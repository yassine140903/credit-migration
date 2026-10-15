"""The calibration layer for the ML arm: out-of-fold predictions, the per-origin correction, and its application.

    python -m src.calibrate oof --model artifacts/ml/headline_pooled     # 5 by-loan folds -> the OOF stack (+ shipped-model predictions)
    python -m src.calibrate fit --model artifacts/ml/headline_pooled     # fit and save the corrections from the OOF stack
    from src.calibrate import load_correction, apply_correction          # what src.forecast imports; no fitting code runs

Writes under artifacts/calibration/<model name>/. Nothing under data/ is changed, no evaluate row is read.

OUT-OF-FOLD STACK (oof). Folds are assigned BY LOAN: fold = siphash(loan_id, key = zero-padded FOLD_SEED) mod N_FOLDS, so a loan cannot
sit in two folds and the assignment does not depend on row order. For each fold an identically configured model (the sidecar's own
features, categories, lgb_params and 100 rounds) is trained on the other folds' fit rows and predicts the held-out fold. The population
is train_ml's: role == fit, the four transient origin states, valid label, the whole fit window; its counts are checked against the
shipped model's sidecar. The shipped model itself also predicts every fit row (in-sample) because the correction is applied to it.

THE CORRECTION, per transient origin state o (four separate corrections), fitted on that state's OOF rows:
  step one   stay vs leave. s = the model's mass on o; target = 1[to_state == o]. ISOTONIC regression (monotone, non-decreasing).
  step two   destination among leavers only. q_k = p_k / sum_{j != o} p_j for the five non-origin classes; each class k gets a PLATT
             sigmoid, expit(a * logit(q_k) + b), fitted one-vs-rest on the leavers, and the five are renormalised to sum to one.
  assemble   row[o] = s_cal;  row[k] = (1 - s_cal) * split_k.   Sums to one by construction; asserted to SUM_TOL on every row, never
             renormalised afterwards.
THE NAIVE SCHEME (comparison only): every one of the six classes calibrated one-vs-rest on its own p_k, then the row renormalised. The
brief does not name its method, so it is built both ways (isotonic per class, Platt per class) and neither is preferred here.

logit() clips its argument to [LOGIT_EPS, 1 - LOGIT_EPS]; the number of values clipped is stored with the fit. Isotonic can return exactly
0 or 1; that is left as fitted and shows up in the log-loss floor count.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from src import panel_io
from src.memwatch import MemWatch
from src.train_ml import STATES, TRANSIENT, keep_rows, label_scan

N_FOLDS = 5
FOLD_SEED = 56
FOLD_HASH_KEY = "%016d" % FOLD_SEED
FOLD_METHOD = ("by loan: fold = pandas.util.hash_array(loan_id, hash_key='%s') %% %d (keyed 64-bit siphash of the loan_id string; "
               "independent of row order and of the other loans)" % (FOLD_HASH_KEY, N_FOLDS))
N_ROUNDS = 100
FOLD_TRAIN_LIMIT_S = 15 * 60          # a fold's Dataset construction + training past this stops the run
CAL_ROOT = os.path.join("artifacts", "calibration")
CHUNK = 2_000_000                     # rows per predict / apply chunk
LOGIT_EPS = 1e-12
SUM_TOL = 1e-9                        # float64, asserted on every calibrated row
FLOOR = 1e-15                         # the 11 Oct log-loss probability floor
GB = 2 ** 30
IDX = {s: i for i, s in enumerate(STATES)}


class CalibrationError(RuntimeError):
    pass


def say(s=""):
    print(s, flush=True)


def cal_dir(model_dir):
    with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as fh:
        return os.path.join(CAL_ROOT, json.load(fh)["name"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fold_of(loan_ids):
    """Fold (int8, 0..N_FOLDS-1) of each loan_id."""
    h = pd.util.hash_array(np.asarray(loan_ids, dtype=object), hash_key=FOLD_HASH_KEY)
    return (h % N_FOLDS).astype(np.int8)


# ==========================================================================
# the primitives: logit, Platt, isotonic
# ==========================================================================
def logit(q):
    """logit with the argument clipped to [LOGIT_EPS, 1 - LOGIT_EPS]."""
    q = np.clip(q, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(q) - np.log1p(-q)


def fit_platt(z, t, max_iter=100):
    """MLE of P(t = 1) = expit(a z + b) by Newton's method with step halving on the exact log-likelihood, started at the identity
    (a, b) = (1, 0). Returns a dict; 'converged' is False if the Newton decrement was still above 1e-9 at max_iter."""
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    def nll(a, b):
        u = a * z + b
        return float(np.sum(np.logaddexp(0.0, u) - t * u))

    a, b = 1.0, 0.0
    f = f_identity = nll(a, b)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        p = expit(a * z + b)
        r, w = p - t, p * (1.0 - p)
        g = np.array([np.dot(r, z), r.sum()])
        H = np.array([[np.dot(w * z, z), np.dot(w, z)], [np.dot(w, z), w.sum()]])
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        if float(g @ step) < 1e-9:
            converged = True
            break
        s = 1.0
        for _ in range(50):
            fn = nll(a - s * step[0], b - s * step[1])
            if fn <= f:
                break
            s *= 0.5
        else:
            break                                          # no decrease along the Newton direction: stop, report not converged
        a, b, f = a - s * step[0], b - s * step[1], fn
    return {"a": float(a), "b": float(b), "n": int(len(t)), "positives": int(t.sum()), "nll_identity": f_identity,
            "nll_fit": f, "iterations": it, "converged": converged}


def fit_isotonic(x, t):
    """Non-decreasing isotonic fit of the 0/1 target t on x. Returns the fitted staircase's (x, y) thresholds, applied by np.interp."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    xt, yt = iso.X_thresholds_.astype(np.float64), iso.y_thresholds_.astype(np.float64)
    probe = np.asarray(x[:200_000], dtype=np.float64)
    worst = float(np.max(np.abs(np.interp(probe, xt, yt) - iso.predict(probe)))) if len(probe) else 0.0
    if worst > 1e-12:
        raise CalibrationError("np.interp on the isotonic thresholds differs from sklearn's predict by %.3e" % worst)
    return xt, yt


# ==========================================================================
# apply (what src.forecast imports)
# ==========================================================================
def load_correction(directory):
    """Read a saved decomposed correction: artifacts/calibration/<name>/ (correction.json + iso_<state>.npz)."""
    with open(os.path.join(directory, "correction.json"), encoding="utf-8") as fh:
        corr = json.load(fh)
    corr["_iso"] = {}
    for o in corr["origin_states"]:
        with np.load(os.path.join(directory, "iso_%s.npz" % o.replace("+", "plus"))) as z:
            corr["_iso"][o] = (z["x"], z["y"])
    return corr


def _others(o):
    return [k for k in range(len(STATES)) if k != IDX[o]]


def _apply_one(p, o, corr, check=True):
    p = np.asarray(p, dtype=np.float64)
    oi, oth = IDX[o], _others(o)
    leave = p[:, oth].sum(axis=1)
    if np.any(leave <= 0.0) or not np.all(np.isfinite(leave)):
        raise CalibrationError("%d %s rows have no (or non-finite) mass off the origin state" % (int((~(leave > 0.0)).sum()), o))
    xs, ys = corr["_iso"][o]
    s_cal = np.interp(p[:, oi], xs, ys)
    c = np.empty((len(p), len(oth)))
    for j, k in enumerate(oth):
        par = corr["origins"][o]["platt"][STATES[k]]
        c[:, j] = expit(par["a"] * logit(p[:, k] / leave) + par["b"])
    tot = c.sum(axis=1)
    if np.any(tot <= 0.0) or not np.all(np.isfinite(tot)):
        raise CalibrationError("%s: a row's Platt-scaled destination masses sum to zero or are not finite" % o)
    out = np.empty_like(p)
    out[:, oi] = s_cal
    out[:, oth] = (1.0 - s_cal)[:, None] * (c / tot[:, None])
    if check:
        worst = float(np.max(np.abs(out.sum(axis=1) - 1.0))) if len(out) else 0.0
        if worst > SUM_TOL:
            raise CalibrationError("%s: a calibrated row sums to 1 with error %.3e > %g (a bug, not to be patched)" % (o, worst, SUM_TOL))
    return out


def apply_correction(proba, origin, corr, check=True):
    """Apply a saved decomposed correction to a raw predict_proba matrix.

    proba   (n, 6) float array in class order STATES
    origin  the origin state of every row: one state name (str), or an array of state names / indices into STATES
    corr    load_correction(...)
    Returns an (n, 6) float64 matrix. Each row sums to 1 within SUM_TOL by construction (asserted when check is True).
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(STATES):
        raise CalibrationError("proba must be (n, %d), got %s" % (len(STATES), proba.shape))
    if isinstance(origin, str):
        return _apply_one(proba, origin, corr, check)
    origin = np.asarray(origin)
    names = np.array([STATES[i] for i in origin], dtype=object) if origin.dtype.kind in "iu" else origin.astype(object)
    out = np.empty_like(proba)
    done = np.zeros(len(proba), dtype=bool)
    for o in corr["origin_states"]:
        m = names == o
        if m.any():
            out[m] = _apply_one(proba[m], o, corr, check)
            done |= m
    if not done.all():
        raise CalibrationError("%d rows have an origin state with no correction (have %s)" % (int((~done).sum()), corr["origin_states"]))
    return out


def apply_naive(p, o, naive, kind):
    """The rejected scheme on origin o's rows: each of the six classes calibrated on its own p_k, then the row renormalised.
    Returns (matrix, n rows whose six calibrated values sum to zero; those rows are NaN)."""
    p = np.asarray(p, dtype=np.float64)
    c = np.empty_like(p)
    for k, st in enumerate(STATES):
        if kind == "isotonic":
            xs, ys = naive["_iso"][(o, st)]
            c[:, k] = np.interp(p[:, k], xs, ys)
        else:
            par = naive["platt"][o][st]
            c[:, k] = expit(par["a"] * logit(p[:, k]) + par["b"])
    tot = c.sum(axis=1)
    bad = ~(tot > 0.0)
    out = c / np.where(bad, 1.0, tot)[:, None]
    out[bad] = np.nan
    return out, int(bad.sum())


def load_naive(directory):
    with open(os.path.join(directory, "naive_platt.json"), encoding="utf-8") as fh:
        naive = {"platt": json.load(fh)}
    naive["_iso"] = {}
    with np.load(os.path.join(directory, "naive_isotonic.npz")) as z:
        for o in TRANSIENT:
            for st in STATES:
                key = "%s__%s" % (o.replace("+", "plus"), st.replace("+", "plus"))
                naive["_iso"][(o, st)] = (z[key + "__x"], z[key + "__y"])
    return naive


# ==========================================================================
# stage 1: the out-of-fold stack
# ==========================================================================
def build_inputs(quarters, side, rows_by_quarter):
    """The feature matrix (as train_ml.build_matrix builds it) plus, per row, the label, from_state, from_period and fold; and the
    unique (loan_id, row fold) pairs for the by-loan check."""
    features, categories = side["features"], side["categories"]
    cols = list(dict.fromkeys(features + ["loan_id", "from_state", "to_state", "from_period"]))
    chunks, ys, fss, fps, folds, pair_ids, pair_folds = [], [], [], [], [], [], []
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[q])
        m, y = keep_rows(d, TRANSIENT, None)
        m &= y >= 0
        if int(m.sum()) != rows_by_quarter[q]:
            raise CalibrationError("%s: pass 2 kept %d rows, pass 1 counted %d" % (q, int(m.sum()), rows_by_quarter[q]))
        d = d[m]
        fold = fold_of(d["loan_id"].to_numpy(dtype=object))
        pairs = pd.DataFrame({"l": d["loan_id"].to_numpy(dtype=object), "f": fold}).drop_duplicates()
        if int(pairs["l"].str.len().max()) > 16:
            raise CalibrationError("a loan_id is longer than 16 characters; the S16 pair check would truncate it")
        pair_ids.append(pairs["l"].to_numpy().astype("S16"))
        pair_folds.append(pairs["f"].to_numpy())
        data = {}
        for c in features:
            if c in categories:
                data[c] = d[c].cat.set_categories(categories[c]).reset_index(drop=True)
            else:
                data[c] = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
        chunks.append(pd.DataFrame(data))
        ys.append(y[m].astype(np.int8))
        fss.append(d["from_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int8))
        fps.append(d["from_period"].to_numpy(dtype=np.int32))
        folds.append(fold)
        del d, m, y, data, pairs
        gc.collect()
        say("  inputs %s: %s rows" % (q, format(len(chunks[-1]), ",")))
    X = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    for c in categories:
        if str(X[c].dtype) != "category" or list(X[c].cat.categories) != categories[c]:
            raise CalibrationError("%s lost its fixed category list in the concat" % c)
    return (X, np.concatenate(ys), np.concatenate(fss), np.concatenate(fps), np.concatenate(folds),
            np.concatenate(pair_ids), np.concatenate(pair_folds))


def predict_chunks(booster, X, rows, out):
    """booster.predict on X.iloc[rows] in CHUNK-row pieces, written to out[rows] (out may be a memmap or a plain array)."""
    for s in range(0, len(rows), CHUNK):
        r = rows[s:s + CHUNK]
        out[r] = booster.predict(X.iloc[r])


def run_oof(model_dir):
    import lightgbm as lgb
    with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as fh:
        side = json.load(fh)
    out = cal_dir(model_dir)
    oof_dir, log_dir = os.path.join(out, "oof"), os.path.join(out, "logs")
    os.makedirs(oof_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    features, categorical, params = side["features"], side["categorical_features"], side["lgb_params"]
    bad = [c for c in ("loan_age", "run_length", "run_length_censored") if c in features]
    if bad or side["origin_states"] != TRANSIENT or side["training_window"]["from_period_max_arg"] is not None or \
            params.get("min_sum_hessian_in_leaf") != 10.0 or params.get("lambda_l2") != 10.0 or side["n_rounds"] != N_ROUNDS:
        raise CalibrationError("%s is not the shipped configuration (features %s, origins %s, params %s)"
                               % (model_dir, features, side["origin_states"], params))

    manifest = panel_io.read_manifest()
    if manifest["build_stamp"] != side["panel_build_stamp"]:
        raise CalibrationError("panel build %s != the model's %s" % (manifest["build_stamp"], side["panel_build_stamp"]))
    quarters = [p["quarter"] for p in manifest["partitions"]]
    say("model %s   sha256 %s" % (side["name"], sha256_file(os.path.join(model_dir, "model.txt"))[:16]))
    say("folds: %d, %s" % (N_FOLDS, FOLD_METHOD))

    def on_abort(msg):
        line = "ABORTED, run did not complete: " + msg
        say(line)
        with open(os.path.join(log_dir, "ABORTED.txt"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    watch = MemWatch(on_abort=on_abort)
    watch.start()
    t_all = time.perf_counter()

    watch.phase = "pass 1: label scan"
    S = label_scan(quarters, TRANSIENT, None)
    n = int(sum(S["rows_by_quarter"].values()))
    sc = {st: int(S["label_counts"][i]) for i, st in enumerate(STATES)}
    if n != side["n_rows"] or sc != side["label_counts"] or S["from_counts"] != side["from_state_counts"]:
        raise CalibrationError("the rebuilt population (%d rows, labels %s, origins %s) differs from the sidecar's" % (n, sc, S["from_counts"]))
    say("population matches the sidecar: %s rows, label counts and from_state counts identical" % format(n, ","))

    watch.phase = "pass 2: matrix"
    X, y, fs, fp, fold, pair_ids, pair_folds = build_inputs(quarters, side, S["rows_by_quarter"])
    if len(X) != n:
        raise CalibrationError("matrix has %d rows, expected %d" % (len(X), n))
    say("matrix %s x %d, %.2f GB" % (format(len(X), ","), X.shape[1], X.memory_usage(deep=True).sum() / GB))

    # ---- by-loan check, from the folds the rows actually carry
    uid, inv = np.unique(pair_ids, return_inverse=True)
    fold_of_loan = np.full(len(uid), -1, dtype=np.int64)
    fold_of_loan[inv] = pair_folds
    loan_in_two_folds = int((fold_of_loan[inv] != pair_folds).sum())
    check = {"n_rows": n, "n_loans": int(len(uid)), "loan_pairs_with_a_second_fold": loan_in_two_folds,
             "rows_by_fold": [int((fold == k).sum()) for k in range(N_FOLDS)],
             "loans_by_fold": [int((fold_of_loan == k).sum()) for k in range(N_FOLDS)],
             "seed": FOLD_SEED, "method": FOLD_METHOD}
    if loan_in_two_folds or sum(check["rows_by_fold"]) != n or sum(check["loans_by_fold"]) != len(uid):
        raise CalibrationError("fold assignment is not by loan: %s" % check)
    del uid, inv, fold_of_loan, pair_ids, pair_folds
    gc.collect()
    np.savez(os.path.join(oof_dir, "meta.npz"), y=y, from_state=fs, from_period=fp, fold=fold)

    # ---- one model per fold
    folds_info = []
    for k in range(N_FOLDS):
        marker = os.path.join(oof_dir, "fold%d.json" % k)
        if os.path.exists(marker):
            with open(marker, encoding="utf-8") as fh:
                folds_info.append(json.load(fh))
            say("fold %d: already done, kept" % k)
            continue
        tr, ho = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        say("fold %d: train %s rows, hold out %s rows" % (k, format(len(tr), ","), format(len(ho), ",")))
        timer = threading.Timer(FOLD_TRAIN_LIMIT_S, lambda: (on_abort("fold %d construct + train exceeded %d s" % (k, FOLD_TRAIN_LIMIT_S)),
                                                              os._exit(3)))
        timer.daemon = True
        watch.phase = "fold %d: Dataset construction" % k
        t0 = time.perf_counter()
        timer.start()
        ds = lgb.Dataset(X.iloc[tr], label=y[tr].astype(np.float32), categorical_feature=categorical, free_raw_data=True,
                         params=params if len(params) > 4 else None)
        gc.collect()
        ds.construct()
        gc.collect()
        construct_s = time.perf_counter() - t0
        watch.phase = "fold %d: training" % k
        t1 = time.perf_counter()
        booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
        train_s = time.perf_counter() - t1
        timer.cancel()
        del ds
        gc.collect()
        if booster.feature_name() != features or booster.current_iteration() != N_ROUNDS or \
                booster.pandas_categorical != [side["categories"][c] for c in categorical]:
            raise CalibrationError("fold %d model does not match the sidecar's features / rounds / categories" % k)
        watch.phase = "fold %d: predict" % k
        t2 = time.perf_counter()
        P = np.empty((len(ho), len(STATES)), dtype=np.float64)
        for s in range(0, len(ho), CHUNK):
            P[s:s + CHUNK] = booster.predict(X.iloc[ho[s:s + CHUNK]])
        np.save(os.path.join(oof_dir, "fold%d_proba.npy" % k), P)
        info = {"fold": k, "train_rows": int(len(tr)), "heldout_rows": int(len(ho)), "construct_seconds": round(construct_s, 1),
                "train_seconds": round(train_s, 1), "predict_seconds": round(time.perf_counter() - t2, 1),
                "peak_rss_bytes_so_far": int(watch.peak_rss()), "min_system_available_bytes_so_far": int(watch.min_avail)}
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        folds_info.append(info)
        say("fold %d: construct %.0f s, train %.0f s (%.1f min), predict %.0f s; peak RSS so far %.2f GB, min available %.2f GB" % (
            k, construct_s, train_s, train_s / 60, info["predict_seconds"], watch.peak_rss() / GB, watch.min_avail / GB))
        del booster, P, tr, ho
        gc.collect()

    # ---- the shipped model, in-sample, on every fit row
    full_path = os.path.join(oof_dir, "full_proba.npy")
    if os.path.exists(os.path.join(oof_dir, "full.json")):
        say("shipped-model predictions: already done, kept")
    else:
        watch.phase = "shipped model: predict all fit rows"
        booster = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
        if booster.feature_name() != features or booster.pandas_categorical != [side["categories"][c] for c in categorical]:
            raise CalibrationError("the shipped model's features / categories differ from its sidecar")
        full = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float64, shape=(n, len(STATES)))
        t0 = time.perf_counter()
        for s in range(0, n, CHUNK):
            full[s:s + CHUNK] = booster.predict(X.iloc[s:s + CHUNK])
        full.flush()
        del full, booster
        gc.collect()
        with open(os.path.join(oof_dir, "full.json"), "w", encoding="utf-8") as fh:
            json.dump({"predict_seconds": round(time.perf_counter() - t0, 1)}, fh)
        say("shipped-model predictions on %s fit rows: %.0f s" % (format(n, ","), time.perf_counter() - t0))
    del X
    gc.collect()

    # ---- stack: every row exactly once
    watch.phase = "stack"
    oof_path = os.path.join(oof_dir, "oof_proba.npy")
    oof = np.lib.format.open_memmap(oof_path, mode="w+", dtype=np.float64, shape=(n, len(STATES)))
    written = np.zeros(n, dtype=np.uint8)
    for k in range(N_FOLDS):
        ho = np.flatnonzero(fold == k)
        P = np.load(os.path.join(oof_dir, "fold%d_proba.npy" % k))
        if len(P) != len(ho):
            raise CalibrationError("fold %d: %d predictions for %d held-out rows" % (k, len(P), len(ho)))
        oof[ho] = P
        written[ho] += 1
        del P, ho
    oof.flush()
    worst_sum, bad_cells = 0.0, 0
    for s in range(0, n, CHUNK):
        blk = np.asarray(oof[s:s + CHUNK])
        worst_sum = max(worst_sum, float(np.abs(blk.sum(axis=1) - 1.0).max()))
        bad_cells += int((~np.isfinite(blk)).sum() + (blk < 0).sum())
    check.update({"rows_with_exactly_one_oof_prediction": int((written == 1).sum()), "rows_with_none": int((written == 0).sum()),
                  "rows_with_more_than_one": int((written > 1).sum()), "oof_worst_row_sum_error": worst_sum,
                  "oof_nonfinite_or_negative_cells": bad_cells, "folds": folds_info,
                  "peak_rss_bytes": int(watch.peak_rss()), "min_system_available_bytes": int(watch.min_avail),
                  "total_seconds": round(time.perf_counter() - t_all, 1)})
    watch.stop()
    if check["rows_with_exactly_one_oof_prediction"] != n or bad_cells:
        raise CalibrationError("the OOF stack is not one prediction per row: %s" % check)
    with open(os.path.join(oof_dir, "fold_check.json"), "w", encoding="utf-8") as fh:
        json.dump(check, fh, indent=2)
    say("OOF stack saved: %s   (every one of %s rows has exactly one prediction; worst |row sum - 1| = %.2e)" % (
        oof_path.replace(os.sep, "/"), format(n, ","), worst_sum))
    say("total %.1f min, peak RSS %.2f GB, min system available %.2f GB" % (
        (time.perf_counter() - t_all) / 60, check["peak_rss_bytes"] / GB, check["min_system_available_bytes"] / GB))


# ==========================================================================
# stage 2: fit and save the corrections
# ==========================================================================
def load_stack(model_dir, which="oof_proba"):
    d = os.path.join(cal_dir(model_dir), "oof")
    with np.load(os.path.join(d, "meta.npz")) as z:
        meta = {k: z[k] for k in z.files}
    return np.load(os.path.join(d, which + ".npy"), mmap_mode="r"), meta


def run_fit(model_dir):
    out = cal_dir(model_dir)
    watch = MemWatch(on_abort=lambda msg: say("ABORTED, run did not complete: " + msg))
    watch.start()
    P_all, meta = load_stack(model_dir)
    y_all, fs_all = meta["y"], meta["from_state"]
    corr = {"model": os.path.basename(model_dir.rstrip("/\\")), "model_sha256": sha256_file(os.path.join(model_dir, "model.txt")),
            "classes": STATES, "origin_states": TRANSIENT, "fold_seed": FOLD_SEED, "n_folds": N_FOLDS, "logit_eps": LOGIT_EPS,
            "sum_tol": SUM_TOL, "step_one": "isotonic (stay vs leave)", "step_two": "Platt one-vs-rest on leavers, renormalised",
            "origins": {}}
    naive_platt, naive_iso = {}, {}
    for o in TRANSIENT:
        t0 = time.perf_counter()
        oi, oth = IDX[o], _others(o)
        watch.phase = "fit %s" % o
        idx = np.flatnonzero(fs_all == oi)
        P = np.asarray(P_all[idx])
        y = y_all[idx].astype(np.int64)
        stay = y == oi
        n, n_leave = len(y), int((~stay).sum())
        dest_counts = {STATES[k]: int((y == k).sum()) for k in oth}
        say("%s: %s rows, %s leavers, destinations %s" % (o, format(n, ","), format(n_leave, ","), dest_counts))
        empty = [d for d, c in dest_counts.items() if c == 0]
        if empty:
            raise CalibrationError("%s: no leaver ever goes to %s; its one-vs-rest sigmoid is unfittable. Stopping." % (o, empty))

        xs, ys = fit_isotonic(P[:, oi], stay)
        np.savez(os.path.join(out, "iso_%s.npz" % o.replace("+", "plus")), x=xs, y=ys)
        say("  step one: isotonic, %s thresholds, stay %.6f -> fitted mean %.6f" % (
            format(len(xs), ","), stay.mean(), float(np.interp(P[:, oi], xs, ys).mean())))

        L = P[~stay][:, oth]
        q = L / L.sum(axis=1)[:, None]
        yl = y[~stay]
        platt, clipped = {}, {}
        for j, k in enumerate(oth):
            platt[STATES[k]] = fit_platt(logit(q[:, j]), yl == k)
            clipped[STATES[k]] = {"below_eps": int((q[:, j] < LOGIT_EPS).sum()), "above_1_minus_eps": int((q[:, j] > 1 - LOGIT_EPS).sum())}
            say("  step two: %-8s events %9s  a %.4f  b %.4f  iters %d  converged %s" % (
                STATES[k], format(dest_counts[STATES[k]], ","), platt[STATES[k]]["a"], platt[STATES[k]]["b"],
                platt[STATES[k]]["iterations"], platt[STATES[k]]["converged"]))
        corr["origins"][o] = {"n_rows": n, "n_leavers": n_leave, "dest_counts": dest_counts,
                              "thin_destinations_under_100_events": [d for d, c in dest_counts.items() if c < 100],
                              "isotonic_thresholds": int(len(xs)), "platt": platt, "logit_clipped_counts": clipped}
        del L, q, yl

        npl, ni = {}, {}
        for k, st in enumerate(STATES):
            tk = y == k
            ni["%s__%s__x" % (o.replace("+", "plus"), st.replace("+", "plus"))], ni["%s__%s__y" % (o.replace("+", "plus"), st.replace("+", "plus"))] = \
                fit_isotonic(P[:, k], tk)
            npl[st] = fit_platt(logit(P[:, k]), tk)
            say("  naive %-8s events %10s  isotonic %s thresholds, Platt a %.4f b %.4f converged %s" % (
                st, format(int(tk.sum()), ","), format(len(ni["%s__%s__x" % (o.replace("+", "plus"), st.replace("+", "plus"))]), ","),
                npl[st]["a"], npl[st]["b"], npl[st]["converged"]))
        naive_platt[o] = npl
        naive_iso.update(ni)
        del P, y, stay, idx
        gc.collect()
        say("  %s done in %.0f s" % (o, time.perf_counter() - t0))

    with open(os.path.join(out, "correction.json"), "w", encoding="utf-8") as fh:
        json.dump(corr, fh, indent=2)
    with open(os.path.join(out, "naive_platt.json"), "w", encoding="utf-8") as fh:
        json.dump(naive_platt, fh, indent=2)
    np.savez(os.path.join(out, "naive_isotonic.npz"), **naive_iso)
    watch.stop()
    say("saved the corrections to %s" % out.replace(os.sep, "/"))
    say("peak RSS %.2f GB, min system available %.2f GB" % (watch.peak_rss() / GB, watch.min_avail / GB))


def run_refold(model_dir, k=0):
    """Retrain fold k exactly as run_oof did and SAVE its booster (run_oof did not persist the fold models), then compare the
    retrained model's held-out predictions with the stored fold<k>_proba.npy. Added for diagnostic 57, by instruction."""
    import lightgbm as lgb
    with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as fh:
        side = json.load(fh)
    out = cal_dir(model_dir)
    oof_dir, log_dir = os.path.join(out, "oof"), os.path.join(out, "logs")
    stored = os.path.join(oof_dir, "fold%d_proba.npy" % k)
    if not os.path.exists(stored):
        raise CalibrationError("no stored fold %d predictions at %s to compare against" % (k, stored))
    features, categorical, params = side["features"], side["categorical_features"], side["lgb_params"]
    manifest = panel_io.read_manifest()
    if manifest["build_stamp"] != side["panel_build_stamp"]:
        raise CalibrationError("panel build %s != the model's %s" % (manifest["build_stamp"], side["panel_build_stamp"]))
    quarters = [p["quarter"] for p in manifest["partitions"]]

    def on_abort(msg):
        line = "ABORTED, run did not complete: " + msg
        say(line)
        with open(os.path.join(log_dir, "ABORTED.txt"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    watch = MemWatch(on_abort=on_abort)
    watch.start()
    watch.phase = "pass 1: label scan"
    S = label_scan(quarters, TRANSIENT, None)
    n = int(sum(S["rows_by_quarter"].values()))
    if n != side["n_rows"] or {st: int(S["label_counts"][i]) for i, st in enumerate(STATES)} != side["label_counts"]:
        raise CalibrationError("the rebuilt population differs from the sidecar's")
    watch.phase = "pass 2: matrix"
    X, y, fs, fp, fold, pair_ids, pair_folds = build_inputs(quarters, side, S["rows_by_quarter"])
    del pair_ids, pair_folds
    with np.load(os.path.join(oof_dir, "meta.npz")) as z:
        if not (np.array_equal(z["fold"], fold) and np.array_equal(z["y"], y) and np.array_equal(z["from_state"], fs)):
            raise CalibrationError("the rebuilt rows / folds differ from the ones diagnostic 56 used (meta.npz)")
    say("rebuilt rows, labels, origin states and fold assignment are identical to diagnostic 56's")
    tr, ho = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    say("fold %d: train %s rows, hold out %s rows" % (k, format(len(tr), ","), format(len(ho), ",")))
    timer = threading.Timer(FOLD_TRAIN_LIMIT_S, lambda: (on_abort("fold %d construct + train exceeded %d s" % (k, FOLD_TRAIN_LIMIT_S)), os._exit(3)))
    timer.daemon = True
    timer.start()
    watch.phase = "fold %d: Dataset construction + training" % k
    t0 = time.perf_counter()
    ds = lgb.Dataset(X.iloc[tr], label=y[tr].astype(np.float32), categorical_feature=categorical, free_raw_data=True,
                     params=params if len(params) > 4 else None)
    gc.collect()
    ds.construct()
    gc.collect()
    booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
    timer.cancel()
    train_s = time.perf_counter() - t0
    del ds
    gc.collect()
    if booster.feature_name() != features or booster.current_iteration() != N_ROUNDS or             booster.pandas_categorical != [side["categories"][c] for c in categorical]:
        raise CalibrationError("the retrained model does not match the sidecar's features / rounds / categories")
    path = os.path.join(oof_dir, "fold%d_model.txt" % k)
    booster.save_model(path)
    P0 = np.load(stored)
    worst, n_diff = 0.0, 0
    for s in range(0, len(ho), CHUNK):
        P = booster.predict(X.iloc[ho[s:s + CHUNK]])
        d = np.abs(P - P0[s:s + CHUNK])
        worst, n_diff = max(worst, float(d.max())), n_diff + int((d > 0).sum())
    info = {"fold": k, "model": path.replace(os.sep, "/"), "model_sha256": sha256_file(path), "train_seconds": round(train_s, 1),
            "heldout_rows": int(len(ho)), "compared_cells": int(P0.size), "max_abs_diff_vs_stored_proba": worst,
            "cells_differing_at_all": n_diff, "peak_rss_bytes": int(watch.peak_rss()), "min_system_available_bytes": int(watch.min_avail),
            "note": "retrained for diagnostic 57 by instruction; the original fold booster was not persisted"}
    watch.stop()
    with open(os.path.join(oof_dir, "fold%d_model.json" % k), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    say("fold %d retrained in %.0f s, saved %s" % (k, train_s, path.replace(os.sep, "/")))
    say("held-out predictions vs the stored diagnostic-56 ones: max |diff| = %.3e, cells differing at all = %s of %s" % (
        worst, format(n_diff, ","), format(P0.size, ",")))
    say("peak RSS %.2f GB, min system available %.2f GB" % (watch.peak_rss() / GB, watch.min_avail / GB))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["oof", "fit", "refold"])
    ap.add_argument("--model", required=True, help="the model artefact directory, e.g. artifacts/ml/headline_pooled")
    args = ap.parse_args(argv)
    {"oof": run_oof, "fit": run_fit, "refold": run_refold}[args.stage](args.model)


if __name__ == "__main__":
    sys.exit(main())
