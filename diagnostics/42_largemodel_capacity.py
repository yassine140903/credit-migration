"""CURRENT and POOLED single-window models: capacity check.   MEASUREMENT ONLY.

    python diagnostics/42_largemodel_capacity.py   # from the repo root

Mirrors diagnostics/41_dq90_fullfit_capacity.py in structure, split logic and report layout. Writes
diagnostics/42_largemodel_capacity.txt and nothing else: no artifacts/ml/ model, nothing under
src/ or data/, no git command. Reuses src.train_ml's constants and feature handling (BASE_FEATURES,
STATES, TRANSIENT, N_ROUNDS; float32 numerics, property_state and from_state native categoricals with
the fixed category lists train_ml declares, nulls left as-is) and its LightGBM base params.

POPULATIONS (selection window: role == 'fit', 200601 <= from_period <= 200706), run separately:
  (a) CURRENT: from_state == 'Current'. Features: the eight base features (from_state omitted: constant).
  (b) POOLED : all four transient from_states. Features: the eight base features + from_state (categorical).
The panel is read once, one partition at a time, five-plus-feature columns only; the CURRENT population is
the Current rows of the same read. Each population gets its own split, its own prior and its own runs.

SPLIT. Same procedure as diagnostics 40/41: the sorted unique loan_ids of the population are shuffled with
numpy default_rng(SEED), the first 80% (rounded) go to train, the rest are held out. To avoid holding 11M
Python strings, loan_ids are factorised per partition (sort=True) with a running offset; this is the same
sorted-unique order because a loan lives in exactly one partition (asserted by the vintage prefix) and the
partition prefixes sort in file order (asserted).

METRICS. Multiclass log-loss = -mean(log p[true class]) with p clipped to [1e-15, 1]; classes are the six
in panel.STATES order. "Predicted distribution": mean predicted probability per class, and the share/count of
rows whose argmax is that class. Wall-clock is the whole lgb.train call (Dataset construction included).
Peak RSS comes from a fresh src.memwatch.MemWatch around each run's lgb.train: "in-run sampled" is the largest
RSS the 0.5 s sampler saw during that call; "process peak" is MemWatch.peak_rss(), which on Windows includes the
OS's process-lifetime peak working set and so never decreases from run to run.
"""

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                         # noqa: E402
from src.memwatch import MemWatch                                       # noqa: E402
from src.train_ml import BASE_FEATURES, N_ROUNDS, STATES, TRANSIENT     # noqa: E402

OUT = os.path.join("diagnostics", "42_largemodel_capacity.txt")
WIN_LO, WIN_HI = 200601, 200706
SEED = 40
TRAIN_FRAC = 0.8
GB = 2 ** 30
POPULATIONS = [("CURRENT", ["Current"], 10911314), ("POOLED", list(TRANSIENT), 11017047)]
CATEGORIES = {"property_state": list(panel.PROPERTY_STATES), "from_state": list(TRANSIENT)}
BASE_PARAMS = {"objective": "multiclass", "num_class": len(STATES), "device_type": "cpu", "verbosity": -1}
CAPACITIES = [("A. LightGBM defaults (as train_ml)", dict(BASE_PARAMS)),
              ("B. reduced capacity", dict(BASE_PARAMS, num_leaves=8, min_data_in_leaf=100))]
SHOW_FROM_MODEL = ["learning_rate", "num_leaves", "min_data_in_leaf", "min_sum_hessian_in_leaf", "max_depth",
                   "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction", "max_bin", "num_threads",
                   "seed", "deterministic"]
RULE_W = 112
EPS = 1e-15
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


def on_abort(msg):
    emit("ABORTED, run did not complete: " + msg)
    write_report()


def load():
    """Selection-window fit rows, all four transient origins: features (incl. from_state), label, loan code."""
    manifest = panel_io.read_manifest()
    prefixes = ["F" + p["quarter"][2:4] + p["quarter"][4:] for p in manifest["partitions"]]
    assert prefixes == sorted(prefixes), "partition vintage prefixes do not sort in file order"
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + BASE_FEATURES
    xs, ys, codes = [], [], []
    offset = n_null = 0
    for p, prefix in zip(manifest["partitions"], prefixes):
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = d["from_state"].isin(TRANSIENT).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        lid = d["loan_id"].astype(str)
        assert lid.str.startswith(prefix).all(), "a loan_id lacks its partition's vintage prefix"
        c, uniq = pd.factorize(lid, sort=True)
        codes.append(c.astype(np.int64) + offset)
        offset += len(uniq)
        data = {}
        for col in BASE_FEATURES:
            if col in CATEGORIES:
                data[col] = d[col].cat.set_categories(CATEGORIES[col]).reset_index(drop=True)
            else:
                data[col] = d[col].to_numpy(dtype=np.float32, na_value=np.nan)
        data["from_state"] = d["from_state"].astype(str).astype(pd.CategoricalDtype(CATEGORIES["from_state"])).reset_index(drop=True)
        xs.append(pd.DataFrame(data))
        ys.append(y[m])
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows, {len(uniq):,} loans", flush=True)
        del d, data, lid
        gc.collect()
    X = pd.concat(xs, ignore_index=True)
    del xs
    for col, cats in CATEGORIES.items():
        assert str(X[col].dtype) == "category" and list(X[col].cat.categories) == cats, col
    return X, np.concatenate(ys), np.concatenate(codes), n_null, manifest


def logloss(P, y):
    p = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def counts(y):
    return np.bincount(y, minlength=len(STATES))


def side_report(name, y, n_loans):
    c = counts(y)
    emit(f"{name}: {len(y):,} rows, {n_loans:,} loans")
    emit("  " + "   ".join(f"{s} {int(n):,}" for s, n in zip(STATES, c)))
    absent = [s for s, n in zip(STATES, c) if n == 0]
    emit(f"  classes absent from {name}: " + (", ".join(absent) if absent else "none (all six present)"))
    return absent


def dist_table(label, P, y):
    c = counts(y)
    am = counts(P.argmax(axis=1))
    mp = P.mean(axis=0)
    n = len(y)
    emit(f"  {label}  ({n:,} rows)")
    emit(f"  {'class':<10}{'actual n':>12}{'actual %':>11}{'mean pred prob %':>19}{'argmax n':>13}{'argmax %':>11}")
    for i, s in enumerate(STATES):
        emit(f"  {s:<10}{int(c[i]):>12,}{100.0 * c[i] / n:>10.4f}%{100.0 * mp[i]:>18.4f}%"
             f"{int(am[i]):>13,}{100.0 * am[i] / n:>10.4f}%")


def run_population(lgb, pop, origin, expected, Xall, yall, call):
    in_pop = Xall["from_state"].isin(origin).to_numpy()
    feats = list(BASE_FEATURES) + (["from_state"] if len(origin) > 1 else [])
    categorical = ["property_state"] + (["from_state"] if len(origin) > 1 else [])
    X, y, codes = Xall[in_pop].reset_index(drop=True), yall[in_pop], call[in_pop]
    title(f"POPULATION {pop}: role == 'fit', from_state in {origin}, {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"rows {len(y):,} (expected {expected:,}: {'MATCH' if len(y) == expected else 'DIFFERENT'})")
    emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")
    emit("NOT features: loan_age, run_length, run_length_censored" + ("" if len(origin) > 1 else ", from_state (constant)"))

    emit()
    emit(f"SPLIT BY LOAN_ID ({pop})")
    uniq, inv = np.unique(codes, return_inverse=True)       # sorted unique loans of this population
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool)
    flag[perm[:n_tr]] = True
    in_train = flag[inv]
    assert np.intersect1d(np.unique(codes[in_train]), np.unique(codes[~in_train])).size == 0, "a loan is on both sides"
    emit(f"seed {SEED} (numpy default_rng on the sorted unique loan_ids); train fraction {TRAIN_FRAC} of loans "
         f"-> {n_tr:,} of {len(uniq):,} loans to train, {len(uniq) - n_tr:,} held out")
    emit("no loan_id appears on both sides (asserted)")
    emit()
    Xtr, ytr = X[in_train].reset_index(drop=True), y[in_train]
    Xte, yte = X[~in_train].reset_index(drop=True), y[~in_train]
    del X, y, codes, inv
    gc.collect()
    absent_tr = side_report("train", ytr, n_tr)
    emit()
    absent_te = side_report("held-out", yte, len(uniq) - n_tr)
    emit()
    emit(f"row share: train {100.0 * len(ytr) / (len(ytr) + len(yte)):.2f}%   held-out "
         f"{100.0 * len(yte) / (len(ytr) + len(yte)):.2f}%")
    if absent_tr or absent_te:
        emit(f"NOTE: classes absent on a side - train: {absent_tr or 'none'}; held-out: {absent_te or 'none'}. "
             "All six classes remain declared in the model (num_class = 6).")

    prior = counts(ytr) / len(ytr)
    prior_tr = logloss(np.tile(prior, (len(ytr), 1)), ytr)
    prior_te = logloss(np.tile(prior, (len(yte), 1)), yte)
    emit()
    emit("prior-only reference (not a model): log-loss of predicting the TRAIN-side class frequencies for every row: "
         f"train {prior_tr:.6f}   held-out {prior_te:.6f}")

    results = []
    for cap_name, params in CAPACITIES:
        name = f"{pop} / {cap_name}"
        title(f"RUN {name}")
        emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")
        emit("params passed to lgb.train: " + ", ".join(f"{k}={v!r}" for k, v in params.items())
             + f";  num_boost_round={N_ROUNDS}")
        print("  training ...", flush=True)
        watch = MemWatch(on_abort=on_abort)
        watch.phase = f"{name}: lgb.train"
        watch.start()
        t0 = time.perf_counter()
        ds = lgb.Dataset(Xtr[feats], label=ytr, categorical_feature=categorical, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
        secs = time.perf_counter() - t0
        watch.stop()
        del ds
        gc.collect()
        assert booster.current_iteration() == N_ROUNDS, booster.current_iteration()
        assert booster.feature_name() == feats
        assert booster.pandas_categorical == [CATEGORIES[c] for c in categorical]
        block = booster.model_to_string().split("parameters:")[-1]
        eff = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]") and ": " in line:
                k, v = line[1:-1].split(": ", 1)
                eff[k] = v
        emit("effective values (from the model's parameters block): "
             + ", ".join(f"{k}={eff.get(k, '?')}" for k in SHOW_FROM_MODEL))
        n_trees = booster.num_trees()
        emit(f"trees {n_trees} ({n_trees // len(STATES)} per class)")
        emit(f"training wall-clock {secs:.1f} s ({secs / 60:.2f} min) = {secs / N_ROUNDS:.2f} s/round, Dataset construction included;"
             f"  peak RSS in-run sampled {watch._peak / GB:.2f} GB, process peak {watch.peak_rss() / GB:.2f} GB;"
             f"  min system available {watch.min_avail / GB:.2f} GB")
        Ptr = booster.predict(Xtr[feats])
        Pte = booster.predict(Xte[feats])
        del booster
        assert Ptr.shape == (len(ytr), len(STATES)) and Pte.shape == (len(yte), len(STATES))
        l_tr, l_te = logloss(Ptr, ytr), logloss(Pte, yte)
        emit()
        emit(f"  multiclass log-loss, train     : {l_tr:.6f}")
        emit(f"  multiclass log-loss, held-out  : {l_te:.6f}")
        emit(f"  gap (held-out - train)         : {l_te - l_tr:+.6f}")
        emit(f"  held-out minus prior-only held-out : {l_te - prior_te:+.6f}")
        emit()
        dist_table("HELD-OUT rows: actual vs predicted destination distribution", Pte, yte)
        emit()
        dist_table("TRAIN rows: actual vs predicted destination distribution", Ptr, ytr)
        results.append((name, l_tr, l_te, secs, watch._peak, watch.peak_rss()))
        del Ptr, Pte
        gc.collect()
    return results, (pop, prior_tr, prior_te)


def main():
    import lightgbm as lgb

    emit("CURRENT and POOLED models, selection window: capacity check   (measurement only; no selection, no recommendation)")
    X, y, codes, n_null, manifest = load()
    emit(f"panel build {manifest['build_stamp']}   lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs")
    emit(f"rows loaded (all four transient origins, selection window, role == 'fit'): {len(y):,}; "
         f"null/unknown labels dropped {n_null:,}")
    emit(f"loans (all four origins): {len(np.unique(codes)):,}")
    emit(f"rounds {N_ROUNDS}, no early stopping, no validation set")

    all_results, priors = [], []
    for pop, origin, expected in POPULATIONS:
        res, prior = run_population(lgb, pop, origin, expected, X, y, codes)
        all_results.append(res)
        priors.append(prior)

    title("SUMMARY  (figures only)")
    emit(f"{'run':<52}{'train log-loss':>16}{'held-out log-loss':>20}{'gap':>12}{'held-out - prior':>18}")
    for res, (pop, p_tr, p_te) in zip(all_results, priors):
        for name, a, b, *_ in res:
            emit(f"{name:<52}{a:>16.6f}{b:>20.6f}{b - a:>+12.6f}{b - p_te:>+18.6f}")
        emit(f"{'(prior-only reference, ' + pop + ')':<52}{p_tr:>16.6f}{p_te:>20.6f}{p_te - p_tr:>+12.6f}{0.0:>+18.6f}")
    emit()
    emit(f"{'run':<52}{'train seconds':>16}{'in-run RSS GB':>16}{'process peak GB':>18}")
    for res in all_results:
        for name, _, _, secs, pk, ppk in res:
            emit(f"{name:<52}{secs:>16.1f}{pk / GB:>16.2f}{ppk / GB:>18.2f}")

    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
