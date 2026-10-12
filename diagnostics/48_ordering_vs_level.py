"""Ordering skill versus level skill on CURRENT-origin loans.   MEASUREMENT ONLY.

    python diagnostics/48_ordering_vs_level.py   # from the repo root

Retrains the two CURRENT runs of diagnostic 47 (R1 full features, R2 borrower-only), exactly as 47 did, and separates two skills the
Brier score conflates: ORDERING (does a riskier loan get a higher predicted score?) and LEVEL (are the predicted probabilities
numerically right?). Writes diagnostics/48_ordering_vs_level.txt, and persists the two models to artifacts/diag48/ (model text files plus a
JSON sidecar; artifacts/ is not covered by .gitignore, which this script does not touch). Nothing under src/ or data/, no git command.
Fit rows only (load_panel(roles=['fit'])); evaluate and condition rows are never read.

Reused from diagnostic 47 (imported, not re-derived): load_rows() (a copy of 44's loader: window 200601..200812, from_state == 'Current',
panel.STATES label mapping, float32 numerics, property_state native categorical with the fixed category list, nulls left as-is),
split_flags() (by-loan 80/20, numpy default_rng(40) on the sorted unique loan codes, no loan on both sides asserted), softmax(),
the LightGBM params dict (objective multiclass, num_class 6, device cpu, defaults otherwise; verbosity 0), 100 rounds, and 44's Brier.
The per-class train / held-out counts must equal diagnostic 47's pass-1 table exactly (hard-coded below) or the script stops.
Reproduction of 47: held-out Brier R1 0.0535804, R2 0.0428301; a mismatch is reported loudly and the run continues.

TARGETS (held-out rows, label = to_state, the panel's next state):
  LEAVES CURRENT: event = (next state != Current); score = 1 - P(Current), as specified. 1 - P(Current) loses nothing the five other
      probabilities do not, except where P(Current) rounds to exactly 1.0; Step 1 therefore also reports the same score computed as the
      sum of the five other probabilities, and a margin score (logsumexp of the five other raw scores minus the Current raw score),
      which cannot saturate. Those two are SUPPLEMENTARY and labelled so; the specified score is the primary one everywhere.
  MOVES TO DQ30 (secondary): event = (next state == DQ30); score = P(DQ30). Event counts are shown beside every figure.

TIES. Every ordering statistic is computed from the table of distinct predicted values (ascending): value, rows, events. AUC counts
a tied positive-negative pair as 0.5 (exact, rank-based: sum over values of positives x (negatives below + half the negatives at the
value) / (P x N)); a top-q% cut takes ceil(q n) rows from the top and credits the events of the block tied at the cutoff pro rata to the
slots remaining; the decile table splits the ascending order into 10 equal-count position groups, with a block straddling a group
boundary shared pro rata (so decile event counts can be fractional; this is the expectation under a random tie-break). The tie structure is
printed first (Step 1) so that every later number can be read against it.

STEP 4 shows two cuts of the "leaves Current" score: (a) the same ten equal-count bins as the Step 2 decile table, with mean predicted
next to the actual rate, and (b) ten bins on the predicted VALUE with fixed edges 0, .005, .01, .015, .02, .03, .05, .1, .2, .5, 1.
"""

import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from src.memwatch import MemWatch                                       # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join("diagnostics", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D47 = _load("diag47", "47_full_window_capacity.py")
D44 = D47.D44
STATES, BASE_FEATURES, CATEGORIES = D47.STATES, D47.BASE_FEATURES, D47.CATEGORIES
K = len(STATES)
I_CUR, I_DQ30 = STATES.index("Current"), STATES.index("DQ30")
PARAMS, N_ROUNDS, R2_DROP = D47.PARAMS, D47.N_ROUNDS, D47.R2_DROP
brier_by_class, counts, softmax = D47.brier_by_class, D47.counts, D47.softmax
emit, title, GB = D47.emit, D47.title, D47.GB

SMOKE = os.environ.get("DIAG48_SMOKE") == "1"
D47.SMOKE = SMOKE                                               # D47.load_rows reads its own flag: first partition only
OUT = os.environ.get("DIAG48_OUT", os.path.join("diagnostics", "48_ordering_vs_level.txt"))
MODEL_DIR = os.environ.get("DIAG48_MODELS", os.path.join("artifacts", "diag48"))
ORIGIN = ["Current"]
EXPECT_COUNTS = {"train": [30959138, 268335, 2289, 1159, 259277, 55], "held-out": [7737923, 67032, 543, 270, 64833, 19]}
EXPECT_BRIER = {"R1 full features": 0.0535804, "R2 borrower-only features": 0.0428301}
MONTHS = [200706, 200712, 200806, 200809, 200811, 200812]
CUTS = [0.01, 0.05, 0.10]
VALUE_EDGES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0000001]
TARGETS = [("LEAVES CURRENT", "leave"), ("MOVES TO DQ30 (secondary)", "dq30")]


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(D47.LINES) + "\n")


D47.write_report = write_report                                 # on_abort inside D47 / MemWatch writes to THIS report
D47.OUT = OUT


class Tab:
    """Distinct predicted values (ascending): value u, rows cnt, events ev; and the derived statistics."""

    def __init__(self, score, event):
        u, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
        self.u, self.cnt = u, cnt.astype(np.float64)
        self.ev = np.bincount(inv, weights=event.astype(np.float64), minlength=len(u))
        self.n, self.P = float(self.cnt.sum()), float(self.ev.sum())
        self.N = self.n - self.P

    def auc(self):
        if self.P == 0 or self.N == 0:
            return float("nan")
        neg = self.cnt - self.ev
        below = np.cumsum(neg) - neg
        return float((self.ev * (below + 0.5 * neg)).sum() / (self.P * self.N))

    def top(self, frac):
        """(events captured in the top `frac` by score, slots k, size of the tie block at the cutoff, straddles?)"""
        cd, ed = self.cnt[::-1], self.ev[::-1]
        cc = np.cumsum(cd)
        k = int(np.ceil(frac * self.n))
        i = int(np.searchsorted(cc, k))
        before_c, before_e = cc[i] - cd[i], np.cumsum(ed)[i] - ed[i]
        got = before_e + ed[i] * (k - before_c) / cd[i]
        return float(got), k, int(cd[i]), bool(cc[i] > k and cd[i] > 1)

    def groups(self, edges_pos):
        """Per group between consecutive positions in the ascending order: rows, events, predicted sum, distinct values touched."""
        cc = np.concatenate([[0.0], np.cumsum(self.cnt)])
        ce = np.concatenate([[0.0], np.cumsum(self.ev)])
        cs = np.concatenate([[0.0], np.cumsum(self.cnt * self.u)])
        e_ev, e_ps = np.interp(edges_pos, cc, ce), np.interp(edges_pos, cc, cs)
        out = []
        for a, b, ea, eb, pa, pb in zip(edges_pos[:-1], edges_pos[1:], e_ev[:-1], e_ev[1:], e_ps[:-1], e_ps[1:]):
            lo = int(np.searchsorted(cc, a, side="right")) - 1
            hi = int(np.searchsorted(cc, b, side="left")) - 1
            out.append((b - a, eb - ea, pb - pa, hi - lo + 1))
        return out

    def deciles(self):
        return self.groups(np.round(np.arange(11) * self.n / 10.0))

    def value_bins(self, edges):
        b = np.clip(np.digitize(self.u, edges) - 1, 0, len(edges) - 2)
        rows = np.bincount(b, weights=self.cnt, minlength=len(edges) - 1)
        ev = np.bincount(b, weights=self.ev, minlength=len(edges) - 1)
        ps = np.bincount(b, weights=self.cnt * self.u, minlength=len(edges) - 1)
        return rows, ev, ps


def fit_and_score(lgb, cap, tag, feats, Xtr, ytr, Xte, yte):
    """Train as diagnostic 47 did; return the held-out scores (probability matrix is not kept)."""
    r = {"tag": tag, "feats": feats}
    categorical = ["property_state"]
    print(f"  [{tag}] training ...", flush=True)
    watch = D47.MemWatch(on_abort=D47.on_abort, abort_mb=700)
    watch.phase = tag
    watch.start()
    cap.msgs.clear()
    t0 = time.perf_counter()
    Xd = Xtr if list(Xtr.columns) == feats else Xtr[feats]
    ds = lgb.Dataset(Xd, label=ytr.astype(np.int64), categorical_feature=categorical, free_raw_data=True, params=PARAMS)
    booster = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
    r["t_train"] = time.perf_counter() - t0
    del ds, Xd
    gc.collect()
    assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
    assert booster.pandas_categorical == [CATEGORIES[c] for c in categorical]
    r["msgs"] = list(cap.msgs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = "current_" + ("R1_full" if tag.startswith("R1") else "R2_borrower_only")
    booster.save_model(os.path.join(MODEL_DIR, stem + ".txt"))
    with open(os.path.join(MODEL_DIR, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"origin_states": ORIGIN, "features": feats, "categorical": categorical, "class_order": STATES, "params": PARAMS,
                   "num_boost_round": N_ROUNDS, "window": [D47.WIN_LO, D47.WIN_HI], "split_seed": D47.SEED, "train_fraction": D47.TRAIN_FRAC,
                   "train_rows": int(len(ytr)), "held_out_rows": int(len(yte)), "made_by": "diagnostics/48_ordering_vs_level.py"}, f, indent=1)
    r["model_path"] = os.path.join(MODEL_DIR, stem + ".txt")
    t1 = time.perf_counter()
    raw = np.asarray(booster.predict(Xte[feats], raw_score=True), dtype=np.float64)
    r["t_pred"] = time.perf_counter() - t1
    del booster
    assert raw.shape == (len(yte), K)
    P = softmax(raw)
    yi = yte.astype(np.int64)
    cls = brier_by_class(P, yi)
    r["brier"], r["cls"] = float(cls.sum()), cls
    r["score_leave"] = 1.0 - P[:, I_CUR]
    alt = P[:, 1:].sum(axis=1)
    _, alt_cnt = np.unique(alt, return_counts=True)
    r["alt_distinct"], r["alt_largest"], r["alt_n"] = int(len(alt_cnt)), int(alt_cnt.max()), int(len(alt))
    r["max_alt_diff"] = float(np.abs(1.0 - P[:, I_CUR] - alt).max())
    del alt, alt_cnt
    r["score_dq30"] = P[:, I_DQ30].copy()
    r["mean_p"] = P.mean(axis=0)
    m = raw[:, 1:].max(axis=1)
    r["score_margin"] = m + np.log(np.exp(raw[:, 1:] - m[:, None]).sum(axis=1)) - raw[:, I_CUR]
    r["n_p0_exact1"] = int((P[:, I_CUR] == 1.0).sum())
    del raw, P
    watch.stop()
    gc.collect()
    r["peak"], r["rss"], r["min_avail"] = watch.peak_rss(), watch._peak, watch.min_avail
    return r


def tie_section(runs):
    title("STEP 1: TIE STRUCTURE of the predicted scores  (held-out rows; reported before any ordering number)")
    for r in runs:
        for tname, key in (("LEAVES CURRENT, score = 1 - P(Current)", "score_leave"), ("MOVES TO DQ30, score = P(DQ30)", "score_dq30")):
            t = Tab(r[key], np.zeros(len(r[key])))
            order = np.argsort(-t.cnt)
            c_sorted = t.cnt[order]
            cum = np.cumsum(c_sorted)
            n50, n90 = int(np.searchsorted(cum, 0.5 * t.n)) + 1, int(np.searchsorted(cum, 0.9 * t.n)) + 1
            emit()
            emit(f"{r['tag']} / {tname}   ({int(t.n):,} held-out rows)")
            emit(f"  distinct predicted values: {len(t.u):,}   share of rows in the single largest tie block: {100 * c_sorted[0] / t.n:.4f}%   "
                 f"distinct values needed to cover 50% of rows: {n50:,}; 90%: {n90:,}")
            if len(t.u) < 500:
                emit(f"  FEW DISTINCT VALUES ({len(t.u):,}): too few for a fine ranking to exist; the ordering numbers below are bounded by this leaf structure.")
            emit(f"  {'rank':>4}{'predicted value':>26}{'rows':>14}{'share of held-out':>20}")
            for i in range(min(10, len(order))):
                emit(f"  {i + 1:>4}{t.u[order[i]]:>26.17g}{int(c_sorted[i]):>14,}{100 * c_sorted[i] / t.n:>19.4f}%")
        t3 = Tab(r["score_margin"], np.zeros(len(r["score_margin"])))
        emit()
        emit(f"{r['tag']} / SUPPLEMENTARY forms of the LEAVES CURRENT score: distinct values, largest tie block share")
        emit(f"  sum of the five other probabilities : {r['alt_distinct']:,} distinct, largest block {100 * r['alt_largest'] / r['alt_n']:.4f}%")
        emit(f"  margin (logsumexp of other raws - Current raw) : {len(t3.u):,} distinct, largest block {100 * t3.cnt.max() / t3.n:.4f}%")
        emit(f"  rows where P(Current) is exactly 1.0 in float64: {r['n_p0_exact1']:,};  max |(1 - P(Current)) - sum of other probabilities| = {r['max_alt_diff']:.3e}")
        del t3
    D47.write_report()


def target_arrays(r, y):
    return {"leave": (r["score_leave"], (y != I_CUR)), "dq30": (r["score_dq30"], (y == I_DQ30))}


def pooled_section(runs, y):
    title("STEP 2: ORDERING, held-out rows, ALL MONTHS POOLED")
    emit("AUC counts tied pairs as 0.5 (exact). Top-q% cuts credit the events of the block tied at the cutoff pro rata ('tied' = rows in that block; "
         "'*' = the cutoff falls inside it). Random ranking: AUC 0.5, top q% captures q% of events.")
    for r in runs:
        for tname, key in TARGETS:
            sc, ev = target_arrays(r, y)[key]
            t = Tab(sc, ev)
            emit()
            emit(f"{r['tag']} / {tname}:  rows {int(t.n):,}, events {int(t.P):,} ({100 * t.P / t.n:.4f}%)")
            emit(f"  AUC = {t.auc():.5f}   (0.5 = chance)")
            emit(f"  {'cut':>6}{'rows k':>12}{'events in top':>16}{'share of events':>17}{'random gives':>14}{'lift':>8}{'tied@cutoff':>14}")
            for q in CUTS:
                got, k, tie, strad = t.top(q)
                emit(f"  {100 * q:>5.0f}%{k:>12,}{got:>16.2f}{100 * got / t.P:>16.2f}%{100 * q:>13.0f}%{(got / t.P) / q:>8.2f}{tie:>13,}{'*' if strad else ' '}")
            dec = t.deciles()
            emit(f"  DECILE TABLE (decile 1 = lowest predicted score; equal-count groups, tied blocks shared pro rata)")
            emit(f"  {'decile':>6}{'rows':>12}{'events':>12}{'actual rate':>14}{'mean predicted':>16}{'distinct values':>17}")
            for i, (n_, e_, ps_, nd) in enumerate(dec):
                emit(f"  {i + 1:>6}{int(n_):>12,}{e_:>12.1f}{100 * e_ / n_:>13.4f}%{100 * ps_ / n_:>15.4f}%{nd:>17,}")
            rates = [e_ / n_ for n_, e_, _, _ in dec]
            emit(f"  top decile rate / bottom decile rate: " + (f"{rates[-1] / rates[0]:.2f}x" if rates[0] > 0 else "n/a (bottom decile has no events)")
                 + f";  monotone non-decreasing across deciles: {all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))}")
            if key == "leave":
                tm = Tab(r["score_margin"], ev)
                emit(f"  SUPPLEMENTARY: AUC of the margin score {tm.auc():.5f} ({len(tm.u):,} distinct values); "
                     f"top-1%/5%/10% shares " + " / ".join(f"{100 * tm.top(q)[0] / tm.P:.2f}%" for q in CUTS))
                del tm
            del t
    D47.write_report()


def month_section(runs, y, fp):
    title("STEP 3: ORDERING WITHIN CALENDAR MONTHS  (each month separately; NOT pooled)")
    emit("Per month: held-out rows of that from_period only. Decile rows show the actual event rate (%) per equal-count group of that month "
         "(decile 1 = lowest predicted score; ties shared pro rata). A flat line / AUC 0.5 within a month means no borrower-level ordering there.")
    for r in runs:
        for tname, key in TARGETS:
            cols = []
            for m in MONTHS:
                mm = fp == m
                sc, ev = target_arrays(r, y)[key]
                t = Tab(sc[mm], ev[mm])
                cols.append((m, t, t.deciles()))
            emit()
            emit(f"{r['tag']} / {tname}")
            emit(f"  {'':<12}" + "".join(f"{m:>12}" for m, _, _ in cols))
            emit(f"  {'rows':<12}" + "".join(f"{int(t.n):>12,}" for _, t, _ in cols))
            emit(f"  {'events':<12}" + "".join(f"{int(t.P):>12,}" for _, t, _ in cols))
            emit(f"  {'AUC':<12}" + "".join(f"{t.auc():>12.5f}" for _, t, _ in cols))
            emit(f"  {'distinct':<12}" + "".join(f"{len(t.u):>12,}" for _, t, _ in cols))
            for i in range(10):
                emit(f"  {'decile ' + str(i + 1) + ' %':<12}" + "".join(f"{100 * d[i][1] / d[i][0]:>12.4f}" for _, _, d in cols))
            emit(f"  {'all rows %':<12}" + "".join(f"{100 * t.P / t.n:>12.4f}" for _, t, _ in cols))
    D47.write_report()


def level_section(runs, y):
    title("STEP 4: LEVEL, for contrast  (LEAVES CURRENT, held-out rows)")
    for r in runs:
        sc, ev = target_arrays(r, y)["leave"]
        t = Tab(sc, ev)
        emit()
        emit(f"{r['tag']}: overall mean predicted {100 * (t.cnt * t.u).sum() / t.n:.4f}% vs actual {100 * t.P / t.n:.4f}% "
             f"(ratio {((t.cnt * t.u).sum() / t.n) / (t.P / t.n):.3f}x)")
        emit(f"  (a) ten equal-count bins (the Step 2 deciles)")
        emit(f"  {'bin':>4}{'rows':>12}{'mean predicted':>17}{'actual rate':>14}{'predicted / actual':>20}")
        for i, (n_, e_, ps_, _) in enumerate(t.deciles()):
            emit(f"  {i + 1:>4}{int(n_):>12,}{100 * ps_ / n_:>16.4f}%{100 * e_ / n_:>13.4f}%" + (f"{(ps_ / e_):>19.2f}x" if e_ > 0 else f"{'n/a (0 events)':>20}"))
        emit(f"  (b) ten bins on the predicted VALUE")
        emit(f"  {'predicted value in':<22}{'rows':>12}{'mean predicted':>17}{'actual rate':>14}{'events':>10}{'predicted / actual':>20}")
        rows, evb, ps = t.value_bins(VALUE_EDGES)
        for i in range(10):
            lo, hi = VALUE_EDGES[i], VALUE_EDGES[i + 1]
            lab = f"[{lo:g}, {min(hi, 1.0):g}{']' if i == 9 else ')'}"
            if rows[i] == 0:
                emit(f"  {lab:<22}{0:>12,}{'-':>17}{'-':>14}{0:>10,}")
            else:
                emit(f"  {lab:<22}{int(rows[i]):>12,}{100 * ps[i] / rows[i]:>16.4f}%{100 * evb[i] / rows[i]:>13.4f}%{int(evb[i]):>10,}"
                     + (f"{ps[i] / evb[i]:>19.2f}x" if evb[i] > 0 else f"{'n/a (0 events)':>20}"))
        del t
    D47.write_report()


def main():
    import lightgbm as lgb

    cap = D47.Capture()
    lgb.register_logger(cap)
    emit("ORDERING vs LEVEL: CURRENT-origin loans, full fit window, retrained diagnostic-47 runs R1 and R2   (measurement only)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs" + ("   SMOKE RUN (first partition only)" if SMOKE else ""))
    title("SETUP")
    X, y, fp, fs, codes, n_null, manifest, n_loans = D47.load_rows(ORIGIN, True)
    emit(f"panel build {manifest['build_stamp']}; window role == 'fit', {D47.WIN_LO} <= from_period <= {D47.WIN_HI}; from_state == Current")
    emit(f"rows {len(y):,}; loans {n_loans:,}; null/unknown labels dropped {n_null:,}")
    in_train, nl, n_tr = D47.split_flags(codes)
    emit(f"split by loan_id: seed {D47.SEED}, train fraction {D47.TRAIN_FRAC} -> {n_tr:,} of {nl:,} loans to train, {nl - n_tr:,} held out; no loan on both sides (asserted)")
    ctr, cte = counts(y[in_train].astype(np.int64)), counts(y[~in_train].astype(np.int64))
    emit(f"  {'class':<10}{'TRAIN rows':>14}{'expected (47)':>16}{'HELD-OUT rows':>16}{'expected (47)':>16}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(ctr[j]):>14,}{EXPECT_COUNTS['train'][j]:>16,}{int(cte[j]):>16,}{EXPECT_COUNTS['held-out'][j]:>16,}")
    same = list(map(int, ctr)) == EXPECT_COUNTS["train"] and list(map(int, cte)) == EXPECT_COUNTS["held-out"]
    emit(f"per-class counts vs diagnostic 47's pass-1 table: {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    if not same and not SMOKE:
        emit("STOP: the population or split does not reproduce diagnostic 47's. Nothing was trained.")
        write_report()
        sys.exit(1)
    del codes, fs
    cols = list(BASE_FEATURES)
    Xtr, ytr = X.loc[in_train, cols].reset_index(drop=True), y[in_train]
    Xte, yte, fpte = X.loc[~in_train, cols].reset_index(drop=True), y[~in_train], fp[~in_train]
    del X, y, fp, in_train
    gc.collect()
    feats2 = [f for f in cols if f not in R2_DROP]
    runs = []
    emit("memory layout: a first attempt of this script stopped on the memory floor (available system memory fell to 621 MB during R2 training, "
         "no rows sampled); this report is from the second attempt, which rebuilds the training frame with R2's six columns after R1 is done "
         "(so R2 needs no second copy) and keeps fewer score arrays between runs. The data, split, params and rounds are unchanged.")
    for tag, fl in (("R1 full features", cols), ("R2 borrower-only features", feats2)):
        if tag.startswith("R2"):
            Xtr = Xtr[feats2].copy()
            gc.collect()
        r = fit_and_score(lgb, cap, tag, fl, Xtr, ytr, Xte, yte)
        runs.append(r)
        want = EXPECT_BRIER[tag]
        ok = abs(round(r["brier"], 7) - want) < 1.5e-7
        emit()
        emit(f"RUN {tag}: features ({len(fl)}) {fl}; categorical ['property_state']; params " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items())
             + f"; num_boost_round={N_ROUNDS}")
        emit(f"  library messages: " + (" | ".join(f"{m} x{r['msgs'].count(m)}" for m in sorted(set(r['msgs']))) if r["msgs"] else "none"))
        emit(f"  wall-clock: lgb.train (Dataset construction included) {r['t_train']:.1f} s ({r['t_train'] / 60:.2f} min); held-out predict {r['t_pred']:.1f} s;"
             f"  peak RSS in-run sampled {r['rss'] / GB:.2f} GB, process peak {r['peak'] / GB:.2f} GB;  min system available {r['min_avail'] / GB:.2f} GB")
        emit(f"  REPRODUCTION of diagnostic 47: held-out Brier {r['brier']:.7f} vs 47's {want:.7f} -> "
             + ("MATCH to seven decimals" if ok else f"!!! MISMATCH (difference {r['brier'] - want:+.3e}); the run continues, read everything below against this !!!"))
        emit(f"  model persisted: {r['model_path']} (+ .json sidecar)")
        r["repro"] = ok
        write_report()
    peak = max(r["peak"] for r in runs)
    del Xtr, Xte
    gc.collect()
    yi = yte.astype(np.int64)
    tie_section(runs)
    pooled_section(runs, yi)
    month_section(runs, yi, fpte)
    level_section(runs, yi)

    title("FINDINGS  (measured only; no interpretation, no recommendation)")
    for r in runs:
        emit(f"{r['tag']}:")
        emit(f"  - reproduces diagnostic 47's held-out Brier: {r['repro']} ({r['brier']:.7f}).")
        for tname, key in TARGETS:
            sc, ev = target_arrays(r, yi)[key]
            t = Tab(sc, ev)
            q = [100 * t.top(c)[0] / t.P for c in CUTS]
            dec = t.deciles()
            rates = [100 * e_ / n_ for n_, e_, _, _ in dec]
            emit(f"  - {tname}: {len(t.u):,} distinct values, largest tie block {100 * t.cnt.max() / t.n:.2f}% of rows; pooled AUC {t.auc():.5f} "
                 f"({int(t.P):,} events in {int(t.n):,} rows); top 1/5/10% hold {q[0]:.2f}% / {q[1]:.2f}% / {q[2]:.2f}% of events; "
                 f"decile 1 rate {rates[0]:.4f}%, decile 10 rate {rates[-1]:.4f}%, all rows {100 * t.P / t.n:.4f}%.")
            aucs = []
            for m in MONTHS:
                mm = fpte == m
                aucs.append(f"{m}: {Tab(sc[mm], ev[mm]).auc():.5f}")
            emit(f"    per-month AUC: " + "; ".join(aucs) + ".")
            if key == "leave":
                n_, e_, ps_, _ = dec[-1]
                emit(f"    level: overall mean predicted {100 * (t.cnt * t.u).sum() / t.n:.4f}% vs actual {100 * t.P / t.n:.4f}%; "
                     f"top decile mean predicted {100 * ps_ / n_:.4f}% vs actual {100 * e_ / n_:.4f}%.")
            del t
    emit()
    emit(f"PEAK RSS (src.memwatch): process peak {peak / GB:.2f} GB; lowest system-available memory seen "
         f"{min(r['min_avail'] for r in runs) / GB:.2f} GB")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped (no rows were sampled): {e}")
        write_report()
        raise
