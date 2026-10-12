"""Leaf-explosion control: can the softmax saturation be stopped without losing the ordering?   MEASUREMENT + ONE PRE-REGISTERED SELECTION.

    python diagnostics/49_leaf_explosion_sweep.py   # from the repo root

Diagnostic 48 found ordering skill (DQ30 AUC 0.730, leaves-Current AUC 0.606 for R2) next to broken levels (62.7% of held-out rows with a
leaves-Current score of exactly 0 for R1, raw-score spreads up to 4.3e8). Mechanism under test: leaf values are unbounded, so a leaf with one
rare event among few rows takes an extreme value and 100 rounds of those accumulate until the softmax saturates. This diagnostic sweeps two
settings that bound leaf values directly and measures saturation, levels and ordering for each. It tunes nothing beyond the specified 3 x 3 grid,
adds no feature and applies no calibration: every figure is on RAW model outputs. Writes diagnostics/49_leaf_explosion_sweep.txt and persists each
run's model to artifacts/diag49/ (lever values in the file name, JSON sidecar). Nothing under src/ or data/, no git command. Fit rows only
(load_panel(roles=['fit'])); evaluate and condition rows are never read.

Population, split, feature handling, label mapping and class order are diagnostic 48's, which are diagnostic 47's (imported, not re-derived): CURRENT
origin only, role == 'fit', 200601 <= from_period <= 200812, by-loan 80/20 split with numpy default_rng(40), panel.STATES class order, float32
numerics, property_state a native categorical with the fixed category list, nulls left as-is, 44's explicit multiclass Brier. The per-class train /
held-out counts must equal diagnostic 48's table (hard-coded below) or the script stops.

FEATURES: R2 BORROWER-ONLY (fico, ltv, dti, rate_spread, property_state, est_ltv). Macro is excluded for a MEASURED reason, not a preference:
diagnostic 48 found DQ30 AUC 0.647 with macro (R1) against 0.730 without (R2). (est_ltv still carries an HPI component, as flagged in 47 and 48.)

GRID (nine runs, full cross; the first value of each lever is the library default, so run 1 is diagnostic 48 R2):
  min_sum_hessian_in_leaf in {1e-3, 10, 1000}   x   lambda_l2 in {0, 10, 1000}
  all other params as diagnostic 48: objective multiclass, num_class 6, device cpu, verbosity 0, 100 rounds, num_leaves 31, min_data_in_leaf 20.
Run 1 must reproduce diagnostic 48 R2's held-out Brier 0.0428301 (to seven decimals) or the script stops after reporting it.

MEMORY. The training frame is built once (six float32/categorical columns) and reused by all nine runs; after a run only scalars and small tables are
kept. The src.memwatch floor stops the run if available system memory falls below 700 MB; rows are never sampled.

PRE-REGISTERED SELECTION RULE (fixed before any number was read; applied mechanically at the end, no judgement substituted):
  GATE : a run is eligible only if its "moves to DQ30" AUC is >= 0.720 (diagnostic 48 R2: 0.72976, minus 0.01).
  PICK : among eligible runs, the LOWEST held-out multiclass Brier wins.
  No eligible run -> report that no setting preserved ordering, pick nothing. Winner is run 1 -> report that the levers made no difference, pick nothing.
  The gate is not relaxed and no setting outside the grid is considered.

Definitions. "leaves Current": event = (next state != Current), score = 1 - P(Current). "moves to DQ30": event = (next state == DQ30), score = P(DQ30).
AUC counts tied pairs as 0.5 (exact, from the table of distinct predicted values, diagnostic 48's Tab). Deciles are equal-count position groups in ascending score
order with a tied block shared pro rata. "Exactly 0" / "exactly 1" are tests on the float64 score 1 - P(Current).
"""

import gc
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join("diagnostics", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D48 = _load("diag48", "48_ordering_vs_level.py")
D47 = D48.D47
STATES, BASE_FEATURES, CATEGORIES = D47.STATES, D47.BASE_FEATURES, D47.CATEGORIES
K = len(STATES)
I_CUR, I_DQ30 = STATES.index("Current"), STATES.index("DQ30")
Tab, brier_by_class, counts, softmax = D48.Tab, D47.brier_by_class, D47.counts, D47.softmax
emit, title, GB = D47.emit, D47.title, D47.GB
R2_DROP, N_ROUNDS = D47.R2_DROP, D47.N_ROUNDS

SMOKE = os.environ.get("DIAG49_SMOKE") == "1"
D47.SMOKE = SMOKE
OUT = os.environ.get("DIAG49_OUT", os.path.join("diagnostics", "49_leaf_explosion_sweep.txt"))
MODEL_DIR = os.environ.get("DIAG49_MODELS", os.path.join("artifacts", "diag49"))
ORIGIN = ["Current"]
EXPECT_COUNTS = {"train": [30959138, 268335, 2289, 1159, 259277, 55], "held-out": [7737923, 67032, 543, 270, 64833, 19]}
EXPECT_BRIER_48_R2, EXPECT_PRIOR = 0.0428301, 0.0332949
REF48_R2 = {"auc_dq30": 0.72976, "auc_leave": 0.60592}
GATE = 0.720
HESS = [1e-3, 10.0, 1000.0]
L2 = [0.0, 10.0, 1000.0]
GRID = [(h, l) for h in HESS for l in L2]                       # run 1 = (1e-3, 0) = the defaults


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(D47.LINES) + "\n")


D47.write_report = write_report                                 # MemWatch's on_abort (in diagnostic 47) writes THIS report
D47.OUT = OUT


def tag(h, l):
    return f"hess{h:g}_l2{l:g}"


def one_run(lgb, cap, k, h, l, Xtr, ytr, Xte, yte, feats):
    params = dict(D47.PARAMS, min_sum_hessian_in_leaf=h, lambda_l2=l)
    r = {"k": k, "h": h, "l2": l, "params": params}
    print(f"  [run {k}: min_sum_hessian_in_leaf={h:g}, lambda_l2={l:g}] training ...", flush=True)
    watch = D47.MemWatch(on_abort=D47.on_abort, abort_mb=700)
    watch.phase = f"run {k}"
    watch.start()
    cap.msgs.clear()
    t0 = time.perf_counter()
    ds = lgb.Dataset(Xtr, label=ytr.astype(np.int64), categorical_feature=["property_state"], free_raw_data=True, params=params)
    booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
    r["t_train"] = time.perf_counter() - t0
    del ds
    gc.collect()
    assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
    assert booster.pandas_categorical == [CATEGORIES["property_state"]]
    r["msgs"] = list(cap.msgs)
    r["trees"] = booster.num_trees()
    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = "current_R2_" + tag(h, l)
    booster.save_model(os.path.join(MODEL_DIR, stem + ".txt"))
    with open(os.path.join(MODEL_DIR, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"origin_states": ORIGIN, "features": feats, "categorical": ["property_state"], "class_order": STATES, "params": params,
                   "num_boost_round": N_ROUNDS, "window": [D47.WIN_LO, D47.WIN_HI], "split_seed": D47.SEED, "train_fraction": D47.TRAIN_FRAC,
                   "run": k, "made_by": "diagnostics/49_leaf_explosion_sweep.py"}, f, indent=1)
    r["model_path"] = os.path.join(MODEL_DIR, stem + ".txt")
    t1 = time.perf_counter()
    raw = np.asarray(booster.predict(Xte, raw_score=True), dtype=np.float64)
    r["t_pred"] = time.perf_counter() - t1
    del booster
    assert raw.shape == (len(yte), K)
    r["spread"] = raw.max(axis=0) - raw.min(axis=0)
    P = softmax(raw)
    del raw
    yi = yte.astype(np.int64)
    cls = brier_by_class(P, yi)
    r["cls"], r["brier"] = cls, float(cls.sum())
    r["mean_p"] = P.mean(axis=0)
    s_leave, s_dq30 = 1.0 - P[:, I_CUR], P[:, I_DQ30].copy()
    del P
    r["share0"], r["share1"] = float((s_leave == 0.0).mean()), float((s_leave == 1.0).mean())
    tl = Tab(s_leave, yi != I_CUR)
    r["auc_leave"], r["distinct_leave"], r["largest_leave"] = tl.auc(), len(tl.u), float(tl.cnt.max() / tl.n)
    td = Tab(s_dq30, yi == I_DQ30)
    r["auc_dq30"], r["distinct_dq30"], r["largest_dq30"] = td.auc(), len(td.u), float(td.cnt.max() / td.n)
    r["dec"] = td.deciles()
    r["events_dq30"], r["events_leave"] = int(td.P), int(tl.P)
    del tl, td, s_leave, s_dq30
    watch.stop()
    gc.collect()
    r["rss"], r["peak"], r["min_avail"] = watch._peak, watch.peak_rss(), watch.min_avail
    r["secs"] = r["t_train"] + r["t_pred"]
    return r


def emit_detail(r, cte, pr_cls):
    title(f"RUN {r['k']}: min_sum_hessian_in_leaf = {r['h']:g}, lambda_l2 = {r['l2']:g}")
    emit("params: " + ", ".join(f"{k}={v!r}" for k, v in r["params"].items()) + f";  num_boost_round={N_ROUNDS};  trees {r['trees']};  model: {r['model_path']}")
    emit("library messages: " + (" | ".join(f"{m} x{r['msgs'].count(m)}" for m in sorted(set(r["msgs"]))) if r["msgs"] else "none"))
    emit(f"COST: wall-clock lgb.train (Dataset construction included) {r['t_train']:.1f} s ({r['t_train'] / 60:.2f} min); held-out predict {r['t_pred']:.1f} s;  "
         f"peak RSS in-run sampled {r['rss'] / GB:.2f} GB, process peak {r['peak'] / GB:.2f} GB;  min system available {r['min_avail'] / GB:.2f} GB")
    emit()
    emit("ORDERING  (held-out rows, raw outputs)")
    emit(f"  AUC, moves to DQ30   (events {r['events_dq30']:,})   : {r['auc_dq30']:.5f}   <- the gate quantity (gate {GATE:.3f}: {'ELIGIBLE' if r['auc_dq30'] >= GATE else 'NOT eligible'})")
    emit(f"  AUC, leaves Current  (events {r['events_leave']:,}) : {r['auc_leave']:.5f}")
    emit(f"  decile table, moves to DQ30 (decile 1 = lowest P(DQ30); equal-count groups, tied blocks shared pro rata)")
    emit(f"  {'decile':>6}{'rows':>12}{'events':>11}{'actual rate':>13}{'mean predicted':>16}{'distinct values':>17}")
    for i, (n_, e_, ps_, nd) in enumerate(r["dec"]):
        emit(f"  {i + 1:>6}{int(n_):>12,}{e_:>11.1f}{100 * e_ / n_:>12.4f}%{100 * ps_ / n_:>15.4f}%{nd:>17,}")
    emit()
    emit("LEVELS  (held-out rows, raw outputs, no calibration)")
    emit(f"  multiclass Brier {r['brier']:.7f};  prior-only reference on this window {EXPECT_PRIOR:.7f};  difference {r['brier'] - EXPECT_PRIOR:+.7f}")
    emit(f"  {'class':<10}{'Brier term':>13}{'prior-only term':>17}{'model - prior':>15}{'held-out freq':>16}{'mean predicted':>16}{'pred / actual':>15}")
    for j, s in enumerate(STATES):
        ratio = f"{r['mean_p'][j] / (cte[j] / cte.sum()):>14.3f}x" if cte[j] else f"{'n/a (0)':>15}"
        emit(f"  {s:<10}{r['cls'][j]:>13.8f}{pr_cls[j]:>17.8f}{r['cls'][j] - pr_cls[j]:>+15.8f}{cte[j] / cte.sum():>16.10f}{r['mean_p'][j]:>16.10f}{ratio}")
    emit(f"  {'TOTAL':<10}{r['cls'].sum():>13.8f}{pr_cls.sum():>17.8f}{r['cls'].sum() - pr_cls.sum():>+15.8f}   (sum of the six terms)")
    emit()
    emit("SATURATION  (reported whether or not it is judged on)")
    emit(f"  share of held-out rows with leaves-Current score exactly 0: {100 * r['share0']:.4f}%;  exactly 1: {100 * r['share1']:.4f}%")
    emit(f"  leaves-Current score: {r['distinct_leave']:,} distinct predicted values, largest tie block {100 * r['largest_leave']:.4f}% of rows")
    emit(f"  P(DQ30) score       : {r['distinct_dq30']:,} distinct predicted values, largest tie block {100 * r['largest_dq30']:.4f}% of rows")
    emit(f"  raw-score spread across held-out rows (max - min), per class: " + "  ".join(f"{s} {v:.4g}" for s, v in zip(STATES, r["spread"]))
         + f";  largest {r['spread'].max():.4g}  (the six log train class frequencies span 13.24)")


def main():
    import lightgbm as lgb

    cap = D47.Capture()
    lgb.register_logger(cap)
    emit("LEAF-EXPLOSION SWEEP: min_sum_hessian_in_leaf x lambda_l2, CURRENT-origin, R2 borrower-only features   (measurement + one pre-registered selection)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs" + ("   SMOKE RUN (first partition only)" if SMOKE else ""))
    emit("Feature set: R2 borrower-only (macro excluded). Reason, measured in diagnostic 48 and not a preference: DQ30 AUC 0.647 with macro (R1) vs 0.730 without (R2).")
    emit("Caveat carried from 47/48: est_ltv is borrower-level but contains an HPI component.")
    emit(f"Pre-registered rule: eligible iff DQ30 AUC >= {GATE:.3f}; among eligible, lowest held-out Brier wins; none eligible -> pick nothing; winner = run 1 -> pick nothing.")
    title("SETUP")
    X, y, fp, fs, codes, n_null, manifest, n_loans = D47.load_rows(ORIGIN, True)
    emit(f"panel build {manifest['build_stamp']}; window role == 'fit', {D47.WIN_LO} <= from_period <= {D47.WIN_HI}; from_state == Current")
    emit(f"rows {len(y):,}; loans {n_loans:,}; null/unknown labels dropped {n_null:,}")
    in_train, nl, n_tr = D47.split_flags(codes)
    emit(f"split by loan_id: seed {D47.SEED}, train fraction {D47.TRAIN_FRAC} -> {n_tr:,} of {nl:,} loans to train, {nl - n_tr:,} held out; no loan on both sides (asserted)")
    ctr, cte = counts(y[in_train].astype(np.int64)), counts(y[~in_train].astype(np.int64))
    emit(f"  {'class':<10}{'TRAIN rows':>14}{'expected (48)':>16}{'HELD-OUT rows':>16}{'expected (48)':>16}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(ctr[j]):>14,}{EXPECT_COUNTS['train'][j]:>16,}{int(cte[j]):>16,}{EXPECT_COUNTS['held-out'][j]:>16,}")
    same = list(map(int, ctr)) == EXPECT_COUNTS["train"] and list(map(int, cte)) == EXPECT_COUNTS["held-out"]
    emit(f"per-class counts vs diagnostic 48's table: {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    if not same and not SMOKE:
        emit("STOP: the population or split does not reproduce diagnostic 48's. Nothing was trained.")
        write_report()
        sys.exit(1)
    del codes, fs
    feats = [f for f in BASE_FEATURES if f not in R2_DROP]
    Xtr, ytr = X.loc[in_train, feats].reset_index(drop=True), y[in_train]
    Xte, yte = X.loc[~in_train, feats].reset_index(drop=True), y[~in_train]
    del X, y, fp, in_train
    gc.collect()
    freq_tr = counts(ytr.astype(np.int64)) / len(ytr)
    pr_cls = brier_by_class(np.broadcast_to(freq_tr, (len(yte), K)), yte.astype(np.int64))
    emit(f"features ({len(feats)}): {feats}; categorical ['property_state']; training frame built once ({len(ytr):,} train rows, {len(yte):,} held-out rows) and reused by all nine runs")
    emit(f"prior-only reference on this window: held-out Brier {pr_cls.sum():.7f} (expected {EXPECT_PRIOR:.7f}: "
         f"{'MATCH' if abs(round(float(pr_cls.sum()), 7) - EXPECT_PRIOR) < 1.5e-7 else 'DIFFERENT'})")
    write_report()

    title("RUN LOG  (one line per run, as they finish)")
    results = []
    for k, (h, l) in enumerate(GRID, start=1):
        r = one_run(lgb, cap, k, h, l, Xtr, ytr, Xte, yte, feats)
        results.append(r)
        emit(f"run {k} (min_sum_hessian_in_leaf {h:g}, lambda_l2 {l:g}): Brier {r['brier']:.7f}, DQ30 AUC {r['auc_dq30']:.5f}, leaves-Current AUC {r['auc_leave']:.5f}, "
             f"share exactly 0 {100 * r['share0']:.2f}%, {r['t_train']:.0f} s, process peak {r['peak'] / GB:.2f} GB, min system available {r['min_avail'] / GB:.2f} GB")
        write_report()
        if k == 1:
            ok = abs(round(r["brier"], 7) - EXPECT_BRIER_48_R2) < 1.5e-7
            emit(f"  REPRODUCTION of diagnostic 48 R2: held-out Brier {r['brier']:.7f} vs {EXPECT_BRIER_48_R2:.7f} -> "
                 + ("MATCH to seven decimals" if ok else f"!!! MISMATCH (difference {r['brier'] - EXPECT_BRIER_48_R2:+.3e}) !!!")
                 + f";  DQ30 AUC {r['auc_dq30']:.5f} vs 48's {REF48_R2['auc_dq30']:.5f};  leaves-Current AUC {r['auc_leave']:.5f} vs {REF48_R2['auc_leave']:.5f}")
            if not ok and not SMOKE:
                emit("STOP: run 1 does not reproduce diagnostic 48 R2. The remaining runs are not executed; this mismatch is the finding.")
                write_report()
                sys.exit(1)
            write_report()
    del Xtr, Xte
    gc.collect()

    title("SUMMARY TABLE  (one row per run; held-out rows, raw outputs)")
    emit(f"{'run':>3}{'min_sum_hess':>14}{'lambda_l2':>11}{'DQ30 AUC':>10}{'leave AUC':>11}{'Brier':>11}{'Brier-prior':>13}{'share =0':>10}{'share =1':>10}"
         f"{'largest raw spread':>20}{'wall-clock s':>14}")
    for r in results:
        emit(f"{r['k']:>3}{r['h']:>14g}{r['l2']:>11g}{r['auc_dq30']:>10.5f}{r['auc_leave']:>11.5f}{r['brier']:>11.7f}{r['brier'] - EXPECT_PRIOR:>+13.7f}"
             f"{100 * r['share0']:>9.3f}%{100 * r['share1']:>9.3f}%{r['spread'].max():>20.4g}{r['t_train']:>14.1f}")
    emit(f"(prior-only Brier on this window {EXPECT_PRIOR:.7f}; AUC gate for eligibility {GATE:.3f} on the DQ30 column)")
    for r in results:
        emit_detail(r, cte, pr_cls)

    title("PRE-REGISTERED SELECTION RULE: mechanical outcome")
    elig = [r for r in results if r["auc_dq30"] >= GATE]
    emit(f"{'run':>3}{'DQ30 AUC':>11}{'gate >= 0.720':>16}{'Brier':>12}")
    for r in results:
        emit(f"{r['k']:>3}{r['auc_dq30']:>11.5f}{('eligible' if r['auc_dq30'] >= GATE else 'not eligible'):>16}{r['brier']:>12.7f}")
    if not elig:
        outcome = "NO RUN IS ELIGIBLE: no setting in the grid preserved ordering at the gate. Nothing is picked."
        winner = None
    else:
        winner = min(elig, key=lambda r: r["brier"])
        if winner["k"] == 1:
            outcome = (f"The winner is run 1 (the defaults, Brier {winner['brier']:.7f}) among {len(elig)} eligible run(s): the levers made no difference under the rule. "
                       f"Nothing is picked.")
        else:
            outcome = (f"The winner is run {winner['k']} (min_sum_hessian_in_leaf {winner['h']:g}, lambda_l2 {winner['l2']:g}): DQ30 AUC {winner['auc_dq30']:.5f}, "
                       f"Brier {winner['brier']:.7f} (run 1: {results[0]['brier']:.7f}), among {len(elig)} eligible run(s). Its model: {winner['model_path']}")
    emit(outcome)

    title("DID SATURATION / BRIER MOVE TOGETHER WITH AUC, OR INDEPENDENTLY OF IT?  (all figures relative to run 1)")
    r1 = results[0]
    emit(f"{'run':>3}{'d DQ30 AUC':>12}{'d leave AUC':>13}{'d Brier':>12}{'d share =0 (pp)':>17}{'d largest spread':>20}")
    for r in results[1:]:
        emit(f"{r['k']:>3}{r['auc_dq30'] - r1['auc_dq30']:>+12.5f}{r['auc_leave'] - r1['auc_leave']:>+13.5f}{r['brier'] - r1['brier']:>+12.7f}"
             f"{100 * (r['share0'] - r1['share0']):>+17.3f}{r['spread'].max() - r1['spread'].max():>+20.4g}")
    better = [r for r in results[1:] if r["brier"] < r1["brier"]]
    hold = [r for r in better if r["auc_dq30"] >= GATE]
    lose = [r for r in better if r["auc_dq30"] < GATE]
    emit(f"runs with a lower Brier than run 1: {[r['k'] for r in better] or 'none'}")
    emit(f"  of those, DQ30 AUC still >= the gate (levels improved while ordering held): {[r['k'] for r in hold] or 'none'}"
         + ("" if not hold else "  (DQ30 AUC change vs run 1: " + ", ".join(f"run {r['k']} {r['auc_dq30'] - r1['auc_dq30']:+.5f}" for r in hold) + ")"))
    emit(f"  of those, DQ30 AUC below the gate (Brier improvement came with an AUC fall below the gate): {[r['k'] for r in lose] or 'none'}"
         + ("" if not lose else "  (DQ30 AUC change vs run 1: " + ", ".join(f"run {r['k']} {r['auc_dq30'] - r1['auc_dq30']:+.5f}" for r in lose) + ")"))
    a, b = np.array([r["auc_dq30"] for r in results]), np.array([r["brier"] for r in results])
    s0 = np.array([r["share0"] for r in results])
    emit(f"across the nine runs: Pearson correlation of DQ30 AUC with Brier {np.corrcoef(a, b)[0, 1]:+.3f}; of DQ30 AUC with share exactly 0 "
         f"{(np.corrcoef(a, s0)[0, 1] if s0.std() > 0 else float('nan')):+.3f}" + "  (n = 9; descriptive only)")
    sat = [r for r in results if r["share0"] < r1["share0"]]
    emit(f"runs with a lower share of exactly-0 scores than run 1: {[r['k'] for r in sat] or 'none'}; their DQ30 AUC changes: "
         + (", ".join(f"run {r['k']} {r['auc_dq30'] - r1['auc_dq30']:+.5f}" for r in sat) or "n/a"))

    title("FINDINGS  (measured only; no interpretation, no recommendation)")
    emit(f"- Run 1 (defaults; reproduction of diagnostic 48 R2's Brier 0.0428301: {'MATCH' if abs(round(r1['brier'], 7) - EXPECT_BRIER_48_R2) < 1.5e-7 else 'MISMATCH'}): Brier {r1['brier']:.7f}, DQ30 AUC {r1['auc_dq30']:.5f}, leaves-Current AUC {r1['auc_leave']:.5f}, "
         f"share exactly 0 {100 * r1['share0']:.2f}%, share exactly 1 {100 * r1['share1']:.2f}%, largest raw spread {r1['spread'].max():.4g}.")
    best_b = min(results, key=lambda r: r["brier"])
    best_a = max(results, key=lambda r: r["auc_dq30"])
    low_s = min(results, key=lambda r: r["share0"])
    low_sp = min(results, key=lambda r: r["spread"].max())
    emit(f"- Lowest Brier in the grid: run {best_b['k']} ({best_b['h']:g}, {best_b['l2']:g}) {best_b['brier']:.7f} (DQ30 AUC {best_b['auc_dq30']:.5f}); prior-only {EXPECT_PRIOR:.7f}.")
    emit(f"- Highest DQ30 AUC: run {best_a['k']} ({best_a['h']:g}, {best_a['l2']:g}) {best_a['auc_dq30']:.5f}; lowest DQ30 AUC: "
         f"{min(results, key=lambda r: r['auc_dq30'])['auc_dq30']:.5f}.")
    emit(f"- Lowest share of exactly-0 scores: run {low_s['k']} ({low_s['h']:g}, {low_s['l2']:g}) {100 * low_s['share0']:.4f}%; smallest largest-raw-spread: "
         f"run {low_sp['k']} ({low_sp['h']:g}, {low_sp['l2']:g}) {low_sp['spread'].max():.4g}.")
    emit(f"- Eligible runs (DQ30 AUC >= {GATE:.3f}): {[r['k'] for r in elig] or 'none'}. Rule outcome: {outcome}")
    emit(f"- Runs with lower Brier than run 1: {[r['k'] for r in better] or 'none'}; with ordering held at the gate: {[r['k'] for r in hold] or 'none'}; with AUC below the gate: {[r['k'] for r in lose] or 'none'}.")
    emit()
    emit(f"PEAK RSS (src.memwatch): process peak {max(r['peak'] for r in results) / GB:.2f} GB; lowest system-available memory seen "
         f"{min(r['min_avail'] for r in results) / GB:.2f} GB; total training wall-clock {sum(r['t_train'] for r in results) / 60:.1f} min")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped (no rows were sampled): {e}")
        write_report()
        raise
