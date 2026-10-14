"""Pooled model: capacity x run length, crossed.   MEASUREMENT ONLY.

    python diagnostics/54_pooled_capacity_x_runlength.py   # from the repo root

Diagnostic 52 (single-origin DQ90+, reduced capacity, with run length) tracked the duration ladder and beat prior-only. Diagnostic 53 (pooled, LightGBM
defaults) failed in both arms. Three things differ between them: origin scope, capacity, run length. This run holds origin scope fixed at POOLED and crosses the
other two, so the cause is isolated. Writes diagnostics/54_pooled_capacity_x_runlength.txt. No plots, no model artefacts, nothing under src/ or data/, no git
command.

EVERYTHING EXCEPT THE ARMS IS DIAGNOSTIC 53'S, imported from diagnostics/53_pooled_runlength_buckets.py and not re-derived: the panel load (two passes, labels
first, then the feature matrix written straight into preallocated column-major float32 matrices), the pooled population (role == 'fit', transient origins,
200601..200812), the by-loan 80/20 split at seed 40, the feature lists, the native categoricals, 100 rounds, seed 40, deterministic, the tables T1-T5 restricted the
same way (T1-T4 to held-out DQ90+-origin rows, T5 to all held-out rows), both priors, and the split-comparability warning block against diagnostic 52.

FOUR ARMS, a 2 x 2 of capacity x run length, columns always in this order:
  A1  LightGBM defaults (num_leaves 31, min_data_in_leaf 20), no run length        (= diagnostic 53's arm A)
  A2  defaults, with run_length and run_length_censored                            (= diagnostic 53's arm B)
  B1  reduced (num_leaves 8, min_data_in_leaf 100), no run length
  B2  reduced, with run length
A1 and A2 must reproduce diagnostic 53's per-class Brier terms (T4 and T5) to six decimals; a PASS/FAIL line says whether they do (a FAIL prints both and the run
continues). The arms are trained SEQUENTIALLY: each arm bins its own training set from the one train matrix, trains, scores the held-out rows in chunks, and frees
its booster and Dataset before the next arm starts; the train matrix is freed once the last arm has binned it.

READING AID (after T2): per arm, the predicted to-Default share of DQ90+-origin rows at run-length bucket 1 and at 7+, and their ratio, next to the realised ratio.
"""

import gc
import importlib.util
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


D53 = _load("diag53", "53_pooled_runlength_buckets.py")
from src.memwatch import MemWatch                                       # noqa: E402

OUT = os.path.join("diagnostics", "54_pooled_capacity_x_runlength.txt")
D53.OUT = OUT                                   # D53's emit / write_report / on_abort now write THIS report
emit, title, table, pct, write_report = D53.emit, D53.title, D53.table, D53.pct, D53.write_report
STATES, K, TRANSIENT, BUCKETS = D53.STATES, D53.K, D53.TRANSIENT, D53.BUCKETS
I_CUR, I_DEF, I_DQ90, I_OTHER, O_DQ90 = D53.I_CUR, D53.I_DEF, D53.I_DQ90, D53.I_OTHER, D53.O_DQ90
FEATS_A, FEATS_B, CATEGORICAL, PARAMS = D53.FEATS_A, D53.FEATS_B, D53.CATEGORICAL, D53.PARAMS
SEED, N_ROUNDS, CHUNK, GB = D53.SEED, D53.N_ROUNDS, D53.CHUNK, D53.GB
REF52, BRIER_FORMULA, bucket_of = D53.REF52, D53.BRIER_FORMULA, D53.bucket_of

DEFAULTS, REDUCED = {}, {"num_leaves": 8, "min_data_in_leaf": 100}
ARMS = [("A1", FEATS_A, DEFAULTS, "defaults, no run length"), ("A2", FEATS_B, DEFAULTS, "defaults, with run length"),
        ("B1", FEATS_A, REDUCED, "reduced, no run length"), ("B2", FEATS_B, REDUCED, "reduced, with run length")]
NAMES = [a[0] for a in ARMS]
# diagnostic 53's per-class Brier terms (class order Current, DQ30, DQ60, DQ90+, Prepaid, Default, then TOTAL), as printed in its report
REF53 = {"T4 (DQ90+ held-out rows)": {"A1": [0.46386125, 0.36585366, 0.02610228, 0.76522839, 0.04257748, 0.03793762, 1.70156068],
                                      "A2": [0.15075801, 0.10284590, 0.05617448, 0.36404234, 0.06684366, 0.04255266, 0.78321705]},
         "T5 (all held-out rows)": {"A1": [0.04851734, 0.02191249, 0.00625187, 0.01258831, 0.01438742, 0.00127358, 0.10493101],
                                    "A2": [0.03637257, 0.01549068, 0.00479790, 0.00720234, 0.01571426, 0.00057862, 0.08015637]}}


def main():
    import lightgbm as lgb

    cap = D53.Capture()
    lgb.register_logger(cap)
    watch = MemWatch(on_abort=D53.on_abort, abort_mb=700)
    watch.phase = "pass 1"
    watch.start()
    t_start = time.perf_counter()
    emit("POOLED MODEL, CAPACITY x RUN LENGTH   (diagnostic 54; measurement only)")
    emit("Population: role == 'fit', pooled transient origins, 200601..200812. T1-T4 are restricted to held-out rows with from_state == 'DQ90+' (comparable "
         "with diagnostics 52 and 53); T5 scores all held-out rows.")
    emit("Arms: " + "; ".join(f"{n} = {d}" for n, _, _, d in ARMS) + ". A1 and A2 are diagnostic 53's arms A and B.")
    emit("All tables are given once, on all rows. The censoring split of diagnostic 52 is NOT repeated: it found 42 of 40,520 held-out DQ90+ rows censored, "
         "immaterial.")

    # ---- pass 1: labels and run_length only (diagnostic 53's code) -------------------------------------------------------------
    print("pass 1: labels", flush=True)
    L = D53.read_partitions(["role", "loan_id", "from_period", "from_state", "to_state", "run_length"])
    y1, fs1, rl1 = L["y"], L["fs"], L["rl"]
    in_train, n_loans, n_tr = D53.split_flags(L["codes"])
    L["codes"] = None
    gc.collect()
    q = (fs1 == O_DQ90) & ~in_train
    cte52 = np.bincount(y1[q].astype(np.int64), minlength=K)
    bte52 = np.bincount(bucket_of(rl1[q]), minlength=4)

    title("BUCKET COUNTS  (DQ90+-origin rows per run_length bucket; reported before any score)")
    for name, sel in (("HELD-OUT", (fs1 == O_DQ90) & ~in_train), ("TRAIN", (fs1 == O_DQ90) & in_train)):
        b = bucket_of(rl1[sel])
        emit(f"{name} DQ90+-origin rows: {int(sel.sum()):,}")
        table(["bucket", "rows", "% of rows"], [[bk, f"{int((b == i).sum()):,}", pct((b == i).mean(), 8, 2)] for i, bk in enumerate(BUCKETS)]
              + [["total", f"{len(b):,}", pct(1.0, 8, 2)]], [8, 12, 12])
        emit()
    title("SPLIT COMPARABILITY  (WARNING, NOT A STOP: held-out DQ90+-origin counts of this run vs diagnostic 52)")
    pct_diff = lambda a, b: f"{100 * (a - b) / b:+.2f}%"
    rows = [[st, f"{int(cte52[j]):,}", f"{REF52['held-out classes'][j]:,}", pct_diff(int(cte52[j]), REF52["held-out classes"][j])] for j, st in enumerate(STATES)]
    rows.append(["TOTAL", f"{int(cte52.sum()):,}", f"{sum(REF52['held-out classes']):,}", pct_diff(int(cte52.sum()), sum(REF52["held-out classes"]))])
    rows += [["run_length " + bk, f"{int(bte52[i]):,}", f"{REF52['held-out buckets'][i]:,}", pct_diff(int(bte52[i]), REF52["held-out buckets"][i])]
             for i, bk in enumerate(BUCKETS)]
    table(["row", "this run", "diagnostic 52", "difference"], rows, [16, 12, 16, 12])
    emit("The two splits draw from different loan lists (52: DQ90+ loans only; this run: all pooled loans), so the same seed assigns loans differently; "
         "diagnostic 52 is a reference point, not a paired control.")
    write_report()

    # ---- priors ------------------------------------------------------------------------------------------------------------
    ytr1, fstr1 = y1[in_train].astype(np.int64), fs1[in_train]
    n_train_origin = np.bincount(fstr1, minlength=4)
    freq_pooled = np.bincount(ytr1, minlength=K) / len(ytr1)
    freq_cond = np.array([np.bincount(ytr1[fstr1 == o], minlength=K) / n_train_origin[o] for o in range(4)])      # (4, K)
    n_te_origin = np.bincount(fs1[~in_train], minlength=4)
    ctr_all, cte_all = np.bincount(ytr1, minlength=K), np.bincount(y1[~in_train].astype(np.int64), minlength=K)
    n_train_rows, n_test_rows = int(in_train.sum()), int((~in_train).sum())
    fp_rng, manifest, n_null, lens = L["fp"], L["manifest"], L["n_null"], L["lens"]
    y_ref, fs_ref = y1.copy(), fs1.copy()
    del L, y1, fs1, rl1, ytr1, fstr1, q
    gc.collect()

    # ---- pass 2: feature matrices (diagnostic 53's code) -------------------------------------------------------------------------
    watch.phase = "pass 2"
    print("pass 2: feature matrix", flush=True)
    Xtr, Xte = D53.read_features(lens, in_train, y_ref, fs_ref, n_train_rows, n_test_rows)
    label_tr = y_ref[in_train].astype(np.float32)
    yte = y_ref[~in_train].astype(np.int64)
    fste = fs_ref[~in_train]
    del y_ref, fs_ref, in_train
    gc.collect()
    sub_idx = np.flatnonzero(fste == O_DQ90)                                        # held-out DQ90+-origin rows
    b_sub = bucket_of(Xte[sub_idx, FEATS_B.index("run_length")].astype(np.int64))
    y_sub = yte[sub_idx]

    # ---- four arms, one at a time -------------------------------------------------------------------------------------------------
    sq, P_sub, t_construct, t_fit, trees, msgs, params_used = {}, {}, {}, {}, {}, {}, {}
    for k, (arm, feats, capacity, desc) in enumerate(ARMS):
        params = dict(PARAMS, **capacity)
        params_used[arm] = params
        print(f"  [arm {arm}: {desc}] binning and training {len(label_tr):,} rows ...", flush=True)
        watch.phase = f"arm {arm}"
        cap.msgs.clear()
        t0 = time.perf_counter()
        ds = lgb.Dataset(Xtr[:, :len(feats)], label=label_tr, feature_name=list(feats), categorical_feature=CATEGORICAL, free_raw_data=True,
                         params=params).construct()
        t_construct[arm] = time.perf_counter() - t0
        if k == len(ARMS) - 1:
            Xtr = None                                                              # last arm has binned: the raw train matrix is no longer needed
            gc.collect()
        t1 = time.perf_counter()
        booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
        del ds
        gc.collect()
        assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
        trees[arm], msgs[arm] = booster.num_trees(), list(cap.msgs)
        s_arm, keep = np.zeros((4, K)), []
        for a in range(0, len(yte), CHUNK):
            b = min(a + CHUNK, len(yte))
            P = np.asarray(booster.predict(Xte[a:b, :len(feats)]), dtype=np.float64)
            onehot = np.zeros_like(P)
            onehot[np.arange(b - a), yte[a:b]] = 1.0
            err = (P - onehot) ** 2
            for o in range(4):
                s_arm[o] += err[fste[a:b] == o].sum(axis=0)
            keep.append(P[fste[a:b] == O_DQ90])
            del P, onehot, err
        del booster
        gc.collect()
        sq[arm] = s_arm
        P_sub[arm] = np.concatenate(keep)
        assert P_sub[arm].shape == (len(sub_idx), K)
        t_fit[arm] = time.perf_counter() - t1
        print(f"  [arm {arm}] done: {trees[arm]} trees, construct {t_construct[arm]:.1f} s, train + predict {t_fit[arm]:.1f} s", flush=True)
    del label_tr
    # prior-only squared-error sums, pooled and origin-conditional
    for name in ("pooled", "cond"):
        s = np.zeros((4, K))
        for a in range(0, len(yte), CHUNK):
            b = min(a + CHUNK, len(yte))
            onehot = np.zeros((b - a, K))
            onehot[np.arange(b - a), yte[a:b]] = 1.0
            err = ((freq_pooled if name == "pooled" else freq_cond[fste[a:b]]) - onehot) ** 2
            for o in range(4):
                s[o] += err[fste[a:b] == o].sum(axis=0)
        sq["prior-" + name] = s
    del Xte
    gc.collect()

    P_prior_pool = np.broadcast_to(freq_pooled, (len(sub_idx), K))
    P_prior_cond = np.broadcast_to(freq_cond[O_DQ90], (len(sub_idx), K))
    cols = [(n, P_sub[n]) for n in NAMES] + [("prior-cond", P_prior_cond), ("prior-pooled", P_prior_pool)]
    heads = ["bucket", "n", "realised"] + [c[0] for c in cols]
    widths = [8, 11, 11] + [11] * len(cols)

    def leave(P):
        return 1.0 - P[:, I_DQ90]

    realised_leave = (y_sub != I_DQ90).astype(np.float64)

    title("T1  DQ90+ HELD-OUT ROWS, PER BUCKET: mean predicted probability of LEAVING DQ90+ (1 - predicted DQ90+ self-transition) vs realised share that left")
    table(heads, [[bk, f"{int((b_sub == i).sum()):,}", pct(realised_leave[b_sub == i].mean(), 8)] + [pct(leave(P)[b_sub == i].mean(), 8) for _, P in cols]
                  for i, bk in enumerate(BUCKETS)], widths)

    title("T2  DQ90+ HELD-OUT ROWS, PER BUCKET: destination split of leaving DQ90+, as shares of ALL rows in the bucket (realised vs mean predicted)")
    emit("THIS IS THE TABLE THAT ANSWERS THE QUESTION. Diagnostic 52 realised to-Default ladder: about 0.3% at run length 1 up to 6.7% at 7+.")
    emit("'other leave' is the rest of the leave (DQ30, DQ60, Prepaid), so that the three blocks sum to T1.")
    for dname, idx in (("to Default", [I_DEF]), ("to Current", [I_CUR]), ("other leave (DQ30 + DQ60 + Prepaid)", I_OTHER)):
        emit(f"  {dname}")
        real = np.isin(y_sub, idx).astype(np.float64)
        table(heads, [[bk, f"{int((b_sub == i).sum()):,}", pct(real[b_sub == i].mean(), 8)] + [pct(P[b_sub == i][:, idx].sum(axis=1).mean(), 8) for _, P in cols]
                      for i, bk in enumerate(BUCKETS)], widths)

    title("READING AID: to-Default LADDER SLOPE  (DQ90+ held-out rows; mean predicted to-Default share at run-length bucket 1 and at 7+, and their ratio)")
    real_def = (y_sub == I_DEF).astype(np.float64)
    r1, r7 = real_def[b_sub == 0].mean(), real_def[b_sub == 3].mean()
    emit(f"  {'':<12}{'bucket 1':>11}{'bucket 7+':>11}{'ratio 7+ / 1':>15}   (realised ratio about 22x)")
    emit(f"  {'REALISED':<12}{pct(r1, 10)}{pct(r7, 10)}{r7 / r1:>14.1f}x")
    for n, P in cols:
        p1, p7 = P[b_sub == 0, I_DEF].mean(), P[b_sub == 3, I_DEF].mean()
        emit(f"  {n:<12}{pct(p1, 10)}{pct(p7, 10)}" + (f"{p7 / p1:>14.1f}x" if p1 > 0 else f"{'n/a (bucket 1 is 0)':>15}"))

    title("T3  POOLED across all DQ90+ held-out rows, same quantities as T1.   *** NOT THE JUDGE: pooling hides within-bucket error ***")
    table(["pooled"] + heads[1:], [["all buckets", f"{len(y_sub):,}", pct(realised_leave.mean(), 8)] + [pct(leave(P).mean(), 8) for _, P in cols]],
          [14] + widths[1:])

    def brier_rows(s_by, n):
        mods = ["prior-pooled", "prior-cond"] + NAMES
        br = {m: s_by[m] / n for m in mods}
        rows = [[st] + [f"{br[m][j]:.8f}" for m in mods] + [f"{br[m][j] - br['prior-cond'][j]:+.8f}" for m in NAMES] for j, st in enumerate(STATES)]
        t = {m: br[m].sum() for m in mods}
        rows.append(["TOTAL"] + [f"{t[m]:.8f}" for m in mods] + [f"{t[m] - t['prior-cond']:+.8f}" for m in NAMES])
        return rows, br

    bh = ["class", "prior-pooled", "prior-cond"] + NAMES + [f"{m} - cond" for m in NAMES]
    bw = [8] + [13] * 6 + [12] * 4

    title("T4  PER-CLASS BRIER on DQ90+ held-out rows (lower is better)")
    emit(BRIER_FORMULA)
    emit(f"rows scored: {len(y_sub):,}   (differences are against the origin-conditional prior, the meaningful reference here)")
    onehot_sub = np.eye(K)[y_sub]
    sub_sq = {n: ((P_sub[n] - onehot_sub) ** 2).sum(axis=0) for n in NAMES}
    sub_sq["prior-pooled"] = ((P_prior_pool - onehot_sub) ** 2).sum(axis=0)
    sub_sq["prior-cond"] = ((P_prior_cond - onehot_sub) ** 2).sum(axis=0)
    for k_ in sub_sq:                                                               # chunked accumulation vs the stored DQ90+ subset
        assert np.allclose(sq[k_][O_DQ90], sub_sq[k_], rtol=1e-9), "chunked and subset Brier sums disagree for " + k_
    rows4, br4 = brier_rows(sub_sq, len(y_sub))
    table(bh, rows4, bw)

    title("T5  PER-CLASS BRIER on ALL held-out rows, every origin state (lower is better): the whole-model cost or gain of capacity and of run length")
    emit(BRIER_FORMULA)
    emit(f"rows scored: {len(yte):,}")
    all_sq = {m: sq[m].sum(axis=0) for m in sq}
    rows5, br5 = brier_rows(all_sq, len(yte))
    table(bh, rows5, bw)
    emit()
    emit("  Total Brier by origin state (held-out rows of that origin)")
    mods = ["prior-pooled", "prior-cond"] + NAMES
    table(["origin", "rows"] + mods + [f"{m} - cond" for m in NAMES],
          [[TRANSIENT[o], f"{int(n_te_origin[o]):,}"] + [f"{sq[m][o].sum() / n_te_origin[o]:.8f}" for m in mods]
           + [f"{(sq[m][o].sum() - sq['prior-cond'][o].sum()) / n_te_origin[o]:+.8f}" for m in NAMES] for o in range(4)], [8, 12] + [13] * 6 + [12] * 4)

    # ---- notes -----------------------------------------------------------------------------------------------------------------
    title("NOTES AND CHECKS")
    got = {"T4 (DQ90+ held-out rows)": {n: list(br4[n]) + [br4[n].sum()] for n in ("A1", "A2")},
           "T5 (all held-out rows)": {n: list(br5[n]) + [br5[n].sum()] for n in ("A1", "A2")}}
    fails = []
    for tname, per_arm in REF53.items():
        for n, ref in per_arm.items():
            for lab, g, r in zip(STATES + ["TOTAL"], got[tname][n], ref):
                if abs(g - r) >= 1e-6:
                    fails.append((tname, n, lab, g, r))
    emit(f"REPRODUCTION OF DIAGNOSTIC 53 (A1 = 53's arm A, A2 = 53's arm B; per-class and total Brier, T4 and T5, agreement to six decimals, |difference| < 1e-6, "
         f"{sum(len(v) for per in REF53.values() for v in per.values())} values compared): {'PASS' if not fails else 'FAIL'}")
    if fails:
        emit(f"  {'table':<26}{'arm':<5}{'class':<9}{'this run':>14}{'diagnostic 53':>16}{'difference':>14}")
        for tname, n, lab, g, r in fails:
            emit(f"  {tname:<26}{n:<5}{lab:<9}{g:>14.8f}{r:>16.8f}{g - r:>+14.8f}")
    emit(f"panel build {manifest['build_stamp']}; role == 'fit', from_state in {TRANSIENT}, 200601 <= from_period <= 200812 (observed {fp_rng[0]}..{fp_rng[1]}); "
         f"rows {n_train_rows + n_test_rows:,}; null labels dropped {n_null:,}")
    emit(f"split by loan: seed {SEED}, {n_tr:,} of {n_loans:,} loans to train, {n_loans - n_tr:,} held out; train rows {n_train_rows:,}, held-out rows {n_test_rows:,}; "
         f"no loan on both sides (asserted); DQ90+ held-out subset vs diagnostic 52: see SPLIT COMPARABILITY (different loan lists, not a paired control)")
    emit("TRAIN rows per origin state, and each state's share of the pooled training objective (every row enters the multiclass loss with equal weight):")
    emit(f"  {'origin':<8}{'train rows':>14}{'share of train rows':>22}{'held-out rows':>16}")
    for o, st in enumerate(TRANSIENT):
        emit(f"  {st:<8}{int(n_train_origin[o]):>14,}{pct(n_train_origin[o] / n_train_rows, 21, 4)}{int(n_te_origin[o]):>16,}")
    emit(f"  {'TOTAL':<8}{n_train_rows:>14,}{pct(1.0, 21, 4)}{n_test_rows:>16,}")
    emit(f"DQ90+-origin rows are {100 * n_train_origin[O_DQ90] / n_train_rows:.4f}% of the pooled training objective. Of the DQ90+-origin train rows, "
         f"{int(np.round(freq_cond[O_DQ90][I_DEF] * n_train_origin[O_DQ90])):,} go to Default; Default is {100 * ctr_all[I_DEF] / n_train_rows:.4f}% of all train rows.")
    emit("class counts, all train rows: " + ", ".join(f"{s} {int(c):,}" for s, c in zip(STATES, ctr_all)) + "; all held-out rows: "
         + ", ".join(f"{s} {int(c):,}" for s, c in zip(STATES, cte_all)))
    emit("prior-only, pooled (all train rows): " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, freq_pooled)))
    for o, st in enumerate(TRANSIENT):
        emit(f"prior-only, conditional on {st:<7}: " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, freq_cond[o])))
    emit(f"features, no run length (A1, B1; {len(FEATS_A)}): {FEATS_A}")
    emit(f"features, with run length (A2, B2; {len(FEATS_B)}): {FEATS_B}")
    emit("native categoricals: " + str(CATEGORICAL) + f"; common params: " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items()) + f"; num_boost_round={N_ROUNDS}")
    emit(f"  {'arm':<5}{'capacity':<44}{'trees':>7}{'construct s':>13}{'train + predict s':>19}   library messages")
    for arm, _, capacity, desc in ARMS:
        cap_txt = ("num_leaves 8, min_data_in_leaf 100" if capacity else "num_leaves 31, min_data_in_leaf 20 (defaults)") + ", " + desc.split(", ")[1]
        m = " | ".join(f"{x} x{msgs[arm].count(x)}" for x in sorted(set(msgs[arm]))) if msgs[arm] else "none"
        emit(f"  {arm:<5}{cap_txt:<44}{trees[arm]:>7}{t_construct[arm]:>13.1f}{t_fit[arm]:>19.1f}   {m}")
    emit(f"lightgbm {lgb.__version__}; total wall-clock {(time.perf_counter() - t_start) / 60:.1f} min")
    watch.stop()
    emit(f"PEAK RSS (src.memwatch): process peak {watch.peak_rss() / GB:.2f} GB; lowest system-available memory seen {watch.min_avail / GB:.2f} GB")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped (no rows were sampled): {e}")
        write_report()
        raise
