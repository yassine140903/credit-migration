"""What does run length mean INSIDE each origin state?   WRITE-UP EVIDENCE ONLY.

    python diagnostics/55_single_origin_runlength.py   # from the repo root

Diagnostic 54 showed a pooled model mis-learns run length on DQ90+ rows. This run fits each origin group separately, with and without run length, to show what
run length does inside each state. No model is shipped, no forecaster, no chaining, no evaluate rows. Writes diagnostics/55_single_origin_runlength.txt. No plots,
no model artefacts, nothing under src/ or data/, no git command.

POPULATION AND SPLIT are diagnostic 54's (through diagnostic 53's code, imported): role == 'fit', transient origins, 200601..200812, by-loan 80/20 at seed 40 over
the POOLED loan list (so the held-out loans are 54's, not 52's). The rows are then subset by origin state.

GROUPS AND MODELS (each with and without run length; 100 rounds, seed 40, deterministic, objective multiclass, num_class 6 in every model):
  Current      LightGBM defaults (num_leaves 31, min_data_in_leaf 20)    source: diagnostic 44 (Current and pooled capacity check)      2 models
  DQ30 + DQ60  BOTH capacities: defaults and reduced (8 / 100)             source: none; nothing has measured this group                   4 models
               (one model over the two origins, as in the grouped architecture; from_state is a feature)
  DQ90+        reduced (num_leaves 8, min_data_in_leaf 100)                 source: diagnostic 41 (reduced capacity wins on this population) 2 models
FEATURES. Diagnostic 54's: fico ltv dti rate_spread property_state est_ltv unemp_level unemp_chg_12m, plus run_length and run_length_censored in the "with run
length" arm; from_state is dropped for the single-origin models (Current, DQ90+) and kept for DQ30 + DQ60. property_state (and from_state where used) are native
categoricals. No loan_age.

MEMORY. Each group's features are read straight into its own preallocated column-major float32 matrix (Current alone; DQ30, DQ60 and DQ90+ together in one small
matrix), so the 31.5M Current rows are never held alongside the pooled matrices. Within a group all models are binned first, the raw matrix is freed, and the models
are then trained one at a time, each booster freed before the next.

SCORING. Held-out rows of the group only. Per-class Brier against the group's CONDITIONAL prior (the group's train class frequencies applied to every held-out row of
the group; for DQ30 + DQ60 the per-origin conditional prior is also reported). Per run-length bucket (1, 2-3, 4-6, 7+) and origin: mean predicted probability of every
class, with the origin's own class being the probability of STAYING, against the realised share, from the with-run-length models only (T2).
"""

import gc
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join("diagnostics", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D53 = _load("diag53", "53_pooled_runlength_buckets.py")
from src import panel_io                                                # noqa: E402
from src.memwatch import MemWatch                                       # noqa: E402

OUT = os.path.join("diagnostics", "55_single_origin_runlength.txt")
D53.OUT = OUT                                   # D53's emit / write_report / on_abort now write THIS report
emit, title, table, pct, write_report = D53.emit, D53.title, D53.table, D53.pct, D53.write_report
STATES, K, TRANSIENT, BUCKETS = D53.STATES, D53.K, D53.TRANSIENT, D53.BUCKETS
CATEGORIES, PARAMS, SEED, N_ROUNDS, CHUNK, GB = D53.CATEGORIES, D53.PARAMS, D53.SEED, D53.N_ROUNDS, D53.CHUNK, D53.GB
BRIER_FORMULA, bucket_of = D53.BRIER_FORMULA, D53.bucket_of
WIN_LO, WIN_HI = D53.WIN_LO, D53.WIN_HI

BASE8 = ["fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv", "unemp_level", "unemp_chg_12m"]
RL2 = ["run_length", "run_length_censored"]
DEFAULTS, REDUCED = {}, {"num_leaves": 8, "min_data_in_leaf": 100}
FLAT_PP = 0.005                                 # direction of travel: |change| below 0.5 percentage point is called flat

# model spec: (name, features, capacity, capacity label, has run length)
GROUPS = [
    {"name": "Current", "origins": [0], "matrix": "cur", "cap_source": "LightGBM defaults (num_leaves 31, min_data_in_leaf 20), source: diagnostic 44",
     "models": [("noRL", BASE8, DEFAULTS, "defaults", False), ("RL", BASE8 + RL2, DEFAULTS, "defaults", True)]},
    {"name": "DQ30+DQ60", "origins": [1, 2], "matrix": "small",
     "cap_source": "BOTH defaults (31 / 20) and reduced (8 / 100); source: none, nothing has measured this group",
     "models": [("D-noRL", BASE8 + ["from_state"], DEFAULTS, "defaults", False), ("D-RL", BASE8 + ["from_state"] + RL2, DEFAULTS, "defaults", True),
                ("R-noRL", BASE8 + ["from_state"], REDUCED, "reduced", False), ("R-RL", BASE8 + ["from_state"] + RL2, REDUCED, "reduced", True)]},
    {"name": "DQ90+", "origins": [3], "matrix": "small", "cap_source": "reduced (num_leaves 8, min_data_in_leaf 100), source: diagnostic 41",
     "models": [("noRL", BASE8, REDUCED, "reduced", False), ("RL", BASE8 + RL2, REDUCED, "reduced", True)]},
]
MATRIX_COLS = {"cur": BASE8 + RL2, "small": BASE8 + ["from_state"] + RL2}
MATRIX_ORIGINS = {"cur": [0], "small": [1, 2, 3]}


def read_group(lens, in_train, y_ref, fs_ref, origins, cols):
    """Diagnostic 53's read_features, restricted to the rows whose origin is in `origins`: the same rows in the same order as pass 1, written into two
    preallocated column-major float32 matrices (train, held-out) with columns `cols`. Returns Xtr, Xte, ytr, yte, fs_tr, fs_te."""
    manifest = panel_io.read_manifest()
    sel_all = np.isin(fs_ref, origins)
    n_tr, n_te = int((sel_all & in_train).sum()), int((sel_all & ~in_train).sum())
    Xtr = np.empty((n_tr, len(cols)), dtype=np.float32, order="F")
    Xte = np.empty((n_te, len(cols)), dtype=np.float32, order="F")
    read_cols = ["role", "from_period", "from_state", "to_state"] + [c for c in cols if c != "from_state"]
    off = otr = ote = 0
    for p, n_expect in zip(manifest["partitions"], lens):
        d = panel_io.load_panel(roles=["fit"], columns=read_cols, quarters=[p["quarter"]])
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = d["from_state"].isin(TRANSIENT).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        m &= y >= 0
        y = y[m]
        d = d[m]
        n = len(d)
        assert n == n_expect, "pass 2 row count differs from pass 1 in " + p["quarter"]
        fs = d["from_state"].astype(str).astype(pd.CategoricalDtype(TRANSIENT)).cat.codes.to_numpy().astype(np.int8)
        assert np.array_equal(y.astype(np.int8), y_ref[off:off + n]) and np.array_equal(fs, fs_ref[off:off + n]), "pass 2 rows differ from pass 1"
        sel = np.isin(fs, origins)
        seg_tr, seg_te = sel & in_train[off:off + n], sel & ~in_train[off:off + n]
        a_tr, a_te = int(seg_tr.sum()), int(seg_te.sum())
        for j, c in enumerate(cols):
            if c == "from_state":
                v = fs.astype(np.float32)
            elif c in CATEGORIES:
                v = d[c].cat.set_categories(CATEGORIES[c]).cat.codes.to_numpy().astype(np.float32)
                v[v < 0] = np.nan
            elif c in RL2:
                assert not d[c].isna().any(), c + " has nulls"
                v = d[c].astype(np.float32).to_numpy()
                if c == "run_length":
                    assert v[sel].min() >= 1 if sel.any() else True, "run_length below 1"
            else:
                v = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
            Xtr[otr:otr + a_tr, j] = v[seg_tr]
            Xte[ote:ote + a_te, j] = v[seg_te]
        otr += a_tr
        ote += a_te
        off += n
        print(f"  loaded {p['quarter']}: {n:,} rows ({a_tr + a_te:,} in the group)", flush=True)
        del d, v, fs, y
        gc.collect()
    assert off == len(y_ref) and otr == n_tr and ote == n_te, "row totals differ from pass 1"
    ytr, yte = y_ref[sel_all & in_train].astype(np.int64), y_ref[sel_all & ~in_train].astype(np.int64)
    return Xtr, Xte, ytr, yte, fs_ref[sel_all & in_train], fs_ref[sel_all & ~in_train]


def cols_of(X, cols, feats, rows=None):
    """The model's columns of a group matrix: a contiguous leading-block view where it is one, else a (small) copy."""
    idx = [cols.index(f) for f in feats]
    if rows is None:
        return X[:, :len(idx)] if idx == list(range(len(idx))) else X[:, idx]
    a, b = rows
    return X[a:b, :len(idx)] if idx == list(range(len(idx))) else X[a:b][:, idx]


def fit_group(lgb, cap, watch, grp, holder, X_te, ytr, yte, fs_tr, fs_te, cols):
    """Bin every model of the group, free the train matrix, then train and score the models one at a time.
    The train matrix arrives in a one-element list so that the caller holds no reference to it while the models train."""
    X_tr = holder.pop()
    origins = grp["origins"]
    o_te = np.searchsorted(origins, fs_te)                                           # local origin index of every held-out row
    n_o = len(origins)
    rl = X_te[:, cols.index("run_length")].astype(np.int64)
    b_te = bucket_of(rl).astype(np.int64)
    gi_te = o_te * 4 + b_te
    R = {"o_te": o_te, "n_o": n_o,
         "cnt": np.bincount(gi_te * K + yte, minlength=n_o * 4 * K).reshape(n_o, 4, K),   # realised class counts by origin and bucket
         "ctr": np.array([np.bincount(ytr[fs_tr == o], minlength=K) for o in origins]),   # train class counts by origin
         "ctr_grp": np.bincount(ytr, minlength=K), "n_tr": len(ytr), "n_te": len(yte),
         "n_tr_o": np.array([(fs_tr == o).sum() for o in origins]), "n_te_o": np.array([(fs_te == o).sum() for o in origins]), "models": {}}
    label_tr = ytr.astype(np.float32)
    watch.phase = f"{grp['name']}: binning"
    dsets, t_construct = {}, {}
    for name, feats, capacity, cap_label, has_rl in grp["models"]:
        params = dict(PARAMS, **capacity)
        t0 = time.perf_counter()
        dsets[name] = lgb.Dataset(cols_of(X_tr, cols, feats), label=label_tr, feature_name=list(feats),
                                  categorical_feature=[f for f in feats if f in CATEGORIES], free_raw_data=True, params=params).construct()
        t_construct[name] = time.perf_counter() - t0
    del X_tr, label_tr
    gc.collect()
    for name, feats, capacity, cap_label, has_rl in grp["models"]:
        params = dict(PARAMS, **capacity)
        print(f"  [{grp['name']} / {name}: {cap_label}, {'with' if has_rl else 'no'} run length] training {R['n_tr']:,} rows ...", flush=True)
        watch.phase = f"{grp['name']} / {name}"
        cap.msgs.clear()
        t1 = time.perf_counter()
        booster = lgb.train(params, dsets.pop(name), num_boost_round=N_ROUNDS)
        gc.collect()
        assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
        S, Psum = np.zeros((n_o, K)), np.zeros((n_o * 4, K))
        for a in range(0, len(yte), CHUNK):
            b = min(a + CHUNK, len(yte))
            P = np.asarray(booster.predict(cols_of(X_te, cols, feats, (a, b))), dtype=np.float64)
            onehot = np.zeros_like(P)
            onehot[np.arange(b - a), yte[a:b]] = 1.0
            err = (P - onehot) ** 2
            for o in range(n_o):
                S[o] += err[o_te[a:b] == o].sum(axis=0)
            for j in range(K):
                Psum[:, j] += np.bincount(gi_te[a:b], weights=P[:, j], minlength=n_o * 4)
            del P, onehot, err
        R["models"][name] = {"S": S, "Pmean": (Psum.reshape(n_o, 4, K) / np.maximum(R["cnt"].sum(axis=2), 1)[:, :, None]), "trees": booster.num_trees(),
                             "msgs": list(cap.msgs), "t_construct": t_construct[name], "t_fit": time.perf_counter() - t1, "cap": cap_label, "rl": has_rl}
        del booster
        gc.collect()
        print(f"  [{grp['name']} / {name}] done: {R['models'][name]['trees']} trees, {R['models'][name]['t_fit']:.1f} s", flush=True)
    return R


def prior_brier_sums(R, p_by_origin):
    """Exact Brier sums of a prior-only predictor on the held-out rows: sum_i (p_j - o_ij)^2 = n p_j^2 - 2 p_j c_j + c_j, per origin."""
    c = R["cnt"].sum(axis=1)                                                         # (n_o, K) realised class counts
    n = R["n_te_o"][:, None]
    return n * p_by_origin ** 2 - 2 * p_by_origin * c + c


def main():
    import lightgbm as lgb

    cap = D53.Capture()
    lgb.register_logger(cap)
    watch = MemWatch(on_abort=D53.on_abort, abort_mb=700)
    watch.phase = "pass 1"
    watch.start()
    t_start = time.perf_counter()
    emit("WHAT DOES RUN LENGTH MEAN INSIDE EACH ORIGIN STATE?   (diagnostic 55; write-up evidence only: no model is shipped, no forecaster, no chaining, no evaluate rows)")
    emit("Population and split: diagnostic 54's (role == 'fit', transient origins, 200601..200812, by-loan 80/20 at seed 40 over the pooled loan list), then subset by "
         "origin state. All tables use HELD-OUT rows of the group only.")
    emit("Capacity choices and sources:")
    for g in GROUPS:
        emit(f"  {g['name']:<10} {g['cap_source']}")

    # ---- pass 1: labels and run_length only (diagnostic 53's code) -------------------------------------------------------------
    print("pass 1: labels", flush=True)
    L = D53.read_partitions(["role", "loan_id", "from_period", "from_state", "to_state", "run_length"])
    y1, fs1, rl1 = L["y"], L["fs"], L["rl"]
    in_train, n_loans, n_tr_loans = D53.split_flags(L["codes"])
    L["codes"] = None
    gc.collect()
    fp_rng, manifest, n_null, lens = L["fp"], L["manifest"], L["n_null"], L["lens"]

    title("BUCKET COUNTS  (rows per run_length bucket, by origin state; reported before any score)")
    for o, st in enumerate(TRANSIENT):
        bt, bh = bucket_of(rl1[(fs1 == o) & in_train]), bucket_of(rl1[(fs1 == o) & ~in_train])
        emit(f"origin {st}: train rows {len(bt):,}, held-out rows {len(bh):,}")
        table(["bucket", "train rows", "% of train", "held-out rows", "% of held-out"],
              [[bk, f"{int((bt == i).sum()):,}", pct((bt == i).mean(), 8, 2), f"{int((bh == i).sum()):,}", pct((bh == i).mean(), 8, 2)] for i, bk in enumerate(BUCKETS)]
              + [["total", f"{len(bt):,}", pct(1.0, 8, 2), f"{len(bh):,}", pct(1.0, 8, 2)]], [8, 14, 12, 16, 15])
        emit()
    write_report()
    y_ref, fs_ref = y1.copy(), fs1.copy()
    del L, y1, fs1, rl1
    gc.collect()

    # ---- fit the three groups, Current first ---------------------------------------------------------------------------------------
    results = {}
    order = ["cur", "small"]
    for mkey in order:
        cols = MATRIX_COLS[mkey]
        watch.phase = f"pass 2 ({mkey})"
        print(f"pass 2: feature matrix for origins {[TRANSIENT[o] for o in MATRIX_ORIGINS[mkey]]}", flush=True)
        Xtr, Xte, ytr, yte, fs_tr, fs_te = read_group(lens, in_train, y_ref, fs_ref, MATRIX_ORIGINS[mkey], cols)
        for grp in [g for g in GROUPS if g["matrix"] == mkey]:
            sel_tr, sel_te = np.isin(fs_tr, grp["origins"]), np.isin(fs_te, grp["origins"])
            if mkey == "cur":                                                        # the whole matrix is the group: no copy
                holder, Xtr = [Xtr], None
                R = fit_group(lgb, cap, watch, grp, holder, Xte, ytr, yte, fs_tr, fs_te, cols)
            else:
                R = fit_group(lgb, cap, watch, grp, [np.asfortranarray(Xtr[sel_tr])], np.asfortranarray(Xte[sel_te]), ytr[sel_tr], yte[sel_te],
                              fs_tr[sel_tr], fs_te[sel_te], cols)
            results[grp["name"]] = R
        del Xtr, Xte, ytr, yte, fs_tr, fs_te
        gc.collect()
        write_report()

    # ---- T1 ----------------------------------------------------------------------------------------------------------------------------
    title("T1  PER-CLASS BRIER on the group's held-out rows, both arms, against the group's conditional prior (lower is better)")
    emit(BRIER_FORMULA)
    for grp in GROUPS:
        R = results[grp["name"]]
        names = [m[0] for m in grp["models"]]
        n = R["n_te"]
        freq_grp = R["ctr_grp"] / R["n_tr"]
        S_prior = prior_brier_sums(R, np.broadcast_to(freq_grp, (R["n_o"], K))).sum(axis=0)
        br = {"prior-grp": S_prior / n}
        for nm in names:
            br[nm] = R["models"][nm]["S"].sum(axis=0) / n
        emit()
        emit(f"--- {grp['name']}: {n:,} held-out rows, {R['n_tr']:,} train rows; prior = this group's train class frequencies")
        mods = ["prior-grp"] + names
        rows = [[st] + [f"{br[m][j]:.8f}" for m in mods] + [f"{br[m][j] - br['prior-grp'][j]:+.8f}" for m in names] for j, st in enumerate(STATES)]
        rows.append(["TOTAL"] + [f"{br[m].sum():.8f}" for m in mods] + [f"{br[m].sum() - br['prior-grp'].sum():+.8f}" for m in names])
        table(["class"] + mods + [f"{m} - prior" for m in names], rows, [8] + [14] * len(mods) + [15] * len(names))
        if R["n_o"] > 1:
            emit("  Total Brier by origin, against the origin-conditional prior (the stricter reference for this two-origin group)")
            freq_o = R["ctr"] / R["ctr"].sum(axis=1, keepdims=True)
            So = prior_brier_sums(R, freq_o).sum(axis=1)
            rows = [[TRANSIENT[o], f"{int(R['n_te_o'][i]):,}", f"{So[i] / R['n_te_o'][i]:.8f}"] + [f"{R['models'][m]['S'][i].sum() / R['n_te_o'][i]:.8f}" for m in names]
                    + [f"{(R['models'][m]['S'][i].sum() - So[i]) / R['n_te_o'][i]:+.8f}" for m in names] for i, o in enumerate(grp["origins"])]
            table(["origin", "rows", "prior-origin"] + names + [f"{m} - prior" for m in names], rows, [8, 11, 14] + [14] * len(names) + [15] * len(names))

    # ---- T2 ----------------------------------------------------------------------------------------------------------------------------
    title("T2  THE MECHANISM TABLE: per run-length bucket and origin, mean predicted vs realised probability of each next state (with-run-length models only)")
    emit("The origin's own state is STAY. Realised = share of the bucket's held-out rows that went to the state; predicted = the model's mean probability for it.")
    for grp in GROUPS:
        R = results[grp["name"]]
        rl_models = [m[0] for m in grp["models"] if m[4]]
        for i, o in enumerate(grp["origins"]):
            emit()
            emit(f"=== origin {TRANSIENT[o]} ({grp['name']} group), held-out rows by run-length bucket; predicted columns: " + ", ".join(rl_models))
            n_ob = R["cnt"][i].sum(axis=1)
            for j, st in enumerate(STATES):
                emit(f"  {'STAY in ' + st if j == o else 'to ' + st}")
                rows = [[bk, f"{int(n_ob[b]):,}", pct(R["cnt"][i, b, j] / max(n_ob[b], 1), 10, 4)] + [pct(R["models"][m]["Pmean"][i, b, j], 10, 4) for m in rl_models]
                        for b, bk in enumerate(BUCKETS)]
                table(["bucket", "n", "realised"] + rl_models, rows, [8, 12, 12] + [12] * len(rl_models))

    # ---- T3 ----------------------------------------------------------------------------------------------------------------------------
    title("T3  DIRECTION OF TRAVEL: predicted probability of STAYING in the origin state at run-length bucket 1 and at 7+, next to the realised")
    emit(f"direction = rising if the 7+ value exceeds the bucket-1 value by more than {100 * FLAT_PP:.1f} percentage point, falling if it is lower by more than that, else flat")

    def direction(d):
        return "rising" if d > FLAT_PP else "falling" if d < -FLAT_PP else "flat"

    for grp in GROUPS:
        R = results[grp["name"]]
        for i, o in enumerate(grp["origins"]):
            n_ob = R["cnt"][i].sum(axis=1)
            real = R["cnt"][i, :, o] / np.maximum(n_ob, 1)
            rd = real[3] - real[0]
            for m in [m[0] for m in grp["models"] if m[4]]:
                pm = R["models"][m]["Pmean"][i, :, o]
                pd_ = pm[3] - pm[0]
                emit(f"{grp['name']:<10} origin {TRANSIENT[o]:<8} model {m:<7} predicted stay {100 * pm[0]:6.2f}% -> {100 * pm[3]:6.2f}% ({100 * pd_:+.2f} pp, {direction(pd_):<7}) | "
                     f"realised stay {100 * real[0]:6.2f}% -> {100 * real[3]:6.2f}% ({100 * rd:+.2f} pp, {direction(rd):<7}) | "
                     f"{'SAME direction' if direction(pd_) == direction(rd) else 'DIFFERENT direction'}")

    # ---- notes -----------------------------------------------------------------------------------------------------------------------------
    title("NOTES AND CHECKS")
    emit(f"panel build {manifest['build_stamp']}; role == 'fit', from_state in {TRANSIENT}, {WIN_LO} <= from_period <= {WIN_HI} (observed {fp_rng[0]}..{fp_rng[1]}); "
         f"null labels dropped {n_null:,}; pooled split by loan: seed {SEED}, {n_tr_loans:,} of {n_loans:,} loans to train, {n_loans - n_tr_loans:,} held out; "
         f"no loan on both sides (asserted)")
    emit(f"common params: " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items()) + f"; num_boost_round={N_ROUNDS}; native categoricals: property_state (and from_state in the "
         f"DQ30+DQ60 models)")
    emit(f"  {'group':<10}{'model':<8}{'capacity':<10}{'run length':<11}{'train rows':>12}{'held-out rows':>15}{'trees':>7}{'construct s':>13}{'train + predict s':>19}   library messages")
    for grp in GROUPS:
        R = results[grp["name"]]
        for nm, feats, capacity, cap_label, has_rl in grp["models"]:
            r = R["models"][nm]
            m = " | ".join(f"{x} x{r['msgs'].count(x)}" for x in sorted(set(r["msgs"]))) if r["msgs"] else "none"
            emit(f"  {grp['name']:<10}{nm:<8}{cap_label:<10}{'with' if has_rl else 'no':<11}{R['n_tr']:>12,}{R['n_te']:>15,}{r['trees']:>7}{r['t_construct']:>13.1f}"
                 f"{r['t_fit']:>19.1f}   {m}")
    emit("features: " + "; ".join(f"{g['name']}: {[f for f in g['models'][-1][1]]}" for g in GROUPS))
    for grp in GROUPS:
        R = results[grp["name"]]
        for i, o in enumerate(grp["origins"]):
            emit(f"conditional prior, origin {TRANSIENT[o]:<7}: " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, R["ctr"][i] / R["ctr"][i].sum())))
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
