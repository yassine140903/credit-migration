"""Does run length let the DQ90+-origin model get exit behaviour right WITHIN run-length buckets?   MEASUREMENT ONLY.

    python diagnostics/52_dq90_runlength_buckets.py   # from the repo root

A model without run length learns one average exit rate for the state; it can be right in aggregate while wrong in every run-length bucket. This
diagnostic trains two arms that differ only in features and compares predicted with realised exit behaviour bucket by bucket. Writes
diagnostics/52_dq90_runlength_buckets.txt. No plots, no model artefacts, nothing under src/ or data/, no git command.

POPULATION. role == 'fit', from_state == 'DQ90+' (the brief's "origin_state"; the panel column is from_state), read partition by partition with
usecols-style column projection through panel_io.load_panel. Rows with a null to_state are dropped and counted. Expect diagnostic 51's 201,792 rows.
SPLIT. By loan, as diagnostic 51: sorted unique loan_id strings shuffled with numpy default_rng(40), first 80% (rounded) to train. The per-class
train / held-out counts are compared with diagnostic 51's (hard-coded below) and any difference is printed; it does not stop the run.

ARMS. A: fico ltv dti rate_spread property_state est_ltv unemp_level unemp_chg_12m (the brief's "original_ltv" is the panel's ltv column).
B: A plus run_length and run_length_censored. property_state is a LightGBM native categorical with the panel's fixed category list. No loan_age.
Both arms: objective multiclass, num_class 6, num_leaves 8, min_data_in_leaf 100, 100 rounds, LightGBM defaults otherwise, seed 40 both arms.
Both arms train on ALL train rows (censored included); the uncensored tables only restrict which held-out rows are scored.
PRIOR-ONLY. Train-side class frequencies (all train rows) applied to every held-out row, in every table.
BUCKETS. run_length 1 / 2-3 / 4-6 / 7+. Counts are printed before any score.
LEAVING DQ90+. Predicted: 1 - P(DQ90+). Realised: to_state != DQ90+. Destinations: Default, Current, and the remaining leave (DQ30, DQ60, Prepaid).
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

OUT = os.path.join("diagnostics", "52_dq90_runlength_buckets.txt")
STATES = list(panel.STATES)
K = len(STATES)
I_CUR, I_DEF, I_DQ90 = STATES.index("Current"), STATES.index("Default"), STATES.index("DQ90+")
I_OTHER = [j for j in range(K) if j not in (I_CUR, I_DEF, I_DQ90)]
ORIGIN = "DQ90+"
CATEGORIES = {"property_state": list(panel.PROPERTY_STATES)}
FEATS_A = ["fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv", "unemp_level", "unemp_chg_12m"]
FEATS_B = FEATS_A + ["run_length", "run_length_censored"]
SEED, TRAIN_FRAC, N_ROUNDS = 40, 0.8, 100
PARAMS = {"objective": "multiclass", "num_class": K, "device_type": "cpu", "verbosity": 0, "num_leaves": 8, "min_data_in_leaf": 100,
          "seed": SEED, "deterministic": True, "force_row_wise": True}
BUCKETS = ["1", "2-3", "4-6", "7+"]
EXPECTED_ROWS = 201792
EXPECT_COUNTS = {"train": [9851, 2131, 3117, 139715, 852, 5606], "held-out": [2514, 503, 765, 35122, 231, 1385]}   # diagnostic 51 / 41
BRIER_FORMULA = ("Brier per class j = mean_i (p_ij - o_ij)^2 over the held-out rows, where p_ij is the predicted probability of class j and o_ij is 1 if the "
                 "row's actual next state is j, else 0; the total is the sum over the six classes")
GB = 2 ** 30
RULE_W = 118
LINES = []


def emit(line=""):
    LINES.append(line)
    print(line, flush=True)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


def on_abort(msg):
    emit("ABORTED BY MEMWATCH: " + msg)
    write_report()


def pct(x, w=9, d=3):
    return f"{100 * x:>{w}.{d}f}%"


def load():
    """All DQ90+ fit rows, one partition at a time, only the columns the arms use."""
    manifest = panel_io.read_manifest()
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + FEATS_B
    xs, ys, ids, fps = [], [], [], []
    n_all = n_null = n_rl_null = 0
    for p in manifest["partitions"]:
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        assert set(d["role"].astype(str).unique()) <= {"fit"}, "a non-fit row was read"
        m0 = (d["from_state"] == ORIGIN).to_numpy()
        n_all += int(m0.sum())
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m0 & (y < 0)).sum())
        m = m0 & (y >= 0)
        d = d[m]
        n_rl_null += int(d["run_length"].isna().sum() + d["run_length_censored"].isna().sum())
        data = {}
        for c in FEATS_B:
            if c in CATEGORIES:
                data[c] = d[c].cat.set_categories(CATEGORIES[c]).reset_index(drop=True)
            elif c == "run_length_censored":
                data[c] = d[c].fillna(False).to_numpy(dtype=bool)
            elif c == "run_length":
                data[c] = d[c].to_numpy(dtype=np.int64, na_value=0)
            else:
                data[c] = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
        xs.append(pd.DataFrame(data))
        ys.append(y[m])
        ids.append(d["loan_id"].astype(str).to_numpy())
        fps.append(d["from_period"].to_numpy(dtype=np.int64, na_value=0))
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows", flush=True)
        del d, data
    X = pd.concat(xs, ignore_index=True)
    assert str(X["property_state"].dtype) == "category" and list(X["property_state"].cat.categories) == CATEGORIES["property_state"]
    return X, np.concatenate(ys), np.concatenate(ids), np.concatenate(fps), n_all, n_null, n_rl_null, manifest


def bucket_of(rl):
    return np.where(rl <= 1, 0, np.where(rl <= 3, 1, np.where(rl <= 6, 2, 3)))


def brier_by_class(P, y):
    return np.array([float(np.mean((P[:, j] - (y == j)) ** 2)) for j in range(K)])


def table(headers, rows, widths):
    emit("  " + "".join(f"{h:>{w}}" if i else f"{h:<{w}}" for i, (h, w) in enumerate(zip(headers, widths))))
    for r in rows:
        emit("  " + "".join(f"{c:>{w}}" if i else f"{c:<{w}}" for i, (c, w) in enumerate(zip(r, widths))))


def subsets(cens):
    return [("(i) ALL HELD-OUT ROWS", np.ones(len(cens), dtype=bool)),
            ("(ii) UNCENSORED ROWS ONLY (run_length_censored == False)", ~cens)]


def main():
    import lightgbm as lgb

    watch = MemWatch(on_abort=on_abort, abort_mb=700)
    watch.phase = "load"
    watch.start()
    t_start = time.perf_counter()
    emit("DQ90+-ORIGIN MODEL: DOES RUN LENGTH FIX EXIT BEHAVIOUR WITHIN RUN-LENGTH BUCKETS?   (diagnostic 52; measurement only)")
    X, y, ids, fps, n_all, n_null, n_rl_null, manifest = load()
    uniq = np.unique(ids)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    in_train = np.isin(ids, uniq[perm[:n_tr]])
    assert not (set(ids[in_train]) & set(ids[~in_train])), "a loan_id is on both sides"
    Xtr, ytr = X[in_train].reset_index(drop=True), y[in_train]
    Xte, yte = X[~in_train].reset_index(drop=True), y[~in_train]
    n_loans_te = len(uniq) - n_tr
    fp_rng = (int(fps.min()), int(fps.max()))
    del X, y, ids, fps, perm
    gc.collect()
    assert Xtr["run_length"].min() >= 1 and Xte["run_length"].min() >= 1, "run_length below 1"

    b_tr, b_te = bucket_of(Xtr["run_length"].to_numpy()), bucket_of(Xte["run_length"].to_numpy())
    c_tr, c_te = Xtr["run_length_censored"].to_numpy(), Xte["run_length_censored"].to_numpy()

    title("BUCKET COUNTS  (rows per run_length bucket; reported before any score)")
    for name, rows_b, rows_c in (("HELD-OUT", b_te, c_te), ("TRAIN", b_tr, c_tr)):
        emit(f"{name} rows: {len(rows_b):,}")
        table(["bucket", "all rows", "% of rows", "censored", "uncensored", "% censored"],
              [[b, f"{int((rows_b == i).sum()):,}", pct((rows_b == i).mean(), 8, 2), f"{int(((rows_b == i) & rows_c).sum()):,}",
                f"{int(((rows_b == i) & ~rows_c).sum()):,}", pct(((rows_b == i) & rows_c).sum() / max(1, (rows_b == i).sum()), 8, 2)]
               for i, b in enumerate(BUCKETS)] +
              [["total", f"{len(rows_b):,}", pct(1.0, 8, 2), f"{int(rows_c.sum()):,}", f"{int((~rows_c).sum()):,}", pct(rows_c.mean(), 8, 2)]],
              [8, 12, 12, 12, 13, 13])
        emit()
    write_report()

    freq_tr = np.bincount(ytr, minlength=K) / len(ytr)
    P_prior = np.broadcast_to(freq_tr, (len(yte), K))
    preds = {}
    secs = {}
    for arm, feats in (("A", FEATS_A), ("B", FEATS_B)):
        print(f"  [arm {arm}] training {len(ytr):,} rows ...", flush=True)
        watch.phase = f"arm {arm}"
        t0 = time.perf_counter()
        Xa = Xtr[feats].copy()
        Xa_te = Xte[feats]
        ds = lgb.Dataset(Xa, label=ytr, categorical_feature=["property_state"], free_raw_data=True, params=PARAMS)
        booster = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
        assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
        P = np.asarray(booster.predict(Xa_te), dtype=np.float64)
        assert P.shape == (len(yte), K) and np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        preds[arm] = P
        secs[arm] = time.perf_counter() - t0
        del booster, ds, Xa
        gc.collect()
    preds["prior"] = np.asarray(P_prior)
    arms = [("A", preds["A"]), ("B", preds["B"]), ("prior-only", preds["prior"])]

    def leave(P):
        return 1.0 - P[:, I_DQ90]

    realised_leave = (yte != I_DQ90).astype(np.float64)

    title("T1  PER BUCKET: mean predicted probability of LEAVING DQ90+ (1 - predicted DQ90+ self-transition) vs realised share that left")
    for label, sub in subsets(c_te):
        emit(f"--- {label}")
        rows = []
        for i, b in enumerate(BUCKETS):
            m = sub & (b_te == i)
            rows.append([b, f"{int(m.sum()):,}", pct(realised_leave[m].mean())] + [pct(leave(P)[m].mean()) for _, P in arms])
        table(["bucket", "n", "realised", "A", "B", "prior-only"], rows, [8, 11, 12, 12, 12, 12])
        emit()

    title("T2  PER BUCKET: destination split of leaving DQ90+, as shares of ALL rows in the bucket (realised vs mean predicted)")
    emit("Default and Current are the two named destinations; 'other leave' is the rest of the leave (DQ30, DQ60, Prepaid), so that the three tables sum to T1.")
    dests = [("to Default", [I_DEF]), ("to Current", [I_CUR]), ("other leave (DQ30 + DQ60 + Prepaid)", I_OTHER)]
    for label, sub in subsets(c_te):
        emit(f"--- {label}")
        for dname, idx in dests:
            emit(f"  {dname}")
            rows = []
            real = np.isin(yte, idx).astype(np.float64)
            for i, b in enumerate(BUCKETS):
                m = sub & (b_te == i)
                rows.append([b, f"{int(m.sum()):,}", pct(real[m].mean())] + [pct(P[m][:, idx].sum(axis=1).mean()) for _, P in arms])
            table(["bucket", "n", "realised", "A", "B", "prior-only"], rows, [8, 11, 12, 12, 12, 12])
        emit()

    title("T3  POOLED across all DQ90+ held-out rows, same quantities as T1.   *** NOT THE JUDGE: pooling hides within-bucket error ***")
    for label, sub in subsets(c_te):
        emit(f"--- {label}")
        table(["pooled", "n", "realised", "A", "B", "prior-only"],
              [["all buckets", f"{int(sub.sum()):,}", pct(realised_leave[sub].mean())] + [pct(leave(P)[sub].mean()) for _, P in arms]],
              [14, 11, 12, 12, 12, 12])
        emit()

    title("T4  PER-CLASS BRIER on held-out rows, pooled (lower is better)")
    emit(BRIER_FORMULA)
    for label, sub in subsets(c_te):
        emit(f"--- {label}   ({int(sub.sum()):,} rows)")
        br = {name: brier_by_class(P[sub], yte[sub]) for name, P in arms}
        rows = [[s] + [f"{br[n][j]:.8f}" for n in ("prior-only", "A", "B")] + [f"{br['A'][j] - br['prior-only'][j]:+.8f}", f"{br['B'][j] - br['prior-only'][j]:+.8f}"]
                for j, s in enumerate(STATES)]
        tot = {n: br[n].sum() for n in br}
        rows.append(["TOTAL"] + [f"{tot[n]:.8f}" for n in ("prior-only", "A", "B")] + [f"{tot['A'] - tot['prior-only']:+.8f}", f"{tot['B'] - tot['prior-only']:+.8f}"])
        table(["class", "prior-only", "A", "B", "A - prior", "B - prior"], rows, [8, 14, 14, 14, 14, 14])
        emit()

    title("NOTES AND CHECKS")
    ctr, cte = np.bincount(ytr, minlength=K), np.bincount(yte, minlength=K)
    same = list(map(int, ctr)) == EXPECT_COUNTS["train"] and list(map(int, cte)) == EXPECT_COUNTS["held-out"]
    emit(f"panel build {manifest['build_stamp']}; role == 'fit', from_state == '{ORIGIN}'; DQ90+-origin fit rows {n_all:,}; null labels dropped {n_null:,}; "
         f"rows used {len(ytr) + len(yte):,} (diagnostic 51: {EXPECTED_ROWS:,}: {'MATCH' if len(ytr) + len(yte) == EXPECTED_ROWS else 'DIFFERENT'}); "
         f"null run_length / run_length_censored values in used rows {n_rl_null:,}; from_period range {fp_rng[0]}..{fp_rng[1]}")
    emit(f"split by loan: seed {SEED}, {n_tr:,} of {len(uniq):,} loans to train, {n_loans_te:,} held out; train rows {len(ytr):,}, held-out rows {len(yte):,}; "
         f"no loan on both sides (asserted)")
    emit(f"  {'class':<10}{'TRAIN rows':>14}{'diag 51':>12}{'HELD-OUT rows':>16}{'diag 51':>12}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(ctr[j]):>14,}{EXPECT_COUNTS['train'][j]:>12,}{int(cte[j]):>16,}{EXPECT_COUNTS['held-out'][j]:>12,}")
    emit(f"per-class counts vs diagnostic 51's table: {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    emit("prior-only (train class frequencies, all train rows): " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, freq_tr)))
    emit(f"arm A features ({len(FEATS_A)}): {FEATS_A}")
    emit(f"arm B features ({len(FEATS_B)}): {FEATS_B}")
    emit("params (both arms): " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items()) + f"; num_boost_round={N_ROUNDS}; property_state native categorical")
    emit(f"lightgbm {lgb.__version__}; train+predict wall-clock A {secs['A']:.1f} s, B {secs['B']:.1f} s; total {time.perf_counter() - t_start:.1f} s")
    watch.stop()
    emit(f"PEAK RSS (src.memwatch): process peak {watch.peak_rss() / GB:.2f} GB; lowest system-available memory seen {watch.min_avail / GB:.2f} GB")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped: {e}")
        write_report()
        raise
