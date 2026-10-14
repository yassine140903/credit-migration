"""Does the shipped POOLED, DEFAULT-capacity model learn the run-length default ladder that diagnostic 52's DQ90+-only model learned?   MEASUREMENT ONLY.

    python diagnostics/53_pooled_runlength_buckets.py   # from the repo root

Diagnostic 52 showed a single-origin, reduced-capacity DQ90+ model learns the run-length ladder (realised to-Default share about 0.3% at run length 1 up to
6.7% at 7+). The shipped headline model is pooled at LightGBM default capacity. This asks whether that configuration learns the same ladder, or whether
DQ90+ is too small a share of the pooled objective for duration to register. Writes diagnostics/53_pooled_runlength_buckets.txt. No plots, no model
artefacts, nothing under src/ or data/, no git command.

POPULATION. role == 'fit', from_state in the four transient states (Current, DQ30, DQ60, DQ90+), 200601 <= from_period <= 200812 (the full fit window, as
diagnostic 47's POOLED population: 40,210,035 rows). Rows with a null to_state are dropped and counted. Read partition by partition with column projection
through panel_io.load_panel; pass 1 reads labels and run_length only, pass 2 the feature matrix, and the two passes must give identical label vectors.
SPLIT. By loan, 80/20, seed 40: sorted unique loan_id strings of the POOLED population (per-partition sorted codes with an offset; loan_ids carry their
partition's vintage prefix, asserted, so this equals the global sorted order) shuffled with numpy default_rng(40), first 80% (rounded) to train. This is
diagnostic 47's POOLED split. The DQ90+-origin subset of the held-out rows is then compared with diagnostic 52's held-out DQ90+ class counts and bucket counts; if it
does not, that is a printed WARNING (SPLIT COMPARABILITY: both sets of counts with percentage differences), not a stop. The pooled split draws from all
pooled loans while 52 drew from DQ90+ loans only, so the same seed assigns loans differently: 52 is a reference point, not a paired control.

ARMS. A: fico ltv dti rate_spread property_state est_ltv unemp_level unemp_chg_12m from_state. B: A plus run_length and run_length_censored. property_state
and from_state are LightGBM native categoricals with fixed category lists. No loan_age. LightGBM DEFAULT capacity (num_leaves 31, min_data_in_leaf 20 are not
overridden), objective multiclass, num_class 6, 100 rounds, seed 40, deterministic. Numerics are float32 throughout, including run_length and the 0/1
run_length_censored flag, so no column forces a float64 copy of the matrix.
PRIOR-ONLY. Train class frequencies, two ways: POOLED over all train rows, and CONDITIONAL on origin state (each held-out row gets its own origin state's
train frequencies). The conditional one is the meaningful reference for the DQ90+ tables.
TABLES. T1-T4 are restricted to held-out rows with from_state == 'DQ90+', so they compare directly with diagnostic 52; T5 scores ALL held-out rows. Brier is
computed in chunks. All tables once on all rows: the censoring split is NOT repeated (diagnostic 52 found 42 of 40,520 held-out DQ90+ rows censored).
Leaving DQ90+ = 1 - P(DQ90+); destinations Default, Current, and the remaining leave (DQ30, DQ60, Prepaid).
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

OUT = os.path.join("diagnostics", "53_pooled_runlength_buckets.txt")
STATES = list(panel.STATES)
K = len(STATES)
TRANSIENT = STATES[:4]
I_CUR, I_DEF, I_DQ90 = STATES.index("Current"), STATES.index("Default"), STATES.index("DQ90+")
I_OTHER = [j for j in range(K) if j not in (I_CUR, I_DEF, I_DQ90)]
O_DQ90 = TRANSIENT.index("DQ90+")
CATEGORIES = {"property_state": list(panel.PROPERTY_STATES), "from_state": list(TRANSIENT)}
FEATS_A = ["fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv", "unemp_level", "unemp_chg_12m", "from_state"]
FEATS_B = FEATS_A + ["run_length", "run_length_censored"]
CATEGORICAL = ["property_state", "from_state"]
WIN_LO, WIN_HI = 200601, 200812
SEED, TRAIN_FRAC, N_ROUNDS = 40, 0.8, 100
PARAMS = {"objective": "multiclass", "num_class": K, "device_type": "cpu", "verbosity": 0, "seed": SEED, "deterministic": True, "force_row_wise": True}
BUCKETS = ["1", "2-3", "4-6", "7+"]
REF52 = {"held-out classes": [2514, 503, 765, 35122, 231, 1385], "held-out buckets": [10362, 13148, 10007, 7003]}   # diagnostic 52
BRIER_FORMULA = ("Brier per class j = mean_i (p_ij - o_ij)^2 over the rows scored, where p_ij is the predicted probability of class j and o_ij is 1 if the "
                 "row's actual next state is j, else 0; the total is the sum over the six classes")
CHUNK = 2_000_000
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
    emit("ABORTED BY MEMWATCH (no rows were sampled): " + msg)
    write_report()


class Capture:
    def __init__(self):
        self.msgs = []

    def info(self, m):
        self.msgs.append(str(m))

    def warning(self, m):
        self.msgs.append(str(m))


def pct(x, w=9, d=3):
    return f"{100 * x:>{w}.{d}f}%"


def table(headers, rows, widths):
    emit("  " + "".join(f"{h:>{w}}" if i else f"{h:<{w}}" for i, (h, w) in enumerate(zip(headers, widths))))
    for r in rows:
        emit("  " + "".join(f"{c:>{w}}" if i else f"{c:<{w}}" for i, (c, w) in enumerate(zip(r, widths))))


def bucket_of(rl):
    return np.where(rl <= 1, 0, np.where(rl <= 3, 1, np.where(rl <= 6, 2, 3)))


def brier_by_class(P, y):
    return np.array([float(np.mean((P[:, j] - (y == j)) ** 2)) for j in range(K)])


def read_partitions(cols):
    """Pass 1: role == 'fit', pooled transient origins, the full window, non-null label. One partition at a time; labels and run_length only."""
    manifest = panel_io.read_manifest()
    prefixes = ["F" + p["quarter"][2:4] + p["quarter"][4:] for p in manifest["partitions"]]
    assert prefixes == sorted(prefixes), "partition vintage prefixes do not sort in file order"
    ys, fss, rls, codes, lens, fp_lo, fp_hi = [], [], [], [], [], 10 ** 9, 0
    offset = n_null = n_in = 0
    for p, prefix in zip(manifest["partitions"], prefixes):
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        assert set(d["role"].astype(str).unique()) <= {"fit"}, "a non-fit row was read"
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = d["from_state"].isin(TRANSIENT).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        n_in += len(d)
        lens.append(len(d))
        fp_lo, fp_hi = min(fp_lo, int(fp[m].min())), max(fp_hi, int(fp[m].max()))
        lid = d["loan_id"].astype(str)
        assert lid.str.startswith(prefix).all(), "a loan_id lacks its partition's vintage prefix"
        c, uniq = pd.factorize(lid, sort=True)
        codes.append(c.astype(np.int64) + offset)
        offset += len(uniq)
        ys.append(y[m].astype(np.int8))
        fss.append(d["from_state"].astype(str).astype(pd.CategoricalDtype(TRANSIENT)).cat.codes.to_numpy().astype(np.int8))
        rls.append(d["run_length"].to_numpy(dtype=np.int32, na_value=0))
        print(f"  loaded {p['quarter']}: {len(d):,} rows, {len(uniq):,} loans", flush=True)
        del d, lid
        gc.collect()
    return dict(lens=lens, y=np.concatenate(ys), fs=np.concatenate(fss), rl=np.concatenate(rls), codes=np.concatenate(codes), n_null=n_null,
                n_loans=offset, fp=(fp_lo, fp_hi), manifest=manifest)


def read_features(lens, in_train, y_ref, fs_ref, n_train_rows, n_test_rows):
    """Pass 2: the same rows in the same order as pass 1, written straight into two preallocated column-major float32 matrices (train, held-out), columns in
    FEATS_B order, so FEATS_A is the leading block and a view of it is contiguous. Categoricals are stored as their category codes (-1 -> NaN), which is what
    LightGBM's pandas path does. No full-panel frame and no split copy is ever built."""
    manifest = panel_io.read_manifest()
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + [c for c in FEATS_B if c != "from_state"]
    Xtr = np.empty((n_train_rows, len(FEATS_B)), dtype=np.float32, order="F")
    Xte = np.empty((n_test_rows, len(FEATS_B)), dtype=np.float32, order="F")
    off = otr = ote = 0
    rl_min = 10 ** 9
    for p, n_expect in zip(manifest["partitions"], lens):
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
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
        seg = in_train[off:off + n]
        for j, c in enumerate(FEATS_B):
            if c == "from_state":
                v = fs.astype(np.float32)
            elif c in CATEGORIES:
                v = d[c].cat.set_categories(CATEGORIES[c]).cat.codes.to_numpy().astype(np.float32)
                v[v < 0] = np.nan
            elif c in ("run_length", "run_length_censored"):
                assert not d[c].isna().any(), c + " has nulls"
                v = d[c].astype(np.float32).to_numpy()
                if c == "run_length":
                    rl_min = min(rl_min, int(v.min()))
            else:
                v = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
            Xtr[otr:otr + int(seg.sum()), j] = v[seg]
            Xte[ote:ote + int((~seg).sum()), j] = v[~seg]
        otr += int(seg.sum())
        ote += int((~seg).sum())
        off += n
        print(f"  loaded {p['quarter']}: {n:,} rows", flush=True)
        del d, v, fs, y
        gc.collect()
    assert off == len(y_ref) and otr == n_train_rows and ote == n_test_rows, "row totals differ from pass 1"
    assert rl_min >= 1, "run_length below 1"
    j_fs = FEATS_B.index("from_state")
    assert not np.isnan(Xtr[:, j_fs]).any() and not np.isnan(Xte[:, j_fs]).any(), "null from_state"
    return Xtr, Xte


def split_flags(codes):
    """Diagnostic 47's split: sorted unique loan codes shuffled with default_rng(SEED), first TRAIN_FRAC (rounded) to train."""
    uniq, inv = np.unique(codes, return_inverse=True)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool)
    flag[perm[:n_tr]] = True
    in_train = flag[inv]
    assert np.intersect1d(np.unique(codes[in_train]), np.unique(codes[~in_train])).size == 0, "a loan is on both sides"
    return in_train, len(uniq), n_tr


def main():
    import lightgbm as lgb

    cap = Capture()
    lgb.register_logger(cap)
    watch = MemWatch(on_abort=on_abort, abort_mb=700)
    watch.phase = "pass 1"
    watch.start()
    t_start = time.perf_counter()
    emit("POOLED, DEFAULT-CAPACITY MODEL: DOES IT LEARN THE DQ90+ RUN-LENGTH LADDER?   (diagnostic 53; measurement only)")
    emit("Population: role == 'fit', pooled transient origins, 200601..200812. T1-T4 are restricted to held-out rows with from_state == 'DQ90+' (comparable "
         "with diagnostic 52); T5 scores all held-out rows.")
    emit("All tables are given once, on all rows. The censoring split of diagnostic 52 is NOT repeated: it found 42 of 40,520 held-out DQ90+ rows censored, "
         "immaterial.")

    # ---- pass 1: labels and run_length only ---------------------------------------------------------------------------------
    print("pass 1: labels", flush=True)
    L = read_partitions(["role", "loan_id", "from_period", "from_state", "to_state", "run_length"])
    y1, fs1, rl1 = L["y"], L["fs"], L["rl"]
    in_train, n_loans, n_tr = split_flags(L["codes"])
    L["codes"] = None
    gc.collect()
    q = (fs1 == O_DQ90) & ~in_train
    cte52 = np.bincount(y1[q].astype(np.int64), minlength=K)
    bte52 = np.bincount(bucket_of(rl1[q]), minlength=4)
    # ---- bucket counts, before any score ----------------------------------------------------------------------------------
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
    y_ref = y1.copy()
    fs_ref = fs1.copy()
    del L, y1, fs1, rl1, ytr1, fstr1, q
    gc.collect()

    # ---- pass 2: feature matrix ---------------------------------------------------------------------------------------------
    watch.phase = "pass 2"
    print("pass 2: feature matrix", flush=True)
    Xtr, Xte = read_features(lens, in_train, y_ref, fs_ref, n_train_rows, n_test_rows)
    ytr, yte = y_ref[in_train].astype(np.int64), y_ref[~in_train].astype(np.int64)
    fste = fs_ref[~in_train]
    del y_ref, fs_ref, in_train
    gc.collect()

    # ---- bin both arms' training sets first, then free the raw matrix, so training holds only binned data -----------------------
    sub_idx = np.flatnonzero(fste == O_DQ90)                                        # held-out DQ90+-origin rows
    rl_sub = Xte[sub_idx, FEATS_B.index("run_length")].astype(np.int64)
    b_sub, y_sub = bucket_of(rl_sub), yte[sub_idx]
    sq = {}                                                                         # per-arm squared-error sums: by origin (4, K)
    P_sub = {}
    secs, msgs, trees = {}, {}, {}
    watch.phase = "dataset construction"
    t0 = time.perf_counter()
    label_tr = ytr.astype(np.float32)
    dsets = {arm: lgb.Dataset(Xtr[:, :len(feats)], label=label_tr, feature_name=list(feats), categorical_feature=CATEGORICAL, free_raw_data=True,
                              params=PARAMS).construct() for arm, feats in (("A", FEATS_A), ("B", FEATS_B))}
    t_construct = time.perf_counter() - t0
    n_trained = len(ytr)
    del Xtr, label_tr, ytr
    gc.collect()
    for arm, feats in (("A", FEATS_A), ("B", FEATS_B)):
        print(f"  [arm {arm}] training {n_trained:,} rows ...", flush=True)
        watch.phase = f"arm {arm}"
        cap.msgs.clear()
        t0 = time.perf_counter()
        booster = lgb.train(PARAMS, dsets.pop(arm), num_boost_round=N_ROUNDS)
        gc.collect()
        assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
        trees[arm], msgs[arm] = booster.num_trees(), list(cap.msgs)
        s_arm = np.zeros((4, K))
        keep = []
        for a in range(0, len(yte), CHUNK):
            b = min(a + CHUNK, len(yte))
            P = np.asarray(booster.predict(Xte[a:b, :len(feats)]), dtype=np.float64)
            onehot = np.zeros_like(P)
            onehot[np.arange(b - a), yte[a:b]] = 1.0
            err = (P - onehot) ** 2
            for o in range(4):
                s_arm[o] += err[fste[a:b] == o].sum(axis=0)
            m = fste[a:b] == O_DQ90
            keep.append(P[m])
            del P, onehot, err
        del booster
        gc.collect()
        sq[arm] = s_arm
        P_sub[arm] = np.concatenate(keep)
        assert P_sub[arm].shape == (len(sub_idx), K)
        secs[arm] = time.perf_counter() - t0
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
    arms = [("A", P_sub["A"]), ("B", P_sub["B"]), ("prior (origin-cond.)", P_prior_cond), ("prior (pooled)", P_prior_pool)]
    heads = ["bucket", "n", "realised", "A", "B", "prior-cond", "prior-pooled"]
    widths = [8, 11, 12, 12, 12, 13, 14]

    def leave(P):
        return 1.0 - P[:, I_DQ90]

    realised_leave = (y_sub != I_DQ90).astype(np.float64)

    title("T1  DQ90+ HELD-OUT ROWS, PER BUCKET: mean predicted probability of LEAVING DQ90+ (1 - predicted DQ90+ self-transition) vs realised share that left")
    rows = [[bk, f"{int((b_sub == i).sum()):,}", pct(realised_leave[b_sub == i].mean())] + [pct(leave(P)[b_sub == i].mean()) for _, P in arms]
            for i, bk in enumerate(BUCKETS)]
    table(heads, rows, widths)

    title("T2  DQ90+ HELD-OUT ROWS, PER BUCKET: destination split of leaving DQ90+, as shares of ALL rows in the bucket (realised vs mean predicted)")
    emit("THIS IS THE TABLE THAT ANSWERS THE QUESTION. Diagnostic 52 realised to-Default ladder: about 0.3% at run length 1 up to 6.7% at 7+.")
    emit("'other leave' is the rest of the leave (DQ30, DQ60, Prepaid), so that the three blocks sum to T1.")
    for dname, idx in (("to Default", [I_DEF]), ("to Current", [I_CUR]), ("other leave (DQ30 + DQ60 + Prepaid)", I_OTHER)):
        emit(f"  {dname}")
        real = np.isin(y_sub, idx).astype(np.float64)
        rows = [[bk, f"{int((b_sub == i).sum()):,}", pct(real[b_sub == i].mean())] + [pct(P[b_sub == i][:, idx].sum(axis=1).mean()) for _, P in arms]
                for i, bk in enumerate(BUCKETS)]
        table(heads, rows, widths)

    title("T3  POOLED across all DQ90+ held-out rows, same quantities as T1.   *** NOT THE JUDGE: pooling hides within-bucket error ***")
    table(["pooled"] + heads[1:], [["all buckets", f"{len(y_sub):,}", pct(realised_leave.mean())] + [pct(leave(P).mean()) for _, P in arms]],
          [14] + widths[1:])

    def brier_rows(s_by_arm, n):
        names = ["prior-pooled", "prior-cond", "A", "B"]
        keys = ["prior-pooled", "prior-cond", "A", "B"]
        br = {nm: s_by_arm[k] / n for nm, k in zip(names, keys)}
        rows = [[st] + [f"{br[nm][j]:.8f}" for nm in names] + [f"{br['A'][j] - br['prior-cond'][j]:+.8f}", f"{br['B'][j] - br['prior-cond'][j]:+.8f}",
                                                                 f"{br['B'][j] - br['A'][j]:+.8f}"] for j, st in enumerate(STATES)]
        t = {nm: br[nm].sum() for nm in names}
        rows.append(["TOTAL"] + [f"{t[nm]:.8f}" for nm in names] + [f"{t['A'] - t['prior-cond']:+.8f}", f"{t['B'] - t['prior-cond']:+.8f}", f"{t['B'] - t['A']:+.8f}"])
        return rows

    bh = ["class", "prior-pooled", "prior-cond", "A", "B", "A - cond", "B - cond", "B - A"]
    bw = [8, 14, 14, 14, 14, 14, 14, 14]

    title("T4  PER-CLASS BRIER on DQ90+ held-out rows (lower is better)")
    emit(BRIER_FORMULA)
    emit(f"rows scored: {len(y_sub):,}   (differences are against the origin-conditional prior, the meaningful reference here)")
    sub_sq = {"A": ((P_sub["A"] - np.eye(K)[y_sub]) ** 2).sum(axis=0), "B": ((P_sub["B"] - np.eye(K)[y_sub]) ** 2).sum(axis=0),
              "prior-pooled": ((P_prior_pool - np.eye(K)[y_sub]) ** 2).sum(axis=0), "prior-cond": ((P_prior_cond - np.eye(K)[y_sub]) ** 2).sum(axis=0)}
    for k in ("A", "B", "prior-pooled", "prior-cond"):                              # chunked accumulation vs the stored DQ90+ subset
        assert np.allclose(sq[k][O_DQ90], sub_sq[k], rtol=1e-9), "chunked and subset Brier sums disagree for " + k
    table(bh, brier_rows(sub_sq, len(y_sub)), bw)

    title("T5  PER-CLASS BRIER on ALL held-out rows, every origin state (lower is better): the whole-model cost or gain of adding run length")
    emit(BRIER_FORMULA)
    emit(f"rows scored: {len(yte):,}")
    all_sq = {"A": sq["A"].sum(axis=0), "B": sq["B"].sum(axis=0), "prior-pooled": sq["prior-pooled"].sum(axis=0), "prior-cond": sq["prior-cond"].sum(axis=0)}
    table(bh, brier_rows(all_sq, len(yte)), bw)
    emit()
    emit("  Total Brier by origin state (held-out rows of that origin)")
    table(["origin", "rows", "prior-pooled", "prior-cond", "A", "B", "A - cond", "B - cond", "B - A"],
          [[TRANSIENT[o], f"{int(n_te_origin[o]):,}"] + [f"{sq[k][o].sum() / n_te_origin[o]:.8f}" for k in ("prior-pooled", "prior-cond", "A", "B")]
           + [f"{(sq['A'][o].sum() - sq['prior-cond'][o].sum()) / n_te_origin[o]:+.8f}", f"{(sq['B'][o].sum() - sq['prior-cond'][o].sum()) / n_te_origin[o]:+.8f}",
              f"{(sq['B'][o].sum() - sq['A'][o].sum()) / n_te_origin[o]:+.8f}"] for o in range(4)], [8, 12, 14, 14, 14, 14, 14, 14, 14])

    # ---- notes -----------------------------------------------------------------------------------------------------------------
    title("NOTES AND CHECKS")
    emit(f"panel build {manifest['build_stamp']}; role == 'fit', from_state in {TRANSIENT}, {WIN_LO} <= from_period <= {WIN_HI} (observed {fp_rng[0]}..{fp_rng[1]}); "
         f"rows {n_train_rows + n_test_rows:,} (diagnostic 47 POOLED: 40,210,035); null labels dropped {n_null:,}")
    emit(f"split by loan: seed {SEED}, {n_tr:,} of {n_loans:,} loans to train, {n_loans - n_tr:,} held out; train rows {n_train_rows:,}, held-out rows {n_test_rows:,}; "
         f"no loan on both sides (asserted); DQ90+ held-out subset vs diagnostic 52: see SPLIT COMPARABILITY (different loan lists, not a paired control)")
    emit("TRAIN rows per origin state, and each state's share of the pooled training objective (every row enters the multiclass loss with equal weight):")
    emit(f"  {'origin':<8}{'train rows':>14}{'share of train rows':>22}{'held-out rows':>16}")
    for o, st in enumerate(TRANSIENT):
        emit(f"  {st:<8}{int(n_train_origin[o]):>14,}{pct(n_train_origin[o] / n_train_rows, 21, 4)}{int(n_te_origin[o]):>16,}")
    emit(f"  {'TOTAL':<8}{n_train_rows:>14,}{pct(1.0, 21, 4)}{n_test_rows:>16,}")
    emit(f"DQ90+-origin rows are {100 * n_train_origin[O_DQ90] / n_train_rows:.4f}% of the pooled training objective (brief's figure from diagnostic 39: ~0.0033%). "
         f"Of the DQ90+-origin train rows, {int(np.round(freq_cond[O_DQ90][I_DEF] * n_train_origin[O_DQ90])):,} go to Default; Default is "
         f"{100 * ctr_all[I_DEF] / n_train_rows:.4f}% of all train rows.")
    emit("class counts, all train rows: " + ", ".join(f"{s} {int(c):,}" for s, c in zip(STATES, ctr_all)) + "; all held-out rows: "
         + ", ".join(f"{s} {int(c):,}" for s, c in zip(STATES, cte_all)))
    emit("prior-only, pooled (all train rows): " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, freq_pooled)))
    for o, st in enumerate(TRANSIENT):
        emit(f"prior-only, conditional on {st:<7}: " + ", ".join(f"{s} {f:.6f}" for s, f in zip(STATES, freq_cond[o])))
    emit(f"arm A features ({len(FEATS_A)}): {FEATS_A}")
    emit(f"arm B features ({len(FEATS_B)}): {FEATS_B}")
    emit("native categoricals: " + str(CATEGORICAL) + "; params (both arms): " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items())
         + f"; num_boost_round={N_ROUNDS}; num_leaves / min_data_in_leaf left at the LightGBM defaults (31 / 20)")
    for arm in ("A", "B"):
        emit(f"arm {arm}: {trees[arm]} trees; train + predict wall-clock {secs[arm]:.1f} s (Dataset construction for both arms, {t_construct:.1f} s, not included); library messages: "
             + (" | ".join(f"{m} x{msgs[arm].count(m)}" for m in sorted(set(msgs[arm]))) if msgs[arm] else "none"))
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
