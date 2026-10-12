"""Full fit window: does the prior-beating failure survive, and is any lift more examples or more macro variation?   MEASUREMENT ONLY.

    python diagnostics/47_full_window_capacity.py   # from the repo root

Diagnostics 40-45 found no model beating a prior-only predictor on the SELECTION window (200601..200706); diagnostic 46 located the
mechanism (a leaf can hold one real rare-class row among twenty). This diagnostic repeats the diagnostic-44 CURRENT and POOLED
single-window models on the FULL fit window, role == 'fit', 200601 <= from_period <= 200812, and adds a borrower-only variant and a
within-month discrimination check. It measures; it does not tune, select or recommend. Writes diagnostics/47_full_window_capacity.txt
and nothing else: no artifacts/ml/ model, nothing under src/ or data/, no git command. Fit rows only: evaluate and condition rows are
never read (load_panel(roles=['fit'])).

Reused from diagnostics 44 / 46 (imported, not re-derived): panel.STATES class order, the label mapping (to_state categorical ->
codes into STATES), BASE_FEATURES, float32 numerics, property_state and from_state as native categoricals with the fixed category
lists, nulls left as-is, the LightGBM params dict (objective multiclass, num_class 6, device cpu, library defaults; verbosity 0
instead of -1 as in 46, with every library message captured and reported), by-loan 80/20 split with numpy default_rng(40) on the
sorted unique loan codes of the population, the explicit multiclass Brier (sum over the six classes of the squared error against the
one-hot vector, averaged over rows, range 0..2). The row loader is a copy of diagnostic 44's load() with exactly three changes: the
window upper bound is 200812; it filters to a population's origin states while reading; and it also returns from_period (needed for
the within-month section). A reproduction check restricts the loaded rows to from_period <= 200706 and requires diagnostic 44's row
counts (10,911,314 CURRENT, 11,017,047 POOLED). The split cannot reproduce 44's exactly, because the population's loan set is larger on
the full window (loans first seen after 200706 join it); the scheme is identical.

POPULATIONS (separate split, prior and runs each):
  CURRENT: from_state == 'Current'.                       Features R1: the eight base features.
  POOLED : the four transient origin states; from_state is a native categorical feature (ninth).
RUNS (per population), LightGBM defaults (num_leaves 31, min_data_in_leaf 20), 100 rounds, no early stopping, no validation set:
  R1  full features.
  R2  BORROWER-ONLY: unemp_level and unemp_chg_12m dropped; fico, ltv, dti, rate_spread, property_state, est_ltv kept (+ from_state in
      POOLED). CAVEAT, flagged and not resolved: est_ltv is borrower-level but contains an HPI component (the loan's ltv rolled
      forward with a house-price index), so R2 is NOT macro-free and R2 still varies over calendar time through est_ltv.

TWO PASSES. (1) A label pass reads only role, loan_id, from_period, from_state, to_state for all four origin states, builds the split
of each population and prints the CLASS COUNTS and row counts BEFORE any model is trained. (2) Per population, the feature matrix is
read (one partition at a time) and the run uses the pass-(2) split, which is asserted to give the same class counts as pass (1). If
memory pressure appears (src.memwatch available-memory floor, or MemoryError) the report says so and the script stops; rows are never
sampled.

SCORING. Held-out predictions come from one predict(raw_score=True) call; probabilities are softmax(raw), checked on 100,000 rows
against predict() (diagnostic 46 found them identical). Train-side Brier is computed in chunks of 4,000,000 rows. The prior-only
reference is recomputed on THIS window: the TRAIN class frequencies predicted for every held-out row.

WITHIN-MONTH DISCRIMINATION (R1 only). Macro columns are constant within a calendar month, so skill measured inside one month is
borrower-level by construction. Months are those of from_period, held-out rows only, each month scored SEPARATELY (never pooled).
Months chosen per population: the three held-out months with the most Default events, plus three mid-window months: the middle
third of the 36-month window (200701..200712), excluding any month already chosen, the three with the most held-out rows. Ranking is
by the predicted Default (or DQ90+) probability, descending; "top 1%" = the ceil(0.01 n) highest rows of that month, "top 10%" likewise.
Predicted probabilities tie often; the cutoff block is handled tie-aware (events in the tied block at the cutoff are credited
pro rata to the slots remaining), and the size of the tied block is printed. Random ranking gives 1% / 10%.
"""

import gc
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                         # noqa: E402
from src.memwatch import MemWatch                                       # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join("diagnostics", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D44 = _load("diag44", "44_capacity_brier.py")
D46 = _load("diag46", "46_init_start_point.py")
BASE_FEATURES, CATEGORIES = D44.BASE_FEATURES, D44.CATEGORIES
STATES = list(D44.STATES)
TRANSIENT = list(D44.TRANSIENT)
SEED, TRAIN_FRAC = D44.SEED, D44.TRAIN_FRAC
BRIER_FORMULA, brier_by_class, counts = D44.BRIER_FORMULA, D44.brier_by_class, D44.counts
K = len(STATES)
I_DQ90, I_DEF = STATES.index("DQ90+"), STATES.index("Default")

SMOKE = os.environ.get("DIAG47_SMOKE") == "1"           # test runs only: first partition, report to the scratch path
WIN_LO, WIN_HI = 200601, 200812
SEL_HI = 200706
SEL_ROWS = {"CURRENT": 10911314, "POOLED": 11017047}    # diagnostic 44 selection-window row counts
POPULATIONS = [("CURRENT", ["Current"]), ("POOLED", list(TRANSIENT))]
PARAMS = dict(D44.BASE_PARAMS, verbosity=0)
R2_DROP = ["unemp_level", "unemp_chg_12m"]
N_ROUNDS = 100
CHUNK = 4_000_000
OUT = os.environ.get("DIAG47_OUT", os.path.join("diagnostics", "47_full_window_capacity.txt"))
GB = 2 ** 30
RULE_W = 118
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
    emit("MEMORY PRESSURE, run stopped (no rows were sampled): " + msg)
    write_report()


class Capture:
    def __init__(self):
        self.msgs = []

    def info(self, m):
        self.msgs.append(str(m))

    def warning(self, m):
        self.msgs.append(str(m))


def softmax(Z):
    Z = np.asarray(Z, dtype=np.float64)
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    E /= E.sum(axis=1, keepdims=True)
    return E


def load_rows(origin, with_features):
    """Copy of diagnostic 44 load(): window 200601..200812, rows of `origin` states only, from_period also returned.
    Returns (X or None, y int8, fp int32, from_state code int8, loan codes int64, n_null, manifest, n_loans)."""
    manifest = panel_io.read_manifest()
    parts = manifest["partitions"][:1] if SMOKE else manifest["partitions"]
    prefixes = ["F" + p["quarter"][2:4] + p["quarter"][4:] for p in manifest["partitions"]]
    assert prefixes == sorted(prefixes), "partition vintage prefixes do not sort in file order"
    pooled = len(origin) > 1
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + (BASE_FEATURES if with_features else [])
    xs, ys, fps, fss, codes = [], [], [], [], []
    offset = n_null = 0
    for p, prefix in zip(parts, prefixes):
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        assert set(d["role"].astype(str).unique()) <= {"fit"}, "a non-fit row was read"
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = d["from_state"].isin(origin).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        lid = d["loan_id"].astype(str)
        assert lid.str.startswith(prefix).all(), "a loan_id lacks its partition's vintage prefix"
        c, uniq = pd.factorize(lid, sort=True)
        codes.append(c.astype(np.int64) + offset)
        offset += len(uniq)
        ys.append(y[m].astype(np.int8))
        fps.append(fp[m].astype(np.int32))
        fss.append(d["from_state"].astype(str).astype(pd.CategoricalDtype(TRANSIENT)).cat.codes.to_numpy().astype(np.int8))
        if with_features:
            data = {}
            for col in BASE_FEATURES:
                if col in CATEGORIES:
                    data[col] = d[col].cat.set_categories(CATEGORIES[col]).reset_index(drop=True)
                else:
                    data[col] = d[col].to_numpy(dtype=np.float32, na_value=np.nan)
            if pooled:
                data["from_state"] = d["from_state"].astype(str).astype(pd.CategoricalDtype(CATEGORIES["from_state"])).reset_index(drop=True)
            xs.append(pd.DataFrame(data))
            del data
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows, {len(uniq):,} loans", flush=True)
        del d, lid
        gc.collect()
    X = None
    if with_features:
        X = pd.concat(xs, ignore_index=True)
        del xs
        for col, cats in CATEGORIES.items():
            if col in X.columns:
                assert str(X[col].dtype) == "category" and list(X[col].cat.categories) == cats, col
    return X, np.concatenate(ys), np.concatenate(fps), np.concatenate(fss), np.concatenate(codes), n_null, manifest, offset


def split_flags(codes):
    """The 44/46 split: sorted unique loan codes shuffled with default_rng(SEED), first TRAIN_FRAC (rounded) to train."""
    uniq, inv = np.unique(codes, return_inverse=True)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool)
    flag[perm[:n_tr]] = True
    in_train = flag[inv]
    assert np.intersect1d(np.unique(codes[in_train]), np.unique(codes[~in_train])).size == 0, "a loan is on both sides"
    return in_train, len(uniq), n_tr


def parse_44_train_counts(pop):
    tr, te, _ = D46.ref44_split(pop)
    return tr


def count_table(pop, ytr, yte, sel_tr):
    ctr, cte = counts(ytr.astype(np.int64)), counts(yte.astype(np.int64))
    emit(f"  {'class':<10}{'TRAIN rows':>14}{'TRAIN freq':>14}{'HELD-OUT rows':>16}{'HELD-OUT freq':>16}{'sel-window TRAIN rows (diag 44)':>34}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(ctr[j]):>14,}{ctr[j] / len(ytr):>14.10f}{int(cte[j]):>16,}{cte[j] / len(yte):>16.10f}{sel_tr[j]:>34,}")
    emit(f"  {'TOTAL':<10}{len(ytr):>14,}{'':>14}{len(yte):>16,}")
    return ctr, cte


def label_pass():
    """Pass 1: labels only, all four origin states. Prints the row counts and CLASS COUNTS of both populations before any model."""
    title("PASS 1: ROW COUNTS AND CLASS COUNTS  (labels only; no model has been trained, no score exists yet)")
    emit(f"window: role == 'fit', {WIN_LO} <= from_period <= {WIN_HI} (the selection window was {WIN_LO}..{SEL_HI})")
    y, fp, fs, codes, n_null, manifest, _ = load_rows(TRANSIENT, False)[1:]
    emit(f"panel build {manifest['build_stamp']}; rows loaded (all four origin states): {len(y):,}; null/unknown labels dropped {n_null:,}")
    plan = {}
    for pop, origin in POPULATIONS:
        sel = np.isin(fs, [TRANSIENT.index(o) for o in origin])
        ypop, fppop, cpop = y[sel], fp[sel], codes[sel]
        in_train, n_loans, n_tr = split_flags(cpop)
        n_sel = int((fppop <= SEL_HI).sum())
        emit()
        emit(f"POPULATION {pop}: from_state in {origin}")
        emit(f"  rows {len(ypop):,} on the full window, {n_loans:,} loans; rows with from_period <= {SEL_HI}: {n_sel:,} "
             f"(diagnostic 44: {SEL_ROWS[pop]:,}: {'MATCH' if n_sel == SEL_ROWS[pop] else 'DIFFERENT' if not SMOKE else 'not checked in a smoke run'})")
        emit(f"  split by loan_id: seed {SEED}, train fraction {TRAIN_FRAC} -> {n_tr:,} of {n_loans:,} loans to train, {n_loans - n_tr:,} held out; "
             f"train rows {int(in_train.sum()):,}, held-out rows {int((~in_train).sum()):,}; no loan on both sides (asserted)")
        if not SMOKE and n_sel != SEL_ROWS[pop]:
            emit("STOP: the loader does not reproduce diagnostic 44's selection-window population.")
            write_report()
            sys.exit(1)
        ctr, cte = count_table(pop, ypop[in_train], ypop[~in_train], parse_44_train_counts(pop))
        emit(f"  >>> TRAIN rows of the rare classes on the full window: Default {int(ctr[I_DEF]):,}, DQ90+ {int(ctr[I_DQ90]):,}, "
             f"DQ60 {int(ctr[STATES.index('DQ60')]):,}   (selection window, diagnostic 44: Default {parse_44_train_counts(pop)[I_DEF]:,}, "
             f"DQ90+ {parse_44_train_counts(pop)[I_DQ90]:,}, DQ60 {parse_44_train_counts(pop)[STATES.index('DQ60')]:,})")
        months = np.unique(fppop[~in_train])
        emit(f"  held-out rows and events by from_period month (rows / DQ90+ / Default):")
        row = []
        for mth in months:
            mm = (fppop == mth) & ~in_train
            row.append(f"{mth}: {int(mm.sum()):,}/{int((ypop[mm] == I_DQ90).sum())}/{int((ypop[mm] == I_DEF).sum())}")
        for i in range(0, len(row), 4):
            emit("    " + "   ".join(f"{r:<30}" for r in row[i:i + 4]))
        plan[pop] = {"ctr": ctr, "cte": cte, "n_loans": n_loans, "rows": len(ypop)}
        del sel, ypop, fppop, cpop, in_train
        gc.collect()
    write_report()
    del y, fp, fs, codes
    gc.collect()
    return plan


def topshare(p, ev, frac):
    """Tie-aware share of the events that fall in the top `frac` of the ranking by p (descending).
    Returns (expected events captured, slots k, size of the tied block at the cutoff, tied events share)."""
    n = len(p)
    k = int(np.ceil(frac * n))
    v = np.partition(p, n - k)[n - k]
    gt = p > v
    eq = p == v
    n_gt, n_eq = int(gt.sum()), int(eq.sum())
    got = float(ev[gt].sum()) + float(ev[eq].sum()) * (k - n_gt) / n_eq
    return got, k, n_eq


def within_month(pop, prob, yte, fpte):
    title(f"WITHIN-MONTH DISCRIMINATION  ({pop}, R1, held-out rows only, each month scored separately)")
    emit("Macro columns are constant within a calendar month, so skill measured inside one month cannot come from the calendar. Months are NOT pooled.")
    if pop == "POOLED":
        emit("NOTE: POOLED rows within a month differ in from_state (a feature of the loan's own state, not a calendar feature); the ranking is not "
             "conditioned on from_state, so part of any within-month skill here can be the origin-state mix.")
    months = np.unique(fpte)
    nrow = {int(m): int((fpte == m).sum()) for m in months}
    ndef = {int(m): int(((fpte == m) & (yte == I_DEF)).sum()) for m in months}
    top_def = sorted(nrow, key=lambda m: (-ndef[m], m))[:3]
    mid = [m for m in nrow if 200701 <= m <= 200712 and m not in top_def]
    mid = sorted(sorted(mid, key=lambda m: m)[:], key=lambda m: -nrow[m])[:3]
    emit(f"top-3 held-out months by Default events: {top_def} (Default events {[ndef[m] for m in top_def]}, rows {[nrow[m] for m in top_def]})")
    emit(f"three mid-window months by row count (middle third 200701..200712, excluding the above): {mid} (rows {[nrow[m] for m in mid]}, "
         f"Default events {[ndef[m] for m in mid]})")
    emit("random ranking would capture 1% of events in the top 1% and 10% in the top 10%. 'tied' = rows sharing the predicted value at the cutoff "
         "(events in that block are credited pro rata). THIN = fewer than 10 events: read it as a count, not a share.")
    for label, j in (("DEFAULT", I_DEF), ("DQ90+", I_DQ90)):
        emit()
        emit(f"  target {label}: ranking by predicted {label} probability, against actual {label}")
        emit(f"  {'month':<8}{'group':<12}{'rows':>11}{'events':>8}{'top 1%: events (share)':>26}{'tied@1%':>10}{'top 10%: events (share)':>27}{'tied@10%':>10}  note")
        for grp, ms in (("top-Default", top_def), ("mid-window", mid)):
            for m in ms:
                mm = fpte == m
                p, ev = prob[mm, j], (yte[mm] == j).astype(np.float64)
                n_ev = int(ev.sum())
                if n_ev == 0:
                    emit(f"  {m:<8}{grp:<12}{int(mm.sum()):>11,}{0:>8}{'n/a (no events)':>26}{'':>10}{'n/a (no events)':>27}{'':>10}  THIN (0 events)")
                    continue
                g1, k1, t1 = topshare(p, ev, 0.01)
                g10, k10, t10 = topshare(p, ev, 0.10)
                emit(f"  {m:<8}{grp:<12}{int(mm.sum()):>11,}{n_ev:>8}{f'{g1:.2f} ({100 * g1 / n_ev:.1f}%)':>26}{t1:>10,}"
                     f"{f'{g10:.2f} ({100 * g10 / n_ev:.1f}%)':>27}{t10:>10,}  " + ("THIN (<10 events)" if n_ev < 10 else ""))
    write_report()


def run_model(lgb, cap, pop, tag, Xtr, ytr, Xte, yte, fpte, feats, categorical, freq_tr, pr_te_cls, do_within):
    ntr = len(ytr)
    r = {"pop": pop, "tag": tag}
    title(f"RUN {pop} / {tag}")
    emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")
    emit("params: " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items()) + f";  num_boost_round={N_ROUNDS}")
    if tag.startswith("R2"):
        emit("CAVEAT: est_ltv is borrower-level but contains an HPI component, so this run is NOT macro-free.")
    print("  training ...", flush=True)
    watch = MemWatch(on_abort=on_abort, abort_mb=700)
    watch.phase = f"{pop} {tag}"
    watch.start()
    cap.msgs.clear()
    t0 = time.perf_counter()
    Xd = Xtr if list(Xtr.columns) == feats else Xtr[feats]
    ds = lgb.Dataset(Xd, label=ytr.astype(np.int64), categorical_feature=categorical, free_raw_data=True, params=PARAMS)
    booster = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
    t_train = time.perf_counter() - t0
    del ds, Xd
    gc.collect()
    assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
    assert booster.pandas_categorical == [CATEGORIES[c] for c in categorical]
    msgs = list(cap.msgs)
    # --- held-out
    t1 = time.perf_counter()
    Xe = Xte[feats]
    raw = np.asarray(booster.predict(Xe, raw_score=True), dtype=np.float64)
    assert raw.shape == (len(yte), K)
    prob = softmax(raw)
    sub = np.arange(0, len(yte), max(1, len(yte) // 100_000))
    r["softmax_vs_predict"] = float(np.abs(prob[sub] - booster.predict(Xe.iloc[sub])).max())
    del Xe
    t_pred_te = time.perf_counter() - t1
    yi = yte.astype(np.int64)
    cls_te = brier_by_class(prob, yi)
    b_te = float(cls_te.sum())
    mean_p = prob.mean(axis=0)
    rmin, rmax = raw.min(axis=0), raw.max(axis=0)
    del raw
    # --- train side, chunked
    t2 = time.perf_counter()
    sse = np.zeros(K)
    for a in range(0, ntr, CHUNK):
        b = min(a + CHUNK, ntr)
        P = softmax(booster.predict(Xtr.iloc[a:b][feats], raw_score=True))
        P[np.arange(b - a), ytr[a:b].astype(np.int64)] -= 1.0          # p - one-hot, in place
        sse += (P ** 2).sum(axis=0)
        del P
    cls_tr = sse / ntr
    b_tr = float(cls_tr.sum())
    t_pred_tr = time.perf_counter() - t2
    watch.stop()
    del booster
    gc.collect()
    ctr = counts(ytr.astype(np.int64))
    cte = counts(yi)
    pr_te = float(pr_te_cls.sum())
    emit(f"library messages: " + (" | ".join(f"{m} x{msgs.count(m)}" for m in sorted(set(msgs))) if msgs else "none"))
    emit(f"wall-clock: lgb.train (Dataset construction included) {t_train:.1f} s ({t_train / 60:.2f} min = {t_train / N_ROUNDS:.2f} s/round); "
         f"held-out predict {t_pred_te:.1f} s; train-side predict {t_pred_tr:.1f} s;  peak RSS in-run sampled {watch._peak / GB:.2f} GB, "
         f"process peak {watch.peak_rss() / GB:.2f} GB;  min system available {watch.min_avail / GB:.2f} GB")
    emit(f"check: max |softmax(raw) - predict()| on 100,000 held-out rows = {r['softmax_vs_predict']:.3e}")
    emit()
    emit("  MULTICLASS BRIER SCORE  (" + BRIER_FORMULA + ")")
    emit(f"  prior-only reference, THIS window (train frequencies for every held-out row) : {pr_te:.7f}")
    emit(f"  held-out Brier                                                  : {b_te:.7f}")
    emit(f"  held-out minus prior-only                                       : {b_te - pr_te:+.7f}")
    emit(f"  train Brier                                                     : {b_tr:.7f}")
    emit(f"  gap (held-out - train)                                          : {b_te - b_tr:+.7f}")
    emit()
    emit("  PER-CLASS BRIER CONTRIBUTION, held-out  (term j = mean_i (p_ij - o_ij)^2; the six terms add to the total)")
    emit(f"  {'class':<10}{'model term':>14}{'% of total':>12}{'prior-only term':>18}{'model - prior':>16}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{cls_te[j]:>14.8f}{100 * cls_te[j] / b_te:>11.4f}%{pr_te_cls[j]:>18.8f}{cls_te[j] - pr_te_cls[j]:>+16.8f}")
    emit(f"  {'TOTAL':<10}{b_te:>14.8f}{100.0:>11.4f}%{pr_te:>18.8f}{b_te - pr_te:>+16.8f}   (sum of the six terms {cls_te.sum():.8f})")
    emit()
    emit("  PER-CLASS DISTRIBUTION  (mean predicted probability on held-out rows vs the held-out actual frequency)")
    emit(f"  {'class':<10}{'train rows':>13}{'train freq':>15}{'held-out rows':>15}{'held-out freq':>15}{'mean pred prob':>16}{'pred / actual':>15}")
    for j, s in enumerate(STATES):
        ratio = f"{mean_p[j] / (cte[j] / len(yte)):>14.3f}x" if cte[j] else f"{'n/a (0 rows)':>15}"
        emit(f"  {s:<10}{int(ctr[j]):>13,}{ctr[j] / ntr:>15.10f}{int(cte[j]):>15,}{cte[j] / len(yte):>15.10f}{mean_p[j]:>16.10f}{ratio}")
    scale = float(np.log(freq_tr).max() - np.log(freq_tr).min())
    emit()
    emit("  RAW-SCORE SPREAD across held-out rows, per class (max - min of the class's raw score)")
    emit(f"  {'class':<10}{'min raw':>16}{'max raw':>16}{'spread':>16}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{rmin[j]:>16.4f}{rmax[j]:>16.4f}{rmax[j] - rmin[j]:>16.4f}")
    emit(f"  largest spread of any class: {float((rmax - rmin).max()):.4f}   (for scale: the six log train frequencies span {scale:.2f}; "
         f"diagnostic 46, 1 round, init ON: 194 on the same scale for CURRENT)")
    r.update(b_te=b_te, b_tr=b_tr, pr_te=pr_te, spread=float((rmax - rmin).max()), t_train=t_train, mean_p=mean_p, cte=cte,
             cls_te=cls_te, peak=watch.peak_rss(), rss=watch._peak, min_avail=watch.min_avail, ctr=ctr, msgs=msgs)
    write_report()
    if do_within:
        within_month(pop, prob, yte, fpte)
    del prob
    gc.collect()
    return r


def run_population(lgb, cap, pop, origin, plan):
    pooled = len(origin) > 1
    title(f"POPULATION {pop}: pass 2, feature matrix")
    X, y, fp, fs, codes, n_null, manifest, n_loans = load_rows(origin, True)
    emit(f"rows {len(y):,}; loans {n_loans:,}; null/unknown labels dropped {n_null:,}")
    in_train, nl, n_tr = split_flags(codes)
    ctr_chk, cte_chk = counts(y[in_train].astype(np.int64)), counts(y[~in_train].astype(np.int64))
    same = (list(ctr_chk) == list(plan[pop]["ctr"])) and (list(cte_chk) == list(plan[pop]["cte"]))
    emit(f"split consistency with pass 1: train and held-out class counts {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    if not same:
        emit("STOP: pass 2 does not reproduce the pass-1 split.")
        write_report()
        sys.exit(1)
    del codes, fs
    feats = list(BASE_FEATURES) + (["from_state"] if pooled else [])
    cols = feats
    Xtr, ytr = X.loc[in_train, cols].reset_index(drop=True), y[in_train]
    Xte, yte, fpte = X.loc[~in_train, cols].reset_index(drop=True), y[~in_train], fp[~in_train]
    del X, y, fp, in_train
    gc.collect()
    freq_tr = counts(ytr.astype(np.int64)) / len(ytr)
    assert (freq_tr > 0).all()
    pr_te_cls = brier_by_class(np.broadcast_to(freq_tr, (len(yte), K)), yte.astype(np.int64))
    pr_tr = float((freq_tr * (1 - freq_tr)).sum())
    emit(f"prior-only reference on this window: held-out Brier {pr_te_cls.sum():.7f} (train rows {pr_tr:.7f}); "
         f"train frequencies " + ", ".join(f"{s} {f:.8f}" for s, f in zip(STATES, freq_tr)))
    feats2 = [f for f in feats if f not in R2_DROP]
    cat1 = ["property_state"] + (["from_state"] if pooled else [])
    res = []
    for tag, fl, do_w in ((("R1 full features"), feats, True), (("R2 borrower-only features"), feats2, False)):
        res.append(run_model(lgb, cap, pop, tag, Xtr, ytr, Xte, yte, fpte, fl, cat1, freq_tr, pr_te_cls, do_w))
    del Xtr, Xte
    gc.collect()
    return res, float(pr_te_cls.sum()), pr_tr


def main():
    import lightgbm as lgb

    cap = Capture()
    lgb.register_logger(cap)
    emit("FULL FIT WINDOW capacity check: CURRENT and POOLED, full vs borrower-only features, plus within-month discrimination   "
         "(measurement only; no selection, no recommendation)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs" + ("   SMOKE RUN (first partition only)" if SMOKE else ""))
    emit("CAVEAT carried through the report: est_ltv is borrower-level but contains an HPI component, so the R2 'borrower-only' runs are "
         "not macro-free; the R1-vs-R2 comparison cannot separate macro from est_ltv's HPI part.")
    plan = label_pass()
    out = []
    for pop, origin in POPULATIONS:
        out.append(run_population(lgb, cap, pop, origin, plan))
        write_report()

    title("SUMMARY  (figures only)")
    emit(f"{'run':<34}{'prior-only':>12}{'held-out':>12}{'held-out - prior':>18}{'train Brier':>13}{'gap':>12}{'max raw spread':>16}{'train s':>9}{'RSS GB':>8}")
    for (res, pr_te, pr_tr), (pop, _) in zip(out, POPULATIONS):
        for r in res:
            emit(f"{pop + ' / ' + r['tag']:<34}{pr_te:>12.7f}{r['b_te']:>12.7f}{r['b_te'] - pr_te:>+18.7f}{r['b_tr']:>13.7f}"
                 f"{r['b_te'] - r['b_tr']:>+12.7f}{r['spread']:>16.2f}{r['t_train']:>9.1f}{r['peak'] / GB:>8.2f}")
    title("FINDINGS  (measured only; no interpretation, no recommendation)")
    for (res, pr_te, pr_tr), (pop, _) in zip(out, POPULATIONS):
        ctr = res[0]["ctr"]
        emit(f"{pop}:")
        emit(f"  - TRAIN rows on the full window: Default {int(ctr[I_DEF]):,}, DQ90+ {int(ctr[I_DQ90]):,}, DQ60 {int(ctr[STATES.index('DQ60')]):,} "
             f"(selection window: {', '.join(str(v) for v in [parse_44_train_counts(pop)[I_DEF], parse_44_train_counts(pop)[I_DQ90], parse_44_train_counts(pop)[STATES.index('DQ60')]])}).")
        for r in res:
            emit(f"  - {r['tag']}: held-out Brier {r['b_te']:.7f} vs prior-only {pr_te:.7f} (difference {r['b_te'] - pr_te:+.7f}); train Brier {r['b_tr']:.7f} "
                 f"(gap {r['b_te'] - r['b_tr']:+.7f}); largest raw-score spread {r['spread']:.2f}.")
            ratios = ", ".join(f"{s} {r['mean_p'][j] / (r['cte'][j] / r['cte'].sum()):.2f}x" if r["cte"][j] else f"{s} n/a" for j, s in enumerate(STATES))
            emit(f"    mean predicted / held-out actual by class: {ratios}.")
        emit(f"  - R2 minus R1 held-out Brier: {res[1]['b_te'] - res[0]['b_te']:+.7f}  (est_ltv carries an HPI component in both runs).")
        msgs = sorted({m for r in res for m in r["msgs"]})
        emit("  - library messages: " + (" | ".join(msgs) if msgs else "none") + ".")
    emit("  - within-month discrimination (R1): see the WITHIN-MONTH sections above; each month is reported separately.")
    peak = max(r["peak"] for res, _, _ in out for r in res)
    emit()
    emit(f"PEAK RSS (src.memwatch): process peak {peak / GB:.2f} GB; lowest system-available memory seen "
         f"{min(r['min_avail'] for res, _, _ in out for r in res) / GB:.2f} GB")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped (no rows were sampled): {e}")
        write_report()
        raise
