"""Does regularisation revive the DQ90+-origin model?   MEASUREMENT ONLY (eight runs, no selection, no recommendation).

    python diagnostics/51_dq90_origin_regularised.py   # from the repo root

Diagnostics 40/41 found the DQ90+-origin model never beating its own base rate, on runs that predate diagnostic 49's finding that unbounded leaf values
(raw-score spread 3.4e8, 32.94% of Current-origin held-out rows saturating to exactly 0) were the cause of the Current model's failure. This diagnostic
re-runs the DQ90+-origin model with and without the regularisation of diagnostic 49 run 5, at both of diagnostic 41's capacities, with and without the
two unemployment columns, and measures levels AND ordering on RAW held-out outputs (no calibration). Writes diagnostics/51_dq90_origin_regularised.txt and
persists each run's model to artifacts/diag51/ (cell in the file name, JSON sidecar). Nothing under src/ or data/, no git command. Fit rows only
(load_panel(roles=['fit'])); evaluate and condition rows are never read.

POPULATION. role == 'fit', from_state == 'DQ90+', 200601 <= from_period <= 200812 (the whole fit window, as diagnostic 41). Diagnostic 41 applied no
from_period filter; its observed range was 200605..200812, so the 200601 lower bound cannot bind and the row count must equal 41's 201,792 (the script
stops if it does not). The split is diagnostic 41's: sorted unique loan_id strings shuffled with numpy default_rng(40), first 80% (rounded) to train;
per-class train / held-out counts must equal 41's table (hard-coded below) or the script stops. Label mapping (to_state categorical -> codes into
panel.STATES), class order, float32 numerics, property_state a native categorical with the fixed category list, nulls left as-is: the same code path
as 41, whose handling diagnostics 47/48/49/50 imported unchanged. Brier (44's explicit multiclass Brier), softmax, Tab (exact tie-aware AUC, top-q%, deciles)
and Boot (loan-resampling bootstrap) are imported from 47/48/50, not re-derived.

FEATURES. Full: fico ltv dti rate_spread property_state est_ltv unemp_level unemp_chg_12m (41's WITH MACRO). Borrower-only: unemp_level and unemp_chg_12m
dropped (41's WITHOUT MACRO; est_ltv still carries an HPI component). run_length is NOT a feature (deliberately).

GRID (2 x 2 x 2). capacity {DEFAULTS num_leaves 31 / min_data_in_leaf 20; REDUCED 8 / 100} x regularisation {OFF min_sum_hessian_in_leaf 1e-3, lambda_l2 0;
ON 10, 10} x features {full, borrower-only}. objective multiclass, num_class 6, device cpu, 100 rounds, verbosity 0 (41 used -1; library messages are captured).
The (DEFAULTS, OFF, full) cell is compared explicitly with diagnostic 41's WITH MACRO / A run.

SCORING. Held-out rows, one predict(raw_score=True), probabilities = softmax(raw). AUC counts tied pairs 0.5 (exact). The 95% bootstrap interval is a
percentile interval over 500 draws resampling HELD-OUT LOANS (a loan contributes all of its held-out rows or none), numpy default_rng(51), the same 500
draws for every run and target. Two scores per class: P(class) (the specified target; a softmax output, so it varies with the other five classes' raw
scores even when the class's own raw column is constant) and the class's OWN raw score (the figure that establishes class-specific signal).
"""

import gc
import importlib.util
import json
import os
import sys
import time

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


D50 = _load("diag50", "50_tail_ordering.py")                            # imports 49 -> 48 -> 47 (-> 44, 46)
D48 = D50.D48
D47 = D48.D47
STATES, BASE_FEATURES, CATEGORIES = D47.STATES, D47.BASE_FEATURES, D47.CATEGORIES
K = len(STATES)
IDX = {s: STATES.index(s) for s in STATES}
I_CUR, I_DEF, I_DQ90 = IDX["Current"], IDX["Default"], IDX["DQ90+"]
Tab, Boot, fmt_ci = D48.Tab, D50.Boot, D50.fmt_ci
brier_by_class, counts, softmax = D47.brier_by_class, D47.counts, D47.softmax
BRIER_FORMULA, GB, N_ROUNDS, R2_DROP = D47.BRIER_FORMULA, D47.GB, D47.N_ROUNDS, D47.R2_DROP

SMOKE = os.environ.get("DIAG51_SMOKE") == "1"
OUT = os.environ.get("DIAG51_OUT", os.path.join("diagnostics", "51_dq90_origin_regularised.txt"))
MODEL_DIR = os.environ.get("DIAG51_MODELS", os.path.join("artifacts", "diag51"))
ORIGIN = "DQ90+"
WIN_LO, WIN_HI = 200601, 200812
SEED, TRAIN_FRAC = 40, 0.8
EXPECTED_ROWS = 201792
EXPECT_COUNTS = {"train": [9851, 2131, 3117, 139715, 852, 5606], "held-out": [2514, 503, 765, 35122, 231, 1385]}   # diagnostic 41
REF41 = {"prior_te": 0.570739, "ll_te": 0.565810, "ll_diff": -0.004930,
         "mean_p": [6.3449, 1.3664, 2.0076, 86.1600, 0.6171, 3.5040], "argmax": [15, 0, 0, 40432, 73, 0]}           # 41 WITH MACRO / A, held-out
N_BOOT = 10 if SMOKE else 500
BOOT_SEED = 51
CUTS = [0.01, 0.10]
EPS = 1e-15
BASE_PARAMS = {"objective": "multiclass", "num_class": K, "device_type": "cpu", "verbosity": 0}
CAPS = [("DEFAULTS", dict(num_leaves=31, min_data_in_leaf=20)), ("REDUCED", dict(num_leaves=8, min_data_in_leaf=100))]
REGS = [("OFF", dict(min_sum_hessian_in_leaf=1e-3, lambda_l2=0.0)), ("ON", dict(min_sum_hessian_in_leaf=10.0, lambda_l2=10.0))]
FEATS_FULL = list(BASE_FEATURES)
FEATS_BORR = [f for f in FEATS_FULL if f not in R2_DROP]
FSETS = [("full", FEATS_FULL), ("borrower-only", FEATS_BORR)]
GRID = [(c, g, f) for c in CAPS for g in REGS for f in FSETS]
# reference figures from the Current-origin diagnostics, quoted as printed in their reports
REF_CUR = {"49 run 1 (OFF) Brier-prior": 0.0095352, "49 run 5 (ON) Brier-prior": -0.0002429, "49 run 1 share exactly 0": 32.938,
           "49 run 5 share exactly 0": 0.0, "48 R1 full DQ30 AUC": 0.647, "48 R2 borrower DQ30 AUC": 0.730}


def emit(line=""):
    D47.emit(line)


title = D47.title


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(D47.LINES) + "\n")


D47.write_report = write_report                                 # MemWatch on_abort (D47.on_abort) writes THIS report
D47.OUT = OUT
D47.RULE_W = 118


def cell_name(cap, reg, fs):
    return f"{cap[0]}/{reg[0]}/{fs[0]}"


def cell_stem(cap, reg, fs):
    return f"dq90_cap-{cap[0].lower()}_reg-{reg[0].lower()}_feat-{fs[0].replace('-', '')}"


def load():
    """All DQ90+ fit rows in the window. Same code path as diagnostic 41 load(), plus role and from_period."""
    manifest = panel_io.read_manifest()
    parts = manifest["partitions"][:1] if SMOKE else manifest["partitions"]
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + FEATS_FULL
    xs, ys, ids, fps = [], [], [], []
    n_null = n_all = 0
    for p in parts:
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        assert set(d["role"].astype(str).unique()) <= {"fit"}, "a non-fit row was read"
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m0 = (d["from_state"] == ORIGIN).to_numpy()
        n_all += int(m0.sum())
        m = m0 & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        data = {}
        for c in FEATS_FULL:
            if c in CATEGORIES:
                data[c] = d[c].cat.set_categories(CATEGORIES[c]).reset_index(drop=True)
            else:
                data[c] = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
        xs.append(pd.DataFrame(data))
        ys.append(y[m])
        ids.append(d["loan_id"].astype(str).to_numpy())
        fps.append(fp[m])
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows", flush=True)
        del d, data
    X = pd.concat(xs, ignore_index=True)
    for c, cats in CATEGORIES.items():
        if c in X.columns:
            assert str(X[c].dtype) == "category" and list(X[c].cat.categories) == cats
    return X, np.concatenate(ys), np.concatenate(ids), np.concatenate(fps), n_null, n_all, manifest


def run_cell(lgb, cap_msgs, k, cap, reg, fs, Xtr, ytr, Xte, yte, loan_idx, n_loans, freq_tr, pr_cls):
    name = cell_name(cap, reg, fs)
    feats = fs[1]
    params = dict(BASE_PARAMS, **cap[1], **reg[1])
    r = {"k": k, "name": name, "cap": cap[0], "reg": reg[0], "fs": fs[0], "params": params, "feats": feats}
    print(f"  [run {k}: {name}] training ...", flush=True)
    watch = MemWatch(on_abort=D47.on_abort, abort_mb=700)
    watch.phase = f"run {k} {name}"
    watch.start()
    cap_msgs.clear()
    t0 = time.perf_counter()
    ds = lgb.Dataset(Xtr[feats], label=ytr, categorical_feature=["property_state"], free_raw_data=True, params=params)
    booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
    r["t_train"] = time.perf_counter() - t0
    del ds
    gc.collect()
    assert booster.current_iteration() == N_ROUNDS and booster.feature_name() == feats
    assert booster.pandas_categorical == [CATEGORIES["property_state"]]
    r["msgs"] = list(cap_msgs)
    r["trees"] = booster.num_trees()
    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = cell_stem(cap, reg, fs)
    booster.save_model(os.path.join(MODEL_DIR, stem + ".txt"))
    with open(os.path.join(MODEL_DIR, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump({"origin_states": [ORIGIN], "features": feats, "categorical": ["property_state"], "class_order": STATES, "params": params,
                   "num_boost_round": N_ROUNDS, "window": [WIN_LO, WIN_HI], "split_seed": SEED, "train_fraction": TRAIN_FRAC,
                   "cell": {"capacity": cap[0], "regularisation": reg[0], "features": fs[0]}, "run": k,
                   "made_by": "diagnostics/51_dq90_origin_regularised.py"}, f, indent=1)
    r["model_path"] = os.path.join(MODEL_DIR, stem + ".txt")
    t1 = time.perf_counter()
    raw = np.asarray(booster.predict(Xte[feats], raw_score=True), dtype=np.float64)
    r["t_pred"] = time.perf_counter() - t1
    del booster
    assert raw.shape == (len(yte), K)
    r["spread"] = raw.max(axis=0) - raw.min(axis=0)
    P = softmax(raw)
    yi = yte
    cls = brier_by_class(P, yi)
    r["cls"], r["brier"] = cls, float(cls.sum())
    r["pr"] = float(pr_cls.sum())
    r["mean_p"] = P.mean(axis=0)
    r["logloss"] = float(-np.log(np.clip(P[np.arange(len(yi)), yi], EPS, 1.0)).mean())
    r["argmax"] = counts(P.argmax(axis=1))
    r["share0"] = (P == 0.0).mean(axis=0)
    r["share1"] = (P == 1.0).mean(axis=0)
    r["t"] = {}
    for cname in ("Default", "Current"):
        j = IDX[cname]
        sc, ev = P[:, j].copy(), (yi == j)
        t = Tab(sc, ev)
        tr = Tab(raw[:, j].copy(), ev)
        b = Boot(sc, ev, loan_idx, n_loans)
        if not b.constant and k == 1 and cname == "Default":
            r["boot_check"] = (b.auc(np.ones(n_loans, dtype=np.int32)), t.auc())
        d = {"events": int(t.P), "auc": t.auc(), "ci": b.interval(N_BOOT, BOOT_SEED), "auc_raw": tr.auc(), "distinct_raw": len(tr.u),
             "distinct": len(t.u), "largest": float(t.cnt.max() / t.n), "spread": float(r["spread"][j]),
             "tops": [t.top(q) for q in CUTS], "dec": t.deciles()}
        r["t"][cname] = d
        del t, tr, b, sc, ev
    del P, raw
    watch.stop()
    gc.collect()
    r["rss"], r["peak"], r["min_avail"] = watch._peak, watch.peak_rss(), watch.min_avail
    r["secs"] = r["t_train"] + r["t_pred"]
    return r


def emit_detail(r, cte):
    title(f"RUN {r['k']}: capacity {r['cap']} / regularisation {r['reg']} / features {r['_fs']}")
    emit("params: " + ", ".join(f"{k}={v!r}" for k, v in r["params"].items()) + f";  num_boost_round={N_ROUNDS};  trees {r['trees']};  model: {r['model_path']}")
    emit(f"features ({len(r['feats'])}): {r['feats']}; categorical ['property_state']")
    emit("library messages: " + (" | ".join(f"{m} x{r['msgs'].count(m)}" for m in sorted(set(r["msgs"]))) if r["msgs"] else "none"))
    emit(f"COST: wall-clock lgb.train (Dataset construction included) {r['t_train']:.1f} s; held-out predict {r['t_pred']:.1f} s;  peak RSS in-run sampled "
         f"{r['rss'] / GB:.2f} GB, process peak {r['peak'] / GB:.2f} GB (src.memwatch);  min system available {r['min_avail'] / GB:.2f} GB")
    emit()
    emit("LEVELS  (held-out rows, raw outputs, no calibration)   " + BRIER_FORMULA)
    emit(f"  multiclass Brier {r['brier']:.7f};  prior-only on THIS population {r['pr']:.7f};  difference {r['brier'] - r['pr']:+.7f}")
    emit(f"  {'class':<10}{'Brier term':>13}{'prior-only term':>17}{'model - prior':>15}{'held-out freq':>16}{'mean predicted':>16}{'pred / actual':>15}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{r['cls'][j]:>13.8f}{r['pr_cls'][j]:>17.8f}{r['cls'][j] - r['pr_cls'][j]:>+15.8f}{cte[j] / cte.sum():>16.10f}{r['mean_p'][j]:>16.10f}"
             f"{r['mean_p'][j] / (cte[j] / cte.sum()):>14.3f}x")
    emit(f"  {'TOTAL':<10}{r['cls'].sum():>13.8f}{r['pr_cls'].sum():>17.8f}{r['cls'].sum() - r['pr_cls'].sum():>+15.8f}   (sum of the six terms)")
    emit()
    emit("ORDERING  (held-out rows, raw outputs; AUC with 95% bootstrap interval over held-out loans, " + f"{N_BOOT} draws)")
    for cname, direction in (("Default", "P(Default) vs actual next state == Default"), ("Current", "P(Current) vs actual next state == Current (cure direction)")):
        d = r["t"][cname]
        emit(f"--- {cname}: {direction}; events {d['events']:,} of {int(cte.sum()):,} held-out rows")
        emit(f"  AUC of P({cname}) : {fmt_ci(d['auc'], d['ci'])}")
        emit(f"  the class's OWN raw score: AUC {d['auc_raw']:.5f}; spread (max - min) {d['spread']:.6g}; {d['distinct_raw']:,} distinct values")
        if d["spread"] == 0.0:
            emit(f"  *** RAW SPREAD EXACTLY 0: this model assigns every loan the identical raw {cname} score, so its {cname} column says nothing class-specific "
                 f"(own raw-score AUC exactly 0.5). P({cname}) still differs between loans only through the softmax normalisation by the other five classes' scores; "
                 f"its AUC above is then the other classes' ordering. ***")
        for q, (got, k, tie, strad) in zip(CUTS, d["tops"]):
            emit(f"  top {100 * q:.0f}% ({k:,} rows): {got:.2f} of {d['events']:,} {cname} events = {100 * got / d['events']:.2f}% (random: {100 * q:.0f}%); "
                 f"rows tied at the cutoff {tie:,}{'*' if strad else ''}")
    emit("  ('*' = the cutoff falls inside a tied block; its events are credited pro rata)")
    emit()
    emit("  DEFAULT decile table (decile 1 = lowest P(Default); equal-count groups, tied blocks shared pro rata so counts can be fractional)")
    emit(f"  {'decile':>6}{'rows':>10}{'events':>10}{'actual rate':>13}{'mean predicted':>16}{'distinct values':>17}")
    for i, (n_, e_, ps_, nd) in enumerate(r["t"]["Default"]["dec"]):
        emit(f"  {i + 1:>6}{int(n_):>10,}{e_:>10.2f}{100 * e_ / n_:>12.4f}%{100 * ps_ / n_:>15.4f}%{nd:>17,}")
    emit()
    emit("SATURATION  (held-out rows)")
    emit(f"  {'class':<10}{'share P exactly 0':>20}{'share P exactly 1':>20}{'raw spread':>16}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{100 * r['share0'][j]:>19.4f}%{100 * r['share1'][j]:>19.4f}%{r['spread'][j]:>16.6g}")
    for cname in ("Default", "Current"):
        d = r["t"][cname]
        emit(f"  P({cname}): {d['distinct']:,} distinct predicted values; largest tie block {100 * d['largest']:.4f}% of rows")
    emit(f"  argmax counts (rows whose largest probability is that class): " + "  ".join(f"{s} {int(n):,}" for s, n in zip(STATES, r["argmax"])))


def main():
    import lightgbm as lgb

    cap = D47.Capture()
    lgb.register_logger(cap)
    emit("DQ90+-ORIGIN MODEL, REGULARISATION x CAPACITY x FEATURES   (diagnostic 51; measurement only; no tuning beyond the grid, no selection, no recommendation)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs" + ("   SMOKE RUN (first partition only; 10 bootstrap draws)" if SMOKE else ""))
    title("SETUP")
    X, y, ids, fps, n_null, n_all, manifest = load()
    emit(f"panel build {manifest['build_stamp']}; role == 'fit', from_state == '{ORIGIN}', {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"rows loaded {len(y):,} (diagnostic 41: {EXPECTED_ROWS:,}: {'MATCH' if len(y) == EXPECTED_ROWS else 'DIFFERENT'}); DQ90+-origin fit rows before the window filter "
         f"{n_all:,}; null/unknown labels dropped {n_null:,}; observed from_period range {int(fps.min())}..{int(fps.max())} "
         f"(diagnostic 41 observed 200605..200812 with no filter)")
    if len(y) != EXPECTED_ROWS and not SMOKE:
        emit("STOP: the row count disagrees with diagnostic 41. Nothing was trained.")
        write_report()
        sys.exit(1)
    uniq = np.unique(ids)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    in_train = np.isin(ids, uniq[perm[:n_tr]])
    assert not (set(ids[in_train]) & set(ids[~in_train])), "a loan_id is on both sides"
    emit(f"split by loan_id (diagnostic 41's scheme): seed {SEED}, sorted unique loan_ids shuffled with numpy default_rng, train fraction {TRAIN_FRAC} -> "
         f"{n_tr:,} of {len(uniq):,} loans to train, {len(uniq) - n_tr:,} held out; train rows {int(in_train.sum()):,}, held-out rows {int((~in_train).sum()):,}; "
         f"no loan on both sides (asserted)")
    ctr, cte = counts(y[in_train]), counts(y[~in_train])
    emit(f"  {'class':<10}{'TRAIN rows':>14}{'diag 41':>12}{'HELD-OUT rows':>16}{'diag 41':>12}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(ctr[j]):>14,}{EXPECT_COUNTS['train'][j]:>12,}{int(cte[j]):>16,}{EXPECT_COUNTS['held-out'][j]:>12,}")
    same = list(map(int, ctr)) == EXPECT_COUNTS["train"] and list(map(int, cte)) == EXPECT_COUNTS["held-out"]
    emit(f"per-class counts vs diagnostic 41's table: {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    if not same and not SMOKE:
        emit("STOP: the split does not reproduce diagnostic 41's. Nothing was trained.")
        write_report()
        sys.exit(1)
    _, loan_idx = np.unique(ids[~in_train], return_inverse=True)
    loan_idx = loan_idx.astype(np.int32)
    n_loans_te = int(loan_idx.max()) + 1
    Xtr, ytr = X[in_train].reset_index(drop=True), y[in_train]
    Xte, yte = X[~in_train].reset_index(drop=True), y[~in_train]
    del X, y, ids, fps, in_train, perm
    gc.collect()
    freq_tr = counts(ytr) / len(ytr)
    pr_cls = brier_by_class(np.broadcast_to(freq_tr, (len(yte), K)), yte)
    emit(f"held-out loans {n_loans_te:,} (the bootstrap resamples these)")
    emit(f"prior-only on THIS population (train class frequencies for every held-out row): held-out Brier {pr_cls.sum():.7f} "
         f"(per-class terms {', '.join(f'{s} {v:.7f}' for s, v in zip(STATES, pr_cls))}); train frequencies "
         + ", ".join(f"{s} {f:.8f}" for s, f in zip(STATES, freq_tr)))
    prior_ll = float(-np.log(np.clip(freq_tr[yte], EPS, 1.0)).mean())
    emit(f"prior-only multiclass log-loss on held-out rows {prior_ll:.6f} (diagnostic 41: {REF41['prior_te']:.6f})")
    emit("features: full " + str(FEATS_FULL) + "; borrower-only " + str(FEATS_BORR))
    emit("CAVEAT: est_ltv is borrower-level but carries an HPI component, so 'borrower-only' is not macro-free (the same caveat as 41/47/48).")
    emit("NOT features: from_state (constant), loan_age, run_length, run_length_censored. run_length is deliberately excluded (7 Oct decision) although "
         "diagnostic 25 showed it drives DQ90+ destinations.")
    emit(f"grid: {len(GRID)} runs = capacity {[c[0] for c in CAPS]} x regularisation {[g[0] for g in REGS]} x features {[f[0] for f in FSETS]}; "
         f"num_boost_round {N_ROUNDS}; training frame ({len(ytr):,} rows) and held-out frame ({len(yte):,} rows) built once and reused")
    emit("LIMITATION (stated, not resolved): this is the full fit window. If the architecture comparison runs on the selection window (200601..200706, "
         "8,787 DQ90+ rows), this diagnostic characterises a population about 23x larger.")
    write_report()

    title("RUN LOG  (one line per run, as they finish)")
    results = []
    for k, (c, g, f) in enumerate(GRID, start=1):
        r = run_cell(lgb, cap.msgs, k, c, g, f, Xtr, ytr, Xte, yte, loan_idx, n_loans_te, freq_tr, pr_cls)
        r["_fs"], r["pr_cls"] = f[0], pr_cls
        results.append(r)
        d, cu = r["t"]["Default"], r["t"]["Current"]
        emit(f"run {k} [{r['name']}]: Brier {r['brier']:.7f} ({r['brier'] - r['pr']:+.7f} vs prior-only), Default AUC {fmt_ci(d['auc'], d['ci'])}, "
             f"Current AUC {fmt_ci(cu['auc'], cu['ci'])}, share P(Default) exactly 0 {100 * r['share0'][I_DEF]:.2f}%, {r['t_train']:.0f} s, "
             f"process peak {r['peak'] / GB:.2f} GB")
        if k == 1:
            bc = r["boot_check"]
            emit(f"  bootstrap code check (unit loan weights vs plain AUC, Default): {bc[0]:.10f} vs {bc[1]:.10f} -> {'MATCH' if abs(bc[0] - bc[1]) < 1e-9 else '!!! MISMATCH !!!'}")
            if abs(bc[0] - bc[1]) >= 1e-9:
                emit("STOP: the bootstrap check failed.")
                write_report()
                sys.exit(1)
        write_report()
    del Xtr, Xte
    gc.collect()

    title("REPRODUCTION: (DEFAULTS, OFF, full) vs diagnostic 41 WITH MACRO / A. LightGBM defaults   (differences noted, not explained)")
    r1 = results[0]
    emit(f"  held-out prior-only log-loss : this run {prior_ll:.6f}   diagnostic 41 {REF41['prior_te']:.6f}")
    emit(f"  held-out log-loss            : this run {r1['logloss']:.6f}   diagnostic 41 {REF41['ll_te']:.6f}   difference {r1['logloss'] - REF41['ll_te']:+.6f}")
    emit(f"  held-out log-loss minus prior: this run {r1['logloss'] - prior_ll:+.6f}   diagnostic 41 {REF41['ll_diff']:+.6f}")
    emit(f"  {'class':<10}{'mean pred % (this)':>20}{'mean pred % (41)':>19}{'argmax n (this)':>18}{'argmax n (41)':>15}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{100 * r1['mean_p'][j]:>20.4f}{REF41['mean_p'][j]:>19.4f}{int(r1['argmax'][j]):>18,}{REF41['argmax'][j]:>15,}")
    emit("  (41 reported log-loss and mean predicted/argmax, not Brier; 41's run used verbosity -1 and predict(); this one verbosity 0 and softmax(predict(raw_score=True)).)")

    title("SUMMARY TABLE  (one row per run; held-out rows, raw outputs; AUC with 95% bootstrap interval over held-out loans)")
    emit(f"prior-only held-out Brier on this population: {pr_cls.sum():.7f}")
    emit(f"{'run':>3} {'capacity':<9}{'regularisation':<15}{'features':<14}{'Brier':>10}{'Brier-prior':>13}{'Default AUC [95% CI]':>33}{'Def own-raw AUC':>17}{'Def raw spread':>16}"
         f"{'Current AUC [95% CI]':>33}{'P(Def)=0 share':>16}")
    for r in results:
        d, cu = r["t"]["Default"], r["t"]["Current"]
        emit(f"{r['k']:>3} {r['cap']:<9}{r['reg']:<15}{r['fs']:<14}{r['brier']:>10.7f}{r['brier'] - r['pr']:>+13.7f}{fmt_ci(d['auc'], d['ci']):>33}{d['auc_raw']:>17.5f}"
             f"{d['spread']:>16.5g}{fmt_ci(cu['auc'], cu['ci']):>33}{100 * r['share0'][I_DEF]:>15.3f}%")
    emit(f"(Default events {results[0]['t']['Default']['events']:,}, Current events {results[0]['t']['Current']['events']:,} of {len(yte):,} held-out rows. "
         f"[no ranking] = the score is one constant for every loan.)")
    emit()
    emit(f"{'run':>3} {'capacity':<9}{'regularisation':<15}{'features':<14}{'Cur own-raw AUC':>17}{'Cur raw spread':>16}{'Def top1% n (share)':>22}{'Def top10% n (share)':>23}"
         f"{'Cur top10% n (share)':>23}{'Def argmax n':>14}{'wall s':>8}{'peak GB':>9}")
    for r in results:
        d, cu = r["t"]["Default"], r["t"]["Current"]
        t1 = "%.1f (%.1f%%)" % (d["tops"][0][0], 100 * d["tops"][0][0] / d["events"])
        t10 = "%.1f (%.1f%%)" % (d["tops"][1][0], 100 * d["tops"][1][0] / d["events"])
        c10 = "%.0f (%.1f%%)" % (cu["tops"][1][0], 100 * cu["tops"][1][0] / cu["events"])
        emit(f"{r['k']:>3} {r['cap']:<9}{r['reg']:<15}{r['fs']:<14}{cu['auc_raw']:>17.5f}{cu['spread']:>16.5g}"
             f"{t1:>22}{t10:>23}{c10:>23}{int(r['argmax'][I_DEF]):>14,}{r['t_train']:>8.1f}{r['peak'] / GB:>9.2f}")

    for r in results:
        emit_detail(r, cte)

    def pick(cap_, reg_, fs_):
        return next(r for r in results if r["cap"] == cap_ and r["reg"] == reg_ and r["fs"] == fs_)

    def row(r):
        d, cu = r["t"]["Default"], r["t"]["Current"]
        return (r["brier"] - r["pr"], d["auc"], cu["auc"], d["auc_raw"], 100 * r["share0"][I_DEF], r["spread"].max())

    title("DESCRIPTIVE: DOES REGULARISATION CHANGE ANYTHING ON THIS POPULATION?  (OFF -> ON, same capacity and features)")
    emit(f"{'capacity':<10}{'features':<14}{'state':<6}{'Brier-prior':>13}{'Default AUC':>13}{'Current AUC':>13}{'Def own-raw':>13}{'P(Def)=0 %':>12}{'largest raw spread':>20}")
    for c in CAPS:
        for f in FSETS:
            for g in REGS:
                v = row(pick(c[0], g[0], f[0]))
                emit(f"{c[0]:<10}{f[0]:<14}{g[0]:<6}{v[0]:>+13.7f}{v[1]:>13.5f}{v[2]:>13.5f}{v[3]:>13.5f}{v[4]:>12.3f}{v[5]:>20.5g}")
            a, b = row(pick(c[0], "OFF", f[0])), row(pick(c[0], "ON", f[0]))
            emit(f"{'':<10}{'':<14}{'ON-OFF':<6}{b[0] - a[0]:>+13.7f}{b[1] - a[1]:>+13.5f}{b[2] - a[2]:>+13.5f}{b[3] - a[3]:>+13.5f}{b[4] - a[4]:>+12.3f}{b[5] - a[5]:>+20.5g}")
    emit("Current-origin reference (diagnostics 48/49, borrower-only features, defaults capacity, as printed there): "
         f"Brier-prior OFF {REF_CUR['49 run 1 (OFF) Brier-prior']:+.7f} -> ON {REF_CUR['49 run 5 (ON) Brier-prior']:+.7f}; share exactly 0 (leaves-Current score) "
         f"{REF_CUR['49 run 1 share exactly 0']:.3f}% -> {REF_CUR['49 run 5 share exactly 0']:.3f}%; DQ30 AUC 0.72976 -> 0.77469; DQ90+ AUC 0.81295 -> 0.88329 (diagnostic 50).")
    nb = sum(1 for c in CAPS for f in FSETS if row(pick(c[0], "ON", f[0]))[0] < row(pick(c[0], "OFF", f[0]))[0])
    emit(f"cells (of 4 capacity x feature cells) where ON gives a lower Brier-prior than OFF: {nb}; where ON gives a higher Default AUC: "
         f"{sum(1 for c in CAPS for f in FSETS if row(pick(c[0], 'ON', f[0]))[1] > row(pick(c[0], 'OFF', f[0]))[1])}; "
         f"higher Current AUC: {sum(1 for c in CAPS for f in FSETS if row(pick(c[0], 'ON', f[0]))[2] > row(pick(c[0], 'OFF', f[0]))[2])}.")
    emit("Brier-prior below zero (model beats prior-only on THIS population's Brier): " + (", ".join(f"{r['k']} [{r['name']}]" for r in results if r["brier"] < r["pr"]) or "none"))

    title("DESCRIPTIVE: DOES REDUCED CAPACITY STILL WIN ONCE REGULARISATION IS ON?  (same regularisation and features)")
    emit(f"{'regularisation':<16}{'features':<14}{'capacity':<10}{'Brier-prior':>13}{'Default AUC':>13}{'Current AUC':>13}{'Def own-raw':>13}")
    for g in REGS:
        for f in FSETS:
            for c in CAPS:
                v = row(pick(c[0], g[0], f[0]))
                emit(f"{g[0]:<16}{f[0]:<14}{c[0]:<10}{v[0]:>+13.7f}{v[1]:>13.5f}{v[2]:>13.5f}{v[3]:>13.5f}")
            a, b = row(pick("DEFAULTS", g[0], f[0])), row(pick("REDUCED", g[0], f[0]))
            emit(f"{'':<16}{'':<14}{'RED-DEF':<10}{b[0] - a[0]:>+13.7f}{b[1] - a[1]:>+13.5f}{b[2] - a[2]:>+13.5f}{b[3] - a[3]:>+13.5f}")
    emit("cells where REDUCED has the lower Brier: OFF " + str([f[0] for f in FSETS if row(pick("REDUCED", "OFF", f[0]))[0] < row(pick("DEFAULTS", "OFF", f[0]))[0]] or "none")
         + "; ON " + str([f[0] for f in FSETS if row(pick("REDUCED", "ON", f[0]))[0] < row(pick("DEFAULTS", "ON", f[0]))[0]] or "none"))
    emit("Diagnostic 41 reference (regularisation off, held-out log-loss, DEFAULTS vs REDUCED): with macro 0.565810 vs 0.542909; without macro 0.553580 vs 0.547922. "
         "Diagnostic 44 (Current and pooled, selection window) found the opposite direction.")

    title("DESCRIPTIVE: DOES MACRO HELP OR HARM HERE?  (full vs borrower-only, same capacity and regularisation)")
    emit(f"{'capacity':<10}{'regularisation':<16}{'features':<14}{'Brier-prior':>13}{'Default AUC':>13}{'Current AUC':>13}{'Def own-raw':>13}")
    for c in CAPS:
        for g in REGS:
            for f in FSETS:
                v = row(pick(c[0], g[0], f[0]))
                emit(f"{c[0]:<10}{g[0]:<16}{f[0]:<14}{v[0]:>+13.7f}{v[1]:>13.5f}{v[2]:>13.5f}{v[3]:>13.5f}")
            a, b = row(pick(c[0], g[0], "borrower-only")), row(pick(c[0], g[0], "full"))
            emit(f"{'':<10}{'':<16}{'FULL-BORR':<14}{b[0] - a[0]:>+13.7f}{b[1] - a[1]:>+13.5f}{b[2] - a[2]:>+13.5f}{b[3] - a[3]:>+13.5f}")
    emit(f"Current-origin reference (diagnostic 48, defaults, regularisation off): DQ30 AUC {REF_CUR['48 R1 full DQ30 AUC']:.3f} with macro vs {REF_CUR['48 R2 borrower DQ30 AUC']:.3f} without.")
    emit("cells where the full set has the higher Default AUC: " + str([f"{c[0]}/{g[0]}" for c in CAPS for g in REGS if row(pick(c[0], g[0], 'full'))[1] > row(pick(c[0], g[0], 'borrower-only'))[1]] or "none")
         + "; higher Current AUC: " + str([f"{c[0]}/{g[0]}" for c in CAPS for g in REGS if row(pick(c[0], g[0], 'full'))[2] > row(pick(c[0], g[0], 'borrower-only'))[2]] or "none")
         + "; lower Brier: " + str([f"{c[0]}/{g[0]}" for c in CAPS for g in REGS if row(pick(c[0], g[0], 'full'))[0] < row(pick(c[0], g[0], 'borrower-only'))[0]] or "none"))

    title("FINDINGS  (measured only; no interpretation, no recommendation)")
    emit(f"- Population: {len(yte) + len(ytr):,} DQ90+-origin fit rows (diagnostic 41: {EXPECTED_ROWS:,}); train {len(ytr):,}, held-out {len(yte):,} ({n_loans_te:,} loans); "
         f"split counts vs 41: {'IDENTICAL' if same else 'DIFFERENT'}. Held-out Default events {int(cte[I_DEF]):,}, Current events {int(cte[I_CUR]):,}. "
         f"Prior-only held-out Brier on this population {pr_cls.sum():.7f}.")
    emit(f"- Reproduction, (DEFAULTS, OFF, full): held-out log-loss {r1['logloss']:.6f} vs diagnostic 41's {REF41['ll_te']:.6f} (difference {r1['logloss'] - REF41['ll_te']:+.6f}); "
         f"argmax Default count {int(r1['argmax'][I_DEF])} (41: 0).")
    for r in results:
        d, cu = r["t"]["Default"], r["t"]["Current"]
        emit(f"- run {r['k']} [{r['name']}]: Brier {r['brier']:.7f} ({r['brier'] - r['pr']:+.7f} vs prior-only); Default AUC {fmt_ci(d['auc'], d['ci'])}, own raw-score AUC {d['auc_raw']:.5f}, "
             f"raw spread {d['spread']:.5g}, top 1% holds {d['tops'][0][0]:.1f} / top 10% {d['tops'][1][0]:.1f} of {d['events']:,} Default events; "
             f"Current AUC {fmt_ci(cu['auc'], cu['ci'])}, own raw-score AUC {cu['auc_raw']:.5f}, raw spread {cu['spread']:.5g}; "
             f"share P(Default) exactly 0 {100 * r['share0'][I_DEF]:.3f}% / exactly 1 {100 * r['share1'][I_DEF]:.3f}%; largest raw spread {r['spread'].max():.5g}; argmax Default {int(r['argmax'][I_DEF]):,}.")
    zero = [r["k"] for r in results if r["t"]["Default"]["spread"] == 0.0]
    emit(f"- Runs whose Default raw-score spread is exactly 0: {zero or 'none'}; whose Current raw-score spread is exactly 0: {[r['k'] for r in results if r['t']['Current']['spread'] == 0.0] or 'none'}.")
    emit(f"- Regularisation, macro and capacity contrasts: see the three DESCRIPTIVE sections above (figures only).")
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
