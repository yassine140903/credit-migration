"""Ordering on the DQ90+, DQ60 and DEFAULT columns across the nine diagnostic-49 models.   MEASUREMENT ONLY (scoring, no training).

    python diagnostics/50_tail_ordering.py   # from the repo root

Diagnostic 49's gate was DQ30 ordering; DQ90+ and Default contribute almost nothing to the aggregate Brier, so the headline metric cannot see a model that
gives up on them (runs 7-9 reported a raw-score spread of exactly 0 for both). This diagnostic measures ordering on those columns for every swept model.
It does not retrain, tune, select or recommend. Writes diagnostics/50_tail_ordering.txt and nothing else: no model is written, nothing under src/ or
data/, no git command. Fit rows only (load_panel(roles=['fit'])); evaluate and condition rows are never read.

MODELS: the nine persisted models in artifacts/diag49/ (each current_R2_<levers>.txt with its .json sidecar). The lever values are READ FROM THE SIDECAR
(params.min_sum_hessian_in_leaf, params.lambda_l2), not from the file name. Feature order and class order are read from the sidecar; predicted columns are
mapped to the panel.STATES order through the recorded class names, never positionally. Sanity check before anything else: run 5 is rescored and must give
held-out Brier 0.0330520 and DQ30 AUC 0.77469 (diagnostic 49) or the script stops; every model's rescored Brier and DQ30 AUC are also compared with
diagnostic 49's printed summary table.

POPULATION / SPLIT / FEATURES: identical to diagnostics 48 and 49 (imported from 47's loader, not re-derived): CURRENT origin, role == 'fit', 200601 <=
from_period <= 200812, by-loan 80/20 split with numpy default_rng(40), R2 borrower-only features, property_state a native categorical with the fixed
category list, panel.STATES class order. The per-class held-out counts must equal 48/49's table (hard-coded below) or the script stops. Only the
held-out frame is kept (built once and reused for all nine models); no training frame is built.

TARGETS per model (held-out rows, RAW model outputs, no calibration; score = the model's probability for the class, event = actual next state is that class):
  PRIMARY DQ90+ ; SECONDARY DQ60 ; THIN Default (19 held-out events: TOO THIN TO MEASURE ORDERING; reported as counts with a bootstrap interval).
  DQ30 AUC is rescored for the side-by-side column and checked against diagnostic 49.
Statistics: AUC with ties counted 0.5 (exact, rank-based, from the table of distinct predicted values); distinct predicted values and the share of rows in
the largest tie block; raw-score spread (max - min of the class's raw score; exactly 0 means the model says nothing about the class); events captured in
the top 1% and top 10% as a count and a share, with the rows tied at the cutoff (the events of a block straddling the cutoff are credited pro rata); equal-count
decile table (ties shared pro rata) with distinct predicted values per decile.

TWO SCORES PER CLASS. P(class) is a softmax output, so it divides by the other five classes' scores: a class whose RAW column is constant for every loan
still gets a P(class) that differs between loans, and that variation is the other classes' ordering, not class-specific signal. The specified target
(P(class) against the actual class) is therefore reported together with the AUC of the class's OWN raw score (the raw column itself), which is exactly 0.5
with one distinct value when the raw column is constant. Neither is relabelled: both are printed, and where they differ the text says so.

BOOTSTRAP: 95% percentile interval on the AUC from 500 draws that resample HELD-OUT LOANS with replacement (a loan contributes all its held-out rows or none),
numpy default_rng(50), the same 500 loan draws for every model and target. A draw is weighted AUC with each row weighted by its loan's multiplicity; draws with
no event (or no non-event) are dropped and counted. A constant score has AUC 0.5 and no interval width (no ranking exists). The bootstrap code is checked
against the plain AUC with unit weights before use.
"""

import gc
import glob
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join("diagnostics", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D49 = _load("diag49", "49_leaf_explosion_sweep.py")
D48 = D49.D48
D47 = D48.D47
STATES, BASE_FEATURES, CATEGORIES = D47.STATES, D47.BASE_FEATURES, D47.CATEGORIES
K = len(STATES)
IDX = {s: STATES.index(s) for s in STATES}
Tab, brier_by_class, counts, softmax = D48.Tab, D47.brier_by_class, D47.counts, D47.softmax
emit, title, GB = D47.emit, D47.title, D47.GB

SMOKE = os.environ.get("DIAG50_SMOKE") == "1"
D47.SMOKE = SMOKE
OUT = os.environ.get("DIAG50_OUT", os.path.join("diagnostics", "50_tail_ordering.txt"))
MODEL_DIR = os.environ.get("DIAG50_MODELS", os.path.join("artifacts", "diag49"))
REF49 = os.path.join("diagnostics", "49_leaf_explosion_sweep.txt")
ORIGIN = ["Current"]
EXPECT_HELD = [7737923, 67032, 543, 270, 64833, 19]
EXPECT_TRAIN = [30959138, 268335, 2289, 1159, 259277, 55]
SANITY_RUN, SANITY_BRIER, SANITY_AUC = 5, 0.0330520, 0.77469
N_BOOT = 10 if SMOKE else 500
BOOT_SEED = 50
CUTS = [0.01, 0.10]
TARGETS = [("DQ90+", "PRIMARY"), ("DQ60", "SECONDARY"), ("Default", "THIN")]
THIN_TAG = "TOO THIN TO MEASURE ORDERING (19 held-out events)"


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(D47.LINES) + "\n")


D47.write_report = write_report
D47.OUT = OUT


def parse_49():
    """Diagnostic 49's printed summary table: run -> (hess, l2, DQ30 AUC, leave AUC, Brier)."""
    out = {}
    pat = re.compile(r"^\s*(\d)\s+(\S+)\s+(\S+)\s+(0\.\d{5})\s+(0\.\d{5})\s+(0\.\d{7})\s+[+-]\d\.\d{7}\s")
    with open(REF49, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if m and int(m.group(1)) not in out:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return out


class Boot:
    """Loan-resampling bootstrap of the AUC for one (score, event) pair."""

    def __init__(self, score, event, loan_idx, n_loans):
        n = len(score)
        self.L = n_loans
        order = np.argsort(score, kind="stable")
        s = score[order]
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = s[1:] != s[:-1]
        starts = np.flatnonzero(change)
        self.constant = len(starts) == 1
        if self.constant:
            return
        ev_sorted = event[order]
        epos = np.flatnonzero(ev_sorted)
        blk = np.searchsorted(starts, epos, side="right") - 1
        ub, self.einv = np.unique(blk, return_inverse=True)
        ends = np.append(starts, n)
        self.start_b, self.end_b = starts[ub], ends[ub + 1]
        self.nub = len(ub)
        self.lsorted = loan_idx[order]
        self.eloan = self.lsorted[epos]
        del order, s, change, starts, ends, ev_sorted, epos, blk

    def auc(self, m):
        w = m[self.lsorted]
        S = np.cumsum(w, dtype=np.int64)
        sb = self.start_b
        before = np.where(sb > 0, S[np.maximum(sb - 1, 0)], 0)
        cnt = S[self.end_b - 1] - before
        ev = np.bincount(self.einv, weights=m[self.eloan], minlength=self.nub)
        P, W = ev.sum(), float(S[-1])
        N = W - P
        if P <= 0 or N <= 0:
            return float("nan")
        neg = cnt - ev
        negbelow = before - (np.cumsum(ev) - ev)
        return float((ev * (negbelow + 0.5 * neg)).sum() / (P * N))

    def interval(self, n_draw, seed):
        if self.constant:
            return 0.5, 0.5, 0, n_draw
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_draw):
            m = np.bincount(rng.integers(0, self.L, self.L), minlength=self.L).astype(np.int32)
            v = self.auc(m)
            if v == v:
                vals.append(v)
        if not vals:
            return float("nan"), float("nan"), 0, n_draw
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi), len(vals), n_draw - len(vals)


def fmt_ci(auc, ci):
    lo, hi, nv, nd = ci
    if lo == lo and lo == hi == 0.5 and auc == 0.5:
        return f"{auc:.5f} [no ranking]"
    return f"{auc:.5f} [{lo:.4f}, {hi:.4f}]" + (f" ({nd} draws dropped)" if nd else "")


def score_model(lgb, cap, spec, Xte, yte, loan_idx, n_loans):
    """Score one persisted model on the held-out frame. Returns the per-target statistics."""
    r = {"k": spec["run"], "h": spec["params"]["min_sum_hessian_in_leaf"], "l2": spec["params"]["lambda_l2"], "path": spec["path"]}
    feats = spec["features"]
    watch = D47.MemWatch(on_abort=D47.on_abort, abort_mb=700)
    watch.phase = f"run {r['k']}"
    watch.start()
    t0 = time.perf_counter()
    booster = lgb.Booster(model_file=spec["path"])
    assert booster.feature_name() == feats, (booster.feature_name(), feats)
    assert booster.pandas_categorical == [CATEGORIES["property_state"]]
    raw = np.asarray(booster.predict(Xte[feats], raw_score=True), dtype=np.float64)
    assert raw.shape == (len(yte), K)
    perm = [spec["class_order"].index(s) for s in STATES]            # sidecar class names -> panel.STATES order, never positional
    raw = raw[:, perm]
    del booster
    r["t_score"] = time.perf_counter() - t0
    yi = yte.astype(np.int64)
    P = softmax(raw)
    r["brier"] = float(brier_by_class(P, yi).sum())
    r["spread"] = raw.max(axis=0) - raw.min(axis=0)
    rawcols = {name: raw[:, IDX[name]].copy() for name, _ in TARGETS}
    del raw
    r["dq30"] = Tab(P[:, IDX["DQ30"]].copy(), yi == IDX["DQ30"]).auc()
    r["t"] = {}
    for name, _ in TARGETS:
        j = IDX[name]
        sc, ev = P[:, j].copy(), (yi == j)
        t = Tab(sc, ev)
        tr = Tab(rawcols[name], ev)
        d = {"auc_raw": tr.auc(), "distinct_raw": len(tr.u), "events": int(t.P), "auc": t.auc(), "distinct": len(t.u), "largest": float(t.cnt.max() / t.n), "spread": float(r["spread"][j]),
             "dec": t.deciles(), "tops": [t.top(q) for q in CUTS]}
        b = Boot(sc, ev, loan_idx, n_loans)
        if not b.constant and r["k"] == SANITY_RUN and name == "DQ90+":
            ones = np.ones(n_loans, dtype=np.int32)
            d["boot_check"] = (b.auc(ones), t.auc())
        d["ci"] = b.interval(N_BOOT, BOOT_SEED)
        r["t"][name] = d
        del t, tr, b, sc, ev
        gc.collect()
    del P, rawcols
    watch.stop()
    gc.collect()
    r["rss"], r["peak"], r["min_avail"] = watch._peak, watch.peak_rss(), watch.min_avail
    r["secs"] = time.perf_counter() - t0
    return r


def emit_detail(r, held):
    title(f"MODEL run {r['k']}: min_sum_hessian_in_leaf = {r['h']:g}, lambda_l2 = {r['l2']:g}   ({r['path']})")
    emit(f"COST: wall-clock {r['secs']:.1f} s (model load + held-out scoring {r['t_score']:.1f} s, rest statistics and bootstrap);  peak RSS in-run sampled "
         f"{r['rss'] / GB:.2f} GB, process peak {r['peak'] / GB:.2f} GB;  min system available {r['min_avail'] / GB:.2f} GB")
    emit(f"rescored: held-out Brier {r['brier']:.7f}; DQ30 AUC {r['dq30']:.5f} (diagnostic 49: Brier {r['ref'][4]:.7f}, DQ30 AUC {r['ref'][2]:.5f})")
    for name, role in TARGETS:
        d = r["t"][name]
        thin = name == "Default"
        emit()
        emit(f"--- {name} ({role})  events {d['events']:,} of {int(held.sum()):,} held-out rows" + (f"   *** {THIN_TAG} ***" if thin else ""))
        emit(f"  AUC {fmt_ci(d['auc'], d['ci'])}   (95% bootstrap interval over held-out loans, {N_BOOT} draws)")
        emit(f"  P({name}): {d['distinct']:,} distinct predicted values;  largest tie block {100 * d['largest']:.4f}% of rows")
        emit(f"  the class's OWN raw score: spread (max - min) {d['spread']:.6g}; {d['distinct_raw']:,} distinct values; AUC of the raw score {d['auc_raw']:.5f}")
        if d["spread"] == 0.0:
            emit(f"  *** RAW SPREAD EXACTLY 0: this model assigns every loan the identical raw {name} score, so its {name} column says NOTHING about {name} "
                 f"(raw-score AUC exactly 0.5, one distinct value: no class-specific ranking). P({name}) above still differs between loans only because the softmax "
                 f"normalises by the other five classes' scores; its AUC of {d['auc']:.5f} therefore measures how the OTHER classes order loans, not {name}-specific signal. ***")
        for q, (got, k, tie, strad) in zip(CUTS, d["tops"]):
            emit(f"  top {100 * q:.0f}% ({k:,} rows): {got:.2f} of {d['events']} {name} events = {100 * got / d['events']:.2f}% "
                 f"(random: {100 * q:.0f}%); rows tied at the cutoff {tie:,}{'*' if strad else ''}"
                 + ("  [Default: a count, not a rate; read with the interval above]" if thin else ""))
    emit()
    emit("DECILE TABLES (decile 1 = lowest predicted score; equal-count groups; tied blocks shared pro rata so counts can be fractional)")
    emit(f"  {'':>6}{'DQ90+ rate':>13}{'distinct':>11}{'DQ60 rate':>12}{'distinct':>11}{'Default events':>16}{'distinct':>11}")
    for i in range(10):
        a, b, c = (r["t"][n]["dec"][i] for n in ("DQ90+", "DQ60", "Default"))
        emit(f"  {i + 1:>6}{100 * a[1] / a[0]:>12.4f}%{a[3]:>11,}{100 * b[1] / b[0]:>11.4f}%{b[3]:>11,}{c[1]:>16.2f}{c[3]:>11,}")
    emit(f"  rows per decile {int(r['t']['DQ90+']['dec'][0][0]):,}; DQ90+ events per decile sum to {sum(x[1] for x in r['t']['DQ90+']['dec']):.1f}, DQ60 "
         f"{sum(x[1] for x in r['t']['DQ60']['dec']):.1f}, Default {sum(x[1] for x in r['t']['Default']['dec']):.1f}")


def main():
    import lightgbm as lgb

    cap = D47.Capture()
    lgb.register_logger(cap)
    print("loading the held-out frame ...", flush=True)
    specs = []
    for jp in sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))):
        with open(jp, encoding="utf-8") as f:
            s = json.load(f)
        s["path"] = jp[:-5] + ".txt"
        assert os.path.exists(s["path"]), s["path"]
        specs.append(s)
    specs.sort(key=lambda s: s["run"])
    assert [s["run"] for s in specs] == list(range(1, 10)), [s["run"] for s in specs]
    feats0 = specs[0]["features"]
    assert all(s["features"] == feats0 and s["origin_states"] == ORIGIN for s in specs)
    assert feats0 == [f for f in BASE_FEATURES if f not in D47.R2_DROP], feats0
    grid = [(h, l) for h in D49.HESS for l in D49.L2]
    assert [(s["params"]["min_sum_hessian_in_leaf"], s["params"]["lambda_l2"]) for s in specs] == grid, "sidecar levers differ from the 49 grid"
    ref49 = parse_49()
    assert sorted(ref49) == list(range(1, 10)), ref49

    X, y, fp, fs, codes, n_null, manifest, n_loans_all = D47.load_rows(ORIGIN, True)
    in_train, nl, n_tr = D47.split_flags(codes)
    ctr, cte = counts(y[in_train].astype(np.int64)), counts(y[~in_train].astype(np.int64))
    same = list(map(int, cte)) == EXPECT_HELD and list(map(int, ctr)) == EXPECT_TRAIN

    title("EVENT COUNTS  (before any ordering number)")
    emit(f"held-out DQ90+ events: {int(cte[IDX['DQ90+']]):,} (train {int(ctr[IDX['DQ90+']]):,})")
    emit(f"held-out DQ60 events : {int(cte[IDX['DQ60']]):,} (train {int(ctr[IDX['DQ60']]):,})")
    emit(f"held-out Default events: {int(cte[IDX['Default']]):,} (train {int(ctr[IDX['Default']]):,})   *** DEFAULT IS {THIN_TAG.upper()} ***")
    emit("Every Default figure below is a count, carries a bootstrap interval, and is marked as too thin. The aggregate Brier cannot see DQ90+ or Default: "
         "in diagnostic 49 run 5 the Default term was 0.0000024 of 0.0331.")
    emit()
    emit("DIAGNOSTIC 50: ordering on DQ90+, DQ60 and Default across the nine diagnostic-49 models (scoring only; measurement only; no selection, no recommendation)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs" + ("   SMOKE RUN (first partition only; 10 bootstrap draws)" if SMOKE else ""))
    title("SETUP")
    emit(f"panel build {manifest['build_stamp']}; window role == 'fit', {D47.WIN_LO} <= from_period <= {D47.WIN_HI}; from_state == Current; rows {len(y):,}; loans {n_loans_all:,}")
    emit(f"split by loan_id: seed {D47.SEED}, train fraction {D47.TRAIN_FRAC} -> {n_tr:,} of {nl:,} loans to train, {nl - n_tr:,} held out; no loan on both sides (asserted)")
    emit(f"  {'class':<10}{'HELD-OUT rows':>16}{'expected (48/49)':>18}{'TRAIN rows':>14}{'expected':>12}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(cte[j]):>16,}{EXPECT_HELD[j]:>18,}{int(ctr[j]):>14,}{EXPECT_TRAIN[j]:>12,}")
    emit(f"per-class counts vs diagnostic 48/49: {'IDENTICAL' if same else '!!! DIFFERENT !!!'}")
    if not same and not SMOKE:
        emit("STOP: the population or split does not reproduce diagnostic 48/49's. Nothing was scored.")
        write_report()
        sys.exit(1)
    te = ~in_train
    codes_te = codes[te]
    _, loan_idx = np.unique(codes_te, return_inverse=True)
    loan_idx = loan_idx.astype(np.int32)
    n_loans_te = int(loan_idx.max()) + 1
    Xte, yte = X.loc[te, feats0].reset_index(drop=True), y[te]
    del X, y, fp, fs, codes, codes_te, in_train
    gc.collect()
    emit(f"held-out frame built ONCE: {len(yte):,} rows, {n_loans_te:,} held-out loans (the bootstrap resamples these loans); features {feats0}; "
         f"sidecar class order {specs[0]['class_order']} {'== panel.STATES' if specs[0]['class_order'] == STATES else 'DIFFERS from panel.STATES (columns are re-mapped by name)'}")
    emit(f"models: nine files in {MODEL_DIR}; lever values read from the sidecars. Bootstrap: {N_BOOT} draws, loans resampled, seed {BOOT_SEED}.")
    write_report()

    results = []
    order = [SANITY_RUN] + [k for k in range(1, 10) if k != SANITY_RUN]
    for k in order:
        spec = specs[k - 1]
        print(f"  scoring run {k} ...", flush=True)
        r = score_model(lgb, cap, spec, Xte, yte, loan_idx, n_loans_te)
        r["ref"] = ref49[k]
        results.append(r)
        if k == SANITY_RUN:
            ok = abs(round(r["brier"], 7) - SANITY_BRIER) < 1.5e-7 and abs(round(r["dq30"], 5) - SANITY_AUC) < 1.5e-5
            title("SANITY CHECK: rescore run 5 against diagnostic 49")
            emit(f"run 5 (min_sum_hessian_in_leaf {r['h']:g}, lambda_l2 {r['l2']:g}): held-out Brier {r['brier']:.7f} vs {SANITY_BRIER:.7f}; DQ30 AUC {r['dq30']:.5f} vs {SANITY_AUC:.5f} -> "
                 + ("MATCH" if ok else "!!! MISMATCH !!!"))
            bc = r["t"]["DQ90+"].get("boot_check")
            if bc:
                emit(f"bootstrap code check (unit loan weights vs plain AUC, DQ90+): {bc[0]:.10f} vs {bc[1]:.10f} -> {'MATCH' if abs(bc[0] - bc[1]) < 1e-9 else '!!! MISMATCH !!!'}")
            if (not ok and not SMOKE) or (bc and abs(bc[0] - bc[1]) >= 1e-9):
                emit("STOP: the sanity checks did not pass. The remaining models are not scored.")
                write_report()
                sys.exit(1)
            title("RUN LOG  (one line per model, in scoring order)")
        d = r["t"]
        emit(f"run {k} ({r['h']:g}, {r['l2']:g}): DQ90+ AUC {fmt_ci(d['DQ90+']['auc'], d['DQ90+']['ci'])}; DQ60 AUC {d['DQ60']['auc']:.5f}; Default AUC {d['Default']['auc']:.5f} "
             f"(19 events, too thin); {r['secs']:.0f} s; process peak {r['peak'] / GB:.2f} GB; min system available {r['min_avail'] / GB:.2f} GB")
        write_report()
    results.sort(key=lambda r: r["k"])
    held = counts(yte.astype(np.int64))

    title("SUMMARY TABLE  (the deliverable; held-out rows, raw outputs; AUC with 95% bootstrap interval over held-out loans)")
    emit(f"{'run':>3}{'min_sum_hess':>13}{'lambda_l2':>10}{'DQ30 AUC':>10}{'DQ90+ AUC [95% CI]':>34}{'DQ60 AUC [95% CI]':>34}{'Default AUC [95% CI] *':>38}"
         f"{'DQ90+ spread':>14}{'DQ90+ raw AUC':>15}{'Default spread':>16}{'Default raw AUC':>17}{'Brier (49)':>12}  flag")
    for r in results:
        d = r["t"]
        flag = []
        if d["DQ90+"]["spread"] == 0.0:
            flag.append("DQ90+ RAW COLUMN CONSTANT (says nothing class-specific)")
        if d["Default"]["spread"] == 0.0:
            flag.append("DEFAULT RAW COLUMN CONSTANT (says nothing class-specific)")
        emit(f"{r['k']:>3}{r['h']:>13g}{r['l2']:>10g}{r['ref'][2]:>10.5f}{fmt_ci(d['DQ90+']['auc'], d['DQ90+']['ci']):>34}{fmt_ci(d['DQ60']['auc'], d['DQ60']['ci']):>34}"
             f"{fmt_ci(d['Default']['auc'], d['Default']['ci']):>38}{d['DQ90+']['spread']:>14.5g}{d['DQ90+']['auc_raw']:>15.5f}{d['Default']['spread']:>16.5g}{d['Default']['auc_raw']:>17.5f}{r['ref'][4]:>12.7f}  " + "; ".join(flag))
    emit(f"* Default: {int(held[IDX['Default']])} held-out events: {THIN_TAG}. DQ30 AUC and Brier are diagnostic 49's printed figures (rescored values agree: "
         f"max |DQ30 AUC difference| {max(abs(r['dq30'] - r['ref'][2]) for r in results):.2e}, max |Brier difference| {max(abs(r['brier'] - r['ref'][4]) for r in results):.2e}). "
         f"[no ranking] = the score is one constant for every loan. 'raw AUC' = AUC of the class's own raw score (0.5 when the raw column is constant); the bracketed AUCs are of P(class), "
         f"which also varies with the other classes' scores through the softmax normalisation.")
    for r in results:
        emit_detail(r, held)

    title("DOES DQ90+ ORDERING VARY WITH min_sum_hessian_in_leaf, lambda_l2, BOTH, OR NEITHER?  (descriptive, from the nine cells)")
    A = np.array([[results[3 * i + j]["t"]["DQ90+"]["auc"] for j in range(3)] for i in range(3)])
    W = np.array([[results[3 * i + j]["t"]["DQ90+"]["ci"][1] - results[3 * i + j]["t"]["DQ90+"]["ci"][0] for j in range(3)] for i in range(3)])
    emit("DQ90+ AUC by cell (rows = min_sum_hessian_in_leaf, columns = lambda_l2); constant-score cells show 0.5 with no interval")
    emit(f"  {'':<16}" + "".join(f"{'lambda_l2 ' + format(l, 'g'):>16}" for l in D49.L2) + f"{'row mean':>12}")
    for i, h in enumerate(D49.HESS):
        emit(f"  {'hess ' + format(h, 'g'):<16}" + "".join(f"{A[i, j]:>16.5f}" for j in range(3)) + f"{A[i].mean():>12.5f}")
    emit(f"  {'column mean':<16}" + "".join(f"{A[:, j].mean():>16.5f}" for j in range(3)))
    R = np.array([[results[3 * i + j]["t"]["DQ90+"]["auc_raw"] for j in range(3)] for i in range(3)])
    emit("DQ90+ AUC of the class's OWN raw score by cell (0.5 exactly = constant raw column)")
    emit(f"  {'':<16}" + "".join(f"{'lambda_l2 ' + format(l, 'g'):>16}" for l in D49.L2))
    for i, h in enumerate(D49.HESS):
        emit(f"  {'hess ' + format(h, 'g'):<16}" + "".join(f"{R[i, j]:>16.5f}" for j in range(3)))
    emit("95% interval width by cell: " + "; ".join(f"({D49.HESS[i]:g}, {D49.L2[j]:g}) {W[i, j]:.4f}" for i in range(3) for j in range(3)))
    spread_h, spread_l = A.mean(axis=1).max() - A.mean(axis=1).min(), A.mean(axis=0).max() - A.mean(axis=0).min()
    resid = A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()
    nonconst = W[W > 0]
    emit(f"spread (max - min) of the marginal means: across min_sum_hessian_in_leaf {spread_h:.5f}; across lambda_l2 {spread_l:.5f}; largest interaction residual |cell - row mean - column mean + grand mean| {np.abs(resid).max():.5f}; "
         f"range of the nine cells {A.max() - A.min():.5f}; mean interval width of the non-constant cells {nonconst.mean() if len(nonconst) else float('nan'):.4f}")
    for i, h in enumerate(D49.HESS):
        emit(f"  within min_sum_hessian_in_leaf = {h:g}: AUC range across lambda_l2 {A[i].max() - A[i].min():.5f}")
    for j, l in enumerate(D49.L2):
        emit(f"  within lambda_l2 = {l:g}: AUC range across min_sum_hessian_in_leaf {A[:, j].max() - A[:, j].min():.5f}")
    emit(f"raw-score DQ90+ AUC: spread of the marginal means across min_sum_hessian_in_leaf {R.mean(axis=1).max() - R.mean(axis=1).min():.5f}; across lambda_l2 "
         f"{R.mean(axis=0).max() - R.mean(axis=0).min():.5f}; range of the nine cells {R.max() - R.min():.5f}")
    const_cells = [(r["h"], r["l2"]) for r in results if r["t"]["DQ90+"]["spread"] == 0.0]
    constd = [(r["h"], r["l2"]) for r in results if r["t"]["Default"]["spread"] == 0.0]
    emit(f"cells whose DQ90+ raw spread is exactly 0 (constant score): {const_cells or 'none'}; whose Default raw spread is exactly 0: {constd or 'none'}")

    title("FINDINGS  (measured only; no interpretation, no recommendation)")
    emit(f"- Held-out events: DQ90+ {int(held[IDX['DQ90+']])} (train {int(ctr[IDX['DQ90+']]):,}); DQ60 {int(held[IDX['DQ60']])} (train {int(ctr[IDX['DQ60']]):,}); Default {int(held[IDX['Default']])} "
         f"(train {int(ctr[IDX['Default']])}): {THIN_TAG}.")
    emit(f"- Sanity: run 5 rescored to Brier {results[4]['brier']:.7f} and DQ30 AUC {results[4]['dq30']:.5f} (diagnostic 49: {SANITY_BRIER:.7f}, {SANITY_AUC:.5f}).")
    for r in results:
        d = r["t"]
        emit(f"- run {r['k']} (min_sum_hessian_in_leaf {r['h']:g}, lambda_l2 {r['l2']:g}): Brier {r['ref'][4]:.7f}, DQ30 AUC {r['ref'][2]:.5f}; DQ90+ AUC {fmt_ci(d['DQ90+']['auc'], d['DQ90+']['ci'])}, "
             f"own raw-score AUC {d['DQ90+']['auc_raw']:.5f}, raw spread {d['DQ90+']['spread']:.5g}, {d['DQ90+']['distinct']:,} distinct P values, top 1% holds {d['DQ90+']['tops'][0][0]:.1f} of {d['DQ90+']['events']} events, top 10% "
             f"{d['DQ90+']['tops'][1][0]:.1f}; DQ60 AUC {fmt_ci(d['DQ60']['auc'], d['DQ60']['ci'])}; Default AUC {fmt_ci(d['Default']['auc'], d['Default']['ci'])} "
             f"({d['Default']['events']} events, too thin), own raw-score AUC {d['Default']['auc_raw']:.5f}, raw spread {d['Default']['spread']:.5g}, top 10% holds {d['Default']['tops'][1][0]:.2f} of {d['Default']['events']}.")
    emit("- For a class whose raw column is constant, P(class) still varies between loans through the softmax normalisation by the other classes' scores; its AUC is then the other "
         "classes' ordering, and the class's own raw-score AUC is exactly 0.5. Both are listed per model above.")
    emit(f"- Models with a raw-score spread of exactly 0 (a constant class column: nothing class-specific): DQ90+ {[r['k'] for r in results if r['t']['DQ90+']['spread'] == 0.0] or 'none'}; "
         f"DQ60 {[r['k'] for r in results if r['t']['DQ60']['spread'] == 0.0] or 'none'}; Default {[r['k'] for r in results if r['t']['Default']['spread'] == 0.0] or 'none'}.")
    emit(f"- DQ90+ AUC across the nine cells: min {A.min():.5f}, max {A.max():.5f}; marginal-mean spread across min_sum_hessian_in_leaf {spread_h:.5f}, across lambda_l2 {spread_l:.5f}; "
         f"mean interval width of non-constant cells {nonconst.mean() if len(nonconst) else float('nan'):.4f}.")
    emit()
    emit(f"PEAK RSS (src.memwatch): process peak {max(r['peak'] for r in results) / GB:.2f} GB; lowest system-available memory seen "
         f"{min(r['min_avail'] for r in results) / GB:.2f} GB; total scoring wall-clock {sum(r['secs'] for r in results) / 60:.1f} min")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped (no rows were sampled): {e}")
        write_report()
        raise
