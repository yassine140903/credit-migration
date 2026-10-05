"""Milestones 2b/2c/2d: persistence, marginal, baseline, the ORACLE (not
achievable) marginal floor and the ORACLE BY START STATE (not achievable)
floor, through one harness pass, scored per start state as well as per cohort.

    python -m src.horizon_nulls                      # artifacts/horizon_nulls/
    python -m src.horizon_nulls --out-dir some/dir

Eligibility, realised outcomes and cohorts are computed ONCE and shared by all
five; every one is scored by the unmodified src.scoring.score. The baseline is
re-run here (load_baseline + the same chainer) and checked against the saved
artifacts/horizon_baseline/ run, so the comparison is like-for-like.

Both oracles are diagnostic yardsticks built from test-period outcomes, NOT
models: each carries its own label (ORACLE (not achievable), ORACLE BY START
STATE (not achievable)) and its own CSV columns in every table and CSV.

STRATIFIED REPORTING
    The live book at the cut is ~95% Current, so a cohort Brier is almost
    entirely a statement about Current loans. Every model is therefore also
    scored per (cohort, start state, horizon) cell, and the primary table is
    the baseline against the ORACLE BY START STATE floor per cell. Every
    aggregate is quoted beside the cohort's start-state mix.

CHECKS - beyond the like-for-like check; any failure stops the run before a
CSV is written, with the report at nulls_report_FAILED.txt
    floor         no model scores below ORACLE BY START STATE in any cell, and
                  each model's excess equals ||p - q||^2 (p its constant cell
                  prediction, q the cell's realised distribution)
    pooling       the mass-weighted mean of per-start-state Briers reproduces
                  the cohort Brier, this run and the saved model_comparison.csv,
                  within 1e-9 relative
    sums          every ORACLE BY START STATE cell distribution sums to 1 within
                  1e-9 relative
    cells         per-cell n, distribution and Brier against
                  reports/cell_counts.csv
    fingerprint   against the saved nulls_report.txt

OUTPUT
    nulls_report.txt       every check and table
    model_comparison.csv   horizon x cohort x start_state -> n, the cohort's n
                           and the cell's share of it, Brier for each model and
                           both oracles, baseline excess over each oracle, the
                           gap between the oracles. start_state == ALL rows are
                           the cohort aggregates.
    compounding_all.csv    horizon x cohort x model -> Default / Prepaid,
                           predicted vs realised, signed gap
"""

import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd

from src import baseline, horizon as hz, nulls, scoring
from src.horizon_baseline import (COHORTS, Report as _Report, fmt, pct,
                                  peak_memory_mb, OUT_DIR as BASELINE_RUN_DIR)

OUT_DIR = os.path.join("artifacts", "horizon_nulls")
REPORT = "nulls_report.txt"
REPORT_FAILED = "nulls_report_FAILED.txt"
COMPARISON_CSV = "model_comparison.csv"
CELL_COUNTS_CSV = os.path.join("reports", "cell_counts.csv")
ORACLE = nulls.ORACLE
BY_START = nulls.ORACLE_BY_START
ORACLES = (ORACLE, BY_START)
MODELS = ("persistence", "marginal", "baseline", ORACLE, BY_START)
LABEL = {"persistence": "persistence", "marginal": "marginal",
         "baseline": "baseline", ORACLE: nulls.ORACLE_LABEL,
         BY_START: nulls.ORACLE_BY_START_LABEL}
ORACLE_CSV = "ORACLE_not_achievable"
BY_START_CSV = "ORACLE_BY_START_not_achievable"
ALL = "ALL"
FULL_TABLE_HORIZONS = (6, 60)
SUMMARY_HORIZON = 60
MATCH_TOL = 1e-12
ORACLE_RTOL = 1e-9
SUM_RTOL = 1e-9
POOL_RTOL = 1e-9
IDENTITY_TOL = 1e-12
CELL_BRIER_TOL = 1e-12
RULE_W = 154

STATES = hz.STATES
I_DEF, I_PRE = STATES.index("Default"), STATES.index("Prepaid")
STRATA = [(STATES.index(s), s) for s in hz.TRANSIENT]


class Report(_Report):
    def rule(self, char="-", width=RULE_W):
        self.emit(char * width)


def fingerprint(ids, arrays):
    """sha256 over loan ids (sorted) and aligned integer arrays."""
    order = np.argsort(ids, kind="stable")
    h = hashlib.sha256()
    h.update("\n".join(np.asarray(ids)[order]).encode("ascii"))
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)[order]).tobytes())
    return h.hexdigest()[:16]


def saved_fingerprint(path=os.path.join(OUT_DIR, REPORT)):
    """The fingerprint recorded in a saved nulls_report.txt."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("fingerprint"))
    return lines[i + 1].strip()


def saved_aggregates(path=os.path.join(OUT_DIR, COMPARISON_CSV)):
    """The cohort rows of a saved model_comparison.csv, before or after it
    gained the start_state column."""
    df = pd.read_csv(path, float_precision="round_trip")
    if "start_state" in df:
        df = df[df["start_state"] == ALL]
    return df.set_index(["horizon", "cohort"])


def brier_col(m):
    return "brier_" + {ORACLE: ORACLE_CSV, BY_START: BY_START_CSV}.get(m, m)


def stop(rep, out_dir, message):
    """Write the report so far beside - never over - the saved report, whose
    fingerprint the next run checks against, and stop."""
    rep.emit()
    rep.emit("HARD ERROR: " + message)
    path = os.path.join(out_dir, REPORT_FAILED)
    rep.write(path)
    raise SystemExit("%s - see %s" % (message, path))


def compare_saved(results, emit, rule):
    """The re-run baseline against the saved artifacts/horizon_baseline run."""
    sc = pd.read_csv(os.path.join(BASELINE_RUN_DIR, "scores.csv"))
    cp = pd.read_csv(os.path.join(BASELINE_RUN_DIR, "compounding.csv"))
    emit("Against the saved baseline run (%s):" % BASELINE_RUN_DIR)
    rule()
    emit("%-5s%-9s%12s%12s%14s%14s%14s%14s%8s"
         % ("h", "cohort", "n saved", "n now", "Default real", "saved",
            "Prepaid real", "saved", ""))
    rule()
    ok_all = True
    for h in hz.HORIZONS:
        for c in COHORTS:
            r = results["baseline"][h, c]
            s = sc[(sc["horizon"] == h) & (sc["cohort"] == c)].iloc[0]
            k = cp[(cp["horizon"] == h) & (cp["cohort"] == c)].iloc[0]
            ok = (int(s["n"]) == r["n"]
                  and int(k["default_real_n"]) == r["realised"][I_DEF]
                  and int(k["prepaid_real_n"]) == r["realised"][I_PRE]
                  and abs(float(s["brier"]) - r["brier"]) <= MATCH_TOL
                  and abs(float(k["default_pred_n"]) - r["expected"][I_DEF])
                  <= 1e-6
                  and abs(float(k["prepaid_pred_n"]) - r["expected"][I_PRE])
                  <= 1e-6)
            ok_all &= ok
            emit("%-5d%-9s%12s%12s%14s%14s%14s%14s%8s"
                 % (h, c, fmt(s["n"]), fmt(r["n"]), fmt(r["realised"][I_DEF]),
                    fmt(k["default_real_n"]), fmt(r["realised"][I_PRE]),
                    fmt(k["prepaid_real_n"]), "PASS" if ok else "FAIL"))
    rule()
    emit("also compared per row: baseline Brier (|diff| <= %g) and predicted"
         % MATCH_TOL)
    emit("Default / Prepaid expected counts (|diff| <= 1e-6).")
    return ok_all


def mix_line(by_start, h, c):
    """The cohort's start-state mix, quoted beside every aggregate."""
    n = sum(by_start["baseline"][h, c, s]["n"] for _, s in STRATA)
    return "    start-state mix: " + "   ".join(
        "%s %s (%s)" % (s, fmt(by_start["baseline"][h, c, s]["n"]),
                        pct(by_start["baseline"][h, c, s]["n"], n, 2))
        for _, s in STRATA)


def six_state(emit, rule, r, title, mix=None):
    n = r["n"]
    emit(title)
    if mix:
        emit(mix)
    rule()
    emit("%-10s%18s%14s%16s%14s%14s" % ("state", "pred (exp. n)", "pred share",
                                        "realised n", "real share", "gap (pp)"))
    rule()
    for k, s in enumerate(STATES):
        e, a = r["expected"][k], r["realised"][k]
        emit("%-10s%18s%14s%16s%14s%+14.4f"
             % (s, "{:,.1f}".format(e), pct(e, n), fmt(a), pct(a, n),
                100.0 * (e - a) / n))
    rule()
    emit("%-10s%18s%14s%16s%14s" % ("TOTAL", "{:,.1f}".format(r["expected"].sum()),
                                    pct(r["expected"].sum(), n), fmt(n),
                                    pct(n, n)))
    emit("multiclass Brier score: %.6f   (n = %s loans)" % (r["brier"], fmt(n)))
    emit()


def comparison_row(h, c, s, n, n_cohort, b):
    """One model_comparison.csv row from {model: Brier} over one cell."""
    exc = b["baseline"] - b[ORACLE]
    exc_bs = b["baseline"] - b[BY_START]
    gap = b[ORACLE] - b[BY_START]
    return {"horizon": h, "cohort": c, "start_state": s, "n": n,
            "n_cohort": n_cohort, "share_of_cohort": n / n_cohort,
            "brier_persistence": b["persistence"],
            "brier_marginal": b["marginal"],
            "brier_baseline": b["baseline"],
            "brier_" + ORACLE_CSV: b[ORACLE],
            "baseline_excess_over_" + ORACLE_CSV: exc,
            "baseline_excess_pct_of_" + ORACLE_CSV: 100.0 * exc / b[ORACLE],
            "oracle_label": nulls.ORACLE_LABEL,
            "brier_" + BY_START_CSV: b[BY_START],
            "baseline_excess_over_" + BY_START_CSV: exc_bs,
            "baseline_excess_pct_of_" + BY_START_CSV:
                100.0 * exc_bs / b[BY_START],
            "oracle_gap_%s_minus_%s" % (ORACLE_CSV, BY_START_CSV): gap,
            "oracle_gap_pct_of_" + ORACLE_CSV: 100.0 * gap / b[ORACLE],
            "oracle_by_start_label": nulls.ORACLE_BY_START_LABEL}


# ==========================================================================
# stratification checks
# ==========================================================================
def check_floor(preds, real, masks, start, by_start, by_start_dists):
    """No model below ORACLE BY START STATE in any cell; excess == ||p - q||^2.

    Every model here is constant within a (cohort, start state, horizon) cell,
    and for a constant prediction p on a cell with realised distribution q,
    Brier(p) = Brier(q) + ||p - q||^2 exactly. A negative excess, or one that
    disagrees with ||p - q||^2, means the loans scored are not the loans the
    oracle was built from."""
    negative, mismatch, worst = [], [], {}
    for m in MODELS:
        if m == BY_START:
            continue
        for (h, c), mask in masks.items():
            pm = preds[m][h, c] if m in ORACLES else preds[m][h]
            for code, s in STRATA:
                cell = mask & (start == code)
                block = pm[cell]
                floor = by_start[BY_START][h, c, s]["brier"]
                exc = by_start[m][h, c, s]["brier"] - floor
                key = (c, s, h)
                if exc < 0:
                    negative.append((m, key, exc, by_start[m][h, c, s]["brier"],
                                     floor, int(cell.sum())))
                if not np.array_equal(block, np.broadcast_to(block[0],
                                                             block.shape)):
                    mismatch.append((m, key, "prediction not constant in cell"))
                    continue
                q = by_start_dists[h, c, s][0]
                dist2 = float(((block[0] - q) ** 2).sum())
                dev = abs(exc - dist2)
                if dev > IDENTITY_TOL:
                    mismatch.append((m, key, "excess %.17g vs ||p-q||^2 %.17g"
                                     % (exc, dist2)))
                key0, exc0, dev0 = worst.get(m, (key, exc, 0.0))
                worst[m] = ((key, exc) if exc < exc0 else (key0, exc0)) + \
                    (max(dev0, dev),)
    return negative, mismatch, worst


def check_pooling(results, by_start, saved):
    """Mass-weighted per-start-state Brier vs the cohort Brier, per model."""
    out = {}
    for m in MODELS:
        here, there, n_ok = 0.0, 0.0, True
        for h in hz.HORIZONS:
            for c in COHORTS:
                parts = [by_start[m][h, c, s] for _, s in STRATA]
                n_ok &= sum(p["n"] for p in parts) == results[m][h, c]["n"]
                pooled = scoring.pooled_brier(parts)
                agg = results[m][h, c]["brier"]
                here = max(here, abs(pooled - agg) / abs(agg))
                col = brier_col(m)
                if col in saved:
                    ref = float(saved.loc[(h, c), col])
                    there = max(there, abs(pooled - ref) / abs(ref))
                else:
                    there = None
        out[m] = (here, there, n_ok)
    return out


def check_cells(cells_saved, by_start, by_start_dists):
    """Per-cell n, distribution and Brier against reports/cell_counts.csv."""
    problems, max_p, max_b = [], 0.0, 0.0
    pcols = ["p_" + k for k in STATES]
    seen = set()
    for _, r in cells_saved.iterrows():
        key = (int(r.horizon), r.cohort, r.start_state)
        seen.add(key)
        if key not in by_start_dists:
            problems.append("cell (%s, %s, h=%d) in %s, not here"
                            % (r.cohort, r.start_state, r.horizon,
                               CELL_COUNTS_CSV))
            continue
        dist, counts = by_start_dists[key]
        if int(counts.sum()) != int(r.n):
            problems.append("n: (%s, %s, h=%d) here %s, %s %s"
                            % (r.cohort, r.start_state, r.horizon,
                               fmt(counts.sum()), CELL_COUNTS_CSV, fmt(r.n)))
        max_p = max(max_p, float(np.abs(dist - r[pcols].to_numpy(float)).max()))
        b = by_start[BY_START][key]["brier"]
        max_b = max(max_b, abs(b - float(r.cell_brier)))
    for key in sorted(set(by_start_dists) - seen):
        problems.append("cell (%s, %s, h=%d) here, not in %s"
                        % (key[1], key[2], key[0], CELL_COUNTS_CSV))
    if max_p > 0:
        problems.append("distributions differ from %s p_* (max |diff| %.3e)"
                        % (CELL_COUNTS_CSV, max_p))
    if max_b > CELL_BRIER_TOL:
        problems.append("cell Brier differs from %s cell_brier (max |diff| "
                        "%.3e)" % (CELL_COUNTS_CSV, max_b))
    return problems, max_p, max_b


def run(out_dir=OUT_DIR):
    t0 = time.time()
    rep = Report()
    emit, rule = rep.emit, rep.rule

    # ---- saved references, read before anything is written ----------------------
    fp_saved = saved_fingerprint()
    saved_agg = saved_aggregates()
    cells_saved = pd.read_csv(CELL_COUNTS_CSV, float_precision="round_trip")

    # ---- shared inputs, computed once -----------------------------------------
    L, stamp = hz.read_test_loans(progress=lambda s: print(s, flush=True))
    reason = hz.eligibility(L).to_numpy()
    E = L[reason < 0]
    start = E["state_cut"].to_numpy()
    real = hz.realised(E)
    varying, fixed = hz.cohorts(real)
    n = len(E)
    masks = {(h, c): (varying[h] if c == "varying" else fixed)
             for h in hz.HORIZONS for c in COHORTS}

    # ---- the three models and the two oracles -----------------------------------
    base_counts, P_df = baseline.load_baseline()
    P = P_df.to_numpy()
    fit_next, n_fit = nulls.fit_next_state_counts()
    marg = nulls.marginal_distribution(fit_next)
    col_sums = base_counts.to_numpy().sum(axis=0)

    preds = {"persistence": nulls.persistence_predictions(start),
             "marginal": nulls.marginal_predictions(marg, n),
             "baseline": hz.chain(start, hz.homogeneous_step(P))}
    preds[ORACLE], oracle_dists = nulls.oracle_predictions(real, masks)
    preds[BY_START], by_start_dists = nulls.oracle_by_start_predictions(
        real, masks, start)
    hz.assert_same_by_start(preds["persistence"], start)
    hz.assert_same_by_start(preds["baseline"], start)
    persist_exact = all(np.array_equal(preds["persistence"][h], hz.one_hot(start))
                        for h in hz.HORIZONS)

    results = {m: {} for m in MODELS}
    by_start = {m: {} for m in MODELS}
    for m in MODELS:
        for (h, c), mask in masks.items():
            pm = preds[m][h, c] if m in ORACLES else preds[m][h]
            results[m][h, c] = scoring.score(pm, real[h], mask)
            for s, r in scoring.stratify(pm, real[h], mask, start,
                                         STRATA).items():
                by_start[m][h, c, s] = r
    # each oracle's aggregate prediction must reproduce the realised counts it
    # was built from - per cohort, and for ORACLE BY START STATE per cell too.
    # Relative tolerance: the expected count is a float sum over ~400k
    # identical rows and carries summation error.
    oracle_exact = all(np.allclose(results[m][hc]["expected"],
                                   results[m][hc]["realised"],
                                   rtol=ORACLE_RTOL, atol=0)
                       for m in ORACLES for hc in masks)
    oracle_exact &= all(np.allclose(r["expected"], r["realised"],
                                    rtol=ORACLE_RTOL, atol=0)
                        for r in by_start[BY_START].values())

    # ---- header ------------------------------------------------------------------
    rule("=")
    emit("HORIZON NULLS - persistence, marginal, baseline, %s"
         % nulls.ORACLE_LABEL)
    emit("and %s, per cohort and per start state"
         % nulls.ORACLE_BY_START_LABEL)
    rule("=")
    emit("panel        : data/panel/panel_*.parquet via src.panel_io (build %s)"
         % stamp)
    emit("cut          : %s" % hz.ym(hz.CUT))
    emit("horizons     : %s" % ", ".join("h=%d -> %s" % (h, hz.ym(hz.CUT + h))
                                         for h in hz.HORIZONS))
    emit("scorer       : src.scoring.score, unmodified, for all five; per start")
    emit("               state through src.scoring.stratify (the same score on")
    emit("               each start state's loans)")
    emit()
    emit("MODEL DEFINITIONS")
    rule()
    emit("persistence  identity matrix chained by src.horizon.chain /")
    emit("             homogeneous_step - the baseline's own path. Every loan keeps")
    emit("             its state at the cut with probability 1.0 at every horizon.")
    emit("marginal     the empirical distribution of next_state across ALL fit rows")
    emit("             (role == fit), applied to every loan regardless of its state")
    emit("             at the cut, identical at every horizon; not chained.")
    emit("baseline     the pooled matrix from src.baseline.load_baseline(), chained.")
    emit(nulls.ORACLE_LABEL)
    emit("             NOT A MODEL - a diagnostic floor built from TEST-PERIOD")
    emit("             OUTCOMES; it could not be built at forecast time. For each")
    emit("             horizon AND each cohort separately: the realised state")
    emit("             distribution across that cohort, given identically to every")
    emit("             loan in it. Not chained; ignores the state at the cut.")
    emit("             Unlike the marginal null (ONE fit-window distribution for")
    emit("             every horizon and cohort), it DIFFERS per horizon and per")
    emit("             cohort.")
    emit(nulls.ORACLE_BY_START_LABEL)
    emit("             NOT A MODEL - the same construction as %s,"
         % nulls.ORACLE_LABEL)
    emit("             conditioned on the state at the cut as well: for each")
    emit("             (cohort, start state, horizon) cell, the realised state")
    emit("             distribution across that cell, given identically to every")
    emit("             loan in it. Each cell's own empirical distribution, so")
    emit("             Brier-optimal on that cell: the floor for any model that")
    emit("             uses the start state and nothing else. Both oracles are")
    emit("             kept; %s minus this one is what" % nulls.ORACLE_LABEL)
    emit("             the start state alone buys.")
    rule()
    emit()
    emit("Marginal distribution (unit: fit rows, next_state):")
    rule()
    emit("%-10s%18s%14s%22s" % ("state", "fit rows", "share", "baseline col. sum"))
    rule()
    for k, s in enumerate(STATES):
        emit("%-10s%18s%14s%22s" % (s, fmt(fit_next[k]), pct(fit_next[k], n_fit),
                                    fmt(col_sums[k])))
    rule()
    emit("%-10s%18s%14s%22s" % ("TOTAL", fmt(fit_next.sum()),
                                pct(fit_next.sum(), n_fit), fmt(col_sums.sum())))
    emit("marginal sums to 1.0 within %g: PASS   (sum = %.17f)"
         % (nulls.TOL, marg.sum()))
    same_cols = np.array_equal(fit_next, col_sums)
    emit("equals the column sums of artifacts/baseline/counts.csv: %s"
         % ("PASS" if same_cols else "FAIL"))
    emit("persistence predictions equal one-hot(state at cut) at every h: %s"
         % ("PASS" if persist_exact else "FAIL"))
    emit("every predicted distribution sums to 1 within %g (chain) and %g"
         % (hz.TOL, scoring.TOL))
    emit("(scorer) for all five, in every cohort and every start-state cell: PASS")
    emit()
    emit("%s distributions (unit: loans in the cohort; realised states):"
         % nulls.ORACLE_LABEL)
    rule()
    emit("%-5s%-9s%11s" % ("h", "cohort", "n") + "".join("%14s" % s for s in STATES)
         + "%8s" % "sum=1")
    rule()
    for h in hz.HORIZONS:
        for c in COHORTS:
            dist, cnt = oracle_dists[h, c]
            emit("%-5d%-9s%11s" % (h, c, fmt(cnt.sum()))
                 + "".join("%14s" % fmt(v) for v in cnt)
                 + "%8s" % ("PASS" if abs(dist.sum() - 1.0) <= nulls.TOL
                            else "FAIL"))
            emit("%-25s" % "" + "".join("%14s" % pct(v, cnt.sum()) for v in cnt))
    rule()
    emit("%s and %s aggregate predictions" % (nulls.ORACLE_LABEL,
                                              nulls.ORACLE_BY_START_LABEL))
    emit("reproduce the realised counts they were built from, per cohort and (the")
    emit("latter) per start-state cell (relative tolerance %g): %s"
         % (ORACLE_RTOL, "PASS" if oracle_exact else "FAIL"))
    emit("%s cell distributions: %s, columns p_*." % (nulls.ORACLE_BY_START_LABEL,
                                                    CELL_COUNTS_CSV))
    emit()

    # ---- 1 like-for-like -----------------------------------------------------------
    rule("=")
    emit("1. LIKE-FOR-LIKE CHECK   (unit: loans)")
    rule("=")
    emit("One eligibility, outcome and cohort computation is shared by all three")
    emit("models and both oracles. The counts it produced:")
    rule()
    emit("%-40s%16s" % ("test-split loans in the panel", fmt(len(L))))
    for k, why in enumerate(hz.EXCLUDED):
        emit("%-40s%16s" % ("  excluded: " + why, fmt((reason == k).sum())))
    emit("%-40s%16s" % ("  eligible", fmt(n)))
    for s in hz.TRANSIENT:
        c = int((start == STATES.index(s)).sum())
        emit("%-40s%16s%12s" % ("    start " + s, fmt(c), pct(c, n)))
    rule()
    emit("%-6s%14s%14s%14s%14s" % ("h", "observed", "censored", "varying",
                                   "fixed"))
    rule()
    for h in hz.HORIZONS:
        obs = int((real[h] != hz.CENSORED).sum())
        emit("%-6d%14s%14s%14s%14s" % (h, fmt(obs), fmt(n - obs),
                                       fmt(varying[h].sum()), fmt(fixed.sum())))
    rule()
    emit("Realised state counts per horizon x cohort:")
    rule()
    emit("%-5s%-9s" % ("h", "cohort") + "".join("%15s" % s for s in STATES)
         + "%15s" % "n")
    rule()
    for h in hz.HORIZONS:
        for c in COHORTS:
            r = results["baseline"][h, c]
            emit("%-5d%-9s" % (h, c) + "".join("%15s" % fmt(v) for v in r["realised"])
                 + "%15s" % fmt(r["n"]))
            emit(mix_line(by_start, h, c))
    rule()
    same = all(results[m][hc]["n"] == results["baseline"][hc]["n"]
               and np.array_equal(results[m][hc]["realised"],
                                  results["baseline"][hc]["realised"])
               for m in MODELS for hc in masks)
    same &= all(by_start[m][k]["n"] == by_start["baseline"][k]["n"]
                and np.array_equal(by_start[m][k]["realised"],
                                   by_start["baseline"][k]["realised"])
                for m in MODELS for k in by_start["baseline"])
    emit("all five scored on identical n and realised counts, per cohort and per")
    emit("start-state cell: %s" % ("PASS" if same else "FAIL"))
    fp = fingerprint(E.index.to_numpy(), [start] + [real[h] for h in hz.HORIZONS]
                     + [fixed])
    emit("fingerprint (eligible ids, start state, realised per h, fixed cohort):")
    emit("    %s" % fp)
    fp_ok = fp == fp_saved
    emit("against the saved nulls run (%s): %s   %s"
         % (os.path.join(OUT_DIR, REPORT), fp_saved, "PASS" if fp_ok else "FAIL"))
    emit()
    saved_ok = compare_saved(results, emit, rule)
    emit("re-run baseline matches the saved baseline run: %s"
         % ("PASS" if saved_ok else "FAIL"))
    emit()
    if not (same and saved_ok and same_cols and persist_exact and oracle_exact):
        stop(rep, out_dir, "like-for-like check failed")
    if not fp_ok:
        stop(rep, out_dir, "data fingerprint %s differs from the saved nulls run "
             "%s" % (fp, fp_saved))

    # ---- 1b stratification checks ----------------------------------------------------
    rule("=")
    emit("1b. STRATIFICATION CHECKS   (unit: loans in the cell)")
    rule("=")
    cell_problems, max_p, max_b = check_cells(cells_saved, by_start,
                                              by_start_dists)
    emit("cells       : %d = %d cohorts x %d start states x %d horizons"
         % (len(by_start_dists), len(COHORTS), len(STRATA), len(hz.HORIZONS)))
    emit("vs %s: n per cell %s; %s distribution max |diff|"
         % (CELL_COUNTS_CSV, "PASS" if not any(p.startswith(("n:", "cell"))
                                               for p in cell_problems)
            else "FAIL", nulls.ORACLE_BY_START_LABEL))
    emit("    %.3e vs p_*; its cell Brier max |diff| %.3e vs cell_brier (tol %g):"
         " %s" % (max_p, max_b, CELL_BRIER_TOL,
                  "PASS" if not cell_problems else "FAIL"))
    for p in cell_problems:
        emit("    DISCREPANCY  " + p)
    sum_dev = max(abs(d.sum() - 1.0) for d, _ in by_start_dists.values())
    sums_ok = sum_dev <= SUM_RTOL
    emit("%s cell distributions sum to 1 within %g (relative),"
         % (nulls.ORACLE_BY_START_LABEL, SUM_RTOL))
    emit("    all %d cells: %s   (max |sum - 1| %.3e)"
         % (len(by_start_dists), "PASS" if sums_ok else "FAIL", sum_dev))
    emit()
    pool = check_pooling(results, by_start, saved_agg)
    pool_ok = True
    emit("Pooling: mass-weighted mean of the per-start-state Briers vs the cohort")
    emit("Brier, max relative |diff| over horizon x cohort (tolerance %g):"
         % POOL_RTOL)
    rule()
    emit("%-40s%26s%34s%14s%8s" % ("model", "vs this run's cohort Brier",
                                   "vs saved model_comparison.csv",
                                   "n sums", ""))
    rule()
    for m in MODELS:
        here, there, n_ok = pool[m]
        ok = n_ok and here <= POOL_RTOL and (there is None or there <= POOL_RTOL)
        pool_ok &= ok
        emit("%-40s%26.3e%34s%14s%8s"
             % (LABEL[m], here, "%.3e" % there if there is not None
                else "n/a (new)", "PASS" if n_ok else "FAIL",
                "PASS" if ok else "FAIL"))
    rule()
    emit()
    negative, mismatch, worst = check_floor(preds, real, masks, start, by_start,
                                            by_start_dists)
    emit("Floor: Brier(model) - Brier(%s) per cell; the" % nulls.ORACLE_BY_START_LABEL)
    emit("latter is each cell's own empirical distribution, so for a prediction p")
    emit("constant on the cell the excess is exactly ||p - q||^2 >= 0.")
    rule()
    emit("%-40s%28s%18s%22s%14s%8s"
         % ("model", "smallest excess at", "smallest excess",
            "max |exc - ||p-q||^2|", "cells < 0", ""))
    rule()
    for m in MODELS[:-1]:
        key, exc, dev = worst[m]
        k_neg = sum(1 for x in negative if x[0] == m)
        k_mis = sum(1 for x in mismatch if x[0] == m)
        emit("%-40s%28s%+18.3e%22.3e%14d%8s"
             % (LABEL[m], "(%s, %s, h=%d)" % key, exc, dev, k_neg,
                "PASS" if not (k_neg or k_mis) else "FAIL"))
    rule()
    emit("baseline never beats %s in any cell: %s"
         % (nulls.ORACLE_BY_START_LABEL,
            "PASS" if not any(x[0] == "baseline" for x in negative) else "FAIL"))
    for m, key, exc, b, floor, k in negative:
        emit("    NEGATIVE EXCESS  %s at (%s, %s, h=%d), n = %s: Brier %.17g, "
             "floor %.17g, excess %.3e" % ((LABEL[m],) + key + (fmt(k), b, floor,
                                                                exc)))
    for m, key, why in mismatch:
        emit("    IDENTITY  %s at (%s, %s, h=%d): %s" % ((LABEL[m],) + key + (why,)))
    emit()
    if negative:
        stop(rep, out_dir, "a model scores below %s in %d cell(s), first %s at "
             "(%s, %s, h=%d) - the scored loans and the oracle's loans are "
             "misaligned" % ((nulls.ORACLE_BY_START_LABEL, len(negative),
                              LABEL[negative[0][0]]) + negative[0][1]))
    if mismatch or cell_problems or not (sums_ok and pool_ok):
        stop(rep, out_dir, "stratification check failed")

    # ---- the comparison rows: cohort aggregates and per start state --------------
    rows = []
    for h in hz.HORIZONS:
        for c in COHORTS:
            k_c = results["baseline"][h, c]["n"]
            rows.append(comparison_row(h, c, ALL, k_c, k_c,
                                       {m: results[m][h, c]["brier"]
                                        for m in MODELS}))
            for _, s in STRATA:
                rows.append(comparison_row(h, c, s, by_start["baseline"][h, c, s]
                                           ["n"], k_c,
                                           {m: by_start[m][h, c, s]["brier"]
                                            for m in MODELS}))
    cmp_df = pd.DataFrame(rows)
    agg = cmp_df[cmp_df["start_state"] == ALL]
    cell = cmp_df[cmp_df["start_state"] != ALL]
    EXC_BS = "baseline_excess_over_" + BY_START_CSV
    PCT_BS = "baseline_excess_pct_of_" + BY_START_CSV
    GAP = "oracle_gap_%s_minus_%s" % (ORACLE_CSV, BY_START_CSV)
    GAP_PCT = "oracle_gap_pct_of_" + ORACLE_CSV
    if (cmp_df[GAP] < 0).any():
        stop(rep, out_dir, "%s scores below %s somewhere"
             % (nulls.ORACLE_LABEL, nulls.ORACLE_BY_START_LABEL))

    # ---- 2 primary: per start state ----------------------------------------------
    rule("=")
    emit("2. PRIMARY VIEW - BASELINE VS %s PER (COHORT, START STATE, HORIZON)"
         % nulls.ORACLE_BY_START_LABEL)
    emit("   (unit: loans in the cell)")
    rule("=")
    emit("floor = %s Brier: the cell's realised" % nulls.ORACLE_BY_START_LABEL)
    emit("distribution given to every loan in it - built from test-period outcomes,")
    emit("not achievable. excess = baseline - floor; % of floor = excess / floor x")
    emit("100; share of agg. excess = (cell share x cell excess) / cohort excess, the")
    emit("cell's contribution to the cohort-level excess (sums to 100% per cohort")
    emit("and horizon).")
    rule()
    emit("%-9s%-8s%4s%10s%10s%14s%48s%14s%12s%20s"
         % ("cohort", "start", "h", "n", "of cohort", "baseline",
            "floor: " + nulls.ORACLE_BY_START_LABEL, "excess", "% of floor",
            "share of agg. exc."))
    rule()
    agg_exc = {(r.horizon, r.cohort): r[EXC_BS] for _, r in agg.iterrows()}
    for c in COHORTS:
        for _, s in STRATA:
            for _, r in cell[(cell.cohort == c) & (cell.start_state == s)].iterrows():
                emit("%-9s%-8s%4d%10s%10s%14.6f%48.6f%+14.6f%11.2f%%%19.2f%%"
                     % (c, s, r.horizon, fmt(r.n), pct(r.n, r.n_cohort, 2),
                        r.brier_baseline, r["brier_" + BY_START_CSV], r[EXC_BS],
                        r[PCT_BS], 100.0 * r.share_of_cohort * r[EXC_BS]
                        / agg_exc[r.horizon, c]))
            rule()
    for _, r in agg.iterrows():
        if r.cohort == COHORTS[0] and r.horizon == hz.HORIZONS[0]:
            emit("cohort aggregates (mass-weighted over the four start states):")
            rule()
        emit("%-9s%-8s%4d%10s%10s%14.6f%48.6f%+14.6f%11.2f%%%19.2f%%"
             % (r.cohort, ALL, r.horizon, fmt(r.n), pct(r.n, r.n_cohort, 2),
                r.brier_baseline, r["brier_" + BY_START_CSV], r[EXC_BS],
                r[PCT_BS], 100.0))
    rule()
    emit()

    # ---- 3 the two oracles -------------------------------------------------------
    rule("=")
    emit("3. THE TWO ORACLES - WHAT THE START STATE ALONE BUYS   (unit: loans in the "
         "cohort)")
    rule("=")
    emit("gap = %s Brier - %s Brier." % (nulls.ORACLE_LABEL,
                                         nulls.ORACLE_BY_START_LABEL))
    emit("Both are built from test-period outcomes. The gap is the mass-weighted")
    emit("mean over start states of ||q_cohort - q_cell||^2 - the Brier bought by")
    emit("knowing the state at the cut and nothing else. baseline excess = baseline -")
    emit("%s." % nulls.ORACLE_BY_START_LABEL)
    rule()
    emit("%-5s%-9s%12s%27s%40s%14s%14s%14s%18s"
         % ("h", "cohort", "n", nulls.ORACLE_LABEL, nulls.ORACLE_BY_START_LABEL,
            "gap", "% of ORACLE", "baseline", "baseline excess"))
    rule()
    for _, r in agg.iterrows():
        emit("%-5d%-9s%12s%27.6f%40.6f%+14.6f%13.2f%%%14.6f%+18.6f"
             % (r.horizon, r.cohort, fmt(r.n), r["brier_" + ORACLE_CSV],
                r["brier_" + BY_START_CSV], r[GAP], r[GAP_PCT],
                r.brier_baseline, r[EXC_BS]))
        emit(mix_line(by_start, r.horizon, r.cohort))
        if r.cohort == COHORTS[-1]:
            rule()
    emit("gap >= 0 in every cohort and cell: PASS")
    emit()

    # ---- 4 every model per start state -------------------------------------------
    rule("=")
    emit("4. MODEL COMPARISON PER START STATE - multiclass Brier score   (unit: "
         "loans in the cell)")
    rule("=")
    emit("Both oracle columns are floors built from test-period outcomes, not")
    emit("competing models. %s here is the COHORT-wide" % nulls.ORACLE_LABEL)
    emit("distribution scored on the cell's loans.")
    rule()
    emit("%-9s%-8s%4s%10s%10s" % ("cohort", "start", "h", "n", "of cohort")
         + "".join("%14s" % LABEL[m] for m in MODELS[:3])
         + "%27s%40s" % (LABEL[ORACLE], LABEL[BY_START]))
    rule()
    for c in COHORTS:
        for _, s in STRATA:
            for _, r in cell[(cell.cohort == c) & (cell.start_state == s)].iterrows():
                emit("%-9s%-8s%4d%10s%10s" % (c, s, r.horizon, fmt(r.n),
                                              pct(r.n, r.n_cohort, 2))
                     + "".join("%14.6f" % r[brier_col(m)] for m in MODELS[:3])
                     + "%27.6f%40.6f" % (r[brier_col(ORACLE)],
                                         r[brier_col(BY_START)]))
            rule()
    emit()

    # ---- 5 Brier comparison, cohort aggregates -----------------------------------------
    rule("=")
    emit("5. MODEL COMPARISON - multiclass Brier score, cohort aggregates   (unit: "
         "loans in the cohort)")
    rule("=")
    emit("The last two columns are %s and %s:" % (nulls.ORACLE_LABEL,
                                                 nulls.ORACLE_BY_START_LABEL))
    emit("floors built from test-period outcomes, not competing models. Each")
    emit("aggregate is ~95% Current loans - see the start-state mix beneath it.")
    rule()
    emit("%-5s%-9s%12s" % ("h", "cohort", "n")
         + "".join("%14s" % LABEL[m] for m in MODELS[:3])
         + "%27s%40s" % (LABEL[ORACLE], LABEL[BY_START]))
    rule()
    for _, r in agg.iterrows():
        emit("%-5d%-9s%12s" % (r.horizon, r.cohort, fmt(r.n))
             + "".join("%14.6f" % r[brier_col(m)] for m in MODELS[:3])
             + "%27.6f%40.6f" % (r[brier_col(ORACLE)], r[brier_col(BY_START)]))
        emit(mix_line(by_start, r.horizon, r.cohort))
        if r.cohort == COHORTS[-1]:
            rule()
    os.makedirs(out_dir, exist_ok=True)
    cmp_df.to_csv(os.path.join(out_dir, COMPARISON_CSV), index=False,
                  float_format="%.17g")
    emit()

    # ---- 6 excess over the oracle floor ----------------------------------------------
    rule("=")
    emit("6. BASELINE EXCESS OVER THE ORACLE FLOOR, cohort aggregates   (unit: loans "
         "in the cohort)")
    rule("=")
    emit("floor = %s Brier: the realised cohort distribution given"
         % nulls.ORACLE_LABEL)
    emit("to every loan, built from test-period outcomes - not achievable at")
    emit("forecast time. excess = baseline Brier - floor; % of floor = excess /")
    emit("floor x 100. The per-start-state floor is in sections 2 and 3.")
    rule()
    emit("%-5s%-9s%12s%34s%14s%14s%14s"
         % ("h", "cohort", "n", "floor: " + LABEL[ORACLE], "baseline", "excess",
            "% of floor"))
    rule()
    for _, row in agg.iterrows():
        emit("%-5d%-9s%12s%34.6f%14.6f%+14.6f%13.2f%%"
             % (row["horizon"], row["cohort"], fmt(row["n"]),
                row["brier_" + ORACLE_CSV], row["brier_baseline"],
                row["baseline_excess_over_" + ORACLE_CSV],
                row["baseline_excess_pct_of_" + ORACLE_CSV]))
        emit(mix_line(by_start, row["horizon"], row["cohort"]))
        if row["cohort"] == COHORTS[-1]:
            rule()
    emit()

    # ---- 7 compounding ---------------------------------------------------------------
    rule("=")
    emit("7. DEFAULT AND PREPAID, PREDICTED VS REALISED - THREE MODELS AND BOTH")
    emit("   ORACLES, cohort aggregates")
    rule("=")
    emit("pred = expected loans (sum of predicted probabilities) and share of the")
    emit("cohort; real = realised loans and share; gap = pred share - real share,")
    emit("in percentage points.")
    rule()
    emit("%-4s%-9s%-40s%9s %-19s%-17s%8s %-20s%-18s%8s"
         % ("h", "cohort", "model", "n", "Default pred", "Default real", "gap pp",
            "Prepaid pred", "Prepaid real", "gap pp"))
    rule()
    comp = []
    for h in hz.HORIZONS:
        for c in COHORTS:
            for m in MODELS:
                r = results[m][h, c]
                k = r["n"]
                row = {"horizon": h, "cohort": c, "model": LABEL[m], "n": k}
                cells = []
                for name, i in (("default", I_DEF), ("prepaid", I_PRE)):
                    e, a = r["expected"][i], int(r["realised"][i])
                    gap = 100.0 * (e - a) / k
                    row.update({name + "_pred_n": e, name + "_pred_share": e / k,
                                name + "_real_n": a, name + "_real_share": a / k,
                                name + "_gap_pp": gap})
                    cells += ["{:,.1f} ({})".format(e, pct(e, k, 2)),
                              "{} ({})".format(fmt(a), pct(a, k, 2)),
                              "%+.2f" % gap]
                comp.append(row)
                emit("%-4d%-9s%-40s%9s %-19s%-17s%8s %-20s%-18s%8s"
                     % tuple([h, c, LABEL[m], fmt(k)] + cells))
            emit(mix_line(by_start, h, c))
        rule()
    pd.DataFrame(comp).to_csv(os.path.join(out_dir, "compounding_all.csv"),
                              index=False, float_format="%.17g")
    emit()

    # ---- 8 six-state tables for the nulls and the oracle -----------------------------
    rule("=")
    emit("8. SIX-STATE PREDICTED VS REALISED, NULL MODELS AND %s,"
         % nulls.ORACLE_LABEL)
    emit("   h = %s, cohort aggregates" % " and ".join(map(str, FULL_TABLE_HORIZONS)))
    rule("=")
    for m in ("persistence", "marginal", ORACLE):
        for h in FULL_TABLE_HORIZONS:
            for c in COHORTS:
                six_state(emit, rule, results[m][h, c],
                          "%s - h = %d (%s), %s cohort"
                          % (LABEL[m], h, hz.ym(hz.CUT + h), c),
                          mix_line(by_start, h, c))

    # ---- summary -----------------------------------------------------------------------
    rule("=")
    emit("SUMMARY - h = %d, baseline excess over %s"
         % (SUMMARY_HORIZON, nulls.ORACLE_BY_START_LABEL))
    rule("=")
    for c in COHORTS:
        sub = cell[(cell.horizon == SUMMARY_HORIZON) & (cell.cohort == c)]
        top = sub.loc[sub[EXC_BS].idxmax()]
        top_pct = sub.loc[sub[PCT_BS].idxmax()]
        a = agg[(agg.horizon == SUMMARY_HORIZON) & (agg.cohort == c)].iloc[0]
        emit("%s cohort: largest excess in %s, %+.6f (%.2f%% of its floor %.6f;"
             % (c, top.start_state, top[EXC_BS], top[PCT_BS],
                top["brier_" + BY_START_CSV]))
        emit("    n = %s, %s of the cohort, %.2f%% of the cohort-level excess)."
             % (fmt(top.n), pct(top.n, top.n_cohort, 2),
                100.0 * top.share_of_cohort * top[EXC_BS] / a[EXC_BS]))
        if top_pct.start_state != top.start_state:
            emit("    largest as %% of floor: %s, %.2f%%."
                 % (top_pct.start_state, top_pct[PCT_BS]))
        emit("    cohort aggregate: excess %+.6f (%.2f%% of floor); the two-oracle"
             % (a[EXC_BS], a[PCT_BS]))
        emit("    gap is %+.6f (%.2f%% of %s)."
             % (a[GAP], a[GAP_PCT], nulls.ORACLE_LABEL))
    emit()

    elapsed = time.time() - t0
    mem, how = peak_memory_mb()
    rule("=")
    emit("OUTPUTS")
    rule("=")
    for f in (REPORT, COMPARISON_CSV, "compounding_all.csv"):
        emit("    %s" % os.path.join(out_dir, f))
    emit("runtime    : %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    emit("peak memory: %.0f MB  (%s)" % (mem, how))
    rule("=")
    rep.write(os.path.join(out_dir, REPORT))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score persistence, marginal, baseline, the ORACLE (not "
                    "achievable) marginal floor and the ORACLE BY START STATE "
                    "(not achievable) floor in one harness pass, per cohort and "
                    "per start state.")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output directory (default: %(default)s)")
    run(parser.parse_args(argv).out_dir)


if __name__ == "__main__":
    main()
