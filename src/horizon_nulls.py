"""Milestones 2b/2c: persistence, marginal, baseline and the ORACLE (not
achievable) marginal floor, through one harness pass.

    python -m src.horizon_nulls                      # artifacts/horizon_nulls/
    python -m src.horizon_nulls --out-dir some/dir

Eligibility, realised outcomes and cohorts are computed ONCE and shared by all
four; every one is scored by the unmodified src.scoring.score. The baseline is
re-run here (load_baseline + the same chainer) and checked against the saved
artifacts/horizon_baseline/ run, so the comparison is like-for-like.

The oracle marginal is a diagnostic yardstick built from test-period outcomes,
NOT a model: it is labelled ORACLE (not achievable) in every table and CSV.

OUTPUT
    nulls_report.txt       every check and table
    model_comparison.csv   horizon x cohort -> n, Brier for each model and the
                           oracle, baseline excess over the oracle floor
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
from src.horizon_baseline import (COHORTS, Report, fmt, pct, peak_memory_mb,
                                  OUT_DIR as BASELINE_RUN_DIR)

OUT_DIR = os.path.join("artifacts", "horizon_nulls")
REPORT = "nulls_report.txt"
ORACLE = nulls.ORACLE
MODELS = ("persistence", "marginal", "baseline", ORACLE)
LABEL = {"persistence": "persistence", "marginal": "marginal",
         "baseline": "baseline", ORACLE: nulls.ORACLE_LABEL}
ORACLE_CSV = "ORACLE_not_achievable"
FULL_TABLE_HORIZONS = (6, 60)
MATCH_TOL = 1e-12
ORACLE_RTOL = 1e-9

STATES = hz.STATES
I_DEF, I_PRE = STATES.index("Default"), STATES.index("Prepaid")


def fingerprint(ids, arrays):
    """sha256 over loan ids (sorted) and aligned integer arrays."""
    order = np.argsort(ids, kind="stable")
    h = hashlib.sha256()
    h.update("\n".join(np.asarray(ids)[order]).encode("ascii"))
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.int64)[order]).tobytes())
    return h.hexdigest()[:16]


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


def six_state(emit, rule, r, title):
    n = r["n"]
    emit(title)
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


def run(out_dir=OUT_DIR):
    t0 = time.time()
    rep = Report()
    emit, rule = rep.emit, rep.rule

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

    # ---- the three models and the oracle ------------------------------------------
    base_counts, P_df = baseline.load_baseline()
    P = P_df.to_numpy()
    fit_next, n_fit = nulls.fit_next_state_counts()
    marg = nulls.marginal_distribution(fit_next)
    col_sums = base_counts.to_numpy().sum(axis=0)

    preds = {"persistence": nulls.persistence_predictions(start),
             "marginal": nulls.marginal_predictions(marg, n),
             "baseline": hz.chain(start, hz.homogeneous_step(P))}
    preds[ORACLE], oracle_dists = nulls.oracle_predictions(real, masks)
    hz.assert_same_by_start(preds["persistence"], start)
    hz.assert_same_by_start(preds["baseline"], start)
    persist_exact = all(np.array_equal(preds["persistence"][h], hz.one_hot(start))
                        for h in hz.HORIZONS)

    results = {m: {} for m in MODELS}
    for m in MODELS:
        for (h, c), mask in masks.items():
            pm = preds[m][h, c] if m == ORACLE else preds[m][h]
            results[m][h, c] = scoring.score(pm, real[h], mask)
    # the oracle's aggregate prediction must reproduce each cohort's realised
    # counts - it is built from them. Relative tolerance: the expected count is
    # a float sum over ~400k identical rows and carries summation error.
    oracle_exact = all(np.allclose(results[ORACLE][hc]["expected"],
                                   results[ORACLE][hc]["realised"],
                                   rtol=ORACLE_RTOL, atol=0) for hc in masks)

    # ---- header ------------------------------------------------------------------
    rule("=")
    emit("HORIZON NULLS - persistence, marginal, baseline and %s"
         % nulls.ORACLE_LABEL)
    rule("=")
    emit("panel        : data/panel/panel_*.parquet via src.panel_io (build %s)"
         % stamp)
    emit("cut          : %s" % hz.ym(hz.CUT))
    emit("horizons     : %s" % ", ".join("h=%d -> %s" % (h, hz.ym(hz.CUT + h))
                                         for h in hz.HORIZONS))
    emit("scorer       : src.scoring.score, unmodified, for all four")
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
    emit("(scorer) for all four: PASS")
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
    emit("%s aggregate prediction reproduces each cohort's realised"
         % nulls.ORACLE_LABEL)
    emit("counts (relative tolerance %g): %s"
         % (ORACLE_RTOL, "PASS" if oracle_exact else "FAIL"))
    emit()

    # ---- 1 like-for-like -----------------------------------------------------------
    rule("=")
    emit("1. LIKE-FOR-LIKE CHECK   (unit: loans)")
    rule("=")
    emit("One eligibility, outcome and cohort computation is shared by all three")
    emit("models and %s. The counts it produced:" % nulls.ORACLE_LABEL)
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
    rule()
    same = all(results[m][hc]["n"] == results["baseline"][hc]["n"]
               and np.array_equal(results[m][hc]["realised"],
                                  results["baseline"][hc]["realised"])
               for m in MODELS for hc in masks)
    emit("all four scored on identical n and realised counts: %s"
         % ("PASS" if same else "FAIL"))
    fp = fingerprint(E.index.to_numpy(), [start] + [real[h] for h in hz.HORIZONS]
                     + [fixed])
    emit("fingerprint (eligible ids, start state, realised per h, fixed cohort):")
    emit("    %s" % fp)
    emit()
    saved_ok = compare_saved(results, emit, rule)
    emit("re-run baseline matches the saved baseline run: %s"
         % ("PASS" if saved_ok else "FAIL"))
    emit()
    if not (same and saved_ok and same_cols and persist_exact and oracle_exact):
        rep.write(os.path.join(out_dir, REPORT))
        raise SystemExit("like-for-like check failed - see %s"
                         % os.path.join(out_dir, REPORT))

    # ---- 2 Brier comparison ----------------------------------------------------------
    rule("=")
    emit("2. MODEL COMPARISON - multiclass Brier score   (unit: loans in the cohort)")
    rule("=")
    emit("The last column is %s: a floor built from test-period"
         % nulls.ORACLE_LABEL)
    emit("outcomes, not a competing model.")
    rule()
    emit("%-5s%-9s%12s" % ("h", "cohort", "n")
         + "".join("%14s" % LABEL[m] for m in MODELS[:3])
         + "%27s" % LABEL[ORACLE])
    rule()
    rows = []
    for h in hz.HORIZONS:
        for c in COHORTS:
            b = {m: results[m][h, c]["brier"] for m in MODELS}
            k = results["baseline"][h, c]["n"]
            emit("%-5d%-9s%12s" % (h, c, fmt(k))
                 + "".join("%14.6f" % b[m] for m in MODELS[:3])
                 + "%27.6f" % b[ORACLE])
            excess = b["baseline"] - b[ORACLE]
            rows.append({"horizon": h, "cohort": c, "n": k,
                         "brier_persistence": b["persistence"],
                         "brier_marginal": b["marginal"],
                         "brier_baseline": b["baseline"],
                         "brier_" + ORACLE_CSV: b[ORACLE],
                         "baseline_excess_over_" + ORACLE_CSV: excess,
                         "baseline_excess_pct_of_" + ORACLE_CSV:
                             100.0 * excess / b[ORACLE],
                         "oracle_label": nulls.ORACLE_LABEL})
        rule()
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "model_comparison.csv"),
                              index=False, float_format="%.17g")
    emit()

    # ---- 3 excess over the oracle floor ----------------------------------------------
    rule("=")
    emit("3. BASELINE EXCESS OVER THE ORACLE FLOOR   (unit: loans in the cohort)")
    rule("=")
    emit("floor = %s Brier: the realised cohort distribution given"
         % nulls.ORACLE_LABEL)
    emit("to every loan, built from test-period outcomes - not achievable at")
    emit("forecast time. excess = baseline Brier - floor; % of floor = excess /")
    emit("floor x 100.")
    rule()
    emit("%-5s%-9s%12s%34s%14s%14s%14s"
         % ("h", "cohort", "n", "floor: " + LABEL[ORACLE], "baseline", "excess",
            "% of floor"))
    rule()
    for row in rows:
        emit("%-5d%-9s%12s%34.6f%14.6f%+14.6f%13.2f%%"
             % (row["horizon"], row["cohort"], fmt(row["n"]),
                row["brier_" + ORACLE_CSV], row["brier_baseline"],
                row["baseline_excess_over_" + ORACLE_CSV],
                row["baseline_excess_pct_of_" + ORACLE_CSV]))
        if row["cohort"] == COHORTS[-1]:
            rule()
    emit()

    # ---- 3 compounding ---------------------------------------------------------------
    rule("=")
    emit("4. DEFAULT AND PREPAID, PREDICTED VS REALISED - THREE MODELS AND")
    emit("   %s" % nulls.ORACLE_LABEL)
    rule("=")
    emit("pred = expected loans (sum of predicted probabilities) and share of the")
    emit("cohort; real = realised loans and share; gap = pred share - real share,")
    emit("in percentage points.")
    rule()
    emit("%-4s%-9s%-24s%9s %-19s%-17s%8s %-20s%-18s%8s"
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
                emit("%-4d%-9s%-24s%9s %-19s%-17s%8s %-20s%-18s%8s"
                     % tuple([h, c, LABEL[m], fmt(k)] + cells))
        rule()
    pd.DataFrame(comp).to_csv(os.path.join(out_dir, "compounding_all.csv"),
                              index=False, float_format="%.17g")
    emit()

    # ---- 5 six-state tables for the nulls and the oracle -----------------------------
    rule("=")
    emit("5. SIX-STATE PREDICTED VS REALISED, NULL MODELS AND %s,"
         % nulls.ORACLE_LABEL)
    emit("   h = %s" % " and ".join(map(str, FULL_TABLE_HORIZONS)))
    rule("=")
    for m in ("persistence", "marginal", ORACLE):
        for h in FULL_TABLE_HORIZONS:
            for c in COHORTS:
                six_state(emit, rule, results[m][h, c],
                          "%s - h = %d (%s), %s cohort"
                          % (LABEL[m], h, hz.ym(hz.CUT + h), c))

    elapsed = time.time() - t0
    mem, how = peak_memory_mb()
    rule("=")
    emit("OUTPUTS")
    rule("=")
    for f in (REPORT, "model_comparison.csv", "compounding_all.csv"):
        emit("    %s" % os.path.join(out_dir, f))
    emit("runtime    : %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    emit("peak memory: %.0f MB  (%s)" % (mem, how))
    rule("=")
    rep.write(os.path.join(out_dir, REPORT))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score persistence, marginal, baseline and the ORACLE (not "
                    "achievable) marginal floor in one harness pass.")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output directory (default: %(default)s)")
    run(parser.parse_args(argv).out_dir)


if __name__ == "__main__":
    main()
