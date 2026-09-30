"""Milestone 2: the homogeneous baseline run through the horizon harness.

    python -m src.horizon_baseline                      # artifacts/horizon_baseline/
    python -m src.horizon_baseline --out-dir some/dir

The pooled matrix comes from src.baseline.load_baseline() (derived from
artifacts/baseline/counts.csv). src.horizon supplies eligibility, outcomes,
cohorts and the chainer; src.scoring scores the resulting distributions without
knowing where they came from.

OUTPUT
    horizon_report.txt      every check and table
    predicted_by_start.csv  horizon x start state -> predicted distribution
    scores.csv              horizon x cohort -> n, Brier score
    compounding.csv         horizon x cohort -> Default / Prepaid, predicted
                            vs realised, signed gap
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from src import baseline, horizon as hz, scoring

OUT_DIR = os.path.join("artifacts", "horizon_baseline")
REPORT = "horizon_report.txt"
RULE_W = 121
COHORTS = ("varying", "fixed")

STATES = hz.STATES
I_DEF, I_PRE = STATES.index("Default"), STATES.index("Prepaid")


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def peak_memory_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            return peak / 2 ** 20, "peak working set (psutil)"
    except ImportError:
        pass
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (kb / 2 ** 20 if sys.platform == "darwin" else kb / 2 ** 10,
                "max resident set size (getrusage)")
    except ImportError:
        return float("nan"), "unavailable"


class Report(baseline.Report):
    def rule(self, char="-", width=RULE_W):
        self.emit(char * width)


def run(out_dir=OUT_DIR):
    t0 = time.time()
    rep = Report()
    emit, rule = rep.emit, rep.rule

    counts_df, P_df = baseline.load_baseline()
    if list(P_df.columns) != STATES:
        raise SystemExit("baseline state order %s != harness %s"
                         % (list(P_df.columns), STATES))
    P = P_df.to_numpy()
    L, stamp = hz.read_test_loans(progress=lambda s: print(s, flush=True))
    targets = hz.target_months()

    rule("=")
    emit("HORIZON HARNESS - homogeneous baseline, test-split loans")
    rule("=")
    emit("panel        : data/panel/panel_*.parquet via src.panel_io (build %s)"
         % stamp)
    emit("rows read    : role in {condition, evaluate}; columns %s"
         % ", ".join(hz.COLUMNS))
    emit("model        : pooled matrix from src.baseline.load_baseline() "
         "(artifacts/baseline/counts.csv)")
    emit("cut          : %s" % hz.ym(hz.CUT))
    emit("horizons     : %s" % ", ".join("h=%d -> %s" % (h, hz.ym(targets[h]))
                                         for h in hz.HORIZONS))
    emit("state order  : %s" % ", ".join(STATES))
    emit("A month is observed by a row's origin_state (month t) or next_state")
    emit("(month t+1). The 2008-12 -> 2009-01 transition is a condition row; its")
    emit("next_state is read as the first post-cut outcome (see section 2).")
    emit()

    # ---- 1 ---------------------------------------------------------------------
    rule("=")
    emit("1. ELIGIBILITY AT THE CUT   (unit: test-split loans with a panel row)")
    rule("=")
    reason = hz.eligibility(L).to_numpy()
    n_all = len(L)
    elig = reason < 0
    E = L[elig]
    start = E["state_cut"].to_numpy()
    emit("%-52s%16s%12s" % ("", "loans", "share"))
    rule()
    emit("%-52s%16s%12s" % ("test-split loans in the panel", fmt(n_all),
                            pct(n_all, n_all)))
    for k, why in enumerate(hz.EXCLUDED):
        c = int((reason == k).sum())
        emit("%-52s%16s%12s" % ("  excluded: " + why, fmt(c), pct(c, n_all)))
    emit("%-52s%16s%12s" % ("  ELIGIBLE: transient state at the cut",
                            fmt(elig.sum()), pct(elig.sum(), n_all)))
    rule()
    emit("excluded reasons: absorbed = first Prepaid/Default observation at or")
    emit("before %s; the other three have no observation at %s and are not"
         % (hz.ym(hz.CUT), hz.ym(hz.CUT)))
    emit("absorbed: first observation after the cut / last observation before it /")
    emit("observed on both sides but not at it.")
    emit()
    ab = L[reason == 0]
    emit("absorbed at or before the cut, by state: %s"
         % ", ".join("%s %s" % (s, fmt((ab["absorb_state"] == STATES.index(s)).sum()))
                     for s in hz.ABSORBING))
    emit()
    emit("%-14s%16s%12s" % ("state at cut", "eligible", "share"))
    rule()
    for s in hz.TRANSIENT:
        c = int((start == STATES.index(s)).sum())
        emit("%-14s%16s%12s" % (s, fmt(c), pct(c, len(E))))
    rule()
    emit("%-14s%16s%12s" % ("TOTAL", fmt(len(E)), pct(len(E), len(E))))
    emit()

    # ---- 2 ---------------------------------------------------------------------
    rule("=")
    emit("2. REALISED OUTCOMES   (unit: eligible loans)")
    rule("=")
    real = hz.realised(E)
    n = len(E)
    for h in hz.HORIZONS:
        if not np.all(np.isin(real[h], [hz.CENSORED] + list(range(len(STATES))))):
            raise SystemExit("h=%d: realised code out of range" % h)
    emit("%-6s%10s%14s%16s%12s%16s%12s"
         % ("h", "T", "eligible", "observed", "share", "censored", "share"))
    rule()
    for h in hz.HORIZONS:
        obs = int((real[h] != hz.CENSORED).sum())
        emit("%-6d%10s%14s%16s%12s%16s%12s"
             % (h, hz.ym(targets[h]), fmt(n), fmt(obs), pct(obs, n, 2),
                fmt(n - obs), pct(n - obs, n, 2)))
    rule()
    emit()
    emit("Observed, split into absorbed (Prepaid + Default) and transient;")
    emit("shares are of observed:")
    rule()
    emit("%-6s%14s%14s%10s%14s%10s%14s%10s%14s%10s"
         % ("h", "observed", "absorbed", "share", "Prepaid", "share", "Default",
            "share", "transient", "share"))
    rule()
    for h in hz.HORIZONS:
        r = real[h]
        obs = int((r != hz.CENSORED).sum())
        pre, de = int((r == I_PRE).sum()), int((r == I_DEF).sum())
        tr = obs - pre - de
        emit("%-6d%14s%14s%10s%14s%10s%14s%10s%14s%10s"
             % (h, fmt(obs), fmt(pre + de), pct(pre + de, obs, 2), fmt(pre),
                pct(pre, obs, 2), fmt(de), pct(de, obs, 2), fmt(tr),
                pct(tr, obs, 2)))
    rule()
    first_post = int((E["absorb_month"] == hz.CUT + 1).sum())
    emit("eligible loans absorbing in %s - read from the condition row with origin"
         % hz.ym(hz.CUT + 1))
    emit("%s, the only row carrying that absorption: %s"
         % (hz.ym(hz.CUT), fmt(first_post)))
    emit()

    # ---- 3 ---------------------------------------------------------------------
    rule("=")
    emit("3. COHORTS   (unit: eligible loans)")
    rule("=")
    varying, fixed = hz.cohorts(real)
    emit("varying: observable at h.  fixed: observable at every horizon, incl. 60.")
    rule()
    emit("%-6s%18s%12s%18s%12s" % ("h", "varying", "of elig.", "fixed",
                                   "of elig."))
    rule()
    for h in hz.HORIZONS:
        emit("%-6d%18s%12s%18s%12s" % (h, fmt(varying[h].sum()),
                                       pct(varying[h].sum(), n, 2),
                                       fmt(fixed.sum()), pct(fixed.sum(), n, 2)))
    rule()
    sub = all(np.all(varying[h][fixed]) for h in hz.HORIZONS)
    emit("fixed cohort is a subset of every varying cohort: %s"
         % ("PASS" if sub else "FAIL"))
    emit()
    emit("Starting-state breakdown (count, share of the cohort):")
    rule()
    cols = [("eligible", np.ones(n, dtype=bool))] + \
        [("var h=%d" % h, varying[h]) for h in hz.HORIZONS] + [("fixed", fixed)]
    emit("%-10s" % "state" + "".join("%18s" % c for c, _ in cols))
    rule()
    for s in hz.TRANSIENT:
        i = STATES.index(s)
        emit("%-10s" % s + "".join("%18s" % fmt(((start == i) & m).sum())
                                   for _, m in cols))
        emit("%-10s" % "" + "".join("%18s" % pct(((start == i) & m).sum(),
                                                 m.sum(), 2) for _, m in cols))
    rule()
    emit("%-10s" % "TOTAL" + "".join("%18s" % fmt(m.sum()) for _, m in cols))
    emit()

    # ---- 4 ---------------------------------------------------------------------
    rule("=")
    emit("4. BASELINE PREDICTIONS   (pooled matrix chained month by month)")
    rule("=")
    pred = hz.chain(start, hz.homogeneous_step(P))
    emit("every predicted distribution sums to 1 within %g: PASS" % hz.TOL)
    hz.assert_same_by_start(pred, start)
    emit("every loan with the same start state has an identical distribution: PASS")
    emit()
    emit("Predicted distribution by start state. Line 1: shares. Line 2: expected")
    emit("loans (share x the eligible loans starting in that state, n).")
    rows_csv = []
    w = 14
    for h in hz.HORIZONS:
        emit()
        emit("h = %d  (%s)" % (h, hz.ym(targets[h])))
        rule()
        emit("%-10s%10s" % ("start", "n") + "".join("%*s" % (w, s) for s in STATES))
        rule()
        for s in hz.TRANSIENT:
            i = STATES.index(s)
            m = start == i
            if not m.any():
                emit("%-10s%10s   no eligible loan starts here" % (s, "0"))
                continue
            dist = pred[h][m][0]
            k = int(m.sum())
            emit("%-10s%10s" % (s, fmt(k)) + "".join("%*s" % (w, "%.4f%%"
                                                              % (100 * v))
                                                     for v in dist))
            emit("%-10s%10s" % ("", "") + "".join("%*s" % (w, "{:,.1f}".format(v * k))
                                                  for v in dist))
            rows_csv.append([h, s, k] + list(dist))
        rule()
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows_csv, columns=["horizon", "start_state", "n_loans"] + STATES) \
        .to_csv(os.path.join(out_dir, "predicted_by_start.csv"), index=False,
                float_format="%.17g")
    emit()

    # ---- 5 ---------------------------------------------------------------------
    rule("=")
    emit("5. SCORING   (src.scoring; unit: loans in the cohort)")
    rule("=")
    results = {}
    for h in hz.HORIZONS:
        for c in COHORTS:
            mask = varying[h] if c == "varying" else fixed
            results[h, c] = scoring.score(pred[h], real[h], mask)
            scoring.render(emit, rule, results[h, c], STATES,
                           "h = %d (%s), %s cohort, n = %s"
                           % (h, hz.ym(targets[h]), c, fmt(results[h, c]["n"])))
    pd.DataFrame([{"horizon": h, "cohort": c, "n": results[h, c]["n"],
                   "brier": results[h, c]["brier"]}
                  for h in hz.HORIZONS for c in COHORTS]) \
        .to_csv(os.path.join(out_dir, "scores.csv"), index=False,
                float_format="%.17g")
    emit("Brier scores:")
    rule()
    emit("%-6s" % "h" + "".join("%26s" % c for c in COHORTS))
    rule()
    for h in hz.HORIZONS:
        emit("%-6d" % h + "".join("%26s" % ("%.6f  (n=%s)" % (results[h, c]["brier"],
                                                             fmt(results[h, c]["n"])))
                                  for c in COHORTS))
    rule()
    emit()

    # ---- 6 ---------------------------------------------------------------------
    rule("=")
    emit("6. COMPOUNDING TABLE   (predicted vs realised, Default and Prepaid)")
    rule("=")
    emit("pred = expected loans (sum of predicted probabilities) and share of the")
    emit("cohort; real = realised loans and share; gap = pred share - real share,")
    emit("in percentage points.")
    rule()
    emit("%-4s%-9s%10s  %-20s%-20s%9s  %-20s%-20s%9s"
         % ("h", "cohort", "n", "Default pred", "Default real", "gap pp",
            "Prepaid pred", "Prepaid real", "gap pp"))
    rule()
    comp = []
    for h in hz.HORIZONS:
        for c in COHORTS:
            r = results[h, c]
            k = r["n"]
            row = {"horizon": h, "cohort": c, "n": k}
            cells = []
            for name, i in (("default", I_DEF), ("prepaid", I_PRE)):
                e, a = r["expected"][i], int(r["realised"][i])
                gap = 100.0 * (e - a) / k if k else float("nan")
                row.update({name + "_pred_n": e, name + "_pred_share": e / k,
                            name + "_real_n": a, name + "_real_share": a / k,
                            name + "_gap_pp": gap})
                cells += ["{:,.1f} ({})".format(e, pct(e, k, 2)),
                          "{} ({})".format(fmt(a), pct(a, k, 2)), "%+.2f" % gap]
            comp.append(row)
            emit("%-4d%-9s%10s  %-20s%-20s%9s  %-20s%-20s%9s"
                 % tuple([h, c, fmt(k)] + cells))
        rule()
    pd.DataFrame(comp).to_csv(os.path.join(out_dir, "compounding.csv"),
                              index=False, float_format="%.17g")
    emit()

    elapsed = time.time() - t0
    mem, how = peak_memory_mb()
    rule("=")
    emit("OUTPUTS")
    rule("=")
    for f in (REPORT, "predicted_by_start.csv", "scores.csv", "compounding.csv"):
        emit("    %s" % os.path.join(out_dir, f))
    emit("runtime    : %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    emit("peak memory: %.0f MB  (%s)" % (mem, how))
    rule("=")
    rep.write(os.path.join(out_dir, REPORT))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the homogeneous baseline through the horizon harness.")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output directory (default: %(default)s)")
    run(parser.parse_args(argv).out_dir)


if __name__ == "__main__":
    main()
