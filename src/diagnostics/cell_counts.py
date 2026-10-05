"""Cell-count diagnostic for the conditional oracle. Read-only.

    python -m src.diagnostics.cell_counts            # reports/cell_counts.csv

The conditional oracle will give every loan the realised outcome distribution
of its (cohort, starting state, horizon) cell. Scored on the same loans, it
memorises each cell's sampling noise. This diagnostic sizes that effect per
cell and nothing else.

DEFINITIONS - all imported, none reimplemented
    eligibility, starting state   src.horizon.eligibility / state_cut (the
                                  state observed at the cut, condition rows)
    realised state at T_h         src.horizon.realised (evaluate rows, plus the
                                  2008-12 -> 2009-01 condition row; absorbed by
                                  T_h stays absorbed)
    cohorts                       src.horizon.cohorts (varying: observable at
                                  h; fixed: observable at every horizon)
    cell distribution             src.nulls.oracle_distribution over the cell
    cell_brier                    src.scoring.score of that distribution, given
                                  to every loan in the cell, against the cell
    noise_inflation               cell_brier / n. In-sample, E[cell_brier] =
                                  B_true * (n - 1) / n, where B_true = 1 -
                                  sum_k p_k^2 is the Brier of the true cell
                                  distribution; the optimism is B_true / n,
                                  which cell_brier / n approximates.

RECONCILIATION - against the saved harness runs, before anything is written
    artifacts/horizon_baseline/horizon_report.txt   starting-state x cohort
                                                    counts (section 3)
    artifacts/horizon_baseline/scores.csv           n per horizon x cohort
    artifacts/horizon_baseline/compounding.csv      realised Default / Prepaid
                                                    per horizon x cohort
    artifacts/horizon_nulls/nulls_report.txt        data fingerprint
A disagreement is printed and the run exits non-zero (the CSV is still
written so the disagreement can be inspected).
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd

from src import horizon as hz, nulls, scoring
from src.horizon_baseline import (COHORTS, OUT_DIR as BASELINE_RUN_DIR, fmt,
                                  peak_memory_mb)
from src.horizon_nulls import fingerprint, saved_fingerprint

OUT_PATH = os.path.join("reports", "cell_counts.csv")
STATES = hz.STATES
I_DEF, I_PRE = STATES.index("Default"), STATES.index("Prepaid")
SUM_RTOL = 1e-9
BRIER_TOL = 1e-12
RULE_W = 132


def rule(char="-"):
    print(char * RULE_W)


# ==========================================================================
# saved harness runs
# ==========================================================================
def saved_breakdown(path=os.path.join(BASELINE_RUN_DIR, "horizon_report.txt")):
    """{(cohort, state, h): n} from section 3's starting-state breakdown."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("Starting-state breakdown"))
    header = next(ln for ln in lines[i:] if ln.startswith("state "))
    cols = re.split(r"\s{2,}", header.strip())[1:]
    out = {}
    for ln in lines[i:]:
        parts = ln.split()
        if not parts or parts[0] not in hz.TRANSIENT:
            continue
        vals = dict(zip(cols, (int(v.replace(",", "")) for v in parts[1:])))
        for h in hz.HORIZONS:
            out["varying", parts[0], h] = vals["var h=%d" % h]
            out["fixed", parts[0], h] = vals["fixed"]
        if len(out) == 2 * len(hz.TRANSIENT) * len(hz.HORIZONS):
            break
    return out


# ==========================================================================
# the grid
# ==========================================================================
def cell(real_h, member):
    """(n, proportions, counts, cell_brier) for one cell."""
    n = int(member.sum())
    if n == 0:
        return 0, np.full(hz.N_STATES, np.nan), np.zeros(hz.N_STATES, int), \
            float("nan")
    dist, counts = nulls.oracle_distribution(real_h, member)
    r = real_h[member]
    res = scoring.score(np.tile(dist, (n, 1)), r, np.ones(n, dtype=bool))
    closed = 1.0 - float((dist ** 2).sum())
    if abs(res["brier"] - closed) > BRIER_TOL:
        raise hz.HorizonError("cell Brier %.17g != 1 - sum p^2 = %.17g"
                              % (res["brier"], closed))
    return n, dist, counts, res["brier"]


def run(out_path=OUT_PATH):
    t0 = time.time()
    L, stamp = hz.read_test_loans(progress=lambda s: print(s, flush=True))
    reason = hz.eligibility(L).to_numpy()
    E = L[reason < 0]
    start = E["state_cut"].to_numpy()
    real = hz.realised(E)
    varying, fixed = hz.cohorts(real)
    masks = {(h, c): (varying[h] if c == "varying" else fixed)
             for h in hz.HORIZONS for c in COHORTS}

    rows, counts = [], {}
    for c in COHORTS:
        for s in hz.TRANSIENT:
            at_s = start == STATES.index(s)
            for h in hz.HORIZONS:
                n, dist, cnt, b = cell(real[h], masks[h, c] & at_s)
                counts[c, s, h] = cnt
                rows.append({"cohort": c, "start_state": s, "horizon": h, "n": n,
                             **{"p_" + k: v for k, v in zip(STATES, dist)},
                             "cell_brier": b,
                             "noise_inflation": b / n if n else float("nan")})
    df = pd.DataFrame(rows)
    pcols = ["p_" + k for k in STATES]

    # ---- checks -------------------------------------------------------------------
    problems = []
    sums = df[pcols].sum(axis=1).to_numpy()
    bad = ~(np.abs(sums - 1.0) <= SUM_RTOL)
    for _, r in df[bad].iterrows():
        problems.append("proportions: %s %s h=%d sum to %r (n=%d)"
                        % (r.cohort, r.start_state, r.horizon,
                           sums[r.name], r.n))

    saved = saved_breakdown()
    for _, r in df.iterrows():
        want = saved[r.cohort, r.start_state, r.horizon]
        if r.n != want:
            problems.append("n: %s %s h=%d: here %s, saved baseline run %s"
                            % (r.cohort, r.start_state, r.horizon, fmt(r.n),
                               fmt(want)))
    sc = pd.read_csv(os.path.join(BASELINE_RUN_DIR, "scores.csv"))
    cp = pd.read_csv(os.path.join(BASELINE_RUN_DIR, "compounding.csv"))
    for h in hz.HORIZONS:
        for c in COHORTS:
            tot = sum(counts[c, s, h] for s in hz.TRANSIENT)
            s_row = sc[(sc["horizon"] == h) & (sc["cohort"] == c)].iloc[0]
            k_row = cp[(cp["horizon"] == h) & (cp["cohort"] == c)].iloc[0]
            for what, here, there in (("n", tot.sum(), s_row["n"]),
                                      ("Default", tot[I_DEF],
                                       k_row["default_real_n"]),
                                      ("Prepaid", tot[I_PRE],
                                       k_row["prepaid_real_n"])):
                if int(here) != int(there):
                    problems.append("%s: %s h=%d summed over start states %s, "
                                    "saved baseline run %s"
                                    % (what, c, h, fmt(here), fmt(there)))
    fp = fingerprint(E.index.to_numpy(), [start] + [real[h] for h in hz.HORIZONS]
                     + [fixed])
    fp_saved = saved_fingerprint()
    if fp != fp_saved:
        problems.append("fingerprint: here %s, saved nulls run %s" % (fp, fp_saved))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.17g")

    # ---- stdout ---------------------------------------------------------------------
    rule("=")
    print("CELL COUNTS - conditional oracle cells (cohort x start state x horizon)")
    rule("=")
    print("panel build  : %s" % stamp)
    print("cut          : %s;  horizons %s" % (hz.ym(hz.CUT),
                                              ", ".join(map(str, hz.HORIZONS))))
    print("unit         : eligible test-split loans; proportions are of the cell")
    rule()
    w = 11
    print("%-9s%-8s%4s%10s" % ("cohort", "start", "h", "n")
          + "".join("%*s" % (w, k) for k in STATES)
          + "%12s%16s" % ("cell_brier", "noise_infl."))
    rule()
    for c in COHORTS:
        for s in hz.TRANSIENT:
            for _, r in df[(df.cohort == c) & (df.start_state == s)].iterrows():
                print("%-9s%-8s%4d%10s" % (c, s, r.horizon, fmt(r.n))
                      + "".join("%*.4f%%" % (w - 1, 100 * r[p]) for p in pcols)
                      + "%12.6f%16.3e" % (r.cell_brier, r.noise_inflation))
        rule()
    print()
    print("CHECKS")
    rule()
    print("proportions sum to 1 within %g (relative), all %d cells: %s"
          % (SUM_RTOL, len(df), "PASS" if not bad.any() else "FAIL"))
    print("cell_brier == 1 - sum p^2 within %g, all cells: PASS" % BRIER_TOL)
    n_checks = [p for p in problems if not p.startswith(("proportions",
                                                         "fingerprint"))]
    print("n per cell vs saved baseline run (section 3 breakdown); per-cohort n,")
    print("realised Default and Prepaid vs scores.csv / compounding.csv: %s"
          % ("PASS" if not n_checks else "FAIL"))
    print("data fingerprint %s vs saved nulls run %s: %s"
          % (fp, fp_saved, "PASS" if fp == fp_saved else "FAIL"))
    for p in problems:
        print("  DISCREPANCY  " + p)
    rule()
    print()

    nmin = df["n"].min()
    small = df[df["n"] == nmin]
    big = df.loc[df["noise_inflation"].idxmax()]
    print("SMALLEST CELL: n = %s at %s"
          % (fmt(nmin), "; ".join("(%s, %s, h=%d)" % (r.cohort, r.start_state,
                                                        r.horizon)
                                  for _, r in small.iterrows())))
    print("LARGEST noise_inflation: %.6e at (%s, %s, h=%d), n = %s, "
          "cell_brier = %.6f"
          % (big.noise_inflation, big.cohort, big.start_state, big.horizon,
             fmt(big.n), big.cell_brier))
    print()
    elapsed = time.time() - t0
    mem, how = peak_memory_mb()
    print("output     : %s" % out_path)
    print("runtime    : %.1f s" % elapsed)
    print("peak memory: %.0f MB  (%s)" % (mem, how))
    rule("=")
    return 1 if problems else 0


def main():
    sys.exit(run())


if __name__ == "__main__":
    main()
