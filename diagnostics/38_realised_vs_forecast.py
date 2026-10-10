"""Realised vs forecast for the canary loans: what they did, what the baseline says, what the two ML arms say.   READ-ONLY.

    python diagnostics/38_realised_vs_forecast.py [--with-age DIR] [--without-age DIR] [--batch 0]

Writes diagnostics/38_realised_vs_forecast.txt and nothing else. Descriptive: it recommends no feature set, scores the arms
against each other with no metric (no log-loss, no Brier, no aggregate distance), and states numbers and per-state gaps.

THE LOANS are read from the forecast output itself (data/forecast/<run>/batch=NNNNN/part.parquet, the h = 0 rows), not
re-derived; the two runs must hold the identical loan set. Each loan's forecast start state is checked against the state
the panel shows it in at the cut.

A - REALISED. src.horizon.read_test_loans (the existing harness reader: condition + evaluate rows) and
src.horizon.realised: a loan that reached Prepaid or Default is carried there; a loan with no observation at the horizon
month and no absorption by it is CENSORED and is kept as its own column, never redistributed. Shares are of the loans in the
batch.
B - BASELINE. src.baseline.load_baseline() (the pooled homogeneous matrix) pushed through src.horizon.chain /
homogeneous_step for the same loans' start states, same cut, same horizons. h = 0 is the start state itself (chain starts
at month 1). No scoring code is written here.
C - SIDE BY SIDE. Per horizon: realised, baseline, ML with loan_age, ML without loan_age, as shares of the loans; then the
forecast-minus-realised gap per state in percentage points. Then the realised monthly prepayment rate by calendar month.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import baseline, horizon as hz                                    # noqa: E402

OUT = os.path.join("diagnostics", "38_realised_vs_forecast.txt")
STATES = hz.STATES
NS = len(STATES)
I_PRE, I_DEF = STATES.index("Prepaid"), STATES.index("Default")
HS = (0, 6, 12, 24, 60)
MONTHS = 60
RULE_W = 118
LINES = []


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def rule(c="-"):
    emit(c * RULE_W)


def f(n):
    return format(int(n), ",")


def read_forecast(run_dir, batch):
    """(loan ids, start-state codes, (61, 6) summed probabilities, run.json) of one run's batch."""
    bdir = os.path.join(run_dir, "batch=%05d" % batch)
    if not os.path.exists(os.path.join(bdir, "_DONE.json")):
        raise SystemExit("%s has no _DONE.json: that batch is not complete" % bdir)
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    path = os.path.join(bdir, "part.parquet")
    t0 = pq.read_table(path, columns=["loan_id", "state", "probability"], filters=[("h", "==", 0)]).to_pandas()
    t0 = t0[t0["probability"] > 0.5]                                   # the initial vector is one-hot
    if t0["loan_id"].duplicated().any():
        raise SystemExit("%s: h = 0 does not give one start state per loan" % path)
    ids = t0["loan_id"].astype(str).to_numpy()
    start = t0["state"].astype(str).map(STATES.index).to_numpy()
    t = pq.read_table(path, columns=["h", "state", "probability"])
    st = t.column("state").combine_chunks()
    if st.dictionary.to_pylist() != STATES:
        raise SystemExit("%s: state order %s" % (path, st.dictionary.to_pylist()))
    key = t.column("h").to_numpy().astype(np.int64) * NS + st.indices.to_numpy().astype(np.int64)
    sums = np.bincount(key, weights=t.column("probability").to_numpy().astype(np.float64),
                       minlength=61 * NS).reshape(61, NS)
    if t.num_rows != len(ids) * 61 * NS:
        raise SystemExit("%s: %d rows, expected %d" % (path, t.num_rows, len(ids) * 61 * NS))
    return ids, start, sums, cfg


def share_row(label, vec, n, censored=None):
    cells = "".join("%11.4f%%" % (100.0 * v / n) for v in vec)
    cen = "" if censored is None else "%11.4f%%" % (100.0 * censored / n)
    return "  %-26s%s%s" % (label, cells, cen)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-age", default=os.path.join("data", "forecast", "pooled_v1"))
    ap.add_argument("--without-age", default=os.path.join("data", "forecast", "pooled_no_loan_age_v1"))
    ap.add_argument("--batch", type=int, default=0)
    args = ap.parse_args()

    ids_a, start_a, sums_a, cfg_a = read_forecast(args.with_age, args.batch)
    ids_b, start_b, sums_b, cfg_b = read_forecast(args.without_age, args.batch)
    if set(ids_a) != set(ids_b):
        raise SystemExit("the two forecast runs do not hold the same loans in batch %d" % args.batch)
    order_b = pd.Series(np.arange(len(ids_b)), index=ids_b).loc[ids_a].to_numpy()
    if not np.array_equal(start_a, start_b[order_b]):
        raise SystemExit("the two runs disagree on a loan's start state")
    n = len(ids_a)

    L, stamp = hz.read_test_loans(horizons=HS, progress=lambda s: print(s, flush=True))
    reason = hz.eligibility(L)
    missing = [i for i in ids_a if i not in L.index]
    if missing:
        raise SystemExit("%d forecast loans are not test-split loans in the panel, e.g. %s" % (len(missing), missing[:5]))
    E = L.loc[ids_a]
    not_elig = int((reason.loc[ids_a].to_numpy() >= 0).sum())
    real = hz.realised(E, horizons=HS)
    cut_state = E["state_cut"].to_numpy()

    rule("=")
    emit("REALISED vs FORECAST   (descriptive: no feature-set recommendation, no model-selection metric)")
    rule("=")
    emit("panel build %s; cut %s; horizons h = %s months (%s)" % (
        stamp, hz.ym(hz.CUT), ", ".join(map(str, HS)), ", ".join(hz.ym(hz.CUT + h) for h in HS)))
    emit()
    emit("FORECAST RUNS (batch %d of each)" % args.batch)
    for lab, d, cfg in (("ML with loan_age   ", args.with_age, cfg_a), ("ML without loan_age", args.without_age, cfg_b)):
        mods = {v["directory"] + " sha256 " + v["sha256"][:16] for v in cfg["models"].values()}
        emit("  %s  %s   model: %s" % (lab, d.replace(os.sep, "/"), "; ".join(sorted(mods))))
    emit()

    rule("=")
    emit("1. THE LOANS")
    rule("=")
    emit("  identified from the forecast output: the h = 0 rows of batch %d in each run (not re-derived)." % args.batch)
    emit("  loans: %s in each run; the two runs hold the identical set: yes; same start state for every loan: yes" % f(n))
    emit("  found among the panel's test-split (condition + evaluate) loans: %s of %s" % (f(n), f(n)))
    emit("  eligible at the cut by src.horizon.eligibility: %s of %s" % (f(n - not_elig), f(n)))
    bad = int((cut_state != start_a).sum())
    emit("  forecast start state == the state the panel shows at the cut, loan by loan: %s of %s%s" % (
        f(n - bad), f(n), "" if not bad else "   *** %s DISAGREE ***" % f(bad)))
    emit("  start states: %s" % ", ".join("%s %s" % (STATES[i], f((start_a == i).sum())) for i in range(4)))
    emit()

    rule("=")
    emit("2. CENSORING   (no observation at the horizon month and not absorbed by it: its own column, never redistributed)")
    rule("=")
    emit("  %-4s %-8s %10s %12s %14s %12s %16s %12s" % ("h", "month", "loans", "observed", "of which", "censored",
                                                      "ends before T", "hole at T"))
    emit("  %-4s %-8s %10s %12s %14s %12s %16s %12s" % ("", "", "", "", "absorbed", "(total)", "", ""))
    last = E["last"].to_numpy()
    for h in HS:
        r = real[h]
        cen = r == hz.CENSORED
        T = hz.CUT + h
        ends = cen & (last < T)
        hole = cen & (last >= T)
        absorbed = np.isin(r, hz.I_ABSORBING)
        emit("  %-4d %-8s %10s %12s %14s %12s %16s %12s" % (h, hz.ym(T), f(n), f((~cen).sum()), f(absorbed.sum()),
                                                          f(cen.sum()), f(ends.sum()), f(hole.sum())))
    emit("  'ends before T': the loan's last observation is earlier than the horizon month.  'hole at T': observed on both")
    emit("  sides of T but not at it.  A loan absorbed (Prepaid/Default) at or before T is observed by construction.")
    emit()

    # ---- baseline ------------------------------------------------------------------------------------------
    _, P_df = baseline.load_baseline()
    if list(P_df.columns) != STATES:
        raise SystemExit("baseline state order %s != harness %s" % (list(P_df.columns), STATES))
    base = hz.chain(cut_state, hz.homogeneous_step(P_df.to_numpy()), horizons=tuple(range(1, MONTHS + 1)))
    base_sum = np.vstack([hz.one_hot(cut_state).sum(axis=0)] + [base[k].sum(axis=0) for k in range(1, MONTHS + 1)])

    rule("=")
    emit("3. SIDE BY SIDE   (shares of the %s loans; each forecast row sums to 100%%)" % f(n))
    rule("=")
    head = "  %-26s" % "" + "".join("%12s" % s for s in STATES) + "%12s" % "Censored"
    rows = {}
    for h in HS:
        r = real[h]
        cnt = np.array([(r == i).sum() for i in range(NS)])
        cen = int((r == hz.CENSORED).sum())
        rows[h] = {"realised": cnt, "baseline": base_sum[h], "with": sums_a[h], "without": sums_b[h]}
        emit("h = %d  (%s)" % (h, hz.ym(hz.CUT + h)))
        emit(head)
        emit(share_row("realised", cnt, n, cen))
        emit("  %-26s%s%s" % ("   realised, loans", "".join("%12s" % f(c) for c in cnt), "%12s" % f(cen)))
        emit(share_row("baseline (homogeneous)", base_sum[h], n))
        emit(share_row("ML with loan_age", sums_a[h], n))
        emit(share_row("ML without loan_age", sums_b[h], n))
        emit("  censored loans at h = %d: %s of %s" % (h, f(cen), f(n)))
        emit()

    rule("=")
    emit("4. FORECAST MINUS REALISED, percentage points of the %s loans   (a censored loan is not in any state column)" % f(n))
    rule("=")
    for h in HS:
        emit("h = %d  (%s)" % (h, hz.ym(hz.CUT + h)))
        emit("  %-26s" % "" + "".join("%12s" % s for s in STATES))
        for lab, key in (("baseline (homogeneous)", "baseline"), ("ML with loan_age", "with"),
                         ("ML without loan_age", "without")):
            gap = 100.0 * (rows[h][key] - rows[h]["realised"]) / n
            emit("  %-26s%s" % (lab, "".join("%+11.4f " % g for g in gap)))
        emit()

    # ---- prepayment time series ---------------------------------------------------------------------------
    am, ast = E["absorb_month"].to_numpy(), E["absorb_state"].to_numpy()
    rule("=")
    emit("5. REALISED MONTHLY PREPAYMENT, calendar months %s .. %s, these %s loans" % (
        hz.ym(hz.CUT + 1), hz.ym(hz.CUT + MONTHS), f(n)))
    rule("=")
    emit("  at risk in month m: not absorbed before m and observed through m-1 (absorbed loans carried, censored loans")
    emit("  leave the denominator once unobserved). rate = new Prepaid in m / at risk.  cumulative = Prepaid by m / %s." % f(n))
    emit("  'implied' columns: the forecasts' own monthly rate from their portfolio sums, (Prepaid_h - Prepaid_h-1) /")
    emit("  (1 - Prepaid_h-1 - Default_h-1); shown only so the shapes can be read side by side.")
    emit("  %-8s %10s %10s %10s %11s %11s | %9s %9s %9s" % ("month", "at risk", "new prepd", "rate %", "cumul. %", "", "baseline",
                                                          "ML w/ age", "ML no age"))
    emit("  %-8s %10s %10s %10s %11s %11s | %9s %9s %9s" % ("", "", "", "", "", "", "implied %", "implied %", "implied %"))

    def implied(sums, h):
        den = 1.0 - sums[h - 1, I_PRE] / n - sums[h - 1, I_DEF] / n
        return 100.0 * (sums[h, I_PRE] - sums[h - 1, I_PRE]) / n / den if den > 0 else float("nan")

    cum = 0
    series = []
    for k in range(1, MONTHS + 1):
        m = hz.CUT + k
        at_risk = int((((am < 0) | (am >= m)) & (last >= m - 1)).sum())
        new = int(((ast == I_PRE) & (am == m)).sum())
        cum += new
        series.append((new, at_risk, cum))
        emit("  %-8s %10s %10s %10.4f %11.4f %11s | %9.4f %9.4f %9.4f" % (
            hz.ym(m), f(at_risk), f(new), 100.0 * new / at_risk if at_risk else float("nan"), 100.0 * cum / n, "",
            implied(base_sum, k), implied(sums_a, k), implied(sums_b, k)))
    for h in HS[1:]:                                    # the series must agree with the horizon tables
        if series[h - 1][2] != int((real[h] == I_PRE).sum()):
            raise SystemExit("h=%d: cumulative Prepaid in the series (%d) != the horizon table (%d)" % (
                h, series[h - 1][2], int((real[h] == I_PRE).sum())))
    emit("  cumulative Prepaid by month equals the horizon tables' Prepaid counts at h = %s: yes" % ", ".join(map(str, HS[1:])))
    emit()

    # ---- plain statements ---------------------------------------------------------------------------------
    rule("=")
    emit("6. WHICH FORECAST IS CLOSER TO REALISED, state by state   (smallest |forecast share - realised share|, in pp)")
    rule("=")
    names = {"baseline": "baseline", "with": "ML with loan_age", "without": "ML without loan_age"}
    for h in HS[1:]:
        emit("h = %d (%s)" % (h, hz.ym(hz.CUT + h)))
        for i, s in enumerate(STATES):
            real_pct = 100.0 * rows[h]["realised"][i] / n
            g = {k: 100.0 * (rows[h][k][i] - rows[h]["realised"][i]) / n for k in names}
            ranked = sorted(g, key=lambda k: abs(g[k]))
            emit("  %-8s realised %8.4f%%   closest: %-20s (%+.4f pp);  then %s (%+.4f pp);  then %s (%+.4f pp)" % (
                s, real_pct, names[ranked[0]], g[ranked[0]], names[ranked[1]], g[ranked[1]], names[ranked[2]],
                g[ranked[2]]))
        emit()
    emit("These are numbers, not a ranking of the arms: no aggregate across states or horizons is computed, no feature set is")
    emit("recommended, and no log-loss, Brier or other selection metric appears anywhere in this report.")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
