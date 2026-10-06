"""Report on the state monthly HPI bridge.   READ-ONLY.

    python diagnostics/26_hpi_bridge_report.py   # from the repo root, after
                                                 # python -m src.hpi_state_monthly

Writes diagnostics/26_hpi_bridge_report.txt and nothing else. It does not touch the
panel or any tracked file and runs no git command.

The anchor checks re-read the quarterly state file and the written CSVs rather
than reusing the build's in-memory frames, so a bug in the build cannot verify
itself.

Sections: 0 inputs, 1 division mapping, 2 coverage per state, 3 anchor
verification, 4 gap distribution, 5 largest gaps, 6 gap by calendar year,
7 quarter-boundary discontinuity.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import hpi_state_monthly as h                                  # noqa: E402

OUT = os.path.join("diagnostics", "26_hpi_bridge_report.txt")
RULE_W = 110
TOL = 1e-9          # relative tolerance for the anchor identity
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def rule(char="-"):
    emit(char * RULE_W)


def title(text):
    emit()
    rule("=")
    emit(text)
    rule("=")


def dist(x):
    """mean / median / p90 / p99 / max as a dict."""
    x = pd.Series(x)
    return {"mean": x.mean(), "median": x.median(), "p90": x.quantile(0.90),
            "p99": x.quantile(0.99), "max": x.max()}


def dist_line(label, x, unit, fmt="{:10.4f}"):
    d = dist(x)
    emit(f"  {label:<44}" + "".join(f"{k:>8}={fmt.format(v)}" for k, v in d.items()) + f"  {unit}")


def main():
    state_raw = h.load_state_quarterly()
    div_wide, div_long = h.load_division_monthly()
    out = pd.read_csv(h.OUT_CSV, parse_dates=["month"])
    mapping = pd.read_csv(h.DIV_CSV)
    assert list(out.columns) == ["state_code", "month", "hpi_step", "hpi_disagg"]

    emit("State monthly HPI bridge report   (seasonally adjusted, purchase-only)")
    emit(f"series  {h.OUT_CSV}   rows {len(out)}")

    # ---------------------------------------------------------------- 0 inputs
    title("0. INPUT INSPECTION")
    for line in h.describe_inputs(state_raw, div_wide):
        emit(line)
    emit()
    emit("Seasonal adjustment confirmed from file contents: state file has index_nsa and index_sa "
         "columns (index_sa used;")
    emit(f"  equal to index_nsa on {(state_raw.index_nsa == state_raw.index_sa).mean():.2%} of rows); "
         "division file has an '(SA)' and '(NSA)' column per division (SA used).")

    # ------------------------------------------------------------- 1 mapping
    title("1. DIVISION MAPPING  (data/macro/census_divisions.csv)")
    emit(f"rows {len(mapping)}   distinct divisions {mapping.division.nunique()}   "
         f"distinct states {mapping.state_code.nunique()}")
    assert len(mapping) == 51 and mapping.division.nunique() == 9
    assert mapping.equals(h.division_table()), "csv differs from the specified mapping"
    for d, g in mapping.groupby("division", sort=False):
        emit(f"  {d:<20}{len(g):>3}   {' '.join(g.state_code)}")
    emit()
    emit("  state_code,division")
    for r in mapping.itertuples(index=False):
        emit(f"  {r.state_code},{r.division}")
    emit()
    emit(f"Territories GU/PR/VI in mapping: {sorted(set(mapping.state_code) & {'GU', 'PR', 'VI'})} "
         "(none assigned a division, by design)")
    emit(f"HPI-file states without a division: {sorted(set(state_raw.state) - set(mapping.state_code))}")

    # -------------------------------------------------------------- 2 coverage
    title("2. DATE COVERAGE AND ROW COUNT PER STATE")
    emit(f"{'state':<7}{'first':>9}{'last':>9}{'rows':>7}{'expected':>10}{'missing':>9}  flag")
    flagged = []
    for s, g in out.groupby("state_code"):
        exp = pd.date_range(g.month.min(), g.month.max(), freq="MS")
        miss = exp.difference(g.month)
        flag = "" if len(miss) == 0 else f"GAP: {len(miss)} months, first {miss[0]:%Y-%m}"
        if flag:
            flagged.append(s)
        emit(f"{s:<7}{g.month.min():%Y-%m}  {g.month.max():%Y-%m}{len(g):>7}{len(exp):>10}{len(miss):>9}  {flag}")
    first, last = out.groupby("state_code").month.min(), out.groupby("state_code").month.max()
    emit()
    emit(f"states with gaps: {flagged if flagged else 'none'}")
    emit(f"required window {h.COVER_FROM:%Y-%m}..{h.COVER_TO:%Y-%m} covered by every state: "
         f"{bool((first <= h.COVER_FROM).all() and (last >= h.COVER_TO).all())}")
    emit(f"states: {out.state_code.nunique()}   total rows: {len(out)}   "
         f"duplicate state-months: {int(out.duplicated(['state_code', 'month']).sum())}")

    # ------------------------------------------------------- 3 anchor identity
    title("3. ANCHOR VERIFICATION  (mean of three months == quarterly anchor)")
    out["yr"], out["qtr"] = out.month.dt.year, out.month.dt.quarter
    anchor = state_raw[["state", "yr", "qtr", "index_sa"]].rename(
        columns={"state": "state_code", "index_sa": "anchor"})
    g = out.groupby(["state_code", "yr", "qtr"]).agg(
        n=("month", "size"), mean_step=("hpi_step", "mean"), mean_disagg=("hpi_disagg", "mean"),
        min_step=("hpi_step", "min"), max_step=("hpi_step", "max")).reset_index()
    g = g.merge(anchor, on=["state_code", "yr", "qtr"], how="outer", indicator=True)
    emit(f"state-quarters in quarterly file: {len(anchor)}   in output: {int(g.n.notna().sum())}")
    emit(f"state-quarters without exactly 3 months: {int((g.n != 3).sum())}   "
         f"in quarterly file but not output: {int((g._merge == 'right_only').sum())}   "
         f"in output but not file: {int((g._merge == 'left_only').sum())}")
    both = g[g._merge == "both"].copy()
    both["err_disagg"] = (both.mean_disagg - both.anchor).abs() / both.anchor
    both["err_step"] = (both.mean_step - both.anchor).abs() / both.anchor
    both["step_const"] = both.min_step == both.max_step
    emit(f"tolerance: relative {TOL:g}")
    for name, col in (("hpi_disagg", "err_disagg"), ("hpi_step", "err_step")):
        bad = both[both[col] > TOL]
        emit(f"  {name:<11} checked {len(both)}   max relative error {both[col].max():.3e}   "
             f"FAILURES {len(bad)}")
        if len(bad):
            emit(bad.head(20)[["state_code", "yr", "qtr", "anchor", "mean_step", "mean_disagg"]].to_string(index=False))
    emit(f"  hpi_step constant within every quarter: {bool(both.step_const.all())}")
    # lag check: step must equal the quarterly value of the SAME quarter
    chk = out.merge(anchor, on=["state_code", "yr", "qtr"], how="left")
    emit(f"  hpi_step == same-quarter anchor on every month (no lag): "
         f"{bool(np.isclose(chk.hpi_step, chk.anchor, rtol=1e-12, atol=0).all())}")

    # --------------------------------------------------------- 4 gap distribution
    out["gap"] = out.hpi_disagg - out.hpi_step
    out["abs_gap"] = out.gap.abs()
    out["abs_gap_pct"] = 100 * out.abs_gap / out.hpi_step
    title("4. |hpi_disagg - hpi_step|  DISTRIBUTION  (all state-months)")
    dist_line("absolute gap (index points)", out.abs_gap, "index pts")
    dist_line("absolute gap, % of index level (hpi_step)", out.abs_gap_pct, "% of level")
    emit(f"  mean signed gap {out.gap.mean():+.6f} index pts  (should be ~0: months average to the anchor)")

    # ------------------------------------------------------------- 5 top 20
    title("5. 20 LARGEST GAPS  (by absolute gap, index points)")
    q = state_raw[["state", "yr", "qtr", "index_sa"]].sort_values(["state", "yr", "qtr"]).copy()
    q["prev"] = q.groupby("state").index_sa.shift(1)
    q["qq_move_pct"] = 100 * (q.index_sa / q.prev - 1)
    q = q.rename(columns={"state": "state_code"})
    top = out.sort_values("abs_gap", ascending=False).head(20).merge(
        q[["state_code", "yr", "qtr", "qq_move_pct"]], on=["state_code", "yr", "qtr"], how="left")
    emit("quarter move = state quarterly value vs the previous quarter, %; blank for a state's first quarter")
    emit(f"{'#':>2} {'state':<6}{'month':<9}{'hpi_step':>10}{'hpi_disagg':>12}{'gap':>9}{'gap %':>8}{'qtr move %':>12}")
    for i, r in enumerate(top.itertuples(index=False), 1):
        mv = "" if pd.isna(r.qq_move_pct) else f"{r.qq_move_pct:+.2f}"
        emit(f"{i:>2} {r.state_code:<6}{r.month:%Y-%m}  {r.hpi_step:>10.3f}{r.hpi_disagg:>12.3f}"
             f"{r.gap:>+9.3f}{r.abs_gap_pct:>8.3f}{mv:>12}")

    # ------------------------------------------------------------ 6 by year
    title("6. GAP BY CALENDAR YEAR  (2008-2009 marked)")
    emit(f"{'year':<6}{'rows':>7}{'mean |gap|':>12}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}"
         f"{'mean % lvl':>12}{'max % lvl':>11}")
    for yr, gg in out.groupby(out.month.dt.year):
        mark = "  <-- 2008-09 window" if yr in (2008, 2009) else ""
        a = gg.abs_gap
        emit(f"{yr:<6}{len(gg):>7}{a.mean():>12.4f}{a.median():>9.4f}{a.quantile(.9):>9.4f}"
             f"{a.quantile(.99):>9.4f}{a.max():>9.4f}{gg.abs_gap_pct.mean():>12.4f}{gg.abs_gap_pct.max():>11.4f}{mark}")
    w = out[out.month.dt.year.isin([2008, 2009])]
    r = out[~out.month.dt.year.isin([2008, 2009])]
    emit()
    emit(f"2008-2009 vs all other years, mean |gap| % of level: {w.abs_gap_pct.mean():.4f} vs {r.abs_gap_pct.mean():.4f}"
         f"   (ratio {w.abs_gap_pct.mean() / r.abs_gap_pct.mean():.2f}x)")

    # ------------------------------------------------- 7 boundary discontinuity
    title("7. QUARTER-BOUNDARY DISCONTINUITY IN hpi_disagg")
    emit("Boundary move = log change from the last month of a quarter to the first month of the next, |%|.")
    emit("Within move   = |log change| between consecutive months inside a quarter (m1->m2, m2->m3).")
    emit("Division move = the same boundary change in the division's own SA series (what a faithful path does).")
    emit("Artificial    = |disagg boundary change - division boundary change|: the jump the method adds.")
    o = out.merge(mapping, on="state_code").sort_values(["state_code", "month"]).reset_index(drop=True)
    o = o.merge(div_long.rename(columns={"hpi": "div"}), on=["division", "month"], how="left")
    o["l_dis"], o["l_step"], o["l_div"] = np.log(o.hpi_disagg), np.log(o.hpi_step), np.log(o["div"])
    for c in ("dis", "step", "div"):
        o[f"d_{c}"] = 100 * o.groupby("state_code")[f"l_{c}"].diff()          # change INTO this month
    contiguous = o.groupby("state_code").month.diff().dt.days.between(28, 31)
    first_of_q = o.month.dt.month.isin([1, 4, 7, 10])
    bnd = o[first_of_q & contiguous & o.d_dis.notna()].copy()
    win = o[~first_of_q & contiguous & o.d_dis.notna()].copy()
    bnd["artificial"] = (bnd.d_dis - bnd.d_div).abs()
    emit(f"boundaries measured {len(bnd)}   within-quarter month pairs {len(win)}")
    emit()
    for tag, mask_b, mask_w in (
            ("ALL YEARS", slice(None), slice(None)),
            ("2008-2009 only", bnd.month.dt.year.isin([2008, 2009]), win.month.dt.year.isin([2008, 2009]))):
        b, wi = bnd[mask_b], win[mask_w]
        emit(f"{tag}   (boundaries {len(b)}, within pairs {len(wi)})")
        dist_line("within-quarter |month move|, disagg", wi.d_dis.abs(), "% per month")
        dist_line("boundary |move|, disagg", b.d_dis.abs(), "% per month")
        dist_line("boundary |move|, division (reference)", b.d_div.abs(), "% per month")
        dist_line("boundary |artificial jump|", b.artificial, "% per month")
        dist_line("boundary |move|, step (reference)", b.d_step.abs(), "% per month")
        ratio = b.d_dis.abs().median() / wi.d_dis.abs().median()
        emit(f"  median boundary |move| / median within |move| (disagg): {ratio:.2f}x;   "
             f"artificial jump / median within |move|: {b.artificial.median() / wi.d_dis.abs().median():.2f}x "
             f"(median), {b.artificial.quantile(.99) / wi.d_dis.abs().quantile(.99):.2f}x (p99 vs p99)")
        emit(f"  share of boundaries whose artificial jump exceeds the median within-quarter move: "
             f"{(b.artificial > wi.d_dis.abs().median()).mean():.1%}")
        emit()
    emit("Read: 'artificial' is the departure of the bridged boundary step from the division's own")
    emit("boundary step. The method forces each quarter to average to the anchor, so the level of one")
    emit("quarter's last month and the next quarter's first month are scaled by different anchors.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
