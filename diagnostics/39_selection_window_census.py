"""Transition-cell census of the selection window.   READ-ONLY.

    python diagnostics/39_selection_window_census.py   # from the repo root

Writes diagnostics/39_selection_window_census.txt and nothing else. It reads the
panel through src.panel_io.load_panel, one partition at a time, five columns
(role, from_period, from_state, to_state, loan_id), and only role == 'fit' rows:
'condition' and 'evaluate' rows are filtered out at read time and never held. It
writes no data file, trains nothing, touches nothing in src/, runs no git command.

WINDOWS. Selection window: role == 'fit' and 200601 <= from_period <= 200706.
Full fit window: every role == 'fit' row. Its observed from_period range is
reported and compared with the expected 200512..200812; nothing is filtered by it.

LOANS. A loan lives in exactly one partition (asserted: every loan_id carries its
partition's vintage prefix), so per-partition distinct counts add to the total.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel                                                   # noqa: E402
from src.panel_io import load_panel, read_manifest                      # noqa: E402

OUT = os.path.join("diagnostics", "39_selection_window_census.txt")
WIN_LO, WIN_HI = 200601, 200706
FULL_LO, FULL_HI = 200512, 200812
STATES = list(panel.STATES)                 # Current DQ30 DQ60 DQ90+ Prepaid Default
FROM = STATES[:4]
THIN = 100
CORNER = "from/to"
GROUPS = [("Current alone", ["Current"]),
          ("DQ30+DQ60 combined", ["DQ30", "DQ60"]),
          ("DQ90+ alone", ["DQ90+"])]
RULE_W = 96
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def scan():
    """One pass over the partitions; returns the 4x6 grids and bookkeeping."""
    manifest = read_manifest()
    sel = np.zeros((len(FROM), len(STATES)), dtype=np.int64)
    full = np.zeros_like(sel)
    odd = {"sel": {}, "full": {}}               # (from, to) outside the 4x6 grid -> rows
    sel_loans = fit_rows = 0
    lo, hi = None, None
    for p in manifest["partitions"]:
        q = p["quarter"]
        d = load_panel(roles=["fit"], quarters=[q],
                       columns=["role", "from_period", "from_state", "to_state", "loan_id"])
        assert (d["role"].astype(str) == "fit").all()
        prefix = "F" + q[2:4] + q[4:]                       # 2007Q3 -> F07Q3
        assert d["loan_id"].str.startswith(prefix).all(), "a loan_id lacks its partition's vintage prefix"
        fit_rows += len(d)
        if len(d):
            lo = d["from_period"].min() if lo is None else min(lo, d["from_period"].min())
            hi = d["from_period"].max() if hi is None else max(hi, d["from_period"].max())
        d["from_state"] = d["from_state"].astype(object)
        d["to_state"] = d["to_state"].astype(object)
        in_win = ((d["from_period"] >= WIN_LO) & (d["from_period"] <= WIN_HI)).to_numpy()
        for tag, grid, sub in (("full", full, d), ("sel", sel, d[in_win])):
            ct = sub.groupby(["from_state", "to_state"], dropna=False).size()
            for (fs, ts), n in ct.items():
                if fs in FROM and ts in STATES:
                    grid[FROM.index(fs), STATES.index(ts)] += int(n)
                else:
                    odd[tag][(fs, ts)] = odd[tag].get((fs, ts), 0) + int(n)
        sel_loans += int(d.loc[in_win, "loan_id"].nunique())
        print(f"  scanned {q}: {len(d):,} fit rows, {int(in_win.sum()):,} in the selection window", flush=True)
        del d
    return sel, full, odd, sel_loans, fit_rows, (lo, hi), manifest


def grid_table(g, with_total=True):
    w = 12
    emit(f"{CORNER:<10}" + "".join(f"{s:>{w}}" for s in STATES) + (f"{'ROW TOTAL':>{w + 2}}" if with_total else ""))
    for i, f in enumerate(FROM):
        emit(f"{f:<10}" + "".join(f"{int(g[i, j]):>{w},}" for j in range(len(STATES)))
             + (f"{int(g[i].sum()):>{w + 2},}" if with_total else ""))
    if with_total:
        emit(f"{'COL TOTAL':<10}" + "".join(f"{int(g[:, j].sum()):>{w},}" for j in range(len(STATES)))
             + f"{int(g.sum()):>{w + 2},}")


def pct_table(g):
    w = 12
    emit(f"{CORNER:<10}" + "".join(f"{s:>{w}}" for s in STATES) + f"{'ROW TOTAL':>{w}}")
    for i, f in enumerate(FROM):
        t = g[i].sum()
        cells = [(100.0 * g[i, j] / t) if t else float("nan") for j in range(len(STATES))]
        emit(f"{f:<10}" + "".join(f"{c:>{w - 1}.4f}%" for c in cells)
             + (f"{sum(cells):>{w - 1}.4f}%" if t else f"{'n/a':>{w}}"))


def group_report(name, rows_idx, g):
    sub = g[rows_idx].sum(axis=0)
    present = [s for s, n in zip(STATES, sub) if n > 0]
    absent = [s for s, n in zip(STATES, sub) if n == 0]
    thin = [f"{s} ({int(n):,})" for s, n in zip(STATES, sub) if 0 < n < THIN]
    emit(f"{name}")
    emit(f"  training rows available : {int(sub.sum()):,}")
    emit("  per-destination counts  : " + "   ".join(f"{s} {int(n):,}" for s, n in zip(STATES, sub)))
    emit(f"  destination classes present ({len(present)} of {len(STATES)}): {', '.join(present) if present else 'none'}")
    emit(f"  classes it would LEARN  : {len(present)} of {len(STATES)}"
         + (f"   (never sees: {', '.join(absent)})" if absent else ""))
    if thin:
        emit(f"  present but under {THIN} rows : {', '.join(thin)}")
    emit()


def main():
    emit("Selection-window transition census   (fit rows only; panel validated against its manifest by load_panel)")
    sel, full, odd, sel_loans, fit_rows, (lo, hi), manifest = scan()
    emit(f"panel build {manifest['build_stamp']}   partitions {manifest['n_partitions']}   "
         f"manifest rows {manifest['total_rows']:,}")
    emit(f"selection window: role == 'fit' and {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"fit rows in the whole panel {fit_rows:,}; observed fit from_period range {lo}..{hi} "
         f"(expected {FULL_LO}..{FULL_HI}: {'MATCH' if (lo, hi) == (FULL_LO, FULL_HI) else 'DIFFERENT'})")
    assert int(full.sum()) + sum(odd["full"].values()) == fit_rows
    for tag, label in (("sel", "selection window"), ("full", "full fit window")):
        if odd[tag]:
            emit(f"WARNING {label}: rows outside the 4x6 grid (excluded from every table below): {odd[tag]}")
    sel_rows = int(sel.sum())

    title("1. FIT ROWS IN THE SELECTION WINDOW")
    emit(f"rows           {sel_rows:,}  (in-grid; plus {sum(odd['sel'].values()):,} outside the grid)")
    emit(f"distinct loans {sel_loans:,}")

    title("2. ROW COUNTS BY from_state  (selection window)")
    emit(f"{'from_state':<12}{'rows':>14}{'% of rows':>12}")
    for i, f in enumerate(FROM):
        emit(f"{f:<12}{int(sel[i].sum()):>14,}{(100.0 * sel[i].sum() / sel_rows if sel_rows else float('nan')):>11.4f}%")
    emit(f"{'TOTAL':<12}{sel_rows:>14,}")

    title("3. CONTINGENCY TABLE, RAW COUNTS  (selection window; rows = from_state, columns = to_state)")
    grid_table(sel)

    title("4. SAME TABLE, ROW-NORMALISED PERCENTAGES  (each from_state row sums to 100)")
    pct_table(sel)

    title("5. EMPTY CELLS  (count == 0, selection window)")
    empty = [(f, s) for i, f in enumerate(FROM) for j, s in enumerate(STATES) if sel[i, j] == 0]
    for f, s in empty:
        emit(f"{f} -> {s} : EMPTY")
    if not empty:
        emit("none: every one of the 24 cells has at least one row.")
    emit(f"({len(empty)} of {len(FROM) * len(STATES)} cells empty)")

    title(f"6. THIN CELLS  (0 < count < {THIN}, selection window)")
    thin = [(f, s, int(sel[i, j])) for i, f in enumerate(FROM) for j, s in enumerate(STATES)
            if 0 < sel[i, j] < THIN]
    for f, s, n in thin:
        emit(f"{f} -> {s} : {n:,}")
    if not thin:
        emit("none.")
    emit(f"({len(thin)} cells with 1..{THIN - 1} rows; empty cells are listed in section 5, not repeated here)")

    title("7. GROUPED ARCHITECTURE  (three models, one per origin group; selection window)")
    for name, members in GROUPS:
        group_report(name, [FROM.index(m) for m in members], sel)

    title("8. POOLED ARCHITECTURE  (one model, all four origin states together; selection window)")
    group_report("Pooled (Current + DQ30 + DQ60 + DQ90+)", list(range(len(FROM))), sel)

    title(f"9. FULL FIT WINDOW, RAW COUNTS  (every role == 'fit' row, observed {lo}..{hi})")
    grid_table(full)
    emit()
    emit("selection window as a share of the full fit window, by cell (selection / full):")
    emit(f"{CORNER:<10}" + "".join(f"{s:>12}" for s in STATES))
    for i, f in enumerate(FROM):
        emit(f"{f:<10}" + "".join(
            (f"{100.0 * sel[i, j] / full[i, j]:>11.2f}%" if full[i, j] else f"{'n/a':>12}")
            for j in range(len(STATES))))
    full_empty = [f"{f} -> {s}" for i, f in enumerate(FROM) for j, s in enumerate(STATES) if full[i, j] == 0]
    emit()
    emit("cells empty in the full fit window: " + (", ".join(full_empty) if full_empty else "none"))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
