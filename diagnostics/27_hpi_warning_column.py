"""Inspect the Warning column of the state HPI file.   READ-ONLY.

    python diagnostics/27_hpi_warning_column.py   # from the repo root

Writes diagnostics/27_hpi_warning_column.txt and nothing else. It reads
data/hpi_po_state.xlsx only; it does not rebuild the series, join the panel,
count loans, drop or filter anything, or run any git command. Sizing exercise.

Warning values are quoted verbatim (repr), never normalised.
"""

import os
import re
import sys
import textwrap
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

sys.path.insert(0, os.getcwd())
from src import hpi_state_monthly as h                                  # noqa: E402

OUT = os.path.join("diagnostics", "27_hpi_warning_column.txt")
RULE_W = 110
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def qlabel(r):
    return f"{int(r.yr)}Q{int(r.qtr)}"


def window(df, y0, y1):
    return df[(df.yr >= y0) & (df.yr <= y1)]


def sizing(w, label):
    n, k = len(w), int(w.warned.sum())
    emit(f"{label}: state-quarters {n}   warned {k}   {100 * k / n:.2f}%   "
         f"states warned {w.loc[w.warned, 'state'].nunique()} of {w.state.nunique()}")


def per_state(w):
    t = w.groupby("state").agg(quarters=("warned", "size"), warned=("warned", "sum")).reset_index()
    t["pct"] = 100 * t.warned / t.quarters
    t = t.sort_values(["warned", "state"], ascending=[False, True])
    emit(f"{'state':<7}{'warned':>8}{'quarters':>10}{'pct':>9}")
    for r in t.itertuples(index=False):
        emit(f"{r.state:<7}{int(r.warned):>8}{r.quarters:>10}{r.pct:>8.1f}%")


def per_year(w, with_total=False):
    emit(f"{'year':<6}{'warned':>8}" + (f"{'quarters':>10}{'pct':>9}{'states warned':>15}" if with_total else ""))
    for yr, g in w.groupby("yr"):
        k = int(g.warned.sum())
        line = f"{yr:<6}{k:>8}"
        if with_total:
            line += f"{len(g):>10}{100 * k / len(g):>8.1f}%{g.loc[g.warned, 'state'].nunique():>15}"
        emit(line)


def concentration(w, label):
    ws = sorted(w.loc[w.warned, "state"].unique())
    if not ws:
        emit(f"{label}: no state is warned in this window.")
        return
    for s in ws:
        qs = w[(w.state == s) & w.warned].sort_values(["yr", "qtr"])
        emit(f"{s}  ({len(qs)} warned quarters)")
        emit(textwrap.fill("  ".join(qlabel(r) for r in qs.itertuples(index=False)),
                           width=RULE_W - 6, initial_indent="    ", subsequent_indent="    "))


def main():
    df = h.load_state_quarterly()
    blank = df["Warning"].isna() | df["Warning"].astype(str).str.strip().eq("")
    df["warned"] = ~blank
    emit("HPI state file: Warning column")
    emit(f"file {os.path.abspath(h.STATE_XLSX)}   shape {df.shape}")

    # ------------------------------------------------------------- what's in it
    title("1. WHAT IS IN THE WARNING COLUMN  (whole file, values verbatim via repr)")
    vc = df["Warning"].map(repr).value_counts(dropna=False)
    emit(f"{'count':>7} {'share':>8}  value")
    for v, c in vc.items():
        emit(f"{c:>7} {100 * c / len(df):>7.2f}%  {v}")
    emit()
    emit(f"distinct values (null counted as a value): {len(vc)}   "
         f"distinct non-null: {df.loc[df.warned, 'Warning'].nunique()}")
    emit(f"blank/null: {int(blank.sum())} of {len(df)} = {100 * blank.mean():.2f}%   "
         f"(null = empty cell; whitespace-only strings also counted as blank: "
         f"{int((df['Warning'].notna() & blank).sum())})")
    emit(f"non-blank: {int(df.warned.sum())} = {100 * df.warned.mean():.2f}%")

    # ----------------------------------------------------------- documentation
    title("2. DOCUMENTATION INSIDE THE WORKBOOK")
    with zipfile.ZipFile(h.STATE_XLSX) as z:
        names = z.namelist()
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = [s.get("name") for s in wb.find("m:sheets", h._NS)]
        defined = wb.find("m:definedNames", h._NS)
        shared = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", h._NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % h._NS["m"])))
        sheet_xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "replace")
    emit(f"sheets in workbook: {sheets}")
    emit(f"package parts: {', '.join(names)}")
    comment_parts = [n for n in names if re.search(r"comment|drawing|vml|legend|note", n, re.I)]
    emit(f"comment/drawing/note parts: {comment_parts if comment_parts else 'none'}")
    emit(f"defined names: {[d.get('name') for d in defined] if defined is not None else 'none'}")
    _, rows = h.read_xlsx_sheet(h.STATE_XLSX)
    wide = [i + 1 for i, r in enumerate(rows) if len(r) > 6]
    emit(f"sheet rows {len(rows)}; header {rows[0]}; rows with data beyond column F: "
         f"{wide if wide else 'none'}")
    odd = [i + 1 for i, r in enumerate(rows[1:], 1) if not r or r[0] is None or not re.fullmatch(r"[A-Z]{2}", r[0])]
    emit(f"rows whose first cell is not a two-letter state code (possible notes/footer rows): "
         f"{odd if odd else 'none'}")
    hf = re.search(r"<headerFooter[^>]*?(/>|>(.*?)</headerFooter>)", sheet_xml, re.S)
    hf_text = bool(hf and hf.group(2) and hf.group(2).strip())
    emit(f"header/footer text in the sheet: {'PRESENT' if hf_text else 'none (headerFooter element is empty or absent)'}")
    emit("defined names above are bare names (IQ_*); they carry no explanatory text")
    emit(f"every string stored in the workbook ({len(shared)}):")
    for s in shared:
        emit(f"    {s!r}")
    non_data = [s for s in shared if not re.fullmatch(r"[A-Z]{2}|state|yr|qtr|index_nsa|index_sa|Warning", s)]
    emit()
    if len(sheets) == 1 and not comment_parts and not odd and not wide:
        emit("RESULT: the workbook has NO notes, legend or readme sheet and no comments; the only "
             "explanatory text anywhere in it is")
        emit("the Warning cell value(s) quoted in section 1. No further definition of the flag is "
             "present, and none is inferred here.")
    else:
        emit("RESULT: possible documentation found - see the lines above. Non-code, non-header strings:")
        for s in non_data:
            emit(f"    {s!r}")

    # ---------------------------------------------------- window 2004-01..2014-12
    w = window(df, 2004, 2014)
    title("3. WINDOW 2004-01 .. 2014-12  (2004Q1 .. 2014Q4)")
    sizing(w, "window")
    emit()
    emit("Per state, worst first (percentage of that state's quarters in the window):")
    per_state(w)
    emit()
    emit("Per year, warned quarters across all states:")
    per_year(w, with_total=True)

    title("4. CONCENTRATION IN THE WINDOW  (warned quarters of each warned state)")
    if w.warned.any():
        emit(f"{w.loc[w.warned, 'state'].nunique()} state(s) carry at least one warning in the window.")
    else:
        emit("HEADLINE: no state is warned in 2004-2014.")
    concentration(w, "window")

    # ------------------------------------------------------------ crisis window
    c = window(df, 2007, 2010)
    title("5. CRISIS WINDOW 2007-01 .. 2010-12  (2007Q1 .. 2010Q4)")
    sizing(c, "crisis window")
    emit()
    emit("Per state, worst first:")
    per_state(c)
    emit()
    emit("Per year:")
    per_year(c, with_total=True)
    emit()
    concentration(c, "crisis window")

    # ------------------------------------------------------------- whole series
    title("6. WHOLE-SERIES CONTEXT  (1991Q1 .. 2026Q2)")
    sizing(df, "whole file")
    emit()
    per_year(df, with_total=True)
    emit()
    s = df.groupby("state").warned.agg(["sum", "size"])
    full, part = s[s["sum"] == s["size"]].index.tolist(), s[(s["sum"] > 0) & (s["sum"] < s["size"])].index.tolist()
    none = s[s["sum"] == 0].index.tolist()
    emit(f"states warned in EVERY quarter ({len(full)}): {' '.join(full) if full else 'none'}")
    part_txt = " ".join("%s(%d/%d)" % (x, s.loc[x, "sum"], s.loc[x, "size"]) for x in part)
    emit(f"states warned in SOME but not all quarters ({len(part)}): {part_txt if part else 'none'}")
    emit(f"states never warned ({len(none)}): {' '.join(none)}")
    emit()
    first = df[df.warned].groupby("state").apply(lambda g: qlabel(g.sort_values(['yr', 'qtr']).iloc[0]),
                                                 include_groups=False)
    last = df[df.warned].groupby("state").apply(lambda g: qlabel(g.sort_values(['yr', 'qtr']).iloc[-1]),
                                                include_groups=False)
    if len(first):
        emit("first and last warned quarter, per warned state:")
        for st in first.index:
            emit(f"    {st}  {first[st]} .. {last[st]}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
