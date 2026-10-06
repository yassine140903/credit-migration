"""State-level monthly HPI series, built from FHFA quarterly state and monthly
division purchase-only indexes (seasonally adjusted).   Standalone macro prep.

    python -m src.hpi_state_monthly        # from the repo root

Reads   data/hpi_po_state.xlsx          quarterly, States
        data/hpi_po_monthly_hist.xlsx   monthly, Census Divisions and U.S.
Writes  data/macro/census_divisions.csv     state_code, division (51 rows)
        data/macro/hpi_state_monthly.csv    state_code, month, hpi_step, hpi_disagg

It does not read or modify the panel and runs no git command.

SERIES
  hpi_step    the state's quarterly value repeated over the three months of that
              same calendar quarter (Q1 -> Jan, Feb, Mar). No lag.
  hpi_disagg  state quarterly value * (division value in month) / (mean of the
              division's three values in that quarter). The three months average
              exactly to the quarterly anchor and follow the division's
              within-quarter movement. Division LEVELS are never used.

SEASONAL ADJUSTMENT is confirmed from the file contents, never the filename:
the state sheet must carry both an index_nsa and an index_sa column (the SA one
is used, and must differ from NSA); the monthly sheet must carry a "(SA)" and a
"(NSA)" column for each division (the SA ones are used, and must differ from NSA).

FAILS LOUDLY: a state with no division, a division month missing where a state
quarter needs it, a non-numeric / non-positive index, or coverage short of
2004-01 .. 2014-12 for any state. No national or division value is ever used to
fill a state gap.

The xlsx files are read with the standard library (zip + XML), so no Excel
engine (openpyxl) is required.
"""

import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import numpy as np
import pandas as pd

STATE_XLSX = os.path.join("data", "hpi_po_state.xlsx")
DIV_XLSX = os.path.join("data", "hpi_po_monthly_hist.xlsx")
MACRO_DIR = os.path.join("data", "macro")
DIV_CSV = os.path.join(MACRO_DIR, "census_divisions.csv")
OUT_CSV = os.path.join(MACRO_DIR, "hpi_state_monthly.csv")

COVER_FROM = pd.Timestamp("2004-01-01")
COVER_TO = pd.Timestamp("2014-12-01")

# The mapping is fixed by the task. Do not infer or extend it. US territories
# (GU, PR, VI) belong to no census division and are deliberately absent.
DIVISIONS = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


class HpiError(RuntimeError):
    pass


# ----------------------------------------------------------------- xlsx reader

def _col_index(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_sheet(path):
    """Return (sheet_name, rows) for the single sheet of an xlsx; each row is a
    list of strings (None where the cell is empty). Fails if not one sheet."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
        sheets = ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS)
        if len(sheets) != 1:
            raise HpiError(f"{path}: expected exactly one sheet, found {len(sheets)}")
        rels = {r.get("Id"): r.get("Target")
                for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        target = rels[sheets[0].get("{%s}id" % _NS["r"])].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).find("m:sheetData", _NS):
            cells = {}
            for c in row:
                v, kind = c.find("m:v", _NS), c.get("t")
                if kind == "inlineStr":
                    cells[_col_index(c.get("r"))] = "".join(
                        t.text or "" for t in c.iter("{%s}t" % _NS["m"]))
                elif v is not None:
                    cells[_col_index(c.get("r"))] = shared[int(v.text)] if kind == "s" else v.text
            rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
    return sheets[0].get("name"), rows


# --------------------------------------------------------------------- loaders

def load_state_quarterly(path=STATE_XLSX):
    """Raw state sheet as a DataFrame (all columns), SA confirmed from its columns."""
    _, rows = read_xlsx_sheet(path)
    header = rows[0]
    df = pd.DataFrame([r + [None] * (len(header) - len(r)) for r in rows[1:]], columns=header)
    for need in ("state", "yr", "qtr", "index_nsa", "index_sa"):
        if need not in df.columns:
            raise HpiError(f"{path}: column {need!r} missing; cannot confirm seasonal "
                           f"adjustment. Columns: {list(df.columns)}")
    for c in ("yr", "qtr", "index_nsa", "index_sa"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    if df[["yr", "qtr", "index_nsa", "index_sa"]].isna().any().any():
        raise HpiError(f"{path}: null yr/qtr/index values")
    share_equal = float((df["index_nsa"] == df["index_sa"]).mean())
    if share_equal > 0.5:
        raise HpiError(f"{path}: index_sa equals index_nsa on {share_equal:.1%} of rows; "
                       "index_sa does not look seasonally adjusted")
    if not df["qtr"].isin([1, 2, 3, 4]).all():
        raise HpiError(f"{path}: qtr outside 1..4")
    df["yr"], df["qtr"] = df["yr"].astype(int), df["qtr"].astype(int)
    return df


def load_division_monthly(path=DIV_XLSX):
    """Wide monthly sheet -> (raw wide DataFrame, long SA DataFrame
    [division, month, hpi]). SA columns are identified by their '(SA)' label."""
    _, rows = read_xlsx_sheet(path)
    h = next((i for i, r in enumerate(rows) if r and r[0] == "Month"), None)
    if h is None:
        raise HpiError(f"{path}: no header row starting with 'Month'")
    header = [re.sub(r"\s+", " ", c).strip() for c in rows[h]]   # 'Pacific\n\n(SA)' -> 'Pacific (SA)'
    body = [r + [None] * (len(header) - len(r)) for r in rows[h + 1:] if r and r[0]]
    wide = pd.DataFrame(body, columns=header)
    base = date(1899, 12, 30)                                    # Excel serial origin
    wide["Month"] = [pd.Timestamp(base + timedelta(days=int(float(v)))) for v in wide["Month"]]
    if not (wide["Month"].dt.day == 1).all():
        raise HpiError(f"{path}: Month values are not all first-of-month")
    for c in header[1:]:
        wide[c] = pd.to_numeric(wide[c], errors="raise")
    sa = {c[: -len(" (SA)")]: c for c in header if c.endswith(" (SA)")}
    nsa = {c[: -len(" (NSA)")] for c in header if c.endswith(" (NSA)")}
    if not sa or set(sa) != nsa:
        raise HpiError(f"{path}: cannot confirm SA/NSA column pairs from labels: {header}")
    for name in sa:
        if (wide[f"{name} (SA)"] == wide[f"{name} (NSA)"]).mean() > 0.5:
            raise HpiError(f"{path}: '{name} (SA)' equals its NSA column; not seasonally adjusted")
    long = pd.concat([pd.DataFrame({"division": name, "month": wide["Month"], "hpi": wide[col]})
                      for name, col in sa.items() if name != "USA"], ignore_index=True)
    return wide, long


def division_table():
    return pd.DataFrame([(s, d) for d, ss in DIVISIONS.items() for s in ss],
                        columns=["state_code", "division"])


# ---------------------------------------------------------------------- step 0

def describe_inputs(state_raw, div_wide):
    """Step 0 lines: path, shape, columns, distinct geography values."""
    out = []
    out.append(f"STATE FILE  {os.path.abspath(STATE_XLSX)}")
    out.append(f"  shape    {state_raw.shape}")
    out.append(f"  columns  {list(state_raw.columns)}")
    geo = sorted(state_raw["state"].unique())
    out.append(f"  geography identifier 'state': {len(geo)} distinct: {' '.join(geo)}")
    out.append(f"  quarters {int(state_raw.yr.min())}Q{int(state_raw.loc[state_raw.yr == state_raw.yr.min(), 'qtr'].min())}"
               f" .. {int(state_raw.yr.max())}Q{int(state_raw.loc[state_raw.yr == state_raw.yr.max(), 'qtr'].max())}")
    two_letter = all(isinstance(g, str) and re.fullmatch(r"[A-Z]{2}", g) for g in geo)
    out.append(f"  format   two-letter uppercase codes: {two_letter}  "
               f"(panel property_state is two-letter text: "
               f"{'MATCH, no coercion needed' if two_letter else 'MISMATCH - report, do not coerce'})")
    out.append("  note     index_sa vs index_nsa both present; index_sa is used")
    out.append("")
    out.append(f"DIVISION FILE  {os.path.abspath(DIV_XLSX)}")
    out.append(f"  shape    {div_wide.shape}")
    out.append(f"  columns  {list(div_wide.columns)}")
    names = sorted({c.rsplit(" (", 1)[0] for c in div_wide.columns[1:]})
    out.append(f"  geography identifier (column-label prefix): {len(names)} distinct: {'; '.join(names)}")
    out.append(f"  months   {div_wide['Month'].min():%Y-%m} .. {div_wide['Month'].max():%Y-%m} "
               f"({len(div_wide)} rows)")
    out.append("  format   division NAMES, not state codes; joined to states through "
               "data/macro/census_divisions.csv")
    out.append("")
    out.append("PANEL-SIDE NOTE  GU, PR, VI may appear as property_state in the panel; the HPI "
               "file has none of them and the mapping must not assign them a division.")
    return out


# ---------------------------------------------------------------------- build

def build(state_raw, div_long, mapping):
    """Return the monthly state frame [state_code, month, hpi_step, hpi_disagg]."""
    # -- step 3: every HPI state must have a division
    hpi_states = sorted(state_raw["state"].unique())
    unmapped = sorted(set(hpi_states) - set(mapping["state_code"]))
    if unmapped:
        raise HpiError(f"states in the HPI file with no division: {unmapped}")
    missing_div = sorted(set(mapping["division"]) - set(div_long["division"]))
    if missing_div:
        raise HpiError(f"mapped divisions with no monthly series: {missing_div}")

    q = state_raw[["state", "yr", "qtr", "index_sa"]].rename(
        columns={"state": "state_code", "index_sa": "q_hpi"})
    if q.duplicated(["state_code", "yr", "qtr"]).any():
        raise HpiError("duplicate state-quarter rows in state file")
    if (q["q_hpi"] <= 0).any() or (div_long["hpi"] <= 0).any():
        raise HpiError("non-positive index value")
    q = q.merge(mapping, on="state_code", how="left", validate="m:1")

    # three months per state quarter, no lag
    q["q_first"] = pd.to_datetime(dict(year=q["yr"], month=3 * (q["qtr"] - 1) + 1, day=1))
    m = q.loc[q.index.repeat(3)].reset_index(drop=True)
    offset = np.tile([0, 1, 2], len(q))
    m["month"] = (m["q_first"].dt.to_period("M") + offset).dt.to_timestamp()

    m = m.merge(div_long, on=["division", "month"], how="left", validate="m:1")
    gaps = m[m["hpi"].isna()]
    if len(gaps):
        keys = gaps[["state_code", "division", "month"]].drop_duplicates()
        raise HpiError(f"{len(gaps)} state-months need a division month that is missing "
                       f"(no fallback is applied):\n{keys.head(30).to_string(index=False)}")

    qmean = m.groupby(["state_code", "yr", "qtr"])["hpi"].transform("mean")
    m["hpi_step"] = m["q_hpi"]
    m["hpi_disagg"] = m["q_hpi"] * m["hpi"] / qmean
    out = (m[["state_code", "month", "hpi_step", "hpi_disagg"]]
           .sort_values(["state_code", "month"]).reset_index(drop=True))

    # -- step 3: coverage 2004-01 .. 2014-12 for every state
    need = pd.date_range(COVER_FROM, COVER_TO, freq="MS")
    for s, g in out.groupby("state_code"):
        lacking = need.difference(g["month"])
        if len(lacking):
            raise HpiError(f"{s}: {len(lacking)} months missing inside "
                           f"{COVER_FROM:%Y-%m}..{COVER_TO:%Y-%m}, first {lacking[0]:%Y-%m}")
    if out.duplicated(["state_code", "month"]).any():
        raise HpiError("duplicate state-months in output")
    return out


def main():
    os.makedirs(MACRO_DIR, exist_ok=True)
    state_raw = load_state_quarterly()
    div_wide, div_long = load_division_monthly()

    print("=== STEP 0: inspection ===")
    print("\n".join(describe_inputs(state_raw, div_wide)))

    print("\n=== STEP 1: division mapping ===")
    mapping = division_table()
    assert len(mapping) == 51, len(mapping)
    assert mapping["division"].nunique() == 9
    assert mapping["state_code"].is_unique
    assert not set(mapping["state_code"]) & {"GU", "PR", "VI"}
    mapping.to_csv(DIV_CSV, index=False, lineterminator="\n")
    print(f"wrote {DIV_CSV}: {len(mapping)} rows, {mapping['division'].nunique()} divisions")

    print("\n=== STEP 2/3: build ===")
    out = build(state_raw, div_long, mapping)
    out.to_csv(OUT_CSV, index=False, lineterminator="\n", date_format="%Y-%m-%d")
    print(f"wrote {OUT_CSV}: {len(out)} rows, {out['state_code'].nunique()} states, "
          f"{out['month'].min():%Y-%m} .. {out['month'].max():%Y-%m}")


if __name__ == "__main__":
    try:
        main()
    except HpiError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
