"""Zero-balance effective date vs the reporting period of the row carrying the code.

For every loan that exits with a zero-balance code in {01,02,03,09,15,16,96},
this measures where the zero-balance effective date sits relative to the monthly
reporting period stamped on the row that carries the code:

    offset_months = (calendar month of effective date) - (reporting period)

in whole calendar months. The reporting period has no day component, so no
day-difference is computed across months; a day-of-month histogram is reported
for the offset_months == 0 rows only, and only from effective dates that
actually carry a day component. The raw character width of the effective-date
field is tabulated first so the availability of a day component is a reported
fact rather than an assumption.

COLUMN INDEX. The zero-balance effective date index is not hardcoded: it is read
out of docs/file_layout_july_2026.xlsx at run time, off the MONTHLY PERFORMANCE
sheet, by matching the attribute name and taking its published field position
minus one. The resolved 0-based index is printed in the output header. The
script aborts if the attribute cannot be located rather than falling back to a
guess.

UNITS. Sections A-D count LOANS, one row per loan: the row carrying the
zero-balance code. A loan carrying more than one such row is collapsed to its
latest-period code row; the number of loans that needed that tie-break is
reported in section D.

TWO PASSES, both chunked and order-independent:
    pass 1  [0,1,3,8,eff]  every row carrying a zero-balance code (a sparse
                           subset), collapsed to one row per loan
    pass 2  [0,1]          maximum reporting period per loan, needed by section
                           D to test whether the code-carrying row is the loan's
                           last row. A loan's rows may straddle chunk and file
                           boundaries, so this is a running max over all chunks,
                           never an inference from row adjacency.
"""

import glob
import os
import re
import time
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "12_zero_balance_date_offset.txt")
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_ZBC = 8

EFF_ATTRIBUTE = "zero balance effective date"
PERF_SHEET_MARKER = "monthly performance"

CODES = ["01", "02", "03", "09", "15", "16", "96"]
CREDIT_EVENT = ["02", "03", "09"]

ZBC_LABELS = {
    "01": "prepaid or matured (voluntary payoff)",
    "02": "third-party sale",
    "03": "short sale or charge-off",
    "06": "repurchase prior to property disposition",
    "09": "REO disposition",
    "15": "note sale",
    "16": "reperforming loan sale",
    "96": "removal (non-standard code, see 08_code96_disposition)",
}

MISSING = "effective date MISSING (blank)"
UNPARSEABLE = "effective date UNPARSEABLE"
BUCKET_NAMES = ["-3 and below", "-2", "-1", "0", "+1", "+2", "+3 and above"]
ALL_BUCKETS = BUCKET_NAMES + [MISSING, UNPARSEABLE]

BUCKET_DISPLAY = dict([(n, n) for n in BUCKET_NAMES]
                      + [(MISSING, "MISSING"), (UNPARSEABLE, "UNPARSEABLE")])

DAY_BINS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15),
            ("16-20", 16, 20), ("21-25", 21, 25), ("26-31", 26, 31)]

_LINES = []


def emit(line=""):
    print(line, flush=True)
    _LINES.append(line)


def write_out():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")


def rule(char="-", width=110):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den):
    return "n/a" if not den else "%.4f%%" % (100.0 * num / den)


# ---------------------------------------------------------------------------
# column index, resolved from the published file layout
# ---------------------------------------------------------------------------

def resolve_eff_index(path):
    """Return (0-based index, sheet, 1-based position, attribute text, n_matches).

    Reads the xlsx with the standard library only (no openpyxl dependency): the
    workbook is a zip of XML and shared strings are resolved by index.
    """
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root:
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

    def cell_text(c):
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(ns + "t"))
        v = c.find(ns + "v")
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    hits = []
    sheets = sorted(n for n in z.namelist()
                    if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    for name in sheets:
        root = ET.fromstring(z.read(name))
        sheet_is_perf = False
        for row in root.iter(ns + "row"):
            texts = [cell_text(c) for c in row.iter(ns + "c")]
            joined = " ".join(texts).lower()
            if PERF_SHEET_MARKER in joined:
                sheet_is_perf = True
            if EFF_ATTRIBUTE in joined:
                position = next((int(t.strip()) for t in texts
                                 if re.match(r"^\d+$", t.strip())), None)
                attribute = next((t for t in texts if EFF_ATTRIBUTE in t.lower()), "")
                hits.append((name, sheet_is_perf, position, attribute))

    perf_hits = [h for h in hits if h[1]]
    chosen = perf_hits[0] if perf_hits else (hits[0] if hits else None)
    if chosen is None or chosen[2] is None:
        raise SystemExit("could not locate %r with a numeric field position in %s"
                         % (EFF_ATTRIBUTE, path))
    name, _, position, attribute = chosen
    if not 1 <= position <= 40:
        raise SystemExit("implausible field position %r for %r in %s"
                         % (position, EFF_ATTRIBUTE, path))
    return position - 1, name, position, attribute, len(hits)


# ---------------------------------------------------------------------------
# date parsing
# ---------------------------------------------------------------------------

def parse_eff(raw):
    """Split raw effective-date strings into status, YYYYMM, day and raw width.

    Accepts YYYYMM (no day component) and YYYYMMDD. Blank is MISSING, anything
    else is UNPARSEABLE; neither is ever dropped.
    """
    raw = raw.str.strip()
    width = raw.str.len()
    status = pd.Series(UNPARSEABLE, index=raw.index, dtype=object)
    status[raw == ""] = MISSING
    ym = pd.Series(-1, index=raw.index, dtype="int64")
    day = pd.Series(-1, index=raw.index, dtype="int64")

    digits = raw.str.isdigit()
    for w in (6, 8):
        sel = (width == w) & digits
        if not sel.any():
            continue
        v = raw[sel]
        y = v.str[:4].astype("int64")
        m = v.str[4:6].astype("int64")
        ok = (y >= 1900) & (y <= 2100) & (m >= 1) & (m <= 12)
        if w == 8:
            d = v.str[6:8].astype("int64")
            ok &= (d >= 1) & (d <= 31)
        idx = v.index[ok]
        ym.loc[idx] = y[ok] * 100 + m[ok]
        if w == 8:
            day.loc[idx] = d[ok]
        status.loc[idx] = "ok"

    return pd.DataFrame({"status": status, "ym": ym, "day": day, "width": width})


def months_between(ym_eff, ym_per):
    """Whole calendar months from reporting period to effective date."""
    return ((ym_eff // 100 - ym_per // 100) * 12
            + (ym_eff % 100 - ym_per % 100))


def bucket_offsets(status, offset):
    """Vectorised bucket label per loan, covering every row exactly once."""
    o = offset.values
    ok = (status == "ok").values
    labels = np.select(
        [ok & (o <= -3), ok & (o == -2), ok & (o == -1), ok & (o == 0),
         ok & (o == 1), ok & (o == 2), ok & (o >= 3)],
        BUCKET_NAMES,
        default="")
    out = pd.Series(labels, index=status.index, dtype=object)
    out[~ok] = status[~ok]
    return out


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------

def pass1_code_rows(paths, col_eff):
    """Every row carrying a zero-balance code. Unit: (loan, code) rows."""
    parts = []
    for path in paths:
        t0 = time.time()
        n_hits = 0
        reader = pd.read_csv(
            path, sep="|", header=None, dtype=str,
            usecols=[COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC, col_eff],
            chunksize=CHUNKSIZE, engine="c", na_filter=False,
        )
        for chunk in reader:
            code = chunk[COL_ZBC].str.strip()
            keep = code != ""
            if not keep.any():
                continue
            parts.append(pd.DataFrame({
                "loan": chunk.loc[keep, COL_LOAN].str.strip(),
                "period": chunk.loc[keep, COL_PERIOD].str.strip(),
                "dq": chunk.loc[keep, COL_DQ].str.strip(),
                "code": code[keep],
                "eff": chunk.loc[keep, col_eff].str.strip(),
            }))
            n_hits += int(keep.sum())
        emit("    %-32s zero_balance rows %12s   (%.1fs)"
             % (os.path.basename(path), fmt(n_hits), time.time() - t0))
    if not parts:
        return pd.DataFrame(columns=["loan", "period", "dq", "code", "eff"])
    return pd.concat(parts, ignore_index=True, copy=False)


def pass2_last_period(paths):
    """Maximum reporting period per loan. Unit: loans."""
    running = None
    for path in paths:
        t0 = time.time()
        rows = 0
        reader = pd.read_csv(
            path, sep="|", header=None, dtype=str,
            usecols=[COL_LOAN, COL_PERIOD],
            chunksize=CHUNKSIZE, engine="c", na_filter=False,
        )
        for chunk in reader:
            rows += len(chunk)
            g = pd.DataFrame({
                "loan": chunk[COL_LOAN].str.strip(),
                "period": pd.to_numeric(chunk[COL_PERIOD].str.strip(),
                                        errors="coerce").fillna(-1).astype("int64"),
            }).groupby("loan")["period"].max()
            running = g if running is None else pd.concat([running, g]).groupby(level=0).max()
        emit("    %-32s rows %14s   loans so far %12s   (%.1fs)"
             % (os.path.basename(path), fmt(rows),
                fmt(0 if running is None else len(running)), time.time() - t0))
    return running if running is not None else pd.Series(dtype="int64")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))
    col_eff, sheet, position, attribute, n_layout_hits = resolve_eff_index(LAYOUT)

    rule("=")
    emit("12_zero_balance_date_offset - zero-balance effective date vs reporting period")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("layout file : %s" % LAYOUT)
    emit("            : matched attribute %r on %s (%d match(es) in workbook)"
         % (attribute.strip(), sheet, n_layout_hits))
    emit("            : published field position %d (1-based)" % position)
    emit("ZERO-BALANCE EFFECTIVE DATE COLUMN INDEX USED: %d  (0-based)" % col_eff)
    emit("columns read: %d loan_id, %d reporting_period, %d dq_status, %d zero_balance_code,"
         " %d zero_balance_effective_date"
         % (COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC, col_eff))
    emit("codes in scope: %s" % ", ".join(CODES))
    emit("collapse rule : one row per loan, the latest-period row carrying a code")
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/2 - rows carrying a zero-balance code")
    rule()
    rows = pass1_code_rows(paths, col_eff)
    emit("    pooled zero-balance rows: %s" % fmt(len(rows)))

    emit()
    emit("PASS 2/2 - last reporting period per loan")
    rule()
    last_period = pass2_last_period(paths)
    emit("    distinct loans in panel: %s" % fmt(len(last_period)))

    rows["period_i"] = pd.to_numeric(rows["period"], errors="coerce").fillna(-1).astype("int64")
    rows = rows.sort_values("period_i", kind="mergesort")

    per_loan_rows = rows.groupby("loan")["code"].size()
    per_loan_codes = rows.groupby("loan")["code"].nunique()
    same_code_rows = rows.groupby(["loan", "code"]).size()
    n_multi_row = int((per_loan_rows > 1).sum())
    n_multi_code = int((per_loan_codes > 1).sum())
    n_same_code_repeat = int(
        same_code_rows[same_code_rows > 1].index.get_level_values(0).nunique())

    loans = rows.drop_duplicates("loan", keep="last").set_index("loan")
    loans = loans.join(parse_eff(loans["eff"]))
    loans["offset"] = np.where(
        loans["status"].values == "ok",
        months_between(loans["ym"].values, loans["period_i"].values),
        np.nan)
    loans["bucket"] = bucket_offsets(loans["status"], loans["offset"])

    in_scope = loans[loans["code"].isin(CODES)]
    out_scope = loans[~loans["code"].isin(CODES)]
    n_in = len(in_scope)

    # ---- field format -------------------------------------------------------
    emit()
    rule("=")
    emit("FIELD FORMAT - raw character width of the zero-balance effective date")
    rule("=")
    emit("Unit: loans in scope (one code row each). Determines whether a day-of-month")
    emit("exists to bin in section B.")
    emit()
    emit("%-32s %14s %12s" % ("raw width (characters)", "loans", "share"))
    rule()
    for w, c in in_scope["width"].value_counts().sort_index().items():
        label = {0: "0  (blank)", 6: "6  (YYYYMM, no day)", 8: "8  (YYYYMMDD)"} \
            .get(int(w), "%d  (unrecognised)" % int(w))
        emit("%-32s %14s %12s" % (label, fmt(c), pct(c, n_in)))
    rule()
    emit("%-32s %14s %12s" % ("TOTAL", fmt(n_in), pct(n_in, n_in)))
    n_with_day = int((in_scope["day"] > 0).sum())
    has_day = n_with_day > 0
    emit()
    emit("effective dates carrying a day component: %s of %s loans in scope (%s)"
         % (fmt(n_with_day), fmt(n_in), pct(n_with_day, n_in)))

    # ---- A ------------------------------------------------------------------
    emit()
    rule("=")
    emit("A. OFFSET_MONTHS BY ZERO-BALANCE CODE   (unit: loans)")
    rule("=")
    emit("offset_months = calendar month of the effective date minus the reporting period")
    emit("of the row carrying the code, in whole calendar months.")
    for code in CODES:
        sub = in_scope[in_scope["code"] == code]
        n = len(sub)
        counts = sub["bucket"].value_counts()
        emit()
        emit("code %s - %s   (loans: %s)" % (code, ZBC_LABELS.get(code, "unlabelled"), fmt(n)))
        rule()
        emit("%-34s %14s %12s" % ("offset_months", "loans", "share"))
        rule()
        for name in ALL_BUCKETS:
            c = int(counts.get(name, 0))
            emit("%-34s %14s %12s" % (name, fmt(c), pct(c, n)))
        rule()
        emit("%-34s %14s %12s" % ("TOTAL", fmt(n), pct(n, n)))
        off = sub.loc[sub["status"] == "ok", "offset"]
        emit("%-34s %14s" % ("min / max observed offset",
                             "n/a" if off.empty else "%d / %d" % (off.min(), off.max())))

    if len(out_scope):
        emit()
        rule()
        emit("codes observed but OUT OF SCOPE for this diagnostic (reported, not analysed):")
        for code, c in out_scope["code"].value_counts().sort_index().items():
            emit("    code %-4s %14s loans   %s"
                 % (code, fmt(c), ZBC_LABELS.get(code, "unlabelled")))

    # ---- B ------------------------------------------------------------------
    emit()
    rule("=")
    emit("B. DAY-OF-MONTH OF THE EFFECTIVE DATE, offset_months == 0 ONLY   (unit: loans)")
    rule("=")
    if not has_day:
        emit("NO DAY COMPONENT IS PRESENT IN THE DATA. Per the FIELD FORMAT table above, the")
        emit("zero-balance effective date is published at month granularity for every loan in")
        emit("scope, so no day-of-month can be binned and no mean day computed. The table is")
        emit("printed with its counts so the empty result is explicit rather than omitted.")
    emit()
    emit("%-6s %10s %10s %10s %10s %10s %10s %14s %10s"
         % ("code", "1-5", "6-10", "11-15", "16-20", "21-25", "26-31",
            "loans off=0", "mean day"))
    rule()
    for code in CODES:
        sub = in_scope[(in_scope["code"] == code) & (in_scope["bucket"] == "0")]
        days = sub.loc[sub["day"] > 0, "day"]
        cells = [int(((days >= lo) & (days <= hi)).sum()) for _, lo, hi in DAY_BINS]
        emit("%-6s %10s %10s %10s %10s %10s %10s %14s %10s"
             % tuple([code] + [fmt(c) for c in cells] + [fmt(len(sub)),
                     "n/a" if days.empty else "%.2f" % days.mean()]))
    rule()
    emit("bins count only loans whose effective date carries a day component; 'loans off=0'")
    emit("is the full offset_months == 0 population for that code either way.")

    # ---- C ------------------------------------------------------------------
    emit()
    rule("=")
    emit("C. OFFSET_MONTHS x DELINQUENCY STATUS ON THE CODE-CARRYING ROW, codes 02/03/09")
    rule("=")
    emit("Unit: loans. Status is column %d exactly as written on the row carrying the code."
         % COL_DQ)
    for code in CREDIT_EVENT:
        sub = in_scope[in_scope["code"] == code]
        n = len(sub)
        emit()
        emit("code %s - %s   (loans: %s)" % (code, ZBC_LABELS.get(code, ""), fmt(n)))
        rule()
        if not n:
            emit("no loans with this code")
            continue
        ct = pd.crosstab(sub["dq"].replace("", "(blank)"), sub["bucket"])
        for col in ALL_BUCKETS:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[ALL_BUCKETS]
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
        emit("%-12s %s %14s %10s"
             % ("dq_status", " ".join("%13s" % BUCKET_DISPLAY[c] for c in ALL_BUCKETS),
                "TOTAL", "share"))
        rule()
        for status, row in ct.iterrows():
            emit("%-12s %s %14s %10s"
                 % (status, " ".join("%13s" % fmt(v) for v in row.values),
                    fmt(row.sum()), pct(row.sum(), n)))
        rule()
        emit("%-12s %s %14s %10s"
             % ("TOTAL", " ".join("%13s" % fmt(v) for v in ct.sum().values),
                fmt(ct.values.sum()), pct(ct.values.sum(), n)))

    # ---- D ------------------------------------------------------------------
    code_period = rows.groupby("loan")["period_i"].max()
    aligned_last = last_period.reindex(code_period.index)
    n_term = len(code_period)
    n_missing_last = int(aligned_last.isna().sum())
    n_not_last = int((code_period < aligned_last).fillna(False).sum())

    emit()
    rule("=")
    emit("D. PANEL-INTEGRITY CHECKS ON THE CODE-CARRYING ROW   (unit: loans)")
    rule("=")
    emit("Population: every loan carrying any zero-balance code, in scope or not: %s"
         % fmt(n_term))
    emit()
    emit("%-58s %14s %12s" % ("check", "loans", "share"))
    rule()
    emit("%-58s %14s %12s" % ("a code appears on more than one row",
                              fmt(n_multi_row), pct(n_multi_row, n_term)))
    emit("%-58s %14s %12s" % ("  ... the SAME code repeating on 2+ rows",
                              fmt(n_same_code_repeat), pct(n_same_code_repeat, n_term)))
    emit("%-58s %14s %12s" % ("  ... two or more DISTINCT codes",
                              fmt(n_multi_code), pct(n_multi_code, n_term)))
    emit("%-58s %14s %12s" % ("code-carrying row is NOT the loan's last row",
                              fmt(n_not_last), pct(n_not_last, n_term)))
    emit("%-58s %14s %12s" % ("loan absent from the pass-2 period index",
                              fmt(n_missing_last), pct(n_missing_last, n_term)))
    rule()

    material = [label for label, count in (
        ("a code appears on more than one row", n_multi_row),
        ("code-carrying row is not the loan's last row", n_not_last),
        ("loan absent from the pass-2 period index", n_missing_last),
    ) if n_term and 100.0 * count / n_term > 0.10]
    emit()
    if material:
        rule("!")
        emit("!! MATERIAL - above 0.10%% of terminated loans: %s" % "; ".join(material))
        emit("!! This bears on panel construction as a whole, not only the terminal row.")
        rule("!")
    else:
        emit("All checks are below 0.10% of terminated loans.")

    emit()
    rule("=")
    emit("done in %.1fs" % (time.time() - t_start))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
