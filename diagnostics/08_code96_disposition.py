"""Does the never-RA code-96 population behave like prepayments or like credit events?

Zero-balance code 96 terminates 18,626 loans. Diagnostic 04 found 879 of them
(4.72%) show "RA" somewhere in their history; those already carry a project
decision. This script concerns ONLY the remainder - loans terminating with 96
that NEVER show RA - referred to throughout as the TARGET SET.

Every measure is reported alongside two reference groups computed identically:
    REF-01  loans terminating with zero-balance code 01 (prepaid / matured)
    REF-CE  loans terminating with 02 / 03 / 09 (credit events)
A target-set number alone is not interpretable; the comparison is the point.

ONE ASYMMETRY IS FLAGGED RATHER THAN HIDDEN. The target set excludes ever-RA
loans by construction, while the reference groups do not. REF-CE in particular is
~99% ever-RA (diagnostic 04). Where that confound bites hardest - the exit-status
discriminator - a fourth column, REF-CE(no RA), applies the same never-RA
restriction to the credit-event group so a like-for-like comparison is available.

UNITS. Two quantities appear and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation

TWO PASSES over the files. Loan rows are not assumed sorted or contiguous and may
straddle chunk and file boundaries; every reduction is order-independent.
    pass 1  [0,3,8]    terminal zero-balance code per loan, and RA presence,
                       which together fix group membership.
    pass 2  [0,1,3,7]  per-loan aggregates over the whole panel: first and last
                       month, months observed, deepest numeric status, the
                       modification flags, and the two latest rows with their
                       statuses. Pass 2 deliberately aggregates ALL loans rather
                       than filtering to the groups: the groups cover ~95% of the
                       panel, so a per-chunk membership test would cost more than
                       it saves.

The "row before the last" is taken as the two largest month indices per loan,
which lets the script distinguish an immediately preceding calendar month from a
non-adjacent earlier row across a reporting gap; both are reported.

Every numeric cast of column 3 is guarded by .str.isdigit(); "RA" is never
coerced.
"""

import glob
import zipfile
import xml.etree.ElementTree as ET
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "08_code96_disposition.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_MOD = 7
COL_ZBC = 8

TARGET_CODE = "96"
PREPAID = ["01"]
CREDIT_EVENT = ["02", "03", "09"]
RA = "RA"
CURRENT = "00"

EXPECTED_TARGET = 17_747
EXPECTED_96_TOTAL = 18_626

# The documented domain of modification_flag, per the official glossary:
#   Y = Current Period Modification, P = Prior Period Modification, Null = Not Modified
MOD_DOMAIN = {"Y", "P", ""}

# Official layout, shipped in the repository. Field positions in that sheet are
# 1-based, so field N is 0-based index N - 1.
LAYOUT_XLSX = os.path.join("docs", "file_layout_july_2026.xlsx")
LAYOUT_SHEET = "xl/worksheets/sheet2.xml"  # Monthly Performance Data File
EXPECTED_FIELDS = {
    COL_LOAN: "Loan Identifier",
    COL_PERIOD: "Period",
    COL_DQ: "Current Loan Delinquency Status",
    COL_MOD: "Modification Flag",
    COL_ZBC: "Zero Balance Code",
}

_LINES = []


def emit(line=""):
    print(line, flush=True)
    _LINES.append(line)


def write_out():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")


def rule(char="-", width=94):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den):
    return "n/a" if not den else "%.4f%%" % (100.0 * num / den)


def show(v):
    return v if v not in ("", None) else "(blank)"


def yyyymm_to_index(s):
    v = pd.to_numeric(s, errors="coerce")
    year = v // 100
    month = v % 100
    return (year * 12 + month).where((month >= 1) & (month <= 12))


def index_to_yyyymm(idx):
    if idx is None or not np.isfinite(idx):
        return "n/a"
    year, month = divmod(int(idx) - 1, 12)
    return "%04d%02d" % (year, month + 1)


def reader_for(path, cols):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=cols,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


# --------------------------------------------------------------------------
# layout confirmation for column 7
# --------------------------------------------------------------------------
def read_layout():
    """Field position -> attribute name, parsed from the official layout workbook.

    The workbook is read straight out of its zip container with the standard
    library, so no spreadsheet package has to be installed into the environment.
    Returns {0-based index: attribute name}, or None if the layout is absent.
    """
    if not os.path.exists(LAYOUT_XLSX):
        return None
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(LAYOUT_XLSX) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
        sheet = ET.fromstring(z.read(LAYOUT_SHEET))
    fields = {}
    for row in sheet.iter(ns + "row"):
        cells = {}
        for c in row.iter(ns + "c"):
            col = "".join(ch for ch in c.get("r") if ch.isalpha())
            v = c.find(ns + "v")
            if v is None:
                continue
            cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        pos, name = cells.get("A"), cells.get("B")
        if pos and name and pos.strip().isdigit():
            fields[int(pos) - 1] = name.strip()
    return fields


def confirm_mod_column(paths):
    """Confirm column positions against the official layout, then check values.

    Two independent checks are reported:
      documentary - the layout workbook shipped in docs/ names the attribute at
                    each field position used by this script.
      empirical   - the observed value domain of column 7 in the data itself.
    Returns (ok, domain, layout, mismatches).
    """
    layout = read_layout()
    mismatches = []
    if layout is not None:
        for idx, expected in EXPECTED_FIELDS.items():
            actual = layout.get(idx, "(absent)")
            if actual.strip().lower() != expected.strip().lower():
                mismatches.append((idx, expected, actual))
    domain = {}
    for path in paths:
        for chunk in reader_for(path, [COL_MOD]):
            for v, c in chunk[COL_MOD].str.strip().value_counts().items():
                domain[v] = domain.get(v, 0) + int(c)
            break  # first chunk of each file is enough to fix the domain
    ok = (set(domain) <= MOD_DOMAIN) and layout is not None and not mismatches
    return ok, domain, layout, mismatches


# --------------------------------------------------------------------------
# pass 1 - group membership
# --------------------------------------------------------------------------
def pass1(paths):
    code_parts, ra_parts = [], []
    for path in paths:
        t0 = time.time()
        n_term = 0
        f_ra = []
        for chunk in reader_for(path, [COL_LOAN, COL_DQ, COL_ZBC]):
            loan = chunk[COL_LOAN].str.strip()
            code = chunk[COL_ZBC].str.strip()
            has = code != ""
            if has.any():
                n_term += int(has.sum())
                code_parts.append(pd.DataFrame({
                    "loan": loan[has].to_numpy(),
                    "code": code[has].to_numpy(),
                }))
            m = chunk[COL_DQ].str.strip() == RA
            if m.any():
                f_ra.append(loan[m].unique())
        r = pd.unique(np.concatenate(f_ra)) if f_ra else np.array([])
        ra_parts.append(r)
        emit("    %-30s zero-balance rows %10s  RA loans %9s  (%.1fs)"
             % (os.path.basename(path), fmt(n_term), fmt(len(r)), time.time() - t0))
    term = pd.concat(code_parts, ignore_index=True, copy=False)
    term_code = term.groupby("loan")["code"].first()
    ra_loans = pd.unique(np.concatenate(ra_parts)) if ra_parts else np.array([])
    return term_code, set(ra_loans.tolist())


# --------------------------------------------------------------------------
# pass 2 - per-loan aggregates over the whole panel
# --------------------------------------------------------------------------
def pass2(paths):
    agg_parts, top_parts = [], []
    for path in paths:
        t0 = time.time()
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD]):
            loan = chunk[COL_LOAN].str.strip()
            status = chunk[COL_DQ].str.strip()
            mod = chunk[COL_MOD].str.strip()
            idx = yyyymm_to_index(chunk[COL_PERIOD].str.strip())

            digit = status.str.isdigit()
            num = pd.to_numeric(status.where(digit), errors="coerce")

            d = pd.DataFrame({
                "loan": loan.to_numpy(),
                "idx": idx.to_numpy(),
                "status": status.to_numpy(),
                "num": num.to_numpy(),
                "y": (mod == "Y").to_numpy(),
                "p": (mod == "P").to_numpy(),
            })
            g = d.groupby("loan", sort=False)
            agg_parts.append(pd.DataFrame({
                "first_idx": g["idx"].min(),
                "last_idx": g["idx"].max(),
                "n_months": g.size(),
                "max_num": g["num"].max(),
                "ever_y": g["y"].max(),
                "ever_p": g["p"].max(),
            }))
            ds = d[["loan", "idx", "status"]].sort_values(
                ["loan", "idx"], ascending=[True, False], kind="mergesort")
            top_parts.append(ds.groupby("loan", sort=False).head(2))
        emit("    %-30s aggregated   (%.1fs)"
             % (os.path.basename(path), time.time() - t0))

    cat = pd.concat(agg_parts, copy=False)
    g = cat.groupby(level=0, sort=False)
    agg = pd.DataFrame({
        "first_idx": g["first_idx"].min(),
        "last_idx": g["last_idx"].max(),
        "n_months": g["n_months"].sum(),
        "max_num": g["max_num"].max(),
        "ever_y": g["ever_y"].max(),
        "ever_p": g["ever_p"].max(),
    })
    tops = pd.concat(top_parts, ignore_index=True, copy=False)
    tops = tops.sort_values(["loan", "idx"], ascending=[True, False],
                            kind="mergesort")
    tops = tops.groupby("loan", sort=False).head(2)
    rank = tops.groupby("loan", sort=False).cumcount()
    last = tops[rank == 0].set_index("loan")
    prev = tops[rank == 1].set_index("loan")
    return agg, last, prev


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def side_by_side(groups, series_map, title, note, top_n=None):
    """One frequency table with a column per group, shares within each group."""
    emit(title)
    rule()
    if note:
        emit(note)
        rule()
    keys = set()
    for name, _ in groups:
        keys |= set(series_map[name].dropna().unique())
    order = sorted(keys, key=lambda k: -sum(
        int((series_map[n] == k).sum()) for n, _ in groups))
    if top_n:
        order = order[:top_n]
    emit("%-12s %s" % ("value", " ".join("%22s" % n for n, _ in groups)))
    rule()
    for k in order:
        cells = []
        for name, n in groups:
            c = int((series_map[name] == k).sum())
            cells.append("%10s %10s" % (fmt(c), pct(c, n)))
        emit("%-12s %s" % (show(k), " ".join("%22s" % c for c in cells)))
    rule()
    emit("%-12s %s" % ("group size", " ".join("%22s" % fmt(n) for _, n in groups)))


def dist_row(label, values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return "%-22s %s" % (label, "(empty)")
    cells = [fmt(v.size), "%.2f" % v.mean()]
    for q in (10, 25, 50, 75, 90, 99):
        cells.append("%.1f" % np.percentile(v, q))
    cells.append("%d" % int(v.max()))
    return "%-22s %s" % (label, " ".join("%9s" % c for c in cells))


def dist_header():
    cols = ["n", "mean", "p10", "p25", "p50", "p75", "p90", "p99", "max"]
    return "%-22s %s" % ("group", " ".join("%9s" % c for c in cols))


def decile_row(label, values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return "%-22s %s" % (label, "(empty)")
    cells = ["%.1f" % np.percentile(v, q) for q in range(10, 100, 10)]
    return "%-22s %s" % (label, " ".join("%8s" % c for c in cells))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("08_code96_disposition - do never-RA code-96 loans look prepaid or defaulted?")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("files found : %d" % len(paths))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    # ---- column 7 confirmation ----
    emit()
    rule("=")
    emit("COLUMN POSITIONS - CONFIRMATION AGAINST THE OFFICIAL LAYOUT")
    rule("=")
    mod_ok, domain, layout, mismatches = confirm_mod_column(paths)
    if layout is None:
        emit("The official layout workbook was NOT found at %s." % LAYOUT_XLSX)
        emit("Column 7 cannot be confirmed documentarily; only the empirical")
        emit("domain check below is available.")
    else:
        emit("Source: %s, sheet 'Monthly Performance Data File'." % LAYOUT_XLSX)
        emit("Field positions there are 1-based; field N is 0-based index N - 1.")
        rule()
        emit("%-8s %-8s %-42s %s" % ("0-based", "field", "layout attribute", "check"))
        rule()
        for idx in sorted(EXPECTED_FIELDS):
            actual = layout.get(idx, "(absent)")
            good = actual.strip().lower() == EXPECTED_FIELDS[idx].strip().lower()
            emit("%-8d %-8d %-42s %s"
                 % (idx, idx + 1, actual, "OK" if good else "*** MISMATCH ***"))
        rule()
        emit("documentary check    : %s"
             % ("CONFIRMED" if not mismatches else "*** MISMATCH ***"))
        emit("layout fields parsed : %d" % len(layout))
    emit()
    emit("Empirical cross-check - observed value set of column %d, sampled from"
         % COL_MOD)
    emit("the first chunk of every file:")
    rule()
    emit("%-16s %16s" % ("value", "loan-months"))
    rule()
    for v in sorted(domain, key=lambda k: -domain[k]):
        emit("%-16s %16s" % (show(v), fmt(domain[v])))
    rule()
    emit("observed domain      : {%s}"
         % ", ".join(sorted(show(v) for v in domain)))
    emit("documented domain    : {%s}"
         % ", ".join(sorted(show(v) for v in MOD_DOMAIN)))
    emit("domain check         : %s"
         % ("CONSISTENT" if set(domain) <= MOD_DOMAIN else "*** INCONSISTENT ***"))
    emit()
    if mod_ok:
        emit("BASIS FOR PROCEEDING: the official layout names field 8 (0-based 7)")
        emit("'Modification Flag', and the observed values are exactly its")
        emit("documented domain (Y = Current Period Modification, P = Prior Period")
        emit("Modification, Null = Not Modified). Section 6 proceeds on a")
        emit("documentary confirmation, not an inference.")
    else:
        emit("The position of column %d could NOT be confirmed against the official"
             % COL_MOD)
        emit("layout. SECTION 6 IS SKIPPED rather than guessed.")

    emit()
    emit("PASS 1/2 - terminal zero-balance code and RA presence")
    rule()
    term_code, ra_set = pass1(paths)

    emit()
    emit("PASS 2/2 - per-loan aggregates")
    rule()
    agg, last, prev = pass2(paths)

    # ---- group membership ----
    code = term_code.reindex(agg.index)
    ever_ra = pd.Series(agg.index.isin(ra_set), index=agg.index)

    is96 = code == TARGET_CODE
    target = is96 & ~ever_ra
    ref01 = code.isin(PREPAID)
    refce = code.isin(CREDIT_EVENT)
    refce_nora = refce & ~ever_ra

    n_t = int(target.sum())
    n_1 = int(ref01.sum())
    n_c = int(refce.sum())
    n_cn = int(refce_nora.sum())
    n_96 = int(is96.sum())

    GROUPS = [("TARGET 96 no-RA", n_t), ("REF-01 prepaid", n_1),
              ("REF-CE 02/03/09", n_c)]
    MASKS = {"TARGET 96 no-RA": target, "REF-01 prepaid": ref01,
             "REF-CE 02/03/09": refce}

    # ---------------- 1. set sizes ----------------
    emit()
    rule("=")
    emit("1. SET SIZES   (unit: loans)")
    rule("=")
    emit("%-46s %14s %12s" % ("group", "loans", "share of panel"))
    rule()
    n_all = len(agg)
    emit("%-46s %14s %12s" % ("all loans in panel", fmt(n_all), pct(n_all, n_all)))
    emit("%-46s %14s %12s" % ("terminating with code 96 (all)", fmt(n_96),
                              pct(n_96, n_all)))
    emit("%-46s %14s %12s" % ("  of which ever RA (excluded)",
                              fmt(n_96 - n_t), pct(n_96 - n_t, n_all)))
    emit("%-46s %14s %12s" % ("  TARGET SET: code 96, never RA", fmt(n_t),
                              pct(n_t, n_all)))
    emit("%-46s %14s %12s" % ("REF-01: prepaid / matured", fmt(n_1), pct(n_1, n_all)))
    emit("%-46s %14s %12s" % ("REF-CE: credit events 02/03/09", fmt(n_c),
                              pct(n_c, n_all)))
    emit("%-46s %14s %12s" % ("  of which never RA", fmt(n_cn), pct(n_cn, n_all)))
    rule()
    emit("%-46s %14s" % ("expected code-96 total (diagnostic 02)",
                         fmt(EXPECTED_96_TOTAL)))
    emit("%-46s %14s" % ("observed code-96 total", fmt(n_96)))
    emit("%-46s %14s" % ("expected TARGET SET (diagnostic 04)", fmt(EXPECTED_TARGET)))
    emit("%-46s %14s" % ("observed TARGET SET", fmt(n_t)))
    if n_t != EXPECTED_TARGET or n_96 != EXPECTED_96_TOTAL:
        emit("%-46s %14s" % ("status", "*** DISCREPANCY ***"))
        emit("Difference target: %+d ; code-96 total: %+d. Reported, not absorbed."
             % (n_t - EXPECTED_TARGET, n_96 - EXPECTED_96_TOTAL))
    else:
        emit("%-46s %14s" % ("status", "MATCH"))
    emit()
    emit("NOTE ON COMPARABILITY: the target set excludes ever-RA loans by")
    emit("construction; REF-01 and REF-CE do not. REF-CE is %s ever-RA."
         % pct(n_c - n_cn, n_c))
    emit("Section 2 therefore also reports REF-CE(no RA) as a like-for-like column.")

    # ---------------- 2. status before exit ----------------
    last_status = last["status"].reindex(agg.index)
    prev_status = prev["status"].reindex(agg.index)
    prev_idx = prev["idx"].reindex(agg.index)
    adjacent = (agg["last_idx"] - prev_idx) == 1

    G4 = GROUPS + [("REF-CE no-RA", n_cn)]
    M4 = dict(MASKS)
    M4["REF-CE no-RA"] = refce_nora

    emit()
    rule("=")
    emit("2. DELINQUENCY STATUS IMMEDIATELY BEFORE EXIT   (unit: loans)")
    rule("=")
    side_by_side(G4, {n: last_status[M4[n]] for n, _ in G4},
                 "2a. STATUS ON THE LOAN'S LAST OBSERVED ROW", None, top_n=14)
    emit()
    side_by_side(G4, {n: prev_status[M4[n]] for n, _ in G4},
                 "2b. STATUS ON THE ROW BEFORE THE LAST", None, top_n=14)
    emit()
    emit("Of the 'row before the last', the share that is the immediately preceding")
    emit("calendar month rather than an earlier row across a gap:")
    rule()
    for name, n in G4:
        m = M4[name]
        adj = int((adjacent & m).fillna(False).sum())
        emit("%-22s %14s %12s" % (name, fmt(adj), pct(adj, n)))

    emit()
    emit("HEADLINE DISCRIMINATORS - status on the LAST observed row   (unit: loans)")
    rule()
    emit("%-22s %22s %22s" % ("group", "exit from 00 (Current)", "exit from >= 03"))
    rule()
    for name, n in G4:
        m = M4[name]
        s = last_status[m]
        d = s.str.isdigit()
        num = pd.to_numeric(s.where(d), errors="coerce")
        c0 = int((s == CURRENT).sum())
        c3 = int((num >= 3).sum())
        emit("%-22s %22s %22s"
             % (name, "%10s %10s" % (fmt(c0), pct(c0, n)),
                "%10s %10s" % (fmt(c3), pct(c3, n))))
    emit()
    emit("A prepayment exits from Current; a credit event exits from deep")
    emit("delinquency. This is the primary discriminator.")

    # ---------------- 3. deepest status ----------------
    emit()
    rule("=")
    emit("3. DEEPEST NUMERIC STATUS EVER REACHED   (unit: loans)")
    rule("=")
    emit("DECILES OF THE PER-LOAN MAXIMUM")
    rule()
    emit("%-22s %s" % ("group", " ".join("%8s" % ("d%d" % i) for i in range(1, 10))))
    rule()
    for name, n in GROUPS:
        emit(decile_row(name, agg.loc[MASKS[name], "max_num"].to_numpy()))
    emit()
    emit("BUCKETED   (unit: loans)")
    rule()
    emit("%-22s %18s %18s %18s %14s"
         % ("group", "never above 00", "tops out 01-02", ">= 03", "no numeric row"))
    rule()
    for name, n in GROUPS:
        v = agg.loc[MASKS[name], "max_num"]
        nn = int(v.isna().sum())
        a = int((v == 0).sum())
        b = int(((v >= 1) & (v <= 2)).sum())
        c = int((v >= 3).sum())
        emit("%-22s %18s %18s %18s %14s"
             % (name, "%8s %8s" % (fmt(a), pct(a, n)),
                "%8s %8s" % (fmt(b), pct(b, n)),
                "%8s %8s" % (fmt(c), pct(c, n)), fmt(nn)))

    # ---------------- 4. calendar timing ----------------
    years = ((agg["last_idx"] - 1) // 12)
    emit()
    rule("=")
    emit("4. CALENDAR YEAR OF EXIT   (unit: loans, shares within group)")
    rule("=")
    emit("%-8s %s" % ("year", " ".join("%22s" % n for n, _ in GROUPS)))
    rule()
    allyears = sorted(years.dropna().unique())
    for y in allyears:
        cells = []
        for name, n in GROUPS:
            c = int((years[MASKS[name]] == y).sum())
            cells.append("%10s %10s" % (fmt(c), pct(c, n)))
        emit("%-8d %s" % (int(y), " ".join("%22s" % c for c in cells)))
    rule()
    emit("%-8s %s" % ("total", " ".join("%22s" % fmt(n) for _, n in GROUPS)))

    # ---------------- 5. loan age at exit ----------------
    emit()
    rule("=")
    emit("5. MONTHS OBSERVED FROM FIRST ROW TO EXIT   (unit: months per loan)")
    rule("=")
    emit(dist_header())
    rule()
    for name, n in GROUPS:
        emit(dist_row(name, agg.loc[MASKS[name], "n_months"].to_numpy()))

    # ---------------- 6. modification history ----------------
    emit()
    rule("=")
    emit("6. MODIFICATION HISTORY   (unit: loans)")
    rule("=")
    if not mod_ok:
        emit("SKIPPED. Column 7 does not carry the documented modification_flag")
        emit("domain and the position cannot be confirmed. Not guessed.")
    else:
        emit("Basis: documentary - the official layout names field 8 (0-based 7)")
        emit("'Modification Flag'. See the confirmation block at the top.")
        rule()
        emit("%-22s %22s %22s %14s"
             % ("group", "ever 'Y'", "ever 'P'", "neither"))
        rule()
        for name, n in GROUPS:
            m = MASKS[name]
            y = int(agg.loc[m, "ever_y"].sum())
            p = int(agg.loc[m, "ever_p"].sum())
            nz = int((~agg.loc[m, "ever_y"] & ~agg.loc[m, "ever_p"]).sum())
            emit("%-22s %22s %22s %14s"
                 % (name, "%10s %10s" % (fmt(y), pct(y, n)),
                    "%10s %10s" % (fmt(p), pct(p, n)), fmt(nz)))
        emit()
        emit("'Y' and 'P' are counted separately and never summed.")

    # ---------------- 7. within-target heterogeneity ----------------
    tmax = agg.loc[target, "max_num"]
    deep = target & (agg["max_num"] >= 3).fillna(False)
    shallow = target & ~(agg["max_num"] >= 3).fillna(False)
    n_deep = int(deep.sum())
    n_shal = int(shallow.sum())

    emit()
    rule("=")
    emit("7. WITHIN-TARGET HETEROGENEITY   (unit: loans)")
    rule("=")
    emit("%-46s %14s %12s" % ("split of the target set", "loans", "share"))
    rule()
    emit("%-46s %14s %12s" % ("TARGET total", fmt(n_t), pct(n_t, n_t)))
    emit("%-46s %14s %12s" % ("  ever reached status >= 03", fmt(n_deep),
                              pct(n_deep, n_t)))
    emit("%-46s %14s %12s" % ("  never exceeded status 02", fmt(n_shal),
                              pct(n_shal, n_t)))
    emit()
    SPLIT = [("TARGET >= 03", n_deep), ("TARGET < 03", n_shal),
             ("REF-01 prepaid", n_1), ("REF-CE 02/03/09", n_c)]
    SMASK = {"TARGET >= 03": deep, "TARGET < 03": shallow,
             "REF-01 prepaid": ref01, "REF-CE 02/03/09": refce}
    emit("CALENDAR YEAR OF EXIT, TARGET HALVES vs REFERENCES   (shares within group)")
    rule()
    emit("%-8s %s" % ("year", " ".join("%20s" % n for n, _ in SPLIT)))
    rule()
    for y in allyears:
        cells = []
        for name, n in SPLIT:
            c = int((years[SMASK[name]] == y).sum())
            cells.append("%9s %9s" % (fmt(c), pct(c, n)))
        emit("%-8d %s" % (int(y), " ".join("%20s" % c for c in cells)))
    rule()
    emit("%-8s %s" % ("total", " ".join("%20s" % fmt(n) for _, n in SPLIT)))
    emit()
    emit("EXIT STATUS OF THE TWO TARGET HALVES   (unit: loans)")
    rule()
    emit("%-22s %22s %22s" % ("group", "exit from 00 (Current)", "exit from >= 03"))
    rule()
    for name, n in SPLIT:
        s = last_status[SMASK[name]]
        d = s.str.isdigit()
        num = pd.to_numeric(s.where(d), errors="coerce")
        c0 = int((s == CURRENT).sum())
        c3 = int((num >= 3).sum())
        emit("%-22s %22s %22s"
             % (name, "%10s %10s" % (fmt(c0), pct(c0, n)),
                "%10s %10s" % (fmt(c3), pct(c3, n))))
    emit()
    emit("A pooled average over a mixture of two populations would hide whichever")
    emit("half is smaller; the split is reported so that cannot happen silently.")

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
