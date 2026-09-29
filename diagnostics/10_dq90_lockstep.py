"""Does the field-4 status move in lockstep with time already spent in DQ90+?

Within an unbroken DQ90+ run the delinquency counter at field 4 and the
run-length both measure the same clock: field 4 counts months of payments owed
(derived from DDLPI), the run-length counts months observed in DQ90+. They
advance together unless something intervenes - a partial payment that rolls
DDLPI forward, a modification or payment deferral that re-bases it, or the
documented cap at 99. This diagnostic measures how often the two diverge.

For every DQ90+ loan-month:
    entry_status     numeric field-4 status in the first month of its run
    k                run-month, 1-based
    expected_status  min(entry_status + (k - 1), 99)
    deviation        observed_status - expected_status
    on_step          deviation == 0
    off_below        deviation < 0
    off_above        deviation > 0   (should not occur; reported as an anomaly)
    cap_row          observed == 99 AND entry_status + (k - 1) > 99
                     (on_step by construction; reported separately)

CONVENTIONS REUSED FROM 09_dq90_duration_hazard.py, UNCHANGED
    DQ90+ loan-month   numeric field-4 status >= 3. 98 and 99 are inside it
                       (D12: 99 is the documented cap, not a sentinel). RA and
                       XX are packed to negative codes so that "status >= 3" can
                       never catch them.
    run start / end    a run continues only across an OBSERVED, immediately
                       preceding DQ90+ month of the same loan. Any other row
                       opens a new run, and the counter is the distance from the
                       run's opening row.
    gap handling       a hole in the reporting periods breaks the run and the
                       counter restarts at 1. A repeated (loan, period) is not
                       adjacent either, so it also restarts, and is counted.
    run-length buckets 1, 2, 3, 4-6, 7-12, 13-24, 25+.
    terminal ZB row    a DQ90+ month that is itself the loan's last observed row
                       and carries a terminating zero-balance code. 09 keeps
                       these in the slice; so does T1 primary here, with T1
                       sensitivity removing them.
    chunk carry-over   there is none to carry. Both scripts run two chunked
                       passes per file: pass 1 finds the loans that ever reach
                       DQ90+, pass 2 keeps every row of exactly those loans as
                       packed integer arrays, and the arrays are lexsorted by
                       (loan, period) once the whole file has been read. A loan
                       whose rows straddle a chunk boundary is therefore
                       reassembled by the sort, not by boundary state. Loans do
                       not straddle FILES - the identifier carries its own
                       vintage prefix - and the prefix seen in each file is
                       reported so that the assumption is checked.

The only departure from 09 is that pass 1 reads three columns rather than four:
it does not need the zero-balance code. Pass 2 reads seven. Nothing about the
definitions above changes.

Every numeric cast of column 3 is guarded by .str.isdigit(); "RA" is never
coerced. No file is ever held whole.
"""

import glob
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "10_dq90_lockstep.txt")
RULE_W = 121

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_MOD = 7
COL_ZBC = 8
COL_DEFER = 24
COL_ASSIST = 29

PASS1_COLS = [COL_LOAN, COL_PERIOD, COL_DQ]
PASS2_COLS = [COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD, COL_ZBC, COL_DEFER, COL_ASSIST]

LAYOUT_XLSX = os.path.join("docs", "file_layout_july_2026.xlsx")
LAYOUT_SHEET = "xl/worksheets/sheet2.xml"  # Monthly Performance Data File

EXPECTED_FIELDS = {
    COL_LOAN: "Loan Identifier",
    COL_PERIOD: "Period",
    COL_DQ: "Current Loan Delinquency Status",
    COL_MOD: "Modification Flag",
    COL_ZBC: "Zero Balance Code",
    COL_DEFER: "Payment Deferral Flag",
    COL_ASSIST: "Borrower Assistance Plan",
}

# Indices the brief expects for the three workout flags, checked against the
# layout workbook at run time rather than trusted.
BRIEF_EXPECTED = {COL_MOD: 7, COL_DEFER: 24, COL_ASSIST: 29}

# Field-4 status packing, identical to 09.
ST_RA = -1
ST_OTHER = -2

ZB_BLANK = 0
ZB_OTHER = -1
ZB_MAP = {"": 0, "01": 1, "02": 2, "03": 3, "09": 9, "15": 15, "16": 16, "96": 96}
ZB_TERMINATING = (1, 2, 3, 9, 15, 16, 96)

# Workout-flag packing. 0 is always the blank.
MOD_NULL, MOD_Y, MOD_P, MOD_OTHER = 0, 1, 2, 3
MOD_MAP = {"": MOD_NULL, "Y": MOD_Y, "P": MOD_P}

DEF_NULL, DEF_C, DEF_P, DEF_OTHER = 0, 1, 2, 3
DEF_MAP = {"": DEF_NULL, "C": DEF_C, "P": DEF_P}

AST_NULL, AST_F, AST_R, AST_T, AST_OTHER = 0, 1, 2, 3, 4
AST_MAP = {"": AST_NULL, "F": AST_F, "R": AST_R, "T": AST_T}

STATUS_CAP = 99

# Run-length buckets, identical to 09.
BUCKETS = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-6", 4, 6),
    ("7-12", 7, 12),
    ("13-24", 13, 24),
    ("25+", 25, 10 ** 9),
]

# Observed field-4 status buckets for the T4 cross-tab. Disjoint, and together
# they cover every DQ90+ status: 3 .. 99.
ST_BUCKETS = [
    ("3", 3, 3),
    ("4-6", 4, 6),
    ("7-12", 7, 12),
    ("13-24", 13, 24),
    ("25-98", 25, 98),
    ("99", 99, 99),
]

# Deviation-size buckets for T5, over off_below rows only.
DEV_BUCKETS = [
    ("-1", -1, -1),
    ("-2", -2, -2),
    ("-3 to -5", -5, -3),
    ("-6 or lower", -(10 ** 9), -6),
]

# Thin-cell thresholds, as briefed.
THIN_MONTHS = 500
THIN_LOANS = 100

BIG_PERIOD = 1 << 30

# Slice totals published by 09, re-checked here so that any drift in the
# shared definitions shows up rather than passing silently.
EXPECTED_ROWS = 150_967_486
EXPECTED_DQ90_MONTHS = 6_463_333
EXPECTED_DQ90_LOANS = 373_546

N_OFF_EXAMPLES = 10

_LINES = []
_THIN = []


def emit(line=""):
    print(line, flush=True)
    _LINES.append(line)


def write_out():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")


def rule(char="-", width=RULE_W):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den, dp=3):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out


def note_thin(table, cell, n, unit):
    """Record a cell whose denominator falls under the briefed threshold."""
    thr = THIN_MONTHS if unit == "loan-months" else THIN_LOANS
    if n < thr:
        _THIN.append((table, cell, int(n), unit, thr))
    return n < thr


def thin_mark(n, unit):
    thr = THIN_MONTHS if unit == "loan-months" else THIN_LOANS
    return "  << THIN" if n < thr else ""


def reader_for(path, cols):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=cols,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


# --------------------------------------------------------------------------
# PART A - documentation lookup
# --------------------------------------------------------------------------
# Verbatim transcriptions from docs/general_user_guide_july_2026.pdf. Curly
# quotes in the source are rendered as ASCII; nothing else is altered.
FLAG_QUOTES = [
    (COL_MOD, "Modification Flag", "PERSISTENT", "Y", """\
  Description
    "A flag indicating if the loan has been modified in the current period or a
     prior period.

     Note: Due to the various loan modification programs available to borrowers,
     a loan could have been modified more than once. In these cases, the
     modification flag will be set to "Y" in more than one monthly reporting
     period."

  Enumerations
    "Y = Current Period Modification
     P = Prior Period Modification

     Null = Not Modified\"""",
     "'or a prior period', and the separate enumeration P = Prior Period "
     "Modification, make this an EVER flag: once a modification happens the "
     "field stays populated. Onset is therefore Y, not 'non-null'."),

    (COL_DEFER, "Payment Deferral Flag", "PERSISTENT", "C", """\
  Description
    "A flag indicating a loan has been granted a Payment Deferral in the current
     or prior period."

  Enumerations
    "C = Current Period
     P = Prior Period\"""",
     "'in the current or prior period', with P = Prior Period, makes this an "
     "EVER flag on the same pattern. Onset is C."),

    (COL_ASSIST, "Borrower Assistance Plan", "CURRENT-MONTH", None, """\
  Description
    "Regardless of delinquency status, the type of assistance plan that the
     borrower is enrolled in that provides temporary mortgage payment relief or
     an opportunity to cure a mortgage delinquency over a defined period.

     Note: Only populated for January 2014 and following periods."

  Enumerations
    "F = Forbearance
     R = Repayment
     T = Trial Period

     Null = No workout plan or not applicable\"""",
     "the enumeration carries no prior-period value and the description is in "
     "the present tense - the plan the borrower 'is enrolled in'. This is a "
     "CURRENT-MONTH indicator, so the flag VALUE is used, not an onset. The "
     "January 2014 restriction bites hard on these vintages: the 2007-2011 "
     "window this cohort exists to observe is entirely before it."),
]


def read_layout():
    """Field position -> attribute name for the monthly performance file.

    Parsed straight out of the xlsx zip container with the standard library, so
    no spreadsheet package has to be installed. Returns {0-based index: name}
    or None if the workbook is absent. Identical to 09.
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
            fields[int(pos) - 1] = re.sub(r"\s+", " ", name).strip()
    return fields


def part_a():
    rule("=")
    emit("PART A - conventions reused, and the documentation behind the three workout flags")
    rule("=")
    emit("A.1  CONVENTIONS REUSED FROM 09_dq90_duration_hazard.py")
    rule()
    for label, text in [
        ("DQ90+ loan-month",
         "numeric field-4 status >= 3. 98 and 99 are inside it (D12: 99 is the documented "
         "cap, not a sentinel). RA packs to -1 and any other non-numeric value to -2, so the "
         "test 'status >= 3' can never catch RA or XX."),
        ("run start / end",
         "a run continues only across an OBSERVED, immediately preceding DQ90+ month of the "
         "same loan. Every other row opens a new run. Run-length is the distance from the "
         "run's opening row, 1-based."),
        ("gap handling",
         "a hole in the reporting periods breaks the run and the counter restarts at 1. A "
         "repeated (loan, period) is not adjacent either, so it restarts a run too; the count "
         "is reported."),
        ("run-length buckets",
         "1, 2, 3, 4-6, 7-12, 13-24, 25+ - exactly 09's."),
        ("terminal zero-balance row",
         "a DQ90+ month that is itself the loan's last observed row and carries a terminating "
         "zero-balance code (01/02/03/09/15/16/96). 09 keeps these inside the slice. T1 "
         "primary keeps them; T1 sensitivity removes them."),
        ("chunk-boundary carry-over",
         "none is needed. Pass 1 finds the loans that ever reach DQ90+ in the file; pass 2 "
         "keeps every row of exactly those loans as packed integer arrays; the arrays are "
         "lexsorted by (loan, period) after the whole file is read. A loan straddling a chunk "
         "boundary is reassembled by that sort, not by boundary state. This is 09's mechanism "
         "unchanged."),
        ("loans do not straddle files",
         "the identifier carries its own vintage prefix. The prefix seen in each file is "
         "reported below so the assumption is checked rather than assumed."),
    ]:
        emit("    %-28s %s" % (label, ""))
        for line in wrap(text, RULE_W - 10):
            emit("        %s" % line)
    emit()
    for line in wrap(
        "Departure from 09: pass 1 reads three columns (0, 1, 3) instead of four - it has no "
        "use for the zero-balance code - and pass 2 reads seven (0, 1, 3, 7, 8, 24, 29) "
        "instead of four. This is a read-width change only. No definition above is affected, "
        "and the slice totals are re-checked against 09's published figures in B.1.",
            RULE_W - 4):
        emit("    %s" % line)
    emit()

    emit("A.2  COLUMN POSITIONS, CONFIRMED AGAINST THE LAYOUT WORKBOOK")
    rule()
    layout = read_layout()
    if layout is None:
        emit("    !! layout workbook not found at %s - positions NOT confirmed" % LAYOUT_XLSX)
        emit()
        return
    emit("    source : %s, sheet 'Monthly Performance Data File'" % LAYOUT_XLSX)
    emit("    layout lists %d field positions for the monthly performance file." % len(layout))
    emit()
    emit("    %-6s %-6s %-42s %s" % ("index", "field", "attribute name in layout", "check"))
    mismatches = []
    for idx, expected in sorted(EXPECTED_FIELDS.items()):
        actual = layout.get(idx, "(absent)")
        ok = actual.strip().lower() == expected.strip().lower()
        if not ok:
            mismatches.append(idx)
        emit("    %-6d %-6d %-42s %s" % (idx, idx + 1, actual, "OK" if ok else "MISMATCH"))
    emit()
    emit("    The brief expected the three workout flags at indices 7, 24 and 29:")
    for idx, brief_idx in sorted(BRIEF_EXPECTED.items()):
        actual = layout.get(idx, "(absent)")
        agree = (idx == brief_idx) and actual.strip().lower() == \
            EXPECTED_FIELDS[idx].strip().lower()
        emit("      brief index %-3d -> %-26s %s"
             % (brief_idx, EXPECTED_FIELDS[idx],
                "CONFIRMED by the layout" if agree else "NOT confirmed - see above"))
    if mismatches:
        emit("    !! layout mismatch at index(es) %s - the guide's positions are used and the"
             % ", ".join(str(i) for i in mismatches))
        emit("       mismatch is on record here.")
    else:
        emit("    All seven positions confirmed. The brief's expectation holds; nothing moved.")
    emit()

    emit("A.3  ARE THE THREE FLAGS CURRENT-MONTH OR PERSISTENT? THE GUIDE'S OWN WORDS")
    rule()
    emit("    source : docs/general_user_guide_july_2026.pdf, Monthly Performance Data File")
    emit()
    for idx, name, kind, onset, quote, reading in FLAG_QUOTES:
        emit("    field %2d (index %2d)  %s" % (idx + 1, idx, name))
        emit("    %s" % ("." * (RULE_W - 8)))
        for line in quote.splitlines():
            emit("    %s" % line.rstrip())
        emit()
        emit("      CLASSIFICATION: %s" % kind)
        for line in wrap(reading, RULE_W - 10):
            emit("        %s" % line)
        if onset:
            emit("        Activity test used in T7: flag value == '%s' (onset)." % onset)
        else:
            emit("        Activity test used in T7: flag value is non-null (F, R or T).")
        emit()

    emit("A.4  CONSEQUENCE FOR T7")
    rule()
    for line in wrap(
        "Because Modification Flag and Payment Deferral Flag are EVER flags, testing them for "
        "'non-null' at a departure month would mark every month after the first modification, "
        "not the month the modification happened. T7 therefore tests ONSET for those two - Y "
        "and C respectively - and the VALUE for Borrower Assistance Plan, which carries no "
        "prior-period code. A departure is credited to a flag if the onset or value appears at "
        "the departure month itself or at the immediately preceding observed month, where that "
        "preceding month exists and is exactly one period earlier.", RULE_W - 4):
        emit("    %s" % line)
    emit()


PRE_REGISTRATION = """\
PRIMARY METRIC
    Share of DQ90+ loan-months with deviation != 0, using each run's own
    entry_status, on the same slice as diagnostic 09.

PREDICTIONS, RECORDED BEFORE THE DATA WAS READ
    Yassine : under 5%
    Claude  : 20-50%, point estimate ~30%"""


def pre_registration():
    rule("=")
    emit("PRE-REGISTRATION - recorded before any result below was computed")
    rule("=")
    for line in PRE_REGISTRATION.splitlines():
        emit(line)
    emit()


# --------------------------------------------------------------------------
# encoding helpers - identical to 09 where shared
# --------------------------------------------------------------------------
def encode_period(s):
    """YYYYMM -> a contiguous month index. Invalid months give -1."""
    v = pd.to_numeric(s, errors="coerce")
    year, month = v // 100, v % 100
    idx = (year * 12 + month).where((month >= 1) & (month <= 12))
    return idx.fillna(-1).astype(np.int32)


def encode_status(s):
    """Field 4 -> int16. Numeric keeps its value; RA -> -1; anything else -> -2."""
    digits = s.str.isdigit()
    num = pd.to_numeric(s.where(digits), errors="coerce")
    code = num.fillna(ST_OTHER).astype(np.int16)
    code[~digits & (s == "RA")] = ST_RA
    return code


def encode_zbc(s):
    """Field 9 -> int16 over the documented enumeration; unknown -> -1."""
    return s.map(ZB_MAP).fillna(ZB_OTHER).astype(np.int16)


def encode_flag(s, mapping, other):
    return s.map(mapping).fillna(other).astype(np.int8)


def ym(idx):
    if idx is None:
        return "n/a"
    y, m = divmod(int(idx) - 1, 12)
    return "%04d%02d" % (y, m + 1)


# --------------------------------------------------------------------------
# per-file passes
# --------------------------------------------------------------------------
def pass1_dq90_loans(path):
    """Loans reaching DQ90+ at least once, plus row tallies and a sort audit.

    The sort audit carries the last (loan, period) across chunk boundaries, so
    a violation sitting exactly on a boundary is caught like any other.
    """
    seen, rows, bad_period, prefixes = [], 0, 0, {}
    pmin, pmax = None, None
    loan_desc = period_desc = 0
    last_loan, last_period = None, None
    for chunk in reader_for(path, PASS1_COLS):
        rows += len(chunk)
        loan = chunk[COL_LOAN].str.strip()
        status = encode_status(chunk[COL_DQ].str.strip())
        period = encode_period(chunk[COL_PERIOD].str.strip()).values
        bad_period += int((period < 0).sum())
        good = period[period >= 0]
        if good.size:
            lo, hi = int(good.min()), int(good.max())
            pmin = lo if pmin is None else min(pmin, lo)
            pmax = hi if pmax is None else max(pmax, hi)
        for p, c in loan.str[:5].value_counts().items():
            prefixes[p] = prefixes.get(p, 0) + int(c)

        lv = loan.values
        prev_l = np.empty(lv.shape, dtype=object)
        prev_l[1:] = lv[:-1]
        prev_l[0] = last_loan if last_loan is not None else lv[0]
        prev_p = np.empty(period.shape, dtype=np.int64)
        prev_p[1:] = period[:-1]
        prev_p[0] = last_period if last_period is not None else period[0]
        same = lv == prev_l
        loan_desc += int((~same & (lv < prev_l)).sum())
        period_desc += int((same & (period < prev_p)).sum())
        last_loan, last_period = lv[-1], int(period[-1])

        hit = lv[status.values >= 3]
        if hit.size:
            seen.append(pd.unique(hit))
    loans = pd.unique(np.concatenate(seen)) if seen else np.array([], dtype=object)
    order = {"loan_desc": loan_desc, "period_desc": period_desc}
    return loans, rows, bad_period, prefixes, pmin, pmax, order


def pass2_collect(path, loan_index):
    """Every row belonging to the DQ90+ loan set, packed into integer arrays."""
    codes, periods, statuses, zbcs, mods, defers, assists = [], [], [], [], [], [], []
    bad_period = 0
    for chunk in reader_for(path, PASS2_COLS):
        loan = chunk[COL_LOAN].str.strip().values
        code = loan_index.get_indexer(loan)
        keep = code >= 0
        if not keep.any():
            continue
        period = encode_period(chunk[COL_PERIOD].str.strip()).values[keep]
        ok = period >= 0
        bad_period += int((~ok).sum())
        codes.append(code[keep][ok].astype(np.int32))
        periods.append(period[ok])
        statuses.append(encode_status(chunk[COL_DQ].str.strip()).values[keep][ok])
        zbcs.append(encode_zbc(chunk[COL_ZBC].str.strip()).values[keep][ok])
        mods.append(encode_flag(chunk[COL_MOD].str.strip(), MOD_MAP, MOD_OTHER)
                    .values[keep][ok])
        defers.append(encode_flag(chunk[COL_DEFER].str.strip(), DEF_MAP, DEF_OTHER)
                      .values[keep][ok])
        assists.append(encode_flag(chunk[COL_ASSIST].str.strip(), AST_MAP, AST_OTHER)
                       .values[keep][ok])
    if not codes:
        e32 = np.array([], dtype=np.int32)
        e16 = np.array([], dtype=np.int16)
        e8 = np.array([], dtype=np.int8)
        return e32, e32, e16, e16, e8, e8, e8, bad_period
    return (
        np.concatenate(codes), np.concatenate(periods), np.concatenate(statuses),
        np.concatenate(zbcs), np.concatenate(mods), np.concatenate(defers),
        np.concatenate(assists), bad_period,
    )


def bucket_of(run):
    """Run length -> bucket row index, vectorised. Identical to 09."""
    out = np.full(run.shape, len(BUCKETS) - 1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(BUCKETS):
        out[(run >= lo) & (run <= hi)] = i
    return out


def st_bucket_of(status):
    out = np.full(status.shape, -1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(ST_BUCKETS):
        out[(status >= lo) & (status <= hi)] = i
    return out


def dev_bucket_of(dev):
    out = np.full(dev.shape, -1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(DEV_BUCKETS):
        out[(dev >= lo) & (dev <= hi)] = i
    return out


ACC_KEYS = [
    "rows", "bad_period", "dq90_months", "dq90_loans", "runs",
    "gap_breaks", "duplicate_period_rows", "terminal_dq90_rows",
    "unknown_zbc_rows", "unknown_mod_rows", "unknown_def_rows", "unknown_ast_rows",
    "first_dep_no_prior_month",
]

# Column order for the on_step / off_below / off_above / cap_row tables.
C_ON, C_BELOW, C_ABOVE, C_CAP = 0, 1, 2, 3
STEP_NAMES = ["on_step", "off_below", "off_above"]


def blank_tables():
    return {
        # T1: 2 slices (primary, ex-terminal) x 4 columns
        "t1": np.zeros((2, 4), dtype=np.int64),
        # T2: run bucket x 4 columns, primary slice
        "t2": np.zeros((len(BUCKETS), 4), dtype=np.int64),
        # T2b: run bucket x 4 columns, ex-terminal slice
        "t2b": np.zeros((len(BUCKETS), 4), dtype=np.int64),
        # T3: entry-status bucket (3,4,5,6+) x {runs, loans-first-run, loans-any-run}
        "t3": np.zeros((4, 3), dtype=np.int64),
        # T4: run bucket x observed-status bucket
        "t4": np.zeros((len(BUCKETS), len(ST_BUCKETS)), dtype=np.int64),
        # T5: deviation-size bucket
        "t5": np.zeros(len(DEV_BUCKETS), dtype=np.int64),
        # T6: [first departure, downstream]
        "t6": np.zeros(2, dtype=np.int64),
        # T7: run bucket x [n, mod, defer, assist, any, none, no-prior-month]
        "t7": np.zeros((len(BUCKETS), 7), dtype=np.int64),
        # T8: secondary reference, entry fixed at 3. 2 slices x 4 columns
        "t8": np.zeros((2, 4), dtype=np.int64),
    }


ENTRY_LABELS = ["3", "4", "5", "6+"]


def entry_bucket_of(entry):
    out = np.full(entry.shape, 3, dtype=np.int8)
    out[entry == 3] = 0
    out[entry == 4] = 1
    out[entry == 5] = 2
    return out


def walk_file(loan, period, status, zbc, mod, defer, assist, loan_index):
    """Lockstep tabulation for every DQ90+ loan-month in one file.

    Arrays arrive unsorted and are lexsorted by (loan, period) here - this is
    what makes chunk boundaries immaterial. The whole reduction is vectorised;
    there is no per-loan Python loop.
    """
    tab = blank_tables()
    stats = {k: 0 for k in ACC_KEYS}
    examples = []
    n = loan.size
    if n == 0:
        return tab, stats, examples

    order = np.lexsort((period, loan))
    loan = loan[order]
    period = period[order].astype(np.int64)
    status = status[order]
    zbc = zbc[order]
    mod = mod[order]
    defer = defer[order]
    assist = assist[order]

    is90 = status >= 3
    stats["dq90_months"] = int(is90.sum())
    stats["unknown_zbc_rows"] = int((zbc == ZB_OTHER).sum())
    stats["unknown_mod_rows"] = int((mod == MOD_OTHER).sum())
    stats["unknown_def_rows"] = int((defer == DEF_OTHER).sum())
    stats["unknown_ast_rows"] = int((assist == AST_OTHER).sum())
    if stats["dq90_months"] == 0:
        return tab, stats, examples
    stats["dq90_loans"] = int(np.unique(loan[is90]).size)

    # ---- run construction, identical to 09 ---------------------------------
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = loan[1:] == loan[:-1]
    prev_period = np.empty(n, dtype=np.int64)
    prev_period[0] = -BIG_PERIOD
    prev_period[1:] = period[:-1]
    prev_is90 = np.zeros(n, dtype=bool)
    prev_is90[1:] = is90[:-1]

    adjacent = same_prev & (period == prev_period + 1)
    cont = is90 & prev_is90 & adjacent
    stats["gap_breaks"] = int(
        (is90 & prev_is90 & same_prev & (period > prev_period + 1)).sum())
    stats["duplicate_period_rows"] = int((same_prev & (period == prev_period)).sum())

    pos = np.arange(n, dtype=np.int64)
    start = np.maximum.accumulate(np.where(~cont, pos, 0))
    run = pos - start + 1

    # ---- lockstep arithmetic ----------------------------------------------
    entry = status[start].astype(np.int64)          # entry_status of the row's run
    obs = status.astype(np.int64)
    ladder = entry + run - 1                        # uncapped expected position
    expected = np.minimum(ladder, STATUS_CAP)
    dev = obs - expected

    on_step = is90 & (dev == 0)
    off_below = is90 & (dev < 0)
    off_above = is90 & (dev > 0)
    cap_row = is90 & (obs == STATUS_CAP) & (ladder > STATUS_CAP)

    # ---- terminal zero-balance rows, per 09's definition -------------------
    has_next = np.zeros(n, dtype=bool)
    has_next[:-1] = loan[:-1] == loan[1:]
    terminal = is90 & ~has_next & np.isin(zbc, ZB_TERMINATING)
    stats["terminal_dq90_rows"] = int(terminal.sum())
    keep = is90 & ~terminal                         # sensitivity slice

    # ---- T1 ----------------------------------------------------------------
    for r, m in ((0, is90), (1, keep)):
        tab["t1"][r, C_ON] = int((on_step & m).sum())
        tab["t1"][r, C_BELOW] = int((off_below & m).sum())
        tab["t1"][r, C_ABOVE] = int((off_above & m).sum())
        tab["t1"][r, C_CAP] = int((cap_row & m).sum())

    # ---- T2 ----------------------------------------------------------------
    rb = bucket_of(run)
    for key, m in (("t2", is90), ("t2b", keep)):
        for col, flagm in ((C_ON, on_step), (C_BELOW, off_below),
                           (C_ABOVE, off_above), (C_CAP, cap_row)):
            sel = flagm & m
            if sel.any():
                np.add.at(tab[key][:, col], rb[sel], 1)

    # ---- T3 ----------------------------------------------------------------
    run_start = is90 & ~cont
    stats["runs"] = int(run_start.sum())
    if run_start.any():
        eb = entry_bucket_of(status[run_start].astype(np.int64))
        np.add.at(tab["t3"][:, 0], eb, 1)
        # Loans by their FIRST run's entry status: partitions the loan set.
        # Keyed on ROW INDEX, not period. Two run starts can share a period
        # when a (loan, period) row is duplicated - each opens its own run,
        # because a repeated period is not adjacent - and selecting on period
        # would then count that loan twice and break the partition.
        nloan = int(loan.max()) + 1
        first_start = np.full(nloan, n, dtype=np.int64)
        np.minimum.at(first_start, loan[run_start], pos[run_start])
        picked = first_start[first_start < n]
        is_first = np.zeros(n, dtype=bool)
        is_first[picked] = True
        np.add.at(tab["t3"][:, 1],
                  entry_bucket_of(status[is_first].astype(np.int64)), 1)
        # loans with AT LEAST ONE run at that entry status: does not partition
        for b in range(4):
            sel = run_start & (entry_bucket_of(entry) == b)
            tab["t3"][b, 2] = int(np.unique(loan[sel]).size) if sel.any() else 0

    # ---- T4 ----------------------------------------------------------------
    sb = st_bucket_of(obs)
    sel = is90 & (sb >= 0)
    if sel.any():
        np.add.at(tab["t4"], (rb[sel], sb[sel]), 1)

    # ---- T5 ----------------------------------------------------------------
    if off_below.any():
        db = dev_bucket_of(dev[off_below])
        np.add.at(tab["t5"], db[db >= 0], 1)

    # ---- T6 : stickiness ---------------------------------------------------
    # Departures counted within each run: a row is the FIRST departure of its
    # run when the number of off-step rows from the run's opening row up to and
    # including this one is exactly 1.
    off = is90 & (dev != 0)
    cs = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(off, out=cs[1:])
    within = cs[pos + 1] - cs[start]
    first_dep = off & (within == 1)
    tab["t6"][0] = int(first_dep.sum())
    tab["t6"][1] = int((off & (within > 1)).sum())

    # ---- T7 : what sits at the first departure -----------------------------
    if first_dep.any():
        prev_ok = adjacent
        prev_mod = np.empty(n, dtype=np.int8)
        prev_mod[0] = MOD_NULL
        prev_mod[1:] = mod[:-1]
        prev_def = np.empty(n, dtype=np.int8)
        prev_def[0] = DEF_NULL
        prev_def[1:] = defer[:-1]
        prev_ast = np.empty(n, dtype=np.int8)
        prev_ast[0] = AST_NULL
        prev_ast[1:] = assist[:-1]

        mod_act = (mod == MOD_Y) | (prev_ok & (prev_mod == MOD_Y))
        def_act = (defer == DEF_C) | (prev_ok & (prev_def == DEF_C))
        ast_act = (assist != AST_NULL) | (prev_ok & (prev_ast != AST_NULL))
        any_act = mod_act | def_act | ast_act

        b = rb[first_dep]
        np.add.at(tab["t7"][:, 0], b, 1)
        for col, m in ((1, mod_act), (2, def_act), (3, ast_act), (4, any_act)):
            sel = first_dep & m
            if sel.any():
                np.add.at(tab["t7"][:, col], rb[sel], 1)
        sel = first_dep & ~any_act
        if sel.any():
            np.add.at(tab["t7"][:, 5], rb[sel], 1)
        sel = first_dep & ~prev_ok
        if sel.any():
            np.add.at(tab["t7"][:, 6], rb[sel], 1)
        stats["first_dep_no_prior_month"] = int((first_dep & ~prev_ok).sum())

    # ---- T8 : secondary reference, entry fixed at 3 ------------------------
    ladder3 = 3 + run - 1
    expected3 = np.minimum(ladder3, STATUS_CAP)
    dev3 = obs - expected3
    on3 = is90 & (dev3 == 0)
    below3 = is90 & (dev3 < 0)
    above3 = is90 & (dev3 > 0)
    cap3 = is90 & (obs == STATUS_CAP) & (ladder3 > STATUS_CAP)
    for r, m in ((0, is90), (1, keep)):
        tab["t8"][r, C_ON] = int((on3 & m).sum())
        tab["t8"][r, C_BELOW] = int((below3 & m).sum())
        tab["t8"][r, C_ABOVE] = int((above3 & m).sum())
        tab["t8"][r, C_CAP] = int((cap3 & m).sum())

    # ---- off_above examples ------------------------------------------------
    if off_above.any():
        idx = np.flatnonzero(off_above)[:N_OFF_EXAMPLES]
        for i in idx:
            examples.append((
                str(loan_index[int(loan[i])]), ym(int(period[i])),
                int(obs[i]), int(entry[i]), int(run[i]), int(expected[i]), int(dev[i]),
            ))
    return tab, stats, examples


# --------------------------------------------------------------------------
# report helpers
# --------------------------------------------------------------------------
def step_row(label, row, den_label=None):
    """One on_step / off_below / off_above / cap_row line with shares."""
    den = int(row[C_ON] + row[C_BELOW] + row[C_ABOVE])
    line = "%-26s %13s" % (label, fmt(den))
    for col in (C_ON, C_BELOW, C_ABOVE, C_CAP):
        line += " | %12s %8s" % (fmt(row[col]), pct(row[col], den, 2))
    return line, den


def step_header(first="slice"):
    head = "%-26s %13s" % (first, "")
    sub = "%-26s %13s" % ("", "loan-months")
    for name in ["on_step", "off_below", "off_above", "cap_row"]:
        head += " |%s" % name.center(22)
        sub += " | %12s %8s" % ("n", "share")
    emit(head)
    emit(sub)
    rule()


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    rule("=")
    emit("10_dq90_lockstep - does field-4 status move in lockstep with run-length inside DQ90+?")
    rule("=")
    paths = sorted(glob.glob(GLOB))
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : pass 1  0 loan, 1 period, 3 dq_status")
    emit("              pass 2  0 loan, 1 period, 3 dq_status, 7 mod_flag, 8 zero_balance_code,")
    emit("                      24 payment_deferral_flag, 29 borrower_assistance_plan")
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-64s %6.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    emit()

    pre_registration()
    part_a()

    rule("=")
    emit("PART B - lockstep measurement")
    rule("=")
    emit("entry_status    : numeric field-4 status in the run's first month")
    emit("k               : run-month, 1-based")
    emit("expected_status : min(entry_status + (k - 1), 99)")
    emit("deviation       : observed_status - expected_status")
    emit("on_step         : deviation == 0")
    emit("off_below       : deviation < 0")
    emit("off_above       : deviation > 0   - should not occur; reported as an anomaly")
    emit("cap_row         : observed == 99 AND entry_status + (k - 1) > 99")
    emit("                  on_step by construction, so it is a SUBSET of the on_step column")
    emit("                  and is reported beside it, never added to it")
    emit()
    emit("Denominator for every share in T1, T2 and T8 is on_step + off_below + off_above,")
    emit("i.e. all DQ90+ loan-months in that slice. cap_row shares use the same denominator.")
    emit()
    emit("THIN-CELL THRESHOLDS: %s loan-months, %s loans. Every cell below either threshold is"
         % (fmt(THIN_MONTHS), fmt(THIN_LOANS)))
    emit("marked '<< THIN' in place and listed again at the end.")
    emit()
    if not paths:
        emit("!! no files matched the glob - nothing to do")
        write_out()
        return

    tot = blank_tables()
    totals = {k: 0 for k in ACC_KEYS}
    examples, prefix_report = [], []
    period_lo = period_hi = None
    order_report = []

    emit("PROGRESS")
    rule()
    for path in paths:
        name = os.path.basename(path)
        t0 = time.time()
        loans, rows, bad1, prefixes, pmin, pmax, order = pass1_dq90_loans(path)
        t1 = time.time()
        emit("    %-18s pass 1  rows %12s   DQ90+ loans %9s            (%5.1fs)"
             % (name, fmt(rows), fmt(loans.size), t1 - t0))

        lc, pe, st, zb, md, df, ast, bad2 = pass2_collect(path, pd.Index(loans))
        t2 = time.time()
        emit("    %-18s pass 2  rows kept %8s%36s(%5.1fs)" % (name, fmt(lc.size), "", t2 - t1))

        tabs, stats, ex = walk_file(lc, pe, st, zb, md, df, ast, pd.Index(loans))
        del lc, pe, st, zb, md, df, ast
        for k in tot:
            tot[k] += tabs[k]
        if len(examples) < N_OFF_EXAMPLES:
            examples.extend(ex[:N_OFF_EXAMPLES - len(examples)])
        t3 = time.time()
        emit("    %-18s walk    DQ90+ months %7s  runs %10s          (%5.1fs)"
             % (name, fmt(stats["dq90_months"]), fmt(stats["runs"]), t3 - t2))

        totals["rows"] += rows
        totals["bad_period"] += bad1 + bad2
        for k in ACC_KEYS:
            if k not in ("rows", "bad_period"):
                totals[k] += stats[k]
        prefix_report.append((name, sorted(prefixes.items(), key=lambda kv: -kv[1])))
        order_report.append((name, order["loan_desc"], order["period_desc"]))
        period_lo = pmin if period_lo is None else min(period_lo, pmin)
        period_hi = pmax if period_hi is None else max(period_hi, pmax)
        emit()

    # ---------------------------------------------------------------- B.1 --
    rule("=")
    emit("B.1  SLICE SIZE, AND AGREEMENT WITH DIAGNOSTIC 09")
    rule("=")
    emit("%-58s %18s" % ("panel rows scanned (loan-months)", fmt(totals["rows"])))
    emit("%-58s %18s" % ("total DQ90+ loan-months", fmt(totals["dq90_months"])))
    emit("%-58s %18s" % ("distinct loans contributing >= 1 DQ90+ month",
                         fmt(totals["dq90_loans"])))
    emit("%-58s %18s" % ("distinct DQ90+ runs", fmt(totals["runs"])))
    emit("%-58s %18s" % ("DQ90+ share of all panel loan-months",
                         pct(totals["dq90_months"], totals["rows"], 4)))
    emit("%-58s %18s" % ("reporting periods observed",
                         "%s - %s" % (ym(period_lo), ym(period_hi))))
    emit("%-58s %18s" % ("rows dropped for an unparseable period", fmt(totals["bad_period"])))
    emit()
    emit("Cross-check against 09's published slice - the definitions are shared, so these must")
    emit("match exactly:")
    for label, got, want in [
        ("panel rows scanned", totals["rows"], EXPECTED_ROWS),
        ("DQ90+ loan-months", totals["dq90_months"], EXPECTED_DQ90_MONTHS),
        ("DQ90+ loans", totals["dq90_loans"], EXPECTED_DQ90_LOANS),
    ]:
        emit("    %-28s this run %14s   09 %14s   %s"
             % (label, fmt(got), fmt(want), "MATCH" if got == want else "!! DIFFERS"))
    emit()
    emit("Loan identifiers do not straddle files. Vintage prefix observed per file:")
    for name, prefs in prefix_report:
        emit("    %-18s %s" % (name, ", ".join("%s (%s rows)" % (p, fmt(c)) for p, c in prefs)))
    emit()

    # ---------------------------------------------------------------- B.2 --
    rule("=")
    emit("B.2  SORT-ORDER VERIFICATION (checked, not assumed)")
    rule("=")
    for line in wrap(
        "Each file is scanned in physical order and every row compared with the row before it, "
        "with the last (loan, period) carried across chunk boundaries so a violation sitting "
        "on a boundary is caught like any other. A 'loan out of order' is a row whose loan "
        "identifier is strictly less than the previous row's; a 'period out of order' is a row "
        "with the same loan as the row before but a strictly earlier period. Nothing in this "
        "script depends on the answer - the arrays are lexsorted by (loan, period) before use, "
        "as in 09 - so this is an audit of the files, not a precondition.", RULE_W - 4):
        emit("    %s" % line)
    emit()
    emit("    %-20s %20s %22s" % ("file", "loan out of order", "period out of order"))
    rule()
    tl = tp = 0
    for name, ld, pd_ in order_report:
        tl += ld
        tp += pd_
        emit("    %-20s %20s %22s" % (name, fmt(ld), fmt(pd_)))
    rule()
    emit("    %-20s %20s %22s" % ("all files", fmt(tl), fmt(tp)))
    emit()
    if tl == 0 and tp == 0:
        emit("    No violations. Every file is sorted by loan identifier, then by period.")
    else:
        emit("    !! %s loan-order and %s period-order violations. Rows are NOT fully sorted;"
             % (fmt(tl), fmt(tp)))
        emit("       the lexsort in walk_file() is load-bearing, not merely defensive.")
    emit()

    # ----------------------------------------------------------------- T1 --
    rule("=")
    emit("T1  OVERALL - on_step, off_below, off_above, cap_row")
    rule("=")
    emit("Unit: DQ90+ loan-months. 'primary' is 09's slice exactly. 'ex-terminal ZB' removes")
    emit("loan-months that are themselves the loan's last observed row carrying a terminating")
    emit("zero-balance code (01/02/03/09/15/16/96).")
    emit()
    step_header()
    prim_den = 0
    for r, label in ((0, "primary"), (1, "ex-terminal ZB")):
        line, den = step_row(label, tot["t1"][r])
        note_thin("T1", label, den, "loan-months")
        emit(line + thin_mark(den, "loan-months"))
        if r == 0:
            prim_den = den
    rule()
    emit()
    off_prim = int(tot["t1"][0, C_BELOW] + tot["t1"][0, C_ABOVE])
    off_ex = int(tot["t1"][1, C_BELOW] + tot["t1"][1, C_ABOVE])
    den_ex = int(tot["t1"][1, C_ON] + tot["t1"][1, C_BELOW] + tot["t1"][1, C_ABOVE])
    emit("PRIMARY METRIC - share of DQ90+ loan-months with deviation != 0, own entry_status:")
    emit("    primary slice        %14s / %14s  =  %s"
         % (fmt(off_prim), fmt(prim_den), pct(off_prim, prim_den, 3)))
    emit("    ex-terminal ZB slice %14s / %14s  =  %s"
         % (fmt(off_ex), fmt(den_ex), pct(off_ex, den_ex, 3)))
    emit()
    emit("    Pre-registered: Yassine under 5%; Claude 20-50%, point ~30%.")
    emit("    Recorded for the register; no reading of the result is offered here.")
    emit()
    emit("    cap_row is a subset of on_step: %s of %s on_step loan-months (%s) sit at the 99"
         % (fmt(tot["t1"][0, C_CAP]), fmt(tot["t1"][0, C_ON]),
            pct(tot["t1"][0, C_CAP], tot["t1"][0, C_ON], 3)))
    emit("    cap, where the uncapped ladder had already passed 99.")
    emit()

    # ----------------------------------------------------------------- T2 --
    rule("=")
    emit("T2  BY RUN-LENGTH BUCKET (09's buckets)")
    rule("=")
    for key, title in (("t2", "T2a  primary slice"), ("t2b", "T2b  ex-terminal ZB slice")):
        emit(title)
        rule()
        step_header("run (months)")
        for i, (label, _, _) in enumerate(BUCKETS):
            line, den = step_row(label, tot[key][i])
            note_thin("T2a" if key == "t2" else "T2b", "run %s" % label, den, "loan-months")
            emit(line + thin_mark(den, "loan-months"))
        rule()
        line, den = step_row("all", tot[key].sum(axis=0))
        emit(line)
        emit()

    # ----------------------------------------------------------------- T3 --
    rule("=")
    emit("T3  DISTRIBUTION OF entry_status")
    rule("=")
    emit("entry_status is the numeric field-4 status in the first month of a run. Three counts,")
    emit("on three different units - read the column headings, they do not agree by design:")
    emit("    runs              every DQ90+ run, counted once. Partitions the run set.")
    emit("    loans, first run  each loan filed under the entry_status of its EARLIEST run.")
    emit("                      Partitions the loan set.")
    emit("    loans, any run    each loan counted under every entry_status it ever enters at.")
    emit("                      Does NOT partition - a loan with two runs at different entry")
    emit("                      statuses appears twice, so this column over-sums.")
    emit()
    run_tot = int(tot["t3"][:, 0].sum())
    loan_tot = int(tot["t3"][:, 1].sum())
    any_tot = int(tot["t3"][:, 2].sum())
    emit("%-14s | %14s %8s | %14s %8s | %14s %8s"
         % ("entry_status", "runs", "share", "loans, 1st run", "share", "loans, any run", "share"))
    rule()
    for i, label in enumerate(ENTRY_LABELS):
        r = tot["t3"][i]
        note_thin("T3", "entry_status %s (runs)" % label, r[0], "loan-months")
        note_thin("T3", "entry_status %s (loans, first run)" % label, r[1], "loans")
        note_thin("T3", "entry_status %s (loans, any run)" % label, r[2], "loans")
        emit("%-14s | %14s %8s | %14s %8s | %14s %8s%s"
             % (label, fmt(r[0]), pct(r[0], run_tot, 2),
                fmt(r[1]), pct(r[1], loan_tot, 2),
                fmt(r[2]), pct(r[2], loan_tot, 2),
                thin_mark(min(int(r[1]), int(r[2])), "loans")))
    rule()
    emit("%-14s | %14s %8s | %14s %8s | %14s %8s"
         % ("all", fmt(run_tot), "100.00%", fmt(loan_tot), "100.00%", fmt(any_tot), ""))
    emit()
    emit("    'loans, first run' sums to %s, the distinct DQ90+ loan count in B.1 (%s)."
         % (fmt(loan_tot), fmt(totals["dq90_loans"])))
    emit("    'loans, any run' sums to %s; the excess over %s is loans entering DQ90+ at more"
         % (fmt(any_tot), fmt(loan_tot)))
    emit("    than one distinct entry_status across separate runs.")
    emit()

    # ----------------------------------------------------------------- T4 --
    rule("=")
    emit("T4  OBSERVED STATUS x RUN LENGTH - loan-month counts")
    rule("=")
    emit("Rows: run length in months. Columns: observed field-4 status. Cell = DQ90+ loan-months,")
    emit("with the share of that ROW (that run-length bucket) beneath it. If status and run")
    emit("length were locked together the mass would sit on the diagonal.")
    emit()
    head = "%-12s %13s" % ("run (months)", "row total")
    for label, _, _ in ST_BUCKETS:
        head += " |%s" % ("status " + label).center(17)
    emit(head)
    rule()
    for i, (label, _, _) in enumerate(BUCKETS):
        den = int(tot["t4"][i].sum())
        line = "%-12s %13s" % (label, fmt(den))
        for j in range(len(ST_BUCKETS)):
            c = int(tot["t4"][i, j])
            note_thin("T4", "run %s x status %s" % (label, ST_BUCKETS[j][0]), c, "loan-months")
            line += " | %9s %6s" % (fmt(c), pct(c, den, 1))
        emit(line + thin_mark(den, "loan-months"))
    rule()
    col = tot["t4"].sum(axis=0)
    den = int(col.sum())
    line = "%-12s %13s" % ("all", fmt(den))
    for j in range(len(ST_BUCKETS)):
        line += " | %9s %6s" % (fmt(col[j]), pct(col[j], den, 1))
    emit(line)
    emit()

    # ----------------------------------------------------------------- T5 --
    rule("=")
    emit("T5  DEVIATION SIZE AMONG off_below")
    rule("=")
    emit("Unit: DQ90+ loan-months with deviation < 0, primary slice. Denominator is the")
    emit("off_below count, not the whole slice.")
    emit()
    below_tot = int(tot["t5"].sum())
    emit("%-16s %16s %10s %18s" % ("deviation", "loan-months", "share", "share of slice"))
    rule()
    for i, (label, _, _) in enumerate(DEV_BUCKETS):
        c = int(tot["t5"][i])
        note_thin("T5", "deviation %s" % label, c, "loan-months")
        emit("%-16s %16s %10s %18s%s"
             % (label, fmt(c), pct(c, below_tot, 2), pct(c, prim_den, 3),
                thin_mark(c, "loan-months")))
    rule()
    emit("%-16s %16s %10s %18s"
         % ("all off_below", fmt(below_tot), "100.00%", pct(below_tot, prim_den, 3)))
    emit()
    recon = int(tot["t1"][0, C_BELOW])
    emit("    Reconciles with T1 off_below (%s): %s"
         % (fmt(recon), "yes" if recon == below_tot else "NO - %s unbucketed" % fmt(recon - below_tot)))
    emit()

    # ----------------------------------------------------------------- T6 --
    rule("=")
    emit("T6  STICKINESS - is an off-step month the run's first departure, or downstream of one?")
    rule("=")
    for line in wrap(
        "Within a run, departures are counted from the run's opening row forward. A loan-month "
        "is the FIRST departure when it is the earliest off-step month of its own run; it is "
        "DOWNSTREAM when an earlier month of the same run was already off-step. Once field 4 "
        "falls behind the ladder it cannot rejoin it, because the ladder keeps advancing by one "
        "a month and field 4 can advance by at most one a month - so a high downstream share is "
        "arithmetic, and the first-departure count is the number of distinct departure EVENTS.",
            RULE_W - 4):
        emit("    %s" % line)
    emit()
    off_all = int(tot["t6"].sum())
    emit("%-34s %16s %12s %20s" % ("", "loan-months", "share", "share of slice"))
    rule()
    for i, label in enumerate(["first departure of its run", "downstream of an earlier one"]):
        c = int(tot["t6"][i])
        note_thin("T6", label, c, "loan-months")
        emit("%-34s %16s %12s %20s%s"
             % (label, fmt(c), pct(c, off_all, 2), pct(c, prim_den, 3),
                thin_mark(c, "loan-months")))
    rule()
    emit("%-34s %16s %12s %20s"
         % ("all off-step loan-months", fmt(off_all), "100.00%", pct(off_all, prim_den, 3)))
    emit()
    emit("    Reconciles with T1 off_below + off_above (%s): %s"
         % (fmt(off_prim), "yes" if off_prim == off_all else "NO"))
    emit("    Distinct departure events per DQ90+ run: %s first departures / %s runs = %.3f"
         % (fmt(tot["t6"][0]), fmt(totals["runs"]),
            (tot["t6"][0] / totals["runs"]) if totals["runs"] else 0.0))
    emit()

    # ----------------------------------------------------------------- T7 --
    rule("=")
    emit("T7  FIRST DEPARTURES - when they happen, and what is flagged at the time")
    rule("=")
    emit("Unit: first-departure loan-months (T6 row 1), filed under the run-month at which the")
    emit("departure occurs. A departure is credited to a flag if the flag fires at the departure")
    emit("month itself OR at the immediately preceding observed month, one period earlier:")
    emit("    modification   flag value 'Y'  (ONSET - the field is persistent, A.3)")
    emit("    deferral       flag value 'C'  (ONSET - the field is persistent, A.3)")
    emit("    assistance     flag value F/R/T (VALUE - the field is current-month, A.3)")
    emit("The three are not exclusive, so mod + defer + assist may exceed 'any of the three'.")
    emit("'no prior month' counts departures whose preceding observed row is absent or is not")
    emit("exactly one period earlier; for those, only the departure month itself was testable.")
    emit("The run-month 1 row is empty by construction, not for want of data: at k = 1 the")
    emit("expected status IS the entry status, so deviation is 0 and no run can first depart there.")
    emit()
    head = "%-12s %12s" % ("run-month", "first dep.")
    for nm in ["modification", "deferral", "assistance", "any of three", "none of three"]:
        head += " |%s" % nm.center(17)
    head += " | no prior mo."
    emit(head)
    rule()
    for i, (label, _, _) in enumerate(BUCKETS):
        r = tot["t7"][i]
        den = int(r[0])
        note_thin("T7", "run-month %s" % label, den, "loan-months")
        line = "%-12s %12s" % (label, fmt(den))
        for j in (1, 2, 3, 4, 5):
            line += " | %9s %6s" % (fmt(r[j]), pct(r[j], den, 1))
        line += " | %12s" % fmt(r[6])
        emit(line + thin_mark(den, "loan-months"))
    rule()
    r = tot["t7"].sum(axis=0)
    den = int(r[0])
    line = "%-12s %12s" % ("all", fmt(den))
    for j in (1, 2, 3, 4, 5):
        line += " | %9s %6s" % (fmt(r[j]), pct(r[j], den, 1))
    line += " | %12s" % fmt(r[6])
    emit(line)
    emit()
    emit("    'none of the three' overall: %s of %s first departures (%s)."
         % (fmt(r[5]), fmt(den), pct(r[5], den, 2)))
    emit("    Borrower Assistance Plan is null before January 2014 by the Guide's own note, so")
    emit("    its column is structurally empty for departures before that date (A.3).")
    emit()

    # ----------------------------------------------------------------- T8 --
    rule("=")
    emit("T8  PRIMARY METRIC WITH THE SECONDARY REFERENCE - entry_status fixed at 3")
    rule("=")
    emit("Identical arithmetic to T1, except expected_status = min(3 + (k - 1), 99) for every")
    emit("run regardless of the status it actually entered at. For comparison only.")
    emit()
    step_header()
    for r, label in ((0, "primary"), (1, "ex-terminal ZB")):
        line, den = step_row(label, tot["t8"][r])
        note_thin("T8", label, den, "loan-months")
        emit(line + thin_mark(den, "loan-months"))
    rule()
    emit()
    off8 = int(tot["t8"][0, C_BELOW] + tot["t8"][0, C_ABOVE])
    emit("SECONDARY METRIC - share with deviation != 0, entry fixed at 3:")
    emit("    primary slice        %14s / %14s  =  %s"
         % (fmt(off8), fmt(prim_den), pct(off8, prim_den, 3)))
    emit("    same slice, own entry_status (T1) :  %s" % pct(off_prim, prim_den, 3))
    emit()

    # ------------------------------------------------------------- THIN ----
    rule("=")
    emit("THIN CELLS (< %s loan-months, or < %s loans)" % (fmt(THIN_MONTHS), fmt(THIN_LOANS)))
    rule("=")
    if not _THIN:
        emit("    None. Every cell reported above clears its threshold.")
    else:
        emit("    %-8s %-46s %14s %14s" % ("table", "cell", "n", "threshold"))
        rule()
        for table, cell, n, unit, thr in _THIN:
            emit("    %-8s %-46s %14s %14s"
                 % (table, cell, fmt(n), "%s %s" % (fmt(thr), unit)))
    emit()

    # ------------------------------------------------------- ANOMALIES -----
    rule("=")
    emit("ANOMALIES AND CONSTRUCTION COUNTS")
    rule("=")
    n_above = int(tot["t1"][0, C_ABOVE])
    emit("(a) off_above - observed status ABOVE the ladder")
    emit("      %s loan-months, %s of the primary slice."
         % (fmt(n_above), pct(n_above, prim_den, 5)))
    for line in wrap(
        "Field 4 counts months owed from DDLPI, so it can rise by at most one per calendar "
        "month, while the ladder rises by exactly one. Inside an unbroken run the deviation is "
        "therefore non-increasing and starts at zero, which makes off_above impossible unless a "
        "row's status jumps by more than one month in one month.", RULE_W - 10):
        emit("      %s" % line)
    if n_above:
        emit("      First %d examples:" % min(N_OFF_EXAMPLES, len(examples)))
        emit("        %-16s %-8s %8s %8s %6s %10s %8s"
             % ("loan id", "period", "observed", "entry", "k", "expected", "dev"))
        for lid, per, obs_, ent, k, exp_, d in examples[:N_OFF_EXAMPLES]:
            emit("        %-16s %-8s %8d %8d %6d %10d %8d" % (lid, per, obs_, ent, k, exp_, d))
    else:
        emit("      None. No example loan IDs to list.")
    emit()
    emit("(b) Runs broken by a reporting gap")
    emit("      %s DQ90+ loan-months open a new run because the immediately preceding month was"
         % fmt(totals["gap_breaks"]))
    emit("      never reported although the previous observed row was itself DQ90+. Share of all")
    emit("      DQ90+ loan-months: %s. Each of these resets k to 1, which also resets"
         % pct(totals["gap_breaks"], totals["dq90_months"], 4))
    emit("      entry_status to that row's status - so a gap manufactures an on_step row.")
    emit()
    emit("(c) Duplicate (loan, period) rows")
    emit("      %s, among rows belonging to DQ90+ loans. A repeated period is not adjacent, so"
         % fmt(totals["duplicate_period_rows"]))
    emit("      it restarts a run exactly as 09 has it.")
    emit()
    emit("(d) Terminal DQ90+ rows")
    emit("      %s DQ90+ loan-months are themselves the loan's last observed row with a"
         % fmt(totals["terminal_dq90_rows"]))
    emit("      terminating zero-balance code. These are the rows removed in the ex-terminal")
    emit("      slice of T1, T2b and T8.")
    emit()
    emit("(e) First departures with no testable prior month")
    emit("      %s of %s first departures. Their preceding observed row is absent or is not"
         % (fmt(totals["first_dep_no_prior_month"]), fmt(tot["t6"][0])))
    emit("      exactly one period earlier, so only the departure month itself was tested in T7.")
    emit()
    emit("(f) Values outside the documented enumerations, among rows of DQ90+ loans")
    emit("      zero-balance code         %12s" % fmt(totals["unknown_zbc_rows"]))
    emit("      modification flag         %12s" % fmt(totals["unknown_mod_rows"]))
    emit("      payment deferral flag     %12s" % fmt(totals["unknown_def_rows"]))
    emit("      borrower assistance plan  %12s" % fmt(totals["unknown_ast_rows"]))
    emit("      Documented: ZB 01/02/03/09/15/16/96 or blank; mod Y/P or blank; deferral C/P or")
    emit("      blank; assistance F/R/T or blank.")
    emit()
    rule("=")
    emit("done in %.1fs" % (time.time() - t_start))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
