"""Exit hazards out of DQ90+ as a function of time already spent in DQ90+.

PART A - documentation lookup. Settled before any data was touched, against the
General User Guide and File Layout (July 2026) shipped in docs/. The question
was whether the monthly performance file carries a foreclosure-status marker -
referral, in-process, attorney assignment or similar - distinct from the
delinquency counter at field 4 (index 3). It does not. The finding and the
Guide's own words on field 4 are reproduced at the top of the output file; the
script re-confirms the four column positions it reads against the layout
workbook at run time, but does not re-derive Part A.

PART B - exit hazard by duration. For every loan-month in DQ90+, the run-length
so far and the state next month are recorded, and the pair is tabulated.

DEFINITIONS USED (inherited from the project decision register, not re-opened):
    DQ90+      numeric field-4 status >= 3. 98 and 99 are inside it: per D12
               99 is the documented CAP, not a sentinel, and 98 is the ordinary
               rung below it. The Guide gives "XX = Not Available" for missing.
    Default    zero-balance code 02 / 03 / 09, or field-4 status "RA", per D7.
               Code 96 WITH a prior RA is also Default, per D9.
    Prepaid    zero-balance code 01.
    Censored   codes 15 / 16 (D16), code 96 without prior RA (D17), or the loan
               running off the end of the panel with no zero-balance code.
    Cure       next-month numeric status 0, 1 or 2.
    Stay       next-month numeric status >= 3.

THREE JUDGEMENT CALLS, MADE EXPLICIT BECAUSE THEY MOVE NUMBERS:

  (1) Reporting gaps. The brief fixes the run rule: a gap in reporting periods
      breaks the run and the counter restarts at 1. The brief does NOT list a
      gap among the censoring conditions, so the OUTCOME for a loan-month is
      read from the next OBSERVED row whatever its period. The number of
      outcomes classified across a gap is reported as a footnote, broken out by
      outcome, so the size of the call is on record.

  (2) Terminal rows. A terminated loan is not updated after its zero-balance
      row (User Guide, "Zero Balance Codes"), so a DQ90+ month that is itself
      the zero-balance row has no successor. It is classified from its OWN
      zero-balance code rather than as censored-at-cutoff. Consequence: when a
      loan is DQ90+ in month p and the zero-balance row at p+1 is ALSO DQ90+,
      the same exit is attributed at two adjacent run-lengths - once as the
      next-month state of p, once as the terminal state of p+1. The count of
      terminal DQ90+ rows bounds that overlap and is reported.

  (3) RA is absorbing (D8) but records are NOT truncated at RA here, because
      the brief did not ask for truncation. This is immaterial by construction:
      RA is not a numeric status, so post-RA months cannot be DQ90+ months. The
      count of DQ90+ loan-months at or after a first RA is reported to confirm.

  The D10 exclusion (15 loans with an unbroken 00 run terminating in RA) is
  immaterial here by construction: those loans have no DQ90+ month at all and
  so enter no denominator.

UNITS. Two quantities appear and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation

TWO PASSES PER FILE, both chunked, both restricted to usecols [0, 1, 3, 8].
No file is ever held whole.
    pass 1  the set of loans that ever reach DQ90+ in that file.
    pass 2  every row of those loans only, kept as packed integer arrays
            (loan code int32, period index int32, status int16, zbc int16).
Loans do not straddle files - the loan identifier carries its own vintage
prefix - and the prefix seen in each file is reported so that assumption is
checked rather than assumed. Each file is reduced to a bucket x outcome count
matrix and its arrays are released before the next file is opened.

Every numeric cast of column 3 is guarded by .str.isdigit(); "RA" is never
coerced.
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
OUT = os.path.join("diagnostics", "09_dq90_duration_hazard.txt")
RULE_W = 121

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_ZBC = 8
USECOLS = [COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC]

LAYOUT_XLSX = os.path.join("docs", "file_layout_july_2026.xlsx")
LAYOUT_SHEET = "xl/worksheets/sheet2.xml"  # Monthly Performance Data File

EXPECTED_FIELDS = {
    COL_LOAN: "Loan Identifier",
    COL_PERIOD: "Period",
    COL_DQ: "Current Loan Delinquency Status",
    COL_ZBC: "Zero Balance Code",
}

# Field-4 status packing. Numeric statuses keep their own value (0..99, the
# documented cap); the two non-numeric outcomes get negative codes so that
# "status >= 3" is exactly the DQ90+ test and can never catch "RA" or "XX".
ST_RA = -1
ST_OTHER = -2

# Zero-balance packing. 0 is the blank (loan still active); -1 is any value
# outside the documented enumeration, which should not occur and is counted.
ZB_BLANK = 0
ZB_OTHER = -1
ZB_MAP = {"": 0, "01": 1, "02": 2, "03": 3, "09": 9, "15": 15, "16": 16, "96": 96}
ZB_DEFAULT = (2, 3, 9)
ZB_PREPAID = 1
ZB_ADMIN_CENSOR = (15, 16)
ZB_DEFECT = 96
ZB_TERMINATING = (1, 2, 3, 9, 15, 16, 96)

# Outcome codes, in the order the table columns appear.
O_CURE, O_STAY, O_DEFAULT, O_PREPAID, O_CENSORED, O_UNCL = 0, 1, 2, 3, 4, 5
OUTCOME_NAMES = ["Cure", "Stay", "Default", "Prepaid", "Censored"]
N_OUT = 6

BUCKETS = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-6", 4, 6),
    ("7-12", 7, 12),
    ("13-24", 13, 24),
    ("25+", 25, 10 ** 9),
]
THIN = 1_000

BIG_PERIOD = 1 << 30

_LINES = []


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


def reader_for(path):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=USECOLS,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


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


# --------------------------------------------------------------------------
# PART A - settled finding, reproduced verbatim
# --------------------------------------------------------------------------
PART_A_FINDING = (
    "No foreclosure-status field exists in the monthly performance file. "
    "Foreclosure progression is not observable as distinct from the delinquency "
    "counter at field 4 (index 3). Confirmed against the General User Guide and "
    "File Layout, July 2026, in docs/."
)

# Verbatim transcription of the Current Loan Delinquency Status entry from
# docs/general_user_guide_july_2026.pdf. Em-dashes and curly quotes in the
# source are rendered as their ASCII equivalents; nothing else is altered.
FIELD4_QUOTE = """\
Current Loan Delinquency Status

  Description
    "A value corresponding to the number of days the borrower is delinquent,
     based on the Due Date of Last Paid Installment ("DDLPI") reported by
     servicers to Freddie Mac.

     If a loan is acquired through REO, then the Current Loan Delinquency
     Status will reflect "RA" for REO Acquisition (instead of the value
     corresponding to the number of days the borrower is delinquent).

     This value for any given month will be capped at 99.

     Note: For periods on or before September 2025, 00 could mean either
     Current or Cycle before first payment date."

  Enumerations
    "00 = Current, or less than 30 days delinquent
     01 = 30-59 days delinquent
     02 = 60 - 89 days delinquent
     03 = 90 - 119 days delinquent etc.
     RA = REO Acquisition

     XX = Not Available\""""

# Fields that touch foreclosure or workout at all, with why each one does not
# answer the question. Keyed by 0-based index.
ADJACENT = [
    (3, "Current Loan Delinquency Status",
     "carries 'RA' = REO Acquisition. RA is the completion of foreclosure (title taken), "
     "not a process stage, and it is this same field rather than a distinct one."),
    (6, "Underwriting Defect and Major Servicing Defect Settlement Date",
     "a repurchase / make-whole settlement date. Unrelated to foreclosure process, and "
     "populated at settlement only."),
    (8, "Zero Balance Code",
     "02 / 03 / 09 identify a foreclosure-related TERMINATION after the fact. Blank until "
     "the terminating event, so it carries no in-process signal."),
    (9, "Zero Balance Effective Date",
     "the period of the terminating event. Terminal only."),
    (24, "Payment Deferral Flag",
     "a workout instrument, and the opposite of foreclosure."),
    (28, "Delinquency Due to Disaster",
     "hardship / forbearance flag. Guide: 'Only populated for January 2014 and following "
     "periods.'"),
    (29, "Borrower Assistance Plan",
     "F = Forbearance, R = Repayment, T = Trial Period. The closest thing in the file to a "
     "during-delinquency process marker, but it is a RETENTION workout, not foreclosure, and "
     "the Guide states 'Only populated for January 2014 and following periods' - i.e. null "
     "across the entire 2007-2011 window these vintages exist to observe."),
]


def read_layout():
    """Field position -> attribute name for the monthly performance file.

    Parsed straight out of the xlsx zip container with the standard library so
    no spreadsheet package has to be installed. Returns {0-based index: name}
    or None if the workbook is absent.
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
    emit("PART A - is foreclosure status observable separately from field 4?")
    rule("=")
    emit("sources : docs/general_user_guide_july_2026.pdf")
    emit("          docs/file_layout_july_2026.xlsx")
    emit()
    emit("A.1  FINDING")
    rule()
    for line in wrap(PART_A_FINDING, RULE_W - 4):
        emit("    %s" % line)
    emit()
    emit("    Field number: none. Index: none. Enumeration: none. There is nothing to report")
    emit("    on population during the delinquency period, because no such field exists.")
    emit()

    emit("A.2  FIELDS THAT TOUCH FORECLOSURE OR WORKOUT, AND WHY NONE OF THEM IS THAT FIELD")
    rule()
    for idx, name, why in ADJACENT:
        emit("    field %2d (index %2d)  %s" % (idx + 1, idx, name))
        for line in wrap(why, RULE_W - 10):
            emit("        %s" % line)
        emit()

    emit("A.3  CONSEQUENCE FOR PART B")
    rule()
    for line in wrap(
        "Foreclosure timing is not separately observable during the delinquency period. "
        "Between entry into DQ90+ and the terminal event, field 4 is the only state signal "
        "the file carries, so 'months in DQ90+' is simultaneously arrears depth and elapsed "
        "foreclosure clock, and the two cannot be separated in this data. The process is "
        "observed only at its endpoint: 'RA' in field 4 at REO acquisition, or a zero-balance "
        "code at termination. A run-length of k months in DQ90+ therefore cannot be read as "
        "'k months since foreclosure referral'.", RULE_W - 4):
        emit("    %s" % line)
    emit()

    emit("A.4  FIELD 4 (INDEX 3) - THE GUIDE'S EXACT WORDS, INCLUDING CAP BEHAVIOUR")
    rule()
    for line in FIELD4_QUOTE.splitlines():
        emit("    %s" % line.rstrip())
    emit()
    emit("    The cap sentence, quoted alone, since the DQ90+ definition in Part B rests on it:")
    emit()
    emit('        "This value for any given month will be capped at 99."')
    emit()
    for line in wrap(
        "Read with the enumeration ('03 = 90 - 119 days delinquent etc.'), 99 is the top rung "
        "of the same ladder - 99 or more months delinquent - and not a missing-value sentinel; "
        "the Guide supplies 'XX = Not Available' for that purpose. The Guide says nothing "
        "special about 98 at all, which is itself the point: 98 is simply the ordinary rung "
        "below the cap. Both therefore fold into DQ90+ with every other status >= 3, matching "
        "decision D12 and diagnostic 06.", RULE_W - 4):
        emit("    %s" % line)
    emit()
    for line in wrap(
        "The cap has moved historically: the release notes record 'Remove the Delinquency "
        "Status cap of D180', so the 99 cap quoted above is the current regime, which is the "
        "one these files are published under.", RULE_W - 4):
        emit("    %s" % line)
    emit()

    emit("A.5  COLUMN POSITIONS USED BY PART B, RE-CONFIRMED AGAINST THE LAYOUT WORKBOOK")
    rule()
    layout = read_layout()
    if layout is None:
        emit("    !! layout workbook not found at %s - positions NOT confirmed" % LAYOUT_XLSX)
        emit()
        return
    emit("    layout lists %d field positions for the monthly performance file." % len(layout))
    mismatches = []
    for idx, expected in sorted(EXPECTED_FIELDS.items()):
        actual = layout.get(idx, "(absent)")
        ok = actual.strip().lower() == expected.strip().lower()
        if not ok:
            mismatches.append(idx)
        emit("      index %d  ->  %-38s %s" % (idx, actual, "OK" if ok else "MISMATCH"))
    if mismatches:
        emit("    !! layout mismatch at index(es) %s - column positions are NOT confirmed"
             % ", ".join(str(i) for i in mismatches))
    else:
        emit("    all four positions confirmed.")
    emit()


# --------------------------------------------------------------------------
# PART B - encoding helpers
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


# --------------------------------------------------------------------------
# PART B - per-file passes
# --------------------------------------------------------------------------
def pass1_dq90_loans(path):
    """Loans in this file that reach DQ90+ at least once, plus row/period tallies."""
    seen, rows, bad_period, prefixes = [], 0, 0, {}
    pmin, pmax = None, None
    for chunk in reader_for(path):
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
        hit = loan.values[status.values >= 3]
        if hit.size:
            seen.append(pd.unique(hit))
    loans = pd.unique(np.concatenate(seen)) if seen else np.array([], dtype=object)
    return loans, rows, bad_period, prefixes, pmin, pmax


def pass2_collect(path, loan_index):
    """Every row belonging to the DQ90+ loan set, packed into integer arrays."""
    codes, periods, statuses, zbcs = [], [], [], []
    bad_period = 0
    for chunk in reader_for(path):
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
    if not codes:
        empty_i32 = np.array([], dtype=np.int32)
        empty_i16 = np.array([], dtype=np.int16)
        return empty_i32, empty_i32, empty_i16, empty_i16, bad_period
    return (
        np.concatenate(codes), np.concatenate(periods),
        np.concatenate(statuses), np.concatenate(zbcs), bad_period,
    )


def bucket_of(run):
    """Run length -> bucket row index, vectorised."""
    out = np.full(run.shape, len(BUCKETS) - 1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(BUCKETS):
        out[(run >= lo) & (run <= hi)] = i
    return out


EMPTY_STATS = (
    "dq90_months", "dq90_loans", "gap_breaks", "duplicate_period_rows",
    "terminal_dq90_rows", "months_after_first_ra", "unknown_zbc_rows",
    "gap_classified",
)


def walk_file(loan, period, status, zbc):
    """Run lengths and next-month outcomes for every DQ90+ loan-month in a file.

    Arrays arrive unsorted; they are sorted by (loan, period) here. The whole
    reduction is vectorised - no per-loan Python loop - and returns a
    bucket x outcome count matrix plus the diagnostic tallies.
    """
    stats = {k: 0 for k in EMPTY_STATS}
    zero_table = np.zeros((len(BUCKETS), N_OUT), dtype=np.int64)
    zero_gap = np.zeros(N_OUT, dtype=np.int64)
    n = loan.size
    if n == 0:
        return zero_table, stats, zero_gap

    order = np.lexsort((period, loan))
    loan = loan[order]
    period = period[order].astype(np.int64)
    status = status[order]
    zbc = zbc[order]

    is90 = status >= 3
    stats["dq90_months"] = int(is90.sum())
    stats["unknown_zbc_rows"] = int((zbc == ZB_OTHER).sum())
    if stats["dq90_months"] == 0:
        return zero_table, stats, zero_gap
    stats["dq90_loans"] = int(np.unique(loan[is90]).size)

    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = loan[1:] == loan[:-1]
    prev_period = np.empty(n, dtype=np.int64)
    prev_period[0] = -BIG_PERIOD
    prev_period[1:] = period[:-1]
    prev_is90 = np.zeros(n, dtype=bool)
    prev_is90[1:] = is90[:-1]

    adjacent = same_prev & (period == prev_period + 1)
    # A run continues only across an observed, immediately preceding DQ90+ month.
    cont = is90 & prev_is90 & adjacent
    # Broken by a hole in the reporting periods: the previous observed row is
    # the same loan and is itself DQ90+, but the month in between was never
    # reported. A break caused by an OBSERVED non-DQ90+ month is an ordinary
    # restart and is deliberately not counted here.
    stats["gap_breaks"] = int((is90 & prev_is90 & same_prev & (period > prev_period + 1)).sum())
    stats["duplicate_period_rows"] = int((same_prev & (period == prev_period)).sum())

    # Run length: every row that does not continue opens a new run, and the
    # distance from the run's opening row is the length so far. Non-DQ90+ rows
    # open runs of their own, which is what forces the restart after a cure;
    # their run values are never read.
    pos = np.arange(n, dtype=np.int64)
    start = np.maximum.accumulate(np.where(~cont, pos, 0))
    run = pos - start + 1

    # First RA month per loan, for the code-96 split (D9 / D17). RA rows are few.
    first_ra = np.full(int(loan.max()) + 1, BIG_PERIOD, dtype=np.int64)
    ra = status == ST_RA
    if ra.any():
        np.minimum.at(first_ra, loan[ra], period[ra])
    row_first_ra = first_ra[loan]
    stats["months_after_first_ra"] = int((is90 & (period >= row_first_ra)).sum())

    has_next = np.zeros(n, dtype=bool)
    has_next[:-1] = loan[:-1] == loan[1:]
    nxt_status = np.empty(n, dtype=np.int16)
    nxt_status[:-1] = status[1:]
    nxt_status[-1] = ST_OTHER
    nxt_zbc = np.empty(n, dtype=np.int16)
    nxt_zbc[:-1] = zbc[1:]
    nxt_zbc[-1] = ZB_BLANK
    nxt_period = np.empty(n, dtype=np.int64)
    nxt_period[:-1] = period[1:]
    nxt_period[-1] = BIG_PERIOD

    outcome = np.full(n, O_UNCL, dtype=np.int8)

    # ---- rows with a successor: that successor row is "next month" ---------
    # Precedence is terminal-event first. A zero-balance row ends the loan, so
    # its code overrides whatever field 4 happens to say on the same row.
    A = is90 & has_next
    ra_by_next = row_first_ra <= nxt_period
    a_def = A & ((nxt_status == ST_RA) | np.isin(nxt_zbc, ZB_DEFAULT)
                 | ((nxt_zbc == ZB_DEFECT) & ra_by_next))
    a_pre = A & ~a_def & (nxt_zbc == ZB_PREPAID)
    a_cen = A & ~a_def & ~a_pre & (np.isin(nxt_zbc, ZB_ADMIN_CENSOR)
                                   | ((nxt_zbc == ZB_DEFECT) & ~ra_by_next))
    rest = A & ~a_def & ~a_pre & ~a_cen
    outcome[a_def] = O_DEFAULT
    outcome[a_pre] = O_PREPAID
    outcome[a_cen] = O_CENSORED
    outcome[rest & (nxt_status >= 0) & (nxt_status <= 2)] = O_CURE
    outcome[rest & (nxt_status >= 3)] = O_STAY

    # ---- rows with no successor: the loan's record ends here ---------------
    B = is90 & ~has_next
    ra_here = row_first_ra <= period
    b_def = B & (np.isin(zbc, ZB_DEFAULT) | ((zbc == ZB_DEFECT) & ra_here))
    b_pre = B & ~b_def & (zbc == ZB_PREPAID)
    outcome[b_def] = O_DEFAULT
    outcome[b_pre] = O_PREPAID
    outcome[B & ~b_def & ~b_pre] = O_CENSORED
    stats["terminal_dq90_rows"] = int((B & np.isin(zbc, ZB_TERMINATING)).sum())

    table = np.zeros((len(BUCKETS), N_OUT), dtype=np.int64)
    np.add.at(table, (bucket_of(run[is90]), outcome[is90]), 1)

    gap_mask = A & (nxt_period > period + 1)
    stats["gap_classified"] = int(gap_mask.sum())
    gap_by_outcome = np.zeros(N_OUT, dtype=np.int64)
    np.add.at(gap_by_outcome, outcome[gap_mask], 1)
    return table, stats, gap_by_outcome


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def hazard_table(table):
    head = "%-8s %12s" % ("run", "")
    sub = "%-8s %12s" % ("(months)", "denominator")
    for name in OUTCOME_NAMES:
        head += " |%s" % name.center(18)
        sub += " | %10s %6s" % ("n", "rate")
    emit(head)
    emit(sub)
    rule()
    for i, (label, _, _) in enumerate(BUCKETS):
        den = int(table[i].sum())
        line = "%-8s %12s" % (label, fmt(den))
        for o in range(len(OUTCOME_NAMES)):
            line += " | %10s %6s" % (fmt(table[i, o]), pct(table[i, o], den, 2))
        if den < THIN:
            line += "   << THIN (n < %s)" % fmt(THIN)
        emit(line)
    rule()
    tot = table.sum(axis=0)
    den = int(tot.sum())
    line = "%-8s %12s" % ("all", fmt(den))
    for o in range(len(OUTCOME_NAMES)):
        line += " | %10s %6s" % (fmt(tot[o]), pct(tot[o], den, 2))
    emit(line)


def main():
    t_start = time.time()
    rule("=")
    emit("09_dq90_duration_hazard - exit hazards from DQ90+ by time already spent in DQ90+")
    rule("=")
    paths = sorted(glob.glob(GLOB))
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : 0 loan, 1 period, 3 dq_status, 8 zero_balance_code")
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-64s %6.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    emit()

    part_a()

    rule("=")
    emit("PART B - exit hazard by duration in DQ90+")
    rule("=")
    emit("DQ90+      : field-4 numeric status >= 3, 98 and 99 included (D12: 99 is the cap)")
    emit("run length : consecutive months in DQ90+; a reporting gap restarts the counter at 1")
    emit("outcome    : state at the next observed row, with a terminal zero-balance code on that")
    emit("             row taking precedence over its field 4")
    emit("Cure       : next status 0 / 1 / 2")
    emit("Stay       : next status >= 3")
    emit("Default    : zero-balance 02 / 03 / 09, or status RA, per D7; also 96 with a prior RA")
    emit("Prepaid    : zero-balance 01")
    emit("Censored   : zero-balance 15 / 16, 96 without a prior RA, or no next observation")
    emit()
    if not paths:
        emit("!! no files matched the glob - nothing to do")
        write_out()
        return

    table = np.zeros((len(BUCKETS), N_OUT), dtype=np.int64)
    gap_by_outcome = np.zeros(N_OUT, dtype=np.int64)
    totals = {k: 0 for k in EMPTY_STATS}
    totals.update({"rows": 0, "bad_period": 0, "kept_rows": 0})
    prefix_report, period_lo, period_hi = [], None, None

    emit("PROGRESS")
    rule()
    for path in paths:
        name = os.path.basename(path)
        t0 = time.time()
        loans, rows, bad1, prefixes, pmin, pmax = pass1_dq90_loans(path)
        t1 = time.time()
        emit("    %-18s pass 1  rows %12s   DQ90+ loans %9s            (%5.1fs)"
             % (name, fmt(rows), fmt(loans.size), t1 - t0))

        lc, pe, st, zb, bad2 = pass2_collect(path, pd.Index(loans))
        t2 = time.time()
        emit("    %-18s pass 2  rows kept %8s%36s(%5.1fs)" % (name, fmt(lc.size), "", t2 - t1))

        tab, stats, gaps = walk_file(lc, pe, st, zb)
        del lc, pe, st, zb
        table += tab
        gap_by_outcome += gaps
        t3 = time.time()
        emit("    %-18s walk    DQ90+ months %7s  contributing loans %8s  (%5.1fs)"
             % (name, fmt(stats["dq90_months"]), fmt(stats["dq90_loans"]), t3 - t2))

        totals["rows"] += rows
        totals["bad_period"] += bad1 + bad2
        for k in EMPTY_STATS:
            totals[k] += stats[k]
        prefix_report.append((name, sorted(prefixes.items(), key=lambda kv: -kv[1])))
        period_lo = pmin if period_lo is None else min(period_lo, pmin)
        period_hi = pmax if period_hi is None else max(period_hi, pmax)
        emit()

    def ym(idx):
        if idx is None:
            return "n/a"
        y, m = divmod(int(idx) - 1, 12)
        return "%04d%02d" % (y, m + 1)

    rule("=")
    emit("B.1  SLICE SIZE")
    rule("=")
    emit("%-58s %18s" % ("panel rows scanned (loan-months)", fmt(totals["rows"])))
    emit("%-58s %18s" % ("total DQ90+ loan-months", fmt(totals["dq90_months"])))
    emit("%-58s %18s" % ("distinct loans contributing >= 1 DQ90+ month",
                         fmt(totals["dq90_loans"])))
    emit("%-58s %18s" % ("DQ90+ share of all panel loan-months",
                         pct(totals["dq90_months"], totals["rows"], 4)))
    emit("%-58s %18s" % ("reporting periods observed", "%s - %s" % (ym(period_lo), ym(period_hi))))
    emit("%-58s %18s" % ("rows dropped for an unparseable period", fmt(totals["bad_period"])))
    emit()
    emit("Loan identifiers do not straddle files. Vintage prefix observed per file:")
    for name, prefs in prefix_report:
        emit("    %-18s %s" % (name, ", ".join("%s (%s rows)" % (p, fmt(c)) for p, c in prefs)))
    emit()

    rule("=")
    emit("B.2  EXIT HAZARD FROM DQ90+ BY RUN LENGTH")
    rule("=")
    emit("Unit: loan-months. One row per (loan, month) observed in DQ90+, filed under the number")
    emit("of months that loan had then been continuously in DQ90+. The denominator column is that")
    emit("bucket's DQ90+ loan-month count; each rate is that column's count / the denominator.")
    emit()
    hazard_table(table)
    emit()
    thin = [BUCKETS[i][0] for i in range(len(BUCKETS)) if int(table[i].sum()) < THIN]
    if thin:
        emit("THIN BUCKETS (denominator < %s): %s" % (fmt(THIN), ", ".join(thin)))
    else:
        emit("THIN BUCKETS (denominator < %s): none - every bucket clears the threshold."
             % fmt(THIN))
    emit()

    rule("=")
    emit("B.3  FOOTNOTES - counts behind the construction")
    rule("=")
    unclassified = int(table[:, O_UNCL].sum())
    emit("(a) Runs broken by a reporting gap")
    emit("      %s DQ90+ loan-months open a new run because the immediately preceding month was"
         % fmt(totals["gap_breaks"]))
    emit("      never reported, although the previous observed row for that loan was itself")
    emit("      DQ90+. The counter restarts at 1 at each of these, as briefed. Runs broken by an")
    emit("      OBSERVED non-DQ90+ month are ordinary restarts and are not counted here.")
    emit("      Share of all DQ90+ loan-months: %s"
         % pct(totals["gap_breaks"], totals["dq90_months"], 4))
    emit()
    emit("(b) Outcomes classified across a reporting gap")
    emit("      %s DQ90+ loan-months have their next OBSERVED row more than one month later, so"
         % fmt(totals["gap_classified"]))
    emit("      their 'state next month' is in fact 'state at next observation'. Share of all")
    emit("      DQ90+ loan-months: %s. By outcome:"
         % pct(totals["gap_classified"], totals["dq90_months"], 4))
    for o, nm in enumerate(OUTCOME_NAMES):
        emit("        %-10s %12s" % (nm, fmt(gap_by_outcome[o])))
    if gap_by_outcome[O_UNCL]:
        emit("        %-10s %12s" % ("unclass.", fmt(gap_by_outcome[O_UNCL])))
    emit()
    emit("(c) Terminal DQ90+ rows")
    emit("      %s DQ90+ loan-months are themselves the loan's zero-balance row, and are"
         % fmt(totals["terminal_dq90_rows"]))
    emit("      classified from their own code rather than as censored-at-cutoff. Where the row")
    emit("      before one of these is also DQ90+, the same exit is attributed at two adjacent")
    emit("      run-lengths. This count is the upper bound on that overlap.")
    emit()
    emit("(d) DQ90+ loan-months at or after a first RA")
    emit("      %s. RA is not a numeric status, so post-RA months cannot be DQ90+; a non-zero"
         % fmt(totals["months_after_first_ra"]))
    emit("      value here would mean a loan returned to a numeric status after REO acquisition")
    emit("      (D8 records 2 such loans in 118,026 across the panel).")
    emit()
    emit("(e) Duplicate (loan, period) rows")
    emit("      %s, among rows belonging to DQ90+ loans. A repeated period neither continues nor"
         % fmt(totals["duplicate_period_rows"]))
    emit("      gap-breaks a run.")
    emit()
    emit("(f) Zero-balance values outside the documented enumeration")
    emit("      %s rows, among rows belonging to DQ90+ loans. Documented enumeration is"
         % fmt(totals["unknown_zbc_rows"]))
    emit("      01 / 02 / 03 / 09 / 15 / 16 / 96, plus blank while the loan is still active.")
    emit()
    emit("(g) Unclassified DQ90+ loan-months")
    emit("      %s. A loan-month lands here only if its successor row carries no zero-balance"
         % fmt(unclassified))
    emit("      code and a field 4 that is neither numeric nor RA ('XX = Not Available', or")
    emit("      blank).")
    if unclassified:
        emit("      Per bucket: %s" % ", ".join(
            "%s: %s" % (BUCKETS[i][0], fmt(table[i, O_UNCL])) for i in range(len(BUCKETS))))
        emit("      These sit inside every denominator above, which is why the five rates need")
        emit("      not sum to exactly 100%.")
    else:
        emit("      Zero, so the five outcome rates sum to 100% in every bucket.")
    emit()
    rule("=")
    emit("done in %.1fs" % (time.time() - t_start))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
