"""Does the delinquency status ever drop by more than one rung in a single month?

The mirror of diagnostic 14. Field 4 counts months of payments owed, so between
two strictly consecutive observed months it can fall by more than one rung only
if the borrower paid off several months of arrears at once, or if the counter was
re-based by something other than payment - a modification, a payment deferral,
the completion of a workout. Those two are indistinguishable in the status column
alone, and a chained forecast reads both as a large cure. This script counts the
falls, sizes them, locates them in the state space, tests them against the
modification and assistance flags, and follows them forward.

SCOPE - strictly consecutive pairs only
    A pair is (row at period p, row at period p+1) for the same loan, both
    observed. Pairs spanning an interior reporting gap are EXCLUDED, as in
    diagnostic 14; the excluded count is reported in section A and reconciled
    against diagnostics 13 and 14, whose output files are read if present.

DEFINITION
    fall   a consecutive pair with numeric status(t) - status(t+1) >= 2
    Field 4 is read as text and cast only where .str.isdigit() holds, so "RA"
    and any blank are never coerced. A pair with a non-numeric end is not a
    fall; those pairs are counted and broken down, never silently dropped.
    A fall is a FULL CURE when status(t+1) == 0 and a PARTIAL FALL otherwise.
    Status 0 and status 1 cannot produce a fall, so every fall starts at
    status >= 2; those rows are carried in the tables and show zero.

COLUMN INDICES - resolved from the published layout, never guessed
    docs/file_layout_july_2026.xlsx is read with the standard library only (the
    workbook is a zip of XML). The MONTHLY PERFORMANCE DATA FILE sheet is located
    by its own title, each attribute is matched by name, and the published
    1-based field position minus one is the 0-based index used here. Every index
    actually used is printed in the output header together with the attribute
    text it was matched on, so a layout revision shows up as a changed header
    rather than as a silently wrong column.

STATE MAPPING - the project's existing one, unchanged
        numeric 0   Current      numeric 1   DQ30      numeric 2   DQ60
        numeric >= 3  DQ90+   (98 and 99 included; per D12 the 99 is the cap)
        "RA"  and  anything else  are carried as their own categories

FLAG SEMANTICS
    A flag is SET when its field is non-blank. The full observed value
    distribution of each flag column is printed before section D, so "non-blank"
    can be read against what the codes actually are rather than assumed.
    Two forms are reported for every flag:
        at t+1              the flag on the row the fall lands on
        at or before t+1    the flag on that row or on any earlier observed row
                            of the same loan, computed as a within-loan running
                            "any", which never leaks across loans
    The comparison group is NON-FALL consecutive pairs leaving the SAME
    status(t), so the two shares are conditioned identically. Skips and ordinary
    one-rung moves are both inside the non-fall group.

METHOD - one pass, one file at a time
    Loan ids are a per-file constant 5-character prefix plus a 7-digit tail and
    diagnostic 01 reports zero cross-file loan overlap, so a loan lives inside
    one file. Each file is packed one int64 per row:

        key = (loan << 36) | (period << 16) | payload

    where the 16-bit payload carries the status and the four flag codes (the
    exact bit layout is defined just below the imports), and sorted once on that
    single key, so adjacency in the sorted array is adjacency in the loan's
    history. Nothing assumes the raw file is sorted or a loan's rows contiguous.
    Look-ahead in section E locates period p+k by binary search on the key, not
    by row offset, so a hole is reported as a hole rather than mistaken for the
    next month.

No plots. No panel-construction code. No git command is run.
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
OUT = os.path.join("diagnostics", "15_status_falls.txt")
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")
DIAG13 = os.path.join("diagnostics", "13_reporting_gaps.txt")
DIAG14 = os.path.join("diagnostics", "14_status_skips.txt")
RULE_W = 121

PERF_SHEET_MARKER = "monthly performance data file"

# attributes resolved from the layout: (label, attribute text to match)
WANTED = [
    ("loan", "loan identifier"),
    ("period", "period"),
    ("status", "current loan delinquency status"),
    ("mod", "modification flag"),
    ("defer", "payment deferral flag"),
    ("disaster", "delinquency due to disaster"),
    ("assist", "borrower assistance plan"),
]
# the three structural columns are additionally pinned to the indices the rest
# of the project already uses; a mismatch stops the run rather than proceeding.
EXPECTED = {"loan": 0, "period": 1, "status": 3}

# ---- bit layout of the packed key (low bits first) ----
ST_BITS = 7           # status + 2, range 0..101
MOD_BITS = 2
DEF_BITS = 2
DIS_BITS = 2
ASS_BITS = 3
PAY_BITS = ST_BITS + MOD_BITS + DEF_BITS + DIS_BITS + ASS_BITS   # 16
PER_BITS = 20
ST_SH, MOD_SH = 0, ST_BITS
DEF_SH = MOD_SH + MOD_BITS
DIS_SH = DEF_SH + DEF_BITS
ASS_SH = DIS_SH + DIS_BITS
PER_SH = PAY_BITS
LOAN_SH = PAY_BITS + PER_BITS
ST_MASK = (1 << ST_BITS) - 1
PER_MASK = (1 << PER_BITS) - 1
ST_OFFSET = 2

RAW_RA = -1
RAW_OTHER = -2

ST_CUR, ST_DQ30, ST_DQ60, ST_DQ90, ST_RA, ST_OTHER = range(6)
ST_LABELS = ["Current", "DQ30", "DQ60", "DQ90+", "RA", "other"]
NSTATE = 6
NCORE = 4

DROP_LABELS = ["2", "3", "4", "5", "6-11", "12+"]
NDROP = len(DROP_LABELS)

STAT_LABELS = ["0", "1", "2", "3", "4", "5", "6-11", "12+"]
NSTAT = len(STAT_LABELS)

RUN_LABELS = ["1", "2-3", "4-6", "7-12", "13+"]
NRUN = len(RUN_LABELS)

FUT_LABELS = ["status 0", "status 1", "status 2", "status >= 3", "RA",
              "other", "loan ends", "gap"]
NFUT = len(FUT_LABELS)
F_END = -100
F_GAP = -101

# flag value encodings; anything unlisted lands in the final "other" slot
FLAG_CODES = {
    "mod": (["", "P", "Y"], MOD_SH, (1 << MOD_BITS) - 1),
    "defer": (["", "P", "C"], DEF_SH, (1 << DEF_BITS) - 1),
    "disaster": (["", "Y"], DIS_SH, (1 << DIS_BITS) - 1),
    "assist": (["", "F", "R", "T"], ASS_SH, (1 << ASS_BITS) - 1),
}
FLAG_ORDER = ["mod", "defer", "disaster", "assist"]
FLAG_TITLES = {
    "mod": "MODIFICATION FLAG",
    "defer": "PAYMENT DEFERRAL FLAG",
    "disaster": "DELINQUENCY DUE TO DISASTER",
    "assist": "BORROWER ASSISTANCE PLAN",
}

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


def pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def cs(count, den, cw=12, sw=10, dp=2):
    return ("%" + str(cw) + "s%" + str(sw) + "s") % (fmt(count), pct(count, den, dp))


# --------------------------------------------------------------------------
# layout resolution
# --------------------------------------------------------------------------
def load_layout(path):
    """[(field position, attribute text)] from the performance sheet of the xlsx."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

    def cell_text(c):
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(ns + "t"))
        v = c.find(ns + "v")
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    for name in sorted(n for n in z.namelist()
                       if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
        root = ET.fromstring(z.read(name))
        rows = [[cell_text(c) for c in row.iter(ns + "c")]
                for row in root.iter(ns + "row")]
        head = " ".join(" ".join(r) for r in rows[:6]).lower()
        if PERF_SHEET_MARKER not in head:
            continue
        out = []
        for r in rows:
            cells = [x.strip() for x in r if x.strip()]
            if len(cells) >= 2 and re.match(r"^\d+$", cells[0]):
                out.append((int(cells[0]), cells[1], name))
        if out:
            return out, name
    raise SystemExit("could not locate the %r sheet in %s"
                     % (PERF_SHEET_MARKER, path))


def resolve_indices():
    entries, sheet = load_layout(LAYOUT)
    resolved = {}
    for label, want in WANTED:
        hits = [e for e in entries if e[1].strip().lower() == want]
        if not hits:
            hits = [e for e in entries if want in e[1].strip().lower()]
        if len(hits) != 1:
            raise SystemExit(
                "layout lookup for %r matched %d row(s) in %s - refusing to guess"
                % (want, len(hits), LAYOUT))
        pos, attribute, _ = hits[0]
        if not 1 <= pos <= 40:
            raise SystemExit("implausible field position %r for %r" % (pos, want))
        resolved[label] = (pos - 1, pos, attribute)
    for label, expect in EXPECTED.items():
        got = resolved[label][0]
        if got != expect:
            raise SystemExit(
                "layout gives index %d for %r but the project uses %d - stopping"
                % (got, label, expect))
    return resolved, sheet


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def encode_period(s):
    v = pd.to_numeric(s, errors="coerce")
    year, month = v // 100, v % 100
    idx = (year * 12 + (month - 1)).where((month >= 1) & (month <= 12))
    idx = idx.fillna(-1).astype(np.int64)
    idx[(idx < 0) | (idx > PER_MASK)] = -1
    return idx


def encode_status(s):
    digits = s.str.isdigit().to_numpy()
    num = pd.to_numeric(s.where(s.str.isdigit()), errors="coerce")
    num = num.fillna(RAW_OTHER).to_numpy()
    code = np.full(len(s), RAW_OTHER, dtype=np.int16)
    code[digits] = num[digits]
    code[(~digits) & (s.to_numpy() == "RA")] = RAW_RA
    code[(code > 99) | (code < RAW_OTHER)] = RAW_OTHER
    return code


def encode_flag(s, name):
    values, _, mask = FLAG_CODES[name]
    other = len(values)
    if other > mask:
        raise RuntimeError("flag %r needs more bits" % name)
    code = np.full(len(s), other, dtype=np.int8)
    arr = s.to_numpy()
    for i, v in enumerate(values):
        code[arr == v] = i
    return code


def flag_labels(name):
    values, _, _ = FLAG_CODES[name]
    return ["(blank)" if v == "" else v for v in values] + ["other"]


def map4(raw):
    st = np.full(raw.shape, ST_OTHER, dtype=np.int8)
    st[raw == RAW_RA] = ST_RA
    st[raw == 0] = ST_CUR
    st[raw == 1] = ST_DQ30
    st[raw == 2] = ST_DQ60
    st[raw >= 3] = ST_DQ90
    return st


def bucket_drop(d):
    return np.select([d == 2, d == 3, d == 4, d == 5, d <= 11],
                     [0, 1, 2, 3, 4], default=5).astype(np.int64)


def bucket_status(s):
    return np.select([s == 0, s == 1, s == 2, s == 3, s == 4, s == 5, s <= 11],
                     [0, 1, 2, 3, 4, 5, 6], default=7).astype(np.int64)


def bucket_run(r):
    return np.select([r == 1, r <= 3, r <= 6, r <= 12],
                     [0, 1, 2, 3], default=4).astype(np.int64)


def bucket_future(s):
    b = np.full(s.shape, -1, dtype=np.int8)
    b[s == 0] = 0
    b[s == 1] = 1
    b[s == 2] = 2
    b[s >= 3] = 3
    b[s == RAW_RA] = 4
    b[s == RAW_OTHER] = 5
    b[s == F_END] = 6
    b[s == F_GAP] = 7
    if (b < 0).any():
        raise RuntimeError("bucket_future left %d row(s) unclassified"
                           % int((b < 0).sum()))
    return b


# --------------------------------------------------------------------------
class Acc(object):
    def __init__(self):
        self.rows = 0
        self.bad_period = 0
        self.bad_loan = 0
        self.loans = 0
        self.pairs_cons = 0
        self.pairs_gap = 0
        self.pairs_dup = 0
        self.pair_both_num = 0
        self.pair_t_nonnum = 0
        self.pair_t1_nonnum = 0
        self.pair_both_nonnum = 0
        self.falls = 0
        self.skips = 0
        self.full_cures = 0
        self.partial = 0
        self.drop_min = None
        self.drop_max = None
        self.drop_hist = np.zeros(NDROP, dtype=np.int64)
        self.drop_by_stat = np.zeros((NSTAT, NDROP), dtype=np.int64)
        self.cure_by_stat = np.zeros(NSTAT, dtype=np.int64)
        self.fall_cells = np.zeros((NSTATE, NSTATE), dtype=np.int64)
        self.all_cells = np.zeros((NSTATE, NSTATE), dtype=np.int64)
        # D: per flag, per status bucket, [pairs, at t+1, ever by t+1]
        self.d_fall = {f: np.zeros((NSTAT, 3), dtype=np.int64) for f in FLAG_ORDER}
        self.d_all = {f: np.zeros((NSTAT, 3), dtype=np.int64) for f in FLAG_ORDER}
        self.flag_values = {f: np.zeros(len(flag_labels(f)), dtype=np.int64)
                            for f in FLAG_ORDER}
        # E: [origin group 0 = status 2, 1 = status >=3][t+2 / t+3][bucket]
        self.future = np.zeros((2, 2, NFUT), dtype=np.int64)
        self.run_map = np.zeros((2, NRUN), dtype=np.int64)
        self.year_hist = {}
        self.prefixes = {}

    def bump(self, arr, lo, hi):
        if not len(arr):
            return
        a, b = int(arr.min()), int(arr.max())
        setattr(self, lo, a if getattr(self, lo) is None else min(getattr(self, lo), a))
        setattr(self, hi, b if getattr(self, hi) is None else max(getattr(self, hi), b))


def reader_for(path, cols):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=cols,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


def read_keys(path, acc, idx):
    parts = []
    cols = sorted(idx[k][0] for k in idx)
    c_loan = idx["loan"][0]
    c_per = idx["period"][0]
    c_st = idx["status"][0]
    for chunk in reader_for(path, cols):
        acc.rows += len(chunk)
        loan = chunk[c_loan].str.strip()
        per = encode_period(chunk[c_per].str.strip())
        raw = encode_status(chunk[c_st].str.strip())

        pre = loan.str[:5]
        for p, c in pre.value_counts().items():
            acc.prefixes[p] = acc.prefixes.get(p, 0) + int(c)
        tail = loan.str[5:]
        good_loan = tail.str.isdigit().to_numpy()
        acc.bad_loan += int((~good_loan).sum())
        lint = pd.to_numeric(tail.where(tail.str.isdigit()), errors="coerce")
        lint = lint.fillna(-1).astype(np.int64).to_numpy()

        payload = (raw.astype(np.int64) + ST_OFFSET)
        for f in FLAG_ORDER:
            code = encode_flag(chunk[idx[f][0]].str.strip(), f)
            acc.flag_values[f] += np.bincount(
                code, minlength=len(flag_labels(f)))
            payload = payload | (code.astype(np.int64) << FLAG_CODES[f][1])

        okp = (per >= 0).to_numpy()
        acc.bad_period += int((~okp).sum())
        ok = okp & good_loan
        if not ok.any():
            continue
        parts.append((lint[ok] << LOAN_SH)
                     | (per.to_numpy()[ok] << PER_SH)
                     | payload[ok])

    if not parts:
        return np.zeros(0, dtype=np.int64)
    key = np.concatenate(parts)
    del parts
    key.sort(kind="stable")
    return key


def within_loan_ever(ind, is_first, n):
    """True where ind is True on this row or any earlier row of the SAME loan."""
    ar = np.arange(n, dtype=np.int32)
    last_true = np.maximum.accumulate(np.where(ind, ar, np.int32(-1)))
    grp_start = np.maximum.accumulate(np.where(is_first, ar, np.int32(-1)))
    return last_true >= grp_start


def process_file(path, acc, idx):
    key = read_keys(path, acc, idx)
    n = len(key)
    if n == 0:
        return

    payload = key & ((1 << PAY_BITS) - 1)
    raw = ((payload & ST_MASK) - ST_OFFSET).astype(np.int16)
    per = ((key >> PER_SH) & PER_MASK).astype(np.int32)
    loan = key >> LOAN_SH

    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    np.not_equal(loan[1:], loan[:-1], out=is_first[1:])
    acc.loans += int(is_first.sum())

    starts = np.nonzero(is_first)[0]
    is_last = np.empty(n, dtype=bool)
    is_last[:-1] = is_first[1:]
    is_last[-1] = True
    last_per = per[is_last]

    same = ~is_first[1:]
    dp = per[1:] - per[:-1]
    cons = same & (dp == 1)
    acc.pairs_cons += int(cons.sum())
    acc.pairs_gap += int((same & (dp > 1)).sum())
    acc.pairs_dup += int((same & (dp == 0)).sum())

    idxp = np.nonzero(cons)[0]
    a = raw[idxp]
    b = raw[idxp + 1]
    a_num = a >= 0
    b_num = b >= 0
    both = a_num & b_num
    acc.pair_both_num += int(both.sum())
    acc.pair_t_nonnum += int((~a_num & b_num).sum())
    acc.pair_t1_nonnum += int((a_num & ~b_num).sum())
    acc.pair_both_nonnum += int((~a_num & ~b_num).sum())

    np.add.at(acc.all_cells, (map4(a), map4(b)), 1)

    diff = np.where(both, a.astype(np.int32) - b.astype(np.int32), 0)
    fall_m = both & (diff >= 2)
    acc.falls += int(fall_m.sum())
    acc.skips += int((both & (diff <= -2)).sum())

    drops = diff[fall_m]
    acc.bump(drops, "drop_min", "drop_max")
    acc.drop_hist += np.bincount(bucket_drop(drops), minlength=NDROP)

    # ---- D: flag association, falls vs non-falls leaving the same status(t)
    # denominators are built over ALL consecutive pairs with a numeric status(t)
    stat_all = np.where(a_num, bucket_status(a.astype(np.int32)), -1)
    valid_all = stat_all >= 0
    for f in FLAG_ORDER:
        code = ((payload >> FLAG_CODES[f][1]) & FLAG_CODES[f][2]).astype(np.int8)
        setf = code > 0
        ever = within_loan_ever(setf, is_first, n)
        at_t1 = setf[idxp + 1]
        ev_t1 = ever[idxp + 1]
        for target, m in ((acc.d_all[f], valid_all),
                          (acc.d_fall[f], valid_all & fall_m)):
            sb = stat_all[m]
            target[:, 0] += np.bincount(sb, minlength=NSTAT)
            target[:, 1] += np.bincount(sb, weights=at_t1[m],
                                        minlength=NSTAT).astype(np.int64)
            target[:, 2] += np.bincount(sb, weights=ev_t1[m],
                                        minlength=NSTAT).astype(np.int64)
        del code, setf, ever

    fi = idxp[fall_m]
    if not len(fi):
        return
    fa = raw[fi].astype(np.int32)
    fb = raw[fi + 1].astype(np.int32)

    acc.full_cures += int((fb == 0).sum())
    acc.partial += int((fb >= 1).sum())
    sb_f = bucket_status(fa)
    np.add.at(acc.drop_by_stat, (sb_f, bucket_drop(fa - fb)), 1)
    acc.cure_by_stat += np.bincount(sb_f[fb == 0], minlength=NSTAT)
    np.add.at(acc.fall_cells, (map4(raw[fi]), map4(raw[fi + 1])), 1)

    for y, c in zip(*np.unique((per[fi] // 12).astype(np.int64),
                               return_counts=True)):
        acc.year_hist[int(y)] = acc.year_hist.get(int(y), 0) + int(c)

    origin = (fa >= 3).astype(np.int64)      # 0 = from status 2, 1 = from >= 3

    # ---- E: status at t+2 and t+3, located by period, not by row offset
    g = np.searchsorted(starts, fi, side="right") - 1
    loan_last = last_per[g]
    base = (key[fi] >> LOAN_SH) << LOAN_SH
    for col, k in enumerate((2, 3)):
        want = per[fi].astype(np.int64) + k
        target = base | (want << PER_SH)
        pos = np.searchsorted(key, target, side="left")
        safe = np.minimum(pos, n - 1)
        found = (pos < n) & (key[safe] < target + (1 << PER_SH))
        s_k = np.where(found, raw[safe].astype(np.int32), F_GAP)
        s_k = np.where(want > loan_last, F_END, s_k)
        np.add.at(acc.future, (origin, col, bucket_future(s_k)), 1)

    # ---- F: run length on the mapped state, ending at t
    ms = map4(raw)
    ar = np.arange(n, dtype=np.int64)
    newrun = np.empty(n, dtype=bool)
    newrun[0] = True
    newrun[1:] = is_first[1:] | (dp != 1) | (ms[1:] != ms[:-1])
    run_start = np.maximum.accumulate(np.where(newrun, ar, 0))
    np.add.at(acc.run_map, (origin, bucket_run(ar[fi] - run_start[fi] + 1)), 1)


# --------------------------------------------------------------------------
def scrape(path, needle):
    """First integer on the line containing needle, from an earlier diagnostic."""
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8"):
        if needle in line:
            nums = re.findall(r"\b\d[\d,]*\b", line.replace(needle, ""))
            if nums:
                return int(nums[0].replace(",", ""))
    return None


def main():
    t_start = time.time()
    idx, sheet = resolve_indices()
    paths = sorted(glob.glob(GLOB))
    acc = Acc()

    rule("=")
    emit("15_status_falls - single-month drops of two or more delinquency rungs")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("layout file : %s" % LAYOUT)
    emit("            : MONTHLY PERFORMANCE DATA FILE sheet matched on %s" % sheet)
    emit()
    emit("COLUMN INDICES RESOLVED FROM THE LAYOUT (0-based index = position - 1)")
    rule()
    emit("%-12s%10s%10s   %s" % ("role", "position", "index", "attribute name"))
    rule()
    for label, _ in WANTED:
        i, pos, attribute = idx[label]
        emit("%-12s%10d%10d   %s" % (label, pos, i, attribute))
    rule()
    emit("indices 0, 1 and 3 additionally checked against the values the rest of the")
    emit("project uses; a mismatch stops the run instead of proceeding.")
    emit()
    emit("scope       : strictly consecutive month pairs within one loan;")
    emit("              pairs spanning an interior reporting gap are EXCLUDED (see 13)")
    emit("fall        : numeric status(t) - status(t+1) >= 2")
    emit("full cure   : a fall with status(t+1) == 0 ; partial fall: status(t+1) >= 1")
    emit("flag is set : the field is non-blank (value distributions printed below)")
    emit("states      : 0 Current, 1 DQ30, 2 DQ60, >=3 DQ90+ (98/99 inside DQ90+)")
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-64s %6.2f GB" % (p, os.path.getsize(p) / 1024.0 ** 3))
    if not paths:
        emit()
        emit("NO PERFORMANCE FILES MATCHED THE GLOB - nothing measured.")
        write_out()
        return
    emit()

    emit("PASS 1/1 - pack, sort and walk each file")
    rule()
    for path in paths:
        t0 = time.time()
        before = (acc.rows, acc.pairs_cons, acc.falls)
        process_file(path, acc, idx)
        emit("    %-18s rows %12s   consecutive pairs %12s   falls %9s   (%5.1fs)"
             % (os.path.basename(path), fmt(acc.rows - before[0]),
                fmt(acc.pairs_cons - before[1]), fmt(acc.falls - before[2]),
                time.time() - t0))
    emit("    pooled rows %s   loans %s   consecutive pairs %s   falls %s"
         % (fmt(acc.rows), fmt(acc.loans), fmt(acc.pairs_cons), fmt(acc.falls)))
    emit("    rows rejected - unparseable period %s, unparseable loan id %s"
         % (fmt(acc.bad_period), fmt(acc.bad_loan)))
    emit()

    P = acc.pairs_cons
    F = acc.falls

    # ---------------- A ----------------
    rule("=")
    emit("A. HEADLINE")
    rule("=")
    emit("Unit: consecutive month pairs.")
    emit()
    emit("%-58s %18s %12s" % ("quantity", "count", "share"))
    rule()
    emit("%-58s %18s %12s" % ("consecutive pairs (denominator)", fmt(P), "100.0000%"))
    emit("%-58s %18s %12s" % ("falls: status drops by >= 2 in one month",
                              fmt(F), pct(F, P)))
    emit("%-58s %18s %12s" % ("    of which full cures (status(t+1) == 0)",
                              fmt(acc.full_cures), pct(acc.full_cures, P)))
    emit("%-58s %18s %12s" % ("    of which partial falls (status(t+1) >= 1)",
                              fmt(acc.partial), pct(acc.partial, P)))
    rule()
    emit("share of FALLS that are full cures:    %s of %s (%s)"
         % (fmt(acc.full_cures), fmt(F), pct(acc.full_cures, F)))
    emit("share of FALLS that are partial falls: %s of %s (%s)"
         % (fmt(acc.partial), fmt(F), pct(acc.partial, F)))
    rule()
    emit("%-58s %18s %12s" % ("pairs with both ends numeric",
                              fmt(acc.pair_both_num), pct(acc.pair_both_num, P)))
    emit("%-58s %18s %12s" % ("pairs with status(t) non-numeric only",
                              fmt(acc.pair_t_nonnum), pct(acc.pair_t_nonnum, P)))
    emit("%-58s %18s %12s" % ("pairs with status(t+1) non-numeric only",
                              fmt(acc.pair_t1_nonnum), pct(acc.pair_t1_nonnum, P)))
    emit("%-58s %18s %12s" % ("pairs with both ends non-numeric",
                              fmt(acc.pair_both_nonnum),
                              pct(acc.pair_both_nonnum, P)))
    emit()
    emit("RECONCILIATION")
    rule()
    identity = acc.rows - acc.loans - acc.pairs_gap - acc.pairs_dup
    emit("%-58s %18s" % ("gap-spanning pairs EXCLUDED from every table above",
                         fmt(acc.pairs_gap)))
    emit("%-58s %18s" % ("duplicate (loan, period) pairs", fmt(acc.pairs_dup)))
    emit("%-58s %18s" % ("rows - loans - gap pairs - duplicate pairs", fmt(identity)))
    emit("%-58s %18s" % ("consecutive pairs counted above", fmt(P)))
    emit("%-58s %18s" % ("identity holds", "yes" if identity == P else "NO"))
    rule()
    d14_falls = scrape(DIAG14, "falls: status drops by >= 2 in one month")
    d14_skips = scrape(DIAG14, "skips: status rises by >= 2 in one month")
    d14_pairs = scrape(DIAG14, "consecutive pairs (denominator)")
    d13_gaps = scrape(DIAG13, "interior gap events (holes, not months)")
    for label, mine, theirs, src in (
            ("falls", F, d14_falls, "14_status_skips.txt"),
            ("skips (recomputed here)", acc.skips, d14_skips, "14_status_skips.txt"),
            ("consecutive pairs", P, d14_pairs, "14_status_skips.txt"),
            ("gap-spanning pairs", acc.pairs_gap, d13_gaps, "13_reporting_gaps.txt")):
        if theirs is None:
            emit("%-40s here %16s   %s not found"
                 % (label, fmt(mine), src))
        else:
            emit("%-40s here %16s   %s %16s   %s"
                 % (label, fmt(mine), src, fmt(theirs),
                    "match" if mine == theirs else "MISMATCH"))
    emit()

    if F == 0:
        rule("=")
        emit("NO FALLS OBSERVED - sections B to G are empty.")
        rule("=")
        write_out()
        return

    # ---------------- B ----------------
    rule("=")
    emit("B. FALL MAGNITUDE BY status(t)   (unit: fall events)")
    rule("=")
    emit("Drop size is status(t) - status(t+1) in raw field-4 rungs. Status 0 and 1")
    emit("cannot drop by two, so those rows are structurally zero.")
    emit()
    emit("%-14s%13s%12s%s" % ("status(t)", "falls", "full cures",
                              "".join("%12s" % l for l in DROP_LABELS)))
    rule()
    for i, lab in enumerate(STAT_LABELS):
        row = acc.drop_by_stat[i]
        emit("%-14s%13s%12s%s" % (lab, fmt(row.sum()), fmt(acc.cure_by_stat[i]),
                                  "".join("%12s" % fmt(c) for c in row)))
    rule()
    tot = acc.drop_by_stat.sum(axis=0)
    emit("%-14s%13s%12s%s" % ("TOTAL", fmt(F), fmt(acc.full_cures),
                              "".join("%12s" % fmt(c) for c in tot)))
    rule()
    emit("%-14s%13s%12s%s" % ("share", "100.00%", pct(acc.full_cures, F, 2),
                              "".join("%12s" % pct(c, F, 2) for c in tot)))
    emit()
    emit("%-38s %s" % ("min drop size (rungs)", fmt(acc.drop_min)))
    emit("%-38s %s" % ("max drop size (rungs)", fmt(acc.drop_max)))
    emit()

    # ---------------- C ----------------
    rule("=")
    emit("C. STATE-SPACE IMPACT")
    rule("=")
    emit("Both ends mapped to the four-state space. Falls have numeric ends by")
    emit("definition, so they occupy only the 4x4 block; RA and 'other' appear in the")
    emit("all-transitions table because they are part of the panel.")
    emit()
    emit("C.1  TRANSITION CELLS POPULATED BY FALLS   (unit: fall events)")
    rule()
    emit("%-16s%14s%s" % ("from \\ to", "falls",
                          "".join("%14s" % l for l in ST_LABELS[:NCORE])))
    rule()
    for i in range(NCORE):
        row = acc.fall_cells[i][:NCORE]
        emit("%-16s%14s%s" % (ST_LABELS[i], fmt(row.sum()),
                              "".join("%14s" % fmt(c) for c in row)))
    rule()
    emit("%-16s%14s%s" % ("TOTAL into", fmt(acc.fall_cells[:NCORE, :NCORE].sum()),
                          "".join("%14s" % fmt(c)
                                  for c in acc.fall_cells[:NCORE, :NCORE].sum(axis=0))))
    emit()
    emit("C.2  ALL CONSECUTIVE-PAIR TRANSITIONS   (unit: consecutive pairs)")
    rule()
    emit("%-16s%16s%s" % ("from \\ to", "pairs",
                          "".join("%16s" % l for l in ST_LABELS)))
    rule()
    for i in range(NSTATE):
        row = acc.all_cells[i]
        emit("%-16s%16s%s" % (ST_LABELS[i], fmt(row.sum()),
                              "".join("%16s" % fmt(c) for c in row)))
    rule()
    emit("%-16s%16s%s" % ("TOTAL into", fmt(acc.all_cells.sum()),
                          "".join("%16s" % fmt(c)
                                  for c in acc.all_cells.sum(axis=0))))
    emit()
    emit("C.3  FALL SHARE OF EACH POPULATED CELL")
    rule()
    emit("For every cell that falls populate: how many of ALL the consecutive-pair")
    emit("transitions landing in that cell arrived there as a fall.")
    emit()
    emit("%-34s%16s%18s%14s" % ("transition cell", "from falls",
                                "all transitions", "fall share"))
    rule()
    for i in range(NCORE):
        for j in range(NCORE):
            f_ij = int(acc.fall_cells[i][j])
            if f_ij == 0:
                continue
            a_ij = int(acc.all_cells[i][j])
            emit("%-34s%16s%18s%14s"
                 % ("%s -> %s" % (ST_LABELS[i], ST_LABELS[j]),
                    fmt(f_ij), fmt(a_ij), pct(f_ij, a_ij, 4)))
    rule()
    emit("%-34s%16s%18s%14s" % ("ALL POPULATED CELLS", fmt(F),
                                fmt(acc.all_cells.sum()),
                                pct(F, acc.all_cells.sum(), 4)))
    emit()
    empty = ["%s -> %s" % (ST_LABELS[i], ST_LABELS[j])
             for i in range(NCORE) for j in range(NCORE)
             if acc.fall_cells[i][j] == 0]
    emit("cells in the 4x4 block that NO fall populates (%d of 16):" % len(empty))
    for e in empty:
        emit("    %s" % e)
    emit()

    # ---------------- flag distributions ----------------
    rule("=")
    emit("FLAG COLUMNS - observed value distribution over all rows")
    rule("=")
    emit("'Set' below means non-blank. Unit: loan-months (%s rows)." % fmt(acc.rows))
    emit()
    emit("%-32s%14s%16s%12s" % ("flag / value", "index", "rows", "share"))
    rule()
    for f in FLAG_ORDER:
        labs = flag_labels(f)
        for j, lab in enumerate(labs):
            emit("%-32s%14s%16s%12s"
                 % ("%s: %s" % (FLAG_TITLES[f], lab),
                    idx[f][0] if j == 0 else "", fmt(acc.flag_values[f][j]),
                    pct(acc.flag_values[f][j], acc.rows, 4)))
        rule()
    emit()

    # ---------------- D ----------------
    rule("=")
    emit("D. MODIFICATION AND LOSS-MITIGATION ASSOCIATION")
    rule("=")
    emit("Unit: consecutive pairs. Falls are compared with NON-FALL pairs leaving the")
    emit("SAME status(t), so the two columns are conditioned identically. The non-fall")
    emit("group contains every other move out of that status, including skips and")
    emit("ordinary one-rung changes.")
    emit("  at t+1            the flag on the row the pair lands on")
    emit("  by t+1            that row or any earlier observed row of the same loan")
    emit()
    for f in FLAG_ORDER:
        fa_t = acc.d_fall[f]
        al_t = acc.d_all[f]
        nf_t = al_t - fa_t
        emit("D.%d  %s   (column index %d)"
             % (FLAG_ORDER.index(f) + 1, FLAG_TITLES[f], idx[f][0]))
        rule()
        emit("%-10s%12s%10s%8s%10s%8s   %14s%11s%8s%11s%8s"
             % ("", "falls", "at t+1", "share", "by t+1", "share",
                "non-falls", "at t+1", "share", "by t+1", "share"))
        emit("%-10s%12s%10s%8s%10s%8s   %14s%11s%8s%11s%8s"
             % ("status(t)", "pairs", "count", "", "count", "",
                "pairs", "count", "", "count", ""))
        rule()
        for i, lab in enumerate(STAT_LABELS):
            fp, fa1, fe1 = (int(x) for x in fa_t[i])
            np_, na1, ne1 = (int(x) for x in nf_t[i])
            emit("%-10s%12s%10s%8s%10s%8s   %14s%11s%8s%11s%8s"
                 % (lab, fmt(fp), fmt(fa1), pct(fa1, fp, 1), fmt(fe1),
                    pct(fe1, fp, 1), fmt(np_), fmt(na1), pct(na1, np_, 1),
                    fmt(ne1), pct(ne1, np_, 1)))
        rule()
        fp, fa1, fe1 = (int(x) for x in fa_t.sum(axis=0))
        np_, na1, ne1 = (int(x) for x in nf_t.sum(axis=0))
        emit("%-10s%12s%10s%8s%10s%8s   %14s%11s%8s%11s%8s"
             % ("TOTAL", fmt(fp), fmt(fa1), pct(fa1, fp, 1), fmt(fe1),
                pct(fe1, fp, 1), fmt(np_), fmt(na1), pct(na1, np_, 1),
                fmt(ne1), pct(ne1, np_, 1)))
        emit()

    # ---------------- E ----------------
    n2 = int(acc.future[0][0].sum())
    n3 = int(acc.future[1][0].sum())
    rule("=")
    emit("E. PERSISTENCE - what the loan reports after the fall")
    rule("=")
    emit("Unit: fall events. t+2 and t+3 are located by the loan's reporting period,")
    emit("not by row offset.")
    emit("  'loan ends'  no observed row at any period >= that month")
    emit("  'gap'        the loan is observed later but not in that month")
    emit()
    emit("origin groups: status(t) == 2: %s falls   |   status(t) >= 3: %s falls"
         % (fmt(n2), fmt(n3)))
    emit()
    emit("%-24s%22s%22s%22s%22s"
         % ("", "origin 2: t+2", "origin 2: t+3",
            "origin >=3: t+2", "origin >=3: t+3"))
    emit("%-24s%22s%22s%22s%22s"
         % ("status at that month", "count     share", "count     share",
            "count     share", "count     share"))
    rule()
    for k, lab in enumerate(FUT_LABELS):
        emit("%-24s%s%s%s%s"
             % (lab, cs(acc.future[0][0][k], n2), cs(acc.future[0][1][k], n2),
                cs(acc.future[1][0][k], n3), cs(acc.future[1][1][k], n3)))
    rule()
    emit("%-24s%s%s%s%s"
         % ("TOTAL", cs(n2, n2), cs(int(acc.future[0][1].sum()), n2),
            cs(n3, n3), cs(int(acc.future[1][1].sum()), n3)))
    emit()

    # ---------------- F ----------------
    rule("=")
    emit("F. RUN LENGTH BEFORE THE FALL, ON THE MAPPED STATE   (unit: fall events)")
    rule("=")
    emit("Consecutive observed months the loan had already held the mapped state it")
    emit("was in at t, counting t itself as month 1. A reporting hole or a change of")
    emit("mapped state ends a run, so every run is unbroken and strictly monthly.")
    emit()
    tot_all = int(acc.run_map.sum())
    emit("%-24s%22s%22s%22s"
         % ("", "all falls", "origin 2", "origin >=3"))
    emit("%-24s%22s%22s%22s"
         % ("run length (months)", "count     share", "count     share",
            "count     share"))
    rule()
    for i, lab in enumerate(RUN_LABELS):
        emit("%-24s%s%s%s"
             % (lab, cs(acc.run_map[0][i] + acc.run_map[1][i], tot_all),
                cs(acc.run_map[0][i], n2), cs(acc.run_map[1][i], n3)))
    rule()
    emit("%-24s%s%s%s" % ("TOTAL", cs(tot_all, tot_all), cs(n2, n2), cs(n3, n3)))
    emit()

    # ---------------- G ----------------
    rule("=")
    emit("G. CALENDAR CONCENTRATION   (unit: fall events)")
    rule("=")
    emit("Filed under the YEAR of status(t) - the month before the drop.")
    emit()
    emit("%-22s%16s%12s%16s" % ("year of status(t)", "falls", "share",
                                "cumulative"))
    rule()
    run = 0
    for y in sorted(acc.year_hist):
        c = acc.year_hist[y]
        run += c
        emit("%-22s%16s%12s%16s" % (y, fmt(c), pct(c, F), pct(run, F)))
    rule()
    emit("%-22s%16s%12s%16s" % ("TOTAL", fmt(F), "100.0000%", "100.0000%"))
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t_start, (time.time() - t_start) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
