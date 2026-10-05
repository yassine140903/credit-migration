"""Does the delinquency status ever jump by more than one rung in a single month?

Field 4 counts months of payments owed, so between two STRICTLY CONSECUTIVE
observed months it can rise by at most one rung if it is behaving as a monthly
arrears counter. A rise of two or more in one month is therefore either genuine
information the counter cannot express, or a reporting artefact - and the two
look identical in a transition matrix. Whichever it is, a chained forecast reads
those pairs as far off-diagonal migrations, which is where early default comes
from. This script counts them, sizes them, locates them in the state space, and
follows each one forward.

SCOPE - strictly consecutive pairs only
    A pair is (row at period p, row at period p+1) for the same loan, both
    observed. Pairs spanning an interior reporting gap are EXCLUDED: diagnostic
    13 measures those and they are dropped from the panel by decision. The count
    of excluded gap-spanning pairs is reported in section A so that the two
    diagnostics reconcile against one another.

DEFINITION
    skip   a consecutive pair with numeric status(t+1) - status(t) >= 2
    fall   a consecutive pair with numeric status(t) - status(t+1) >= 2, the
           cure-side analogue, reported in section A with its own count
    Field 4 is read as text and cast only where .str.isdigit() holds, so "RA"
    and any blank are never coerced to a number. A pair with a non-numeric end
    cannot have a numeric difference and is NOT a skip; those pairs are counted
    and broken down by which end is non-numeric, never silently dropped.

STATE MAPPING - the project's existing one, unchanged
        numeric 0        Current
        numeric 1        DQ30
        numeric 2        DQ60
        numeric >= 3     DQ90+     (98 and 99 included; per D12 the 99 is the
                                    documented cap, not a missing-value code)
        "RA"             RA        carried as its own category
        anything else    other     blanks and any stray code
    Both ends of a skip are numeric by definition, so skips land only in the
    four delinquency states; RA and "other" appear in the all-transitions
    denominator of section C, where they are part of the panel.

METHOD - one pass, one file at a time, nothing held across files
    Loan sequence numbers are a per-file constant 5-character prefix plus a
    7-digit tail, and diagnostic 01 reports zero cross-file loan overlap, so a
    loan lives entirely inside one file. Each file is read in chunks of columns
    [0, 1, 3] and packed into a single int64 per row:

        key = (loan_int << 27) | (period_index << 7) | (status + 2)

    period_index is year*12 + (month-1), fitting in 20 bits; status + 2 maps
    {-2 other, -1 RA, 0..99} onto 0..101, fitting in 7 bits. The file's rows are
    then sorted once on that single key, which orders them by loan and then by
    period, so adjacency in the sorted array IS adjacency in the loan's history.
    No assumption is made anywhere that the raw file is sorted or that a loan's
    rows are contiguous, and the prefixes actually observed are reported.

    Look-ahead for section D does not assume the next row is the next month: the
    row for period p+k is located by binary search on the key, and a loan's last
    observed period decides whether a missing look-ahead is "loan ends" or
    "gap".

    Sections B to F are reduced to fixed histograms inside each file, so peak
    memory is one file's key array, not the panel.

No plots. No panel-construction code. No git command is run.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "14_status_skips.txt")
RULE_W = 121

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
PASS_COLS = [COL_LOAN, COL_PERIOD, COL_DQ]
COL_NAMES = "0 loan_id, 1 reporting_period, 3 dq_status"

# key packing
ST_BITS = 7
PER_BITS = 20
ST_MASK = (1 << ST_BITS) - 1
PER_MASK = (1 << PER_BITS) - 1
ST_OFFSET = 2          # status + 2, so -2 (other) and -1 (RA) stay non-negative

RAW_RA = -1
RAW_OTHER = -2

# mapped four-state space, plus the two carried categories
ST_CUR, ST_DQ30, ST_DQ60, ST_DQ90, ST_RA, ST_OTHER = range(6)
ST_LABELS = ["Current", "DQ30", "DQ60", "DQ90+", "RA", "other"]
NSTATE = 6
NCORE = 4              # the four delinquency states skips can occupy

JUMP_LABELS = ["2", "3", "4", "5", "6-11", "12+"]
NJUMP = len(JUMP_LABELS)

STAT_LABELS = ["0", "1", "2", "3", "4", "5", "6-11", "12+"]
NSTAT = len(STAT_LABELS)

RUN_LABELS = ["1", "2-3", "4-6", "7-12", "13+"]
NRUN = len(RUN_LABELS)

FUT_LABELS = ["status 0", "status 1", "status 2", ">=3 and still rising",
              ">=3 and flat", ">=3 and falling", "RA", "other",
              "loan ends", "gap"]
NFUT = len(FUT_LABELS)
F_END = -100
F_GAP = -101

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


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def encode_period(s):
    """YYYYMM -> year*12 + (month-1). Invalid or out of packing range -> -1."""
    v = pd.to_numeric(s, errors="coerce")
    year, month = v // 100, v % 100
    idx = (year * 12 + (month - 1)).where((month >= 1) & (month <= 12))
    idx = idx.fillna(-1).astype(np.int64)
    idx[(idx < 0) | (idx > PER_MASK)] = -1
    return idx


def mlabel(i):
    return "%04d-%02d" % (i // 12, i % 12 + 1)


def encode_status(s):
    """Field 4 -> int16 raw. Numeric keeps its value; RA -> -1; else -> -2."""
    digits = s.str.isdigit().to_numpy()
    num = pd.to_numeric(s.where(s.str.isdigit()), errors="coerce")
    num = num.fillna(RAW_OTHER).to_numpy()
    code = np.full(len(s), RAW_OTHER, dtype=np.int16)
    code[digits] = num[digits]
    code[(~digits) & (s.to_numpy() == "RA")] = RAW_RA
    # statuses outside the packable range would corrupt the key; none are
    # expected (the documented cap is 99) and any would be reported as "other".
    code[(code > 99) | (code < RAW_OTHER)] = RAW_OTHER
    return code


def map4(raw):
    """Raw status -> the six-category state space."""
    st = np.full(raw.shape, ST_OTHER, dtype=np.int8)
    st[raw == RAW_RA] = ST_RA
    st[raw == 0] = ST_CUR
    st[raw == 1] = ST_DQ30
    st[raw == 2] = ST_DQ60
    st[raw >= 3] = ST_DQ90
    return st


def bucket_jump(j):
    return np.select([j == 2, j == 3, j == 4, j == 5, j <= 11],
                     [0, 1, 2, 3, 4], default=5).astype(np.int64)


def bucket_status(s):
    return np.select([s == 0, s == 1, s == 2, s == 3, s == 4, s == 5, s <= 11],
                     [0, 1, 2, 3, 4, 5, 6], default=7).astype(np.int64)


def bucket_run(r):
    return np.select([r == 1, r <= 3, r <= 6, r <= 12],
                     [0, 1, 2, 3], default=4).astype(np.int64)


def bucket_future(s, base):
    """Status at t+k -> one of FUT_LABELS. base is status(t+1), the skip landing."""
    b = np.full(s.shape, -1, dtype=np.int8)
    b[s == 0] = 0
    b[s == 1] = 1
    b[s == 2] = 2
    ge3 = s >= 3
    b[ge3 & (s > base)] = 3
    b[ge3 & (s == base)] = 4
    b[ge3 & (s < base)] = 5
    b[s == RAW_RA] = 6
    b[s == RAW_OTHER] = 7
    b[s == F_END] = 8
    b[s == F_GAP] = 9
    if (b < 0).any():
        raise RuntimeError("bucket_future left %d row(s) unclassified"
                           % int((b < 0).sum()))
    return b


def reader_for(path):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=PASS_COLS,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


# --------------------------------------------------------------------------
# accumulators
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
        self.skips = 0
        self.falls = 0
        self.jump_min = None
        self.jump_max = None
        self.fall_min = None
        self.fall_max = None
        self.jump_hist = np.zeros(NJUMP, dtype=np.int64)
        self.fall_hist = np.zeros(NJUMP, dtype=np.int64)
        self.jump_by_stat = np.zeros((NSTAT, NJUMP), dtype=np.int64)
        self.skip_cells = np.zeros((NSTATE, NSTATE), dtype=np.int64)
        self.all_cells = np.zeros((NSTATE, NSTATE), dtype=np.int64)
        # D: [from group][t+2 or t+3][bucket]
        self.future = np.zeros((2, 2, NFUT), dtype=np.int64)
        # E: [from group][run bucket], raw-status run and mapped-state run
        self.run_raw = np.zeros((2, NRUN), dtype=np.int64)
        self.run_map = np.zeros((2, NRUN), dtype=np.int64)
        self.year_hist = {}
        self.prefixes = {}

    def bump_minmax(self, arr, lo_name, hi_name):
        if not len(arr):
            return
        lo, hi = int(arr.min()), int(arr.max())
        cur_lo = getattr(self, lo_name)
        cur_hi = getattr(self, hi_name)
        setattr(self, lo_name, lo if cur_lo is None else min(cur_lo, lo))
        setattr(self, hi_name, hi if cur_hi is None else max(cur_hi, hi))


# --------------------------------------------------------------------------
# per-file work
# --------------------------------------------------------------------------
def read_keys(path, acc):
    """Every row of one file packed into one int64 array."""
    parts = []
    prefixes = {}
    for chunk in reader_for(path):
        acc.rows += len(chunk)
        loan = chunk[COL_LOAN].str.strip()
        per = encode_period(chunk[COL_PERIOD].str.strip())
        raw = encode_status(chunk[COL_DQ].str.strip())

        pre = loan.str[:5]
        for p, c in pre.value_counts().items():
            prefixes[p] = prefixes.get(p, 0) + int(c)
        tail = loan.str[5:]
        good_loan = tail.str.isdigit().to_numpy()
        acc.bad_loan += int((~good_loan).sum())

        lint = pd.to_numeric(tail.where(tail.str.isdigit()), errors="coerce")
        lint = lint.fillna(-1).astype(np.int64).to_numpy()

        okp = (per >= 0).to_numpy()
        acc.bad_period += int((~okp).sum())
        ok = okp & good_loan
        if not ok.any():
            continue

        key = ((lint[ok] << (ST_BITS + PER_BITS))
               | (per.to_numpy()[ok] << ST_BITS)
               | (raw[ok].astype(np.int64) + ST_OFFSET))
        parts.append(key)

    for p, c in prefixes.items():
        acc.prefixes[p] = acc.prefixes.get(p, 0) + c
    if not parts:
        return np.zeros(0, dtype=np.int64)
    key = np.concatenate(parts)
    del parts
    key.sort(kind="stable")
    return key


def process_file(path, acc):
    key = read_keys(path, acc)
    n = len(key)
    if n == 0:
        return

    raw = ((key & ST_MASK) - ST_OFFSET).astype(np.int16)
    per = ((key >> ST_BITS) & PER_MASK).astype(np.int32)
    loan = (key >> (ST_BITS + PER_BITS)).astype(np.int64)

    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    np.not_equal(loan[1:], loan[:-1], out=is_first[1:])
    acc.loans += int(is_first.sum())

    starts = np.nonzero(is_first)[0]
    is_last = np.empty(n, dtype=bool)
    is_last[:-1] = is_first[1:]
    is_last[-1] = True
    last_per = per[is_last]                      # max period of each loan

    same = ~is_first[1:]                         # row i and i+1 share a loan
    dp = per[1:] - per[:-1]
    cons = same & (dp == 1)
    acc.pairs_cons += int(cons.sum())
    acc.pairs_gap += int((same & (dp > 1)).sum())
    acc.pairs_dup += int((same & (dp == 0)).sum())

    idx = np.nonzero(cons)[0]                    # pair is (idx, idx+1)
    a = raw[idx]                                 # status(t)
    b = raw[idx + 1]                             # status(t+1)

    a_num = a >= 0
    b_num = b >= 0
    both = a_num & b_num
    acc.pair_both_num += int(both.sum())
    acc.pair_t_nonnum += int((~a_num & b_num).sum())
    acc.pair_t1_nonnum += int((a_num & ~b_num).sum())
    acc.pair_both_nonnum += int((~a_num & ~b_num).sum())

    # ---- all-transition denominator for section C, over every consecutive pair
    ma = map4(a)
    mb = map4(b)
    np.add.at(acc.all_cells, (ma, mb), 1)

    # ---- skips and falls
    diff = np.where(both, b.astype(np.int32) - a.astype(np.int32), 0)
    skip_m = both & (diff >= 2)
    fall_m = both & (diff <= -2)
    acc.skips += int(skip_m.sum())
    acc.falls += int(fall_m.sum())

    jumps = diff[skip_m]
    falls = (-diff[fall_m])
    acc.bump_minmax(jumps, "jump_min", "jump_max")
    acc.bump_minmax(falls, "fall_min", "fall_max")
    acc.jump_hist += np.bincount(bucket_jump(jumps), minlength=NJUMP)
    acc.fall_hist += np.bincount(bucket_jump(falls), minlength=NJUMP)

    si = idx[skip_m]                             # index of the t row of each skip
    if not len(si):
        return
    sa = raw[si].astype(np.int32)                # status(t)
    sb = raw[si + 1].astype(np.int32)            # status(t+1)

    np.add.at(acc.jump_by_stat, (bucket_status(sa), bucket_jump(sb - sa)), 1)
    np.add.at(acc.skip_cells, (map4(raw[si]), map4(raw[si + 1])), 1)

    # calendar year of status(t)
    yrs = (per[si] // 12).astype(np.int64)
    for y, c in zip(*np.unique(yrs, return_counts=True)):
        acc.year_hist[int(y)] = acc.year_hist.get(int(y), 0) + int(c)

    from_group = (sa >= 1).astype(np.int64)          # 0 = from status 0, 1 = from >=1

    # ---- D: status at t+2 and t+3, located by binary search, not by offset
    g = np.searchsorted(starts, si, side="right") - 1
    loan_last_per = last_per[g]
    base_key = (loan[si] << (ST_BITS + PER_BITS))
    for col, k in enumerate((2, 3)):
        want_per = per[si].astype(np.int64) + k
        target = base_key | (want_per << ST_BITS)
        pos = np.searchsorted(key, target, side="left")
        safe = np.minimum(pos, n - 1)
        found = (pos < n) & (key[safe] < target + (1 << ST_BITS))
        s_k = np.where(found, raw[safe].astype(np.int32), F_GAP)
        s_k = np.where(want_per > loan_last_per, F_END, s_k)
        bk = bucket_future(s_k, sb)
        np.add.at(acc.future, (from_group, col, bk), 1)

    # ---- E: run length in the state held immediately before the skip
    ar = np.arange(n, dtype=np.int64)
    for raw_mode, target_acc in ((True, acc.run_raw), (False, acc.run_map)):
        val = raw if raw_mode else map4(raw)
        newrun = np.empty(n, dtype=bool)
        newrun[0] = True
        newrun[1:] = is_first[1:] | (dp != 1) | (val[1:] != val[:-1])
        run_start = np.maximum.accumulate(np.where(newrun, ar, 0))
        run_len = ar - run_start + 1
        np.add.at(target_acc, (from_group, bucket_run(run_len[si])), 1)
        del newrun, run_start, run_len


# --------------------------------------------------------------------------
# table helpers
# --------------------------------------------------------------------------
def cs(count, den):
    """count and its share, as a fixed-width pair."""
    return "%12s%10s" % (fmt(count), pct(count, den, 2))


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))
    acc = Acc()

    rule("=")
    emit("14_status_skips - single-month jumps of two or more delinquency rungs")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns read: %s" % COL_NAMES)
    emit("scope       : strictly consecutive month pairs within one loan;")
    emit("              pairs spanning an interior reporting gap are EXCLUDED (see 13)")
    emit("skip        : numeric status(t+1) - status(t) >= 2")
    emit("fall        : numeric status(t) - status(t+1) >= 2")
    emit("non-numeric : 'RA' and blanks are a category of their own, never coerced;")
    emit("              a pair with a non-numeric end is neither a skip nor a fall")
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
        before = (acc.rows, acc.pairs_cons, acc.skips)
        process_file(path, acc)
        emit("    %-18s rows %12s   consecutive pairs %12s   skips %9s   (%5.1fs)"
             % (os.path.basename(path), fmt(acc.rows - before[0]),
                fmt(acc.pairs_cons - before[1]), fmt(acc.skips - before[2]),
                time.time() - t0))
    emit("    pooled rows %s   loans %s   consecutive pairs %s   skips %s"
         % (fmt(acc.rows), fmt(acc.loans), fmt(acc.pairs_cons), fmt(acc.skips)))
    emit("    loan-id prefixes seen: %s"
         % ", ".join("%s (%s rows)" % (p, fmt(c))
                     for p, c in sorted(acc.prefixes.items())))
    emit("    rows rejected - unparseable period %s, unparseable loan id %s"
         % (fmt(acc.bad_period), fmt(acc.bad_loan)))
    emit()

    P = acc.pairs_cons

    # ---------------- A ----------------
    rule("=")
    emit("A. HEADLINE")
    rule("=")
    emit("Unit: consecutive month pairs. One pair per (loan, month) that has an")
    emit("observed row in the immediately following month.")
    emit()
    emit("%-58s %18s %12s" % ("quantity", "count", "share"))
    rule()
    emit("%-58s %18s %12s" % ("consecutive pairs (denominator)", fmt(P), "100.0000%"))
    emit("%-58s %18s %12s" % ("skips: status rises by >= 2 in one month",
                              fmt(acc.skips), pct(acc.skips, P)))
    emit("%-58s %18s %12s" % ("falls: status drops by >= 2 in one month",
                              fmt(acc.falls), pct(acc.falls, P)))
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
    emit("RECONCILIATION WITH DIAGNOSTIC 13   (unit: pairs, except where stated)")
    rule()
    emit("%-58s %18s" % ("gap-spanning pairs EXCLUDED from every table above",
                         fmt(acc.pairs_gap)))
    emit("%-58s %18s" % ("duplicate (loan, period) pairs", fmt(acc.pairs_dup)))
    emit("%-58s %18s" % ("rows in panel (unit: loan-months)", fmt(acc.rows)))
    emit("%-58s %18s" % ("loans in panel (unit: loans)", fmt(acc.loans)))
    identity = acc.rows - acc.loans - acc.pairs_gap - acc.pairs_dup
    emit("%-58s %18s" % ("rows - loans - gap pairs - duplicate pairs",
                         fmt(identity)))
    emit("%-58s %18s" % ("consecutive pairs counted above", fmt(P)))
    emit("%-58s %18s" % ("identity holds", "yes" if identity == P else "NO"))
    emit()
    emit("Every loan contributes one fewer pair than it has rows, minus one pair for")
    emit("each interior hole, so the identity above is the arithmetic check that no")
    emit("pair was created or lost. The gap-pair count is diagnostic 13's gap-event")
    emit("count measured independently here.")
    emit()

    if acc.skips == 0:
        rule("=")
        emit("NO SKIPS OBSERVED - sections B to F are empty.")
        rule("=")
        write_out()
        return

    S = acc.skips

    # ---------------- B ----------------
    rule("=")
    emit("B. SKIP MAGNITUDE   (unit: skip events)")
    rule("=")
    emit("Jump size is status(t+1) - status(t) in raw field-4 rungs.")
    emit()
    emit("B.1  JUMP SIZE DISTRIBUTION")
    rule()
    emit("%-22s%16s%12s%20s%12s" % ("jump size (rungs)", "skips", "share",
                                    "falls of same size", "share"))
    rule()
    for i, lab in enumerate(JUMP_LABELS):
        emit("%-22s%16s%12s%20s%12s"
             % (lab, fmt(acc.jump_hist[i]), pct(acc.jump_hist[i], S),
                fmt(acc.fall_hist[i]), pct(acc.fall_hist[i], acc.falls)))
    rule()
    emit("%-22s%16s%12s%20s%12s" % ("TOTAL", fmt(S), "100.0000%",
                                    fmt(acc.falls), "100.0000%"))
    emit()
    emit("%-38s %s" % ("min jump size (rungs)", fmt(acc.jump_min)))
    emit("%-38s %s" % ("max jump size (rungs)", fmt(acc.jump_max)))
    emit("%-38s %s" % ("min fall size (rungs)",
                       fmt(acc.fall_min) if acc.fall_min is not None else "n/a"))
    emit("%-38s %s" % ("max fall size (rungs)",
                       fmt(acc.fall_max) if acc.fall_max is not None else "n/a"))
    emit()
    emit("B.2  JUMP SIZE BY status(t)   (counts; unit: skip events)")
    rule()
    emit("%-22s%13s%s" % ("status(t)", "skips",
                          "".join("%13s" % l for l in JUMP_LABELS)))
    rule()
    for i, lab in enumerate(STAT_LABELS):
        row = acc.jump_by_stat[i]
        emit("%-22s%13s%s" % (lab, fmt(row.sum()),
                              "".join("%13s" % fmt(c) for c in row)))
    rule()
    tot = acc.jump_by_stat.sum(axis=0)
    emit("%-22s%13s%s" % ("TOTAL", fmt(acc.jump_by_stat.sum()),
                          "".join("%13s" % fmt(c) for c in tot)))
    emit()
    emit("B.3  SHARE OF ALL SKIPS in each cell of B.2 - counts are in B.2 above")
    rule()
    emit("%-22s%13s%s" % ("status(t)", "share",
                          "".join("%13s" % l for l in JUMP_LABELS)))
    rule()
    for i, lab in enumerate(STAT_LABELS):
        row = acc.jump_by_stat[i]
        emit("%-22s%13s%s" % (lab, pct(row.sum(), S, 2),
                              "".join("%13s" % pct(c, S, 2) for c in row)))
    rule()
    emit("%-22s%13s%s" % ("TOTAL", "100.00%",
                          "".join("%13s" % pct(c, S, 2) for c in tot)))
    emit()

    # ---------------- C ----------------
    rule("=")
    emit("C. STATE-SPACE IMPACT")
    rule("=")
    emit("Both ends mapped to the four-state space. Skips have numeric ends by")
    emit("definition, so they occupy only the 4x4 block; RA and 'other' appear in the")
    emit("all-transitions table because they are part of the panel.")
    emit()
    emit("C.1  TRANSITION CELLS POPULATED BY SKIPS   (unit: skip events)")
    rule()
    emit("%-22s%14s%s" % ("from \\ to", "skips",
                          "".join("%14s" % l for l in ST_LABELS[:NCORE])))
    rule()
    for i in range(NCORE):
        row = acc.skip_cells[i][:NCORE]
        emit("%-22s%14s%s" % (ST_LABELS[i], fmt(row.sum()),
                              "".join("%14s" % fmt(c) for c in row)))
    rule()
    emit("%-22s%14s%s" % ("TOTAL into", fmt(acc.skip_cells[:NCORE, :NCORE].sum()),
                          "".join("%14s" % fmt(c)
                                  for c in acc.skip_cells[:NCORE, :NCORE].sum(axis=0))))
    emit()
    emit("C.2  ALL CONSECUTIVE-PAIR TRANSITIONS   (unit: consecutive pairs)")
    rule()
    emit("%-22s%16s%s" % ("from \\ to", "pairs",
                          "".join("%16s" % l for l in ST_LABELS)))
    rule()
    for i in range(NSTATE):
        row = acc.all_cells[i]
        emit("%-22s%16s%s" % (ST_LABELS[i], fmt(row.sum()),
                              "".join("%16s" % fmt(c) for c in row)))
    rule()
    emit("%-22s%16s%s" % ("TOTAL into", fmt(acc.all_cells.sum()),
                          "".join("%16s" % fmt(c)
                                  for c in acc.all_cells.sum(axis=0))))
    emit()
    emit("C.3  SKIP SHARE OF EACH POPULATED CELL")
    rule()
    emit("For every cell that skips populate: how many of ALL the consecutive-pair")
    emit("transitions landing in that cell arrived there as a skip.")
    emit()
    emit("%-34s%16s%18s%14s" % ("transition cell", "from skips",
                                "all transitions", "skip share"))
    rule()
    for i in range(NCORE):
        for j in range(NCORE):
            s_ij = int(acc.skip_cells[i][j])
            if s_ij == 0:
                continue
            a_ij = int(acc.all_cells[i][j])
            emit("%-34s%16s%18s%14s"
                 % ("%s -> %s" % (ST_LABELS[i], ST_LABELS[j]),
                    fmt(s_ij), fmt(a_ij), pct(s_ij, a_ij, 4)))
    rule()
    emit("%-34s%16s%18s%14s" % ("ALL POPULATED CELLS", fmt(S),
                                fmt(acc.all_cells.sum()),
                                pct(S, acc.all_cells.sum(), 4)))
    emit()
    empty = [("%s -> %s" % (ST_LABELS[i], ST_LABELS[j]))
             for i in range(NCORE) for j in range(NCORE)
             if acc.skip_cells[i][j] == 0]
    emit("cells in the 4x4 block that no skip populates (%d of 16): %s"
         % (len(empty), ", ".join(empty) if empty else "none"))
    emit()

    # ---------------- D ----------------
    rule("=")
    emit("D. CORRECTION SIGNATURE - what the loan reports after the skip")
    rule("=")
    emit("Unit: skip events. Each skip is followed forward to the months t+2 and t+3,")
    emit("located by the loan's reporting period, not by row offset.")
    emit("  'loan ends'  the loan has no observed row at any period >= that month")
    emit("  'gap'        the loan is observed later but not in that month")
    emit("  rising / flat / falling are measured against status(t+1), the month the")
    emit("  skip landed in, for both the t+2 and the t+3 column.")
    emit()
    n0 = int(acc.future[0][0].sum())
    n1 = int(acc.future[1][0].sum())
    emit("from groups: status(t) == 0: %s skips   |   status(t) >= 1: %s skips"
         % (fmt(n0), fmt(n1)))
    emit()
    emit("%-26s%22s%22s%22s%22s"
         % ("", "from 0: t+2", "from 0: t+3",
            "from >=1: t+2", "from >=1: t+3"))
    emit("%-26s%22s%22s%22s%22s"
         % ("status at that month", "count     share", "count     share",
            "count     share", "count     share"))
    rule()
    for k, lab in enumerate(FUT_LABELS):
        emit("%-26s%s%s%s%s"
             % (lab,
                cs(acc.future[0][0][k], n0), cs(acc.future[0][1][k], n0),
                cs(acc.future[1][0][k], n1), cs(acc.future[1][1][k], n1)))
    rule()
    emit("%-26s%s%s%s%s"
         % ("TOTAL", cs(n0, n0), cs(int(acc.future[0][1].sum()), n0),
            cs(n1, n1), cs(int(acc.future[1][1].sum()), n1)))
    emit()

    # ---------------- E ----------------
    rule("=")
    emit("E. PRE-SKIP RUN LENGTH   (unit: skip events)")
    rule("=")
    emit("How many consecutive observed months the loan had already held the state it")
    emit("was in at t, counting t itself as month 1. A reporting hole or a change of")
    emit("state ends a run, so every run is unbroken and strictly monthly.")
    emit()
    emit("Two readings of 'its state' are reported because they differ for the deep")
    emit("states: field 4 increments every month inside an unbroken DQ90+ spell (see")
    emit("diagnostic 10), so a raw-status run there is short by construction while the")
    emit("mapped-state run measures the spell. They coincide for status 0, 1 and 2.")
    emit()
    for title, table in (("E.1  RUN LENGTH ON RAW FIELD-4 STATUS", acc.run_raw),
                         ("E.2  RUN LENGTH ON THE MAPPED STATE", acc.run_map)):
        tot_all = int(table.sum())
        emit(title)
        rule()
        emit("%-26s%22s%22s%22s"
             % ("", "all skips", "from 0", "from >=1"))
        emit("%-26s%22s%22s%22s"
             % ("run length (months)", "count     share", "count     share",
                "count     share"))
        rule()
        for i, lab in enumerate(RUN_LABELS):
            emit("%-26s%s%s%s"
                 % (lab, cs(table[0][i] + table[1][i], tot_all),
                    cs(table[0][i], n0), cs(table[1][i], n1)))
        rule()
        emit("%-26s%s%s%s"
             % ("TOTAL", cs(tot_all, tot_all), cs(n0, n0), cs(n1, n1)))
        emit()

    # ---------------- F ----------------
    rule("=")
    emit("F. CALENDAR CONCENTRATION   (unit: skip events)")
    rule("=")
    emit("Filed under the YEAR of status(t) - the month before the jump.")
    emit()
    emit("%-22s%16s%12s%16s" % ("year of status(t)", "skips", "share",
                                "cumulative"))
    rule()
    run = 0
    for y in sorted(acc.year_hist):
        c = acc.year_hist[y]
        run += c
        emit("%-22s%16s%12s%16s" % (y, fmt(c), pct(c, S), pct(run, S)))
    rule()
    emit("%-22s%16s%12s%16s" % ("TOTAL", fmt(S), "100.0000%", "100.0000%"))
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t_start, (time.time() - t_start) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
