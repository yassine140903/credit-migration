"""Is the delinquency status on a censoring row a real reading, and three panel checks.

PART A asks one question about a small population: when a loan's last observed
row carries zero-balance code 15, 16 or 96, is the delinquency status on that
row a fresh observation of the loan, or the previous month's value carried
forward by a servicer who had already stopped looking? src/panel.py emits a
transition INTO that row, labelled with the status the row carries, so if the
status is stale the panel contains a transition that never happened.

    population      loans whose LAST observed row carries 15, 16 or 96, each
                    code reported separately and never pooled
    comparison      loans whose last observed row carries NO code at all
                    ('no_further_observation'): their last row is uncontested,
                    so they show what an ordinary final month looks like
    excluded        loans ending on a terminal code 01/02/03/09, which are
                    terminations rather than censoring

    'the row before'    throughout sections A.2, A.3 and A.4 this means the
                        immediately preceding row AND only when it is strictly
                        the previous calendar month, which is the pair
                        src/panel.py would actually emit. Where the loan has a
                        single row, or a reporting hole sits immediately before
                        the code row, there is no such row; those loans are
                        counted as 'previous row missing' in A.1 and excluded
                        from A.2's second table, A.3 and A.4, with the excluded
                        count stated in each.

STATE MAPPING for PART A
    0 Current, 1 DQ30, 2 DQ60, >= 3 DQ90+, 'RA' as its own column, any other
    non-numeric as 'other'. RA is deliberately NOT folded into Default here:
    the question is what the file says, and folding would hide it. Note that
    src/panel.py maps RA to Default and suppresses those rows as absorbing
    from-states; A.3 states how many rows that removes so the two views reconcile.

PART B checks three properties the panel build relies on, independently of
src/panel.py - nothing is imported from it and the panel is only read.

    B.1  every loan_id appears in exactly one quarter's parquet, which is what
         makes a per-quarter run_length correct
    B.2  loans with more than one row that contribute no panel row because the
         gap rule dropped every candidate pair
    B.3  the offset between the first payment month and the first observed
         reporting period, over a fixed-seed sample of 200 loans; src/panel.py
         takes the origination month to be the first payment month minus one

Column indices are resolved at run time from the published layout by attribute
name, the same way src/panel.py resolves them, and are printed in the header.

No plots. Nothing is written except the report. No git command is run.
"""

import glob
import os
import re
import time
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

PERF_GLOB = "data/historical_data_200[67]/*/perf_*.txt"
ORIG_GLOB = "data/historical_data_200[67]/*/orig_*.txt"
PANEL_GLOB = os.path.join("data", "panel", "panel_*.parquet")
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")
OUT = os.path.join("diagnostics", "17_censor_semantics.txt")

CHUNKSIZE = 2_000_000
RULE_W = 121
SAMPLE_N = 200
SAMPLE_SEED = 17

PERF_SHEET = "monthly performance data file"
ORIG_SHEET = "origination data file"
PERF_WANTED = [("loan", "loan identifier"), ("period", "period"),
               ("status", "current loan delinquency status"),
               ("zero_balance", "zero balance code")]
ORIG_WANTED = [("loan", "loan identifier"),
               ("first_payment", "first payment date")]
PERF_EXPECTED = {"loan": 0, "period": 1, "status": 3}

ZB_TERMINAL = ("01", "02", "03", "09")
ZB_CENSOR = ("15", "16", "96")

# groups, in column order: the three codes, then the uncontested comparison
GROUPS = ["code 15", "code 16", "code 96", "no code"]
NGROUP = len(GROUPS)
G_NONE = 3

ST_LABELS = ["Current", "DQ30", "DQ60", "DQ90+", "RA", "other"]
NSTATE = len(ST_LABELS)
S_RA, S_OTHER = 4, 5

CMP_LABELS = ["identical", "changed by exactly 1 rung", "changed by 2+ rungs",
              "changed, non-numeric end", "previous row missing"]
NCMP = len(CMP_LABELS)
C_SAME, C_ONE, C_TWO, C_NONNUM, C_MISSING = range(NCMP)

RUN_LABELS = ["1", "2-3", "4-6", "7-12", "13+"]
NRUN = len(RUN_LABELS)

RAW_RA, RAW_OTHER = -1, -2

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


def cs(count, den, cw=12, sw=9, dp=1):
    return ("%" + str(cw) + "s%" + str(sw) + "s") % (fmt(count),
                                                     pct(count, den, dp))


def to_yyyymm(idx):
    return (idx // 12) * 100 + (idx % 12) + 1


# ==========================================================================
# layout
# ==========================================================================
def _sheets(path):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

    def cell(c):
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(ns + "t"))
        v = c.find(ns + "v")
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    out = {}
    for name in sorted(n for n in z.namelist()
                       if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
        rows = [[cell(c) for c in row.iter(ns + "c")]
                for row in ET.fromstring(z.read(name)).iter(ns + "row")]
        head = " ".join(" ".join(r) for r in rows[:6]).lower()
        fields = [(int(cs_[0]), cs_[1]) for cs_ in
                  ([x.strip() for x in r if x.strip()] for r in rows)
                  if len(cs_) >= 2 and re.match(r"^\d+$", cs_[0])]
        for marker in (PERF_SHEET, ORIG_SHEET):
            if marker in head and fields:
                out[marker] = (fields, name)
    return out


def _norm(t):
    t = t.strip().lower().replace("®", "").replace("™", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()/%.-]+", " ", t)).strip()


def resolve():
    sheets = _sheets(LAYOUT)

    def pick(marker, wanted):
        fields, sheet = sheets[marker]
        got = {}
        for label, want in wanted:
            w = _norm(want)
            hits = [f for f in fields if _norm(f[1]) == w] or \
                   [f for f in fields if w in _norm(f[1])]
            if len(hits) != 1:
                raise SystemExit("layout lookup for %r on %s matched %d row(s)"
                                 % (want, marker, len(hits)))
            got[label] = (hits[0][0] - 1, hits[0][0], hits[0][1])
        return got, sheet

    perf, perf_sheet = pick(PERF_SHEET, PERF_WANTED)
    orig, orig_sheet = pick(ORIG_SHEET, ORIG_WANTED)
    for label, expect in PERF_EXPECTED.items():
        if perf[label][0] != expect:
            raise SystemExit("layout gives %d for performance %r, project uses %d"
                             % (perf[label][0], label, expect))
    return perf, orig, perf_sheet, orig_sheet


# ==========================================================================
# encoding
# ==========================================================================
def encode_status(s):
    digits = s.str.isdigit().to_numpy()
    num = pd.to_numeric(s.where(s.str.isdigit()), errors="coerce")
    code = num.fillna(RAW_OTHER).astype(np.int16).to_numpy()
    code[~digits] = RAW_OTHER
    code[(~digits) & (s.to_numpy() == "RA")] = RAW_RA
    return code


def map_state(raw):
    st = np.full(raw.shape, S_OTHER, dtype=np.int8)
    st[raw == RAW_RA] = S_RA
    st[raw == 0] = 0
    st[raw == 1] = 1
    st[raw == 2] = 2
    st[raw >= 3] = 3
    return st


def bucket_run(r):
    return np.select([r == 1, r <= 3, r <= 6, r <= 12],
                     [0, 1, 2, 3], default=4).astype(np.int64)


# ==========================================================================
class Acc(object):
    def __init__(self):
        self.perf_rows = 0
        self.loans = 0
        self.terminal_loans = 0
        self.group_loans = np.zeros(NGROUP, dtype=np.int64)
        self.cmp_tab = np.zeros((NGROUP, NCMP), dtype=np.int64)
        self.state_code = np.zeros((NGROUP, NSTATE), dtype=np.int64)
        self.state_prev = np.zeros((NGROUP, NSTATE), dtype=np.int64)
        self.xtab = np.zeros((NGROUP, NSTATE, NSTATE), dtype=np.int64)
        self.run_tab = np.zeros((NGROUP, NRUN), dtype=np.int64)
        self.prev_missing = np.zeros(NGROUP, dtype=np.int64)
        # B.2
        self.multi_row_loans = 0
        self.gap_only_loans = 0
        self.absorbing_only_loans = 0
        self.gap_only_examples = []
        # B.3
        self.sample_first_period = {}
        self.indices = None


def process_file(path, pidx, acc, sample_by_prefix):
    c_loan, c_per = pidx["loan"][0], pidx["period"][0]
    c_st, c_zb = pidx["status"][0], pidx["zero_balance"][0]
    cols = sorted({c_loan, c_per, c_st, c_zb})
    tails, pers, sts, zbs = [], [], [], []
    prefixes = {}
    for chunk in pd.read_csv(path, sep="|", header=None, dtype=str, usecols=cols,
                             chunksize=CHUNKSIZE, engine="c", na_filter=False):
        acc.perf_rows += len(chunk)
        loan = chunk[c_loan].str.strip()
        for p, c in loan.str[:5].value_counts().items():
            prefixes[p] = prefixes.get(p, 0) + int(c)
        tails.append(pd.to_numeric(loan.str[5:], errors="coerce")
                     .fillna(-1).astype(np.int64).to_numpy())
        per = pd.to_numeric(chunk[c_per].str.strip(), errors="coerce")
        y, m = per // 100, per % 100
        pers.append((y * 12 + (m - 1)).where((m >= 1) & (m <= 12))
                    .fillna(-1).astype(np.int64).to_numpy())
        sts.append(encode_status(chunk[c_st].str.strip()))
        zbs.append(chunk[c_zb].str.strip().to_numpy().astype("U2"))

    tail = np.concatenate(tails)
    period = np.concatenate(pers)
    raw = np.concatenate(sts)
    zb = np.concatenate(zbs)
    del tails, pers, sts, zbs
    keep = (period >= 0) & (tail >= 0)
    tail, period, raw, zb = tail[keep], period[keep], raw[keep], zb[keep]

    order = np.lexsort((period, tail))
    tail, period, raw, zb = tail[order], period[order], raw[order], zb[order]
    n = len(tail)

    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    np.not_equal(tail[1:], tail[:-1], out=is_first[1:])
    is_last = np.empty(n, dtype=bool)
    is_last[:-1] = is_first[1:]
    is_last[-1] = True
    acc.loans += int(is_first.sum())

    dp = np.zeros(n, dtype=np.int64)
    dp[1:] = period[1:] - period[:-1]
    same = np.zeros(n, dtype=bool)
    same[1:] = ~is_first[1:]
    consecutive = same & (dp == 1)

    state = map_state(raw)

    # run length on the mapped state over the full history, reset by a state
    # change or a hole - the same definition src/panel.py uses for run_length
    ar = np.arange(n, dtype=np.int64)
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    new_run[1:] = is_first[1:] | (dp[1:] != 1) | (state[1:] != state[:-1])
    run_len = ar - np.maximum.accumulate(np.where(new_run, ar, 0)) + 1

    # ---------------- PART A ----------------
    last = np.nonzero(is_last)[0]
    last_zb = zb[last]
    group = np.full(len(last), -1, dtype=np.int8)
    for i, code in enumerate(ZB_CENSOR):
        group[last_zb == code] = i
    group[last_zb == ""] = G_NONE
    acc.terminal_loans += int(np.isin(last_zb, ZB_TERMINAL).sum())

    sel = group >= 0
    rows = last[sel]
    grp = group[sel].astype(np.int64)
    acc.group_loans += np.bincount(grp, minlength=NGROUP)

    # the row before: only when it is strictly the previous calendar month
    has_prev = consecutive[rows]
    prev = rows - 1

    # A.1 - raw status on the code row vs the row before it
    cmp_code = np.full(len(rows), C_MISSING, dtype=np.int64)
    both = has_prev
    a = raw[np.where(both, prev, rows)]
    b = raw[rows]
    numeric = both & (a >= 0) & (b >= 0)
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    cmp_code[both & (a == b)] = C_SAME
    cmp_code[numeric & (a != b) & (diff == 1)] = C_ONE
    cmp_code[numeric & (a != b) & (diff >= 2)] = C_TWO
    cmp_code[both & (a != b) & ~numeric] = C_NONNUM
    np.add.at(acc.cmp_tab, (grp, cmp_code), 1)

    # A.2 - mapped state on the code row (all loans in the group) and on the
    # row before (only where that row exists)
    np.add.at(acc.state_code, (grp, state[rows]), 1)
    np.add.at(acc.state_prev, (grp[both], state[prev[both]]), 1)
    acc.prev_missing += np.bincount(grp[~both], minlength=NGROUP)

    # A.3 - the pair src/panel.py would emit: from = row before, to = code row
    np.add.at(acc.xtab, (grp[both], state[prev[both]], state[rows[both]]), 1)

    # A.4 - run length as of the row before
    np.add.at(acc.run_tab, (grp[both], bucket_run(run_len[prev[both]])), 1)

    # ---------------- B.2 ----------------
    # loans with more than one row and no strictly consecutive pair at all:
    # every candidate pair they have spans a hole
    starts = np.nonzero(is_first)[0]
    rows_per_loan = np.diff(np.append(starts, n))
    cons_per_loan = np.add.reduceat(consecutive.astype(np.int64), starts)
    multi = rows_per_loan > 1
    acc.multi_row_loans += int(multi.sum())
    gap_only = multi & (cons_per_loan == 0)
    acc.gap_only_loans += int(gap_only.sum())
    if gap_only.any() and len(acc.gap_only_examples) < 10:
        prefix = next(iter(prefixes))
        for s in starts[gap_only][:10 - len(acc.gap_only_examples)]:
            k = int(np.searchsorted(starts, s))
            span = slice(s, s + int(rows_per_loan[k]))
            acc.gap_only_examples.append(
                (prefix + "%07d" % tail[s],
                 ", ".join(str(to_yyyymm(p)) for p in period[span])))

    # loans that have consecutive pairs but every one of them starts on an
    # absorbing (RA) row, so R1 would suppress them all
    live_pair = consecutive & (raw[np.maximum(ar - 1, 0)] >= 0)
    live_per_loan = np.add.reduceat(live_pair.astype(np.int64), starts)
    acc.absorbing_only_loans += int((multi & (cons_per_loan > 0)
                                     & (live_per_loan == 0)).sum())

    # ---------------- B.3 ----------------
    prefix = next(iter(prefixes))
    wanted = sample_by_prefix.get(prefix)
    if wanted is not None and len(wanted):
        uniq = tail[is_first]
        first_period = period[is_first]
        pos = np.searchsorted(uniq, wanted)
        safe = np.clip(pos, 0, len(uniq) - 1)
        hit = uniq[safe] == wanted
        for t, p in zip(wanted[hit], first_period[safe][hit]):
            acc.sample_first_period[prefix + "%07d" % t] = int(p)


# ==========================================================================
def part_b1():
    """Every loan_id in exactly one quarter's parquet."""
    paths = sorted(glob.glob(PANEL_GLOB))
    seen = {}
    per_file = []
    for p in paths:
        ids = pd.read_parquet(p, columns=["loan_id"])["loan_id"]
        uniq = pd.unique(ids)
        per_file.append((os.path.basename(p), len(ids), len(uniq)))
        for lid in uniq:
            seen[lid] = seen.get(lid, 0) + 1
        del ids, uniq
    multi = {k: v for k, v in seen.items() if v > 1}
    return paths, per_file, len(seen), multi


def load_sample(oidx):
    """Fixed-seed sample of loan ids, with each one's first payment month."""
    frames = []
    for path in sorted(glob.glob(ORIG_GLOB)):
        cols = sorted({oidx["loan"][0], oidx["first_payment"][0]})
        f = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=cols,
                        engine="c", na_filter=False)
        loan = f[oidx["loan"][0]].str.strip()
        fpd = pd.to_numeric(f[oidx["first_payment"][0]].str.strip(),
                            errors="coerce")
        y, m = fpd // 100, fpd % 100
        idx = (y * 12 + (m - 1)).where((m >= 1) & (m <= 12))
        frames.append(pd.DataFrame({"loan_id": loan.to_numpy(),
                                    "fp": idx.fillna(-1).astype(np.int64)
                                    .to_numpy()}))
    allo = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(SAMPLE_SEED)
    take = rng.choice(len(allo), size=SAMPLE_N, replace=False)
    take.sort()
    sample = allo.iloc[take].reset_index(drop=True)
    by_prefix = {}
    for lid in sample["loan_id"]:
        by_prefix.setdefault(lid[:5], []).append(int(lid[5:]))
    return sample, {k: np.sort(np.array(v, dtype=np.int64))
                    for k, v in by_prefix.items()}, len(allo)


# ==========================================================================
def main():
    t0 = time.time()
    pidx, oidx, perf_sheet, orig_sheet = resolve()
    perf_paths = sorted(glob.glob(PERF_GLOB))

    rule("=")
    emit("17_censor_semantics - is the status on a censoring row a real reading,")
    emit("                      plus three checks on the built panel")
    rule("=")
    emit("performance : %s" % PERF_GLOB)
    emit("origination : %s" % ORIG_GLOB)
    emit("panel       : %s" % PANEL_GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit()
    emit("RESOLVED COLUMN INDICES   (0-based index = published position - 1)")
    rule()
    emit("%-16s%-16s%10s%8s   %s" % ("file", "role", "position", "index",
                                     "attribute name"))
    rule()
    for label, _ in PERF_WANTED:
        i, pos, attribute = pidx[label]
        emit("%-16s%-16s%10d%8d   %s" % ("performance", label, pos, i, attribute))
    for label, _ in ORIG_WANTED:
        i, pos, attribute = oidx[label]
        emit("%-16s%-16s%10d%8d   %s" % ("origination", label, pos, i, attribute))
    rule()
    emit("sheets: %s / %s" % (perf_sheet, orig_sheet))
    emit("indices 0, 1 and 3 asserted against the values the project already uses.")
    emit()
    emit("population  : loans whose LAST observed row carries 15, 16 or 96")
    emit("comparison  : loans whose last observed row carries NO code")
    emit("'row before': the immediately preceding row, and only when it is")
    emit("              strictly the previous calendar month")
    emit("states      : 0 Current, 1 DQ30, 2 DQ60, >=3 DQ90+, RA as its own")
    emit("              column (NOT folded into Default, unlike src/panel.py)")
    emit()

    sample, sample_by_prefix, n_orig = load_sample(oidx)
    acc = Acc()
    emit("PASS 1/1 - performance files")
    rule()
    for path in perf_paths:
        t = time.time()
        before = acc.loans
        process_file(path, pidx, acc, sample_by_prefix)
        emit("    %-18s loans %9s   (%5.1fs)"
             % (os.path.basename(path), fmt(acc.loans - before), time.time() - t))
    emit("    pooled rows %s   loans %s" % (fmt(acc.perf_rows), fmt(acc.loans)))
    emit()

    total_pop = int(acc.group_loans.sum())

    rule("=")
    emit("PART A - CENSORING-ROW SEMANTICS")
    rule("=")
    emit("%-28s%16s%15s" % ("group", "loans", "of all loans"))
    rule()
    for i, g in enumerate(GROUPS):
        emit("%-28s%16s%15s" % (g, fmt(acc.group_loans[i]),
                                pct(acc.group_loans[i], acc.loans)))
    rule()
    emit("%-28s%16s%15s" % ("TOTAL in scope", fmt(total_pop),
                            pct(total_pop, acc.loans)))
    emit("%-28s%16s%15s" % ("excluded: terminal 01/02/03/09",
                            fmt(acc.terminal_loans),
                            pct(acc.terminal_loans, acc.loans)))
    emit("%-28s%16s" % ("loans", fmt(acc.loans)))
    emit()

    # ---------------- A.1 ----------------
    rule("=")
    emit("A.1  STATUS ON THE CODE ROW vs THE ROW IMMEDIATELY BEFORE IT")
    rule("=")
    emit("Unit: loans. Comparison is on the RAW field-4 status, so 'identical'")
    emit("includes RA followed by RA. A group that is almost entirely 'identical'")
    emit("is carrying the previous month's value forward.")
    emit()
    emit("%-30s%s" % ("", "".join("%21s" % g for g in GROUPS)))
    emit("%-30s%s" % ("comparison", "".join("%21s" % "count    share"
                                            for _ in GROUPS)))
    rule()
    for k, lab in enumerate(CMP_LABELS):
        emit("%-30s%s" % (lab, "".join(cs(acc.cmp_tab[g][k], acc.group_loans[g])
                                       for g in range(NGROUP))))
    rule()
    emit("%-30s%s" % ("TOTAL", "".join(cs(acc.group_loans[g], acc.group_loans[g])
                                       for g in range(NGROUP))))
    emit()

    # ---------------- A.2 ----------------
    rule("=")
    emit("A.2  MAPPED STATE ON THE CODE ROW, AND ON THE ROW BEFORE")
    rule("=")
    emit("Unit: loans. The second table covers only loans that have a strictly")
    emit("preceding month; the excluded count is given under it.")
    emit()
    for title, table, note in (
            ("A.2a  STATE ON THE CODE ROW (or last row, for 'no code')",
             acc.state_code, None),
            ("A.2b  STATE ON THE ROW BEFORE IT", acc.state_prev, True)):
        emit(title)
        rule()
        emit("%-16s%14s%s" % ("group", "loans",
                              "".join("%13s" % s for s in ST_LABELS)))
        rule()
        for g in range(NGROUP):
            row = table[g]
            emit("%-16s%14s%s" % (GROUPS[g], fmt(row.sum()),
                                  "".join("%13s" % fmt(c) for c in row)))
        rule()
        for g in range(NGROUP):
            row = table[g]
            tot = int(row.sum())
            emit("%-16s%14s%s" % (GROUPS[g] + " %", "100.00%" if tot else "n/a",
                                  "".join("%13s" % pct(c, tot, 2) for c in row)))
        if note:
            rule()
            for g in range(NGROUP):
                emit("    %-24s previous row missing: %s of %s (%s)"
                     % (GROUPS[g], fmt(acc.prev_missing[g]),
                        fmt(acc.group_loans[g]),
                        pct(acc.prev_missing[g], acc.group_loans[g])))
        emit()

    # ---------------- A.3 ----------------
    rule("=")
    emit("A.3  THE TRANSITION src/panel.py EMITS FOR THESE LOANS")
    rule("=")
    emit("Unit: loans with a strictly preceding month. The from-state is the mapped state")
    emit("on the row before, the to-state is the mapped state on the code row. This is")
    emit("exactly the last pair the panel carries for each of these loans.")
    emit()
    emit("src/panel.py maps RA to Default and suppresses any pair whose from-state is")
    emit("absorbing, so every cell in the RA from-state row below is NOT emitted; the")
    emit("count is stated under each table.")
    emit()
    for g in range(NGROUP):
        tab = acc.xtab[g]
        tot = int(tab.sum())
        emit("A.3.%d  %s   (loans with a preceding month: %s)"
             % (g + 1, GROUPS[g], fmt(tot)))
        rule()
        emit("%-16s%14s%s" % ("from \\ to", "loans",
                              "".join("%13s" % s for s in ST_LABELS)))
        rule()
        for i in range(NSTATE):
            emit("%-16s%14s%s" % (ST_LABELS[i], fmt(tab[i].sum()),
                                  "".join("%13s" % fmt(c) for c in tab[i])))
        rule()
        emit("%-16s%14s%s" % ("TOTAL into", fmt(tot),
                              "".join("%13s" % fmt(c) for c in tab.sum(axis=0))))
        diag = int(np.trace(tab))
        emit("    unchanged across the pair (diagonal): %s of %s (%s)"
             % (fmt(diag), fmt(tot), pct(diag, tot)))
        emit("    FLAGGED  DQ90+ -> Current : %s of %s (%s)"
             % (fmt(tab[3][0]), fmt(tot), pct(tab[3][0], tot)))
        emit("    FLAGGED  DQ90+ -> DQ30    : %s of %s (%s)"
             % (fmt(tab[3][1]), fmt(tot), pct(tab[3][1], tot)))
        emit("    suppressed by R1 (RA from-state, not in the panel): %s of %s (%s)"
             % (fmt(tab[S_RA].sum()), fmt(tot), pct(int(tab[S_RA].sum()), tot)))
        emit()

    # ---------------- A.4 ----------------
    rule("=")
    emit("A.4  RUN LENGTH ON THE MAPPED STATE AS OF THE ROW BEFORE THE CODE ROW")
    rule("=")
    emit("Unit: loans with a strictly preceding month. Consecutive months the loan")
    emit("had held that mapped state, counting that row as 1; a state change or a")
    emit("hole ends a run.")
    emit()
    emit("%-30s%s" % ("", "".join("%21s" % g for g in GROUPS)))
    emit("%-30s%s" % ("run length (months)",
                      "".join("%21s" % "count    share" for _ in GROUPS)))
    rule()
    totals = acc.run_tab.sum(axis=1)
    for k, lab in enumerate(RUN_LABELS):
        emit("%-30s%s" % (lab, "".join(cs(acc.run_tab[g][k], totals[g])
                                       for g in range(NGROUP))))
    rule()
    emit("%-30s%s" % ("TOTAL", "".join(cs(totals[g], totals[g])
                                       for g in range(NGROUP))))
    emit()

    # ---------------- PART B ----------------
    rule("=")
    emit("PART B - CHECKS ON THE BUILT PANEL")
    rule("=")
    emit()
    emit("B.1  LOAN-TO-FILE CONTAINMENT")
    rule()
    paths, per_file, n_unique, multi = part_b1()
    emit("%-28s%18s%18s" % ("parquet file", "rows", "distinct loans"))
    rule()
    tot_rows = 0
    tot_loans = 0
    for name, nrows, nloans in per_file:
        tot_rows += nrows
        tot_loans += nloans
        emit("%-28s%18s%18s" % (name, fmt(nrows), fmt(nloans)))
    rule()
    emit("%-28s%18s%18s" % ("TOTAL", fmt(tot_rows), fmt(tot_loans)))
    emit()
    emit("%-54s%18s" % ("sum of per-file distinct loans", fmt(tot_loans)))
    emit("%-54s%18s" % ("distinct loans across all files", fmt(n_unique)))
    emit("%-54s%18s" % ("loans appearing in more than one file", fmt(len(multi))))
    if multi:
        emit()
        emit("    loans in more than one file (up to 20 shown):")
        for lid in sorted(multi)[:20]:
            emit("        %s  in %d files" % (lid, multi[lid]))
    emit()

    emit("B.2  GAP-ONLY LOANS")
    rule()
    emit("Loans with more than one observed row and NO strictly consecutive pair")
    emit("anywhere: every candidate pair they have spans a hole, so the gap rule")
    emit("alone leaves them with no panel row. Counted from the performance files,")
    emit("independently of src/panel.py.")
    emit()
    emit("%-58s%18s%12s" % ("quantity", "loans", "of loans"))
    rule()
    emit("%-58s%18s%12s" % ("loans with more than one observed row",
                            fmt(acc.multi_row_loans),
                            pct(acc.multi_row_loans, acc.loans)))
    emit("%-58s%18s%12s" % ("of those, zero strictly consecutive pairs (gap-only)",
                            fmt(acc.gap_only_loans),
                            pct(acc.gap_only_loans, acc.loans)))
    emit("%-58s%18s%12s" % ("of those, every consecutive pair has an RA from-state",
                            fmt(acc.absorbing_only_loans),
                            pct(acc.absorbing_only_loans, acc.loans)))
    rule()
    emit("diagnostic 16 reports 0 loans losing every pair to R3 or R1.")
    if acc.gap_only_examples:
        emit()
        emit("    gap-only loans (up to 10 shown), with their observed periods:")
        for lid, periods in acc.gap_only_examples:
            emit("        %s  %s" % (lid, periods))
    emit()

    emit("B.3  FIRST-PAYMENT-DATE OFFSET")
    rule()
    emit("offset = first payment month - first observed reporting period, in whole")
    emit("months, over %d loans sampled without replacement from %s origination"
         % (SAMPLE_N, fmt(n_orig)))
    emit("records with numpy default_rng(seed=%d)." % SAMPLE_SEED)
    emit("src/panel.py takes the origination month to be first payment minus 1.")
    emit()
    offsets = []
    missing = 0
    for _, row in sample.iterrows():
        first = acc.sample_first_period.get(row["loan_id"])
        if first is None or row["fp"] < 0:
            missing += 1
            continue
        offsets.append(int(row["fp"]) - int(first))
    offsets = np.array(offsets, dtype=np.int64)
    emit("%-28s%16s%15s" % ("offset (months)", "loans", "share"))
    rule()
    if len(offsets):
        vals, counts = np.unique(offsets, return_counts=True)
        for v, c in zip(vals, counts):
            emit("%-28s%16s%15s" % (v, fmt(c), pct(int(c), len(offsets))))
        rule()
        emit("%-28s%16s%15s" % ("TOTAL", fmt(len(offsets)), "100.0000%"))
        emit()
        modal = int(vals[int(np.argmax(counts))])
        emit("%-40s %s" % ("modal offset (months)", modal))
        emit("%-40s %s of %s (%s)"
             % ("loans at the modal offset", fmt(counts.max()),
                fmt(len(offsets)), pct(int(counts.max()), len(offsets))))
        emit("%-40s %d / %d" % ("min / max offset", offsets.min(), offsets.max()))
        emit("%-40s %s" % ("sampled loans with no match", fmt(missing)))
    else:
        emit("no sampled loan could be matched")
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
