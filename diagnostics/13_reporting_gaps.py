"""Are a loan's reporting periods strictly consecutive months, and where not, what is missing?

The panel is built by pairing each loan-month with the month that follows it.
That pairing is only meaningful if the next ROW is the next MONTH. Where a
loan's observed history has an interior hole, the naive pair spans more than one
month and silently measures a different quantity from every other pair in the
table. This script measures the holes: how many, how wide, how concentrated, and
what state the loan was in on either side of them.

DEFINITIONS
    interior gap   for a loan's periods sorted ascending, a consecutive pair
                   (p_prev, p_next) with gap = (p_next - p_prev) - 1 > 0.
                   gap == 0 is strictly consecutive. The FIRST and LAST observed
                   periods are spell boundaries, not gaps: a loan that starts
                   late or stops early contributes nothing here.
    gap event      one such pair. A loan with several holes contributes several
                   gap events, and section C counts events per loan.
    row after gap  the row at p_next. Exactly one per gap event, so the count of
                   gap events IS the count of loan-months sitting immediately
                   after a gap.

STATE MAPPING - the project's existing one, unchanged
    Field 4 (current loan delinquency status) is read as text and cast only
    where .str.isdigit() holds, so "RA" and "XX" are never coerced to a number.
        numeric 0        Current
        numeric 1        DQ30
        numeric 2        DQ60
        numeric >= 3     DQ90+     (98 and 99 included; per D12 the 99 is the
                                    documented cap, not a missing-value code)
        "RA"             RA        REO acquisition, reported as its own column
        anything else    other     "XX = Not Available", blanks, stray codes
    RA and "other" are NOT folded into the four delinquency states and are NOT
    dropped. They are carried as their own columns in sections D and E so that
    every gap event is visible in some cell and no row is silently discarded.

TWO PASSES, NO FILE HELD WHOLE
    pass 1  columns [0, 1, 3], every row of all eight files. Accumulates
            (a) the baseline state distribution over ALL loan-months, and
            (b) per loan, five order-independent moments of its period set:
                n (row count), pmin, pmax, sum of periods, sum of squared
                periods. Chunk and file boundaries therefore need no special
                handling and no assumption is made that a loan's rows are
                sorted or contiguous.
    screen  a loan's period multiset is exactly {pmin .. pmax} with no repeats
            if and only if n, sum and sum-of-squares all equal the values that
            set implies. n alone would miss a loan carrying one duplicate month
            AND one hole (the two cancel in the row count); the second and third
            moments cannot both be preserved by such a substitution. Loans
            failing the screen are FLAGGED - a superset of the gapped loans,
            also catching any duplicate (loan, period) pair.
    pass 2  columns [0, 1, 3] again, keeping only rows whose loan is FLAGGED.
            That subset is small enough to hold whole, so it is sorted per loan
            and walked pair by pair. Every number in sections A-F comes from
            this exact walk, not from the screen. Because pass 2 filters all
            eight files by the flagged set, a loan appearing in more than one
            file would still be assembled correctly; the count of flagged loans
            found in more than one file is reported.

    Rows whose period does not parse as YYYYMM with month in 1..12 are excluded
    from the period arithmetic and counted separately.

No plots. No panel-construction code. No git command is run.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "13_reporting_gaps.txt")
RULE_W = 121

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
PASS_COLS = [COL_LOAN, COL_PERIOD, COL_DQ]
COL_NAMES = "0 loan_id, 1 reporting_period, 3 dq_status"

ST_CUR, ST_DQ30, ST_DQ60, ST_DQ90, ST_RA, ST_OTHER = range(6)
ST_LABELS = ["Current", "DQ30", "DQ60", "DQ90+", "RA", "other"]
NSTATE = 6

BUCKET_LABELS = ["1", "2", "3", "4", "5", "6-11", "12-23", "24+"]
NBUCKET = len(BUCKET_LABELS)

XTAB_LABELS = ["gap = 1", "gap = 2", "gap = 3", "gap >= 4"]

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
    """YYYYMM -> a contiguous month index year*12 + (month-1). Invalid -> -1."""
    v = pd.to_numeric(s, errors="coerce")
    year, month = v // 100, v % 100
    idx = (year * 12 + (month - 1)).where((month >= 1) & (month <= 12))
    return idx.fillna(-1).astype(np.int64)


def mlabel(i):
    return "%04d-%02d" % (i // 12, i % 12 + 1)


def myear(i):
    return i // 12


def encode_state(s):
    """Field 4 -> one of the six state codes. Casts are guarded by isdigit()."""
    digits = s.str.isdigit().to_numpy()
    num = pd.to_numeric(s.where(s.str.isdigit()), errors="coerce")
    num = num.fillna(-1).to_numpy()
    code = np.full(len(s), ST_OTHER, dtype=np.int8)
    code[digits & (num == 0)] = ST_CUR
    code[digits & (num == 1)] = ST_DQ30
    code[digits & (num == 2)] = ST_DQ60
    code[digits & (num >= 3)] = ST_DQ90
    code[(~digits) & (s.to_numpy() == "RA")] = ST_RA
    return code


def bucket_of(gap):
    """Gap size in months -> index into BUCKET_LABELS."""
    return np.select(
        [gap <= 5, gap <= 11, gap <= 23],
        [gap - 1, 5, 6],
        default=7,
    ).astype(np.int64)


def reader_for(path):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=PASS_COLS,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


# --------------------------------------------------------------------------
# pass 1 - baseline state counts, and per-loan period moments
# --------------------------------------------------------------------------
def combine(a, b):
    if a is None:
        return b
    g = pd.concat([a, b]).groupby(level=0, sort=False)
    return pd.DataFrame({
        "n": g["n"].sum(),
        "pmin": g["pmin"].min(),
        "pmax": g["pmax"].max(),
        "psum": g["psum"].sum(),
        "psq": g["psq"].sum(),
    })


def pass1(path, state_counts):
    """Per-loan period moments for one file, plus rows / bad-period tally."""
    agg = None
    rows = 0
    bad = 0
    for chunk in reader_for(path):
        rows += len(chunk)
        loan = chunk[COL_LOAN].str.strip()
        per = encode_period(chunk[COL_PERIOD].str.strip())
        st = encode_state(chunk[COL_DQ].str.strip())

        state_counts += np.bincount(st, minlength=NSTATE)

        ok = (per >= 0).to_numpy()
        bad += int((~ok).sum())

        p = per.to_numpy()[ok]
        d = pd.DataFrame({"loan": loan.to_numpy()[ok], "p": p, "psq": p * p})
        g = d.groupby("loan", sort=False)
        part = pd.DataFrame({
            "n": g["p"].size(),
            "pmin": g["p"].min(),
            "pmax": g["p"].max(),
            "psum": g["p"].sum(),
            "psq": g["psq"].sum(),
        })
        agg = combine(agg, part)
    return agg, rows, bad


def screen(agg):
    """Loans whose period multiset is not exactly {pmin..pmax} with no repeats."""
    pmin = agg["pmin"].to_numpy()
    pmax = agg["pmax"].to_numpy()
    span = pmax - pmin + 1
    exp_sum = (pmin + pmax) * span // 2

    def s2(m):
        return m * (m + 1) * (2 * m + 1) // 6

    exp_sq = s2(pmax) - s2(pmin - 1)
    bad = ((agg["n"].to_numpy() != span)
           | (agg["psum"].to_numpy() != exp_sum)
           | (agg["psq"].to_numpy() != exp_sq))
    return agg.index.to_numpy()[bad]


# --------------------------------------------------------------------------
# pass 2 - every row of the flagged loans
# --------------------------------------------------------------------------
def pass2(path, flagged):
    parts = []
    rows = 0
    for chunk in reader_for(path):
        rows += len(chunk)
        loan = chunk[COL_LOAN].str.strip()
        keep = loan.isin(flagged).to_numpy()
        if not keep.any():
            continue
        per = encode_period(chunk[COL_PERIOD].str.strip())
        st = encode_state(chunk[COL_DQ].str.strip())
        ok = keep & (per >= 0).to_numpy()
        if not ok.any():
            continue
        parts.append(pd.DataFrame({
            "loan": loan.to_numpy()[ok],
            "p": per.to_numpy()[ok],
            "st": st[ok],
        }))
    return parts, rows


# --------------------------------------------------------------------------
# table helpers
# --------------------------------------------------------------------------
def state_header(first="bucket"):
    emit("%-22s%12s%s" % (first, "rows",
                          "".join("%14s" % l for l in ST_LABELS)))


def state_row(label, counts):
    counts = np.asarray(counts, dtype=np.int64)
    emit("%-22s%12s%s" % (label, fmt(counts.sum()),
                          "".join("%14s" % fmt(c) for c in counts)))


def share_header(first="bucket"):
    emit("%-22s%12s%s" % (first, "rows",
                          "".join("%10s" % l for l in ST_LABELS)))


def share_row(label, counts):
    counts = np.asarray(counts, dtype=np.int64)
    tot = int(counts.sum())
    cells = "".join("%10s" % (pct(c, tot, 2) if tot else "n/a") for c in counts)
    emit("%-22s%12s%s" % (label, fmt(tot), cells))


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("13_reporting_gaps - interior holes in each loan's monthly reporting history")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns read: %s" % COL_NAMES)
    emit("gap         : (next period - previous period) - 1, in whole calendar months,")
    emit("              over one loan's periods sorted ascending; 0 = strictly consecutive")
    emit("boundaries  : a loan's first and last observed periods are spell edges, not gaps")
    emit("states      : 0 Current, 1 DQ30, 2 DQ60, >=3 DQ90+ (98/99 inside DQ90+);")
    emit("              'RA' and non-numeric codes carried as their own columns, never dropped")
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-64s %6.2f GB" % (p, os.path.getsize(p) / 1024.0 ** 3))
    if not paths:
        emit()
        emit("NO PERFORMANCE FILES MATCHED THE GLOB - nothing measured.")
        write_out()
        return
    emit()

    # ---------------- pass 1 ----------------
    emit("PASS 1/2 - per-loan period moments over every row")
    rule()
    state_counts = np.zeros(NSTATE, dtype=np.int64)
    total_rows = 0
    total_bad = 0
    flagged = set()
    loan_arrays = []
    per_file_loans = []
    for path in paths:
        t0 = time.time()
        agg, rows, bad = pass1(path, state_counts)
        total_rows += rows
        total_bad += bad
        bad_loans = screen(agg)
        flagged.update(bad_loans.tolist())
        loan_arrays.append(agg.index.to_numpy())
        per_file_loans.append(len(agg))
        emit("    %-18s rows %12s   loans %9s   flagged %6s   (%5.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(agg)),
                fmt(len(bad_loans)), time.time() - t0))
    all_loans = np.concatenate(loan_arrays)
    pooled_loans = len(np.unique(all_loans))
    emit("    pooled rows %s   pooled distinct loans %s   flagged loans %s"
         % (fmt(total_rows), fmt(pooled_loans), fmt(len(flagged))))
    emit("    rows with unparseable reporting period: %s" % fmt(total_bad))
    emit("    cross-file loan overlap: %s loan(s) appear in more than one file"
         % fmt(sum(per_file_loans) - pooled_loans))
    emit()

    valid_rows = total_rows - total_bad

    # ---------------- pass 2 ----------------
    emit("PASS 2/2 - every row of the flagged loans, walked pair by pair")
    rule()
    parts = []
    found_per_file = {}
    for path in paths:
        t0 = time.time()
        fparts, rows = pass2(path, flagged)
        n_here = sum(len(x) for x in fparts)
        loans_here = set()
        for x in fparts:
            loans_here.update(x["loan"].unique().tolist())
        for lid in loans_here:
            found_per_file[lid] = found_per_file.get(lid, 0) + 1
        parts.extend(fparts)
        emit("    %-18s rows %12s   flagged rows kept %8s   flagged loans %6s   (%5.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(n_here),
                fmt(len(loans_here)), time.time() - t0))

    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame({"loan": [], "p": [], "st": []}))
    multi_file = sum(1 for v in found_per_file.values() if v > 1)
    emit("    flagged rows held in memory: %s" % fmt(len(df)))
    emit("    flagged loans found in more than one file: %s" % fmt(multi_file))
    emit()

    # ---------------- the walk ----------------
    df = df.sort_values(["loan", "p"], kind="mergesort").reset_index(drop=True)
    prev_loan = df["loan"].shift(1)
    same = (prev_loan == df["loan"]).to_numpy()
    delta = (df["p"] - df["p"].shift(1)).to_numpy()
    gap = np.where(same, delta - 1, np.nan)

    is_gap = same & (gap > 0)
    is_dup = same & (gap == -1)

    gap_sizes = gap[is_gap].astype(np.int64)
    st_after = df["st"].to_numpy()[is_gap]
    st_before = df["st"].shift(1).to_numpy()[is_gap].astype(np.int64)
    p_before = df["p"].shift(1).to_numpy()[is_gap].astype(np.int64)
    loan_of_gap = df["loan"].to_numpy()[is_gap]

    n_gaps = int(is_gap.sum())
    n_dups = int(is_dup.sum())
    gapped_loans = np.unique(loan_of_gap)
    n_gapped_loans = len(gapped_loans)
    missing_months = int(gap_sizes.sum()) if n_gaps else 0

    # ---------------- A ----------------
    rule("=")
    emit("A. HEADLINE")
    rule("=")
    emit("Two units appear below and are never mixed: LOANS (one mortgage) and")
    emit("LOAN-MONTHS (one observed (loan, month) row).")
    emit()
    emit("%-58s %18s %12s" % ("quantity", "count", "share"))
    rule()
    emit("%-58s %18s %12s" % ("loans in panel", fmt(pooled_loans), "100.0000%"))
    emit("%-58s %18s %12s" % ("loans flagged by the moment screen",
                              fmt(len(flagged)), pct(len(flagged), pooled_loans)))
    emit("%-58s %18s %12s" % ("loans with >= 1 interior gap (confirmed by walk)",
                              fmt(n_gapped_loans), pct(n_gapped_loans, pooled_loans)))
    emit("%-58s %18s %12s" % ("loans strictly consecutive throughout",
                              fmt(pooled_loans - n_gapped_loans),
                              pct(pooled_loans - n_gapped_loans, pooled_loans)))
    rule()
    emit("%-58s %18s %12s" % ("loan-months in panel", fmt(valid_rows), "100.0000%"))
    emit("%-58s %18s %12s" % ("interior gap events (holes, not months)",
                              fmt(n_gaps), pct(n_gaps, valid_rows)))
    emit("%-58s %18s %12s" % ("loan-months sitting immediately after a gap",
                              fmt(n_gaps), pct(n_gaps, valid_rows)))
    emit("%-58s %18s %12s" % ("missing loan-months implied by the gaps",
                              fmt(missing_months),
                              pct(missing_months, valid_rows)))
    rule()
    emit("%-58s %18s" % ("duplicate (loan, period) pairs found", fmt(n_dups)))
    emit("%-58s %18s" % ("rows with unparseable reporting period", fmt(total_bad)))
    emit("%-58s %18s" % ("flagged loans with no interior gap",
                         fmt(len(flagged) - n_gapped_loans)))
    emit()
    emit("The row-after-a-gap count equals the gap-event count by construction: each")
    emit("interior gap has exactly one row on its far side.")
    emit()

    if n_gaps == 0:
        rule("=")
        emit("NO INTERIOR GAPS OBSERVED - sections B to F are empty.")
        rule("=")
        emit()
        emit("runtime: %.1f s  (%.2f min)"
             % (time.time() - t_start, (time.time() - t_start) / 60.0))
        write_out()
        return

    buckets = bucket_of(gap_sizes)

    # ---------------- B ----------------
    rule("=")
    emit("B. GAP-SIZE DISTRIBUTION   (unit: gap events)")
    rule("=")
    emit("Gap size is the number of CONSECUTIVE MONTHS MISSING between the two")
    emit("observed rows. A gap of 1 means one month absent, so the pair spans two")
    emit("calendar months rather than one.")
    emit()
    emit("%-22s%16s%12s%22s" % ("gap size (months)", "gap events", "share",
                                "loan-months missing"))
    rule()
    bcount = np.bincount(buckets, minlength=NBUCKET)
    bmissing = np.bincount(buckets, weights=gap_sizes,
                           minlength=NBUCKET).astype(np.int64)
    for i, lab in enumerate(BUCKET_LABELS):
        emit("%-22s%16s%12s%22s" % (lab, fmt(bcount[i]), pct(bcount[i], n_gaps),
                                    fmt(bmissing[i])))
    rule()
    emit("%-22s%16s%12s%22s" % ("TOTAL", fmt(n_gaps), "100.0000%",
                                fmt(missing_months)))
    emit()
    emit("%-38s %s" % ("min observed gap (months)", fmt(gap_sizes.min())))
    emit("%-38s %s" % ("max observed gap (months)", fmt(gap_sizes.max())))
    emit("%-38s %.2f" % ("mean gap (months)", gap_sizes.mean()))
    emit("%-38s %.1f" % ("median gap (months)", float(np.median(gap_sizes))))
    emit()

    # ---------------- C ----------------
    rule("=")
    emit("C. INTERIOR GAPS PER LOAN   (unit: loans with >= 1 interior gap)")
    rule("=")
    emit("Whether a hole is an isolated incident or a recurring property of a loan's")
    emit("servicing record.")
    emit()
    per_loan = pd.Series(loan_of_gap).value_counts().to_numpy()
    c1 = int((per_loan == 1).sum())
    c2 = int((per_loan == 2).sum())
    c3 = int((per_loan == 3).sum())
    c4 = int((per_loan >= 4).sum())
    emit("%-22s%16s%18s%22s" % ("gaps in the loan", "loans", "share of gapped",
                                "share of all loans"))
    rule()
    for lab, c in (("exactly 1", c1), ("exactly 2", c2), ("exactly 3", c3),
                   ("4 or more", c4)):
        emit("%-22s%16s%18s%22s" % (lab, fmt(c), pct(c, n_gapped_loans),
                                    pct(c, pooled_loans)))
    rule()
    emit("%-22s%16s%18s%22s" % ("TOTAL", fmt(n_gapped_loans), "100.0000%",
                                pct(n_gapped_loans, pooled_loans)))
    emit()
    emit("%-38s %s" % ("max interior gaps in one loan", fmt(per_loan.max())))
    emit("%-38s %.2f" % ("mean interior gaps per gapped loan", per_loan.mean()))
    emit()

    # ---------------- D ----------------
    before_by_bucket = np.zeros((NBUCKET, NSTATE), dtype=np.int64)
    np.add.at(before_by_bucket, (buckets, st_before), 1)

    rule("=")
    emit("D. STATE OF THE ROW IMMEDIATELY BEFORE THE GAP, BY GAP SIZE")
    rule("=")
    emit("Unit: gap events, one per hole, filed under the delinquency state recorded")
    emit("in the last month observed BEFORE the hole. The baseline row is the state")
    emit("distribution over every loan-month in the panel - what the rows before gaps")
    emit("would look like if holes fell at random with respect to delinquency.")
    emit()
    emit("D.1  COUNTS   (unit: gap events; the BASELINE row is loan-months)")
    rule()
    state_header("gap size (months)")
    rule()
    for i, lab in enumerate(BUCKET_LABELS):
        state_row(lab, before_by_bucket[i])
    rule()
    state_row("ALL GAPS", before_by_bucket.sum(axis=0))
    state_row("BASELINE loan-months", state_counts)
    emit()
    emit("D.2  ROW SHARES of that same table - every percentage has its exact count")
    emit("     in the corresponding cell of D.1 directly above.")
    rule()
    share_header("gap size (months)")
    rule()
    for i, lab in enumerate(BUCKET_LABELS):
        share_row(lab, before_by_bucket[i])
    rule()
    share_row("ALL GAPS", before_by_bucket.sum(axis=0))
    share_row("BASELINE loan-months", state_counts)
    emit()

    # ---------------- E ----------------
    rule("=")
    emit("E. STATE BEFORE THE GAP vs STATE AFTER THE GAP")
    rule("=")
    emit("Unit: gap events. Rows are the state in the last month before the hole,")
    emit("columns the state in the first month after it. The diagonal is where a naive")
    emit("pair-across would report no transition; everything off it is movement the")
    emit("panel never observed month by month.")
    emit()
    xgroup = np.where(gap_sizes >= 4, 3, gap_sizes - 1)
    for k, xlab in enumerate(XTAB_LABELS):
        sel = xgroup == k
        n_sel = int(sel.sum())
        tab = np.zeros((NSTATE, NSTATE), dtype=np.int64)
        if n_sel:
            np.add.at(tab, (st_before[sel], st_after[sel]), 1)
        emit("E.%d  %s   (gap events: %s, %s of all gaps)"
             % (k + 1, xlab, fmt(n_sel), pct(n_sel, n_gaps)))
        rule()
        emit("%-22s%12s%s" % ("before \\ after", "rows",
                              "".join("%14s" % l for l in ST_LABELS)))
        rule()
        for i, lab in enumerate(ST_LABELS):
            state_row(lab, tab[i])
        rule()
        state_row("TOTAL after", tab.sum(axis=0))
        if n_sel:
            diag = int(np.trace(tab))
            emit("    unchanged across the gap (diagonal): %s of %s (%s)"
                 % (fmt(diag), fmt(n_sel), pct(diag, n_sel)))
        emit()

    # ---------------- F ----------------
    rule("=")
    emit("F. CALENDAR CONCENTRATION   (unit: gap events)")
    rule("=")
    emit("Filed under the YEAR of the reporting period immediately BEFORE the hole -")
    emit("the last month observed before reporting stops.")
    emit()
    years = myear(p_before)
    vc = pd.Series(years).value_counts().sort_index()
    emit("%-22s%16s%12s%16s" % ("year before gap", "gap events", "share",
                                "cumulative"))
    rule()
    run = 0
    for y, c in vc.items():
        run += int(c)
        emit("%-22s%16s%12s%16s" % (int(y), fmt(c), pct(int(c), n_gaps),
                                    pct(run, n_gaps)))
    rule()
    emit("%-22s%16s%12s%16s" % ("TOTAL", fmt(n_gaps), "100.0000%", "100.0000%"))
    emit()
    emit("%-44s %s" % ("earliest month before a gap", mlabel(int(p_before.min()))))
    emit("%-44s %s" % ("latest month before a gap", mlabel(int(p_before.max()))))
    pre_2009 = int((p_before < (2009 * 12)).sum())
    emit("%-44s %s of %s (%s)"
         % ("pre-gap month before 2009-01", fmt(pre_2009),
            fmt(n_gaps), pct(pre_2009, n_gaps)))
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t_start, (time.time() - t_start) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
