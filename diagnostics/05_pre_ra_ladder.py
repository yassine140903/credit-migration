"""What the delinquency ladder looks like BEFORE a loan's first RA month.

Diagnostic 04 tested a single threshold (numeric status >= 6 before first RA) and
found 7,800 ever-RA loans failing it. One threshold cannot separate a loan that
skipped the ladder entirely from one that walked it normally and was foreclosed
before reaching status 6. This script replaces the threshold with the full
distribution: the maximum status reached before first RA, the status in the month
immediately preceding it, and how much history exists before it at all.

Restricted throughout to loans that ever show "RA" in column 3. For each such
loan, only rows strictly before its first RA month are considered.

Counts only. Nothing here classifies, relabels, or recommends a state definition.

UNITS. Two quantities appear and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation

THREE PASSES. A loan's rows are not assumed sorted or contiguous and may straddle
chunk and file boundaries; no per-loan state is inferred from row adjacency.
    pass 1  [0,1,3]  first RA month per loan. The groupby index IS the ever-RA
                     set, so the set and the month come from one scan.
    pass 2  [0,1,3]  restricted to ever-RA, rows strictly before first RA:
                     max numeric status, whether any numeric row exists, the
                     count of prior months, and the (month, status) of the latest
                     prior row.
    pass 3  [0,1,3]  restricted to a handful of named loans: their full status
                     sequences, for direct inspection.
Pass 2 cannot fold into pass 1: "strictly before" is undefined until the first RA
month is known for the whole panel, and a chunk read early may hold rows that
precede a first RA month discovered in a chunk read later. Pass 3 cannot fold in
either: which loans to print is decided by pass 2's results.

The latest-prior-row reduction is an argmax over the month index, applied per
chunk and then again across chunks, which is order-independent.

Numeric casts of column 3 are guarded by .str.isdigit(), so "RA" and any other
non-numeric code are never coerced.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "05_pre_ra_ladder.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3

RA = "RA"
DQ_DEEP = 6
MAX_SEQUENCES = 10

# Cross-checks from diagnostic 04 on the same 8-file glob. A mismatch means one
# of the two scripts is wrong; it is reported as a bug, not absorbed.
EXPECTED_RA_LOANS = 118_026
EXPECTED_NO_PRIOR_DEEP = 7_800

_LINES = []


def emit(line=""):
    print(line, flush=True)
    _LINES.append(line)


def write_out():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")


def rule(char="-", width=78):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den):
    return "n/a" if not den else "%.4f%%" % (100.0 * num / den)


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


def distribution(values, unit, label):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    emit("%-16s %14s   %s" % ("statistic", unit, label))
    rule()
    if not v.size:
        emit("(empty)")
        return
    emit("%-16s %14s" % ("n [loans]", fmt(v.size)))
    emit("%-16s %14.3f" % ("mean", v.mean()))
    emit("%-16s %14.3f" % ("std", v.std(ddof=0)))
    emit("%-16s %14d" % ("min", int(v.min())))
    emit("%-16s %14.1f" % ("p1", np.percentile(v, 1)))
    for q in range(10, 100, 10):
        tag = "p50 (median)" if q == 50 else "p%-2d (d%d)" % (q, q // 10)
        emit("%-16s %14.1f" % (tag, np.percentile(v, q)))
    emit("%-16s %14.1f" % ("p99", np.percentile(v, 99)))
    emit("%-16s %14d" % ("max", int(v.max())))


# --------------------------------------------------------------------------
# pass 1 - first RA month per loan (its index is the ever-RA set)
# --------------------------------------------------------------------------
def pass1(paths):
    parts = []
    ra_months = 0
    total_rows = 0
    all_parts = []
    for path in paths:
        t0 = time.time()
        rows = 0
        f_all = []
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            rows += len(chunk)
            loan = chunk[COL_LOAN].str.strip()
            f_all.append(loan.unique())
            m = chunk[COL_DQ].str.strip() == RA
            if not m.any():
                continue
            ra_months += int(m.sum())
            df = pd.DataFrame({
                "loan": loan[m].to_numpy(),
                "idx": yyyymm_to_index(chunk.loc[m, COL_PERIOD].str.strip()).to_numpy(),
            })
            parts.append(df.groupby("loan", sort=False)["idx"].min())
        total_rows += rows
        a = pd.unique(np.concatenate(f_all)) if f_all else np.array([])
        all_parts.append(a)
        emit("    %-30s rows %13s  loans %11s  (%.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(a)), time.time() - t0))
    first_ra = (pd.concat(parts).groupby(level=0).min()
                if parts else pd.Series(dtype="float64"))
    n_all = len(pd.unique(np.concatenate(all_parts))) if all_parts else 0
    return first_ra, ra_months, total_rows, n_all


# --------------------------------------------------------------------------
# pass 2 - pre-RA aggregates, restricted to the ever-RA set
# --------------------------------------------------------------------------
def pass2(paths, ra_set, first_ra):
    agg_parts, last_parts = [], []
    for path in paths:
        t0 = time.time()
        kept = 0
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(ra_set)
            if not keep.any():
                continue
            loan = loan[keep]
            status = chunk.loc[keep, COL_DQ].str.strip()
            idx = yyyymm_to_index(chunk.loc[keep, COL_PERIOD].str.strip())
            before = idx < loan.map(first_ra)
            if not before.any():
                continue
            kept += int(before.sum())

            digit = status.str.isdigit()
            num = pd.to_numeric(status.where(digit), errors="coerce")

            pre = pd.DataFrame({
                "loan": loan[before].to_numpy(),
                "idx": idx[before].to_numpy(),
                "status": status[before].to_numpy(),
                "num": num[before].to_numpy(),
                "is_num": digit[before].to_numpy(),
            })
            g = pre.groupby("loan", sort=False)
            agg_parts.append(pd.DataFrame({
                "max_num": g["num"].max(),
                "n_numeric": g["is_num"].sum(),
                "n_months": g.size(),
            }))
            # latest prior row in this chunk, carried with its status
            imax = g["idx"].idxmax()
            last_parts.append(pre.loc[imax, ["loan", "idx", "status"]])
        emit("    %-30s pre-RA rows %11s   (%.1fs)"
             % (os.path.basename(path), fmt(kept), time.time() - t0))

    cat = pd.concat(agg_parts, copy=False)
    g = cat.groupby(level=0, sort=False)
    agg = pd.DataFrame({
        "max_num": g["max_num"].max(),
        "n_numeric": g["n_numeric"].sum(),
        "n_months": g["n_months"].sum(),
    })
    lat = pd.concat(last_parts, ignore_index=True, copy=False)
    lat = lat.loc[lat.groupby("loan")["idx"].idxmax()].set_index("loan")
    return agg, lat


# --------------------------------------------------------------------------
# pass 3 - full status sequences for a handful of named loans
# --------------------------------------------------------------------------
def pass3(paths, wanted):
    parts = []
    for path in paths:
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(wanted)
            if not keep.any():
                continue
            parts.append(pd.DataFrame({
                "loan": loan[keep].to_numpy(),
                "idx": yyyymm_to_index(
                    chunk.loc[keep, COL_PERIOD].str.strip()).to_numpy(),
                "status": chunk.loc[keep, COL_DQ].str.strip().to_numpy(),
            }))
    if not parts:
        return pd.DataFrame(columns=["loan", "idx", "status"])
    return pd.concat(parts, ignore_index=True, copy=False)


def freq_table(series, n_total, header, none_label, none_count):
    """Frequency table over contiguous integer statuses 0..max, plus a 'none' row."""
    emit("%-14s %14s %12s %12s" % (header, "loans", "share", "cumulative"))
    rule()
    cum = 0
    v = series.dropna()
    if len(v):
        counts = v.astype(int).value_counts()
        for s in range(0, int(counts.index.max()) + 1):
            c = int(counts.get(s, 0))
            cum += c
            emit("%-14d %14s %12s %12s" % (s, fmt(c), pct(c, n_total), pct(cum, n_total)))
    cum += none_count
    emit("%-14s %14s %12s %12s"
         % (none_label, fmt(none_count), pct(none_count, n_total), pct(cum, n_total)))
    rule()
    emit("%-14s %14s %12s" % ("TOTAL", fmt(n_total), pct(n_total, n_total)))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("05_pre_ra_ladder - delinquency ladder before the first RA month")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : %d loan, %d period, %d dq_status"
         % (COL_LOAN, COL_PERIOD, COL_DQ))
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/3 - first RA month (and thereby the ever-RA set)")
    rule()
    first_ra, ra_months, total_rows, n_all = pass1(paths)
    n_ra = len(first_ra)
    ra_set = set(first_ra.index.tolist())

    emit()
    emit("PASS 2/3 - pre-RA aggregates")
    rule()
    agg, lat = pass2(paths, ra_set, first_ra)
    agg = agg.reindex(first_ra.index)
    agg["n_months"] = agg["n_months"].fillna(0)
    agg["n_numeric"] = agg["n_numeric"].fillna(0)

    # ---- groups ----
    has_num = agg["n_numeric"] > 0
    deep = has_num & (agg["max_num"] >= DQ_DEEP)
    never_deep = ~deep
    n_deep = int(deep.sum())
    n_never = int(never_deep.sum())
    n_no_num = int((~has_num).sum())

    emit()
    rule("=")
    emit("SCOPE AND CROSS-CHECKS")
    rule("=")
    emit("%-52s %14s %10s" % ("all distinct loans in panel   [loans]",
                              fmt(n_all), pct(n_all, n_all)))
    emit("%-52s %14s %10s" % ("loans ever showing 'RA'       [loans]",
                              fmt(n_ra), pct(n_ra, n_all)))
    emit("%-52s %14s" % ("RA observations         [loan-months]", fmt(ra_months)))
    emit("%-52s %14s" % ("total panel rows        [loan-months]", fmt(total_rows)))
    rule()
    emit("%-52s %14s" % ("expected ever-RA loans (diagnostic 04)",
                         fmt(EXPECTED_RA_LOANS)))
    emit("%-52s %14s" % ("observed ever-RA loans", fmt(n_ra)))
    emit("%-52s %14s" % ("status", "MATCH" if n_ra == EXPECTED_RA_LOANS
                         else "*** MISMATCH ***"))
    emit("%-52s %14s" % ("expected never->=%d group (diagnostic 04)" % DQ_DEEP,
                         fmt(EXPECTED_NO_PRIOR_DEEP)))
    emit("%-52s %14s" % ("observed never->=%d group" % DQ_DEEP, fmt(n_never)))
    emit("%-52s %14s" % ("status", "MATCH" if n_never == EXPECTED_NO_PRIOR_DEEP
                         else "*** MISMATCH ***"))
    if n_ra != EXPECTED_RA_LOANS or n_never != EXPECTED_NO_PRIOR_DEEP:
        emit("A mismatch is a BUG in one of the two scripts, not a data finding.")
    rule()
    emit("%-52s %14s %10s" % ("group A: numeric max >= %d before first RA" % DQ_DEEP,
                              fmt(n_deep), pct(n_deep, n_ra)))
    emit("%-52s %14s %10s" % ("group B: never reached >= %d before first RA" % DQ_DEEP,
                              fmt(n_never), pct(n_never, n_ra)))
    emit("%-52s %14s %10s" % ("  of group B, no numeric row at all",
                              fmt(n_no_num), pct(n_no_num, n_ra)))

    # ---------------- (a) ----------------
    emit()
    rule("=")
    emit("(a) MAXIMUM NUMERIC DELINQUENCY STATUS BEFORE FIRST RA   (unit: loans)")
    rule("=")
    emit("ALL ever-RA loans (n = %s)" % fmt(n_ra))
    rule()
    freq_table(agg.loc[has_num, "max_num"], n_ra, "max status",
               "no numeric row", n_no_num)
    emit()
    emit("GROUP B ONLY - loans never reaching >= %d before first RA (n = %s)"
         % (DQ_DEEP, fmt(n_never)))
    rule()
    freq_table(agg.loc[never_deep & has_num, "max_num"], n_never, "max status",
               "no numeric row", n_no_num)

    # ---------------- (b) ----------------
    lat = lat.reindex(first_ra.index)
    gap = (first_ra - lat["idx"])
    n_no_prior = int(lat["idx"].isna().sum())
    emit()
    rule("=")
    emit("(b) STATUS IN THE MONTH IMMEDIATELY PRECEDING FIRST RA   (unit: loans)")
    rule("=")
    emit("Raw values, unbucketed. 'immediately preceding' = the latest row strictly")
    emit("before the first RA month; where the panel has a gap that row is more than")
    emit("one month earlier, which is quantified below the table.")
    rule()
    emit("%-14s %14s %12s %12s" % ("status", "loans", "share", "cumulative"))
    rule()
    vc = lat["status"].value_counts(dropna=True)
    cum = 0
    for s, c in vc.items():
        cum += int(c)
        shown = s if s != "" else "(blank)"
        emit("%-14s %14s %12s %12s"
             % (shown, fmt(int(c)), pct(int(c), n_ra), pct(cum, n_ra)))
    cum += n_no_prior
    emit("%-14s %14s %12s %12s"
         % ("no prior row", fmt(n_no_prior), pct(n_no_prior, n_ra), pct(cum, n_ra)))
    rule()
    emit("%-14s %14s %12s" % ("TOTAL", fmt(n_ra), pct(n_ra, n_ra)))
    emit()
    emit("DISTANCE FROM THAT ROW TO THE FIRST RA MONTH   (unit: loans)")
    rule()
    g1 = int((gap == 1).sum())
    g2 = int((gap > 1).sum())
    emit("%-52s %14s %10s" % ("exactly 1 month before (contiguous)",
                              fmt(g1), pct(g1, n_ra)))
    emit("%-52s %14s %10s" % ("more than 1 month before (gap in panel)",
                              fmt(g2), pct(g2, n_ra)))
    emit("%-52s %14s %10s" % ("no prior row at all",
                              fmt(n_no_prior), pct(n_no_prior, n_ra)))

    # ---------------- (c) ----------------
    emit()
    rule("=")
    emit("(c) MONTHS OBSERVED STRICTLY BEFORE FIRST RA   (unit: months, per loan)")
    rule("=")
    distribution(agg.loc[deep, "n_months"].to_numpy(), "months",
                 "group A: max >= %d" % DQ_DEEP)
    emit()
    distribution(agg.loc[never_deep, "n_months"].to_numpy(), "months",
                 "group B: never >= %d" % DQ_DEEP)

    # ---------------- (d) ----------------
    years = ((first_ra - 1) // 12).astype(int)
    ya = years[deep.reindex(years.index).fillna(False)]
    yb = years[never_deep.reindex(years.index).fillna(False)]
    va, vb = ya.value_counts(), yb.value_counts()
    emit()
    rule("=")
    emit("(d) FIRST RA MONTH BY CALENDAR YEAR, GROUP A vs GROUP B")
    rule("=")
    emit("%-8s %12s %10s   %12s %10s   %10s"
         % ("year", "A loans", "A share", "B loans", "B share", "B minus A"))
    rule()
    for y in sorted(set(va.index) | set(vb.index)):
        a, b = int(va.get(y, 0)), int(vb.get(y, 0))
        sa = 100.0 * a / n_deep if n_deep else 0.0
        sb = 100.0 * b / n_never if n_never else 0.0
        emit("%-8d %12s %9.4f%%   %12s %9.4f%%   %+9.4f%%"
             % (y, fmt(a), sa, fmt(b), sb, sb - sa))
    rule()
    emit("%-8s %12s %9.4f%%   %12s %9.4f%%"
         % ("TOTAL", fmt(n_deep), 100.0, fmt(n_never), 100.0))
    emit()
    emit("A = %s loans reaching numeric >= %d before first RA"
         % (fmt(n_deep), DQ_DEEP))
    emit("B = %s loans that never did. Shares are within each group." % fmt(n_never))

    # ---------------- (e) ----------------
    max0 = has_num & (agg["max_num"] == 0)
    n_max0 = int(max0.sum())
    emit()
    rule("=")
    emit("(e) THE TWO EXTREME CASES   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("max pre-RA numeric status is 0",
                              fmt(n_max0), pct(n_max0, n_ra)))
    emit("%-52s %14s %10s" % ("no numeric row before first RA at all",
                              fmt(n_no_num), pct(n_no_num, n_ra)))
    emit("(shares are of the %s ever-RA loans)" % fmt(n_ra))

    pick_a = agg.index[max0][: MAX_SEQUENCES // 2].tolist()
    pick_b = agg.index[~has_num][: MAX_SEQUENCES - len(pick_a)].tolist()
    wanted = set(pick_a) | set(pick_b)

    emit()
    emit("PASS 3/3 - full status sequences for %d sampled loan(s)" % len(wanted))
    rule()
    seq = pass3(paths, wanted) if wanted else pd.DataFrame(
        columns=["loan", "idx", "status"])

    for group_name, picks in (("max pre-RA numeric status == 0", pick_a),
                              ("no numeric row before first RA", pick_b)):
        emit()
        emit("--- %s : %d sampled loan(s) ---" % (group_name, len(picks)))
        for lid in picks:
            rows = seq[seq["loan"] == lid].sort_values("idx")
            f = first_ra.get(lid)
            emit()
            emit("loan %s   first RA %s   rows %s   pre-RA months %s"
                 % (lid, index_to_yyyymm(f), fmt(len(rows)),
                    fmt(int(agg.loc[lid, "n_months"]))))
            line = []
            for _, r in rows.iterrows():
                mark = "*" if r["idx"] == f else " "
                st = r["status"] if r["status"] != "" else "--"
                line.append("%s%s:%s" % (mark, index_to_yyyymm(r["idx"]), st))
            for i in range(0, len(line), 6):
                emit("    " + "  ".join(line[i:i + 6]))
        if not picks:
            emit("    (no loans in this group)")
    emit()
    emit("'*' marks the first RA month. '--' is a blank status field.")

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
