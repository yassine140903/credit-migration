"""What does current_loan_delinquency_status == "99" actually mean?

Status 99 spikes against an otherwise monotone decay in the status frequency
table. Three readings compete:
  (a) CAP      - "99 or more months delinquent", genuinely on the ladder,
                 censored at the field's two-character ceiling.
  (b) SENTINEL - "status not reported / unknown", not on the ladder at all.
  (c) TERMINAL - 99 appears only as a loan's final observed row.

Delinquency advances by at most one month per month, so under (a) a 99 row must
be preceded by 98 or 99 and cannot be followed by a low status. Under (b)
predecessors and successors resemble the general population. This script reports
the evidence that separates them. Counts only - it does not pick a reading.

UNITS. Three quantities appear and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation
    runs         - one maximal block of consecutive 99 months within one loan

TWO PASSES over the files. A loan's rows are not assumed sorted or contiguous and
may straddle chunk and file boundaries.
    pass 1  [0,1,3]    prevalence: the 90-99 loan-month counts, the loan
                       universe, the dataset's reporting-period range, and the
                       set of loans ever reporting 99.
    pass 2  [0,1,3,8]  the FULL row history of just those loans, gathered into
                       memory (a small slice of the panel), from which every
                       adjacency question is then answered exactly.

ADJACENCY. "Previous month" means the immediately preceding calendar month for
that loan, never merely the nearest earlier row. This is computed by self-merging
the history on (loan, month + 1) and (loan, month - 1), so a loan with a
reporting gap yields no match and is counted in its own "gap" category rather
than silently borrowing a non-adjacent row as its neighbour. Because the merge is
keyed on the month index itself, the result does not depend on row order.

Every numeric cast of column 3 is guarded by .str.isdigit(); "RA" and "XX" are
never coerced.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "06_status_99.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_ZBC = 8

TARGET = "99"
CURRENT = "00"
NEIGHBOUR_CODES = ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"]
NON_NUMERIC = ["RA", "XX"]

PREPAID = ["01"]
CREDIT_EVENT = ["02", "03", "09"]
ADMIN_EXIT = ["06", "15", "16"]

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


def distribution(values, unit):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    emit("%-16s %14s" % ("statistic", unit))
    rule()
    if not v.size:
        emit("(empty)")
        return
    emit("%-16s %14s" % ("n", fmt(v.size)))
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
# pass 1 - prevalence and the set of loans ever reporting 99
# --------------------------------------------------------------------------
def pass1(paths):
    status_counts = {}
    loan_parts, hit_parts = [], []
    total_rows = 0
    min_idx, max_idx = np.inf, -np.inf
    for path in paths:
        t0 = time.time()
        rows = 0
        f_loans, f_hits = [], []
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            rows += len(chunk)
            loan = chunk[COL_LOAN].str.strip()
            status = chunk[COL_DQ].str.strip()
            f_loans.append(loan.unique())

            idx = yyyymm_to_index(chunk[COL_PERIOD].str.strip())
            if idx.notna().any():
                min_idx = min(min_idx, float(idx.min()))
                max_idx = max(max_idx, float(idx.max()))

            vc = status.value_counts()
            for code in NEIGHBOUR_CODES:
                if code in vc.index:
                    status_counts[code] = status_counts.get(code, 0) + int(vc[code])

            m = status == TARGET
            if m.any():
                f_hits.append(loan[m].unique())
        total_rows += rows
        a = pd.unique(np.concatenate(f_loans)) if f_loans else np.array([])
        h = pd.unique(np.concatenate(f_hits)) if f_hits else np.array([])
        loan_parts.append(a)
        hit_parts.append(h)
        emit("    %-30s rows %13s  loans %11s  99-loans %8s  (%.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(a)), fmt(len(h)),
                time.time() - t0))
    all_loans = pd.unique(np.concatenate(loan_parts)) if loan_parts else np.array([])
    hit_loans = pd.unique(np.concatenate(hit_parts)) if hit_parts else np.array([])
    return (status_counts, len(all_loans), hit_loans, total_rows,
            min_idx, max_idx)


# --------------------------------------------------------------------------
# pass 2 - full row history of the 99-reporting loans
# --------------------------------------------------------------------------
def pass2(paths, hit_set):
    parts = []
    for path in paths:
        t0 = time.time()
        kept = 0
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(hit_set)
            if not keep.any():
                continue
            kept += int(keep.sum())
            parts.append(pd.DataFrame({
                "loan": loan[keep].to_numpy(),
                "idx": yyyymm_to_index(
                    chunk.loc[keep, COL_PERIOD].str.strip()).to_numpy(),
                "status": chunk.loc[keep, COL_DQ].str.strip().to_numpy(),
                "code": chunk.loc[keep, COL_ZBC].str.strip().to_numpy(),
            }))
        emit("    %-30s rows kept %11s   (%.1fs)"
             % (os.path.basename(path), fmt(kept), time.time() - t0))
    if not parts:
        return pd.DataFrame(columns=["loan", "idx", "status", "code"])
    hist = pd.concat(parts, ignore_index=True, copy=False)
    hist = hist.dropna(subset=["idx"])
    hist["idx"] = hist["idx"].astype(np.int64)
    return hist


def neighbour_table(merged, col, n99, missing_label, gap_label,
                    n_missing, n_gap, title, note):
    """Frequency table of the adjacent month's status, with gap/edge broken out."""
    emit(title)
    rule()
    emit(note)
    rule()
    emit("%-22s %14s %12s %12s" % ("status", "loan-months", "share", "cumulative"))
    rule()
    vc = merged[col].value_counts()
    cum = 0
    for s, c in vc.items():
        cum += int(c)
        shown = s if s != "" else "(blank)"
        emit("%-22s %14s %12s %12s"
             % (shown, fmt(int(c)), pct(int(c), n99), pct(cum, n99)))
    cum += n_gap
    emit("%-22s %14s %12s %12s"
         % (gap_label, fmt(n_gap), pct(n_gap, n99), pct(cum, n99)))
    cum += n_missing
    emit("%-22s %14s %12s %12s"
         % (missing_label, fmt(n_missing), pct(n_missing, n99), pct(cum, n99)))
    rule()
    emit("%-22s %14s %12s" % ("TOTAL 99 loan-months", fmt(n99), pct(n99, n99)))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("06_status_99 - is delinquency status 99 a cap, a sentinel, or terminal?")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : %d loan, %d period, %d dq_status, %d zero_balance_code"
         % (COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC))
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/2 - prevalence and the set of loans ever reporting 99")
    rule()
    status_counts, n_all, hit_loans, total_rows, min_idx, max_idx = pass1(paths)
    n_hits = len(hit_loans)
    hit_set = set(hit_loans.tolist())

    if not n_hits:
        emit("no loan reports status 99 - nothing further to report")
        write_out()
        return

    emit()
    emit("PASS 2/2 - full row history of those %s loans" % fmt(n_hits))
    rule()
    hist = pass2(paths, hit_set)

    # ---------------- 1. prevalence ----------------
    n99 = int(status_counts.get(TARGET, 0))
    emit()
    rule("=")
    emit("1. PREVALENCE OF STATUS 99")
    rule("=")
    emit("%-52s %16s %10s" % ("quantity", "count", "share"))
    rule()
    emit("%-52s %16s %10s" % ("loan-months at status 99      [loan-months]",
                              fmt(n99), pct(n99, total_rows)))
    emit("%-52s %16s" % ("all panel loan-months         [loan-months]",
                         fmt(total_rows)))
    emit("%-52s %16s %10s" % ("loans ever reporting 99             [loans]",
                              fmt(n_hits), pct(n_hits, n_all)))
    emit("%-52s %16s" % ("all distinct loans in panel         [loans]", fmt(n_all)))
    emit("%-52s %16s" % ("rows gathered in pass 2       [loan-months]", fmt(len(hist))))
    emit()
    emit("dataset reporting period: %s to %s"
         % (index_to_yyyymm(min_idx), index_to_yyyymm(max_idx)))

    emit()
    emit("STATUS 90-99 SIDE BY SIDE   (unit: loan-months)")
    rule()
    emit("%-10s %16s %14s" % ("status", "loan-months", "vs previous"))
    rule()
    prev_c = None
    for code in NEIGHBOUR_CODES:
        c = int(status_counts.get(code, 0))
        ratio = "-" if not prev_c else "x%.2f" % (c / float(prev_c))
        emit("%-10s %16s %14s" % (code, fmt(c), ratio))
        prev_c = c if c else prev_c
    emit()
    emit("A smooth continuation of the decay would keep the ratio near the values")
    emit("seen across 90-98; a spike shows as a large jump in the final row.")

    # ---------------- adjacency ----------------
    loan_min = hist.groupby("loan")["idx"].min()
    loan_max = hist.groupby("loan")["idx"].max()

    h99 = hist[hist["status"] == TARGET][["loan", "idx"]].copy()
    n99_hist = len(h99)

    prev_map = hist[["loan", "idx", "status"]].rename(columns={"status": "prev"})
    prev_map = prev_map.assign(idx=prev_map["idx"] + 1)
    back = h99.merge(prev_map, on=["loan", "idx"], how="left")

    next_map = hist[["loan", "idx", "status"]].rename(columns={"status": "next"})
    next_map = next_map.assign(idx=next_map["idx"] - 1)
    fwd = h99.merge(next_map, on=["loan", "idx"], how="left")

    back_missing = back["prev"].isna()
    is_first = back["idx"] == back["loan"].map(loan_min)
    n_no_prior = int((back_missing & is_first).sum())
    n_gap_back = int((back_missing & ~is_first).sum())
    back_ok = back[~back_missing]

    fwd_missing = fwd["next"].isna()
    is_last = fwd["idx"] == fwd["loan"].map(loan_max)
    n_no_next = int((fwd_missing & is_last).sum())
    n_gap_fwd = int((fwd_missing & ~is_last).sum())
    fwd_ok = fwd[~fwd_missing]

    # ---------------- 2. predecessors ----------------
    emit()
    rule("=")
    emit("2. PREDECESSOR - STATUS IN THE IMMEDIATELY PRECEDING CALENDAR MONTH")
    rule("=")
    neighbour_table(
        back_ok, "prev", n99_hist,
        "no prior row (99 is loan's first row)", "gap (no row at month - 1)",
        n_no_prior, n_gap_back,
        "Every 99 loan-month, by what preceded it   (unit: loan-months)",
        "A non-adjacent earlier row is NEVER used as the predecessor.")

    n_adj_back = len(back_ok)
    n_from_9899 = int(back_ok["prev"].isin(["98", "99"]).sum())
    n_from_cur = int((back_ok["prev"] == CURRENT).sum())
    emit()
    emit("HEADLINE  (denominator: the %s 99 loan-months WITH an adjacent predecessor)"
         % fmt(n_adj_back))
    rule()
    emit("%-52s %14s %10s" % ("arriving from 98 or 99",
                              fmt(n_from_9899), pct(n_from_9899, n_adj_back)))
    emit("%-52s %14s %10s" % ("arriving from anything else",
                              fmt(n_adj_back - n_from_9899),
                              pct(n_adj_back - n_from_9899, n_adj_back)))
    emit("%-52s %14s %10s" % ("arriving from 00 (Current)",
                              fmt(n_from_cur), pct(n_from_cur, n_adj_back)))
    emit()
    emit("Under the CAP reading, 'arriving from 98 or 99' should be ~100%: a loan")
    emit("cannot reach 99 months of arrears except through 98.")

    # ---------------- 3. successors ----------------
    emit()
    rule("=")
    emit("3. SUCCESSOR - STATUS IN THE IMMEDIATELY FOLLOWING CALENDAR MONTH")
    rule("=")
    neighbour_table(
        fwd_ok, "next", n99_hist,
        "no next row (99 is loan's final row)", "gap (no row at month + 1)",
        n_no_next, n_gap_fwd,
        "Every 99 loan-month, by what followed it   (unit: loan-months)",
        "A non-adjacent later row is NEVER used as the successor.")

    n_adj_fwd = len(fwd_ok)
    n_to_cur = int((fwd_ok["next"] == CURRENT).sum())
    n_to_9899 = int(fwd_ok["next"].isin(["98", "99"]).sum())
    emit()
    emit("HEADLINE  (denominator: the %s 99 loan-months WITH an adjacent successor)"
         % fmt(n_adj_fwd))
    rule()
    emit("%-52s %14s %10s" % ("returning to 00 (Current)",
                              fmt(n_to_cur), pct(n_to_cur, n_adj_fwd)))
    emit("%-52s %14s %10s" % ("going to 98 or 99",
                              fmt(n_to_9899), pct(n_to_9899, n_adj_fwd)))
    emit()
    emit("Under the CAP reading a return to 00 in the next month is impossible;")
    emit("under the SENTINEL reading it is expected whenever reporting resumes.")

    # ---------------- 4. run lengths ----------------
    h99s = h99.sort_values(["loan", "idx"], kind="mergesort").reset_index(drop=True)
    seq = h99s.groupby("loan").cumcount()
    h99s["run_key"] = h99s["idx"] - seq
    runs = h99s.groupby(["loan", "run_key"]).size()
    emit()
    rule("=")
    emit("4. CONSECUTIVE-99 RUN LENGTHS")
    rule("=")
    emit("A run is a maximal block of consecutive calendar months at status 99")
    emit("within one loan. Contiguity is on the month index, so a reporting gap")
    emit("ends a run.")
    rule()
    emit("%-52s %14s" % ("runs identified                        [runs]", fmt(len(runs))))
    emit("%-52s %14s" % ("loans contributing them               [loans]",
                         fmt(h99s['loan'].nunique())))
    emit("%-52s %14s" % ("99 observations total           [loan-months]", fmt(n99_hist)))
    emit()
    distribution(runs.to_numpy(), "months")
    emit()
    emit("RUN LENGTH FREQUENCY   (unit: runs)")
    rule()
    emit("%-14s %14s %12s %12s" % ("length", "runs", "share", "cumulative"))
    rule()
    rl = runs.value_counts().sort_index()
    cum = 0
    for length, c in rl.items():
        if length > 24:
            break
        cum += int(c)
        emit("%-14d %14s %12s %12s"
             % (length, fmt(int(c)), pct(int(c), len(runs)), pct(cum, len(runs))))
    tail = int((runs > 24).sum())
    cum += tail
    emit("%-14s %14s %12s %12s" % ("25+", fmt(tail), pct(tail, len(runs)),
                                   pct(cum, len(runs))))
    emit()
    emit("Short runs point to reporting outages (SENTINEL); long unbroken runs")
    emit("point to a censored ladder value (CAP).")

    # ---------------- 5. terminality ----------------
    final = hist.loc[hist.groupby("loan")["idx"].idxmax()].set_index("loan")
    ends99 = final["status"] == TARGET
    n_ends99 = int(ends99.sum())
    ra_loans = set(hist.loc[hist["status"] == "RA", "loan"].unique().tolist())

    emit()
    rule("=")
    emit("5. TERMINALITY AND EXIT CODES   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("loans ever reporting 99", fmt(n_hits), pct(n_hits, n_hits)))
    emit("%-52s %14s %10s" % ("of these, 99 is the FINAL observed row",
                              fmt(n_ends99), pct(n_ends99, n_hits)))
    emit("%-52s %14s %10s" % ("of these, 99 is not the final row",
                              fmt(n_hits - n_ends99), pct(n_hits - n_ends99, n_hits)))
    emit()
    emit("Under the TERMINAL reading the first figure would be ~100%.")

    fin = final[ends99]
    emit()
    emit("ZERO_BALANCE_CODE ON THAT FINAL 99 ROW   (unit: loans, n = %s)" % fmt(n_ends99))
    rule()
    emit("%-46s %12s %10s %10s" % ("outcome", "loans", "share", "RA ever"))
    rule()
    groups = [
        ("01        prepaid / matured", PREPAID),
        ("02/03/09  credit events", CREDIT_EVENT),
        ("06/15/16  administrative exits", ADMIN_EXIT),
    ]
    seen = set()
    for label, codes in groups:
        sel = fin["code"].isin(codes)
        n = int(sel.sum())
        ra = int(fin.index[sel].isin(ra_loans).sum())
        seen.update(codes)
        emit("%-46s %12s %10s %10s"
             % (label, fmt(n), pct(n, n_ends99), pct(ra, n) if n else "n/a"))
    other = fin[(fin["code"] != "") & (~fin["code"].isin(seen))]
    if len(other):
        for code, n in other["code"].value_counts().items():
            ra = int(other.index[other["code"] == code].isin(ra_loans).sum())
            emit("%-46s %12s %10s %10s"
                 % ("%s        other / unclassified" % code, fmt(int(n)),
                    pct(int(n), n_ends99), pct(ra, int(n))))
    nocode = fin[fin["code"] == ""]
    n_nocode = len(nocode)
    ra_nc = int(nocode.index.isin(ra_loans).sum())
    emit("%-46s %12s %10s %10s"
         % ("(none)    still open when the dataset ends / censored",
            fmt(n_nocode), pct(n_nocode, n_ends99),
            pct(ra_nc, n_nocode) if n_nocode else "n/a"))
    rule()
    emit("%-46s %12s %10s" % ("TOTAL", fmt(n_ends99), pct(n_ends99, n_ends99)))
    emit()
    emit("The no-code group is reported separately and is NOT merged into any")
    emit("other category. 'RA ever' is the share of that row's loans showing RA")
    emit("anywhere in their history.")
    emit()
    emit("A material share terminating with 01 would contradict the CAP reading:")
    emit("a voluntary prepayment cannot follow eight years of arrears.")

    if n_nocode:
        emit()
        emit("FINAL-ROW MONTH OF THE NO-CODE (CENSORED) GROUP   (unit: loans)")
        rule()
        emit("dataset reporting period: %s to %s"
             % (index_to_yyyymm(min_idx), index_to_yyyymm(max_idx)))
        rule()
        emit("%-14s %14s %12s %12s" % ("month", "loans", "share", "cumulative"))
        rule()
        vc = nocode["idx"].value_counts().sort_index(ascending=False)
        cum = 0
        for i, (mi, c) in enumerate(vc.items()):
            if i >= 15:
                break
            cum += int(c)
            tag = index_to_yyyymm(mi)
            if mi == max_idx:
                tag += " *"
            emit("%-14s %14s %12s %12s"
                 % (tag, fmt(int(c)), pct(int(c), n_nocode), pct(cum, n_nocode)))
        if len(vc) > 15:
            rest = int(vc.iloc[15:].sum())
            emit("%-14s %14s %12s %12s"
                 % ("other months", fmt(rest), pct(rest, n_nocode), "100.0000%"))
        rule()
        at_end = int((nocode["idx"] == max_idx).sum())
        emit("%-52s %14s %10s" % ("final row IS the dataset's last month",
                                  fmt(at_end), pct(at_end, n_nocode)))
        emit("%-52s %14s %10s" % ("final row is an EARLIER month",
                                  fmt(n_nocode - at_end),
                                  pct(n_nocode - at_end, n_nocode)))
        emit("%-52s %14s" % ("distinct final months represented", fmt(len(vc))))
        emit("('*' marks the dataset's last reporting month.)")
        emit()
        emit("If censoring explains this group, its loans cluster on the dataset's")
        emit("final month. Scatter across many months means something else ends")
        emit("these records.")

    # ---------------- 6. non-numeric neighbours ----------------
    emit()
    rule("=")
    emit("6. NON-NUMERIC NEIGHBOURS OF A 99 ROW   (unit: loan-months)")
    rule("=")
    emit("%-40s %14s %12s %12s"
         % ("", "before a 99", "share", "after a 99"))
    rule()
    for code in NON_NUMERIC:
        b = int((back_ok["prev"] == code).sum())
        f = int((fwd_ok["next"] == code).sum())
        emit("%-40s %14s %12s %12s"
             % ("status %s" % code, fmt(b), pct(b, n_adj_back), fmt(f)))
    b_any = int(back_ok["prev"].isin(NON_NUMERIC).sum())
    f_any = int(fwd_ok["next"].isin(NON_NUMERIC).sum())
    rule()
    emit("%-40s %14s %12s %12s"
         % ("either RA or XX", fmt(b_any), pct(b_any, n_adj_back), fmt(f_any)))
    emit("%-40s %14s %12s %12s"
         % ("adjacent rows examined", fmt(n_adj_back), "", fmt(n_adj_fwd)))
    emit()
    emit("Shares are of the adjacent-predecessor denominator; the 'after' column")
    emit("has its own denominator of %s." % fmt(n_adj_fwd))
    present = sorted(set(hist["status"].unique()) & set(NON_NUMERIC))
    emit()
    emit("non-numeric codes present anywhere in these loans' history: %s"
         % (", ".join(present) if present else "none"))

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
