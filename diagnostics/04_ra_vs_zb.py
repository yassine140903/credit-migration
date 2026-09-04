"""Overlap between the "RA" delinquency-status marker and credit-event zero-balance codes.

"RA" in current_loan_delinquency_status (column 3) is the REO Acquisition
indicator: foreclosure has completed and the property is held pending sale.
Zero-balance codes 02/03/09 (column 8) are credit-event terminations, recorded
once, on the loan's final row. The two are different markers of the same
underlying process, written to different fields at different times. This script
measures how much they overlap. It reports counts only - it does not classify,
relabel, or recommend a Default definition.

UNITS. Two quantities appear throughout and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation
Every table labels which one it is in.

FOUR PASSES. A loan's rows are not assumed sorted or contiguous and may straddle
chunk and file boundaries, so no per-loan state is inferred from row adjacency.
Every reduction uses order-independent operators (min / max / sum / set union):
    pass 1  [0,3]      loan universe, the ever-RA loan set, RA loan-months
    pass 2  [0,1,3,8]  terminal zero-balance code + its month for every loan;
                       first RA month for the ever-RA set
    pass 3  [0,1,3]    restricted to ever-RA: numeric DQ>=6 strictly BEFORE the
                       first RA month, and any status strictly AFTER it
Pass 3 cannot be folded into pass 2: "before" and "after" are only defined once
the first RA month is known for the whole panel, and a chunk read early may hold
rows that precede a first RA month discovered in a chunk read later.

Numeric casts of column 3 are guarded by .str.isdigit() throughout, so "RA" and
any other non-numeric code are never coerced.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "04_ra_vs_zb.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_ZBC = 8

RA = "RA"
CURRENT = "00"
DQ_DEEP = 6

# Cross-check from 03_dq_deep_state on the same 8-file glob. A mismatch means
# one of the two scripts is wrong; it is reported as a bug, not absorbed.
EXPECTED_RA_LOAN_MONTHS = 969_765

ZBC_LABELS = {
    "01": "prepaid or matured (voluntary payoff)",
    "02": "third-party sale",
    "03": "short sale or charge-off",
    "06": "repurchase prior to property disposition",
    "09": "REO disposition",
    "15": "note sale",
    "16": "reperforming loan sale",
}

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
    """YYYYMM -> year*12 + month, so differences are true month counts."""
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
# pass 1 - loan universe, ever-RA set, RA loan-months
# --------------------------------------------------------------------------
def pass1(paths):
    all_parts, ra_parts = [], []
    ra_months = 0
    total_rows = 0
    for path in paths:
        t0 = time.time()
        rows = 0
        f_all, f_ra = [], []
        for chunk in reader_for(path, [COL_LOAN, COL_DQ]):
            rows += len(chunk)
            loan = chunk[COL_LOAN].str.strip()
            status = chunk[COL_DQ].str.strip()
            f_all.append(loan.unique())
            m = status == RA
            n = int(m.sum())
            if n:
                ra_months += n
                f_ra.append(loan[m].unique())
        total_rows += rows
        a = pd.unique(np.concatenate(f_all)) if f_all else np.array([])
        r = pd.unique(np.concatenate(f_ra)) if f_ra else np.array([])
        all_parts.append(a)
        ra_parts.append(r)
        emit("    %-30s rows %13s  loans %11s  ever-RA %10s  (%.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(a)), fmt(len(r)),
                time.time() - t0))
    all_loans = pd.unique(np.concatenate(all_parts)) if all_parts else np.array([])
    ra_loans = pd.unique(np.concatenate(ra_parts)) if ra_parts else np.array([])
    return all_loans, ra_loans, ra_months, total_rows


# --------------------------------------------------------------------------
# pass 2 - terminal zero-balance code (all loans) and first RA month (RA set)
# --------------------------------------------------------------------------
def pass2(paths, ra_set):
    term_parts, ra_first_parts = [], []
    for path in paths:
        t0 = time.time()
        n_term = 0
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ, COL_ZBC]):
            loan = chunk[COL_LOAN].str.strip()
            idx = yyyymm_to_index(chunk[COL_PERIOD].str.strip())

            code = chunk[COL_ZBC].str.strip()
            has = code != ""
            if has.any():
                n_term += int(has.sum())
                term_parts.append(pd.DataFrame({
                    "loan": loan[has].to_numpy(),
                    "code": code[has].to_numpy(),
                    "term_idx": idx[has].to_numpy(),
                }))

            is_ra = (chunk[COL_DQ].str.strip() == RA) & loan.isin(ra_set)
            if is_ra.any():
                df = pd.DataFrame({
                    "loan": loan[is_ra].to_numpy(),
                    "ra_idx": idx[is_ra].to_numpy(),
                })
                ra_first_parts.append(df.groupby("loan", sort=False)["ra_idx"].min())
        emit("    %-30s zero-balance rows %10s   (%.1fs)"
             % (os.path.basename(path), fmt(n_term), time.time() - t0))

    term = (pd.concat(term_parts, ignore_index=True, copy=False)
            if term_parts else pd.DataFrame(columns=["loan", "code", "term_idx"]))
    ra_first = (pd.concat(ra_first_parts).groupby(level=0).min()
                if ra_first_parts else pd.Series(dtype="float64"))
    return term, ra_first


# --------------------------------------------------------------------------
# pass 3 - pre-RA deep delinquency and post-RA statuses, restricted to ever-RA
# --------------------------------------------------------------------------
def pass3(paths, ra_set, ra_first):
    pre_deep, post_numeric, post_current = set(), set(), set()
    post_months = {}
    post_pair_parts = []
    for path in paths:
        t0 = time.time()
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(ra_set)
            if not keep.any():
                continue
            loan = loan[keep]
            status = chunk.loc[keep, COL_DQ].str.strip()
            idx = yyyymm_to_index(chunk.loc[keep, COL_PERIOD].str.strip())
            first = loan.map(ra_first)

            digit = status.str.isdigit()
            num = pd.to_numeric(status.where(digit), errors="coerce")

            before = idx < first
            m = before & digit & (num >= DQ_DEEP)
            if m.any():
                pre_deep.update(loan[m].unique().tolist())

            after = idx > first
            if not after.any():
                continue

            m = after & digit
            if m.any():
                post_numeric.update(loan[m].unique().tolist())
            m = after & (status == CURRENT)
            if m.any():
                post_current.update(loan[m].unique().tolist())

            sa = status[after]
            for code, cnt in sa.value_counts().items():
                post_months[code] = post_months.get(code, 0) + int(cnt)
            post_pair_parts.append(
                pd.DataFrame({"loan": loan[after].to_numpy(),
                              "status": sa.to_numpy()}).drop_duplicates()
            )
        emit("    %-30s scanned   (%.1fs)" % (os.path.basename(path), time.time() - t0))

    post_pairs = (pd.concat(post_pair_parts, ignore_index=True, copy=False)
                  .drop_duplicates() if post_pair_parts
                  else pd.DataFrame(columns=["loan", "status"]))
    return pre_deep, post_numeric, post_current, post_months, post_pairs


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("04_ra_vs_zb - overlap of the RA status marker and credit-event zero-balance codes")
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
    emit("PASS 1/3 - loan universe and ever-RA set")
    rule()
    all_loans, ra_loans, ra_months, total_rows = pass1(paths)
    n_all, n_ra = len(all_loans), len(ra_loans)
    ra_set = set(ra_loans.tolist())

    emit()
    emit("PASS 2/3 - terminal zero-balance code and first RA month")
    rule()
    term, ra_first = pass2(paths, ra_set)

    emit()
    emit("PASS 3/3 - pre-RA deep delinquency and post-RA statuses")
    rule()
    pre_deep, post_numeric, post_current, post_months, post_pairs = pass3(
        paths, ra_set, ra_first)

    # ---------------- (a) ----------------
    emit()
    rule("=")
    emit("(a) THE RA MARKER")
    rule("=")
    emit("%-52s %16s %10s" % ("quantity", "count", "share"))
    rule()
    emit("%-52s %16s %10s" % ("all distinct loans in panel   [loans]",
                              fmt(n_all), pct(n_all, n_all)))
    emit("%-52s %16s %10s" % ("loans ever showing 'RA'       [loans]",
                              fmt(n_ra), pct(n_ra, n_all)))
    emit("%-52s %16s" % ("RA observations         [loan-months]", fmt(ra_months)))
    emit("%-52s %16s" % ("total panel rows        [loan-months]", fmt(total_rows)))
    if n_ra:
        emit("%-52s %16.3f" % ("mean RA months per ever-RA loan", ra_months / float(n_ra)))
    emit()
    emit("cross-check against 03_dq_deep_state (same glob)")
    rule()
    emit("%-52s %16s" % ("expected RA loan-months", fmt(EXPECTED_RA_LOAN_MONTHS)))
    emit("%-52s %16s" % ("observed RA loan-months", fmt(ra_months)))
    if ra_months == EXPECTED_RA_LOAN_MONTHS:
        emit("%-52s %16s" % ("status", "MATCH"))
    else:
        d = ra_months - EXPECTED_RA_LOAN_MONTHS
        emit("%-52s %16s" % ("status", "*** MISMATCH ***"))
        emit("%-52s %16s" % ("difference (observed - expected)", "%+d" % d))
        emit("This is a BUG in one of the two scripts, not a data finding.")

    # ---------------- (b) ----------------
    dup_term = int(term.groupby("loan").size().gt(1).sum()) if len(term) else 0
    term_code = term.groupby("loan")["code"].first()
    term_idx = term.groupby("loan")["term_idx"].first()

    universe = pd.DataFrame({"loan": all_loans})
    universe["code"] = universe["loan"].map(term_code)
    universe["ever_ra"] = universe["loan"].isin(ra_set)

    emit()
    rule("=")
    emit("(b) EVER-RA x TERMINAL ZERO_BALANCE_CODE   (unit: loans, codes raw/ungrouped)")
    rule("=")
    emit("%-6s %-42s %11s %11s %11s"
         % ("code", "label", "ever-RA", "never-RA", "total"))
    rule()
    codes = sorted([c for c in universe["code"].dropna().unique()])
    tot_ra = tot_nra = 0
    for c in codes:
        sel = universe["code"] == c
        a = int((sel & universe["ever_ra"]).sum())
        b = int((sel & ~universe["ever_ra"]).sum())
        tot_ra += a
        tot_nra += b
        emit("%-6s %-42s %11s %11s %11s"
             % (c, ZBC_LABELS.get(c, "not in the documented code list"),
                fmt(a), fmt(b), fmt(a + b)))
    sel = universe["code"].isna()
    a = int((sel & universe["ever_ra"]).sum())
    b = int((sel & ~universe["ever_ra"]).sum())
    emit("%-6s %-42s %11s %11s %11s"
         % ("-", "never terminated / censored at cutoff", fmt(a), fmt(b), fmt(a + b)))
    rule()
    emit("%-6s %-42s %11s %11s %11s"
         % ("", "TOTAL", fmt(tot_ra + a), fmt(tot_nra + b), fmt(n_all)))
    if dup_term:
        emit()
        emit("NOTE: %s loan(s) carry more than one zero-balance row; the first "
             "observed code is used." % fmt(dup_term))

    # ---------------- (c) ----------------
    ra_u = universe[universe["ever_ra"]]
    n_09 = int((ra_u["code"] == "09").sum())
    n_0203 = int(ra_u["code"].isin(["02", "03"]).sum())
    n_01 = int((ra_u["code"] == "01").sum())
    n_admin = int(ra_u["code"].isin(["15", "16", "96"]).sum())
    n_open = int(ra_u["code"].isna().sum())
    ce = universe["code"].isin(["02", "03", "09"])
    n_ce_no_ra = int((ce & ~universe["ever_ra"]).sum())
    n_ce_all = int(ce.sum())

    emit()
    rule("=")
    emit("(c) THE OVERLAP, STATED   (unit: loans)")
    rule("=")
    emit("%-54s %12s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-54s %12s %10s" % ("ever-RA, terminate 09 (REO disposition)",
                              fmt(n_09), pct(n_09, n_ra)))
    emit("%-54s %12s %10s" % ("ever-RA, terminate 02 or 03",
                              fmt(n_0203), pct(n_0203, n_ra)))
    emit("%-54s %12s %10s" % ("ever-RA, terminate 01 (prepaid/matured)",
                              fmt(n_01), pct(n_01, n_ra)))
    emit("%-54s %12s %10s" % ("ever-RA, terminate 15/16/96",
                              fmt(n_admin), pct(n_admin, n_ra)))
    emit("%-54s %12s %10s" % ("ever-RA, never terminated",
                              fmt(n_open), pct(n_open, n_ra)))
    emit("(shares above are of the %s ever-RA loans)" % fmt(n_ra))
    rule()
    emit("%-54s %12s %10s" % ("terminate 02/03/09 but NEVER show RA",
                              fmt(n_ce_no_ra), pct(n_ce_no_ra, n_ce_all)))
    emit("(share is of the %s loans terminating 02/03/09)" % fmt(n_ce_all))
    if n_01:
        emit()
        emit("*** CALLED OUT: %s ever-RA loan(s) terminate with code 01, prepaid or"
             % fmt(n_01))
        emit("    matured. A loan whose foreclosure completed and whose property was")
        emit("    taken into REO should not close as a voluntary payoff. This is")
        emit("    reported, not smoothed over; the count stands as measured.")

    # ---------------- (d) ----------------
    ra09 = ra_u.loc[ra_u["code"] == "09", "loan"]
    hold = (ra09.map(term_idx) - ra09.map(ra_first)).dropna()
    emit()
    rule("=")
    emit("(d) REO HOLDING PERIOD: FIRST RA MONTH -> TERMINATION MONTH (code 09)")
    rule("=")
    emit("Population: the %s ever-RA loans terminating 09.   (unit: months)" % fmt(n_09))
    rule()
    distribution(hold.to_numpy(), "months")
    if len(hold):
        n_neg = int((hold < 0).sum())
        n_zero = int((hold == 0).sum())
        emit()
        emit("%-54s %12s" % ("holding period == 0 months (RA in final month)", fmt(n_zero)))
        emit("%-54s %12s" % ("holding period <  0 months (RA after zero-balance)",
                             fmt(n_neg)))
        if n_neg:
            emit("A negative holding period means the RA row is dated after the")
            emit("zero-balance row. Reported as measured.")

    # ---------------- (e) ----------------
    n_pre = len(pre_deep)
    emit()
    rule("=")
    emit("(e) NUMERIC DQ >= %d STRICTLY BEFORE THE FIRST RA MONTH   (unit: loans)" % DQ_DEEP)
    rule("=")
    emit("%-54s %12s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-54s %12s %10s" % ("ever-RA loans", fmt(n_ra), pct(n_ra, n_ra)))
    emit("%-54s %12s %10s" % ("of these, numeric DQ>=%d before first RA" % DQ_DEEP,
                              fmt(n_pre), pct(n_pre, n_ra)))
    emit("%-54s %12s %10s" % ("of these, reach RA with NO prior numeric DQ>=%d" % DQ_DEEP,
                              fmt(n_ra - n_pre), pct(n_ra - n_pre, n_ra)))
    emit()
    emit("The second figure is the state-labelling edge case: a loan entering RA")
    emit("without a numeric deep-delinquency observation preceding it.")

    # ---------------- (f) ----------------
    n_post_num = len(post_numeric)
    n_post_cur = len(post_current)
    emit()
    rule("=")
    emit("(f) MOVEMENT OUT OF RA: STATUSES STRICTLY AFTER THE FIRST RA MONTH")
    rule("=")
    emit("%-54s %12s %10s" % ("quantity  (unit: loans)", "loans", "share"))
    rule()
    emit("%-54s %12s %10s" % ("ever-RA loans", fmt(n_ra), pct(n_ra, n_ra)))
    emit("%-54s %12s %10s" % ("return to ANY numeric status after first RA",
                              fmt(n_post_num), pct(n_post_num, n_ra)))
    emit("%-54s %12s %10s" % ("return to '00' (Current) after first RA",
                              fmt(n_post_cur), pct(n_post_cur, n_ra)))

    if post_months:
        loans_by_status = post_pairs["status"].value_counts()
        emit()
        emit("DISTRIBUTION OF POST-RA STATUS VALUES")
        rule()
        emit("%-10s %-30s %16s %12s"
             % ("status", "kind", "loan-months", "loans"))
        rule()
        for code in sorted(post_months, key=lambda k: -post_months[k]):
            if code == RA:
                kind = "RA (still in REO)"
            elif code == "":
                kind = "blank"
            elif code.isdigit():
                kind = "numeric, current" if code == CURRENT else "numeric, delinquent"
            else:
                kind = "non-numeric"
            emit("%-10s %-30s %16s %12s"
                 % (code if code else "(blank)", kind, fmt(post_months[code]),
                    fmt(int(loans_by_status.get(code, 0)))))
        rule()
        emit("%-10s %-30s %16s"
             % ("TOTAL", "", fmt(sum(post_months.values()))))
        emit()
        emit("'loan-months' sums observations; 'loans' counts distinct loans showing")
        emit("that status at least once after first RA. Loans appear under every")
        emit("status they show, so the loans column does not sum to the loan total.")
    emit()
    emit("A non-zero return-to-numeric count means RA is not absorbing in the data.")

    # ---------------- (g) ----------------
    sel96 = universe["code"] == "96"
    n96 = int(sel96.sum())
    n96_ra = int((sel96 & universe["ever_ra"]).sum())
    emit()
    rule("=")
    emit("(g) ZERO_BALANCE_CODE 96 AND RA   (unit: loans)")
    rule("=")
    emit("%-54s %12s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-54s %12s %10s" % ("loans terminating with code 96", fmt(n96), pct(n96, n96)))
    emit("%-54s %12s %10s" % ("of these, ever show RA", fmt(n96_ra), pct(n96_ra, n96)))
    emit("%-54s %12s %10s" % ("of these, never show RA",
                              fmt(n96 - n96_ra), pct(n96 - n96_ra, n96)))

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
