"""Deep-delinquency dynamics: loans that ever reach current_loan_delinquency_status >= 6.

Reports, for that subpopulation:
  (a) count and share of all loans
  (b) first-touch month distribution, aggregated by calendar year
  (c) how many ever report status "00" (Current) at a LATER month - cure count/rate
  (d) for cured loans, months from first DQ>=6 to first subsequent "00":
      deciles, median, mean. The SHAPE is the point: a sharp spike near a single
      value points at administrative re-aging, a long right tail at genuine workout.
      Plus modification_flag = "Y" incidence.
  (e) how many terminate with a credit event (02/03/09) - eventual default
      probability conditional on ever reaching DQ>=6
  (f) dwell time: total loan-MONTHS observed at status >= 6, and the implied
      per-month hazard of moving to a credit event.

UNITS. Two different quantities appear throughout and are never mixed:
    loans        - one mortgage, counted once however long it is observed
    loan-months  - one (loan, month) observation
Every table below labels which one it is in.

FOUR PASSES. A loan's rows are not assumed sorted or contiguous and may straddle
chunk and file boundaries, so no per-loan state is inferred from row adjacency.
Each pass reduces with order-independent operators (min / max / sum / any):
    pass 1  [0,3]        loan universe, and the set of loans ever at DQ>=6
    pass 2  [0,1,3,7,8]  restricted to that set: first DQ>=6 month, deep-delinquent
                         loan-months, modification flags, zero-balance code
    pass 3  [0,1,3]      restricted: first "00" strictly after the first DQ>=6 month
                         (needs pass 2's first-touch month, hence a separate pass)
Cure timing cannot be folded into pass 2: "later than first touch" is only defined
once the first touch is known for the whole panel, and an earlier chunk may hold a
"00" that precedes a first touch discovered in a later chunk.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "03_dq_deep_state.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_MOD = 7
COL_ZBC = 8

DQ_DEEP = 6
CURRENT = "00"
CREDIT_EVENT = ["02", "03", "09"]
PREPAID = ["01"]
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


def deep_mask(status):
    """status >= DQ_DEEP, with every numeric cast guarded by .str.isdigit().

    Non-numeric codes ("RA", "XX", "R", blank) are never coerced and never
    counted as deep delinquency.
    """
    digit = status.str.isdigit()
    num = pd.to_numeric(status.where(digit), errors="coerce")
    return digit & (num >= DQ_DEEP)


def reader_for(path, cols):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=cols,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


# --------------------------------------------------------------------------
# pass 1 - loan universe and the DQ>=6 loan set
# --------------------------------------------------------------------------
def pass1(paths):
    all_parts, deep_parts = [], []
    nonnumeric = {}
    total_rows = 0
    for path in paths:
        t0 = time.time()
        rows = 0
        file_all, file_deep = [], []
        for chunk in reader_for(path, [COL_LOAN, COL_DQ]):
            rows += len(chunk)
            loan = chunk[COL_LOAN].str.strip()
            status = chunk[COL_DQ].str.strip()
            file_all.append(loan.unique())

            nn = status[~status.str.isdigit()]
            if len(nn):
                for code, cnt in nn.value_counts().items():
                    nonnumeric[code] = nonnumeric.get(code, 0) + int(cnt)

            m = deep_mask(status)
            if m.any():
                file_deep.append(loan[m].unique())

        total_rows += rows
        a = pd.unique(np.concatenate(file_all)) if file_all else np.array([])
        d = pd.unique(np.concatenate(file_deep)) if file_deep else np.array([])
        all_parts.append(a)
        deep_parts.append(d)
        emit("    %-30s rows %13s  loans %11s  DQ>=6 loans %10s  (%.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(a)), fmt(len(d)),
                time.time() - t0))

    all_loans = pd.unique(np.concatenate(all_parts)) if all_parts else np.array([])
    deep_loans = pd.unique(np.concatenate(deep_parts)) if deep_parts else np.array([])
    return all_loans, deep_loans, total_rows, nonnumeric


# --------------------------------------------------------------------------
# pass 2 - per-loan state for the DQ>=6 set
# --------------------------------------------------------------------------
def pass2(paths, deep_set):
    agg_parts, zbc_parts = [], []
    for path in paths:
        t0 = time.time()
        kept = 0
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD, COL_ZBC]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(deep_set)
            if not keep.any():
                continue
            kept += int(keep.sum())

            loan = loan[keep]
            status = chunk.loc[keep, COL_DQ].str.strip()
            idx = yyyymm_to_index(chunk.loc[keep, COL_PERIOD].str.strip())
            mod = chunk.loc[keep, COL_MOD].str.strip()
            code = chunk.loc[keep, COL_ZBC].str.strip()

            m = deep_mask(status)
            df = pd.DataFrame({
                "loan": loan.to_numpy(),
                # month index only where the row is itself deep-delinquent, so the
                # min over the group is the first-touch month
                "deep_idx": idx.where(m).to_numpy(),
                "deep_months": m.to_numpy(),
                "mod_y": (mod == "Y").to_numpy(),
                "mod_p": (mod == "P").to_numpy(),
            })
            g = df.groupby("loan", sort=False)
            agg_parts.append(pd.DataFrame({
                "first_deep_idx": g["deep_idx"].min(),
                "deep_months": g["deep_months"].sum(),
                "mod_y": g["mod_y"].max(),
                "mod_p": g["mod_p"].max(),
            }))

            has_code = code != ""
            if has_code.any():
                zbc_parts.append(pd.DataFrame({
                    "loan": loan[has_code].to_numpy(),
                    "code": code[has_code].to_numpy(),
                }))
        emit("    %-30s rows kept %13s   (%.1fs)"
             % (os.path.basename(path), fmt(kept), time.time() - t0))

    cat = pd.concat(agg_parts, copy=False)
    g = cat.groupby(level=0, sort=False)
    state = pd.DataFrame({
        "first_deep_idx": g["first_deep_idx"].min(),
        "deep_months": g["deep_months"].sum(),
        "mod_y": g["mod_y"].max(),
        "mod_p": g["mod_p"].max(),
    })
    zbc = (pd.concat(zbc_parts, ignore_index=True, copy=False)
           if zbc_parts else pd.DataFrame(columns=["loan", "code"]))
    return state, zbc


# --------------------------------------------------------------------------
# pass 3 - first "00" strictly after each loan's first DQ>=6 month
# --------------------------------------------------------------------------
def pass3(paths, deep_set, first_deep_idx):
    parts = []
    for path in paths:
        t0 = time.time()
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ]):
            loan = chunk[COL_LOAN].str.strip()
            keep = loan.isin(deep_set) & (chunk[COL_DQ].str.strip() == CURRENT)
            if not keep.any():
                continue
            loan = loan[keep]
            idx = yyyymm_to_index(chunk.loc[keep, COL_PERIOD].str.strip())
            touch = loan.map(first_deep_idx)
            later = idx > touch
            if not later.any():
                continue
            df = pd.DataFrame({
                "loan": loan[later].to_numpy(),
                "cure_idx": idx[later].to_numpy(),
            })
            parts.append(df.groupby("loan", sort=False)["cure_idx"].min())
        emit("    %-30s scanned   (%.1fs)" % (os.path.basename(path), time.time() - t0))

    if not parts:
        return pd.Series(dtype="float64")
    return pd.concat(parts).groupby(level=0).min()


def distribution(values, unit):
    """Deciles plus p1/p99, median, mean for a 1-D numeric array."""
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
        tag = "p%-2d (d%d)" % (q, q // 10)
        if q == 50:
            tag = "p50 (median)"
        emit("%-16s %14.1f" % (tag, np.percentile(v, q)))
    emit("%-16s %14.1f" % ("p99", np.percentile(v, 99)))
    emit("%-16s %14d" % ("max", int(v.max())))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("03_dq_deep_state - loans ever reaching delinquency status >= %d" % DQ_DEEP)
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : %d loan, %d period, %d dq_status, %d modification_flag, "
         "%d zero_balance_code" % (COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD, COL_ZBC))
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/3 - loan universe and DQ>=6 loan set")
    rule()
    all_loans, deep_loans, total_rows, nonnumeric = pass1(paths)
    n_all, n_deep = len(all_loans), len(deep_loans)
    deep_set = set(deep_loans.tolist())
    del all_loans

    if not n_deep:
        emit("no loans reach DQ>=%d - nothing further to report" % DQ_DEEP)
        write_out()
        return

    emit()
    emit("PASS 2/3 - per-loan state restricted to the DQ>=6 set")
    rule()
    state, zbc = pass2(paths, deep_set)

    emit()
    emit("PASS 3/3 - first return to '00' after first touch")
    rule()
    cure_idx = pass3(paths, deep_set, state["first_deep_idx"])

    # ---------------- (a) ----------------
    emit()
    rule("=")
    emit("(a) DEEP-DELINQUENCY INCIDENCE   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-52s %14s %10s" % ("all distinct loans in panel", fmt(n_all), pct(n_all, n_all)))
    emit("%-52s %14s %10s" % ("ever reach status >= %d" % DQ_DEEP, fmt(n_deep), pct(n_deep, n_all)))
    emit("%-52s %14s %10s" % ("never reach status >= %d" % DQ_DEEP,
                              fmt(n_all - n_deep), pct(n_all - n_deep, n_all)))
    emit()
    emit("%-52s %14s" % ("total panel rows (loan-months)", fmt(total_rows)))
    if nonnumeric:
        emit()
        emit("non-numeric delinquency status codes seen (excluded from >=%d by the" % DQ_DEEP)
        emit(".str.isdigit() guard)   (unit: loan-months)")
        rule()
        for code, cnt in sorted(nonnumeric.items(), key=lambda kv: -kv[1]):
            shown = code if code else "(blank)"
            emit("    %-12s %14s" % (shown, fmt(cnt)))

    # ---------------- (b) ----------------
    touch = state["first_deep_idx"].dropna()
    years = ((touch - 1) // 12).astype(int)
    emit()
    rule("=")
    emit("(b) FIRST TOUCH OF DQ>=%d, BY CALENDAR YEAR   (unit: loans)" % DQ_DEEP)
    rule("=")
    emit("%-8s %14s %10s %s" % ("year", "loans", "share", "cumulative"))
    rule()
    vc = years.value_counts().sort_index()
    cum = 0
    for year, cnt in vc.items():
        cum += int(cnt)
        emit("%-8d %14s %10s %10s" % (year, fmt(cnt), pct(int(cnt), n_deep), pct(cum, n_deep)))
    rule()
    emit("%-8s %14s %10s" % ("TOTAL", fmt(int(vc.sum())), pct(int(vc.sum()), n_deep)))
    emit()
    emit("earliest first touch: %s     latest first touch: %s"
         % (index_to_yyyymm(touch.min()), index_to_yyyymm(touch.max())))

    # ---------------- (c) ----------------
    cured = state.index.isin(cure_idx.index)
    n_cured = int(cured.sum())
    emit()
    rule("=")
    emit("(c) CURE - ever report '00' at a month AFTER first touch   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("quantity", "loans", "share of DQ>=6"))
    rule()
    emit("%-52s %14s %10s" % ("loans ever at DQ>=%d" % DQ_DEEP, fmt(n_deep), pct(n_deep, n_deep)))
    emit("%-52s %14s %10s" % ("of these, later report status '00' (cured)",
                              fmt(n_cured), pct(n_cured, n_deep)))
    emit("%-52s %14s %10s" % ("never return to '00'",
                              fmt(n_deep - n_cured), pct(n_deep - n_cured, n_deep)))

    # ---------------- (d) ----------------
    aligned = cure_idx.reindex(state.index)
    months_to_cure = (aligned - state["first_deep_idx"]).dropna()
    emit()
    rule("=")
    emit("(d) MONTHS FROM FIRST DQ>=%d TO FIRST SUBSEQUENT '00'   (unit: months)" % DQ_DEEP)
    rule("=")
    emit("Population: the %s cured loans. Read the SHAPE, not just the centre:" % fmt(n_cured))
    emit("mass concentrated on one value => administrative re-aging; a long right")
    emit("tail => genuine workout.")
    rule()
    distribution(months_to_cure.to_numpy(), "months")

    emit()
    emit("HISTOGRAM OF MONTHS-TO-CURE   (unit: loans)")
    rule()
    mtc = months_to_cure.astype(int)
    emit("%-12s %14s %10s %s" % ("months", "loans", "share", ""))
    rule()
    exact = mtc.value_counts().sort_index()
    for m in range(1, 13):
        cnt = int(exact.get(m, 0))
        bar = "#" * int(round(40.0 * cnt / max(1, int(exact.max()))))
        emit("%-12d %14s %10s %s" % (m, fmt(cnt), pct(cnt, n_cured), bar))
    for lo, hi in [(13, 18), (19, 24), (25, 36), (37, 60), (61, 120), (121, 10 ** 6)]:
        cnt = int(((mtc >= lo) & (mtc <= hi)).sum())
        label = "%d-%d" % (lo, hi) if hi < 10 ** 6 else "121+"
        emit("%-12s %14s %10s" % (label, fmt(cnt), pct(cnt, n_cured)))
    rule()
    emit("%-12s %14s %10s" % ("mode", fmt(int(exact.idxmax())),
                              pct(int(exact.max()), n_cured)))

    n_mod_y_cured = int(state.loc[cured, "mod_y"].sum())
    n_mod_p_cured = int(state.loc[cured, "mod_p"].sum())
    n_mod_y_deep = int(state["mod_y"].sum())
    emit()
    emit("MODIFICATION FLAG (column %d, present in layout)   (unit: loans)" % COL_MOD)
    rule()
    emit("%-52s %14s %10s" % ("cured loans with modification_flag 'Y' ever",
                              fmt(n_mod_y_cured), pct(n_mod_y_cured, n_cured)))
    emit("%-52s %14s %10s" % ("cured loans with modification_flag 'P' ever",
                              fmt(n_mod_p_cured), pct(n_mod_p_cured, n_cured)))
    emit("%-52s %14s %10s" % ("all DQ>=6 loans with flag 'Y' ever",
                              fmt(n_mod_y_deep), pct(n_mod_y_deep, n_deep)))
    emit()
    emit("'Y' and 'P' are reported separately and not summed: the brief asks for 'Y',")
    emit("and 'P' is a distinct value in this column.")

    # ---------------- (e) ----------------
    codes = zbc.groupby("loan")["code"].first()
    codes = codes.reindex(state.index)
    n_ce = int(codes.isin(CREDIT_EVENT).sum())
    n_pp = int(codes.isin(PREPAID).sum())
    n_ae = int(codes.isin(ADMIN_EXIT).sum())
    n_open = int(codes.isna().sum())
    n_other = n_deep - n_ce - n_pp - n_ae - n_open

    emit()
    rule("=")
    emit("(e) TERMINAL OUTCOME OF DQ>=%d LOANS   (unit: loans)" % DQ_DEEP)
    rule("=")
    emit("%-40s %-12s %12s %14s" % ("outcome", "codes", "loans", "share of DQ>=6"))
    rule()
    emit("%-40s %-12s %12s %14s" % ("credit event", "/".join(CREDIT_EVENT),
                                    fmt(n_ce), pct(n_ce, n_deep)))
    emit("%-40s %-12s %12s %14s" % ("prepaid / matured", "/".join(PREPAID),
                                    fmt(n_pp), pct(n_pp, n_deep)))
    emit("%-40s %-12s %12s %14s" % ("administrative exit (censored)", "/".join(ADMIN_EXIT),
                                    fmt(n_ae), pct(n_ae, n_deep)))
    emit("%-40s %-12s %12s %14s" % ("other / unclassified code", "-",
                                    fmt(n_other), pct(n_other, n_deep)))
    emit("%-40s %-12s %12s %14s" % ("still open (censored at cutoff)", "-",
                                    fmt(n_open), pct(n_open, n_deep)))
    rule()
    emit("%-40s %-12s %12s %14s" % ("TOTAL", "", fmt(n_deep), pct(n_deep, n_deep)))
    emit()
    emit("Eventual default probability conditional on ever reaching DQ>=%d" % DQ_DEEP)
    emit("  unadjusted (all DQ>=6 as denominator)        : %s" % pct(n_ce, n_deep))
    resolved = n_ce + n_pp
    emit("  excluding censored (admin exits + still open): %s" % pct(n_ce, resolved))
    emit("  denominator for the second figure: %s resolved loans" % fmt(resolved))

    # ---------------- (f) ----------------
    deep_lm = int(state["deep_months"].sum())
    emit()
    rule("=")
    emit("(f) DWELL TIME AND IMPLIED MONTHLY HAZARD")
    rule("=")
    emit("The two quantities below are different units and are not interchangeable.")
    rule()
    emit("%-56s %18s" % ("loans ever at status >= %d                [loans]" % DQ_DEEP,
                         fmt(n_deep)))
    emit("%-56s %18s" % ("observations at status >= %d        [loan-months]" % DQ_DEEP,
                         fmt(deep_lm)))
    emit("%-56s %18.3f" % ("mean deep-delinquent months per such loan", deep_lm / float(n_deep)))
    emit("%-56s %18s" % ("credit-event terminations among them      [loans]", fmt(n_ce)))
    emit()
    emit("DEEP-DELINQUENT MONTHS PER LOAN   (unit: months, over the %s DQ>=6 loans)"
         % fmt(n_deep))
    rule()
    distribution(state["deep_months"].to_numpy(), "months")
    emit()
    emit("IMPLIED PER-MONTH HAZARD")
    rule()
    emit("  credit events [loans] / deep-delinquent exposure [loan-months]")
    emit("  = %s / %s" % (fmt(n_ce), fmt(deep_lm)))
    emit("  = %.6f per loan-month  (%.4f%% per month)"
         % (n_ce / float(deep_lm), 100.0 * n_ce / float(deep_lm)))
    emit()
    emit("Caveat: the numerator counts loans (each terminates at most once) while the")
    emit("denominator counts loan-months of exposure. This is an exposure-weighted")
    emit("average hazard over the whole deep-delinquent spell, not a per-month")
    emit("transition rate from a duration-specific model; it is not a life-table")
    emit("hazard and should not be annualised by multiplying by 12.")

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
