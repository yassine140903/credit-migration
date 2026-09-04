"""Panel shape of the Freddie Mac monthly performance files (2006-2007 vintages).

Per file and pooled: row count, distinct loans, min/max monthly_reporting_period,
and the distribution of months-tracked per loan (deciles plus p1/p99, mean).

Single chunked pass over columns 0 (loan_sequence_number) and 1
(monthly_reporting_period). Rows for a loan are NOT assumed sorted or contiguous,
and may straddle chunk and file boundaries: per-loan aggregates are built from
small per-chunk groupby results which are concatenated and re-reduced once per
file, then the per-file loan-level frames are re-reduced again to pool. Both
reductions are order-independent (sum / min / max), so the result is exact
regardless of how rows fall across chunks.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "01_panel_shape.txt")

COL_LOAN = 0
COL_PERIOD = 1

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
    return "{:,}".format(n)


def yyyymm_to_index(s):
    """YYYYMM string -> integer month index (year * 12 + month); NaN if unparseable.

    Using a month index rather than the raw YYYYMM integer means differences and
    spans are true month counts, not decimal artefacts across year boundaries.
    """
    v = pd.to_numeric(s, errors="coerce")
    year = v // 100
    month = v % 100
    idx = year * 12 + month
    idx = idx.where((month >= 1) & (month <= 12))
    return idx


def index_to_yyyymm(idx):
    if idx is None or not np.isfinite(idx):
        return "n/a"
    year, month = divmod(int(idx) - 1, 12)
    return "%04d%02d" % (year, month + 1)


def reduce_loan_level(frames):
    """Concatenate per-chunk aggregate frames and reduce to one row per loan."""
    cat = pd.concat(frames, copy=False)
    grouped = cat.groupby(level=0, sort=False)
    return pd.DataFrame(
        {
            "rows": grouped["rows"].sum(),
            "min_idx": grouped["min_idx"].min(),
            "max_idx": grouped["max_idx"].max(),
        }
    )


def scan_file(path):
    """Return (n_rows, n_bad_period, per-loan DataFrame) for one performance file."""
    n_rows = 0
    n_bad_period = 0
    frames = []

    reader = pd.read_csv(
        path,
        sep="|",
        header=None,
        dtype=str,
        usecols=[COL_LOAN, COL_PERIOD],
        chunksize=CHUNKSIZE,
        engine="c",
        na_filter=False,
    )
    for chunk in reader:
        n_rows += len(chunk)
        loan = chunk[COL_LOAN].str.strip()
        idx = yyyymm_to_index(chunk[COL_PERIOD].str.strip())
        n_bad_period += int(idx.isna().sum())

        grouped = pd.DataFrame({"loan": loan, "idx": idx}).groupby("loan", sort=False)[
            "idx"
        ]
        frames.append(
            pd.DataFrame(
                {
                    "rows": grouped.size(),
                    "min_idx": grouped.min(),
                    "max_idx": grouped.max(),
                }
            )
        )

    if not frames:
        return n_rows, n_bad_period, pd.DataFrame(
            columns=["rows", "min_idx", "max_idx"]
        )
    return n_rows, n_bad_period, reduce_loan_level(frames)


def report(label, n_rows, n_bad_period, per_loan):
    emit()
    rule("=")
    emit("SCOPE: %s" % label)
    rule("=")

    n_loans = len(per_loan)
    emit("%-36s %16s" % ("rows", fmt(n_rows)))
    emit("%-36s %16s" % ("distinct loans", fmt(n_loans)))
    emit("%-36s %16s" % ("rows with unparseable period", fmt(n_bad_period)))
    if not n_loans:
        emit("no loans found - nothing further to report")
        return

    emit("%-36s %16s" % ("min monthly_reporting_period",
                         index_to_yyyymm(per_loan["min_idx"].min())))
    emit("%-36s %16s" % ("max monthly_reporting_period",
                         index_to_yyyymm(per_loan["max_idx"].max())))

    months = per_loan["rows"].to_numpy(dtype=np.float64)
    emit()
    emit("MONTHS TRACKED PER LOAN   (unit: months of observation for one loan)")
    rule()
    emit("%-14s %14s" % ("statistic", "months"))
    emit("%-14s %14.2f" % ("mean", months.mean()))
    emit("%-14s %14.2f" % ("std", months.std(ddof=0)))
    emit("%-14s %14d" % ("min", int(months.min())))
    emit("%-14s %14.1f" % ("p1", np.percentile(months, 1)))
    for q in range(10, 100, 10):
        emit("%-14s %14.1f" % ("p%-2d  (d%d)" % (q, q // 10),
                               np.percentile(months, q)))
    emit("%-14s %14.1f" % ("p99", np.percentile(months, 99)))
    emit("%-14s %14d" % ("max", int(months.max())))

    # Reconciliation. For a panel with no duplicate (loan, month) rows and no
    # interior gaps, rows-per-loan equals the span implied by min/max period.
    # Divergence is reported rather than silently absorbed, because it would
    # change how months-tracked should be interpreted above.
    span = per_loan["max_idx"] - per_loan["min_idx"] + 1
    rows = per_loan["rows"]
    emit()
    emit("PANEL INTEGRITY   (unit: loans)")
    rule()
    emit("%-48s %14s" % ("rows == span implied by min/max period",
                         fmt(int((rows == span).sum()))))
    emit("%-48s %14s" % ("rows >  span  (duplicate months present)",
                         fmt(int((rows > span).sum()))))
    emit("%-48s %14s" % ("rows <  span  (interior gaps present)",
                         fmt(int((rows < span).sum()))))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("01_panel_shape - Freddie Mac monthly performance panel shape")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns read: %d loan_sequence_number, %d monthly_reporting_period"
         % (COL_LOAN, COL_PERIOD))
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    per_file = []
    for path in paths:
        t0 = time.time()
        n_rows, n_bad, per_loan = scan_file(path)
        elapsed = time.time() - t0
        report("%s   (%.1fs)" % (os.path.basename(path), elapsed),
               n_rows, n_bad, per_loan)
        per_file.append((n_rows, n_bad, per_loan))

    total_rows = sum(x[0] for x in per_file)
    total_bad = sum(x[1] for x in per_file)
    pooled = reduce_loan_level([x[2] for x in per_file if len(x[2])])
    report("POOLED (all %d files)" % len(paths), total_rows, total_bad, pooled)

    # If any loan id appeared in more than one file, the pooled reduction above is
    # what keeps the pooled figures correct. Report the overlap either way.
    sum_file_loans = sum(len(x[2]) for x in per_file)
    emit()
    emit("CROSS-FILE LOAN OVERLAP   (unit: loans)")
    rule()
    emit("%-48s %14s" % ("sum of per-file distinct loans", fmt(sum_file_loans)))
    emit("%-48s %14s" % ("pooled distinct loans", fmt(len(pooled))))
    emit("%-48s %14s" % ("implied loans appearing in >1 file",
                         fmt(sum_file_loans - len(pooled))))

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
