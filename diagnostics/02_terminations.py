"""Termination accounting for the Freddie Mac monthly performance panel.

Pooled over all matched files: distinct loans, terminated (any zero_balance_code)
vs never terminated (censored at the data cutoff), the full frequency table of
zero_balance_code, and the brief's groupings as counts and shares of all loans.

Zero-balance code groupings (per the task brief):
    01              prepaid / matured
    02, 03, 09      credit events  (third-party sale, short sale, REO disposition)
    06, 15, 16      administrative exits (repurchase, note sale) - CENSORED,
                    counted as neither prepayment nor credit event
Any other code observed in the data is reported in an explicit UNCLASSIFIED
bucket and is never folded into one of the three groups above.

Two chunked passes over columns 0 and 8, so the loan universe and the termination
records are each accumulated with the right unit of observation:
    pass 1  distinct loan ids across the whole panel (the denominator)
    pass 2  the (loan, zero_balance_code) rows only - a sparse subset
Order-independent throughout; loan rows may straddle chunk and file boundaries.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "02_terminations.txt")

COL_LOAN = 0
COL_ZBC = 8

PREPAID = ["01"]
CREDIT_EVENT = ["02", "03", "09"]
ADMIN_EXIT = ["06", "15", "16"]

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
    return "{:,}".format(n)


def pct(num, den):
    return "n/a" if not den else "%.4f%%" % (100.0 * num / den)


def pass1_loan_universe(paths):
    """Distinct loan ids across the whole panel. Unit: loans."""
    per_file_uniques = []
    per_file_rows = []
    for path in paths:
        t0 = time.time()
        rows = 0
        chunk_uniques = []
        reader = pd.read_csv(
            path, sep="|", header=None, dtype=str, usecols=[COL_LOAN],
            chunksize=CHUNKSIZE, engine="c", na_filter=False,
        )
        for chunk in reader:
            rows += len(chunk)
            chunk_uniques.append(chunk[COL_LOAN].str.strip().unique())
        uniq = pd.unique(np.concatenate(chunk_uniques)) if chunk_uniques else np.array([])
        per_file_uniques.append(uniq)
        per_file_rows.append(rows)
        emit("    %-32s rows %14s   distinct loans %12s   (%.1fs)"
             % (os.path.basename(path), fmt(rows), fmt(len(uniq)), time.time() - t0))
    pooled = pd.unique(np.concatenate(per_file_uniques)) if per_file_uniques else np.array([])
    return pooled, per_file_uniques, sum(per_file_rows)


def pass2_termination_records(paths):
    """All rows carrying a zero_balance_code. Unit: (loan, code) rows."""
    parts = []
    for path in paths:
        t0 = time.time()
        n_hits = 0
        reader = pd.read_csv(
            path, sep="|", header=None, dtype=str, usecols=[COL_LOAN, COL_ZBC],
            chunksize=CHUNKSIZE, engine="c", na_filter=False,
        )
        for chunk in reader:
            code = chunk[COL_ZBC].str.strip()
            keep = code != ""
            if not keep.any():
                continue
            sub = pd.DataFrame(
                {"loan": chunk.loc[keep, COL_LOAN].str.strip(), "code": code[keep]}
            )
            n_hits += len(sub)
            parts.append(sub)
        emit("    %-32s zero_balance rows %12s   (%.1fs)"
             % (os.path.basename(path), fmt(n_hits), time.time() - t0))
    if not parts:
        return pd.DataFrame(columns=["loan", "code"])
    return pd.concat(parts, ignore_index=True, copy=False)


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("02_terminations - zero-balance / termination accounting (pooled)")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns read: %d loan_sequence_number, %d zero_balance_code"
         % (COL_LOAN, COL_ZBC))
    emit("files found : %d" % len(paths))
    for p in paths:
        emit("    %-58s %7.2f GB" % (p, os.path.getsize(p) / 1024 ** 3))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/2 - loan universe")
    rule()
    all_loans, per_file_uniques, total_rows = pass1_loan_universe(paths)
    n_loans = len(all_loans)

    emit()
    emit("PASS 2/2 - termination records")
    rule()
    term = pass2_termination_records(paths)

    # ---- loan-level termination status -------------------------------------
    # A loan may in principle carry more than one zero-balance row; collapse to
    # the loan level and report any loan with conflicting codes rather than
    # letting a double-count leak into the group tallies below.
    codes_per_loan = term.groupby("loan")["code"].agg(["nunique", "first", "size"])
    n_terminated = len(codes_per_loan)
    n_conflict = int((codes_per_loan["nunique"] > 1).sum())
    n_repeat_rows = int((codes_per_loan["size"] > 1).sum())

    emit()
    rule("=")
    emit("TERMINATED vs CENSORED   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-52s %14s %10s" % ("all distinct loans in panel", fmt(n_loans), pct(n_loans, n_loans)))
    emit("%-52s %14s %10s" % ("terminated (any zero_balance_code)",
                              fmt(n_terminated), pct(n_terminated, n_loans)))
    emit("%-52s %14s %10s" % ("never terminated (censored at cutoff)",
                              fmt(n_loans - n_terminated), pct(n_loans - n_terminated, n_loans)))
    emit()
    emit("%-52s %14s" % ("total panel rows", fmt(total_rows)))
    emit("%-52s %14s" % ("rows carrying a zero_balance_code", fmt(len(term))))
    emit("%-52s %14s" % ("loans with >1 zero_balance row", fmt(n_repeat_rows)))
    emit("%-52s %14s" % ("loans with conflicting distinct codes", fmt(n_conflict)))

    # ---- full frequency table ----------------------------------------------
    row_freq = term["code"].value_counts()
    loan_freq = codes_per_loan["first"].value_counts()

    emit()
    rule("=")
    emit("ZERO_BALANCE_CODE FREQUENCY TABLE (full, every code observed)")
    rule("=")
    emit("%-6s %-44s %12s %12s" % ("code", "label", "rows", "loans"))
    rule()
    for code in sorted(set(row_freq.index) | set(loan_freq.index)):
        label = ZBC_LABELS.get(code, "NOT IN BRIEF'S CODE LIST - unclassified")
        emit("%-6s %-44s %12s %12s"
             % (code, label, fmt(int(row_freq.get(code, 0))), fmt(int(loan_freq.get(code, 0)))))
    rule()
    emit("%-6s %-44s %12s %12s" % ("", "TOTAL", fmt(int(row_freq.sum())), fmt(int(loan_freq.sum()))))
    emit()
    emit("'rows' counts zero-balance rows; 'loans' counts distinct loans by their")
    emit("first observed code. The two differ only where a loan carries repeats.")

    # ---- groupings ----------------------------------------------------------
    first_code = codes_per_loan["first"]
    known = set(PREPAID) | set(CREDIT_EVENT) | set(ADMIN_EXIT)
    observed = set(first_code.unique())
    unclassified = sorted(observed - known)

    groups = [
        ("prepayment", PREPAID, "terminal, non-credit"),
        ("credit event", CREDIT_EVENT, "terminal, credit loss"),
        ("administrative exit", ADMIN_EXIT, "CENSORED - neither category"),
    ]

    emit()
    rule("=")
    emit("GROUPED OUTCOMES   (unit: loans; share is of all %s loans in panel)" % fmt(n_loans))
    rule("=")
    emit("%-22s %-14s %12s %10s  %s" % ("group", "codes", "loans", "share", "interpretation"))
    rule()
    grouped_total = 0
    for name, codes, note in groups:
        n = int(first_code.isin(codes).sum())
        grouped_total += n
        emit("%-22s %-14s %12s %10s  %s"
             % (name, "/".join(codes), fmt(n), pct(n, n_loans), note))

    n_unclassified = int(first_code.isin(unclassified).sum()) if unclassified else 0
    emit("%-22s %-14s %12s %10s  %s"
         % ("UNCLASSIFIED", "/".join(unclassified) if unclassified else "-",
            fmt(n_unclassified), pct(n_unclassified, n_loans),
            "code(s) absent from brief - NOT grouped"))
    rule()
    emit("%-22s %-14s %12s %10s"
         % ("all terminated", "", fmt(n_terminated), pct(n_terminated, n_loans)))
    emit("%-22s %-14s %12s %10s"
         % ("still open (censored)", "", fmt(n_loans - n_terminated),
            pct(n_loans - n_terminated, n_loans)))

    if unclassified:
        emit()
        emit("NOTE: zero_balance_code(s) %s appear in the data but are not in the"
             % ", ".join(unclassified))
        emit("      brief's mapping. They are reported separately above and are not")
        emit("      assigned to prepayment, credit event, or administrative exit.")

    emit()
    emit("Censoring note: administrative exits (%s) remove a loan from observation"
         % "/".join(ADMIN_EXIT))
    emit("without resolving whether it would have prepaid or defaulted; together with")
    emit("the never-terminated loans they form the censored portion of the panel.")

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
