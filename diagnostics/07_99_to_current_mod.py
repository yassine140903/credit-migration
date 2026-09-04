"""Modification flags on the 99 -> 00 transitions found in diagnostic 06.

Diagnostic 06 found 110 loan-months at status 99 whose immediately following
calendar month reports "00" (Current). Under the CAP reading of 99 that jump is
impossible; a loan modification is the obvious candidate explanation, since a
modification re-sets the delinquency clock. This script reports what
modification_flag (column 7) actually says on and around those transitions.

UNITS. The headline figure from diagnostic 06 is 110 LOAN-MONTHS, not 110 loans:
one loan may contribute more than one 99 -> 00 transition. Both are reported
separately throughout and never conflated.

modification_flag takes three values in this dataset: "Y", "P", and blank. They
are reported separately and never summed.

TWO PASSES. Loan rows are not assumed sorted or contiguous and may straddle chunk
and file boundaries.
    pass 1  [0,3]      the set of loans ever reporting status 99
    pass 2  [0,1,3,7]  the full history of those loans, from which the adjacent
                       99 -> 00 transitions are found by self-merging on
                       (loan, month + 1) - so a reporting gap never supplies a
                       false neighbour, and row order is irrelevant.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "07_99_to_current_mod.txt")

COL_LOAN = 0
COL_PERIOD = 1
COL_DQ = 3
COL_MOD = 7

TARGET = "99"
CURRENT = "00"

# Cross-check from diagnostic 06 on the same 8-file glob.
EXPECTED_TRANSITIONS = 110
EXPECTED_99_LOANS = 569

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


def show(v):
    return v if v != "" else "(blank)"


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


def pass1(paths):
    parts = []
    for path in paths:
        t0 = time.time()
        hits = []
        for chunk in reader_for(path, [COL_LOAN, COL_DQ]):
            m = chunk[COL_DQ].str.strip() == TARGET
            if m.any():
                hits.append(chunk.loc[m, COL_LOAN].str.strip().unique())
        h = pd.unique(np.concatenate(hits)) if hits else np.array([])
        parts.append(h)
        emit("    %-30s 99-loans %8s   (%.1fs)"
             % (os.path.basename(path), fmt(len(h)), time.time() - t0))
    return pd.unique(np.concatenate(parts)) if parts else np.array([])


def pass2(paths, hit_set):
    parts = []
    for path in paths:
        t0 = time.time()
        kept = 0
        for chunk in reader_for(path, [COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD]):
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
                "mod": chunk.loc[keep, COL_MOD].str.strip().to_numpy(),
            }))
        emit("    %-30s rows kept %11s   (%.1fs)"
             % (os.path.basename(path), fmt(kept), time.time() - t0))
    hist = pd.concat(parts, ignore_index=True, copy=False)
    hist = hist.dropna(subset=["idx"])
    hist["idx"] = hist["idx"].astype(np.int64)
    return hist


def flag_table(series, n, title, unit):
    emit(title)
    rule()
    emit("%-16s %14s %12s" % ("modification_flag", unit, "share"))
    rule()
    vc = series.value_counts(dropna=False)
    for v, c in vc.items():
        emit("%-16s %14s %12s" % (show(v), fmt(int(c)), pct(int(c), n)))
    rule()
    emit("%-16s %14s %12s" % ("TOTAL", fmt(n), pct(n, n)))


def main():
    t_start = time.time()
    paths = sorted(glob.glob(GLOB))

    rule("=")
    emit("07_99_to_current_mod - modification flags on the 99 -> 00 transitions")
    rule("=")
    emit("glob        : %s" % GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("columns     : %d loan, %d period, %d dq_status, %d modification_flag"
         % (COL_LOAN, COL_PERIOD, COL_DQ, COL_MOD))
    emit("files found : %d" % len(paths))
    if not paths:
        emit("NO FILES MATCHED - aborting")
        write_out()
        return

    emit()
    emit("PASS 1/2 - loans ever reporting status 99")
    rule()
    hit_loans = pass1(paths)
    n_hits = len(hit_loans)
    emit()
    emit("%-52s %14s" % ("loans ever reporting 99", fmt(n_hits)))
    emit("%-52s %14s" % ("expected (diagnostic 06)", fmt(EXPECTED_99_LOANS)))
    emit("%-52s %14s" % ("status", "MATCH" if n_hits == EXPECTED_99_LOANS
                         else "*** MISMATCH ***"))

    emit()
    emit("PASS 2/2 - full history of those loans")
    rule()
    hist = pass2(paths, set(hit_loans.tolist()))

    # ---- adjacent 99 -> 00 transitions -------------------------------------
    h99 = hist[hist["status"] == TARGET][["loan", "idx", "mod"]].rename(
        columns={"mod": "mod_at_99"})
    nxt = hist[["loan", "idx", "status", "mod"]].rename(
        columns={"status": "next_status", "mod": "mod_at_next"})
    nxt = nxt.assign(idx=nxt["idx"] - 1)
    joined = h99.merge(nxt, on=["loan", "idx"], how="inner")
    tr = joined[joined["next_status"] == CURRENT].copy()

    n_tr = len(tr)
    n_tr_loans = tr["loan"].nunique()

    emit()
    rule("=")
    emit("THE 99 -> 00 TRANSITIONS")
    rule("=")
    emit("%-52s %14s" % ("transitions found        [loan-months]", fmt(n_tr)))
    emit("%-52s %14s" % ("expected (diagnostic 06) [loan-months]",
                         fmt(EXPECTED_TRANSITIONS)))
    emit("%-52s %14s" % ("status", "MATCH" if n_tr == EXPECTED_TRANSITIONS
                         else "*** MISMATCH ***"))
    if n_tr != EXPECTED_TRANSITIONS:
        emit("A mismatch is a BUG in one of the two scripts, not a data finding.")
    rule()
    emit("%-52s %14s" % ("distinct loans contributing them  [loans]", fmt(n_tr_loans)))
    emit("%-52s %14.3f" % ("transitions per such loan", n_tr / float(n_tr_loans)
                           if n_tr_loans else 0.0))
    emit()
    emit("The 110 headline from diagnostic 06 counts LOAN-MONTHS. The loan count")
    emit("above is the distinct-mortgage figure and is the smaller of the two.")

    if not n_tr:
        write_out()
        return

    # ---- flags on the two rows of the transition ---------------------------
    emit()
    rule("=")
    emit("MODIFICATION FLAG ON THE TRANSITION ROWS   (unit: loan-months)")
    rule("=")
    flag_table(tr["mod_at_99"], n_tr,
               "On the 99 row itself (the month before the cure)", "loan-months")
    emit()
    flag_table(tr["mod_at_next"], n_tr,
               "On the 00 row itself (the cure month)", "loan-months")

    emit()
    emit("JOINT: FLAG AT 99 x FLAG AT 00   (unit: loan-months)")
    rule()
    ct = pd.crosstab(tr["mod_at_99"], tr["mod_at_next"])
    cols = list(ct.columns)
    emit("%-16s %s" % ("at 99 \\ at 00",
                       " ".join("%12s" % show(c) for c in cols)))
    rule()
    for i, row in ct.iterrows():
        emit("%-16s %s" % (show(i), " ".join("%12s" % fmt(row[c]) for c in cols)))
    rule()
    emit("%-16s %s" % ("TOTAL", " ".join("%12s" % fmt(ct[c].sum()) for c in cols)))

    # ---- flags anywhere in the loan's history ------------------------------
    tl = sorted(tr["loan"].unique().tolist())
    sub = hist[hist["loan"].isin(set(tl))]
    ever = sub.groupby("loan")["mod"].agg(
        ever_y=lambda s: (s == "Y").any(),
        ever_p=lambda s: (s == "P").any(),
    )
    n_y = int(ever["ever_y"].sum())
    n_p = int(ever["ever_p"].sum())
    n_neither = int((~ever["ever_y"] & ~ever["ever_p"]).sum())

    emit()
    rule("=")
    emit("MODIFICATION FLAG ANYWHERE IN THESE LOANS' HISTORY   (unit: loans)")
    rule("=")
    emit("%-52s %14s %10s" % ("quantity", "loans", "share"))
    rule()
    emit("%-52s %14s %10s" % ("loans with a 99 -> 00 transition",
                              fmt(n_tr_loans), pct(n_tr_loans, n_tr_loans)))
    emit("%-52s %14s %10s" % ("  ever flagged 'Y'", fmt(n_y), pct(n_y, n_tr_loans)))
    emit("%-52s %14s %10s" % ("  ever flagged 'P'", fmt(n_p), pct(n_p, n_tr_loans)))
    emit("%-52s %14s %10s" % ("  never flagged either",
                              fmt(n_neither), pct(n_neither, n_tr_loans)))
    emit()
    emit("'Y' and 'P' are counted separately and never summed; a loan showing both")
    emit("appears in both rows.")

    # ---- baseline: the other 99 loans --------------------------------------
    others = hist[~hist["loan"].isin(set(tl))]
    if len(others):
        oever = others.groupby("loan")["mod"].agg(
            ever_y=lambda s: (s == "Y").any(),
            ever_p=lambda s: (s == "P").any(),
        )
        n_o = len(oever)
        oy = int(oever["ever_y"].sum())
        op = int(oever["ever_p"].sum())
        emit()
        emit("BASELINE - 99-reporting loans WITHOUT a 99 -> 00 transition (unit: loans)")
        rule()
        emit("%-52s %14s %10s" % ("such loans", fmt(n_o), pct(n_o, n_o)))
        emit("%-52s %14s %10s" % ("  ever flagged 'Y'", fmt(oy), pct(oy, n_o)))
        emit("%-52s %14s %10s" % ("  ever flagged 'P'", fmt(op), pct(op, n_o)))
        emit()
        emit("Comparing this against the block above shows whether modification is")
        emit("specific to the loans that jump 99 -> 00 or common to all 99 loans.")

    # ---- timing of the first Y relative to the transition -------------------
    firstY = (sub[sub["mod"] == "Y"].groupby("loan")["idx"].min()
              if (sub["mod"] == "Y").any() else pd.Series(dtype="float64"))
    if len(firstY):
        first_tr = tr.groupby("loan")["idx"].min()
        delta = (first_tr - firstY.reindex(first_tr.index)).dropna()
        emit()
        rule("=")
        emit("TIMING: MONTHS FROM FIRST 'Y' TO THE 99 ROW OF THE FIRST TRANSITION")
        rule("=")
        emit("Positive = the 'Y' precedes the transition.   (unit: months)")
        rule()
        emit("%-20s %14s" % ("loans measured", fmt(len(delta))))
        if len(delta):
            v = delta.to_numpy(dtype=float)
            emit("%-20s %14.2f" % ("mean", v.mean()))
            emit("%-20s %14.1f" % ("min", v.min()))
            for q in (10, 25, 50, 75, 90):
                tag = "p50 (median)" if q == 50 else "p%d" % q
                emit("%-20s %14.1f" % (tag, np.percentile(v, q)))
            emit("%-20s %14.1f" % ("max", v.max()))
            emit()
            emit("%-52s %14s %10s" % ("'Y' strictly BEFORE the 99 row",
                                      fmt(int((v > 0).sum())),
                                      pct(int((v > 0).sum()), len(v))))
            emit("%-52s %14s %10s" % ("'Y' in the same month as the 99 row",
                                      fmt(int((v == 0).sum())),
                                      pct(int((v == 0).sum()), len(v))))
            emit("%-52s %14s %10s" % ("'Y' only AFTER the 99 row",
                                      fmt(int((v < 0).sum())),
                                      pct(int((v < 0).sum()), len(v))))

    # ---- sample sequences ---------------------------------------------------
    emit()
    rule("=")
    emit("SAMPLE SEQUENCES AROUND THE TRANSITION (up to 8 loans)")
    rule("=")
    emit("Each row shows month:status/flag for the 6 months either side of the")
    emit("first 99 -> 00 transition. '>' marks the 99 row, '<' the 00 row.")
    for lid in tl[:8]:
        t0 = int(tr[tr["loan"] == lid]["idx"].min())
        w = hist[(hist["loan"] == lid) & (hist["idx"] >= t0 - 6)
                 & (hist["idx"] <= t0 + 7)].sort_values("idx")
        emit()
        emit("loan %s   transition at %s -> %s"
             % (lid, index_to_yyyymm(t0), index_to_yyyymm(t0 + 1)))
        cells = []
        for _, r in w.iterrows():
            mark = ">" if r["idx"] == t0 else ("<" if r["idx"] == t0 + 1 else " ")
            flag = r["mod"] if r["mod"] != "" else "-"
            cells.append("%s%s:%s/%s" % (mark, index_to_yyyymm(r["idx"]),
                                         r["status"], flag))
        for i in range(0, len(cells), 5):
            emit("    " + "  ".join(cells[i:i + 5]))

    total = time.time() - t_start
    emit()
    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (total, total / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
