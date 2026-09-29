"""Entry-age distribution of the loans in the built panel.

Freddie does not always begin reporting a loan the month after its first
payment; some loans first appear months or years after origination. This
measures the shape of that delay. It draws no conclusion, compares no groups,
joins no origination characteristics and proposes no cut point.

DEFINITION
    entry age   the minimum loan_age over a loan's panel rows. The panel is
                written in (loan_id, origin_period) order, so this is meant to
                equal the loan_age on the loan's first panel row; section
                CHECKS counts any loan where the two disagree.

    Unit: loans with at least one panel row. A loan with no panel row is not in
    the panel and not counted here. The first panel row is the first EMITTED
    origin month, which can sit after the loan's first performance month where
    src/panel.py dropped or suppressed the earliest pairs (R1, R3).

READING
    Partitions are read one at a time, three columns each. The per-loan minimum
    is taken within a partition, which is only correct if no loan spans two
    partitions; that is asserted here from the loan-id prefixes (one set per
    file, pairwise disjoint), independently of src/panel.py. Row order within
    each partition is asserted too.

Percentiles use the inverted-CDF method, so each is an entry age some loan
actually has.

Nothing is imported from src/. Nothing is written except the report. No git
command is run.
"""

import glob
import os
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

PANEL_GLOB = os.path.join("data", "panel", "panel_*.parquet")
OUT = os.path.join("diagnostics", "18_entry_age.txt")

RULE_W = 121
LIST_UPTO = 12                          # ages listed individually up to this
PERCENTILES = [50, 75, 90, 95, 99]
NULL_AGE = np.iinfo(np.int32).max       # stands in for a null loan_age

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


def ranges(values):
    """Sorted ints -> compact 'a-b' run strings."""
    out = []
    start = prev = None
    for v in values:
        if start is None:
            start = prev = v
        elif v == prev + 1:
            prev = v
        else:
            out.append(str(start) if start == prev else "%d-%d" % (start, prev))
            start = prev = v
    if start is not None:
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
    return out


def process_file(path):
    """Per-loan entry ages for one partition, plus the tallies CHECKS needs."""
    t = pq.read_table(path, columns=["loan_id", "origin_period", "loan_age"])
    n = t.num_rows
    res = {"name": os.path.basename(path), "rows": n, "loans": 0,
           "prefixes": set(), "null_rows": 0, "all_null_loans": 0,
           "first_null": 0, "first_ne_min": 0, "entry": np.empty(0, np.int64)}
    if n == 0:
        return res
    ids = t["loan_id"].combine_chunks()
    period = t["origin_period"].to_numpy()
    age_col = t["loan_age"].combine_chunks()

    head, tail = ids.slice(0, n - 1), ids.slice(1)
    if pc.any(pc.less(tail, head)).as_py():
        raise SystemExit("%s is not sorted by loan_id" % res["name"])
    change = pc.not_equal(tail, head).to_numpy(zero_copy_only=False)
    same = ~change
    if np.any(np.diff(period)[same] <= 0):
        raise SystemExit("%s: origin_period not strictly increasing within a "
                         "loan" % res["name"])
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))

    res["prefixes"] = set(pc.unique(pc.utf8_slice_codeunits(ids, 0, 5))
                          .to_pylist())
    is_null = age_col.is_null().to_numpy(zero_copy_only=False)
    age = pc.fill_null(age_col, NULL_AGE).to_numpy().astype(np.int64)
    loan_min = np.minimum.reduceat(age, starts)
    first = age[starts]

    all_null = loan_min == NULL_AGE
    res["loans"] = len(starts)
    res["null_rows"] = int(is_null.sum())
    res["all_null_loans"] = int(all_null.sum())
    res["first_null"] = int(is_null[starts].sum())
    res["first_ne_min"] = int(((first != loan_min) & ~is_null[starts]).sum())
    res["entry"] = loan_min[~all_null]
    return res


def main():
    t0 = time.time()
    paths = sorted(glob.glob(PANEL_GLOB))
    if not paths:
        raise SystemExit("no panel partitions matched %s" % PANEL_GLOB)

    rule("=")
    emit("18_entry_age - loan age at each loan's first panel row")
    rule("=")
    emit("panel       : %s" % PANEL_GLOB)
    emit("definition  : entry age = min(loan_age) over the loan's panel rows")
    emit("unit        : loans with at least one panel row")
    emit()

    per_file = [process_file(p) for p in paths]

    # ---- containment and per-file read --------------------------------------
    emit("PARTITIONS READ")
    rule()
    emit("%-28s%18s%18s   %s" % ("parquet file", "rows", "loans", "prefixes"))
    rule()
    owner = {}
    shared = {}
    for r in per_file:
        emit("%-28s%18s%18s   %s" % (r["name"], fmt(r["rows"]), fmt(r["loans"]),
                                     ", ".join(sorted(r["prefixes"]))))
        for p in r["prefixes"]:
            if p in owner:
                shared.setdefault(p, [owner[p]]).append(r["name"])
            owner.setdefault(p, r["name"])
    rule()
    emit("%-28s%18s%18s" % ("TOTAL", fmt(sum(r["rows"] for r in per_file)),
                            fmt(sum(r["loans"] for r in per_file))))
    if shared:
        raise SystemExit("loan-id prefix in more than one partition, per-file "
                         "minimum would be wrong: %r" % shared)
    emit("loan-id prefixes pairwise disjoint across partitions: yes, so no loan")
    emit("spans two files and the per-file minimum is the per-loan minimum.")
    emit()

    entry = np.concatenate([r["entry"] for r in per_file])
    n_loans = sum(r["loans"] for r in per_file)
    n = len(entry)
    lo, hi = int(entry.min()), int(entry.max())
    counts = np.bincount(entry - lo)                # index i -> age lo + i

    def at(age):
        return int(counts[age - lo]) if lo <= age <= hi else 0

    # ---- distribution ---------------------------------------------------------
    rule("=")
    emit("ENTRY AGE DISTRIBUTION   (unit: loans)")
    rule("=")
    emit("%-58s %18s" % ("loans in the panel", fmt(n_loans)))
    emit("%-58s %18s" % ("loans with an entry age (non-null loan_age)", fmt(n)))
    emit()
    emit("%-22s%18s%12s%18s%12s" % ("entry age (months)", "loans", "share",
                                    "cumulative", "cum share"))
    rule()
    cum = 0
    for age in range(lo, LIST_UPTO + 1):
        c = at(age)
        cum += c
        emit("%-22s%18s%12s%18s%12s%s" % (age, fmt(c), pct(c, n), fmt(cum),
                                          pct(cum, n),
                                          "   NO LOANS" if c == 0 else ""))
    above = int((entry > LIST_UPTO).sum())
    cum += above
    emit("%-22s%18s%12s%18s%12s" % ("> %d" % LIST_UPTO, fmt(above),
                                    pct(above, n), fmt(cum), pct(cum, n)))
    rule()
    emit("%-22s%18s%12s" % ("TOTAL", fmt(n), pct(n, n)))
    emit()

    emit("PERCENTILES   (inverted CDF: each value is an entry age a loan has)")
    rule()
    for q in PERCENTILES:
        emit("%-22s%18d" % ("p%d" % q,
                            int(np.percentile(entry, q, method="inverted_cdf"))))
    emit("%-22s%18d" % ("max", hi))
    emit("%-22s%18d" % ("min", lo))
    emit("%-22s%18.2f" % ("mean", entry.mean()))
    emit()

    # ---- shape ------------------------------------------------------------------
    rule("=")
    emit("SHAPE   (descriptive figures only)")
    rule("=")
    mode = lo + int(np.argmax(counts))
    emit("%-58s %18d" % ("modal entry age", mode))
    emit("%-58s %18s %12s" % ("loans at the modal age", fmt(at(mode)),
                              pct(at(mode), n)))
    emit("%-58s %18s %12s" % ("loans within 2 months of the minimum (%d-%d)"
                              % (lo, lo + 2),
                              fmt(int((entry <= lo + 2).sum())),
                              pct(int((entry <= lo + 2).sum()), n)))
    emit("%-58s %18s %12s" % ("loans at entry age <= %d" % LIST_UPTO,
                              fmt(n - above), pct(n - above, n)))
    emit("%-58s %18s %12s" % ("loans at entry age > %d" % LIST_UPTO,
                              fmt(above), pct(above, n)))
    for step in (24, 36, 60):
        k = int((entry > step).sum())
        emit("%-58s %18s %12s" % ("loans at entry age > %d" % step, fmt(k),
                                  pct(k, n)))
    listed = [at(a) for a in range(max(lo, mode), LIST_UPTO + 1)]
    rises = sum(1 for a, b in zip(listed, listed[1:]) if b > a)
    emit("%-58s %18s" % ("month-on-month rises in count, mode .. %d" % LIST_UPTO,
                         "%d of %d steps" % (rises, max(len(listed) - 1, 0))))
    emit()
    emit("Loans by entry-age band, to show where the tail sits:")
    rule()
    emit("%-22s%18s%12s" % ("band (months)", "loans", "share"))
    rule()
    edges = [lo, 1, 2, 3, 4, 7, 13, 25, 37, 61, 121, hi + 1]
    edges = sorted(set(e for e in edges if lo <= e <= hi + 1))
    for a, b in zip(edges, edges[1:]):
        k = int(((entry >= a) & (entry < b)).sum())
        label = str(a) if b == a + 1 else "%d-%d" % (a, b - 1)
        emit("%-22s%18s%12s" % (label, fmt(k), pct(k, n)))
    rule()
    emit()

    # ---- checks -------------------------------------------------------------------
    rule("=")
    emit("CHECKS AND ANYTHING UNEXPECTED")
    rule("=")
    zero = at(0)
    neg = int((entry < 0).sum())
    empty = [lo + i for i in np.nonzero(counts == 0)[0]]
    emit("%-58s %18s %12s" % ("loans with entry age 0", fmt(zero), pct(zero, n)))
    emit("%-58s %18s %12s" % ("loans with a negative entry age", fmt(neg),
                              pct(neg, n)))
    if neg:
        emit("    negative entry ages present: %s"
             % ", ".join("%d (%s)" % (a, fmt(at(a)))
                         for a in range(lo, min(0, hi + 1))))
    emit("%-58s %18s" % ("ages between min and max with no loans",
                         fmt(len(empty))))
    empty_low = [a for a in empty if a <= LIST_UPTO]
    emit("%-58s %18s" % ("   of which at or below %d" % LIST_UPTO,
                         ", ".join(str(a) for a in empty_low) or "none"))
    if empty:
        runs = ranges(empty)
        emit("    empty ages: %s%s" % (", ".join(runs[:40]),
                                       " ... (%d more runs)" % (len(runs) - 40)
                                       if len(runs) > 40 else ""))
    emit()
    null_rows = sum(r["null_rows"] for r in per_file)
    n_rows = sum(r["rows"] for r in per_file)
    emit("%-58s %18s %12s" % ("panel rows with a null loan_age", fmt(null_rows),
                              pct(null_rows, n_rows)))
    emit("%-58s %18s" % ("loans whose loan_age is null on every row",
                         fmt(sum(r["all_null_loans"] for r in per_file))))
    emit("%-58s %18s" % ("loans whose FIRST row has a null loan_age",
                         fmt(sum(r["first_null"] for r in per_file))))
    emit("%-58s %18s" % ("loans whose first-row loan_age != their minimum",
                         fmt(sum(r["first_ne_min"] for r in per_file))))
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
