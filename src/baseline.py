"""the homogeneous (time-pooled) Markov transition matrix.

    python -m src.baseline                      # writes artifacts/baseline/
    python -m src.baseline --out-dir some/dir

Estimation only - no forecasting, no matrix powers, no lifetime PD.

INPUT
    data/panel/panel_*.parquet through src.panel_io.load_panel, which validates
    the whole panel against its manifest first. Four columns are read, one
    partition at a time, in a single pass: origin_period, origin_state,
    next_state, role. Only rows with role == "fit" enter any count; condition
    and evaluate rows are counted by role for the report and nothing else.

STATE ORDER - fixed everywhere, asserted against src.panel.STATES
    Current, DQ30, DQ60, DQ90+, Prepaid, Default
    The first four rows are ESTIMATED. Prepaid and Default are absorbing: the
    build never emits a pair from them (R1), so their rows are SET BY
    CONSTRUCTION to the identity, never estimated.

STEPS
    1  count matrix     6x6 integer counts of fit rows, origin x destination.
                        Its total must equal the fit-row count taken from the
                        role column alone; the absorbing origin rows must be
                        all zero. Either failure stops the run.
    2  structural zeros Current->DQ60, Current->DQ90+, DQ30->DQ90+ are expected
                        to be zero if the delinquency counter advances at most
                        one step a month. Reported PASS/FAIL, never corrected.
    3  probabilities    each estimated row divided by its total. No smoothing,
                        no floor. A zero row total is a hard error.
    4  sparsity         per estimated row: total, smallest non-zero cell, zero
                        cells. Reported only.
    5  time instability the same estimate per calendar year of origin_period in
                        the fit window. Diagnostic only; used for nothing.

OUTPUT (artifacts/baseline/)
    counts.csv              the integer count matrix - the primary artefact
    transition_matrix.csv   probabilities derived from it, with a row_basis
                            column marking each row estimated / by_construction
    baseline_report.txt     every check and table, counts beside every share

Probabilities are always derived from counts at read time: load_baseline()
reads counts.csv and applies the same probability_matrix() used here, so
transition_matrix.csv is for inspection, not the source of truth.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src import panel
from src.panel_io import load_panel, read_manifest

STATES = ["Current", "DQ30", "DQ60", "DQ90+", "Prepaid", "Default"]
N_STATES = len(STATES)
ESTIMATED = STATES[:4]
ABSORBING = STATES[4:]
FIT_ROLE = "fit"
FIT_YEARS = (2006, 2007, 2008)
CUT_PERIOD = panel.CUT_YEAR * 100 + panel.CUT_MONTH        # 200812
STRUCTURAL_ZEROS = [("Current", "DQ60"), ("Current", "DQ90+"), ("DQ30", "DQ90+")]
COLUMNS = ["origin_period", "origin_state", "next_state", "role"]

OUT_DIR = os.path.join("artifacts", "baseline")
COUNTS_CSV = "counts.csv"
MATRIX_CSV = "transition_matrix.csv"
REPORT_TXT = "baseline_report.txt"

ROW_TOL = 1e-12
RULE_W = 121

if STATES != panel.STATES:
    raise ImportError("baseline state order %s disagrees with src.panel %s"
                      % (STATES, panel.STATES))


class BaselineError(RuntimeError):
    """A check the baseline depends on has failed; nothing is written."""


# ==========================================================================
# estimation - pure functions of counts
# ==========================================================================
def count_transitions(origin, dest):
    """6x6 int64 counts from integer state codes (0..5), origin x destination."""
    origin = np.asarray(origin, dtype=np.int64)
    dest = np.asarray(dest, dtype=np.int64)
    flat = np.bincount(origin * N_STATES + dest, minlength=N_STATES * N_STATES)
    return flat.reshape(N_STATES, N_STATES)


def absorbing_row_violations(counts):
    """{state: row counts} for each absorbing origin row that is not all zero."""
    out = {}
    for s in ABSORBING:
        row = counts[STATES.index(s)]
        if row.any():
            out[s] = row.copy()
    return out


def probability_matrix(counts):
    """Row-normalise the estimated rows; set the absorbing rows to the identity.

    No smoothing and no floor. Raises BaselineError on a zero estimated row
    total or an absorbing row that carries counts, rather than dividing.
    """
    counts = np.asarray(counts)
    bad = absorbing_row_violations(counts)
    if bad:
        raise BaselineError("absorbing origin row(s) carry counts: %s"
                            % {s: r.tolist() for s, r in bad.items()})
    totals = counts.sum(axis=1)
    zero = [s for s in ESTIMATED if totals[STATES.index(s)] == 0]
    if zero:
        raise BaselineError("estimated row(s) with total zero: %s" % zero)
    p = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for s in ESTIMATED:
        i = STATES.index(s)
        p[i] = counts[i] / totals[i]
    for s in ABSORBING:
        i = STATES.index(s)
        p[i, i] = 1.0
    sums = p.sum(axis=1)
    if not np.all(np.abs(sums - 1.0) <= ROW_TOL):
        raise BaselineError("row sums differ from 1.0: %s" % sums.tolist())
    return p


def counts_frame(counts):
    return pd.DataFrame(np.asarray(counts, dtype=np.int64),
                        index=pd.Index(STATES, name="origin_state"),
                        columns=STATES)


def load_baseline(out_dir=OUT_DIR):
    """(counts, probabilities) as DataFrames, probabilities derived from counts."""
    frame = pd.read_csv(os.path.join(out_dir, COUNTS_CSV), index_col=0)
    if list(frame.index) != STATES or list(frame.columns) != STATES:
        raise BaselineError("%s is not labelled in the fixed state order"
                            % COUNTS_CSV)
    counts = frame.to_numpy(dtype=np.int64)
    p = probability_matrix(counts)
    return counts_frame(counts), pd.DataFrame(p, index=frame.index,
                                              columns=STATES)


# ==========================================================================
# the single pass over the panel
# ==========================================================================
def _codes(series):
    """Categorical state column -> int codes in STATES order; -1 if null."""
    cats = list(series.cat.categories)
    unknown = [c for c in cats if c not in STATES]
    if unknown:
        raise BaselineError("panel carries unknown state label(s) %s" % unknown)
    remap = np.array([STATES.index(c) for c in cats] + [-1], dtype=np.int64)
    return remap[series.cat.codes.to_numpy()]      # code -1 indexes the -1 slot


def scan_panel():
    """One pass over every partition. Counts from fit rows only."""
    manifest = read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    scan = {"build_stamp": manifest["build_stamp"], "quarters": quarters,
            "pooled": np.zeros((N_STATES, N_STATES), dtype=np.int64),
            "by_year": {}, "role_rows": {}, "fit_rows": 0, "fit_null_state": 0,
            "fit_after_cut": 0, "fit_period_range": [None, None],
            "total_rows": 0}
    for q in quarters:
        t0 = time.time()
        df = load_panel(columns=COLUMNS, quarters=[q])
        scan["total_rows"] += len(df)
        for r, n in df["role"].value_counts(dropna=False).items():
            key = "(null)" if pd.isna(r) else str(r)
            scan["role_rows"][key] = scan["role_rows"].get(key, 0) + int(n)

        fit = (df["role"] == FIT_ROLE).to_numpy()
        scan["fit_rows"] += int(fit.sum())
        o = _codes(df["origin_state"])[fit]
        d = _codes(df["next_state"])[fit]
        period = df["origin_period"].to_numpy()[fit].astype(np.int64)
        del df

        ok = (o >= 0) & (d >= 0)
        scan["fit_null_state"] += int((~ok).sum())
        o, d, period = o[ok], d[ok], period[ok]
        scan["fit_after_cut"] += int((period > CUT_PERIOD).sum())
        if len(period):
            lo, hi = int(period.min()), int(period.max())
            cur = scan["fit_period_range"]
            cur[0] = lo if cur[0] is None else min(cur[0], lo)
            cur[1] = hi if cur[1] is None else max(cur[1], hi)

        scan["pooled"] += count_transitions(o, d)
        year = period // 100
        for y in np.unique(year):
            m = year == y
            acc = scan["by_year"].setdefault(
                int(y), np.zeros((N_STATES, N_STATES), dtype=np.int64))
            acc += count_transitions(o[m], d[m])
        print("    %s  %s fit rows  (%.1fs)" % (q, "{:,}".format(int(fit.sum())),
                                              time.time() - t0), flush=True)
    return scan


# ==========================================================================
# report
# ==========================================================================
class Report(object):

    def __init__(self):
        self.lines = []

    def emit(self, line=""):
        print(line, flush=True)
        self.lines.append(line)

    def rule(self, char="-", width=RULE_W):
        self.emit(char * width)

    def write(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def matrix_table(rep, counts, rows=STATES, shares=False, basis=None):
    """Counts, and optionally each row's shares on the line beneath."""
    w = 16
    rep.emit("%-10s" % "origin" + "".join("%*s" % (w, s) for s in STATES)
             + "%*s" % (w + 2, "row total") + ("   basis" if basis else ""))
    rep.rule()
    for s in rows:
        i = STATES.index(s)
        tot = counts[i].sum()
        rep.emit("%-10s" % s + "".join("%*s" % (w, fmt(c)) for c in counts[i])
                 + "%*s" % (w + 2, fmt(tot))
                 + ("   " + basis[s] if basis else ""))
        if shares:
            if s in ABSORBING:
                rep.emit("%-10s" % "  p" + "".join(
                    "%*s" % (w, "%.4f%%" % (100.0 * (j == i)))
                    for j in range(N_STATES)))
            else:
                rep.emit("%-10s" % "  p" + "".join("%*s" % (w, pct(c, tot))
                                                   for c in counts[i]))
    rep.rule()


def fail(rep, out_dir, message):
    rep.emit()
    rep.emit("HARD ERROR: " + message)
    rep.emit("No CSV was written. The report up to this point is at %s."
             % os.path.join(out_dir, REPORT_TXT))
    rep.write(os.path.join(out_dir, REPORT_TXT))
    raise SystemExit(1)


def run(out_dir=OUT_DIR):
    t_start = time.time()
    rep = Report()
    scan = scan_panel()
    counts = scan["pooled"]
    totals = counts.sum(axis=1)

    rep.rule("=")
    rep.emit("BASELINE - homogeneous Markov transition matrix, fit rows only")
    rep.rule("=")
    rep.emit("panel       : data/panel/panel_*.parquet via src.panel_io.load_panel")
    rep.emit("build stamp : %s" % scan["build_stamp"])
    rep.emit("partitions  : %s" % ", ".join(scan["quarters"]))
    rep.emit("columns     : %s" % ", ".join(COLUMNS))
    rep.emit("state order : %s" % ", ".join(STATES))
    rep.emit("estimated   : %s" % ", ".join(ESTIMATED))
    rep.emit("by constr.  : %s (identity rows)" % ", ".join(ABSORBING))
    rep.emit()
    rep.emit("%-24s%18s%12s" % ("panel rows by role", "rows", "share"))
    rep.rule()
    for r in sorted(scan["role_rows"]):
        n = scan["role_rows"][r]
        rep.emit("%-24s%18s%12s   %s" % (r, fmt(n), pct(n, scan["total_rows"]),
                                         "counted" if r == FIT_ROLE
                                         else "excluded"))
    rep.rule()
    rep.emit("%-24s%18s" % ("TOTAL", fmt(scan["total_rows"])))
    rep.emit()

    # ---- 1 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("1. COUNT MATRIX   (unit: fit rows; rows = origin, columns = destination)")
    rep.rule("=")
    matrix_table(rep, counts)
    total = int(counts.sum())
    rep.emit("%-58s%18s" % ("matrix total", fmt(total)))
    rep.emit("%-58s%18s" % ("fit rows in the panel (role column)", fmt(scan["fit_rows"])))
    rep.emit("%-58s%18s" % ("fit rows with a null or unmapped state",
                            fmt(scan["fit_null_state"])))
    lo, hi = scan["fit_period_range"]
    rep.emit("%-58s%18s" % ("fit origin_period range",
                            "%s - %s" % (lo, hi) if lo is not None else "n/a"))
    rep.emit("%-58s%18s   %s" % ("fit rows with origin_period after %d" % CUT_PERIOD,
                                 fmt(scan["fit_after_cut"]),
                                 "PASS" if scan["fit_after_cut"] == 0 else "FAIL"))
    if total != scan["fit_rows"]:
        fail(rep, out_dir, "matrix total %s != fit rows %s (difference %+d)"
             % (fmt(total), fmt(scan["fit_rows"]), total - scan["fit_rows"]))
    rep.emit("%-58s%18s" % ("matrix total == fit rows", "PASS"))
    bad = absorbing_row_violations(counts)
    for s in ABSORBING:
        rep.emit("%-58s%18s" % ("%s origin row all zero" % s,
                                "FAIL" if s in bad else "PASS"))
    if bad:
        for s, row in bad.items():
            rep.emit("    %s row: %s" % (s, ", ".join("%s %s" % (d, fmt(c))
                                                     for d, c in zip(STATES, row))))
        fail(rep, out_dir, "the panel emits pairs from an absorbing state, "
             "contradicting R1 of the build")
    rep.emit()

    # ---- 2 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("2. STRUCTURAL ZERO CHECK   (unit: fit rows)")
    rep.rule("=")
    rep.emit("Raw counts of the four estimated rows, before normalising:")
    rep.emit()
    matrix_table(rep, counts, rows=ESTIMATED)
    rep.emit()
    rep.emit("Hypothesis: the delinquency counter advances at most one step per month,")
    rep.emit("so these cells should be exactly zero.")
    rep.rule()
    rep.emit("%-24s%18s%18s%12s   %s" % ("cell", "count", "row total", "share",
                                         "result"))
    rep.rule()
    for a, b in STRUCTURAL_ZEROS:
        i, j = STATES.index(a), STATES.index(b)
        c = int(counts[i, j])
        rep.emit("%-24s%18s%18s%12s   %s"
                 % ("%s -> %s" % (a, b), fmt(c), fmt(totals[i]),
                    pct(c, totals[i]), "PASS (zero)" if c == 0
                    else "FAIL (%s)" % fmt(c)))
    rep.rule()
    rep.emit()

    # ---- 3 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("3. PROBABILITY MATRIX   (each estimated row / its total; no smoothing)")
    rep.rule("=")
    zero_rows = [s for s in ESTIMATED if totals[STATES.index(s)] == 0]
    if zero_rows:
        fail(rep, out_dir, "estimated row(s) with total zero, not divided: %s"
             % ", ".join(zero_rows))
    p = probability_matrix(counts)
    basis = {s: ("estimated" if s in ESTIMATED else "SET BY CONSTRUCTION")
             for s in STATES}
    rep.emit("Each origin: counts on the first line, row shares (p) beneath.")
    rep.emit()
    matrix_table(rep, counts, shares=True, basis=basis)
    rep.emit("Prepaid and Default are absorbing: 1.0 on the diagonal, 0.0 elsewhere,")
    rep.emit("set by construction and not estimated (their count rows are zero).")
    rep.emit()
    rep.emit("%-10s%24s%14s   %s" % ("origin", "row sum", "|sum - 1|", "result"))
    rep.rule()
    for i, s in enumerate(STATES):
        sm = p[i].sum()
        rep.emit("%-10s%24.17f%14.2e   %s" % (s, sm, abs(sm - 1.0),
                                              "PASS" if abs(sm - 1.0) <= ROW_TOL
                                              else "FAIL"))
    rep.rule()
    rep.emit("tolerance: %.0e" % ROW_TOL)
    rep.emit()

    # ---- 4 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("4. ROW SPARSITY   (unit: fit rows; reported only, no action taken)")
    rep.rule("=")
    rep.emit("%-10s%18s%26s%12s%14s   %s" % ("origin", "row total",
                                              "smallest non-zero cell", "share",
                                              "zero cells", "zero-cell destinations"))
    rep.rule()
    for s in ESTIMATED:
        i = STATES.index(s)
        row = counts[i]
        nz = row[row > 0]
        small = int(nz.min()) if len(nz) else 0
        at = STATES[int(np.where(row == small)[0][0])] if len(nz) else "-"
        zeros = [STATES[j] for j in range(N_STATES) if row[j] == 0]
        rep.emit("%-10s%18s%26s%12s%14d   %s"
                 % (s, fmt(totals[i]), "%s (%s)" % (fmt(small), at),
                    pct(small, totals[i]), len(zeros),
                    ", ".join(zeros) or "-"))
    rep.rule()
    rep.emit()

    # ---- 5 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("5. TIME INSTABILITY   (diagnostic only; unit: fit rows)")
    rep.rule("=")
    rep.emit("The pooled matrix of section 3 is the baseline. The yearly matrices")
    rep.emit("below exist only to quantify what pooling averages over; nothing")
    rep.emit("downstream uses them. Year = calendar year of origin_period.")
    rep.emit()
    years = sorted(set(FIT_YEARS) | set(scan["by_year"]))
    by_year = {y: scan["by_year"].get(y, np.zeros((N_STATES, N_STATES),
                                                  dtype=np.int64))
               for y in years}
    extra = [y for y in years if y not in FIT_YEARS]
    rep.emit("fit rows by year: %s" % ", ".join(
        "%d %s" % (y, fmt(by_year[y].sum())) for y in years))
    rep.emit("sum of yearly totals == pooled total: %s"
             % ("PASS" if sum(int(m.sum()) for m in by_year.values()) == total
                else "FAIL"))
    if extra:
        rep.emit("years outside %s present in fit rows: %s"
                 % (list(FIT_YEARS), extra))
    rep.emit()
    for s in ("Current", "DQ30"):
        i = STATES.index(s)
        rep.emit("%s row by year (counts; row shares beneath)" % s)
        rep.rule()
        rep.emit("%-10s" % "year" + "".join("%16s" % d for d in STATES)
                 + "%18s" % "row total")
        rep.rule()
        for y in years:
            row = by_year[y][i]
            tot = int(row.sum())
            rep.emit("%-10d" % y + "".join("%16s" % fmt(c) for c in row)
                     + "%18s" % fmt(tot))
            if tot == 0:
                rep.emit("%-10s%s" % ("  p", "row total zero - not estimated"))
            else:
                rep.emit("%-10s" % "  p" + "".join("%16s" % pct(c, tot)
                                                   for c in row))
        row = counts[i]
        rep.emit("%-10s" % "pooled" + "".join("%16s" % fmt(c) for c in row)
                 + "%18s" % fmt(row.sum()))
        rep.emit("%-10s" % "  p" + "".join("%16s" % pct(c, row.sum())
                                           for c in row))
        rep.rule()
        rep.emit()
    ic, j30, jpp = STATES.index("Current"), STATES.index("DQ30"), \
        STATES.index("Prepaid")
    rep.emit("Current -> DQ30 and Current -> Prepaid by year")
    rep.rule()
    rep.emit("%-10s%18s%18s%12s%18s%12s" % ("year", "Current total",
                                            "-> DQ30", "p", "-> Prepaid", "p"))
    rep.rule()
    for label, m in [(str(y), by_year[y]) for y in years] + [("pooled", counts)]:
        tot = int(m[ic].sum())
        rep.emit("%-10s%18s%18s%12s%18s%12s"
                 % (label, fmt(tot), fmt(m[ic, j30]), pct(m[ic, j30], tot),
                    fmt(m[ic, jpp]), pct(m[ic, jpp], tot)))
    rep.rule()
    rep.emit()

    # ---- outputs ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    counts_frame(counts).to_csv(os.path.join(out_dir, COUNTS_CSV))
    mat = pd.DataFrame(p, index=pd.Index(STATES, name="origin_state"),
                       columns=STATES)
    mat["row_basis"] = ["estimated" if s in ESTIMATED else "by_construction"
                        for s in STATES]
    mat.to_csv(os.path.join(out_dir, MATRIX_CSV), float_format="%.17g")
    # read-back: the written counts reproduce the matrix exactly
    rc, rp = load_baseline(out_dir)
    same = (np.array_equal(rc.to_numpy(), counts)
            and np.array_equal(rp.to_numpy(), p))
    rep.rule("=")
    rep.emit("OUTPUTS")
    rep.rule("=")
    for name in (COUNTS_CSV, MATRIX_CSV, REPORT_TXT):
        rep.emit("    %s" % os.path.join(out_dir, name))
    rep.emit("%-58s%18s" % ("counts.csv read back reproduces counts and p",
                            "PASS" if same else "FAIL"))
    elapsed = time.time() - t_start
    rep.rule("=")
    rep.emit("runtime: %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    rep.rule("=")
    rep.write(os.path.join(out_dir, REPORT_TXT))
    if not same:
        raise SystemExit(1)
    return counts, p


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate the homogeneous transition matrix from fit rows.")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="output directory (default: %(default)s)")
    args = parser.parse_args(argv)
    run(args.out_dir)


if __name__ == "__main__":
    main()
