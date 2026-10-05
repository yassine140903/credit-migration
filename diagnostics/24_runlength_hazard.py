"""Run-length hazard flattening, to set the semi-Markov ladder cap.   READ-ONLY.

    python diagnostics/24_runlength_hazard.py            # from the repo root

Writes diagnostics/24_runlength_hazard.txt and nothing else. It does not modify
the panel or any tracked file and runs no git command.

INPUT      the transition panel via src.panel_io.load_panel, role == 'fit' rows
           only: the cap is a modelling choice and must not be set on condition
           or evaluate rows. Read one partition at a time (a loan lives in one
           partition, so per-loan logic is exact within a partition).

EXCLUSIONS
  seasoned   loans whose first panel month is later than their first payment
             month. Their run_length reads 1 where the true time-in-state is
             higher, which flattens the low rungs artificially. The panel has no
             left-censoring flag, so the condition is DERIVED: loan_age is
             defined as months from the origination month (first payment month
             minus one) to month t, so loan_age == 1 on a loan's first row means
             first panel month == first payment month. The derivation is
             cross-checked here against first_payment read from the origination
             files; any disagreement is reported.
  post_gap   rows flagged post_gap (run_length is reset to 1 there by
             construction).

MEASUREMENT  per transient from_state (Current, DQ30, DQ60, DQ90+), per rung
             (run_length): rows, cumulative share of the state's rows at or
             below the rung, exit hazard = share of rows with
             to_state != from_state, and its change from the previous rung in
             percentage points. Rungs are walked from 1 upward and reporting
             stops at the first rung with fewer than MIN_ROWS rows.

CAP RULES    flatness  the first rung r such that |change| < 1.0 pp at r, r+1
                       and r+2 (three consecutive reported rungs; the change at
                       r is taken against r-1, so rung 1 has none and r >= 2).
             mass      the lowest rung with <= 5% of the state's rows strictly
                       above it (cumulative share at the rung >= 95%).
             binding   the larger of the two; the mass cap alone when flatness
                       is never satisfied.
"""

import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel                                                 # noqa: E402
from src.panel_io import load_panel, read_manifest                    # noqa: E402

OUT = os.path.join("diagnostics", "24_runlength_hazard.txt")
RULE_W = 100
MIN_ROWS = 1000
FLAT_PP = 1.0
FLAT_RUNS = 3
MASS_TAIL = 0.05
MAX_RL = 600
TRANSIENT = ["Current", "DQ30", "DQ60", "DQ90+"]
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def rule(char="-"):
    emit(char * RULE_W)


def fmt(n):
    return "{:,}".format(int(n))


def month_index(yyyymm):
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def first_payment_by_loan(path):
    """{loan_id: first payment YYYYMM} from one origination file."""
    fields, _ = panel._sheet_rows(panel.LAYOUT)[panel.ORIG_SHEET]
    idx = {}
    for key, want in (("loan", "loan identifier"), ("fpd", "first payment date")):
        hits = [f for f in fields if panel._normalise(f[1]) == panel._normalise(want)]
        if len(hits) != 1:
            raise SystemExit("layout lookup for %r matched %d rows" % (want, len(hits)))
        idx[key] = hits[0][0] - 1
    fr = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=list(idx.values()),
                     engine="c", na_filter=False)
    loan = fr[idx["loan"]].str.strip()
    fpd = pd.to_numeric(fr[idx["fpd"]].str.strip(), errors="coerce")
    return pd.Series(fpd.to_numpy(), index=loan.to_numpy())


def main():
    t0 = time.time()
    man = read_manifest()
    quarters = [p["quarter"] for p in man["partitions"]]
    orig_paths = sorted(glob.glob(panel.ORIG_GLOB))
    if [re.search(r"(20\d\dQ[1-4])", os.path.basename(p)).group(1) for p in orig_paths] != quarters:
        raise SystemExit("origination files and panel partitions disagree on quarters")

    n_st = len(TRANSIENT)
    # counts[state, run_length, 0 = stay, 1 = exit]
    counts = np.zeros((n_st, MAX_RL + 1, 2), dtype=np.int64)
    fit_rows = fit_loans = 0
    seen_ids = set()
    seasoned_loans = seasoned_rows = 0
    gap_rows_kept_pop = 0            # post_gap rows among non-seasoned loans
    absorbing_rows = 0               # fit rows whose from_state is not transient (must be 0)
    derive_vs_orig = {"agree": 0, "disagree": 0, "no_orig": 0, "orig_unparsed": 0}
    disagree_examples = []

    for q, op in zip(quarters, orig_paths):
        df = load_panel(quarters=[q], roles=["fit"],
                        columns=["loan_id", "from_period", "from_state", "to_state",
                                 "run_length", "loan_age", "post_gap"])
        if df.empty:
            continue
        ids = set(df["loan_id"].unique())
        if ids & seen_ids:
            raise SystemExit("%s: a loan appears in more than one partition" % q)
        seen_ids |= ids
        # per loan: the first fit row (partition is ordered by loan, from_period)
        lid, uniq = pd.factorize(df["loan_id"].to_numpy(), sort=False)
        first = np.ones(len(df), dtype=bool)
        first[1:] = lid[1:] != lid[:-1]
        if first.sum() != len(uniq):
            raise SystemExit("%s: rows are not contiguous by loan" % q)
        fp_arr = df["from_period"].to_numpy()
        if not (fp_arr[1:][~first[1:]] > fp_arr[:-1][~first[1:]]).all():
            raise SystemExit("%s: from_period not strictly increasing within a loan" % q)
        entry_age = df["loan_age"].to_numpy()[first]
        seasoned_loan = entry_age > 1                    # derived: first panel month > first payment month

        # cross-check against the origination file
        fpd = first_payment_by_loan(op)
        want = fpd.reindex(uniq)
        first_month = fp_arr[first]
        ok = want.notna().to_numpy()
        derived_orig = np.zeros(len(uniq), dtype=bool)
        derived_orig[ok] = month_index(first_month[ok]) > month_index(want.to_numpy()[ok].astype(np.int64))
        derive_vs_orig["no_orig"] += int((~pd.Index(uniq).isin(fpd.index)).sum())
        derive_vs_orig["orig_unparsed"] += int((~ok).sum())
        bad = ok & (derived_orig != seasoned_loan)
        derive_vs_orig["agree"] += int((ok & ~bad).sum())
        derive_vs_orig["disagree"] += int(bad.sum())
        for i in np.nonzero(bad)[0][:5]:
            disagree_examples.append((uniq[i], int(first_month[i]), int(want.iloc[i]),
                                      int(entry_age[i])))

        row_seasoned = seasoned_loan[lid]
        fit_rows += len(df)
        fit_loans += len(uniq)
        seasoned_loans += int(seasoned_loan.sum())
        seasoned_rows += int(row_seasoned.sum())

        keep = ~row_seasoned
        gap = df["post_gap"].to_numpy()
        gap_rows_kept_pop += int((keep & gap).sum())
        keep &= ~gap
        fs = df["from_state"].astype(str).to_numpy()[keep]
        ts = df["to_state"].astype(str).to_numpy()[keep]
        rl = df["run_length"].to_numpy()[keep].astype(np.int64)
        if rl.size and (rl.min() < 1 or rl.max() > MAX_RL):
            raise SystemExit("%s: run_length outside 1..%d" % (q, MAX_RL))
        exit_ = (fs != ts).astype(np.int64)
        for k, s in enumerate(TRANSIENT):
            m = fs == s
            for e in (0, 1):
                counts[k, :, e] += np.bincount(rl[m & (exit_ == e)], minlength=MAX_RL + 1)
        absorbing_rows += int((~np.isin(fs, TRANSIENT)).sum())
        del df, lid, fs, ts, rl, exit_, gap, keep
        emit_progress = "  %s done (%.0fs)" % (q, time.time() - t0)
        print(emit_progress, file=sys.stderr, flush=True)

    kept_rows = int(counts.sum())
    expected = fit_rows - seasoned_rows - gap_rows_kept_pop
    assert kept_rows + absorbing_rows == expected, (kept_rows, absorbing_rows, expected)

    # ---- report -------------------------------------------------------------
    rule("=")
    emit("RUN-LENGTH HAZARD FLATTENING   (diagnostic 24; read-only)")
    rule("=")
    emit("panel build  : %s  (%s rows, %d partitions; validated on every load)"
         % (man["build_stamp"], fmt(man["total_rows"]), man["n_partitions"]))
    emit("input        : role == 'fit' rows only: %s rows, %s loans" % (fmt(fit_rows), fmt(fit_loans)))
    emit()
    emit("EXCLUSION 1. Seasoned entrants (first panel month later than first payment month)")
    rule()
    emit("route taken  : DERIVED. The panel has no left-censoring flag (schema: %s)." % ", ".join(panel.SCHEMA))
    emit("               Condition = loan_age on the loan's first fit row > 1. loan_age is months from")
    emit("               the origination month (first payment month - 1) to month t, so it equals 1")
    emit("               exactly when the first panel month is the first payment month.")
    emit("               A loan's first FIT row is its first panel row: fit = train-split, from month")
    emit("               <= 2008-12, and a loan's rows run in from_period order.")
    emit("excluded     : %s loans of %s fit loans (%s)   |   %s of %s fit rows (%s)"
         % (fmt(seasoned_loans), fmt(fit_loans), "%.4f%%" % (100.0 * seasoned_loans / fit_loans),
            fmt(seasoned_rows), fmt(fit_rows), "%.4f%%" % (100.0 * seasoned_rows / fit_rows)))
    emit("cross-check  : derived condition vs first_payment read from the origination files:")
    emit("               agree %s   disagree %s   loan not in origination file %s   first_payment unparsed %s"
         % (fmt(derive_vs_orig["agree"]), fmt(derive_vs_orig["disagree"]),
            fmt(derive_vs_orig["no_orig"]), fmt(derive_vs_orig["orig_unparsed"])))
    for ex in disagree_examples[:5]:
        emit("               disagreement: loan %s first panel month %d first payment %d entry loan_age %d" % ex)
    emit("note         : a loan whose first pair was dropped by a reporting hole (R3) or whose first")
    emit("               month was suppressed (R1) also has a first panel month after its first")
    emit("               payment month, so it is excluded by the stated condition, whatever the cause.")
    emit()
    emit("EXCLUSION 2. post_gap rows (among the non-seasoned loans): %s rows" % fmt(gap_rows_kept_pop))
    emit("rows measured: %s  (= %s fit - %s seasoned - %s post_gap; %s with a non-transient from_state)"
         % (fmt(kept_rows), fmt(fit_rows), fmt(seasoned_rows), fmt(gap_rows_kept_pop), fmt(absorbing_rows)))
    emit()
    emit("Reporting stops at the first rung with fewer than %s rows. Hazard = share of rows with"
         % fmt(MIN_ROWS))
    emit("to_state != from_state. Change = this rung's hazard minus the previous rung's, in pp.")
    emit()

    summary = []
    for k, s in enumerate(TRANSIENT):
        n_rl = counts[k].sum(axis=1)
        total = int(n_rl.sum())
        rungs, rows, cum, haz, chg = [], [], [], [], []
        stop = None
        run = 0
        top = int(np.nonzero(n_rl)[0].max()) if total else 0
        for r in range(1, top + 2):
            n = int(n_rl[r]) if r <= MAX_RL else 0
            if n < MIN_ROWS:
                stop = (r, n)
                break
            run += n
            rungs.append(r)
            rows.append(n)
            cum.append(run / total)
            haz.append(100.0 * counts[k, r, 1] / n)
            chg.append(None if r == 1 else haz[-1] - haz[-2])
        # flatness
        flat = None
        for i in range(1, len(rungs) - FLAT_RUNS + 1):
            if all(abs(c) < FLAT_PP for c in chg[i:i + FLAT_RUNS]):
                flat = rungs[i]
                break
        # mass: over the state's FULL rung distribution, thin tail included
        cum_all = np.cumsum(n_rl) / total
        mass = int(np.nonzero(cum_all >= 1.0 - MASS_TAIL)[0][0])
        rule("=")
        emit("%s   (%s rows measured)" % (s, fmt(total)))
        rule("=")
        emit("%6s%14s%14s%12s%14s" % ("rung", "rows", "cum share", "hazard %", "change (pp)"))
        rule()
        for r, n, c, h, d in zip(rungs, rows, cum, haz, chg):
            emit("%6d%14s%13.2f%%%12.2f%14s" % (r, fmt(n), 100.0 * c, h,
                                                 "-" if d is None else "%+.2f" % d))
        rule()
        if stop is None:
            emit("reporting stop: none; every rung up to the maximum has >= %s rows" % fmt(MIN_ROWS))
        else:
            r, n = stop
            emit("reporting stop: rung %d has %s rows (< %s); not reported, nor any rung above it."
                 % (r, fmt(n), fmt(MIN_ROWS)))
            beyond = int(n_rl[r:].sum())
            emit("               %s rows (%.3f%% of the state) sit at rungs >= %d, unreported."
                 % (fmt(beyond), 100.0 * beyond / total, r))
        if flat is None:
            emit("flatness rule: NEVER SATISFIED. No run of %d consecutive reported rungs has |change| < %.1f pp"
                 " (rungs reported: 1-%d)." % (FLAT_RUNS, FLAT_PP, rungs[-1] if rungs else 0))
        else:
            emit("flatness rule: rung %d (|change| < %.1f pp at rungs %d, %d, %d)"
                 % (flat, FLAT_PP, flat, flat + 1, flat + 2))
        beyond_stop = stop is not None and mass >= stop[0]
        emit("mass rule    : rung %d (%.2f%% of rows at or below; %.2f%% strictly above)%s"
             % (mass, 100.0 * cum_all[mass], 100.0 * (1.0 - cum_all[mass]),
                "   NOTE: this rung is past the reporting stop" if beyond_stop else ""))
        if flat is None:
            bind, why = mass, "mass (flatness never satisfied)"
        elif flat > mass:
            bind, why = flat, "flatness"
        elif mass > flat:
            bind, why = mass, "mass"
        else:
            bind, why = mass, "both (equal)"
        emit("binding      : %d  (%s)" % (bind, why))
        emit()
        summary.append((s, flat, mass, bind, why))

    rule("=")
    emit("SUMMARY   (selected cap per state; flatness / mass / binding)")
    rule("=")
    for s, flat, mass, bind, why in summary:
        emit("%-8s cap = %3d   bound by %-34s flatness %-14s mass %d"
             % (s, bind, why, "never" if flat is None else flat, mass))
    rule("=")
    emit("runtime: %.1f s" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
