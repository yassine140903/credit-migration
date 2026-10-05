"""Destination mix by run-length.   READ-ONLY.

    python diagnostics/25_runlength_destination_mix.py   # from the repo root

Writes diagnostics/25_runlength_destination_mix.txt and nothing else. It does not
modify the panel or any tracked file and runs no git command.

RELATIONSHIP TO DIAGNOSTIC 24. Same population, same rungs, same exclusions: the
population code below is copied from 24_runlength_hazard.py unchanged (fit rows
only; seasoned entrants excluded by the derived condition, loan_age on the
loan's first fit row > 1, cross-checked against first_payment in the origination
files; post_gap rows excluded; reporting stops at the first rung with fewer than
MIN_ROWS rows). The one change: rows are tallied by DESTINATION (to_state, all
six states) instead of by exit / stay.

MEASUREMENT  per transient from_state, per rung: rows; total exit hazard; then
             one column per destination other than the origin, each as a share
             of ALL rows at that rung (not of exits), so the destination
             columns add to the total hazard. The total is asserted to equal
             the sum of its parts exactly (integer counts), and to equal the
             rows and hazard printed in diagnostics/24_runlength_hazard.txt
             (read back from that file, at its printed precision).

HEADLINE     per state, per destination: value at rung 1 and at the last
             reported rung, side by side, and the change in percentage points.

No decision rule: this measures; it selects no cap.
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

OUT = os.path.join("diagnostics", "25_runlength_destination_mix.txt")
D24 = os.path.join("diagnostics", "24_runlength_hazard.txt")
RULE_W = 124
MIN_ROWS = 1000
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


def read_d24():
    """{state: {rung: (rows, hazard %)}} from diagnostic 24's printed tables."""
    if not os.path.exists(D24):
        raise SystemExit("%s not found: run diagnostic 24 first" % D24)
    out, cur = {}, None
    with open(D24, encoding="utf-8") as fh:
        for ln in fh:
            m = re.match(r"^(Current|DQ30|DQ60|DQ90\+)\s+\(", ln)
            if m:
                cur = m.group(1)
                out[cur] = {}
                continue
            m = re.match(r"^\s*(\d+)\s+([\d,]+)\s+[\d.]+%\s+([\d.]+)\s+(\S+)\s*$", ln)
            if m and cur:
                out[cur][int(m.group(1))] = (int(m.group(2).replace(",", "")), float(m.group(3)))
    if set(out) != set(TRANSIENT) or not all(out.values()):
        raise SystemExit("could not read the four tables from %s" % D24)
    return out


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
    # counts[state, run_length, destination index in STATES]
    panel_states = panel.STATES
    counts = np.zeros((n_st, MAX_RL + 1, len(panel_states)), dtype=np.int64)
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
        for k, s in enumerate(TRANSIENT):
            m = fs == s
            for d, dname in enumerate(panel_states):
                counts[k, :, d] += np.bincount(rl[m & (ts == dname)], minlength=MAX_RL + 1)
        absorbing_rows += int((~np.isin(fs, TRANSIENT)).sum())
        del df, lid, fs, ts, rl, gap, keep
        emit_progress = "  %s done (%.0fs)" % (q, time.time() - t0)
        print(emit_progress, file=sys.stderr, flush=True)


    kept_rows = int(counts.sum())
    expected = fit_rows - seasoned_rows - gap_rows_kept_pop
    assert kept_rows + absorbing_rows == expected, (kept_rows, absorbing_rows, expected)
    d24 = read_d24()

    # ---- report -------------------------------------------------------------
    rule("=")
    emit("DESTINATION MIX BY RUN-LENGTH   (diagnostic 25; read-only)")
    rule("=")
    emit("panel build  : %s  (%s rows, %d partitions; validated on every load)"
         % (man["build_stamp"], fmt(man["total_rows"]), man["n_partitions"]))
    emit("population   : identical to diagnostic 24 (code reused unchanged).")
    emit("input        : role == 'fit' rows only: %s rows, %s loans" % (fmt(fit_rows), fmt(fit_loans)))
    emit("seasoned     : %s loans (%.4f%%) / %s rows (%.4f%%) excluded; derived condition (loan_age on the"
         % (fmt(seasoned_loans), 100.0 * seasoned_loans / fit_loans,
            fmt(seasoned_rows), 100.0 * seasoned_rows / fit_rows))
    emit("               first fit row > 1) agrees with first_payment in the origination files for %s of %s loans"
         % (fmt(derive_vs_orig["agree"]), fmt(fit_loans)))
    emit("post_gap     : %s rows excluded (non-seasoned loans)" % fmt(gap_rows_kept_pop))
    emit("rows measured: %s" % fmt(kept_rows))
    emit()
    emit("Every destination column is a share of ALL rows at the rung (percent), so the destination")
    emit("columns add to the total hazard. 'hazard' = share of rows with to_state != from_state.")
    emit("The origin state's own column is omitted. 'check' compares rows and hazard with diagnostic 24.")
    emit()

    headline = []
    for k, s in enumerate(TRANSIENT):
        n_rl = counts[k].sum(axis=1)
        total = int(n_rl.sum())
        dests = [d for d in panel.STATES if d != s]
        didx = [panel.STATES.index(d) for d in dests]
        sidx = panel.STATES.index(s)
        rungs, rows, haz, shares = [], [], [], []
        stop = None
        for r in range(1, int(np.nonzero(n_rl)[0].max()) + 2):
            n = int(n_rl[r]) if r <= MAX_RL else 0
            if n < MIN_ROWS:
                stop = (r, n)
                break
            exits = n - int(counts[k, r, sidx])
            parts = [int(counts[k, r, d]) for d in didx]
            assert sum(parts) == exits, (s, r)                 # additive decomposition, exact
            rungs.append(r)
            rows.append(n)
            haz.append(100.0 * exits / n)
            shares.append([100.0 * p / n for p in parts])
        # the free correctness check against diagnostic 24
        ref = d24[s]
        assert sorted(ref) == rungs, (s, "rung set differs from diagnostic 24")
        for r, n, h in zip(rungs, rows, haz):
            assert ref[r][0] == n, (s, r, "rows", n, ref[r][0])
            assert abs(round(h, 2) - ref[r][1]) < 1e-9, (s, r, "hazard", h, ref[r][1])
        rule("=")
        emit("%s   (%s rows measured)" % (s, fmt(total)))
        rule("=")
        emit("%5s%12s%9s" % ("rung", "rows", "hazard") + "".join("%11s" % ("to " + d) for d in dests)
             + "%8s" % "check")
        rule()
        for r, n, h, sh in zip(rungs, rows, haz, shares):
            emit("%5d%12s%9.2f" % (r, fmt(n), h) + "".join("%11.2f" % v for v in sh) + "%8s" % "ok")
        rule()
        r_stop, n_stop = stop
        emit("reporting stop: rung %d has %s rows (< %s); not reported, nor any rung above it."
             % (r_stop, fmt(n_stop), fmt(MIN_ROWS)))
        emit("check: %d rungs, rows and hazard (2 dp) reproduce diagnostic 24; destinations sum to the hazard exactly."
             % len(rungs))
        emit()
        headline.append((s, dests, rungs, haz, shares))

    rule("=")
    emit("HEADLINE: first rung vs last reported rung, by destination   (percent of all rows at the rung)")
    rule("=")
    for s, dests, rungs, haz, shares in headline:
        emit("%s:  rung %d  ->  rung %d" % (s, rungs[0], rungs[-1]))
        emit("%-14s%10s%10s%14s" % ("", "rung %d" % rungs[0], "rung %d" % rungs[-1], "change (pp)"))
        for j, d in enumerate(dests):
            emit("%-14s%10.2f%10.2f%+14.2f" % ("to " + d, shares[0][j], shares[-1][j],
                                              shares[-1][j] - shares[0][j]))
        emit("%-14s%10.2f%10.2f%+14.2f" % ("total hazard", haz[0], haz[-1], haz[-1] - haz[0]))
        emit()
    rule("=")
    emit("runtime: %.1f s" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
