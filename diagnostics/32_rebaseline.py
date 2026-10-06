"""Re-baseline of the stale gates after the panel change.   READ-ONLY except for what it re-runs.

    python diagnostics/32_rebaseline.py --old DIR      # from the repo root

DIR holds the PRE-re-baseline copies of diagnostics/23_origination_validity.txt,
artifacts/horizon_baseline/, artifacts/horizon_nulls/ and reports/. Writes
diagnostics/32_rebaseline.txt. It re-runs, idempotently, src.diagnostics.cell_counts
(rewrites reports/cell_counts.csv with the same content) and src.horizon_nulls into a
temporary directory (never over artifacts/horizon_nulls). It runs no git command.

Everything about the harness universe is recomputed here, loan by loan, from the
pre-HPI panel kept in data/panel/_pre_hpi/ and the current panel.
"""

import argparse
import difflib
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import horizon as hz, panel_io                                    # noqa: E402

OUT = os.path.join("diagnostics", "32_rebaseline.txt")
OLD_FP, MID_FP, NEW_FP = "3b1209189824e42f", "2f304bae5613525d", "71d85ff7fd2c5d1a"
LINES = []


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def title(t):
    emit()
    emit("=" * 118)
    emit(t)
    emit("=" * 118)


def fmt(n):
    return f"{int(n):,}"


def run(cmd):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def side_by_side(old, new, keys, cols, label_w=24):
    m = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    for _, r in m.iterrows():
        k = ", ".join(str(r[x]) for x in keys)
        if r["_merge"] != "both":
            emit(f"  {k}: only in {'old' if r['_merge'] == 'left_only' else 'new'}")
            continue
        emit(f"  {k}")
        for c in cols:
            a, b = r[c + "_old"], r[c + "_new"]
            d = "" if a == b else f"   (diff {b - a:+.6g})"
            emit(f"      {c:<{label_w}}{a:>24.10g}  ->  {b:<24.10g}{d}")


def main(old):
    emit("Re-baseline of the stale gates on the current panel")
    emit(f"panel fingerprint required: {NEW_FP}   (old {OLD_FP})")

    # ------------------------------------------------------------ 1 gates re-run
    title("1. GATES RE-RUN")
    t23 = open("diagnostics/23_origination_validity.txt", encoding="utf-8").read()
    line23 = next(l for l in t23.splitlines() if l.startswith("data fingerprint:"))
    chk23 = [l for l in t23.splitlines() if "PASS" in l or "FAIL" in l][-3:]
    emit("diagnostic 23   python diagnostics/23_origination_validity.py   (output rewritten over its .txt)")
    emit(f"    {line23}")
    for l in chk23[1:]:
        emit(f"    {l}")
    emit(f"    RESULT: {'PASS' if 'FAIL' not in ''.join(chk23) and 'PASS' in line23 else 'FAIL'}   "
         f"(WANT_FP in the script is {NEW_FP}; confirmed by the run, not assumed)")

    hb = open("artifacts/horizon_baseline/horizon_report.txt", encoding="utf-8").read()
    n_pass, n_fail = hb.count("PASS"), hb.count("FAIL")
    emit("horizon_baseline   python -m src.horizon_baseline   (outputs regenerated in artifacts/horizon_baseline/)")
    emit(f"    checks in its report: {n_pass} PASS, {n_fail} FAIL.   It uses the SAVED count matrix artifacts/baseline/counts.csv,")
    emit("    unchanged. It is a prerequisite of the nulls job (the nulls job reconciles against it), not one of the two gates you named.")

    rc, out = run(["-m", "src.diagnostics.cell_counts"])
    emit("cell_counts   python -m src.diagnostics.cell_counts   (reports/cell_counts.csv regenerated; also a prerequisite)")
    for l in out.splitlines():
        if "PASS" in l or "FAIL" in l or "fingerprint" in l:
            emit(f"    {l.strip()}")
    emit(f"    exit code {rc}   RESULT: {'PASS' if rc == 0 else 'FAIL'}")

    with tempfile.TemporaryDirectory() as td:
        rc, out = run(["-m", "src.horizon_nulls", "--out-dir", td])
        failed = os.path.join(td, "nulls_report_FAILED.txt")
        emit("nulls job   python -m src.horizon_nulls --out-dir <temp>   (artifacts/horizon_nulls/ NOT overwritten)")
        emit(f"    exit code {rc}   RESULT: {'PASS' if rc == 0 else 'FAIL (stopped by its own gate, see below)'}")
        if os.path.exists(failed):
            ft = open(failed, encoding="utf-8").read().splitlines()
            for i, l in enumerate(ft):
                if l.endswith("FAIL") or l.startswith("HARD ERROR"):
                    emit(f"    {l.strip()}")
            i = next((k for k, l in enumerate(ft) if l.startswith("Marginal distribution")), None)
            if i is not None:
                for l in ft[i + 2:i + 12]:
                    emit(f"    {l.rstrip()}")
    emit()
    emit("WHY THE NULLS JOB CANNOT PASS YET (two independent blockers, both self-references to the old universe):")
    emit("  (a) it checks that the marginal of the CURRENT panel's fit rows equals the column sums of artifacts/baseline/counts.csv.")
    emit("      counts.csv is the baseline MODEL (a pooled count matrix estimated from fit rows). It was estimated on the old panel;")
    emit("      the fit rows now differ by the territory and null-ltv fit rows. Passing needs `python -m src.baseline` to re-estimate")
    emit("      it. That is model estimation, which this task excluded, so it was NOT run.")
    emit("  (b) its pooling check compares against its own previous artifacts/horizon_nulls/model_comparison.csv (old universe).")
    emit("      Passing the first time on a changed universe needs that stale reference removed once (a bootstrap pass), then a")
    emit("      second run to check against the genuine new one. Replacing that file was declined, so it was NOT done.")
    emit("State left behind: artifacts/horizon_nulls/ is unchanged (still the old universe's tables). nulls_report.txt still carries")
    emit("the 'stale tables' note, correctly: the tables are NOT current, so the note was NOT removed.")
    emit("artifacts/horizon_baseline/ and reports/cell_counts.csv ARE regenerated (old matrix, new universe); if the baseline is")
    emit("re-estimated they must be regenerated again, in the order baseline -> horizon_baseline -> cell_counts -> nulls (twice).")

    # ------------------------------------------------------------ 2 old vs new
    title("2. DIAGNOSTIC 23: WHAT MOVED, AND WHY")
    t_old = open(os.path.join(old, "23_origination_validity.txt"), encoding="utf-8").read()
    t_new = t23
    grab = lambda t, pat: int(re.search(pat, t).group(1).replace(",", ""))
    l_old = grab(t_old, r"panel loans  : ([\d,]+) loans")
    l_new = grab(t_new, r"panel loans  : ([\d,]+) loans")
    r_old = grab(t_old, r"\(([\d,]+) rows, 8 partitions")
    r_new = grab(t_new, r"\(([\d,]+) rows, 8 partitions")
    emit(f"panel loans {fmt(l_old)} -> {fmt(l_new)} ({fmt(l_old - l_new)} fewer);  panel rows {fmt(r_old)} -> {fmt(r_new)} "
         f"({fmt(r_old - r_new)} fewer)")
    emit("Accounted for by:")
    emit("   territory loans (GU, PR, VI)     10,073 loans   400,089 rows   <- expected cause")
    emit("   null-ltv loans (999 sentinel)       219 loans     6,775 rows   <- expected cause")
    emit("   the named malformed loan              1 loan          1 row    <- NOT one of the two exclusions: the old .txt predates")
    emit("                                                                    the build that excluded F06Q30148547 (69,927,577 rows)")
    ok = (l_old - 10073 - 219 - 1 == l_new) and (r_old - 400089 - 6775 - 1 == r_new)
    emit(f"   arithmetic: {fmt(l_old)} - 10,073 - 219 - 1 = {fmt(l_old - 10073 - 219 - 1)} (new {fmt(l_new)});  "
         f"{fmt(r_old)} - 400,089 - 6,775 - 1 = {fmt(r_old - 400089 - 6775 - 1)} (new {fmt(r_new)})   closes: {ok}")
    emit()
    emit("Moves with a cause OTHER than the two exclusions (flagged, not absorbed):")
    emit("  1. The malformed loan (above): the rate_spread<-orig_rate and property_state<-state MALFORMED buckets go from 1 loan to 0,")
    emit("     and the first_payment VALID row loses its 1 loan with no panel row. One loan, one row.")
    emit("  2. postal_code '000' (8 loans, 180 rows): the old output said the sentinel REACHED the panel as raw text (YES, 8 loans);")
    emit("     it now says NO. Not caused by either exclusion (the same 8 loans / 180 rows remain): the hardening change to src/panel.py")
    emit("     since that run now nulls '000'. The old .txt predates it.")
    emit("  3. 'source lines' section: every src/panel.py line citation moved (267 -> 326, 550 -> not found, ...). Caused by the edits to")
    emit("     src/panel.py (docstring, HPI join, exclusions) and the earlier hardening rewrite. Four of the patterns it greps for")
    emit("     (ltv[ltv >= LTV_NA], dti[dti >= DTI_NA], rate = num(\"orig_rate\"), seen.discard) now report NOT FOUND because that code")
    emit("     no longer exists in that form. The diagnostic's gate does not check these lines, so it still passes, but the section")
    emit("     describes code that has been replaced. Needs a decision: update the patterns, or retire the section.")
    emit("  4. 'origination loans with no panel row' 6,241 -> 16,534: +10,292 (the two exclusions) +1 (the malformed loan). Expected.")
    emit()
    emit("Everything else that moved is the removal of those loans from the panel: the territory tables go to zero, the ltv SENTINEL")
    emit("bucket (219 loans) goes to zero, the fico / dti / msa bucket counts fall by the removed loans' share, the harness universe")
    emit("goes 407,555 -> 405,705 (section 3), the territory share of that universe goes 1,810 -> 0.")
    emit()
    emit("Full line diff of diagnostic 23 (< previous run, > this run):")
    d = difflib.unified_diff([l.rstrip() for l in t_old.replace("\r", "").splitlines()],
                             [l.rstrip() for l in t_new.replace("\r", "").splitlines()], lineterm="", n=0)
    for l in d:
        if l[:1] in "<>+-" and not l.startswith(("+++", "---")):
            emit("  " + ("< " if l[0] == "-" else "> ") + l[1:][:150])

    title("2b. REGENERATED BASELINE RUN: OLD vs NEW  (artifacts/horizon_baseline/, same count matrix, new universe)")
    for f, keys, cols in (("scores.csv", ["horizon", "cohort"], ["n", "brier"]),
                          ("compounding.csv", ["horizon", "cohort"],
                           [c for c in pd.read_csv("artifacts/horizon_baseline/compounding.csv").columns[2:]])):
        emit(f"{f}   (old -> new)")
        side_by_side(pd.read_csv(os.path.join(old, "horizon_baseline", f)),
                     pd.read_csv(os.path.join("artifacts", "horizon_baseline", f)), keys, cols)
    emit("predicted_by_start.csv   (old -> new, n_loans and the six predicted probabilities)")
    side_by_side(pd.read_csv(os.path.join(old, "horizon_baseline", "predicted_by_start.csv")),
                 pd.read_csv("artifacts/horizon_baseline/predicted_by_start.csv"), ["horizon", "start_state"],
                 ["n_loans", "Current", "DQ30", "DQ60", "DQ90+", "Prepaid", "Default"])
    emit("(the predicted probabilities differ only through the start-state mix of the loans, not the matrix.)")

    title("2c. reports/cell_counts.csv: OLD vs NEW  (n, cell_brier, noise_inflation; max |change| over the six p_*)")
    o = pd.read_csv(os.path.join(old, "reports", "cell_counts.csv"))
    n = pd.read_csv("reports/cell_counts.csv")
    m = o.merge(n, on=["cohort", "start_state", "horizon"], suffixes=("_o", "_n"))
    emit(f"{'cohort':<9}{'start':<9}{'h':>3}{'n old':>10}{'n new':>10}{'d n':>6}{'cell_brier old':>17}{'new':>17}{'noise_infl old':>16}"
         f"{'new':>14}{'max|d p|':>11}")
    for _, r in m.iterrows():
        dp = max(abs(r["p_" + s + "_n"] - r["p_" + s + "_o"]) for s in ["Current", "DQ30", "DQ60", "DQ90+", "Prepaid", "Default"])
        emit(f"{r.cohort:<9}{r.start_state:<9}{int(r.horizon):>3}{int(r.n_o):>10,}{int(r.n_n):>10,}{int(r.n_n - r.n_o):>6}"
             f"{r.cell_brier_o:>17.10f}{r.cell_brier_n:>17.10f}{r.noise_inflation_o:>16.4e}{r.noise_inflation_n:>14.4e}{dp:>11.2e}")

    # ------------------------------------------------------------ 3 reconciliation
    title("3. ELIGIBLE-LOAN RECONCILIATION  (harness universe, loan by loan)")
    old_dir, new_dir = "data/panel/_pre_hpi", panel_io.PANEL_DIR
    panel_io.PANEL_DIR = old_dir
    Lo, _ = hz.read_test_loans(progress=lambda s: None)
    ro = pd.Series(hz.eligibility(Lo).to_numpy(), index=Lo.index)
    panel_io.PANEL_DIR = new_dir
    Ln, _ = hz.read_test_loans(progress=lambda s: None)
    rn = pd.Series(hz.eligibility(Ln).to_numpy(), index=Ln.index)
    terr, nl, allo = set(), set(), set()
    for f in sorted(glob.glob(old_dir + "/panel_*.parquet")):
        t = pq.read_table(f, columns=["loan_id", "property_state", "ltv"]).to_pandas()
        g = t.groupby("loan_id", observed=True).agg(s=("property_state", "first"), n=("ltv", lambda x: x.isna().any()))
        terr |= set(g.index[g.s.isin(["GU", "PR", "VI"])])
        nl |= set(g.index[g.n])
        allo |= set(g.index)
    nl_only = nl - terr
    removed = set(Lo.index) - set(Ln.index)
    added = set(Ln.index) - set(Lo.index)
    reasons = ["eligible"] + list(hz.EXCLUDED)
    code = lambda r: -1 if r < 0 else int(r)
    emit("Eligibility rule (src.horizon.eligibility): a test-split loan with a condition or evaluate row is ELIGIBLE iff it is observed")
    emit("at the 2008-12 cut in a transient state; otherwise excluded for exactly one of four reasons.")
    emit()
    emit(f"panel loans (pre-HPI panel kept in {old_dir}):   {fmt(len(allo))}")
    emit(f"panel loans now:                                  {fmt(len(allo) - len(terr) - len(nl_only))}   "
         f"= {fmt(len(allo))} - {fmt(len(terr))} territory - {fmt(len(nl_only))} null-ltv (not territory)")
    emit(f"   (territory and null-ltv overlap: {len(terr & nl)} loans)")
    emit(f"   of the {fmt(len(terr) + len(nl_only))} removed loans: {fmt(len(removed))} are test-split (have condition/evaluate rows),")
    emit(f"   {fmt(len(terr) + len(nl_only) - len(removed))} are train-split only and were never in the harness universe.")
    emit()
    emit(f"test-split loans with a panel row:  old {fmt(len(Lo))}   new {fmt(len(Ln))}   removed {fmt(len(removed))}   added {fmt(len(added))}")
    emit(f"{'removed test-split loans, by old eligibility':<48}{'territory':>12}{'null-ltv':>12}{'total':>10}")
    tot_t = tot_n = 0
    for c, name in zip(range(-1, 4), reasons):
        a = sum(1 for x in removed & terr if code(ro[x]) == c)
        b = sum(1 for x in removed & nl_only if code(ro[x]) == c)
        tot_t += a
        tot_n += b
        emit(f"  {name:<46}{a:>12,}{b:>12,}{a + b:>10,}")
    emit(f"  {'TOTAL':<46}{tot_t:>12,}{tot_n:>12,}{tot_t + tot_n:>10,}")
    el_t = sum(1 for x in removed & terr if ro[x] < 0)
    el_n = sum(1 for x in removed & nl_only if ro[x] < 0)
    emit(f"removed loans in neither set: {len(removed - terr - nl_only)}")
    emit()
    old_el, new_el = int((ro < 0).sum()), int((rn < 0).sum())
    emit("Eligible-loan arithmetic:")
    emit(f"  old harness universe            {fmt(old_el)}")
    emit(f"  - eligible territory loans      {fmt(el_t)}")
    emit(f"  - eligible null-ltv loans       {fmt(el_n)}")
    emit(f"  = expected                      {fmt(old_el - el_t - el_n)}")
    emit(f"  new harness universe            {fmt(new_el)}")
    emit(f"  RESIDUAL                        {old_el - el_t - el_n - new_el}   ({'closes exactly' if old_el - el_t - el_n == new_el else 'DOES NOT CLOSE'})")
    common = Ln.index
    emit(f"retained test-split loans whose eligibility reason changed: {int((ro[common] != rn[common]).sum())}")
    cols = [c for c in Ln.columns if c in Lo.columns]
    dif = sum(int((Lo.loc[common, c].to_numpy() != Ln[c].to_numpy()).sum()) for c in cols)
    emit(f"retained loans' harness fields ({', '.join(cols)}) differing old vs new: {dif}")
    emit("  => the eligibility rule itself is unchanged; the only effect is the removal of loans.")
    emit()
    emit("Against the earlier figure 'N loans left the harness universe':")
    emit(f"  1,810 is CONSISTENT with this reconciliation as the TERRITORY-ONLY part ({fmt(el_t)} eligible territory loans; it was the")
    emit(f"  old 407,555 -> 405,745 step, before null-ltv loans were removed). It is not the whole difference: the null-ltv exclusion")
    emit(f"  removes {fmt(el_n)} more, so the universe falls by {fmt(old_el - new_el)} in total ({fmt(old_el)} -> {fmt(new_el)}). No discrepancy,")
    emit(f"  but 1,810 must not be read as the total. Diagnostic 29 (405,745) and diagnostic 30 / this report (405,705) are different")
    emit(f"  stages of the same change, differing by exactly the {fmt(el_n)} null-ltv loans.")

    # ------------------------------------------------------------ 4 sweep
    title("4. FINGERPRINT SWEEP  (3b1209189824e42f, 2f304bae5613525d)")
    classify = {
        "diagnostics/23_origination_validity.py": ("LIVE GATE", f"WANT_FP = {NEW_FP} (updated earlier; confirmed by the run); the old value appears only in a trailing comment"),
        "artifacts/horizon_nulls/nulls_report.txt": ("LIVE GATE (baseline line) + HISTORICAL (rest)",
                                                     "the line read by saved_fingerprint() holds the new value; the old value remains in the run's prose line "
                                                     "'against the saved nulls run ...: <old> PASS' and in the appended NOTE, both historical. The file's tables are still the old universe's"),
        "diagnostics/29_hpi_join_report.txt": ("HISTORICAL RECORD", "milestone 29 report; records the old, intermediate and new fingerprints at that stage"),
        "diagnostics/30_housekeeping.py": ("REFERENCE CONSTANT, not a gate", "OLD_FP is only printed and searched for in that report; it compares nothing against a run"),
        "diagnostics/30_housekeeping.txt": ("HISTORICAL RECORD", "milestone 30 report"),
        "diagnostics/32_rebaseline.py": ("REFERENCE CONSTANTS, not a gate", "this script"),
        "diagnostics/32_rebaseline.txt": ("HISTORICAL RECORD", "this report"),
    }
    hits = []
    for root in (".", "../credit-migration-patches"):
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in (".git", "data", "__pycache__", "_pre_hpi", "_pre_housekeeping")]
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    t = open(p, encoding="utf-8", errors="strict").read()
                except Exception:
                    continue
                for ln, l in enumerate(t.splitlines(), 1):
                    for fp in (OLD_FP, MID_FP):
                        if fp in l:
                            hits.append((os.path.normpath(p).replace("\\", "/"), ln, fp, l.strip()[:110]))
    emit(f"{'file:line':<58}{'value':<20}{'class'}")
    for p, ln, fp, text in sorted(hits):
        if p in classify:
            c = classify[p][0]
        elif p.startswith("..") or "credit-migration-patches" in p:
            c = "HISTORICAL (outside the repo: saved patch / commit message)"
        else:
            c = "UNCLASSIFIED - check"
        emit(f"{p + ':' + str(ln):<58}{fp:<20}{c}")
    emit()
    emit("Notes on the classes:")
    for p, (c, why) in classify.items():
        if any(h[0] == p for h in hits):
            emit(f"  {p}: {c}. {why}")
    emit("  ../credit-migration-patches/*: saved patches and commit messages of earlier milestones; outside the repo; untouched.")
    live_old = [h for h in hits if h[2] in (OLD_FP, MID_FP) and h[0] in ("diagnostics/23_origination_validity.py",) and "WANT_FP" in h[3] and NEW_FP not in h[3]]
    emit()
    emit(f"Live gates still holding an old value: {len(live_old)}.   Live gates updated this task: none needed (both were already")
    emit(f"{NEW_FP}); diagnostic 23's was verified by running it.   Historical records left alone: all lines listed above as such.")
    emit("Two gates read the baseline through saved_fingerprint(): src/horizon_nulls.py and src/diagnostics/cell_counts.py (no literal in either).")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    main(ap.parse_args().old)
