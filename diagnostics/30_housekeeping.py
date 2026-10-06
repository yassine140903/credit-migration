"""Panel housekeeping report: exclusions, final shape, reproducibility, fingerprint baselines.   READ-ONLY.

    python diagnostics/30_housekeeping.py --scratch DIR   # from the repo root

DIR is a scratch rebuild made with  python -m src.panel --out-dir DIR  . Writes
diagnostics/30_housekeeping.txt and a copy of the scratch build report to
diagnostics/30_panel_build_report.txt. It reads the committed panel (data/panel/)
and DIR, changes neither, and runs no git command.

REPRODUCIBILITY is compared on: row count, loan count (per partition and total),
column set / order / Arrow types, and the harness fingerprint (src.horizon_nulls);
and, beyond what was asked, on every value of every partition and of the censoring
files (Arrow / DataFrame equality). Any difference is reported, never repaired.
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import horizon as hz, panel                                      # noqa: E402
from src import panel_io                                                  # noqa: E402
from src.horizon_nulls import fingerprint                                 # noqa: E402

OUT = os.path.join("diagnostics", "30_housekeeping.txt")
BUILD_COPY = os.path.join("diagnostics", "30_panel_build_report.txt")
OLD_FP = "3b1209189824e42f"
RULE_W = 118
LINES = []


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def rule(c="-"):
    emit(c * RULE_W)


def title(t):
    emit()
    rule("=")
    emit(t)
    rule("=")


def fmt(n):
    return f"{int(n):,}"


def harness_fingerprint():
    L, _ = hz.read_test_loans(progress=lambda s: None)
    E = L[hz.eligibility(L).to_numpy() < 0]
    real = hz.realised(E)
    _, fixed = hz.cohorts(real)
    fp = fingerprint(E.index.to_numpy(), [E["state_cut"].to_numpy()] + [real[h] for h in hz.HORIZONS] + [fixed])
    return fp, len(E)


def partition_stats(path):
    t = pq.read_table(path)
    enc = t.column("loan_id").combine_chunks().dictionary_encode()
    return t, t.num_rows, len(enc.dictionary)


def grab(text, label):
    """The numbers on the first report line starting with `label`."""
    for ln in text.splitlines():
        if ln.startswith(label):
            return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", ln[len(label):])]
    raise SystemExit("cannot find %r in the scratch build report" % label)


def main(scratch):
    live = panel.PANEL_DIR
    build_report = open(os.path.join(scratch, "build_report.txt"), encoding="utf-8").read()
    shutil.copy2(os.path.join(scratch, "build_report.txt"), BUILD_COPY)
    log_path = os.path.join(live, "_pre_housekeeping", "transform_log.txt")
    emit("Panel housekeeping: restore reproducibility")
    emit(f"committed panel: {live}      scratch rebuild: {scratch}")

    # ----------------------------------------------------------- 1 exclusions
    title("1. COUNTS REMOVED AT EACH EXCLUSION")
    t_l, t_r = grab(build_report, "territory (GU, PR, VI)")
    n_l, n_r = grab(build_report, "null ltv (not already a territory loan)")
    c_t = grab(build_report, "removed from data/panel/censoring/: territory")[0]
    c_n = grab(build_report, "removed from data/panel/censoring/: null ltv")[0]
    emit("From the scratch build (src.panel), in build order:")
    emit(f"  {'exclusion':<58}{'loans':>12}{'rows':>14}")
    emit(f"  {'1. territory loans (GU, PR, VI)':<58}{fmt(t_l):>12}{fmt(t_r):>14}")
    emit(f"  {'2. null-ltv loans (999 sentinel), not already territory':<58}{fmt(n_l):>12}{fmt(n_r):>14}")
    emit(f"  {'(named malformed loan F06Q30148547, unchanged)':<58}{'1':>12}{'1':>14}")
    emit(f"  {'3. hpi_feature column dropped (100% null before)':<58}{'':>12}{'':>14}")
    emit(f"  censoring files: territory loans removed {fmt(c_t)}, null-ltv loans removed {fmt(c_n)}")
    emit("  Same counts on the committed panel (one-off transform log, data/panel/_pre_housekeeping/): see appendix.")
    emit(f"  on record before: territory 10,073 loans / 400,089 rows; null-ltv 219 loans / 6,775 rows.   "
         f"match: {(t_l, t_r, n_l, n_r) == (10073, 400089, 219, 6775)}")
    assert (t_l, t_r, n_l, n_r) == (10073, 400089, 219, 6775)

    # ------------------------------------------------------- 2/3 committed shape
    title("2. FINAL ROW AND LOAN COUNTS  (committed panel)")
    man = panel_io.read_manifest()
    panel_io.validate(man)
    rows = loans = 0
    emit(f"{'partition':<10}{'rows':>14}{'loans':>12}")
    for p in man["partitions"]:
        _, r, l = partition_stats(os.path.join(live, p["file"]))
        rows += r
        loans += l
        emit(f"{p['quarter']:<10}{fmt(r):>14}{fmt(l):>12}")
    rule()
    emit(f"{'TOTAL':<10}{fmt(rows):>14}{fmt(loans):>12}   (manifest rows {fmt(man['total_rows'])}; validated by load_panel)")

    title("3. FINAL COLUMNS BY GROUP  (committed panel manifest)")
    types = {c["name"]: c["type"] for c in man["schema"]}
    groups = [("keys", panel.KEYS), ("states", panel.STATE_COLS), ("features", panel.FEATURES),
              ("metadata", panel.METADATA)]
    assert list(types) == panel.SCHEMA, "manifest columns differ from src.panel.SCHEMA"
    for g, cols in groups:
        emit(f"{g} ({len(cols)}):")
        for c in cols:
            emit(f"    {c:<28}{types[c]}")
    emit(f"total {len(types)} columns; order equals src.panel.SCHEMA: True; hpi_feature present: {'hpi_feature' in types}")
    emit(f"HPI features (features group, not metadata): {panel.HPI_FEATURES}")

    # ------------------------------------------------------- 4 reproducibility
    title("4. REPRODUCIBILITY: scratch rebuild vs committed panel")
    sman = panel_io_manifest(scratch)
    diffs = []
    emit(f"{'check':<44}{'committed':>22}{'scratch':>22}  result")

    def row(name, a, b):
        ok = a == b
        if not ok:
            diffs.append(name)
        if isinstance(a, list):
            a, b = (f"{len(a)} items", f"{len(b)} items") if ok else (a, b)
        emit(f"{name:<44}{str(a):>22}{str(b):>22}  {'MATCH' if ok else 'DIFFERENT'}")

    row("partitions", man["n_partitions"], sman["n_partitions"])
    row("rows (manifest)", fmt(man["total_rows"]), fmt(sman["total_rows"]))
    s_rows = s_loans = 0
    tables_equal = {}
    for p in man["partitions"]:
        tc, rc, lc = partition_stats(os.path.join(live, p["file"]))
        ts, rs, ls = partition_stats(os.path.join(scratch, p["file"]))
        s_rows += rs
        s_loans += ls
        row(f"  {p['quarter']} rows / loans", f"{fmt(rc)} / {fmt(lc)}", f"{fmt(rs)} / {fmt(ls)}")
        tables_equal[p["quarter"]] = (tc.schema.equals(ts.schema, check_metadata=False)
                                      and tc.equals(ts))
        del tc, ts
    row("rows (counted from the files)", fmt(rows), fmt(s_rows))
    row("loans", fmt(loans), fmt(s_loans))
    row("column set", sorted(types), sorted(c["name"] for c in sman["schema"]))
    row("column order", list(types), [c["name"] for c in sman["schema"]])
    row("column types", [types[c] for c in types], [c["type"] for c in sman["schema"]])
    # fingerprint: the committed panel through load_panel; the scratch one by pointing panel_io at it
    fp_c, n_c = harness_fingerprint()
    saved = panel_io.PANEL_DIR
    panel_io.PANEL_DIR = scratch
    try:
        fp_s, n_s = harness_fingerprint()
    finally:
        panel_io.PANEL_DIR = saved
    row("harness fingerprint", fp_c, fp_s)
    row("eligible loans in the harness universe", fmt(n_c), fmt(n_s))
    emit()
    emit("Beyond what was asked: every value of every column of every partition compared (Arrow equality, schema included):")
    for q, ok in tables_equal.items():
        emit(f"    {q}  {'IDENTICAL' if ok else 'DIFFERENT'}")
        if not ok:
            diffs.append(f"values {q}")
    cens_ok = True
    for f in sorted(glob.glob(os.path.join(live, "censoring", "censor_*.parquet"))):
        g = os.path.join(scratch, "censoring", os.path.basename(f))
        same = os.path.exists(g) and pd.read_parquet(f).equals(pd.read_parquet(g))
        cens_ok &= same
        emit(f"    censoring/{os.path.basename(f)}  {'IDENTICAL' if same else 'DIFFERENT'}")
    if not cens_ok:
        diffs.append("censoring files")
    ident = [ln for ln in build_report.splitlines() if "identity holds" in ln or "== written" in ln]
    emit()
    emit("Scratch build's own arithmetic identity lines:")
    for ln in ident:
        emit("    " + ln.strip())
    if any(not ln.rstrip().endswith("yes") for ln in ident):
        diffs.append("arithmetic identity")
    emit()
    emit("RESULT: " + ("CLEAN. Every compared item matches exactly. The committed panel is what python -m src.panel produces."
                       if not diffs else "NOT CLEAN: " + ", ".join(diffs)))
    emit(f"scratch build report copied to {BUILD_COPY}")
    if diffs:
        emit("The committed panel was NOT replaced.")

    # ---------------------------------------------------- 5 fingerprint baselines
    title("5. FINGERPRINT BASELINES")
    emit(f"old fingerprint {OLD_FP}   new fingerprint {fp_c}   ({n_c} eligible loans)")
    emit("(the old universe, recorded in diagnostics/29, had 407,555 eligible loans)")
    emit()
    emit(f"{'file':<52}{'contains old':>14}{'contains new':>14}")
    for f in ["diagnostics/23_origination_validity.py", "artifacts/horizon_nulls/nulls_report.txt",
              "diagnostics/23_origination_validity.txt", "diagnostics/29_hpi_join_report.txt"]:
        t = open(f, encoding="utf-8").read()
        emit(f"{f:<52}{str(OLD_FP in t):>14}{str(fp_c in t):>14}")
    emit("src/horizon_nulls.py and src/diagnostics/cell_counts.py hold no literal: both read the baseline from "
         "artifacts/horizon_nulls/nulls_report.txt (saved_fingerprint()).")
    emit()
    emit("FILES CHANGED (expected value only; no gate's logic touched):")
    emit(f"  diagnostics/23_origination_validity.py   WANT_FP   {OLD_FP} -> {fp_c}")
    emit(f"  artifacts/horizon_nulls/nulls_report.txt  the baseline line read by saved_fingerprint()   {OLD_FP} -> {fp_c}")
    emit("      plus a NOTE appended at the end of that file saying so and that its tables are stale until the run is repeated.")
    emit("      saved_fingerprint() now returns: " + __import__("src.horizon_nulls", fromlist=["x"]).saved_fingerprint())
    emit("LEFT AS HISTORICAL RECORDS (outputs of earlier runs, not gates; rewriting them would misstate what was run):")
    emit(f"  diagnostics/23_origination_validity.txt, diagnostics/29_hpi_join_report.txt, and the prose line")
    emit(f"  'against the saved nulls run ...: {OLD_FP}   PASS' inside nulls_report.txt.")
    emit("NOT RE-RUN: diagnostic 23 (it rewrites its own .txt and its territory sections assume territories in the panel) and")
    emit("src.horizon_nulls (it compares against the saved model_comparison.csv / cell_counts.csv, computed on the old universe).")

    # ----------------------------------------------------------- appendix
    if os.path.exists(log_path):
        title("APPENDIX. ONE-OFF TRANSFORM OF THE COMMITTED PANEL (steps 1-3; log kept in data/panel/_pre_housekeeping/)")
        for ln in open(log_path, encoding="utf-8").read().splitlines():
            emit(ln)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")
    return 0 if not diffs else 1


def panel_io_manifest(root):
    import json
    return json.load(open(os.path.join(root, panel.MANIFEST_NAME), encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    sys.exit(main(ap.parse_args().scratch))
