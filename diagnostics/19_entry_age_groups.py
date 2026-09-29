"""Early- versus late-entering loans, and the entry-age anomalies of diagnostic 18.

GROUPS
    entry age   minimum loan_age over the loan's panel rows, as in diagnostic 18
    early       entry age 0-2
    late        entry age 3 or above
    Unit: loans with at least one panel row.

PART 1  FICO, original LTV, DTI and rate spread, from the panel. Each is an
        origination attribute joined per loan, so it is taken from the loan's
        first panel row. Mean, median, non-null count per group, and the raw
        difference late minus early. No significance tests: at this sample size
        they decide nothing.

PART 2  Channel and seller name, from the origination files (full files, no
        sampling), joined to the panel's loans on loan_id. Channel is shown as
        the raw code in the file. Sellers: top 10 by loan count within each
        group, and the share those ten hold.

PART 3  The loans whose first panel row's loan_age is not their minimum. For
        those loans only: whether loan_age goes backwards down the rows, how
        often and by how much, and whether the rows are in period order - both
        in the panel as written and in the source performance files as stored
        (the panel is sorted by construction, so only the source order says
        anything about how the data arrived).

Plain figures only. No interpretation of whether any difference matters.

Column indices are resolved at run time from the published layout by attribute
name and printed in the header. Nothing is imported from src/. Nothing is
written except the report. No git command is run.
"""

import glob
import os
import re
import time
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

PERF_GLOB = "data/historical_data_200[67]/*/perf_*.txt"
ORIG_GLOB = "data/historical_data_200[67]/*/orig_*.txt"
PANEL_GLOB = os.path.join("data", "panel", "panel_*.parquet")
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")
OUT = os.path.join("diagnostics", "19_entry_age_groups.txt")

CHUNKSIZE = 2_000_000
RULE_W = 121
EARLY_MAX = 2                           # early: entry age 0..EARLY_MAX
TOP_SELLERS = 10

PERF_SHEET = "monthly performance data file"
ORIG_SHEET = "origination data file"
PERF_WANTED = [("loan", "loan identifier"), ("period", "period"),
               ("loan_age", "loan age")]
ORIG_WANTED = [("loan", "loan identifier"), ("channel", "channel"),
               ("seller", "seller name")]
PERF_EXPECTED = {"loan": 0, "period": 1}

GROUPS = ["early", "late"]
FEATURES = [("fico", "FICO"), ("ltv", "original LTV"), ("dti", "DTI"),
            ("rate_spread", "rate spread")]

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


# ==========================================================================
# layout
# ==========================================================================
def _sheets(path):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

    def cell(c):
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(ns + "t"))
        v = c.find(ns + "v")
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    out = {}
    for name in sorted(n for n in z.namelist()
                       if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
        rows = [[cell(c) for c in row.iter(ns + "c")]
                for row in ET.fromstring(z.read(name)).iter(ns + "row")]
        head = " ".join(" ".join(r) for r in rows[:6]).lower()
        fields = [(int(cs_[0]), cs_[1]) for cs_ in
                  ([x.strip() for x in r if x.strip()] for r in rows)
                  if len(cs_) >= 2 and re.match(r"^\d+$", cs_[0])]
        for marker in (PERF_SHEET, ORIG_SHEET):
            if marker in head and fields:
                out[marker] = (fields, name)
    return out


def _norm(t):
    t = t.strip().lower().replace("®", "").replace("™", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()/%.-]+", " ", t)).strip()


def resolve():
    sheets = _sheets(LAYOUT)

    def pick(marker, wanted):
        fields, sheet = sheets[marker]
        got = {}
        for label, want in wanted:
            w = _norm(want)
            hits = [f for f in fields if _norm(f[1]) == w] or \
                   [f for f in fields if w in _norm(f[1])]
            if len(hits) != 1:
                raise SystemExit("layout lookup for %r on %s matched %d row(s)"
                                 % (want, marker, len(hits)))
            got[label] = (hits[0][0] - 1, hits[0][0], hits[0][1])
        return got, sheet

    perf, perf_sheet = pick(PERF_SHEET, PERF_WANTED)
    orig, orig_sheet = pick(ORIG_SHEET, ORIG_WANTED)
    for label, expect in PERF_EXPECTED.items():
        if perf[label][0] != expect:
            raise SystemExit("layout gives %d for performance %r, project uses %d"
                             % (perf[label][0], label, expect))
    return perf, orig, perf_sheet, orig_sheet


# ==========================================================================
# panel pass
# ==========================================================================
def process_partition(path):
    """Per-loan frame (id, entry age, first-row age, features) and, for the
    anomaly loans, their panel rows."""
    cols = ["loan_id", "origin_period", "loan_age"] + [f for f, _ in FEATURES]
    t = pq.read_table(path, columns=cols)
    n = t.num_rows
    ids = t["loan_id"].combine_chunks()
    head, tail = ids.slice(0, n - 1), ids.slice(1)
    if pc.any(pc.less(tail, head)).as_py():
        raise SystemExit("%s is not sorted by loan_id" % os.path.basename(path))
    change = pc.not_equal(tail, head).to_numpy(zero_copy_only=False)
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    loan_of_row = np.repeat(np.arange(len(starts)),
                            np.diff(np.append(starts, n)))

    if t["loan_age"].null_count:
        raise SystemExit("%s has null loan_age; diagnostic 18 found none"
                         % os.path.basename(path))
    age = t["loan_age"].to_numpy().astype(np.int64)
    period = t["origin_period"].to_numpy().astype(np.int64)
    entry = np.minimum.reduceat(age, starts)
    first = age[starts]

    loans = pd.DataFrame({
        "loan_id": ids.take(starts).to_pandas().to_numpy(),
        "entry_age": entry,
    })
    for f, _ in FEATURES:
        loans[f] = t[f].take(starts).to_numpy(zero_copy_only=False)

    anomalous = first != entry
    rows = None
    if anomalous.any():
        mask = anomalous[loan_of_row]
        rows = pd.DataFrame({"loan_id": ids.filter(mask).to_pandas().to_numpy(),
                             "period": period[mask], "loan_age": age[mask]})
    return loans, rows


def steps(frame, order_by_period):
    """Within-loan consecutive steps of a (loan_id, period, loan_age) frame.

    Rows are taken in the frame's own order unless order_by_period.
    """
    if order_by_period:
        frame = frame.sort_values(["loan_id", "period"], kind="stable")
    lid = frame["loan_id"].to_numpy()
    same = lid[1:] == lid[:-1]
    d_age = np.diff(frame["loan_age"].to_numpy())[same]
    d_per = np.diff(frame["period"].to_numpy())[same]
    loan = lid[1:][same]
    return loan, d_age, d_per


def describe_steps(label, loan, d_age, d_per, n_loans):
    back = d_age < 0
    emit("%-66s %18s" % ("%s: within-loan consecutive steps" % label,
                         fmt(len(d_age))))
    emit("%-66s %18s" % ("    steps where period does NOT increase",
                         fmt(int((d_per <= 0).sum()))))
    emit("%-66s %18s" % ("    loans with any such step",
                         fmt(len(np.unique(loan[d_per <= 0])))))
    emit("%-66s %18s" % ("    steps where loan_age goes backwards",
                         fmt(int(back.sum()))))
    emit("%-66s %18s %12s" % ("    loans with at least one backward step",
                              fmt(len(np.unique(loan[back]))),
                              pct(len(np.unique(loan[back])), n_loans)))
    emit("%-66s %18s" % ("    loans whose loan_age is non-decreasing throughout",
                         fmt(n_loans - len(np.unique(loan[back])))))
    if back.any():
        emit("%-66s %18d" % ("    largest backward step (months of loan_age)",
                             int(-d_age[back].min())))
        emit("%-66s %18s" % ("    backward steps with period advancing by exactly 1",
                             fmt(int((back & (d_per == 1)).sum()))))


# ==========================================================================
# source-file pass for the anomaly loans
# ==========================================================================
def source_rows(perf_paths, pidx, wanted):
    """Every source row of the wanted loans, in each file's stored order."""
    c_loan, c_per, c_age = (pidx["loan"][0], pidx["period"][0],
                            pidx["loan_age"][0])
    cols = sorted({c_loan, c_per, c_age})
    prefixes = {w[:5] for w in wanted}
    out = []
    for path in perf_paths:
        # Each perf file carries one loan-id prefix (src/panel.py enforces it
        # during the build); skip files holding none of the wanted loans.
        first = pd.read_csv(path, sep="|", header=None, dtype=str, nrows=1,
                            usecols=[c_loan], na_filter=False)
        if first.iloc[0, 0].strip()[:5] not in prefixes:
            print("    skipped %s (no wanted prefix)" % os.path.basename(path),
                  flush=True)
            continue
        offset = 0
        for chunk in pd.read_csv(path, sep="|", header=None, dtype=str,
                                 usecols=cols, chunksize=CHUNKSIZE, engine="c",
                                 na_filter=False):
            loan = chunk[c_loan].str.strip()
            hit = loan.isin(wanted).to_numpy()
            if hit.any():
                per = pd.to_numeric(chunk[c_per].str.strip()[hit],
                                    errors="coerce")
                idx = (per // 100) * 12 + (per % 100 - 1)
                out.append(pd.DataFrame({
                    "loan_id": loan[hit].to_numpy(),
                    "period": idx.to_numpy(),
                    "loan_age": pd.to_numeric(chunk[c_age].str.strip()[hit],
                                              errors="coerce").to_numpy(),
                    "pos": offset + np.nonzero(hit)[0],
                }))
            offset += len(chunk)
        print("    scanned %s" % os.path.basename(path), flush=True)
    frame = pd.concat(out, ignore_index=True)
    return frame


# ==========================================================================
def main():
    t0 = time.time()
    pidx, oidx, perf_sheet, orig_sheet = resolve()
    panel_paths = sorted(glob.glob(PANEL_GLOB))
    perf_paths = sorted(glob.glob(PERF_GLOB))
    orig_paths = sorted(glob.glob(ORIG_GLOB))

    rule("=")
    emit("19_entry_age_groups - early- versus late-entering loans, and the")
    emit("                      entry-age anomalies of diagnostic 18")
    rule("=")
    emit("panel       : %s" % PANEL_GLOB)
    emit("performance : %s" % PERF_GLOB)
    emit("origination : %s" % ORIG_GLOB)
    emit("groups      : early = entry age 0-%d, late = entry age %d+"
         % (EARLY_MAX, EARLY_MAX + 1))
    emit()
    emit("RESOLVED COLUMN INDICES   (0-based index = published position - 1)")
    rule()
    emit("%-16s%-16s%10s%8s   %s" % ("file", "role", "position", "index",
                                     "attribute name"))
    rule()
    for label, _ in PERF_WANTED:
        i, pos, attribute = pidx[label]
        emit("%-16s%-16s%10d%8d   %s" % ("performance", label, pos, i, attribute))
    for label, _ in ORIG_WANTED:
        i, pos, attribute = oidx[label]
        emit("%-16s%-16s%10d%8d   %s" % ("origination", label, pos, i, attribute))
    rule()
    emit("performance sheet: %s      origination sheet: %s"
         % (perf_sheet, orig_sheet))
    emit()

    loan_parts, anomaly_parts = [], []
    for p in panel_paths:
        loans, rows = process_partition(p)
        loan_parts.append(loans)
        if rows is not None:
            anomaly_parts.append(rows)
    loans = pd.concat(loan_parts, ignore_index=True)
    del loan_parts
    if loans["loan_id"].duplicated().any():
        raise SystemExit("a loan_id appears in more than one partition")
    loans["group"] = np.where(loans["entry_age"] <= EARLY_MAX, 0, 1)
    n_all = len(loans)
    n_g = [int((loans["group"] == g).sum()) for g in range(2)]

    # ---- groups ----------------------------------------------------------
    rule("=")
    emit("GROUPS   (unit: loans)")
    rule("=")
    emit("%-40s%18s%12s" % ("group", "loans", "share"))
    rule()
    for g, name in enumerate(GROUPS):
        emit("%-40s%18s%12s" % ("%s (entry age %s)"
                                % (name, "0-%d" % EARLY_MAX if g == 0
                                   else "%d+" % (EARLY_MAX + 1)),
                                fmt(n_g[g]), pct(n_g[g], n_all)))
    rule()
    emit("%-40s%18s%12s" % ("TOTAL", fmt(n_all), pct(sum(n_g), n_all)))
    emit("ties to diagnostic 18: 2,290,179 at entry age 0-2 of 2,425,495 loans.")
    emit()

    # ---- part 1 ------------------------------------------------------------
    rule("=")
    emit("PART 1 - CONTINUOUS CHARACTERISTICS   (unit: loans, first panel row)")
    rule("=")
    emit("%-14s%-8s%14s%12s%12s%14s%12s" % ("feature", "group", "non-null",
                                            "of group", "mean", "median",
                                            "null"))
    rule()
    diffs = []
    for f, label in FEATURES:
        means, medians = [], []
        for g, name in enumerate(GROUPS):
            v = loans.loc[loans["group"] == g, f].to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            m = float(v[ok].mean()) if ok.any() else float("nan")
            md = float(np.median(v[ok])) if ok.any() else float("nan")
            means.append(m)
            medians.append(md)
            emit("%-14s%-8s%14s%12s%12.3f%14.3f%12s"
                 % (label if g == 0 else "", name, fmt(ok.sum()),
                    pct(int(ok.sum()), len(v), 2), m, md, fmt((~ok).sum())))
        diffs.append((label, means[1] - means[0], medians[1] - medians[0]))
        rule()
    emit()
    emit("%-24s%22s%22s" % ("feature", "mean: late - early",
                            "median: late - early"))
    rule()
    for label, dm, dmd in diffs:
        emit("%-24s%22.3f%22.3f" % (label, dm, dmd))
    rule()
    emit("Units as stored in the panel: FICO points, LTV and DTI in percent,")
    emit("rate spread in percentage points. Means and medians over non-null values.")
    emit()

    # ---- part 2: join origination -------------------------------------------
    c_loan, c_ch, c_se = oidx["loan"][0], oidx["channel"][0], oidx["seller"][0]
    orig_parts = []
    for path in orig_paths:
        f = pd.read_csv(path, sep="|", header=None, dtype=str,
                        usecols=sorted({c_loan, c_ch, c_se}), engine="c",
                        na_filter=False)
        orig_parts.append(pd.DataFrame({
            "loan_id": f[c_loan].str.strip().to_numpy(),
            "channel": f[c_ch].str.strip().to_numpy(),
            "seller": f[c_se].str.strip().to_numpy(),
        }))
    orig = pd.concat(orig_parts, ignore_index=True)
    del orig_parts
    n_orig = len(orig)
    dup_orig = int(orig["loan_id"].duplicated().sum())
    if dup_orig:
        raise SystemExit("%d duplicated loan_id values in origination files"
                         % dup_orig)
    joined = loans[["loan_id", "group"]].merge(orig, on="loan_id", how="left",
                                               validate="one_to_one")
    unmatched = joined["channel"].isna()

    rule("=")
    emit("PART 2 - CHANNEL AND SELLER   (unit: loans, joined on loan_id)")
    rule("=")
    emit("%-58s %18s" % ("origination records read (all files, full)",
                         fmt(n_orig)))
    emit("%-58s %18s" % ("panel loans", fmt(n_all)))
    emit("%-58s %18s" % ("panel loans matched to an origination record",
                         fmt(int((~unmatched).sum()))))
    for g, name in enumerate(GROUPS):
        k = int((unmatched & (joined["group"] == g)).sum())
        emit("%-58s %18s" % ("panel loans with no match, %s" % name, fmt(k)))
    emit()
    joined = joined[~unmatched]
    n_j = [int((joined["group"] == g).sum()) for g in range(2)]

    emit("CHANNEL   (raw code as it appears in the origination file)")
    rule()
    emit("%-14s%18s%12s%18s%12s%16s" % ("channel", "early", "share", "late",
                                        "share", "late - early"))
    rule()
    ct = pd.crosstab(joined["channel"], joined["group"]).reindex(
        columns=[0, 1], fill_value=0)
    for code in sorted(ct.index, key=lambda c: (-int(ct.loc[c].sum()), c)):
        e, l_ = int(ct.loc[code, 0]), int(ct.loc[code, 1])
        se = 100.0 * e / n_j[0] if n_j[0] else float("nan")
        sl = 100.0 * l_ / n_j[1] if n_j[1] else float("nan")
        emit("%-14s%18s%12s%18s%12s%15.4f%%"
             % (repr(code) if code == "" else code, fmt(e), pct(e, n_j[0]),
                fmt(l_), pct(l_, n_j[1]), sl - se))
    rule()
    emit("%-14s%18s%12s%18s%12s" % ("TOTAL", fmt(n_j[0]), pct(n_j[0], n_j[0]),
                                    fmt(n_j[1]), pct(n_j[1], n_j[1])))
    emit("late - early is the difference in share, percentage points.")
    emit()

    emit("SELLER   (top %d by loan count within each group)" % TOP_SELLERS)
    for g, name in enumerate(GROUPS):
        vc = joined.loc[joined["group"] == g, "seller"].value_counts()
        rule()
        emit("%-6s%-58s%18s%12s" % ("rank", "%s - seller name" % name, "loans",
                                    "share"))
        rule()
        for r, (seller, c) in enumerate(vc.head(TOP_SELLERS).items(), 1):
            emit("%-6d%-58s%18s%12s" % (r, seller[:56] if seller else "''",
                                        fmt(c), pct(int(c), n_j[g])))
        rule()
        top = int(vc.head(TOP_SELLERS).sum())
        emit("%-64s%18s%12s" % ("top %d combined" % TOP_SELLERS, fmt(top),
                                pct(top, n_j[g])))
        emit("%-64s%18s" % ("distinct seller names in group", fmt(len(vc))))
        emit()
    emit("CONCENTRATION")
    rule()
    for g, name in enumerate(GROUPS):
        vc = joined.loc[joined["group"] == g, "seller"].value_counts()
        emit("%-58s %18s" % ("%s: share held by the top %d sellers"
                             % (name, TOP_SELLERS),
                             pct(int(vc.head(TOP_SELLERS).sum()), n_j[g])))
    emit()
    del joined, orig

    # ---- part 3 ------------------------------------------------------------
    rule("=")
    emit("PART 3 - LOANS WHOSE FIRST-ROW loan_age IS NOT THEIR MINIMUM")
    rule("=")
    anom = pd.concat(anomaly_parts, ignore_index=True)
    ids = set(anom["loan_id"].unique())
    n_anom = len(ids)
    emit("%-66s %18s" % ("anomaly loans (diagnostic 18 found 2,048)",
                         fmt(n_anom)))
    ga = loans.loc[loans["loan_id"].isin(ids), "group"]
    emit("%-66s %18s" % ("    of which early / late",
                         "%s / %s" % (fmt(int((ga == 0).sum())),
                                      fmt(int((ga == 1).sum())))))
    emit("%-66s %18s" % ("panel rows of those loans", fmt(len(anom))))
    emit()
    emit("A. PANEL, rows in stored order")
    rule()
    describe_steps("panel", *steps(anom, order_by_period=False), n_anom)
    emit()

    print("reading source performance files for %d loans" % n_anom, flush=True)
    src = source_rows(perf_paths, pidx, ids)
    src = src.sort_values(["loan_id", "pos"], kind="stable")
    emit("B. SOURCE PERFORMANCE FILES, rows in the order stored in the file")
    rule()
    emit("%-66s %18s" % ("source rows of the anomaly loans", fmt(len(src))))
    emit("%-66s %18s" % ("anomaly loans found in the source files",
                         fmt(src["loan_id"].nunique())))
    describe_steps("source, file order", *steps(src, order_by_period=False),
                   n_anom)
    emit()
    emit("C. SOURCE PERFORMANCE FILES, rows re-ordered by period")
    rule()
    describe_steps("source, period order", *steps(src, order_by_period=True),
                   n_anom)
    emit()

    # backward-step size distribution, source in period order
    _, d_age, d_per = steps(src, order_by_period=True)
    back = d_age < 0
    if back.any():
        emit("Backward steps by size, source rows in period order:")
        rule()
        emit("%-30s%18s%12s" % ("loan_age drop (months)", "steps", "share"))
        rule()
        sizes = -d_age[back]
        for lo, hi in [(1, 1), (2, 2), (3, 5), (6, 12), (13, 24), (25, 60),
                       (61, 10 ** 6)]:
            k = int(((sizes >= lo) & (sizes <= hi)).sum())
            label = str(lo) if lo == hi else ("%d+" % lo if hi == 10 ** 6
                                              else "%d-%d" % (lo, hi))
            emit("%-30s%18s%12s" % (label, fmt(k), pct(k, len(sizes))))
        rule()
        emit("%-30s%18s" % ("TOTAL", fmt(len(sizes))))
        emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
