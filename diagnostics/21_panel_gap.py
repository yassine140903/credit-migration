"""Loans present at origination that contribute no row to the transition panel.

The origination files for the 2006-2007 vintages hold more loan ids than the
built panel does. This finds every such loan, looks each one up in the raw
performance files, and reports what those loans were doing, so the decision to
drop them can be made on evidence: is it safe, or does it silently remove credit
events? It changes nothing and proposes no treatment.

DEFINITIONS
    gap loan        a loan id in an origination file with no row in any
                    data/panel/panel_*.parquet partition
    performance     a row of a perf_*.txt file carrying that loan id, whatever
    row             its content (rows with an unparseable period are counted
                    and reported, never silently dropped)
    last row        the row with the latest valid reporting period
    bucket          0 rows, exactly 1 row, 2 or more rows
    consecutive     two rows of one loan whose periods differ by exactly one
    pair            month - the only pairs src/panel.py can emit (R3)
    live pair       a consecutive pair whose ORIGIN row is not absorbing:
                    no zero-balance code 01/02/03/09 and a numeric delinquency
                    status (R1; non-numeric status such as RA maps to Default)
    split           the build's 80/20 train/test loan split, recomputed here
                    from the published seed with the same keyed BLAKE2b digest,
                    not imported

CAUSE (2-or-more bucket only) - the build's own rules, applied in its order
    R3  no consecutive pair       every adjacent pair spans a hole or repeats
    R1  no live pair              every consecutive pair has an absorbing origin
    R6  train, all live pairs     a train-split loan whose live pairs all
        after 2008-12             originate after the cut; the build excludes
                                  such rows by design
    UNEXPLAINED                   none of the above - a candidate build defect

SOURCES
    panel           loan_id only, one partition at a time, read directly (not
                    through src/panel_io). This is the panel's only use.
    origination     loan identifier only, chunked
    performance     loan identifier, period, delinquency status, zero-balance
                    code; chunked; only gap-loan rows are retained

Column indices are resolved at run time from the published layout by attribute
name and printed in the header. Nothing is imported from src/. Nothing is
written except the report. No git command is run.
"""

import glob
import hashlib
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
OUT = os.path.join("diagnostics", "21_panel_gap.txt")

CHUNKSIZE = 2_000_000
RULE_W = 121
EXPECTED_GAP = 6241
LIST_B = 50                     # 2+ bucket loans listed individually, at most
MONTH_FLAG = 0.05               # section E: flag a month above this share
FILE_FLAG = 1.5                 # section F: flag share ratio >= this or <= 1/this

CUT = 2008 * 12 + 11            # R6: last origin month in the fit window, 2008-12
SPLIT_SEED = "credit-migration/2006-2007/v1"
TRAIN_PCT = 80

ZB_PREPAID = ("01",)
ZB_DEFAULT = ("02", "03", "09")
ZB_CENSOR = ("15", "16", "96")
ZB_ABSORBING = ZB_PREPAID + ZB_DEFAULT

PERF_SHEET = "monthly performance data file"
ORIG_SHEET = "origination data file"
PERF_WANTED = [("loan", "loan identifier"), ("period", "period"),
               ("status", "current loan delinquency status"),
               ("zero_balance", "zero balance code")]
ORIG_WANTED = [("loan", "loan identifier")]
PERF_EXPECTED = {"loan": 0, "period": 1, "status": 3}

CAUSES = ["R3  no consecutive pair",
          "R1  no live pair",
          "R6  train, all live pairs after 2008-12",
          "UNEXPLAINED"]
BUCKETS = ["0 rows", "1 row", "2+ rows"]

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


def pct(num, den, dp=2):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def ym(idx):
    """Contiguous month index -> 'YYYY-MM'."""
    idx = int(idx)
    return "%04d-%02d" % (idx // 12, idx % 12 + 1)


def quarter_of(path):
    found = re.search(r"(20\d\dQ[1-4])", os.path.basename(path))
    if not found:
        raise SystemExit("cannot read a quarter name from %r" % path)
    return found.group(1)


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
    for marker in (PERF_SHEET, ORIG_SHEET):
        if marker not in sheets:
            raise SystemExit("layout %s has no %r sheet" % (LAYOUT, marker))

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
# step 1 - the gap id set
# ==========================================================================
def panel_ids():
    """Distinct panel loan ids, and the per-partition distinct counts."""
    parts, per_file = [], []
    for path in sorted(glob.glob(PANEL_GLOB)):
        t = pq.read_table(path, columns=["loan_id"])
        u = pc.unique(t["loan_id"]).to_numpy(zero_copy_only=False)
        per_file.append((os.path.basename(path), t.num_rows, len(u)))
        parts.append(u)
        print("    panel %s" % os.path.basename(path), flush=True)
    return pd.unique(np.concatenate(parts)), per_file


def origination_ids(oidx):
    """Every origination loan id with its vintage quarter, in file order."""
    c = oidx["loan"][0]
    frames = []
    for path in sorted(glob.glob(ORIG_GLOB)):
        q = quarter_of(path)
        for chunk in pd.read_csv(path, sep="|", header=None, dtype=str,
                                 usecols=[c], chunksize=CHUNKSIZE, engine="c",
                                 na_filter=False):
            frames.append(pd.DataFrame({"loan_id": chunk[c].str.strip()
                                        .to_numpy(), "vintage": q}))
        print("    orig  %s" % os.path.basename(path), flush=True)
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# step 2 - the performance scan
# ==========================================================================
def scan_perf(paths, pidx, gap_index):
    """Gap-loan rows from every perf file, and each file's total row count."""
    c = {k: pidx[k][0] for k, _ in PERF_WANTED}
    cols = sorted(c.values())
    found, perf_rows = [], {}
    for path in paths:
        q = quarter_of(path)
        n = 0
        t0 = time.time()
        for chunk in pd.read_csv(path, sep="|", header=None, dtype=str,
                                 usecols=cols, chunksize=CHUNKSIZE, engine="c",
                                 na_filter=False):
            n += len(chunk)
            loan = chunk[c["loan"]].str.strip()
            hit = loan.isin(gap_index).to_numpy()
            if hit.any():
                sub = chunk[hit]
                found.append(pd.DataFrame({
                    "loan_id": loan[hit].to_numpy(),
                    "period": sub[c["period"]].str.strip().to_numpy(),
                    "status": sub[c["status"]].str.strip().to_numpy(),
                    "zb": sub[c["zero_balance"]].str.strip().to_numpy(),
                    "file": q}))
        perf_rows[q] = n
        print("    perf  %s  %s rows  %.0f s"
              % (os.path.basename(path), fmt(n), time.time() - t0), flush=True)
    if found:
        rows = pd.concat(found, ignore_index=True)
    else:
        rows = pd.DataFrame({k: pd.Series(dtype=object) for k in
                             ("loan_id", "period", "status", "zb", "file")})
    return rows, perf_rows


def train_flag(loan_id):
    key = SPLIT_SEED.encode("ascii")[:16]
    digest = hashlib.blake2b(loan_id.encode("ascii"), digest_size=8,
                             key=key).digest()
    return (int.from_bytes(digest, "big") % 100) < TRAIN_PCT


def summarise(rows):
    """One record per loan found in the performance files."""
    per = pd.to_numeric(rows["period"], errors="coerce")
    mon = per % 100
    valid = (per.notna() & (mon >= 1) & (mon <= 12)).to_numpy()
    rows = rows.assign(valid=valid,
                       pidx=np.where(valid, (per // 100) * 12 + mon - 1, np.nan))
    # invalid periods sort FIRST, so a loan's final row is its latest valid one
    rows = rows.sort_values(["loan_id", "pidx"], kind="stable",
                            na_position="first").reset_index(drop=True)

    g = rows.groupby("loan_id", sort=True)
    L = pd.DataFrame({"rows": g.size(),
                      "invalid": g["valid"].agg(lambda s: int((~s).sum())),
                      "first": g["pidx"].min(), "last": g["pidx"].max(),
                      "files": g["file"].agg(lambda s: "+".join(sorted(set(s)))),
                      "n_files": g["file"].nunique()})
    tail = rows.drop_duplicates("loan_id", keep="last").set_index("loan_id")
    L["last_status"] = tail["status"]
    L["last_zb"] = tail["zb"]
    event = rows["zb"].isin(ZB_DEFAULT) | ~rows["status"].str.isdigit()
    L["any_default"] = event.groupby(rows["loan_id"]).any()

    # pairs over valid rows only, exactly as the build sees them
    V = rows[rows["valid"]].reset_index(drop=True)
    loan = V["loan_id"].to_numpy()
    p = V["pidx"].to_numpy()
    absorbing = (V["zb"].isin(ZB_ABSORBING) | ~V["status"].str.isdigit()).to_numpy()
    same = np.zeros(len(V), dtype=bool)
    same[1:] = loan[1:] == loan[:-1]
    dp = np.zeros(len(V))
    dp[1:] = p[1:] - p[:-1]
    prev_abs = np.zeros(len(V), dtype=bool)
    prev_abs[1:] = absorbing[:-1]
    prev_p = np.full(len(V), np.nan)
    prev_p[1:] = p[:-1]
    consec = same & (dp == 1)
    live = consec & ~prev_abs
    P = pd.DataFrame({"loan_id": loan,
                      "consec": consec, "hole": same & (dp > 1),
                      "hole_w": np.where(same & (dp > 1), dp - 1, 0),
                      "dup": same & (dp <= 0),
                      "live": live, "live_pre": live & (prev_p <= CUT)})
    agg = P.groupby("loan_id").agg(consec=("consec", "sum"), holes=("hole", "sum"),
                                   widest=("hole_w", "max"), dups=("dup", "sum"),
                                   live=("live", "sum"),
                                   live_pre=("live_pre", "sum"))
    L = L.join(agg).fillna({c: 0 for c in agg.columns})
    for c in agg.columns:
        L[c] = L[c].astype(np.int64)
    L["train"] = [train_flag(i) for i in L.index]

    cause = np.full(len(L), CAUSES[3], dtype=object)
    no_live_pre_train = L["train"] & (L["live_pre"] == 0)
    cause[no_live_pre_train] = CAUSES[2]
    cause[L["live"] == 0] = CAUSES[1]
    cause[L["consec"] == 0] = CAUSES[0]
    L["cause"] = cause
    return L, rows


def periods_label(r):
    if r["holes"] == 0 and r["dups"] == 0 and r["invalid"] == 0:
        return "consecutive"
    bits = []
    if r["holes"]:
        bits.append("%d hole%s, widest %d"
                    % (r["holes"], "" if r["holes"] == 1 else "s", r["widest"]))
    if r["dups"]:
        bits.append("%d dup" % r["dups"])
    if r["invalid"]:
        bits.append("%d bad period" % r["invalid"])
    return "; ".join(bits)


def code_label(z):
    return z if z else "(none)"


# ==========================================================================
# tables
# ==========================================================================
def count_table(label, series, den, order=None, width=44):
    """One categorical column as counts and shares. Zero rows are printed."""
    vc = series.value_counts()
    keys = list(order) if order is not None else sorted(vc.index)
    emit("%-*s%14s%12s" % (width, label, "loans", "share"))
    rule()
    for k in keys:
        n = int(vc.get(k, 0))
        emit("%-*s%14s%12s" % (width, k, fmt(n), pct(n, den)))
    rule()
    emit("%-*s%14s%12s" % (width, "TOTAL", fmt(den), "100.00%" if den else "n/a"))


def crosstab(rows_s, cols_s, row_label):
    """Counts only. Every cell printed, zero included."""
    rk = sorted(set(rows_s), key=status_key)
    ck = sorted(set(cols_s), key=lambda z: (z != "", z))
    tab = pd.crosstab(rows_s, cols_s)
    w = 10
    emit("%-14s" % row_label + "".join("%*s" % (w, code_label(c)) for c in ck)
         + "%*s" % (w + 2, "TOTAL"))
    rule()
    for r in rk:
        emit("%-14s" % r + "".join("%*s" % (w, fmt(tab.loc[r, c])) for c in ck)
             + "%*s" % (w + 2, fmt(tab.loc[r].sum())))
    rule()
    emit("%-14s" % "TOTAL" + "".join("%*s" % (w, fmt(tab[c].sum())) for c in ck)
         + "%*s" % (w + 2, fmt(tab.values.sum())))


def status_key(s):
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def status_name(s):
    return s if s else "(blank)"


def outcome(z, s):
    if z in ZB_PREPAID:
        return "code 01  prepaid"
    if z in ZB_DEFAULT:
        return "code 02/03/09  default-type"
    if z in ZB_CENSOR:
        return "code 15/16/96  censoring"
    if z:
        return "code, other"
    if not s.isdigit():
        return "no code, status non-numeric (RA etc.)"
    s = int(s)
    if s == 0:
        return "no code, status 0  current"
    if s <= 2:
        return "no code, status 1-2"
    return "no code, status 3+"


OUTCOMES = ["code 01  prepaid", "code 02/03/09  default-type",
            "code 15/16/96  censoring", "code, other",
            "no code, status non-numeric (RA etc.)",
            "no code, status 0  current", "no code, status 1-2",
            "no code, status 3+"]


# ==========================================================================
def main():
    t0 = time.time()
    pidx, oidx, perf_sheet, orig_sheet = resolve()
    perf_paths = sorted(glob.glob(PERF_GLOB))

    rule("=")
    emit("21_panel_gap - origination loans that contribute no row to the panel")
    rule("=")
    emit("origination : %s" % ORIG_GLOB)
    emit("performance : %s" % PERF_GLOB)
    emit("panel       : %s   (loan_id only)" % PANEL_GLOB)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
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
    emit("sheets: performance %s, origination %s" % (perf_sheet, orig_sheet))
    emit()

    # ---- step 1 ---------------------------------------------------------------
    pids, per_file = panel_ids()
    orig = origination_ids(oidx)
    n_orig_rec = len(orig)
    dup_orig = int(orig["loan_id"].duplicated().sum())
    orig_u = orig.drop_duplicates("loan_id")
    in_panel = orig_u["loan_id"].isin(pids).to_numpy()
    gap = orig_u[~in_panel].set_index("loan_id")
    panel_only = int((~pd.Index(pids).isin(orig_u["loan_id"])).sum())
    n_gap = len(gap)

    rule("=")
    emit("STEP 1. THE GAP ID SET   (unit: loan ids)")
    rule("=")
    emit("%-24s%16s%16s" % ("panel partition", "rows", "distinct ids"))
    rule()
    for name, r, u in per_file:
        emit("%-24s%16s%16s" % (name, fmt(r), fmt(u)))
    rule()
    emit("%-24s%16s%16s" % ("TOTAL", fmt(sum(r for _, r, _ in per_file)),
                            fmt(sum(u for _, _, u in per_file))))
    emit()
    emit("%-58s%18s" % ("origination records", fmt(n_orig_rec)))
    emit("%-58s%18s" % ("  duplicate ids among them", fmt(dup_orig)))
    emit("%-58s%18s" % ("distinct origination ids", fmt(len(orig_u))))
    emit("%-58s%18s" % ("distinct panel ids", fmt(len(pids))))
    emit("%-58s%18s" % ("panel ids NOT in any origination file", fmt(panel_only)))
    emit("%-58s%18s" % ("gap ids = origination - panel", fmt(n_gap)))
    emit("%-58s%18s" % ("expected", fmt(EXPECTED_GAP)))
    if n_gap == EXPECTED_GAP:
        emit("CHECK PASSED: the gap holds exactly %s ids." % fmt(EXPECTED_GAP))
    else:
        emit("CHECK FAILED: gap %s, expected %s, difference %+d. Continuing."
             % (fmt(n_gap), fmt(EXPECTED_GAP), n_gap - EXPECTED_GAP))
    emit()

    # ---- step 2 ---------------------------------------------------------------
    rows, perf_rows = scan_perf(perf_paths, pidx, gap.index.to_numpy())
    L, rows = summarise(rows)
    stray = L.index.difference(gap.index)
    if len(stray):
        raise SystemExit("%d loans found that are not gap ids" % len(stray))
    L["vintage"] = gap.loc[L.index, "vintage"]
    zero_ids = gap.index.difference(L.index)
    one = L[L["rows"] == 1]
    many = L[L["rows"] >= 2]
    seen = L                                    # loans with at least one row
    n0, n1, n2 = len(zero_ids), len(one), len(many)

    # ---- A --------------------------------------------------------------------
    rule("=")
    emit("A. BUCKET COUNTS   (unit: gap loans; rows = raw performance rows)")
    rule("=")
    emit("%-34s%14s%12s" % ("bucket", "loans", "share"))
    rule()
    for name, k in zip(BUCKETS, (n0, n1, n2)):
        emit("%-34s%14s%12s" % (name, fmt(k), pct(k, n_gap)))
    rule()
    emit("%-34s%14s%12s" % ("TOTAL", fmt(n0 + n1 + n2), pct(n0 + n1 + n2, n_gap)))
    emit("sums to the gap (%s): %s" % (fmt(n_gap),
                                       "yes" if n0 + n1 + n2 == n_gap else "NO"))
    emit("performance rows retained for gap loans: %s; with an unparseable period: %s"
         % (fmt(len(rows)), fmt(int((~rows["valid"]).sum()))))
    emit()

    # ---- B --------------------------------------------------------------------
    rule("=")
    emit("B. THE 2-OR-MORE BUCKET   (unit: loans; %s loans)" % fmt(n2))
    rule("=")
    emit("Each loan below has at least two performance rows yet no panel row. The")
    emit("cause column applies the build's own rules in order (see the docstring);")
    emit("'UNEXPLAINED' means none of them accounts for the loan.")
    emit()
    head = many.head(LIST_B)
    emit("Loans, individually%s:" % ("" if n2 <= LIST_B else
                                    " (first %d of %s, by loan id)"
                                    % (LIST_B, fmt(n2))))
    rule()
    row_fmt = "%-13s%5s%9s%9s  %-22s%-8s%4s%5s%7s%14s  %s"
    emit(row_fmt % ("loan_id", "rows", "first", "last", "periods", "file(s)",
                    "st", "zb", "split", "live pre/all", "cause"))
    rule()
    if n2 == 0:
        emit("(none)")
    for lid, r in head.iterrows():
        emit(row_fmt % (lid, fmt(r["rows"]), ym(r["first"]), ym(r["last"]),
                        periods_label(r)[:22], r["files"][:8],
                        status_name(r["last_status"]), r["last_zb"] or "-",
                        "train" if r["train"] else "test",
                        "%d/%d" % (r["live_pre"], r["live"]), r["cause"].split()[0]))
    rule()
    emit("st, zb: delinquency status and zero-balance code on the last row ('-' = no")
    emit("code). live pre/all: live consecutive pairs originating <= 2008-12 / in all.")
    emit()
    if n2 > LIST_B or n2 == 0:
        emit("Summary of all %s loans in the bucket:" % fmt(n2))
        emit()
    count_table("cause", many["cause"], n2, order=CAUSES)
    emit()
    count_table("split", many["train"].map({True: "train", False: "test"}), n2,
                order=["train", "test"])
    emit()
    cls = many.apply(lambda r: "consecutive" if periods_label(r) == "consecutive"
                     else "contains a hole, dup or bad period", axis=1) \
        if n2 else pd.Series(dtype=object)
    count_table("reporting periods", cls, n2,
                order=["consecutive", "contains a hole, dup or bad period"])
    emit()
    count_table("source file(s)", many["files"], n2,
                order=sorted(set(many["files"])) or ["(none)"])
    emit()
    bands = pd.cut(many["rows"], [1, 2, 3, 6, 12, 24, 60, 10 ** 6],
                   labels=["2", "3", "4-6", "7-12", "13-24", "25-60", "61+"])
    count_table("performance rows per loan", bands.astype(str), n2,
                order=["2", "3", "4-6", "7-12", "13-24", "25-60", "61+"])
    emit()
    for label, col in (("year of first row", "first"), ("year of last row", "last")):
        yr = (many[col] // 12).map(lambda v: "(no valid period)" if pd.isna(v)
                                   else str(int(v)))
        count_table(label, yr, n2)
        emit()
    emit("Last-row delinquency status x last-row zero-balance code (loans):")
    rule()
    if n2:
        crosstab(many["last_status"].map(status_name), many["last_zb"], "status")
    else:
        emit("(none)")
    emit()
    unexp = many[many["cause"] == CAUSES[3]]
    emit("UNEXPLAINED loans (candidate build defects): %s" % fmt(len(unexp)))
    for lid, r in unexp.head(LIST_B).iterrows():
        emit("    %s  rows %s  %s..%s  %s  live pre/all %d/%d  %s"
             % (lid, fmt(r["rows"]), ym(r["first"]), ym(r["last"]),
                periods_label(r), r["live_pre"], r["live"],
                "train" if r["train"] else "test"))
    emit()

    # ---- C --------------------------------------------------------------------
    rule("=")
    emit("C. THE 1-ROW BUCKET   (unit: loans; %s loans)" % fmt(n1))
    rule("=")
    emit("Last-row delinquency status x last-row zero-balance code (counts only):")
    rule()
    if n1:
        crosstab(one["last_status"].map(status_name), one["last_zb"], "status")
    else:
        emit("(none)")
    emit()
    nocode = one[one["last_zb"] == ""]
    emit("Restricted to the %s loans with NO zero-balance code:" % fmt(len(nocode)))
    emit()
    count_table("delinquency status", nocode["last_status"].map(status_name),
                len(nocode), order=sorted(set(nocode["last_status"].map(status_name)),
                                          key=status_key) or ["(none)"])
    emit()

    # ---- D --------------------------------------------------------------------
    rule("=")
    emit("D. THE 0-ROW BUCKET   (unit: loans)")
    rule("=")
    emit("%-58s%18s%12s" % ("gap loans with no performance row in any file",
                            fmt(n0), pct(n0, n_gap)))
    emit()
    zv = gap.loc[zero_ids, "vintage"] if n0 else pd.Series(dtype=object)
    count_table("origination year", zv.str[:4], n0, order=["2006", "2007"])
    emit()
    count_table("origination file (vintage)", zv, n0,
                order=sorted(set(orig["vintage"])))
    emit()

    # ---- E --------------------------------------------------------------------
    rule("=")
    emit("E. CALENDAR MONTH OF THE LAST ROW   (unit: loans with >= 1 row; %s loans)"
         % fmt(len(seen)))
    rule("=")
    emit("Flag: a month holding more than %.0f%% of a column's loans. Columns flagged:"
         % (100 * MONTH_FLAG))
    emit("1 = 1-row bucket, 2 = 2+ bucket, A = all loans with at least one row.")
    rule()
    emit("%-10s%12s%10s%12s%10s%12s%10s   %s"
         % ("month", "1 row", "share", "2+ rows", "share", "all", "share", "flag"))
    rule()
    months = sorted(set(seen["last"].dropna().astype(int)))
    c1 = one["last"].dropna().astype(int).value_counts()
    c2 = many["last"].dropna().astype(int).value_counts()
    flagged = []
    for m in months:
        a, b = int(c1.get(m, 0)), int(c2.get(m, 0))
        f = "".join(t for t, k, d in (("1", a, n1), ("2", b, n2),
                                       ("A", a + b, len(seen)))
                    if d and k / d > MONTH_FLAG)
        if f:
            flagged.append((m, f, a, b))
        emit("%-10s%12s%10s%12s%10s%12s%10s   %s"
             % (ym(m), fmt(a), pct(a, n1), fmt(b), pct(b, n2), fmt(a + b),
                pct(a + b, len(seen)), f))
    if not months:
        emit("(none)")
    rule()
    emit("%-10s%12s%10s%12s%10s%12s%10s"
         % ("TOTAL", fmt(n1), "100.00%" if n1 else "n/a", fmt(n2),
            "100.00%" if n2 else "n/a", fmt(len(seen)),
            "100.00%" if len(seen) else "n/a"))
    no_valid = int(seen["last"].isna().sum())
    emit("loans with no valid period on any row (not in the table): %s"
         % fmt(no_valid))
    emit("flagged months: %s" % (", ".join("%s [%s]" % (ym(m), f)
                                          for m, f, _, _ in flagged) or "none"))
    emit()

    # ---- F --------------------------------------------------------------------
    rule("=")
    emit("F. SOURCE FILE OF THE ROWS   (unit: loans with >= 1 row; %s loans)"
         % fmt(len(seen)))
    rule("=")
    orig_n = orig["vintage"].value_counts()
    tot_perf = sum(perf_rows.values())
    emit("File size is measured two ways: origination records in the vintage")
    emit("(loans the file should carry) and performance rows in the file. 'ratio' =")
    emit("the file's share of gap loans / its share of origination records. Flag:")
    emit("ratio >= %.2f or <= %.2f." % (FILE_FLAG, 1 / FILE_FLAG))
    rule()
    emit("%-9s%9s%10s%9s%10s%9s%10s%12s%9s%14s%9s%7s  %s"
         % ("file", "1 row", "share", "2+ rows", "share", "all", "share",
            "orig recs", "share", "perf rows", "share", "ratio", "flag"))
    rule()
    per_file_loans = rows.drop_duplicates(["loan_id", "file"])
    one_ids, many_ids = set(one.index), set(many.index)
    for q in sorted(perf_rows):
        ids = per_file_loans.loc[per_file_loans["file"] == q, "loan_id"]
        a = int(ids.isin(one_ids).sum())
        b = int(ids.isin(many_ids).sum())
        share_gap = (a + b) / len(seen) if len(seen) else 0.0
        share_orig = orig_n.get(q, 0) / n_orig_rec
        ratio = share_gap / share_orig if share_orig else float("nan")
        flag = "HIGH" if ratio >= FILE_FLAG else "LOW" if ratio <= 1 / FILE_FLAG \
            else ""
        emit("%-9s%9s%10s%9s%10s%9s%10s%12s%9s%14s%9s%7.2f  %s"
             % (q, fmt(a), pct(a, n1), fmt(b), pct(b, n2), fmt(a + b),
                pct(a + b, len(seen)), fmt(orig_n.get(q, 0)),
                pct(orig_n.get(q, 0), n_orig_rec), fmt(perf_rows[q]),
                pct(perf_rows[q], tot_perf), ratio, flag))
    rule()
    emit("%-9s%9s%10s%9s%10s%9s%10s%12s%9s%14s"
         % ("TOTAL", fmt(n1), "", fmt(n2), "", fmt(len(seen)), "",
            fmt(n_orig_rec), "", fmt(tot_perf)))
    multi = int((seen["n_files"] > 1).sum())
    other = int((seen["files"] != seen["vintage"]).sum())
    emit("loans whose rows sit in more than one file: %s" % fmt(multi))
    emit("loans whose rows are not (only) in their own vintage's file: %s"
         % fmt(other))
    emit()
    emit("By the gap loan's origination vintage, for comparison with the table above")
    emit("(all gap loans, including the 0-row bucket):")
    rule()
    emit("%-9s%10s%10s%10s%10s%10s%12s%9s%7s"
         % ("vintage", "0 rows", "1 row", "2+ rows", "all", "share",
            "orig recs", "share", "ratio"))
    rule()
    for q in sorted(orig_n.index):
        k0 = int((gap.loc[zero_ids, "vintage"] == q).sum()) if n0 else 0
        k1 = int((one["vintage"] == q).sum())
        k2 = int((many["vintage"] == q).sum())
        k = k0 + k1 + k2
        so = orig_n[q] / n_orig_rec
        emit("%-9s%10s%10s%10s%10s%10s%12s%9s%7.2f"
             % (q, fmt(k0), fmt(k1), fmt(k2), fmt(k), pct(k, n_gap),
                fmt(orig_n[q]), pct(orig_n[q], n_orig_rec),
                (k / n_gap) / so if n_gap else float("nan")))
    rule()
    emit()

    # ---- G --------------------------------------------------------------------
    rule("=")
    emit("G. CREDIT EVENTS CARRIED BY THE GAP LOANS   (unit: loans)")
    rule("=")
    emit("Outcome on the last row, by bucket. The 0-row bucket carries no row and")
    emit("so no observable event.")
    rule()
    emit("%-42s%12s%10s%12s%10s%12s%10s"
         % ("last-row outcome", "1 row", "share", "2+ rows", "share", "all",
            "share"))
    rule()
    o1 = pd.Series([outcome(z, s) for z, s in
                    zip(one["last_zb"], one["last_status"])]).value_counts()
    o2 = pd.Series([outcome(z, s) for z, s in
                    zip(many["last_zb"], many["last_status"])]).value_counts()
    for o in OUTCOMES:
        a, b = int(o1.get(o, 0)), int(o2.get(o, 0))
        emit("%-42s%12s%10s%12s%10s%12s%10s"
             % (o, fmt(a), pct(a, n1), fmt(b), pct(b, n2), fmt(a + b),
                pct(a + b, len(seen))))
    rule()
    emit("%-42s%12s%10s%12s%10s%12s%10s"
         % ("TOTAL", fmt(n1), "", fmt(n2), "", fmt(len(seen)), ""))
    emit()
    emit("Loans with a default-type code (02/03/09) or a non-numeric status on ANY")
    emit("row, not only the last:")
    rule()
    for name, part in (("1 row", one), ("2+ rows", many)):
        k = int(part["any_default"].sum())
        emit("%-58s%14s%12s" % (name, fmt(k), pct(k, len(part))))
    for cause in CAUSES:
        part = many[many["cause"] == cause]
        k = int(part["any_default"].sum())
        emit("%-58s%14s%12s" % ("2+ rows, cause %s" % cause.split("  ")[0],
                                fmt(k), pct(k, len(part))))
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
