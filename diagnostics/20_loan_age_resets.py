"""Backward loan_age steps in the source performance files, and modification.

Diagnostic 19 found loans whose loan_age decreases between consecutive months.
A check on 2006Q1 showed the decrease is loan_age restarting on the row that
carries the modification flag. This measures that across all eight quarters, as
evidence for deciding what loan_age should mean for a modified loan. It changes
nothing and proposes no treatment.

DEFINITIONS
    reset           a row whose loan_age is LOWER than on the loan's previous
                    observed row (period order; across reporting holes too;
                    a null age on either side is not compared). "The reset row"
                    is that lower-age row, "the row before" its predecessor.
    drop            loan_age on the row before minus loan_age on the reset row
    modification    the performance file's Modification Flag. Values are shown
    flag            as they appear: Y, P, blank, anything else as itself.
    modified loan   a loan carrying Y or P on at least one row. "First flagged
                    period" is the earliest such row.
    unflagged       a loan with at least one reset and NO Y or P on any row
    reset loan

SOURCES
    performance     every row of every file, eight columns, chunked; sorted per
                    file by (loan, period) - files are never held whole
    panel           the built panel_*.parquet, four columns, one partition at a
                    time; read directly, not through src/panel_io, so this check
                    stays independent of the build code

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
PANEL_GLOB = os.path.join("data", "panel", "panel_*.parquet")
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")
OUT = os.path.join("diagnostics", "20_loan_age_resets.txt")

CHUNKSIZE = 2_000_000
RULE_W = 121
LIST_UNFLAGGED = 200            # unflagged reset rows listed individually, at most
NA = -9999

PERF_SHEET = "monthly performance data file"
PERF_WANTED = [("loan", "loan identifier"), ("period", "period"),
               ("status", "current loan delinquency status"),
               ("loan_age", "loan age"),
               ("rmlm", "remaining months to legal maturity"),
               ("mod", "modification flag"),
               ("zero_balance", "zero balance code"),
               ("rate", "current interest rate")]
PERF_EXPECTED = {"loan": 0, "period": 1, "status": 3}

FLAGS = ["Y", "P", "blank", "other"]
F_BLANK, F_Y, F_P, F_OTHER = 0, 1, 2, 3        # stored codes
FLAG_ORDER = [F_Y, F_P, F_BLANK, F_OTHER]      # display order, matches FLAGS

STATES = ["Current", "DQ30", "DQ60", "DQ90+", "Prepaid", "Default"]

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


def to_yyyymm(idx):
    return (idx // 12) * 100 + (idx % 12) + 1


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
        if PERF_SHEET in head and fields:
            out[PERF_SHEET] = (fields, name)
    return out


def _norm(t):
    t = t.strip().lower().replace("®", "").replace("™", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()/%.-]+", " ", t)).strip()


def resolve():
    fields, sheet = _sheets(LAYOUT)[PERF_SHEET]
    got = {}
    for label, want in PERF_WANTED:
        w = _norm(want)
        hits = [f for f in fields if _norm(f[1]) == w] or \
               [f for f in fields if w in _norm(f[1])]
        if len(hits) != 1:
            raise SystemExit("layout lookup for %r matched %d row(s)"
                             % (want, len(hits)))
        got[label] = (hits[0][0] - 1, hits[0][0], hits[0][1])
    for label, expect in PERF_EXPECTED.items():
        if got[label][0] != expect:
            raise SystemExit("layout gives %d for performance %r, project uses %d"
                             % (got[label][0], label, expect))
    return got, sheet


# ==========================================================================
# performance pass
# ==========================================================================
def read_file(path, pidx, other_flags):
    """One file's eight columns as narrow arrays, sorted by (loan, period)."""
    c = {k: pidx[k][0] for k, _ in PERF_WANTED}
    cols = sorted(c.values())
    parts = {k: [] for k in ("tail", "period", "status", "age", "rmlm", "mod",
                             "zb", "rate")}
    prefixes = set()
    for chunk in pd.read_csv(path, sep="|", header=None, dtype=str, usecols=cols,
                             chunksize=CHUNKSIZE, engine="c", na_filter=False):
        loan = chunk[c["loan"]].str.strip()
        prefixes.update(loan.str[:5].unique())
        parts["tail"].append(pd.to_numeric(loan.str[5:], errors="coerce")
                             .fillna(-1).astype(np.int64).to_numpy())
        per = pd.to_numeric(chunk[c["period"]].str.strip(), errors="coerce")
        idx = (per // 100 * 12 + per % 100 - 1).where(
            (per % 100 >= 1) & (per % 100 <= 12))
        parts["period"].append(idx.fillna(-1).astype(np.int32).to_numpy())
        raw = chunk[c["status"]].str.strip()
        parts["status"].append(pd.to_numeric(raw.where(raw.str.isdigit()),
                                             errors="coerce")
                               .fillna(-1).astype(np.int16).to_numpy())
        for k, label in (("age", "loan_age"), ("rmlm", "rmlm")):
            parts[k].append(pd.to_numeric(chunk[c[label]].str.strip(),
                                          errors="coerce")
                            .fillna(NA).astype(np.int32).to_numpy())
        flag = chunk[c["mod"]].str.strip()
        code = np.full(len(flag), F_OTHER, dtype=np.int8)
        arr = flag.to_numpy()
        code[arr == ""] = F_BLANK
        code[arr == "Y"] = F_Y
        code[arr == "P"] = F_P
        for v, n in flag[code == F_OTHER].value_counts().items():
            other_flags[v] = other_flags.get(v, 0) + int(n)
        parts["mod"].append(code)
        parts["zb"].append(pd.to_numeric(chunk[c["zero_balance"]].str.strip(),
                                         errors="coerce")
                           .fillna(-1).astype(np.int16).to_numpy())
        parts["rate"].append(pd.to_numeric(chunk[c["rate"]].str.strip(),
                                           errors="coerce")
                             .astype(np.float32).to_numpy())
    if len(prefixes) != 1:
        raise SystemExit("%s carries %d loan-id prefixes" % (path, len(prefixes)))
    a = {k: np.concatenate(v) for k, v in parts.items()}
    keep = (a["tail"] >= 0) & (a["period"] >= 0)
    order = np.lexsort((a["period"][keep], a["tail"][keep]))
    return {k: v[keep][order] for k, v in a.items()}, prefixes.pop()


def process_file(path, pidx, other_flags):
    a, prefix = read_file(path, pidx, other_flags)
    tail, age = a["tail"], a["age"]
    n = len(tail)
    same = np.zeros(n, dtype=bool)
    same[1:] = tail[1:] == tail[:-1]
    known = np.zeros(n, dtype=bool)
    known[1:] = (age[1:] != NA) & (age[:-1] != NA)
    down = np.zeros(n, dtype=bool)
    down[1:] = age[1:] < age[:-1]
    r = np.nonzero(same & known & down)[0]              # reset rows
    b = r - 1                                           # the row before

    flagged_row = (a["mod"] == F_Y) | (a["mod"] == F_P)
    mod_tails, first_idx = np.unique(tail[flagged_row], return_index=True)
    first_mod = a["period"][flagged_row][first_idx]     # rows sorted: min period
    loans = np.unique(tail)

    rst = {
        "tail": tail[r], "prefix": np.full(len(r), prefix),
        "period": a["period"][r], "dper": a["period"][r] - a["period"][b],
        "before": age[b], "after": age[r],
        "flag": a["mod"][r], "flag_before": a["mod"][b],
        "status_before": a["status"][b], "status_after": a["status"][r],
        "zb": a["zb"][r],
        "rmlm_before": a["rmlm"][b], "rmlm_after": a["rmlm"][r],
        "rate_before": a["rate"][b], "rate_after": a["rate"][r],
        "mod_loan": np.isin(tail[r], mod_tails),
    }
    # rows after the reset row, per reset, for the unflagged listing
    last_of_loan = np.searchsorted(tail, tail[r], side="right") - 1
    rst["rows_after"] = last_of_loan - r
    return {"prefix": prefix, "name": os.path.basename(path), "rows": n,
            "loans": len(loans), "mod_tails": mod_tails, "first_mod": first_mod,
            "reset": rst}


# ==========================================================================
# panel pass
# ==========================================================================
def panel_pass(by_prefix):
    """Panel rows by state, split by the loan's modification status."""
    # columns: unmodified, modified (all), modified before first flag, at/after
    from_counts = np.zeros((len(STATES), 4), dtype=np.int64)
    to_counts = np.zeros((len(STATES), 4), dtype=np.int64)
    in_panel = {}
    for path in sorted(glob.glob(PANEL_GLOB)):
        t = pq.read_table(path, columns=["loan_id", "from_period",
                                         "from_state", "to_state"])
        ids = t["loan_id"].combine_chunks()
        prefix = ids[0].as_py()[:5]
        tails = pc.cast(pc.utf8_slice_codeunits(ids, 5, 12), "int64").to_numpy()
        info = by_prefix[prefix]
        pos = np.searchsorted(info["mod_tails"], tails)
        safe = np.clip(pos, 0, max(len(info["mod_tails"]) - 1, 0))
        is_mod = (len(info["mod_tails"]) > 0) & (info["mod_tails"][safe] == tails)
        first = to_yyyymm(info["first_mod"][safe].astype(np.int64))
        post = is_mod & (t["from_period"].to_numpy() >= first)
        pre = is_mod & ~post
        for arr, col in ((from_counts, "from_state"), (to_counts, "to_state")):
            codes = t[col].combine_chunks().indices.to_numpy()
            names = t[col].combine_chunks().dictionary.to_pylist()
            remap = np.array([STATES.index(s) for s in names])
            s = remap[codes]
            for k, m in enumerate((~is_mod, is_mod, pre, post)):
                arr[:, k] += np.bincount(s[m], minlength=len(STATES))
        uniq = np.unique(tails)
        in_panel[prefix] = uniq
    return from_counts, to_counts, in_panel


def state_table(title, arr):
    emit(title)
    rule()
    heads = ["unmodified", "modified", "  mod: before", "  mod: at/after"]
    emit("%-10s" % "state" + "".join("%16s%10s" % (h, "share") for h in heads))
    rule()
    tot = arr.sum(axis=0)
    for i, s in enumerate(STATES):
        emit("%-10s" % s + "".join("%16s%10s" % (fmt(arr[i, k]),
                                                 pct(arr[i, k], tot[k], 2))
                                   for k in range(4)))
    rule()
    emit("%-10s" % "TOTAL" + "".join("%16s%10s" % (fmt(tot[k]), "100.00%")
                                     for k in range(4)))
    emit()


# ==========================================================================
def band_table(label, values, bands, den=None):
    den = len(values) if den is None else den
    emit("%-30s%18s%12s" % (label, "resets", "share"))
    rule()
    for lo, hi in bands:
        k = int(((values >= lo) & (values <= hi)).sum())
        name = ("< %d" % (hi + 1) if lo <= -10 ** 6 else str(lo) if lo == hi
                else "%d+" % lo if hi >= 10 ** 6 else "%d-%d" % (lo, hi))
        emit("%-30s%18s%12s" % (name, fmt(k), pct(k, den)))
    rule()
    emit("%-30s%18s" % ("TOTAL", fmt(den)))


def main():
    t0 = time.time()
    pidx, sheet = resolve()
    perf_paths = sorted(glob.glob(PERF_GLOB))

    rule("=")
    emit("20_loan_age_resets - backward loan_age steps and the modification flag")
    rule("=")
    emit("performance : %s" % PERF_GLOB)
    emit("panel       : %s" % PANEL_GLOB)
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
    rule()
    emit("performance sheet: %s" % sheet)
    emit()

    other_flags = {}
    files = []
    for p in perf_paths:
        files.append(process_file(p, pidx, other_flags))
        print("    read %s" % os.path.basename(p), flush=True)
    by_prefix = {f["prefix"]: f for f in files}
    R = {k: np.concatenate([f["reset"][k] for f in files])
         for k in files[0]["reset"]}
    n_reset = len(R["tail"])
    key = pd.Series(R["prefix"]).str.cat(pd.Series(R["tail"]).astype(str),
                                         sep="|").to_numpy()
    reset_loans_per = pd.Series(key).value_counts()
    n_reset_loans = len(reset_loans_per)
    n_loans = sum(f["loans"] for f in files)
    n_mod = sum(len(f["mod_tails"]) for f in files)

    # ---- 1. by quarter ---------------------------------------------------------
    rule("=")
    emit("1. LOANS WITH AT LEAST ONE RESET, BY QUARTER   (unit: loans)")
    rule("=")
    emit("%-20s%14s%14s%10s%14s%14s%14s" % ("file", "loans", "reset loans",
                                            "share", "resets",
                                            "modified", "mod, no reset"))
    rule()
    for f in files:
        rt = np.unique(f["reset"]["tail"])
        no_reset = len(np.setdiff1d(f["mod_tails"], rt))
        emit("%-20s%14s%14s%10s%14s%14s%14s"
             % (f["name"], fmt(f["loans"]), fmt(len(rt)),
                pct(len(rt), f["loans"], 2), fmt(len(f["reset"]["tail"])),
                fmt(len(f["mod_tails"])), fmt(no_reset)))
    rule()
    all_no_reset = sum(len(np.setdiff1d(f["mod_tails"],
                                        np.unique(f["reset"]["tail"])))
                       for f in files)
    emit("%-20s%14s%14s%10s%14s%14s%14s"
         % ("TOTAL", fmt(n_loans), fmt(n_reset_loans),
            pct(n_reset_loans, n_loans, 2), fmt(n_reset), fmt(n_mod),
            fmt(all_no_reset)))
    emit("modified = Y or P on at least one row. 'mod, no reset' = modified loans")
    emit("whose loan_age never goes backwards.")
    emit()

    # ---- 2. flag on the reset row ----------------------------------------------
    rule("=")
    emit("2. MODIFICATION FLAG ON THE RESET ROW AND ON THE ROW BEFORE   (unit: resets)")
    rule("=")
    emit("%-14s%18s%12s%18s%12s" % ("flag", "reset row", "share", "row before",
                                    "share"))
    rule()
    for name, code in zip(FLAGS, FLAG_ORDER):
        k1 = int((R["flag"] == code).sum())
        k2 = int((R["flag_before"] == code).sum())
        emit("%-14s%18s%12s%18s%12s" % (name, fmt(k1), pct(k1, n_reset),
                                        fmt(k2), pct(k2, n_reset)))
    rule()
    emit("%-14s%18s" % ("TOTAL", fmt(n_reset)))
    emit("flag values other than Y, P and blank, anywhere in the files: %s"
         % (", ".join("%r x %s" % (k, fmt(v)) for k, v in
                      sorted(other_flags.items())) or "none"))
    emit()

    # ---- 3. drop size and ages --------------------------------------------------
    rule("=")
    emit("3. DROP SIZE AND loan_age EITHER SIDE OF THE RESET   (unit: resets)")
    rule("=")
    drop = R["before"] - R["after"]
    emit("%-40s %s" % ("drop percentiles p1/p10/p50/p90/p99",
                       " / ".join(str(int(np.percentile(drop, q,
                                                        method="inverted_cdf")))
                                  for q in (1, 10, 50, 90, 99))))
    emit("%-40s %d / %d" % ("drop min / max", drop.min(), drop.max()))
    emit()
    band_table("drop (months)", drop, [(1, 1), (2, 2), (3, 5), (6, 12),
                                       (13, 24), (25, 36), (37, 60),
                                       (61, 120), (121, 10 ** 6)])
    emit()
    band_table("loan_age on the row before", R["before"],
               [(0, 12), (13, 24), (25, 36), (37, 48), (49, 60), (61, 84),
                (85, 120), (121, 10 ** 6)])
    emit()
    band_table("loan_age on the reset row", R["after"],
               [(-10 ** 6, -1), (0, 0), (1, 1), (2, 2), (3, 3), (4, 6),
                (7, 12), (13, 24), (25, 10 ** 6)])
    emit()
    dp = pd.Series(R["dper"]).value_counts().sort_index()
    emit("period advance across the reset (months): %s"
         % ", ".join("%d: %s" % (k, fmt(v)) for k, v in dp.items()))
    emit()

    # ---- 4. calendar period -----------------------------------------------------
    rule("=")
    emit("4. WHEN RESETS OCCUR   (unit: resets, by calendar period of the reset row)")
    rule("=")
    ym = to_yyyymm(R["period"].astype(np.int64))
    year = ym // 100
    quarter = (ym % 100 - 1) // 3 + 1
    emit("%-10s%14s%10s%10s" % ("year", "resets", "share", ""))
    rule()
    for y, k in pd.Series(year).value_counts().sort_index().items():
        emit("%-10d%14s%10s   %s" % (y, fmt(k), pct(int(k), n_reset, 2),
                                     "#" * int(round(60.0 * k / n_reset))))
    rule()
    emit()
    emit("by calendar quarter (quarters with at least one reset):")
    rule()
    yq = pd.Series(year * 10 + quarter).value_counts().sort_index()
    row = []
    for v, k in yq.items():
        row.append("%dQ%d %8s" % (v // 10, v % 10, fmt(k)))
        if len(row) == 5:
            emit("    ".join(row))
            row = []
    if row:
        emit("    ".join(row))
    rule()
    emit()

    # ---- 5. unflagged reset loans -------------------------------------------------
    rule("=")
    emit("5. RESETS ON LOANS WITH NO MODIFICATION FLAG ON ANY ROW")
    rule("=")
    unf = ~R["mod_loan"]
    unf_loans = len(np.unique(key[unf]))
    emit("%-58s %18s %12s" % ("unflagged reset loans", fmt(unf_loans),
                              pct(unf_loans, n_reset_loans)))
    emit("%-58s %18s" % ("resets on them", fmt(int(unf.sum()))))
    emit("%-58s %18s" % ("resets with a blank flag on a MODIFIED loan",
                         fmt(int(((R["flag"] == F_BLANK) & R["mod_loan"]).sum()))))
    emit()
    emit("Compared with resets on modified loans (unit: resets):")
    rule()
    emit("%-52s%22s%22s" % ("", "modified loans", "unflagged loans"))
    rule()
    for label, f in [
        ("resets", lambda m: fmt(m.sum())),
        ("median drop (months)",
         lambda m: "%d" % np.median(drop[m]) if m.any() else "n/a"),
        ("median loan_age on the reset row",
         lambda m: "%d" % np.median(R["after"][m]) if m.any() else "n/a"),
        ("remaining months to maturity NULL on the row before",
         lambda m: pct(int((m & (R["rmlm_before"] == NA)).sum()),
                       int(m.sum()), 1)),
        ("remaining months to maturity INCREASES at reset",
         lambda m: pct(int((m & (R["rmlm_before"] != NA)
                            & (R["rmlm_after"] > R["rmlm_before"])).sum()),
                       int(m.sum()), 1)),
        ("current interest rate FALLS at reset",
         lambda m: pct(int((m & (R["rate_after"] < R["rate_before"])).sum()),
                       int(m.sum()), 1)),
        ("current interest rate unchanged at reset",
         lambda m: pct(int((m & (R["rate_after"] == R["rate_before"])).sum()),
                       int(m.sum()), 1)),
        ("status before reset delinquent (>= 1)",
         lambda m: pct(int((m & (R["status_before"] >= 1)).sum()),
                       int(m.sum()), 1)),
        ("status on reset row current (0)",
         lambda m: pct(int((m & (R["status_after"] == 0)).sum()),
                       int(m.sum()), 1)),
        ("zero-balance code on the reset row",
         lambda m: pct(int((m & (R["zb"] >= 0)).sum()), int(m.sum()), 1)),
        ("reset row is the loan's last row",
         lambda m: pct(int((m & (R["rows_after"] == 0)).sum()),
                       int(m.sum()), 1)),
    ]:
        emit("%-52s%22s%22s" % (label, f(R["mod_loan"]), f(unf)))
    rule()
    emit()
    idx = np.nonzero(unf)[0]
    emit("Unflagged resets, individually%s:"
         % ("" if len(idx) <= LIST_UNFLAGGED
            else " (first %d of %s)" % (LIST_UNFLAGGED, fmt(len(idx)))))
    rule()
    row_fmt = "%-14s%8s%12s%8s%12s%6s%18s%14s"
    emit(row_fmt % ("loan_id", "period", "age", "st", "rmlm", "zb", "rate",
                    "rows after"))
    emit(row_fmt % ("", "", "bef -> aft", "bef>aft", "bef->aft", "",
                    "bef->aft", ""))
    rule()
    for i in idx[:LIST_UNFLAGGED]:
        emit(row_fmt
             % (R["prefix"][i] + "%07d" % R["tail"][i], ym[i],
                "%d -> %d" % (R["before"][i], R["after"][i]),
                "%d>%d" % (R["status_before"][i], R["status_after"][i]),
                "%d->%d" % (R["rmlm_before"][i], R["rmlm_after"][i]),
                "%02d" % R["zb"][i] if R["zb"][i] >= 0 else "-",
                "%.3f->%.3f" % (R["rate_before"][i], R["rate_after"][i]),
                fmt(R["rows_after"][i])))
    rule()
    emit("st: delinquency status, -1 = non-numeric (RA). rmlm: remaining months")
    emit("to legal maturity. zb: zero-balance code on the reset row, '-' if none.")
    emit()

    # ---- 6. repeat resets ---------------------------------------------------------
    rule("=")
    emit("6. LOANS THAT RESET MORE THAN ONCE   (unit: loans)")
    rule("=")
    emit("%-22s%18s%12s" % ("resets per loan", "loans", "share"))
    rule()
    vc = reset_loans_per.value_counts().sort_index()
    for k, v in vc.items():
        emit("%-22s%18s%12s" % (k, fmt(v), pct(int(v), n_reset_loans)))
    rule()
    emit("%-22s%18s" % ("TOTAL", fmt(n_reset_loans)))
    emit("%-22s%18s%12s" % ("more than once", fmt(int(vc[vc.index > 1].sum())),
                            pct(int(vc[vc.index > 1].sum()), n_reset_loans)))
    emit()

    # ---- 7. panel ------------------------------------------------------------------
    rule("=")
    emit("7. MODIFIED LOANS IN THE PANEL")
    rule("=")
    from_counts, to_counts, in_panel = panel_pass(by_prefix)
    mod_in = sum(int(np.isin(f["mod_tails"], in_panel.get(f["prefix"],
                                                           np.empty(0))).sum())
                 for f in files)
    all_in = sum(len(v) for v in in_panel.values())
    unmod_total = n_loans - n_mod
    emit("%-58s%18s%18s%12s" % ("(unit: loans)", "loans", "in the panel",
                                "share"))
    rule()
    emit("%-58s%18s%18s%12s" % ("modified loans", fmt(n_mod), fmt(mod_in),
                                pct(mod_in, n_mod)))
    emit("%-58s%18s%18s%12s" % ("unmodified loans", fmt(unmod_total),
                                fmt(all_in - mod_in),
                                pct(all_in - mod_in, unmod_total)))
    emit("%-58s%18s%18s%12s" % ("all loans", fmt(n_loans), fmt(all_in),
                                pct(all_in, n_loans)))
    rule()
    emit()
    emit("Panel rows (unit: rows) by the loan's modification status. 'before' and")
    emit("'at/after' split a modified loan's rows at its first flagged period,")
    emit("by from month.")
    emit()
    state_table("FROM STATE", from_counts)
    state_table("TO STATE", to_counts)

    rule("=")
    emit("runtime: %.1f s  (%.2f min)"
         % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    write_out()


if __name__ == "__main__":
    main()
