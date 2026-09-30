"""Loan age versus calendar time in the fit pool's transition rates.

The baseline's yearly tables show the DQ30 cure rate falling from 2006 to 2008.
Two explanations are confounded there: the credit cycle, and a shift in the age
mix of the fit pool. This tabulates the rates by loan age x calendar year so the
two can be read apart. Diagnostic only - nothing downstream consumes it.

DEFINITIONS
    loan age        origin_period minus the loan's First Payment Date from the
                    origination file, in whole calendar months. A row whose
                    origin month IS the first payment month has age 0.
                    (The panel's own loan_age column counts from the month
                    BEFORE the first payment; the offset is printed as a check.)
    age bucket      0-11, 12-23, 24-35, 36+
    year            calendar year of origin_period; 2006, 2007, 2008 are
                    tabulated. Rows in any other year are listed separately and
                    enter no count or percentage.
    thin            a cell whose origin-row count is under 1,000. The DQ30 row
                    decides thinness in steps 1-4, the Current row in 5-6. Thin
                    cells print counts only; their shares are suppressed.
    structurally    a cell holding NO fit row of any origin state: no loan in
    absent          the pool had that age in that year. Distinct from zero.
    cure            DQ30 -> Current.   worsening   DQ30 -> DQ60

COUNTING
    Transitions are counted with src.baseline.count_transitions and each cell
    is normalised with src.baseline.probability_matrix, unchanged. That function
    refuses a matrix with an empty estimated row; if it refuses a cell, the
    refusal is printed and that cell's shares are not shown.

SOURCES
    panel           fit rows only, via src.panel_io.load_panel, one partition
                    at a time: loan_id, origin_period, origin_state, next_state,
                    loan_age (for the offset check only), role
    origination     loan identifier and first payment date, chunked, indices
                    resolved from the published layout by src.panel.resolve

Nothing is written except the report. No git command is run.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import baseline as bl  # noqa: E402
from src import panel  # noqa: E402
from src.panel_io import load_panel, read_manifest  # noqa: E402

ORIG_GLOB = panel.ORIG_GLOB
OUT = os.path.join("diagnostics", "22_age_vs_calendar.txt")
CHUNKSIZE = 2_000_000

YEARS = [2006, 2007, 2008]
AGE_EDGES = [0, 12, 24, 36]                      # lower edges; last is open
AGE_LABELS = ["0-11", "12-23", "24-35", "36+"]
THIN = 1000
COLUMNS = ["loan_id", "origin_period", "origin_state", "next_state", "loan_age",
           "role"]

S = {s: i for i, s in enumerate(bl.STATES)}
NA, NY, NS = len(AGE_LABELS), len(YEARS), bl.N_STATES

UNASSIGNED = ["loan_id not in any origination file",
              "first payment date missing or unparseable",
              "origin month before first payment month (age < 0)"]


# ==========================================================================
# origination: first payment month per loan
# ==========================================================================
def first_payment_lookup(oidx):
    """{prefix: (sorted tails, first-payment month index or -1)}."""
    c_loan, c_fp = oidx["loan"][0], oidx["first_payment"][0]
    parts = {}
    for path in sorted(glob.glob(ORIG_GLOB)):
        for chunk in pd.read_csv(path, sep="|", header=None, dtype=str,
                                 usecols=[c_loan, c_fp], chunksize=CHUNKSIZE,
                                 engine="c", na_filter=False):
            loan = chunk[c_loan].str.strip()
            fp = pd.to_numeric(chunk[c_fp].str.strip(), errors="coerce")
            mon = fp % 100
            idx = ((fp // 100) * 12 + mon - 1).where((mon >= 1) & (mon <= 12))
            frame = pd.DataFrame({
                "prefix": loan.str[:5].to_numpy(),
                "tail": pd.to_numeric(loan.str[5:], errors="coerce")
                          .fillna(-1).astype(np.int64).to_numpy(),
                "fp": idx.fillna(-1).astype(np.int64).to_numpy()})
            for p, g in frame.groupby("prefix"):
                parts.setdefault(p, []).append(g)
        print("    orig  %s" % os.path.basename(path), flush=True)
    out = {}
    for p, gs in parts.items():
        g = pd.concat(gs).sort_values("tail", kind="stable")
        if g["tail"].duplicated().any():
            raise SystemExit("duplicate loan id under prefix %s" % p)
        out[p] = (g["tail"].to_numpy(), g["fp"].to_numpy())
    return out


def lookup_fp(fp_by_prefix, loan_ids):
    """First-payment month index per loan id; -2 = not found, -1 = unusable."""
    prefix = loan_ids.str[:5].to_numpy()
    tail = pd.to_numeric(loan_ids.str[5:], errors="coerce").fillna(-1) \
        .astype(np.int64).to_numpy()
    out = np.full(len(loan_ids), -2, dtype=np.int64)
    for p in np.unique(prefix):
        m = prefix == p
        if p not in fp_by_prefix:
            continue
        tails, fps = fp_by_prefix[p]
        pos = np.searchsorted(tails, tail[m])
        safe = np.clip(pos, 0, len(tails) - 1)
        hit = tails[safe] == tail[m]
        out[np.nonzero(m)[0][hit]] = fps[safe][hit]
    return out


# ==========================================================================
# the single pass over the panel
# ==========================================================================
def scan(fp_by_prefix):
    manifest = read_manifest()
    acc = {"counts": np.zeros((NA, NY, NS, NS), dtype=np.int64),
           "fit_rows": 0, "null_state": 0,
           "unassigned": np.zeros(len(UNASSIGNED), dtype=np.int64),
           "unassigned_by_state": np.zeros((len(UNASSIGNED), NS), dtype=np.int64),
           "outside": [], "offset": {}, "negative_age": {},
           "build_stamp": manifest["build_stamp"]}
    for q in [p["quarter"] for p in manifest["partitions"]]:
        t0 = time.time()
        df = load_panel(columns=COLUMNS, quarters=[q])
        df = df[(df["role"] == bl.FIT_ROLE).to_numpy()].reset_index(drop=True)
        acc["fit_rows"] += len(df)
        o = bl._codes(df["origin_state"])
        d = bl._codes(df["next_state"])
        ok = (o >= 0) & (d >= 0)
        acc["null_state"] += int((~ok).sum())

        per = df["origin_period"].to_numpy().astype(np.int64)
        month = (per // 100) * 12 + per % 100 - 1
        fp = lookup_fp(fp_by_prefix, df["loan_id"].astype(str))
        age = month - fp
        reason = np.full(len(df), -1, dtype=np.int64)
        reason[fp == -2] = 0
        reason[fp == -1] = 1
        reason[(fp >= 0) & (age < 0)] = 2
        for k in range(len(UNASSIGNED)):
            m = ok & (reason == k)
            acc["unassigned"][k] += int(m.sum())
            acc["unassigned_by_state"][k] += np.bincount(o[m], minlength=NS)
        good = ok & (reason < 0)
        for v, n in pd.Series(age[ok & (reason == 2)]).value_counts().items():
            acc["negative_age"][int(v)] = acc["negative_age"].get(int(v), 0) + int(n)

        # offset of the panel's loan_age column against this definition
        pa = df["loan_age"].to_numpy(dtype="float64", na_value=np.nan)[good]
        diff = pd.Series(pa - age[good]).value_counts(dropna=False)
        for v, n in diff.items():
            key = "(null)" if pd.isna(v) else int(v)
            acc["offset"][key] = acc["offset"].get(key, 0) + int(n)

        year = per // 100
        yi = np.searchsorted(YEARS, year)
        in_years = good & np.isin(year, YEARS)
        out_years = good & ~np.isin(year, YEARS)
        for i in np.nonzero(out_years)[0]:
            acc["outside"].append((df["loan_id"].iat[i], int(per[i]), int(age[i]),
                                   bl.STATES[o[i]], bl.STATES[d[i]]))
        a = np.searchsorted(AGE_EDGES, age[in_years], side="right") - 1
        # one 6x6 count per (age, year) cell, through the baseline's counter
        cell = a * NY + yi[in_years]
        oo, dd = o[in_years], d[in_years]
        for c in np.unique(cell):
            m = cell == c
            acc["counts"][c // NY, c % NY] += bl.count_transitions(oo[m], dd[m])
        print("    panel %s  %s fit rows  (%.1fs)" % (q, bl.fmt(len(df)),
                                                    time.time() - t0), flush=True)
    return acc


def normalise(counts):
    """probability_matrix per non-absent cell; None plus the refusal where it
    refuses. Absent cells (no row at all) are not offered to it."""
    probs, refused = {}, {}
    for a in range(NA):
        for y in range(NY):
            if counts[a, y].sum() == 0:
                probs[a, y] = None
                continue
            try:
                probs[a, y] = bl.probability_matrix(counts[a, y])
            except bl.BaselineError as e:
                probs[a, y] = None
                refused[a, y] = str(e)
    return probs, refused


# ==========================================================================
# tables
# ==========================================================================
def cell_status(counts, a, y, origin):
    """'absent', 'thin' or '' for an (age, year) cell and origin row."""
    if counts[a, y].sum() == 0:
        return "absent"
    if counts[a, y, S[origin]].sum() < THIN:
        return "thin"
    return ""


def cell_text(counts, probs, a, y, origin, dest):
    """'p% (k/n)', or counts only when thin, or the absence marker."""
    st = cell_status(counts, a, y, origin)
    if st == "absent":
        return "structurally absent"
    k = counts[a, y, S[origin], S[dest]]
    n = counts[a, y, S[origin]].sum()
    if st == "thin":
        return "thin %s/%s" % (bl.fmt(k), bl.fmt(n))
    p = probs[a, y]
    if p is None:
        return "refused %s/%s" % (bl.fmt(k), bl.fmt(n))
    return "%.2f%% (%s/%s)" % (100.0 * p[S[origin], S[dest]], bl.fmt(k), bl.fmt(n))


def margin_text(k, n):
    """Margins pool cells; no probability_matrix cell exists for them."""
    if n == 0:
        return "no rows"
    return "%s (%s/%s)" % (bl.pct(k, n, 2), bl.fmt(k), bl.fmt(n))


def contrast(rep, counts, probs, origin, dest, title):
    """The two table shapes for one cell: calendar held, then age held."""
    w = 27
    rep.emit("%s   (%s -> %s)" % (title, origin, dest))
    rep.emit()
    rep.emit("(a) Calendar held, age varies - one row per year:")
    rep.rule()
    rep.emit("%-8s" % "year" + "".join("%*s" % (w, "age " + lab)
                                       for lab in AGE_LABELS))
    rep.rule()
    for y, yr in enumerate(YEARS):
        rep.emit("%-8d" % yr + "".join("%*s" % (w, cell_text(counts, probs, a, y,
                                                              origin, dest))
                                       for a in range(NA)))
    rep.rule()
    rep.emit()
    rep.emit("(b) Age held, calendar varies - one row per age bucket:")
    rep.rule()
    rep.emit("%-8s" % "age" + "".join("%*s" % (w, str(yr)) for yr in YEARS))
    rep.rule()
    for a, lab in enumerate(AGE_LABELS):
        rep.emit("%-8s" % lab + "".join("%*s" % (w, cell_text(counts, probs, a, y,
                                                               origin, dest))
                                        for y in range(NY)))
    rep.rule()
    rep.emit("%-8s" % "all" + "".join(
        "%*s" % (w, margin_text(counts[:, y, S[origin], S[dest]].sum(),
                                counts[:, y, S[origin]].sum()))
        for y in range(NY)))
    rep.emit("'all' pools every age bucket within the year (age-assigned rows only).")
    rep.emit()


def grid(rep, counts, probs, origin, dest, title):
    """One age x year table of a single cell, with margins."""
    w = 27
    rep.emit("%s   (%s -> %s; thin = %s row under %s)"
             % (title, origin, dest, origin, bl.fmt(THIN)))
    rep.rule()
    rep.emit("%-8s" % "age" + "".join("%*s" % (w, str(yr)) for yr in YEARS)
             + "%*s" % (w, "all years"))
    rep.rule()
    for a, lab in enumerate(AGE_LABELS):
        rep.emit("%-8s" % lab + "".join(
            "%*s" % (w, cell_text(counts, probs, a, y, origin, dest))
            for y in range(NY)) + "%*s" % (w, margin_text(
                counts[a, :, S[origin], S[dest]].sum(),
                counts[a, :, S[origin]].sum())))
    rep.rule()
    rep.emit("%-8s" % "all" + "".join(
        "%*s" % (w, margin_text(counts[:, y, S[origin], S[dest]].sum(),
                                counts[:, y, S[origin]].sum()))
        for y in range(NY)) + "%*s" % (w, margin_text(
            counts[:, :, S[origin], S[dest]].sum(), counts[:, :, S[origin]].sum())))
    rep.emit()


# ==========================================================================
def main():
    t0 = time.time()
    rep = bl.Report()
    _, oidx, _, orig_sheet = panel.resolve()
    fp_by_prefix = first_payment_lookup(oidx)
    acc = scan(fp_by_prefix)
    C = acc["counts"]
    probs, refused = normalise(C)

    rep.rule("=")
    rep.emit("22_age_vs_calendar - loan age versus calendar year, fit rows only")
    rep.rule("=")
    rep.emit("panel       : data/panel/panel_*.parquet via src.panel_io.load_panel"
             " (build %s)" % acc["build_stamp"])
    rep.emit("origination : %s" % ORIG_GLOB)
    rep.emit("counting    : src.baseline.count_transitions, "
             "src.baseline.probability_matrix")
    rep.emit()
    rep.emit("LOAN AGE = origin_period - First Payment Date (origination file), in")
    rep.emit("calendar months. Age 0 = the origin month is the first payment month.")
    rep.emit("buckets     : %s" % ", ".join(AGE_LABELS))
    rep.emit("years       : %s (calendar year of origin_period)"
             % ", ".join(map(str, YEARS)))
    rep.emit("thin        : origin-row count under %s; shares suppressed, counts kept"
             % bl.fmt(THIN))
    rep.emit("absent      : no fit row of any origin state in the age x year cell")
    rep.emit()
    rep.emit("RESOLVED ORIGINATION COLUMNS   (0-based index = published position - 1)")
    rep.rule()
    for label in ("loan", "first_payment"):
        i, pos, name = oidx[label]
        rep.emit("%-16s%10d%8d   %s" % (label, pos, i, name))
    rep.emit("sheet: %s" % orig_sheet)
    rep.emit()

    rep.rule("=")
    rep.emit("FIT ROWS AND AGE ASSIGNMENT   (unit: fit rows)")
    rep.rule("=")
    n_fit = acc["fit_rows"]
    tabulated = int(C.sum())
    n_out = len(acc["outside"])
    n_un = int(acc["unassigned"].sum())
    rep.emit("%-64s%16s%12s" % ("fit rows in the panel", bl.fmt(n_fit), "100.0000%"))
    rep.emit("%-64s%16s%12s" % ("null or unmapped state (not age-related)",
                                bl.fmt(acc["null_state"]),
                                bl.pct(acc["null_state"], n_fit)))
    rep.emit("%-64s%16s%12s" % ("could not be assigned an age", bl.fmt(n_un),
                                bl.pct(n_un, n_fit)))
    for k, why in enumerate(UNASSIGNED):
        rep.emit("%-64s%16s%12s" % ("    " + why, bl.fmt(acc["unassigned"][k]),
                                    bl.pct(acc["unassigned"][k], n_fit)))
    rep.emit("%-64s%16s%12s" % ("assigned an age, year outside %s-%s"
                                % (YEARS[0], YEARS[-1]), bl.fmt(n_out),
                                bl.pct(n_out, n_fit)))
    rep.emit("%-64s%16s%12s" % ("tabulated below", bl.fmt(tabulated),
                                bl.pct(tabulated, n_fit)))
    ident = acc["null_state"] + n_un + n_out + tabulated == n_fit
    rep.emit("%-64s%16s" % ("identity: null + unassigned + outside + tabulated "
                            "== fit rows", "PASS" if ident else "FAIL"))
    rep.emit()
    if n_un:
        rep.emit("Unassigned rows by origin state:")
        rep.rule()
        rep.emit("%-52s" % "reason" + "".join("%11s" % s for s in bl.STATES))
        rep.rule()
        for k, why in enumerate(UNASSIGNED):
            rep.emit("%-52s" % why[:51] + "".join(
                "%11s" % bl.fmt(v) for v in acc["unassigned_by_state"][k]))
        rep.rule()
        rep.emit()
        rep.emit("Age values of the rows with age < 0:")
        rep.rule()
        n_neg = sum(acc["negative_age"].values())
        for v in sorted(acc["negative_age"], reverse=True):
            rep.emit("%-20s%16s%12s" % ("age %d" % v, bl.fmt(acc["negative_age"][v]),
                                        bl.pct(acc["negative_age"][v], n_neg)))
        rep.rule()
        rep.emit("%-20s%16s" % ("TOTAL", bl.fmt(n_neg)))
        rep.emit()
    rep.emit("Rows in years outside %s-%s, listed individually; they enter no count"
             % (YEARS[0], YEARS[-1]))
    rep.emit("or percentage in the tables below:")
    rep.rule()
    rep.emit("%-16s%10s%8s%10s  %s" % ("loan_id", "period", "age", "bucket",
                                       "transition"))
    rep.rule()
    for lid, per, age, o, d in acc["outside"]:
        b = AGE_LABELS[np.searchsorted(AGE_EDGES, age, side="right") - 1]
        rep.emit("%-16s%10d%8d%10s  %s -> %s" % (lid, per, age, b, o, d))
    if not acc["outside"]:
        rep.emit("(none)")
    rep.rule()
    rep.emit()
    rep.emit("Check: the panel's loan_age column minus this age (tabulated rows):")
    rep.rule()
    tot_off = sum(acc["offset"].values())
    for k in sorted(acc["offset"], key=lambda v: (isinstance(v, str), v)):
        rep.emit("%-20s%16s%12s" % (k, bl.fmt(acc["offset"][k]),
                                    bl.pct(acc["offset"][k], tot_off)))
    rep.rule()
    rep.emit()
    if refused:
        rep.emit("probability_matrix refused these cells (shares not shown):")
        for (a, y), msg in sorted(refused.items()):
            rep.emit("    age %-6s %d  %s" % (AGE_LABELS[a], YEARS[y], msg))
    else:
        rep.emit("probability_matrix accepted every non-absent cell.")
    rep.emit()

    # ---- 1 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("1. COVERAGE GRID - DQ30-origin fit rows by age x year   (unit: rows)")
    rep.rule("=")
    rep.emit("Read this before any rate: it decides which comparisons are readable.")
    rep.rule()
    w = 22
    rep.emit("%-8s" % "age" + "".join("%*s" % (w, str(yr)) for yr in YEARS)
             + "%*s" % (w, "row total"))
    rep.rule()
    dq = C[:, :, S["DQ30"]].sum(axis=-1)            # (age, year)
    for a, lab in enumerate(AGE_LABELS):
        cells = []
        for y in range(NY):
            st = cell_status(C, a, y, "DQ30")
            cells.append("structurally absent" if st == "absent" else
                         bl.fmt(dq[a, y]) + ("  THIN" if st == "thin" else ""))
        rep.emit("%-8s" % lab + "".join("%*s" % (w, c) for c in cells)
                 + "%*s" % (w, bl.fmt(dq[a].sum())))
    rep.rule()
    rep.emit("%-8s" % "total" + "".join("%*s" % (w, bl.fmt(dq[:, y].sum()))
                                        for y in range(NY))
             + "%*s" % (w, bl.fmt(dq.sum())))
    rep.emit()
    rep.emit("All-state fit rows per cell (the basis of 'structurally absent'):")
    rep.rule()
    allr = C.sum(axis=(2, 3))
    rep.emit("%-8s" % "age" + "".join("%*s" % (w, str(yr)) for yr in YEARS)
             + "%*s" % (w, "row total"))
    rep.rule()
    for a, lab in enumerate(AGE_LABELS):
        rep.emit("%-8s" % lab + "".join("%*s" % (w, bl.fmt(allr[a, y]))
                                        for y in range(NY))
                 + "%*s" % (w, bl.fmt(allr[a].sum())))
    rep.rule()
    rep.emit("%-8s" % "total" + "".join("%*s" % (w, bl.fmt(allr[:, y].sum()))
                                        for y in range(NY))
             + "%*s" % (w, bl.fmt(allr.sum())))
    rep.emit()

    # ---- 2 ---------------------------------------------------------------------
    rep.rule("=")
    rep.emit("2. THE FULL DQ30 ROW BY AGE x YEAR   (unit: DQ30-origin fit rows)")
    rep.rule("=")
    rep.emit("Counts on the first line, shares (from probability_matrix) beneath.")
    rep.emit("Thin cells: counts only.")
    rep.rule()
    w = 13
    rep.emit("%-8s%-6s" % ("age", "year") + "".join("%*s" % (w, s) for s in bl.STATES)
             + "%*s" % (w + 1, "total") + "  note")
    rep.rule()
    i30 = S["DQ30"]
    for a, lab in enumerate(AGE_LABELS):
        for y, yr in enumerate(YEARS):
            st = cell_status(C, a, y, "DQ30")
            row = C[a, y, i30]
            if st == "absent":
                rep.emit("%-8s%-6d%s" % (lab, yr, "structurally absent"))
                continue
            note = {"thin": "THIN - shares suppressed", "": ""}[st]
            if st == "" and probs[a, y] is None:
                note = "probability_matrix refused"
            rep.emit("%-8s%-6d" % (lab, yr) + "".join("%*s" % (w, bl.fmt(c))
                                                      for c in row)
                     + "%*s" % (w + 1, bl.fmt(row.sum())) + "  " + note)
            if st == "" and probs[a, y] is not None:
                rep.emit("%-14s" % "  p" + "".join(
                    "%*s" % (w, "%.4f%%" % (100.0 * v)) for v in probs[a, y][i30]))
        rep.rule()
    rep.emit()

    # ---- 3 - 6 -----------------------------------------------------------------
    rep.rule("=")
    rep.emit("3. CURE CONTRAST")
    rep.rule("=")
    contrast(rep, C, probs, "DQ30", "Current", "Cure")
    rep.rule("=")
    rep.emit("4. WORSENING CONTRAST")
    rep.rule("=")
    contrast(rep, C, probs, "DQ30", "DQ60", "Worsening")
    rep.rule("=")
    rep.emit("5. CURRENT -> DQ30 BY AGE x YEAR")
    rep.rule("=")
    grid(rep, C, probs, "Current", "DQ30", "Roll to DQ30")
    rep.rule("=")
    rep.emit("6. CURRENT -> PREPAID BY AGE x YEAR")
    rep.rule("=")
    grid(rep, C, probs, "Current", "Prepaid", "Prepayment")
    rep.emit("Cells read 'p (count/origin-row total)'. 'all' margins pool cells and")
    rep.emit("are row totals divided directly, since no cell matrix exists for them.")
    rep.emit()

    elapsed = time.time() - t0
    rep.rule("=")
    rep.emit("runtime: %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    rep.rule("=")
    rep.write(OUT)


if __name__ == "__main__":
    main()
