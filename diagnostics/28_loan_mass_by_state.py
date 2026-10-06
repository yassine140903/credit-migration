"""Loan mass by property state.   READ-ONLY.

    python diagnostics/28_loan_mass_by_state.py   # from the repo root

Writes diagnostics/28_loan_mass_by_state.txt and nothing else. It reads the panel
through src.panel_io.load_panel (one partition at a time, five columns), writes no
data file, joins no HPI file, drops or filters nothing, and runs no git command.

UNITS. Loans are distinct loan_id; rows are panel rows (loan-months). A loan lives
in exactly one partition (asserted: every loan_id carries its partition's vintage
prefix) and in exactly one property_state (asserted), so per-state loan counts add
to the panel total exactly. By role, a LOAN is counted in a role when it has at
least one row in that role: a test-split loan has condition and evaluate rows, so
role loan counts do not add to the loan total.

DEFAULT. A loan "reaches Default" if it has at least one row with to_state ==
Default. Rows with from_state == Default are cross-checked and any loan that has
one without a to_state == Default row is reported.

property_state values are tallied verbatim (as they appear, nulls included) and
anything that is not exactly two uppercase letters is flagged, never normalised.
"""

import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel                                                   # noqa: E402
from src.panel_io import load_panel, read_manifest                      # noqa: E402

OUT = os.path.join("diagnostics", "28_loan_mass_by_state.txt")
HPI_CODES_CSV = os.path.join("data", "macro", "census_divisions.csv")   # code list only, no join
RULE_W = 112
LOW_VOLUME = ["AK", "DC", "DE", "HI", "VT", "WV", "WY"]
TERRITORIES = ["GU", "PR", "VI"]
REF_TERRITORY_LOANS = 10073        # diagnostics/23_origination_validity.txt, GU+PR+VI loans
REF_TERRITORY_ROWS = 400089        # same file, GU+PR+VI panel rows
NULL = "<NULL>"
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def rule(char="-"):
    emit(char * RULE_W)


def title(text):
    emit()
    rule("=")
    emit(text)
    rule("=")


def pct(x, tot):
    return 100.0 * x / tot if tot else float("nan")


def scan():
    """One pass over the partitions. Returns the accumulators."""
    manifest = read_manifest()
    rows_by_value = Counter()                 # verbatim value -> rows
    loans_by_value = Counter()                # verbatim value -> loans
    rows_role = defaultdict(Counter)          # role -> value -> rows
    loans_role = defaultdict(Counter)         # role -> value -> loans
    default_loans = Counter()                 # value -> loans reaching Default
    notes = {"multi_state_loans": 0, "bad_prefix": 0, "from_default_only": 0,
             "from_default_rows": 0, "to_default_rows": 0, "loans_seen": 0}
    for p in manifest["partitions"]:
        q = p["quarter"]
        d = load_panel(quarters=[q], columns=["loan_id", "property_state", "role",
                                              "from_state", "to_state"])
        assert len(d) == p["rows"], (q, len(d), p["rows"])
        prefix = "F" + q[2:4] + q[4:]                       # 2007Q3 -> F07Q3
        notes["bad_prefix"] += int((~d["loan_id"].str.startswith(prefix)).sum())
        lid = pd.factorize(d["loan_id"])[0]
        s = d["property_state"].astype(object)
        assert not (s == NULL).any(), "a real property_state value collides with the null sentinel"
        s = s.where(s.notna(), NULL)
        st, labels = pd.factorize(s)
        labels = list(labels)
        roles = d["role"].astype(str).to_numpy()
        to_def = (d["to_state"] == "Default").to_numpy()
        from_def = (d["from_state"] == "Default").to_numpy()
        notes["to_default_rows"] += int(to_def.sum())
        notes["from_default_rows"] += int(from_def.sum())
        f = pd.DataFrame({"lid": lid, "st": st, "role": roles, "to_def": to_def, "from_def": from_def})
        del d, s

        notes["loans_seen"] += int(f["lid"].nunique())
        per_loan = f.groupby("lid").agg(n_st=("st", "nunique"), st=("st", "first"),
                                        to_def=("to_def", "max"), from_def=("from_def", "max"))
        notes["multi_state_loans"] += int((per_loan["n_st"] > 1).sum())
        notes["from_default_only"] += int((per_loan["from_def"] & ~per_loan["to_def"]).sum())
        for code, n in per_loan.groupby("st").size().items():
            loans_by_value[labels[code]] += int(n)
        for code, n in per_loan[per_loan["to_def"]].groupby("st").size().items():
            default_loans[labels[code]] += int(n)
        for code, n in f.groupby("st").size().items():
            rows_by_value[labels[code]] += int(n)
        for (r, code), n in f.groupby(["role", "st"]).size().items():
            rows_role[r][labels[code]] += int(n)
        for (r, code), n in f.drop_duplicates(["lid", "role", "st"]).groupby(["role", "st"]).size().items():
            loans_role[r][labels[code]] += int(n)
        print(f"  scanned {q}: {p['rows']:,} rows", flush=True)
    return rows_by_value, loans_by_value, rows_role, loans_role, default_loans, notes, manifest


def table(keys, loans, rows, tot_loans, tot_rows, label="state"):
    """Per-state table sorted by loans, largest first, with a total row."""
    emit(f"{label:<10}{'loans':>12}{'% loans':>10}{'rows':>14}{'% rows':>10}")
    for k in sorted(keys, key=lambda k: (-loans.get(k, 0), k)):
        emit(f"{k:<10}{loans.get(k, 0):>12,}{pct(loans.get(k, 0), tot_loans):>9.4f}%"
             f"{rows.get(k, 0):>14,}{pct(rows.get(k, 0), tot_rows):>9.4f}%")
    rule()
    emit(f"{'TOTAL':<10}{tot_loans:>12,}{pct(tot_loans, tot_loans):>9.4f}%"
         f"{tot_rows:>14,}{pct(tot_rows, tot_rows):>9.4f}%")


def subtotal(name, codes, loans, rows, tot_loans, tot_rows):
    l = sum(loans.get(c, 0) for c in codes)
    r = sum(rows.get(c, 0) for c in codes)
    emit(f"{name:<22}{l:>12,}{pct(l, tot_loans):>9.4f}%{r:>14,}{pct(r, tot_rows):>9.4f}%")
    return l, r


def main():
    emit("Loan mass by property state   (transition panel, validated against its manifest by load_panel)")
    rows_v, loans_v, rows_role, loans_role, def_loans, notes, manifest = scan()
    tot_rows, tot_loans = sum(rows_v.values()), sum(loans_v.values())
    emit(f"panel build {manifest['build_stamp']}   partitions {manifest['n_partitions']}   "
         f"manifest rows {manifest['total_rows']:,}")
    emit(f"rows counted {tot_rows:,}   loans counted {tot_loans:,}")
    assert tot_rows == manifest["total_rows"]
    assert notes["bad_prefix"] == 0, "a loan_id lacks its partition's vintage prefix"
    assert notes["multi_state_loans"] == 0, "a loan has more than one property_state"
    assert tot_loans == notes["loans_seen"]
    emit("integrity: rows equal the manifest; every loan_id carries its partition's vintage prefix "
         "(no loan spans partitions);")
    emit("           no loan has more than one property_state, so state loan counts add to the total exactly.")

    # ------------------------------------------------------------ 1 verbatim
    title("1. property_state VALUES AS THEY APPEAR  (verbatim, repr; pre-join check)")
    ok = re.compile(r"[A-Z]{2}")
    expected = set(panel.PROPERTY_STATES)
    hpi_codes = set(pd.read_csv(HPI_CODES_CSV)["state_code"])
    emit(f"{'value (repr)':<14}{'rows':>14}{'loans':>12}  flag")
    flagged = []
    for v in sorted(rows_v, key=lambda v: (-loans_v[v], v)):
        raw = None if v == NULL else v
        flag = ""
        if raw is None:
            flag = "FLAG: null"
        elif not ok.fullmatch(raw):
            flag = "FLAG: not two uppercase letters"
        elif raw not in expected:
            flag = "FLAG: two letters but not one of the 54 codes the build allows"
        if flag:
            flagged.append((v, flag))
        emit(f"{repr(raw):<14}{rows_v[v]:>14,}{loans_v[v]:>12,}  {flag}")
    emit()
    emit(f"distinct values: {len(rows_v)}   flagged: {len(flagged)}"
         + (f"   {flagged}" if flagged else "   (none: every value is a two-letter uppercase code)"))
    seen = {v for v in rows_v if v != NULL and ok.fullmatch(v)}
    emit(f"allowed codes (src.panel.PROPERTY_STATES, 54) absent from the panel: "
         f"{sorted(expected - seen) if expected - seen else 'none'}")
    emit(f"HPI-side code list ({HPI_CODES_CSV}, {len(hpi_codes)} codes) vs panel:")
    emit(f"  panel codes with no HPI-side counterpart: {sorted(seen - hpi_codes)}")
    emit(f"  HPI-side codes absent from the panel:     {sorted(hpi_codes - seen) if hpi_codes - seen else 'none'}")
    emit(f"  panel codes matching the HPI side exactly: {len(seen & hpi_codes)} of {len(hpi_codes)}")

    # ------------------------------------------------------------ 2 per state
    title("2. PER STATE  (distinct loans, rows; % of panel total; loans descending)")
    table(rows_v.keys(), loans_v, rows_v, tot_loans, tot_rows)

    # ------------------------------------------------------------ 3 by role
    title("3. PER STATE BY ROLE  (loan counted in a role if it has >=1 row there; % of that role's total)")
    for r in panel.ROLES:
        rl, rr = loans_role[r], rows_role[r]
        emit()
        emit(f"ROLE {r}")
        table(rows_v.keys(), rl, rr, sum(rl.values()), sum(rr.values()))
        absent = sorted(k for k in rows_v if rl.get(k, 0) == 0)
        emit(f"values with no loans in role {r}: {absent if absent else 'none'}")
    emit()
    emit("Loan totals by role: " + "   ".join(f"{r} {sum(loans_role[r].values()):,}" for r in panel.ROLES)
         + f"   (sum {sum(sum(loans_role[r].values()) for r in panel.ROLES):,} vs {tot_loans:,} loans: "
         "test-split loans are in two roles)")

    # ------------------------------------------------------------ 4 low volume
    title("4. THE SEVEN FHFA LOW-VOLUME STATES  (AK DC DE HI VT WV WY)")
    emit(f"{'':<22}{'loans':>12}{'% loans':>10}{'rows':>14}{'% rows':>10}")
    for c in sorted(LOW_VOLUME, key=lambda c: -loans_v.get(c, 0)):
        subtotal(c, [c], loans_v, rows_v, tot_loans, tot_rows)
    rule()
    sl, sr = subtotal("SEVEN COMBINED", LOW_VOLUME, loans_v, rows_v, tot_loans, tot_rows)
    over = [c for c in LOW_VOLUME if pct(loans_v.get(c, 0), tot_loans) > 1.0]
    emit()
    emit(f"Largest single one of the seven: "
         f"{max(LOW_VOLUME, key=lambda c: loans_v.get(c, 0))} at "
         f"{pct(max(loans_v.get(c, 0) for c in LOW_VOLUME), tot_loans):.4f}% of loans.")
    emit("Does any individual one of the seven exceed 1% of loans? "
         + (f"YES: {over}" if over else "NO. None of the seven individually exceeds 1% of loans."))

    # ------------------------------------------------------------ 5 territories
    title("5. TERRITORIES  (GU, PR, VI)")
    emit(f"{'':<22}{'loans':>12}{'% loans':>10}{'rows':>14}{'% rows':>10}")
    for c in sorted(TERRITORIES, key=lambda c: -loans_v.get(c, 0)):
        subtotal(c, [c], loans_v, rows_v, tot_loans, tot_rows)
    rule()
    tl, tr = subtotal("GU+PR+VI COMBINED", TERRITORIES, loans_v, rows_v, tot_loans, tot_rows)
    emit()
    emit(f"On record (diagnostic 23): {REF_TERRITORY_LOANS:,} loans, {REF_TERRITORY_ROWS:,} rows.")
    emit(f"This panel:               {tl:,} loans, {tr:,} rows.")
    emit(f"Loans: {'MATCH' if tl == REF_TERRITORY_LOANS else 'DIFFERENT by %+d' % (tl - REF_TERRITORY_LOANS)}.   "
         f"Rows: {'MATCH' if tr == REF_TERRITORY_ROWS else 'DIFFERENT by %+d' % (tr - REF_TERRITORY_ROWS)}.")
    emit(f"Loans left after excluding the territories: {tot_loans - tl:,} of {tot_loans:,}.")

    # ------------------------------------------------------------ 6 default
    title("6. CREDIT-EVENT CONCENTRATION  (loans with >=1 row where to_state == Default)")
    tot_def = sum(def_loans.values())
    emit(f"loans reaching Default {tot_def:,} of {tot_loans:,} = {pct(tot_def, tot_loans):.4f}%   "
         f"(rows with to_state=Default {notes['to_default_rows']:,}; rows with from_state=Default "
         f"{notes['from_default_rows']:,}; loans with a from_state=Default row but no to_state=Default row "
         f"{notes['from_default_only']:,})")
    emit("default rate = Default loans / that state's loans.  concentration = share of Default loans / "
         "share of loans (1.0 = proportional;")
    emit(">1 = more than its size would imply).")
    emit()
    emit(f"{'state':<8}{'loans':>11}{'% loans':>9}{'default':>10}{'% of defaults':>15}{'default rate':>14}{'concentration':>15}")
    for k in sorted(rows_v, key=lambda k: (-def_loans.get(k, 0), k)):
        dl, ln = def_loans.get(k, 0), loans_v[k]
        conc = (pct(dl, tot_def) / pct(ln, tot_loans)) if ln else float("nan")
        emit(f"{k:<8}{ln:>11,}{pct(ln, tot_loans):>8.3f}%{dl:>10,}{pct(dl, tot_def):>14.3f}%"
             f"{pct(dl, ln):>13.2f}%{conc:>15.2f}")
    rule()
    emit(f"{'TOTAL':<8}{tot_loans:>11,}{100.0:>8.3f}%{tot_def:>10,}{100.0:>14.3f}%{pct(tot_def, tot_loans):>13.2f}%{1.0:>15.2f}")
    emit()
    emit("Subtotals:")
    for name, codes in (("seven low-volume", LOW_VOLUME), ("territories", TERRITORIES)):
        ln = sum(loans_v.get(c, 0) for c in codes)
        dl = sum(def_loans.get(c, 0) for c in codes)
        emit(f"  {name:<18}loans {ln:>10,} ({pct(ln, tot_loans):.3f}%)   default loans {dl:>8,} "
             f"({pct(dl, tot_def):.3f}% of all)   default rate {pct(dl, ln):.2f}%   "
             f"concentration {pct(dl, tot_def) / pct(ln, tot_loans):.2f}")
    top = sorted(rows_v, key=lambda k: (-def_loans.get(k, 0), k))[:5]
    emit(f"  top-5 states by default loans: {', '.join(top)} = "
         f"{pct(sum(def_loans.get(k, 0) for k in top), tot_def):.2f}% of all Default loans")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
