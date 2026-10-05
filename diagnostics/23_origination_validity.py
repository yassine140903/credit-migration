"""Territory coverage and origination-record validity.   READ-ONLY.

    python diagnostics/23_origination_validity.py        # from the repo root

Writes diagnostics/23_origination_validity.txt and nothing else. It does not
modify the panel, src/panel.py or any tracked file, changes no value and drops
no loan.

PART 1  Territory coverage. GU, PR and VI, separately and combined: loans and
        panel rows, share of the panel, rows and loans by role, by origination
        quarter. A state-level house-price index covers the states and DC only.

PART 2  Origination-record validity. Every value of every origination field the
        build reads (plus Number of Units) is put in exactly ONE bucket, per
        loan, across all eight origination files:
            VALID      parses and lies in the plausible range
            SENTINEL   a documented Freddie "not available" code
            MALFORMED  does not parse, or lies outside both the plausible range
                       and the documented sentinel set

PART 3  How the panel currently holds those values: for fico, ltv, dti,
        rate_spread and property_state, what the panel stores for loans in each
        bucket - null, the raw sentinel number, the raw (malformed) value, or
        something else - with counts, and the src/panel.py lines responsible.

SOURCES FOR THE BUCKET DEFINITIONS - docs/general_user_guide_july_2026.pdf,
Origination Data File table (read from the page, not from memory):
    field            documented range             documented sentinel
    Classic FICO     300 - 850                    9999 = Not Available
    First Payment    YYYYMM                       none documented
    DTI              0 - 65 (percent)             999 = Not Available
    LTV              6 - 105 (2018Q1 and prior)   999 = Not Available
    Interest Rate    none documented              none documented
    Loan Identifier  PYYQnXXXXXXX                 none documented
    Property State   "AL, TX, VA, etc."           none documented
    MSA              five-digit code              null (blank) = not an MSA /
                                                  Metropolitan Division, or unknown
    Postal Code      ### (first three digits)     000 = Unknown
    Number of Units  1 - 4                        99 = Not Available
Where the guide gives no range, the "plausible" range below is THIS SCRIPT'S
definition, stated here so it can be argued with:
    first payment   YYYYMM, month 1-12, year 1999-2021 (the dataset's documented
                    origination span, 1999 to the 30 June 2021 cutoff)
    interest rate   a number with 0 < rate <= 25 (percent)
    loan id         [FA]YYQnNNNNNNN, with YYQn equal to the file's quarter
    property state  the 56 USPS state / DC / territory codes (the 50 states, DC,
                    AS, GU, MP, PR, VI). Blank is NOT documented, so it is
                    MALFORMED, not SENTINEL.
Values are compared after stripping whitespace.

Dependencies: src.panel (layout, panel constants, PMMS lookup), src.panel_io
(the validated loader), src.horizon (the harness universe, fingerprint only).
The PMMS series is read from its cache, data/macro/MORTGAGE30US.csv; the script
asserts the file is unchanged when it finishes.
"""

import glob
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import horizon as hz, panel                                  # noqa: E402
from src.horizon_nulls import fingerprint, saved_fingerprint          # noqa: E402
from src.panel_io import load_panel, read_manifest, validate          # noqa: E402

OUT = os.path.join("diagnostics", "23_origination_validity.txt")
RULE_W = 121
WANT_FP = "3b1209189824e42f"
V, S, M = 0, 1, 2
BUCKET = ["VALID", "SENTINEL", "MALFORMED"]
TERRITORIES = ["GU", "PR", "VI"]
USPS = set("AK AL AR AZ CA CO CT DC DE FL GA GU HI IA ID IL IN KS KY LA MA MD ME "
           "MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX "
           "UT VA VI VT WA WI WV WY AS MP".split())
KNOWN_LOAN = "F06Q30148547"
CHECK_LOAN_TAIL = int(KNOWN_LOAN[5:])

# field -> (label in the layout, documented sentinel text)
FIELDS = [
    ("fico", "classic fico", "9999"),
    ("first_payment", "first payment date", "none documented"),
    ("dti", "original debt-to-income (dti) ratio", "999"),
    ("ltv", "original loan-to-value (ltv)", "999"),
    ("orig_rate", "original interest rate", "none documented"),
    ("loan", "loan identifier", "none documented"),
    ("property_state", "property state", "none documented"),
    ("msa", "metropolitan statistical area (msa) or metropolitan division",
     "blank (null)"),
    ("postal_code", "postal code", "000"),
    ("units", "number of units", "99"),
]
NAMES = [f[0] for f in FIELDS]
RANGE_TEXT = {
    "fico": "300-850", "first_payment": "YYYYMM, 1999-2021", "dti": "0-65",
    "ltv": "6-105", "orig_rate": "0 < r <= 25 (script)", "loan": "[FA]YYQnNNNNNNN",
    "property_state": "56 USPS codes", "msa": "5 digits", "postal_code": "3 digits",
    "units": "1-4"}

LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def rule(char="-", width=RULE_W):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


# ==========================================================================
# layout
# ==========================================================================
def resolve_fields():
    fields, _ = panel._sheet_rows(panel.LAYOUT)[panel.ORIG_SHEET]
    out = {}
    for name, want, _ in FIELDS:
        w = panel._normalise(want)
        hits = [f for f in fields if panel._normalise(f[1]) == w]
        if len(hits) != 1:
            raise SystemExit("layout lookup for %r matched %d rows" % (want, len(hits)))
        out[name] = hits[0][0] - 1
    return out


# ==========================================================================
# classification - one bucket per value
# ==========================================================================
def _num(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def classify_range(s, lo, hi, sentinel=None):
    x = _num(s)
    out = np.full(len(x), M, dtype=np.int8)
    ok = np.isfinite(x)
    out[ok & (x >= lo) & (x <= hi)] = V
    if sentinel is not None:
        out[ok & (x == sentinel)] = S
    return out, x


def classify(name, s, file_quarter):
    """(bucket codes, parsed numeric values or None) for one stripped column."""
    if name == "fico":
        return classify_range(s, 300, 850, 9999)
    if name == "dti":
        return classify_range(s, 0, 65, 999)
    if name == "ltv":
        return classify_range(s, 6, 105, 999)
    if name == "units":
        x = _num(s)
        out = np.full(len(x), M, dtype=np.int8)
        ok = np.isfinite(x)
        out[ok & np.isin(x, [1, 2, 3, 4])] = V
        out[ok & (x == 99)] = S
        return out, x
    if name == "orig_rate":
        x = _num(s)
        out = np.full(len(x), M, dtype=np.int8)
        out[np.isfinite(x) & (x > 0) & (x <= 25)] = V
        return out, x
    out = np.full(len(s), M, dtype=np.int8)
    if name == "first_payment":
        ok = s.str.fullmatch(r"\d{6}").to_numpy(dtype=bool)
        yr = pd.to_numeric(s.str[:4], errors="coerce").to_numpy()
        mo = pd.to_numeric(s.str[4:], errors="coerce").to_numpy()
        out[ok & (mo >= 1) & (mo <= 12) & (yr >= 1999) & (yr <= 2021)] = V
        return out, _num(s)
    if name == "loan":
        ok = s.str.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}").to_numpy(dtype=bool)
        ok &= (s.str[1:5] == file_quarter).to_numpy(dtype=bool)
        out[ok] = V
        return out, None
    if name == "property_state":
        out[s.isin(USPS).to_numpy()] = V
        return out, None
    if name == "msa":
        out[s.str.fullmatch(r"\d{5}").to_numpy(dtype=bool)] = V
        out[(s == "").to_numpy()] = S
        return out, None
    if name == "postal_code":
        ok = s.str.fullmatch(r"\d{3}").to_numpy(dtype=bool)
        out[ok & (s != "000").to_numpy()] = V
        out[(s == "000").to_numpy()] = S
        return out, None
    raise KeyError(name)


# ==========================================================================
# panel scan helpers
# ==========================================================================
def source_lines(patterns):
    """{key: (line number, stripped text)} for the first line of src/panel.py
    matching each regex - cited from the file as it is now."""
    with open(os.path.join("src", "panel.py"), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out = {}
    for key, rx in patterns.items():
        for i, ln in enumerate(lines, 1):
            if re.search(rx, ln):
                out[key] = (i, ln.strip())
                break
        else:
            out[key] = (None, "NOT FOUND: %s" % rx)
    return out


def held_class(held, raw, sentinel):
    """Counts [null, == raw sentinel number, == raw parsed value, other] of a
    float32 panel column vs the raw parsed values, over one bucket."""
    h = held.astype(np.float64)
    isnull = np.isnan(h)
    is_sent = ~isnull & (sentinel is not None) & (h == (sentinel if sentinel is not None else np.nan))
    raw32 = raw.astype(np.float32).astype(np.float64)
    is_raw = ~isnull & ~is_sent & (h == raw32)
    other = ~isnull & ~is_sent & ~is_raw
    return np.array([isnull.sum(), is_sent.sum(), is_raw.sum(), other.sum()], dtype=np.int64), other


def main():
    t0 = time.time()
    pmms_path = os.path.join(panel.MACRO_DIR, panel.PMMS + ".csv")
    pmms_stat = os.stat(pmms_path)
    pidx = resolve_fields()
    cols = sorted(set(pidx.values()))
    inv = {v: k for k, v in pidx.items()}

    man = read_manifest()
    validate(man)                                   # the panel must be the trusted one
    pmms, _ = panel.fetch_series(panel.PMMS)
    orig_paths = sorted(glob.glob(panel.ORIG_GLOB))
    quarters = [p["quarter"] for p in man["partitions"]]
    if [re.search(r"(20\d\dQ[1-4])", os.path.basename(p)).group(1) for p in orig_paths] != quarters:
        raise SystemExit("origination files and panel partitions disagree on quarters")

    # ---- accumulators ------------------------------------------------------------
    cnt = {f: np.zeros(3, dtype=np.int64) for f in NAMES}            # loans per bucket
    raw_vals = {(f, b): Counter() for f in NAMES for b in (S, M)}
    n_orig = 0
    n_any_m = n_any_s = 0
    n_m_by_count = Counter()
    pair = np.zeros((len(NAMES), len(NAMES)), dtype=np.int64)
    per_file_m = {}
    examples = []
    known = None
    # part 3: key -> [loans, rows, null, sentinel, raw, other]
    p3 = {}
    p3_other_samples = {}
    n_panel_loans = n_orig_not_in_panel = 0
    static_changes = 0
    # part 1
    terr_codes = None
    t_rows_role = {}
    t_loans_role = {}
    t_loans_q, t_rows_q = {}, {}
    all_rows = all_loans = 0
    state_ids = {}
    cats_ref = None
    role_names = list(panel.ROLES)

    def bump(key, loans, rows, tally):
        a = p3.setdefault(key, np.zeros(6, dtype=np.int64))
        a += np.concatenate([[loans, rows], tally])

    for q, op in zip(quarters, orig_paths):
        # ----- origination file: classify every field, per loan ------------------------
        fr = pd.read_csv(op, sep="|", header=None, dtype=str, usecols=cols,
                         engine="c", na_filter=False)
        raw = {inv[c]: fr[c].str.strip() for c in cols}
        del fr
        n = len(raw["loan"])
        n_orig += n
        codes = np.empty((n, len(NAMES)), dtype=np.int8)
        parsed = {}
        for j, f in enumerate(NAMES):
            codes[:, j], x = classify(f, raw[f], q[2:])
            parsed[f] = x
            cnt[f] += np.bincount(codes[:, j], minlength=3)
            for b in (S, M):
                sel = codes[:, j] == b
                if sel.any():
                    raw_vals[f, b].update(raw[f][sel].value_counts().to_dict())
        isM = codes == M
        nm = isM.sum(axis=1)
        n_any_m += int((nm > 0).sum())
        n_any_s += int((codes == S).any(axis=1).sum())
        per_file_m[q] = (n, int((nm > 0).sum()))
        for k, c in zip(*np.unique(nm, return_counts=True)):
            n_m_by_count[int(k)] += int(c)
        pair += isM.astype(np.int64).T @ isM.astype(np.int64)
        loan_ids = raw["loan"].to_numpy(dtype=object)
        if (nm > 0).any():
            cand = sorted(np.nonzero(nm > 0)[0], key=lambda i: (-nm[i], loan_ids[i]))[:20]
            for i in cand:
                examples.append((-int(nm[i]), loan_ids[i], {NAMES[j]: raw[NAMES[j]].iloc[i]
                                                            for j in range(len(NAMES)) if isM[i, j]}))
        tails = pd.to_numeric(raw["loan"].str[5:], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        for i in np.nonzero(raw["loan"].to_numpy() == KNOWN_LOAN)[0]:
            known = {NAMES[j]: (raw[NAMES[j]].iloc[i], BUCKET[codes[i, j]]) for j in range(len(NAMES))}

        # ----- the panel partition: first row per loan --------------------------------------
        t = pq.read_table(os.path.join(panel.PANEL_DIR, "panel_%s.parquet" % q),
                          columns=["loan_id", "role", "fico", "ltv", "dti", "rate_spread",
                                   "property_state", "msa", "postal_code"])
        ptail = pc.cast(pc.utf8_slice_codeunits(t["loan_id"], 5, 12), pa.int64()) \
            .combine_chunks().to_numpy()
        nrow = len(ptail)
        first = np.ones(nrow, dtype=bool)
        first[1:] = ptail[1:] != ptail[:-1]
        fi = np.nonzero(first)[0]
        nrows = np.diff(np.r_[fi, nrow])
        ft = ptail[fi]
        role = t["role"].to_pandas()
        if list(role.cat.categories) != role_names:
            raise SystemExit("role categories differ from src.panel.ROLES")
        role = role.cat.codes.to_numpy().astype(np.int64)
        ps = t["property_state"].combine_chunks()
        cats = ps.dictionary.to_pylist()
        cats_ref = cats_ref or cats
        code = ps.indices.fill_null(-1).to_numpy().astype(np.int64)
        lid = np.cumsum(first) - 1                                  # loan index per row
        n_panel_loans += len(fi)
        all_rows += nrow
        all_loans += len(fi)

        # position of each panel loan in the origination file
        order = np.argsort(tails, kind="stable")
        pos = order[np.searchsorted(tails[order], ft)]
        if not np.array_equal(tails[pos], ft):
            raise SystemExit("%s: a panel loan is missing from its origination file" % q)
        n_orig_not_in_panel += n - len(fi)
        in_panel = np.zeros(n, dtype=bool)
        in_panel[pos] = True

        # the features are per-loan constants: confirm on this partition
        for c in ("fico", "ltv", "dti", "rate_spread"):
            v = t[c].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
            v0 = v[fi][lid]
            static_changes += int((~((v == v0) | (np.isnan(v) & np.isnan(v0)))).sum())
        static_changes += int((code != code[fi][lid]).sum())

        # ----- part 1: territories --------------------------------------------------------
        if terr_codes is None:
            terr_codes = {s: cats.index(s) for s in TERRITORIES}
        for s, k in terr_codes.items():
            m_rows = code == k
            m_loans = code[fi] == k
            t_rows_q[s, q] = int(m_rows.sum())
            t_loans_q[s, q] = int(m_loans.sum())
            for r in range(3):
                t_rows_role[s, r] = t_rows_role.get((s, r), 0) + int((m_rows & (role == r)).sum())
                uniq = np.unique(lid[m_rows & (role == r)])
                t_loans_role[s, r] = t_loans_role.get((s, r), 0) + len(uniq)
            ids = t["loan_id"].combine_chunks().take(pa.array(fi[m_loans])).to_pylist()
            state_ids.update({i: s for i in ids})
        t_rows_q["ALL", q], t_loans_q["ALL", q] = nrow, len(fi)

        # ----- part 3: what the panel holds ------------------------------------------------
        held = {c: t[c].combine_chunks().to_numpy(zero_copy_only=False)[fi].astype(np.float32)
                for c in ("fico", "ltv", "dti", "rate_spread")}
        for f, sent in (("fico", 9999.0), ("ltv", 999.0), ("dti", 999.0)):
            j = NAMES.index(f)
            b_all = codes[pos, j]
            for b in (V, S, M):
                sel = b_all == b
                if not sel.any():
                    continue
                tally, other = held_class(held[f][sel], parsed[f][pos][sel], sent)
                bump((f, b), int(sel.sum()), int(nrows[sel].sum()), tally)
                if other.any():
                    p3_other_samples.setdefault((f, b), Counter()).update(
                        zip(np.round(parsed[f][pos][sel][other], 4),
                            np.round(held[f][sel][other].astype(np.float64), 4)))
        # rate_spread depends on the rate AND the first payment month
        fpd = parsed["first_payment"]
        year, month = fpd // 100, fpd % 100
        valid_m = np.isfinite(fpd) & (month >= 1) & (month <= 12)
        omon = np.where(valid_m, year * 12 + (month - 1) - 1, -1).astype(np.int64)
        market = panel.lookup(pmms, omon)
        expect = (parsed["orig_rate"].astype(np.float32) - market).astype(np.float32)
        jr, jf = NAMES.index("orig_rate"), NAMES.index("first_payment")
        for label, jj in (("rate_spread<-orig_rate", jr), ("rate_spread<-first_payment", jf)):
            b_all = codes[pos, jj]
            for b in (V, S, M):
                sel = b_all == b
                if not sel.any():
                    continue
                h = held["rate_spread"][sel].astype(np.float64)
                e = expect[pos][sel].astype(np.float64)
                isnull = np.isnan(h)
                same = ~isnull & (h == e)
                other = ~isnull & ~same
                bump((label, b), int(sel.sum()), int(nrows[sel].sum()),
                     np.array([isnull.sum(), 0, same.sum(), other.sum()]))
                if (~isnull).any() and b != V:
                    p3_other_samples.setdefault((label, b), Counter()).update(
                        zip(np.round(parsed["orig_rate"][pos][sel][~isnull], 4),
                            np.round(h[~isnull], 4)))
        # property_state, msa, postal_code: strings
        jps = NAMES.index("property_state")
        hs = np.array([cats[c] if c >= 0 else None for c in code[fi]], dtype=object)
        rs = raw["property_state"].to_numpy(dtype=object)[pos]
        for b in (V, S, M):
            sel = codes[pos, jps] == b
            if not sel.any():
                continue
            isnull = np.array([x is None for x in hs[sel]])
            same = ~isnull & (hs[sel] == rs[sel])
            bump(("property_state", b), int(sel.sum()), int(nrows[sel].sum()),
                 np.array([isnull.sum(), 0, same.sum(), (~isnull & ~same).sum()]))
        for f in ("msa", "postal_code"):
            j = NAMES.index(f)
            hv = np.array(t[f].take(pa.array(fi)).to_pylist(), dtype=object)
            rv = raw[f].to_numpy(dtype=object)[pos]
            for b in (V, S, M):
                sel = codes[pos, j] == b
                if not sel.any():
                    continue
                isnull = np.array([x is None for x in hv[sel]])
                same = ~isnull & (hv[sel] == rv[sel])
                bump((f, b), int(sel.sum()), int(nrows[sel].sum()),
                     np.array([isnull.sum(), 0, same.sum(), (~isnull & ~same).sum()]))
        del t, raw, parsed, codes

    # ---- harness universe in the territories + fingerprint ----------------------------------
    L, stamp = hz.read_test_loans(progress=lambda s: None)
    reason = hz.eligibility(L).to_numpy()
    E = L[reason < 0]
    start = E["state_cut"].to_numpy()
    real = hz.realised(E)
    varying, fixed = hz.cohorts(real)
    fp = fingerprint(E.index.to_numpy(), [start] + [real[h] for h in hz.HORIZONS] + [fixed])
    elig_terr = Counter(state_ids[i] for i in E.index if i in state_ids)
    fixed_terr = Counter(state_ids[i] for i in E.index[fixed] if i in state_ids)

    src = source_lines({
        "fico": r"fico\[\(fico < FICO_RANGE",
        "ltv": r"ltv\[ltv >= LTV_NA\]",
        "dti": r"dti\[dti >= DTI_NA\]",
        "rate": r'rate = num\("orig_rate"\)',
        "num": r"def num\(label\)",
        "coerce": r'errors="coerce"\)\.to_numpy\(\)',
        "spread": r"spread = \(rate - market\)",
        "market": r"market = lookup\(pmms, orig_month\)",
        "text": r"def _text\(frame, col\)",
        "textblank": r's\[s == ""\] = None',
        "cat": r"state_code = pd\.Categorical\(state",
        "catlist": r"seen\.discard\(\"\"\)",
        "ficoconst": r"^FICO_RANGE",
        "ltvconst": r"^DTI_NA|^LTV_NA",
    })

    # ======================================================================================
    # REPORT
    # ======================================================================================
    rule("=")
    emit("ORIGINATION-RECORD VALIDITY AND TERRITORY COVERAGE   (diagnostic 23; read-only)")
    rule("=")
    emit("panel build  : %s  (%s rows, %d partitions; validated against its manifest)"
         % (man["build_stamp"], fmt(man["total_rows"]), man["n_partitions"]))
    emit("origination  : %d files, %s loans" % (len(orig_paths), fmt(n_orig)))
    emit("panel loans  : %s loans have at least one panel row; %s origination loans have none"
         % (fmt(n_panel_loans), fmt(n_orig_not_in_panel)))
    emit("unit         : loans (one origination record each); panel rows where stated")
    emit("sources      : docs/general_user_guide_july_2026.pdf, Origination Data File table")
    emit()

    # ---- PART 1 ---------------------------------------------------------------------------
    rule("=")
    emit("PART 1. TERRITORY COVERAGE   (GU, PR, VI; a state-level HPI covers states and DC only)")
    rule("=")
    emit("A loan sits in one partition and one state, so loans and rows are exact counts.")
    emit("By role: ROWS are rows with that role. LOANS are loans with at least one row in that")
    emit("role; a test-split loan has condition and evaluate rows, a train-split loan has fit")
    emit("rows only, so role loan counts do not sum to the loan total.")
    rule()
    groups = TERRITORIES + ["GU+PR+VI"]

    def tv(kind, g, r=None, q=None):
        parts = TERRITORIES if g == "GU+PR+VI" else [g]
        if kind == "rows_role":
            return sum(t_rows_role[s, r] for s in parts)
        if kind == "loans_role":      # a loan is in one state, so territory loans do not overlap
            return sum(t_loans_role[s, r] for s in parts)
        if kind == "rows_q":
            return sum(t_rows_q[s, q] for s in parts)
        return sum(t_loans_q[s, q] for s in parts)

    emit("%-10s%12s%12s%14s%14s%12s%12s%12s" % ("", "loans", "of loans", "panel rows",
                                               "of rows", "role fit", "condition", "evaluate"))
    rule()
    for g in groups:
        loans = sum(tv("loans_q", g, q=q) for q in quarters)
        rows = sum(tv("rows_q", g, q=q) for q in quarters)
        emit("%-10s%12s%12s%14s%14s%12s%12s%12s   rows" % (
            g, fmt(loans), pct(loans, all_loans, 4), fmt(rows), pct(rows, all_rows, 4),
            *[fmt(tv("rows_role", g, r)) for r in range(3)]))
        emit("%-10s%12s%12s%14s%14s%12s%12s%12s   loans in role" % (
            "", "", "", "", "", *[fmt(tv("loans_role", g, r)) for r in range(3)]))
    rule()
    for what, fn, tot, allv in (("loans", "loans_q", all_loans, t_loans_q),
                                ("panel rows", "rows_q", all_rows, t_rows_q)):
        emit("by origination quarter (%s):" % what)
        rule()
        emit("%-10s" % "" + "".join("%12s" % q for q in quarters) + "%14s" % "total")
        rule()
        for g in groups:
            emit("%-10s" % g + "".join("%12s" % fmt(tv(fn, g, q=q)) for q in quarters)
                 + "%14s" % fmt(sum(tv(fn, g, q=q) for q in quarters)))
        emit("%-10s" % "ALL" + "".join("%12s" % fmt(allv["ALL", q]) for q in quarters)
             + "%14s" % fmt(tot))
        rule()
    emit()
    emit("In the horizon harness (eligible test-split loans at the 2008-12 cut, n = %s):"
         % fmt(len(E)))
    rule()
    emit("%-10s%16s%12s%16s%12s" % ("", "eligible", "share", "fixed cohort", "share"))
    for g in groups:
        parts = TERRITORIES if g == "GU+PR+VI" else [g]
        a, b = sum(elig_terr[s] for s in parts), sum(fixed_terr[s] for s in parts)
        emit("%-10s%16s%12s%16s%12s" % (g, fmt(a), pct(a, len(E), 4), fmt(b),
                                        pct(b, int(fixed.sum()), 4)))
    rule()
    emit()

    # ---- PART 2 ---------------------------------------------------------------------------
    rule("=")
    emit("PART 2. ORIGINATION-RECORD VALIDITY   (unit: loans; %s origination records)" % fmt(n_orig))
    rule("=")
    emit("Each value is in exactly one bucket. Plausible ranges marked (script) are this")
    emit("script's own definition - the guide documents no range for them.")
    rule()
    emit("%-16s%-24s%-18s%14s%14s%14s%12s" % ("field", "plausible range", "sentinel (guide)",
                                              "VALID", "SENTINEL", "MALFORMED", "MALF. share"))
    rule()
    for f, _, sent in FIELDS:
        c = cnt[f]
        assert c.sum() == n_orig
        emit("%-16s%-24s%-18s%14s%14s%14s%12s" % (f, RANGE_TEXT[f], sent, fmt(c[V]), fmt(c[S]),
                                                  fmt(c[M]), pct(c[M], n_orig, 4)))
    rule()
    emit("Sentinels not documented in the guide: first_payment, orig_rate, loan, property_state.")
    emit("(Number of Units: the guide documents 99 = Not Available.)")
    emit()
    emit("Raw values outside VALID, per field (most frequent first, up to 8):")
    rule()
    for f in NAMES:
        for b in (S, M):
            top = raw_vals[f, b].most_common(8)
            if top:
                emit("%-16s%-10s %s" % (f, BUCKET[b],
                                        "   ".join("%r x %s" % (k if k != "" else "<blank>", fmt(v))
                                                    for k, v in top)
                                        + ("   (+%d more values)" % (len(raw_vals[f, b]) - 8)
                                           if len(raw_vals[f, b]) > 8 else "")))
    rule()
    emit()
    emit("a) LOANS WITH AT LEAST ONE MALFORMED FIELD")
    rule()
    emit("%s of %s origination loans (%s)." % (fmt(n_any_m), fmt(n_orig), pct(n_any_m, n_orig, 4)))
    emit("For reference, loans with at least one SENTINEL value: %s (%s)."
         % (fmt(n_any_s), pct(n_any_s, n_orig, 4)))
    emit("%-10s%14s%12s" % ("quarter", "loans", "malformed"))
    for q, (n, m) in per_file_m.items():
        emit("%-10s%14s%12s" % (q, fmt(n), fmt(m)))
    rule()
    emit()
    emit("b) OVERLAP - number of MALFORMED fields per loan, and pairs of fields")
    rule()
    for k in sorted(n_m_by_count):
        emit("%d malformed field(s): %s loans" % (k, fmt(n_m_by_count[k])))
    multi = sum(v for k, v in n_m_by_count.items() if k > 1)
    emit("malformed in MORE THAN ONE field: %s loans" % fmt(multi))
    emit()
    emit("loans malformed in both fields (diagonal = the field alone):")
    emit("%-16s" % "" + "".join("%9s" % n[:8] for n in NAMES))
    for i, a in enumerate(NAMES):
        emit("%-16s" % a + "".join("%9s" % (fmt(pair[i, j]) if pair[i, j] else ".")
                                   for j in range(len(NAMES))))
    rule()
    emit()
    emit("c) EXAMPLES - up to 20 MALFORMED loans (most malformed fields first, then loan id);")
    emit("raw values of the malformed fields only")
    rule()
    examples.sort(key=lambda e: (e[0], e[1]))
    if not examples:
        emit("no MALFORMED loan in any file")
    for _, lid_, vals in examples[:20]:
        emit("%-14s%s" % (lid_, "   ".join("%s=%r" % (k, v) for k, v in vals.items())))
    rule()
    emit()
    emit("d) %s (the loan found with a blank property state)" % KNOWN_LOAN)
    rule()
    if known is None:
        emit("not found in any origination file")
    else:
        emit("present. Every field, raw value and bucket:")
        for f in NAMES:
            emit("    %-16s%-10r %s" % (f, known[f][0], known[f][1]))
        bad = [f for f in NAMES if known[f][1] == "MALFORMED"]
        sent_ = [f for f in NAMES if known[f][1] == "SENTINEL"]
        emit("MALFORMED in: %s   |   SENTINEL in: %s" % (", ".join(bad) or "none",
                                                         ", ".join(sent_) or "none"))
        emit("%s of the %s loans with any MALFORMED field has %d or more; the other has %d."
             % (fmt(sum(v for k, v in n_m_by_count.items() if k >= len(bad))), fmt(n_any_m),
                len(bad), min(k for k in n_m_by_count if k > 0)))
    rule()
    emit()

    # ---- PART 3 ---------------------------------------------------------------------------
    rule("=")
    emit("PART 3. HOW THE PANEL HOLDS THESE VALUES   (unit: loans with a panel row; panel rows)")
    rule("=")
    emit("held = what data/panel stores, read from the parquet. Classes, in this order:")
    emit("    null        the stored value is null")
    emit("    sentinel    the stored number equals the raw sentinel (9999 / 999)")
    emit("    raw value   the stored number equals the raw value, parsed (float32): a value that")
    emit("                is NOT valid and was passed through unchanged")
    emit("    other       anything else (a zero, a parse artefact, a derived number)")
    emit("The VALID rows are the control: every one should be 'raw value'.")
    emit("Each loan's features are constants of the loan; %s rows differed from their loan's first"
         % fmt(static_changes))
    emit("row across fico, ltv, dti, rate_spread, property_state.")
    rule()
    emit("%-32s%-11s%12s%14s%12s%12s%12s%12s" % ("panel column <- source field", "bucket", "loans",
                                                "panel rows", "null", "sentinel", "raw value",
                                                "other"))
    rule()
    order3 = [("fico", "fico <- fico"), ("ltv", "ltv <- ltv"), ("dti", "dti <- dti"),
              ("rate_spread<-orig_rate", "rate_spread <- orig_rate"),
              ("rate_spread<-first_payment", "rate_spread <- first_payment"),
              ("property_state", "property_state <- state"),
              ("msa", "msa <- msa  (metadata)"),
              ("postal_code", "postal_code <- postal (meta)")]
    flagged = []
    for key, label in order3:
        for b in (V, S, M):
            a = p3.get((key, b))
            if a is None:
                emit("%-32s%-11s%12s   (no loan in this bucket)" % (label, BUCKET[b], "0"))
                continue
            emit("%-32s%-11s%12s%14s%12s%12s%12s%12s" % (label, BUCKET[b], fmt(a[0]), fmt(a[1]),
                                                        fmt(a[2]), fmt(a[3]), fmt(a[4]), fmt(a[5])))
            if b != V and key in ("fico", "ltv", "dti") and (a[3] or a[4] or a[5]):
                flagged.append((label, BUCKET[b], a))
            if b == M and key.startswith("rate_spread") and (a[4] or a[5]):
                flagged.append((label, BUCKET[b], a))
        rule()
    emit()
    emit("Values behind the non-null cells in the SENTINEL / MALFORMED buckets (raw -> held),")
    emit("most frequent first, up to 6:")
    rule()
    if not p3_other_samples:
        emit("none - no loan outside VALID holds a number in fico, ltv, dti or rate_spread")
    for k, c in sorted(p3_other_samples.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        emit("%-32s%-11s %s" % (k[0], BUCKET[k[1]], "   ".join(
            "%s->%s x %s" % (a, b, fmt(n)) for (a, b), n in c.most_common(6))))
    rule()
    emit()
    emit("Lines of src/panel.py that produce this behaviour (read from the file as it is now):")
    rule()
    for key, why in (
            ("ficoconst", "FICO_RANGE: what counts as a valid FICO"),
            ("ltvconst", "the LTV / DTI 'not available' constants"),
            ("num", "num(): every numeric field is parsed with errors=\"coerce\""),
            ("coerce", "  ... so a non-numeric string becomes NaN, silently"),
            ("fico", "fico: outside 300-850 -> NaN (this also removes the 9999 sentinel)"),
            ("ltv", "ltv: only values >= 999 -> NaN; nothing below 999 is range-checked"),
            ("dti", "dti: only values >= 999 -> NaN; nothing below 999 is range-checked"),
            ("rate", "rate: parsed with no range check and no sentinel handling"),
            ("market", "market: PMMS rate for the first-payment month minus one (NaN if no month)"),
            ("spread", "rate_spread = rate - market: inherits any bad rate; NaN if either is NaN"),
            ("text", "property_state / msa / postal_code are read as text ..."),
            ("textblank", "  ... a blank becomes None (null); anything else passes through as-is"),
            ("cat", "property_state: coded against the category list"),
            ("catlist", "  ... the list is every non-blank value in the files, so an invalid code "
                        "would be ADDED, not rejected")):
        ln, text = src[key]
        emit("src/panel.py:%-6s %-78s %s" % (ln, why, text[:60]))
    rule()
    emit()
    emit("FLAGS - a non-valid value reaching a feature column as a number")
    rule()
    if not flagged:
        emit("none")
    for label, bucket, a in flagged:
        emit("%s, %s loans: %s loans (%s panel rows) hold a number: %s as the raw sentinel, "
             "%s as the raw value, %s other." % (label, bucket, fmt(a[0]), fmt(a[1]),
                                                 fmt(a[3]), fmt(a[4]), fmt(a[5])))
    sent_reach = [x for x in flagged if x[1] == "SENTINEL"]
    emit()
    emit("LATENT - the build has no range check where a documented range exists:")
    emit("    ltv   only >= 999 is nulled; a value of 1-5 or 106-998 would be stored as a number.")
    emit("          MALFORMED ltv in these files: %s loans, so nothing is stored wrongly today."
         % fmt(cnt["ltv"][M]))
    emit("    dti   only >= 999 is nulled; a value of 66-998 would be stored as a number.")
    emit("          MALFORMED dti in these files: %s loans." % fmt(cnt["dti"][M]))
    emit("    rate  no range check and no sentinel handling: a rate of 0 or 99 would flow into")
    emit("          rate_spread as a number. MALFORMED orig_rate: %s loans, '.' (-> NaN, null) and"
         % fmt(cnt["orig_rate"][M]))
    emit("          '0'; the '0' loan is not in the panel, so no 0% rate reaches rate_spread.")
    emit("    All three are properties of THESE files, not of the code. A different vintage")
    emit("    (or the 2018Q2+ LTV range 1-998) would change what the same lines do.")
    emit()
    emit("A documented sentinel code reaches fico / ltv / dti / rate_spread as a number: %s"
         % ("YES" if any(x[2][3] for x in sent_reach) else "NO"))
    emit("A documented sentinel code reaches a metadata string column as raw text: %s"
         % ("YES: " + ", ".join("%s (%s loans)" % (k, fmt(p3[k, S][4]))
                                for k in ("msa", "postal_code") if (k, S) in p3 and p3[k, S][4])
            if any((k, S) in p3 and p3[k, S][4] for k in ("msa", "postal_code")) else "NO"))
    rule()
    emit()

    # ---- end ------------------------------------------------------------------------------------
    rule("=")
    emit("CHECKS")
    rule("=")
    emit("data fingerprint: %s   required %s   saved nulls run %s   %s"
         % (fp, WANT_FP, saved_fingerprint(), "PASS" if fp == WANT_FP else "FAIL"))
    after = os.stat(pmms_path)
    emit("data/macro/MORTGAGE30US.csv unchanged (size, mtime): %s"
         % ("PASS" if (after.st_size, after.st_mtime_ns) == (pmms_stat.st_size, pmms_stat.st_mtime_ns)
            else "FAIL"))
    emit("every origination loan classified once per field: PASS (asserted per field)")
    emit("runtime: %.1f s  (%.2f min)" % (time.time() - t0, (time.time() - t0) / 60.0))
    rule("=")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return 0 if fp == WANT_FP else 1


if __name__ == "__main__":
    sys.exit(main())
