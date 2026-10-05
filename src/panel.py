"""Transition-panel builder for the Freddie Mac SFLLD 2006-2007 vintages.

Run as a module:      python -m src.panel
Import and drive:     from src.panel import build; build()

WHAT A ROW IS
    One row per OBSERVED consecutive-month transition of one loan: the origin
    month t and the month t+1 that immediately follows it IN THE DATA. A loan
    observed for N months yields at most N-1 rows - fewer where a reporting hole
    breaks the chain (R3) or where the origin month is already absorbing (R1).

================================ SCHEMA ================================
Three groups, kept distinct in the code and in the written file's column order.

keys
    loan_id         str      Freddie loan sequence number
    origin_period   int32    YYYYMM of month t

states                       categorical over the six-state space
    origin_state             Current, DQ30, DQ60, DQ90+, Prepaid, Default
    next_state               same categories; Prepaid and Default are absorbing

features                     everything the model is allowed to read
    run_length      int32    consecutive months in origin_state as of month t,
                             counting t itself as 1 (R5)
    loan_age        int32    months from the origination month (first payment
                             month minus one) to month t. NOT Freddie's reported
                             loan age, which restarts at modification (R7)
    is_modified     bool     month t is at or after the loan's first
                             modification (R7)
    months_since_modification
                    Int32    months from the loan's MOST RECENT modification
                             to month t; 0 on the modification month, null
                             before the first one (R7)
    fico          float32  Classic FICO at origination
    ltv             float32  original LTV
    dti             float32  original DTI
    rate_spread     float32  original note rate minus the PMMS 30-year fixed
                             market rate prevailing in the origination month
    unemp_level     float32  national unemployment rate at month t
    unemp_chg_12m   float32  that rate minus its value twelve months earlier
    hpi_feature     float32  PLACEHOLDER, emitted as all-null. The encoding is
                             not yet decided; the column exists so the join can
                             be added later without rebuilding the panel.
    property_state  category Freddie "Property State", the two-letter code, from
                             the origination file. A parquet dictionary of
                             strings (dictionary<values=string, indices=int8>),
                             NOT a numeric code: the category list is the union
                             over every origination file, sorted, so every
                             partition carries the same dictionary. Null where
                             the field is blank - never imputed. Added so a
                             state-level house-price index can be joined later;
                             no such join exists yet and hpi_feature is untouched.

metadata                     bookkeeping - NEVER a model input
    role            category fit / condition / evaluate (R6)
    post_gap        bool     month t is the first month after a reporting hole
    is_skip         bool     raw status rose by >= 2 rungs across the pair
    is_fall         bool     raw status fell by >= 2 rungs across the pair
    msa             string   Freddie "Metropolitan Statistical Area (MSA) Or
                             Metropolitan Division" code, as text; null where
                             blank. Parked for finer geography; never imputed.
    postal_code     string   Freddie "Postal Code", the 3-digit prefix, as text
                             so leading zeros survive; null where blank. Parked
                             like msa.

    METADATA IS NEVER A FEATURE. is_skip and is_fall are derived from the raw
    status at t+1, which is the label side of the pair. Feeding either into a
    model leaks the outcome: knowing "the status jumped by two or more" already
    tells you next_state is not adjacent to origin_state. They exist so that
    rows can be audited, counted and sliced after the fact, and so that a later
    decision to exclude them can be made without a rebuild. The same applies to
    post_gap and role; msa and postal_code are parked here, not offered as
    inputs. Any model code must select the `features` group by name
    and must never select by "all columns except the states".

================================ RULES =================================
R1  TERMINAL ROW. The row carrying a zero-balance code IS the exit month. The
    transition INTO that month is emitted with next_state = Prepaid (code 01) or
    Default (codes 02, 03, 09). No row is ever emitted with an absorbing state as
    origin_state. The exit is dated by the REPORTING PERIOD; the zero-balance
    effective date is not read, and no loan is excluded for disagreeing with it.

R2  CENSORING. Codes 15, 16 and 96, and loans that simply stop appearing, are
    censored. Rows are emitted up to the last OBSERVED transition and then stop;
    no final self-transition is ever fabricated for the last observed month.
    Censoring is not a state and never appears in next_state - a censoring code
    marks where follow-up ENDS, not what the loan is doing, so the pair landing
    on that month carries the state its own delinquency status maps to. Every
    censored loan is recorded separately with its last observed period and a
    reason, in data/panel/censoring/.

R3  GAPS. Where reporting periods are not strictly consecutive the pair spanning
    the hole is DROPPED - never paired across - and the chain resumes on the far
    side. run_length resets to 1 on the first row after the hole and that row
    carries post_gap = True. No size threshold: every hole is treated this way.

R4  SKIPS AND FALLS. Both are kept as ordinary observed transitions and flagged.

R5  RUN LENGTH. Consecutive months the loan has held its current MAPPED state as
    of the origin month, counting the origin month as 1, computed over the loan's
    FULL observed history before any role filtering, and reset by a state change
    or a gap.

R6  ROLE. Grouped-temporal split, cut at 2008-12.
        fit        train-split loan, origin month <= 2008-12
        condition  test-split loan,  origin month <= 2008-12
        evaluate   test-split loan,  origin month >  2008-12
    The train/test split is over LOANS, by a deterministic hash of loan_id,
    80/20. Train-split rows with an origin month after the cut are excluded
    entirely. `condition` rows exist only to carry state and run_length forward
    into the evaluation window; they are never fitted on.

R7  LOAN AGE AND MODIFICATION. Freddie's loan age restarts when a loan is
    modified (diagnostic 20), so loan_age is recomputed from the origination
    month and is strictly increasing within a loan; the build asserts it.
    A MODIFICATION EVENT is any of:
        - the loan's first row flagged Y or P
        - every later row that starts a run of Y ('Y' marks the modification
          month and is followed by P rows)
        - for a loan with NO Y or P on any row, each row where Freddie's loan
          age decreases. Such loans are counted and listed in the build report.
    A reset with a P or blank flag on an otherwise-flagged loan is not an event.
    is_modified is True from the first event onward. months_since_modification
    counts calendar months from the most recent event, so it goes back to 0 at
    every repeat modification - deliberately: it measures time since the terms
    last changed, not since they first did.

============================== ORDERING ================================
Load-bearing, and implemented in this order inside build_quarter():
    1 read and sort by (loan_id, period)      5 join macro on origin period
    2 detect gaps, mark them                  6 join origination on loan_id,
    3 run_length and modification events        and compute loan_age from it
      over the full history                   7 emit pairs, applying R1 and R2
    4 assign role
Step 3 precedes step 4 so that run_length and the modification columns are
properties of the loan's observed history and not of the window a row happens
to land in.

============================ COLUMN INDICES ============================
Every index is resolved at run time from docs/file_layout_july_2026.xlsx by
attribute name - none is hardcoded from memory. The three indices the rest of
the project already relies on (0 loan id, 1 period, 3 delinquency status) are
additionally asserted, and any mismatch stops the run before a file is read.

============================== MEMORY ==================================
Performance files are read in chunks and only the six needed columns are
retained, as narrow integer arrays. A quarter's retained columns peak at roughly
400 MB for the largest file, against a 2.3 GB source file; no file is ever held
whole. The per-loan sort requires the quarter's six columns to be resident at
once, which is what makes the transition walk possible at all.

============================ DETERMINISM ===============================
Same input, same output, same order: the loan split is a keyed BLAKE2b digest
rather than Python's randomised hash(); the sort is stable and total on
(loan_id, period); rows are written in (loan_id, origin_period) order; and the
column order is fixed by SCHEMA below.
"""

import argparse
import datetime
import glob
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
PERF_GLOB = "data/historical_data_200[67]/*/perf_*.txt"
ORIG_GLOB = "data/historical_data_200[67]/*/orig_*.txt"
LAYOUT = os.path.join("docs", "file_layout_july_2026.xlsx")
PANEL_DIR = os.path.join("data", "panel")
CENSOR_DIR = os.path.join(PANEL_DIR, "censoring")
MACRO_DIR = os.path.join("data", "macro")
REPORT = os.path.join("diagnostics", "16_panel_build_report.txt")
PANEL_FILE = "panel_%s.parquet"                 # % quarter
MANIFEST_NAME = "panel_manifest.json"
STAMP_KEY = b"panel_build_stamp"                # parquet footer metadata key

CHUNKSIZE = 2_000_000
RULE_W = 121

CUT_YEAR, CUT_MONTH = 2008, 12          # R6: last origin month in the fit window
TRAIN_PCT = 80                          # R6: 80/20 loan split
SPLIT_SEED = "credit-migration/2006-2007/v1"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s"
FRED_TIMEOUT = 60
FRED_ATTEMPTS = 3
UNRATE = "UNRATE"
PMMS = "MORTGAGE30US"
LOOKBACK = 12                           # months, for unemp_chg_12m

# ---- the six-state space -------------------------------------------------
S_CUR, S_DQ30, S_DQ60, S_DQ90, S_PREPAID, S_DEFAULT = range(6)
STATES = ["Current", "DQ30", "DQ60", "DQ90+", "Prepaid", "Default"]
ABSORBING = (S_PREPAID, S_DEFAULT)

ZB_PREPAID = ("01",)
ZB_DEFAULT = ("02", "03", "09")
ZB_CENSOR = ("15", "16", "96")

ROLES = ["fit", "condition", "evaluate"]
R_FIT, R_CONDITION, R_EVALUATE, R_EXCLUDED = 0, 1, 2, -1

CENSOR_REASONS = ["code_15", "code_16", "code_96", "no_further_observation"]

# ---- schema, in written column order ------------------------------------
KEYS = ["loan_id", "origin_period"]
STATE_COLS = ["origin_state", "next_state"]
FEATURES = ["run_length", "loan_age", "is_modified", "months_since_modification",
            "fico", "ltv", "dti", "rate_spread",
            "unemp_level", "unemp_chg_12m", "hpi_feature", "property_state"]
METADATA = ["role", "post_gap", "is_skip", "is_fall", "msa", "postal_code"]
SCHEMA = KEYS + STATE_COLS + FEATURES + METADATA

# Non-numeric features, and the exact Arrow type each must carry on disk.
# panel_io.validate enforces this independently of the manifest: a partition
# or manifest that records anything else fails the load, with no override.
CATEGORICAL_FEATURES = {
    "property_state": "dictionary<values=string, indices=int8, ordered=0>"}
MAX_CATEGORIES = 127                    # int8 dictionary indices

# ---- attributes resolved from the published layout ----------------------
PERF_WANTED = [
    ("loan", "loan identifier"),
    ("period", "period"),
    ("status", "current loan delinquency status"),
    ("loan_age", "loan age"),
    ("zero_balance", "zero balance code"),
    ("mod", "modification flag"),
]
MOD_NONE, MOD_Y, MOD_P = 0, 1, 2        # read_perf's encoding of the flag
PERF_EXPECTED = {"loan": 0, "period": 1, "status": 3}
ORIG_WANTED = [
    ("fico", "classic fico"),
    ("first_payment", "first payment date"),
    ("dti", "original debt-to-income (dti) ratio"),
    ("ltv", "original loan-to-value (ltv)"),
    ("orig_rate", "original interest rate"),
    ("loan", "loan identifier"),
    ("property_state", "property state"),
    ("msa", "metropolitan statistical area (msa) or metropolitan division"),
    ("postal_code", "postal code"),
]
# 0-based indices the project relies on (Freddie fields 17, 5 and 19), asserted
# after resolution - the same safeguard the performance file has.
ORIG_EXPECTED = {"property_state": 16, "msa": 4, "postal_code": 18}
PERF_SHEET = "monthly performance data file"
ORIG_SHEET = "origination data file"

# origination sentinels that mean "not available"
FICO_RANGE = (300, 850)
DTI_NA = 999
LTV_NA = 999

_LINES = []


def emit(line=""):
    print(line, flush=True)
    _LINES.append(line)


def rule(char="-", width=RULE_W):
    emit(char * width)


def fmt(n):
    return "{:,}".format(int(n))


def pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def mi(y, m):
    """(year, month) -> contiguous month index."""
    return y * 12 + (m - 1)


def to_yyyymm(idx):
    """Contiguous month index -> YYYYMM."""
    return (idx // 12) * 100 + (idx % 12) + 1


CUT = mi(CUT_YEAR, CUT_MONTH)


# ==========================================================================
# layout resolution
# ==========================================================================
def _sheet_rows(path):
    """{sheet title marker: [(field position, attribute name)]} from the xlsx."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

    def cell_text(c):
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(ns + "t"))
        v = c.find(ns + "v")
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    out = {}
    for name in sorted(n for n in z.namelist()
                       if re.match(r"xl/worksheets/sheet\d+\.xml$", n)):
        root = ET.fromstring(z.read(name))
        rows = [[cell_text(c) for c in row.iter(ns + "c")]
                for row in root.iter(ns + "row")]
        head = " ".join(" ".join(r) for r in rows[:6]).lower()
        fields = []
        for r in rows:
            cells = [x.strip() for x in r if x.strip()]
            if len(cells) >= 2 and re.match(r"^\d+$", cells[0]):
                fields.append((int(cells[0]), cells[1]))
        for marker in (PERF_SHEET, ORIG_SHEET):
            if marker in head and fields:
                out[marker] = (fields, name)
    return out


def _normalise(text):
    """Fold the layout's typographic noise so a plain-ASCII name still matches."""
    t = text.strip().lower()
    t = t.replace("®", "").replace("™", "")
    t = re.sub(r"[^a-z0-9()/%.-]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def resolve(path=LAYOUT):
    """Resolve both files' column indices by attribute name. Never guesses."""
    sheets = _sheet_rows(path)
    for marker in (PERF_SHEET, ORIG_SHEET):
        if marker not in sheets:
            raise SystemExit("layout %s has no %r sheet" % (path, marker))

    def pick(marker, wanted):
        fields, sheet = sheets[marker]
        resolved = {}
        for label, want in wanted:
            w = _normalise(want)
            hits = [f for f in fields if _normalise(f[1]) == w]
            if not hits:
                hits = [f for f in fields if w in _normalise(f[1])]
            if len(hits) != 1:
                raise SystemExit(
                    "layout lookup for %r on %s matched %d row(s) - refusing to "
                    "guess" % (want, marker, len(hits)))
            pos, attribute = hits[0]
            if not 1 <= pos <= 40:
                raise SystemExit("implausible field position %r for %r"
                                 % (pos, want))
            resolved[label] = (pos - 1, pos, attribute)
        return resolved, sheet

    perf, perf_sheet = pick(PERF_SHEET, PERF_WANTED)
    orig, orig_sheet = pick(ORIG_SHEET, ORIG_WANTED)
    for which, found, expected in (("performance", perf, PERF_EXPECTED),
                                   ("origination", orig, ORIG_EXPECTED)):
        for label, expect in expected.items():
            got = found[label][0]
            if got != expect:
                raise SystemExit(
                    "layout gives index %d for %s %r but the project uses "
                    "%d - stopping before any file is read"
                    % (got, which, label, expect))
    return perf, orig, perf_sheet, orig_sheet


# ==========================================================================
# macro series
# ==========================================================================
def fetch_series(sid):
    """Monthly {month index: value} for a FRED series, cache first.

    Weekly series are averaged within the calendar month. Returns
    (values, provenance); provenance carries how the series was obtained so the
    build report can state it rather than implying a download happened.
    """
    path = os.path.join(MACRO_DIR, sid + ".csv")
    prov = {"id": sid, "path": path, "url": FRED_CSV % sid}
    raw = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        prov["source"] = "cache"
        prov["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(os.path.getmtime(path)))
        with open(path, "rb") as fh:
            raw = fh.read()
    else:
        # No custom User-Agent: measured on this host, fredgraph.csv serves the
        # default python-urllib UA promptly but hangs on a custom one.
        errors = []
        for attempt in range(1, FRED_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(urllib.request.Request(prov["url"]),
                                            timeout=FRED_TIMEOUT) as resp:
                    raw = resp.read()
                prov["source"] = "download"
                prov["attempts"] = attempt
                break
            except Exception as exc:
                errors.append("attempt %d: %s: %s"
                              % (attempt, type(exc).__name__, exc))
        if raw is None:
            prov["source"] = "UNAVAILABLE"
            prov["error"] = "; ".join(errors)
            return None, prov
        os.makedirs(MACRO_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
        prov["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")

    frame = pd.read_csv(io.BytesIO(raw))
    date_col, value_col = frame.columns[0], frame.columns[1]
    when = pd.to_datetime(frame[date_col], errors="coerce")
    value = pd.to_numeric(frame[value_col], errors="coerce")
    ok = when.notna() & value.notna()
    months = (when[ok].dt.year * 12 + when[ok].dt.month - 1).to_numpy()
    grouped = pd.Series(value[ok].to_numpy()).groupby(months).mean()
    prov["n_months"] = int(len(grouped))
    if len(grouped):
        prov["range"] = "%d -> %d" % (to_yyyymm(int(grouped.index.min())),
                                      to_yyyymm(int(grouped.index.max())))
    return {int(k): float(v) for k, v in grouped.items()}, prov


def macro_tables():
    """(unemployment level, 12m change, PMMS level) as month-indexed dicts."""
    unrate, u_prov = fetch_series(UNRATE)
    pmms, p_prov = fetch_series(PMMS)
    chg = {}
    if unrate:
        for m, v in unrate.items():
            back = unrate.get(m - LOOKBACK)
            if back is not None:
                chg[m] = v - back
    return unrate, chg, pmms, u_prov, p_prov


def lookup(table, months):
    """Vectorised month-index lookup into a dict, missing -> NaN."""
    out = np.full(len(months), np.nan, dtype=np.float32)
    if not table:
        return out
    keys = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
    vals = np.fromiter(table.values(), dtype=np.float64, count=len(table))
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    pos = np.searchsorted(keys, months)
    safe = np.clip(pos, 0, len(keys) - 1)
    hit = keys[safe] == months
    out[hit] = vals[safe][hit].astype(np.float32)
    return out


# ==========================================================================
# loan split
# ==========================================================================
def split_flags(loan_ids):
    """Deterministic 80/20 train mask over an array of loan id strings.

    Keyed BLAKE2b, not Python's hash(), which is salted per process and would
    reshuffle the split on every run.
    """
    key = SPLIT_SEED.encode("ascii")[:16]
    out = np.empty(len(loan_ids), dtype=bool)
    for i, lid in enumerate(loan_ids):
        digest = hashlib.blake2b(lid.encode("ascii"), digest_size=8,
                                 key=key).digest()
        out[i] = (int.from_bytes(digest, "big") % 100) < TRAIN_PCT
    return out


# ==========================================================================
# origination features
# ==========================================================================
def state_categories(orig_paths, oidx):
    """Sorted union of the Property State values in every origination file.

    Read once, before the first write, so every partition shares one dictionary.
    Blank is a missing value, not a category.
    """
    seen = set()
    for path in orig_paths:
        col = pd.read_csv(path, sep="|", header=None, dtype=str,
                          usecols=[oidx["property_state"][0]], engine="c",
                          na_filter=False).iloc[:, 0].str.strip()
        seen.update(col.unique())
    seen.discard("")
    cats = sorted(seen)
    if len(cats) > MAX_CATEGORIES:
        raise SystemExit("%d distinct Property State values do not fit int8 "
                         "dictionary indices" % len(cats))
    return cats


def _text(frame, col):
    """Stripped text column as an object array, None where blank."""
    s = frame[col].str.strip().to_numpy(dtype=object)
    s[s == ""] = None
    return s


def load_origination(path, oidx, pmms, categories=None):
    """Sorted (loan tail, fico, ltv, dti, rate_spread, geography) for a quarter.

    The loan sequence number is a per-file constant 5-character prefix plus a
    7-digit tail, so the tail alone keys the quarter and sorts numerically.
    rate_spread is the note rate minus the PMMS 30-year fixed rate prevailing in
    the origination month, which is taken as the first payment month minus one -
    the origination date itself is not carried in the file.

    property_state is returned as int8 codes into `categories` (-1 where blank),
    which defaults to this file's own sorted values; a non-blank value outside
    `categories` is an error. msa and postal_code are text, None where blank.
    """
    cols = sorted(oidx[k][0] for k in oidx)
    frame = pd.read_csv(path, sep="|", header=None, dtype=str, usecols=cols,
                        engine="c", na_filter=False)
    loan = frame[oidx["loan"][0]].str.strip()
    tail = pd.to_numeric(loan.str[5:], errors="coerce").fillna(-1).astype(np.int64)

    def num(label):
        return pd.to_numeric(frame[oidx[label][0]].str.strip(),
                             errors="coerce").to_numpy()

    fico = num("fico").astype(np.float32)
    fico[(fico < FICO_RANGE[0]) | (fico > FICO_RANGE[1])] = np.nan
    ltv = num("ltv").astype(np.float32)
    ltv[ltv >= LTV_NA] = np.nan
    dti = num("dti").astype(np.float32)
    dti[dti >= DTI_NA] = np.nan
    rate = num("orig_rate").astype(np.float32)

    fpd = num("first_payment")
    year, month = fpd // 100, fpd % 100
    valid = np.isfinite(fpd) & (month >= 1) & (month <= 12)
    orig_month = np.where(valid, year * 12 + (month - 1) - 1, -1).astype(np.int64)
    market = lookup(pmms, orig_month)
    spread = (rate - market).astype(np.float32)

    state = _text(frame, oidx["property_state"][0])
    present = np.array([v is not None for v in state], dtype=bool)
    if categories is None:
        categories = sorted(set(state[present]))
    state_code = pd.Categorical(state, categories=categories).codes
    unknown = present & (state_code < 0)
    if unknown.any():
        raise RuntimeError("%s: Property State value(s) outside the category "
                           "list: %s" % (path, sorted(set(state[unknown]))))
    msa = _text(frame, oidx["msa"][0])
    postal = _text(frame, oidx["postal_code"][0])

    order = np.argsort(tail.to_numpy(), kind="stable")
    return {"tail": tail.to_numpy()[order], "fico": fico[order],
            "ltv": ltv[order], "dti": dti[order], "rate_spread": spread[order],
            "orig_month": orig_month[order],
            "property_state": state_code[order].astype(np.int8),
            "state_categories": list(categories),
            "msa": msa[order], "postal_code": postal[order],
            "n": len(order), "n_spread": int(np.isfinite(spread).sum())}


def join_origination(orig, tails):
    """Look up per-loan origination features for an array of loan tails."""
    pos = np.searchsorted(orig["tail"], tails)
    safe = np.clip(pos, 0, max(orig["n"] - 1, 0))
    hit = (orig["n"] > 0) & (orig["tail"][safe] == tails)
    out = {}
    for col in ("fico", "ltv", "dti", "rate_spread"):
        vals = np.full(len(tails), np.nan, dtype=np.float32)
        vals[hit] = orig[col][safe][hit]
        out[col] = vals
    out["orig_month"] = np.where(hit, orig["orig_month"][safe], -1)  # -1: none
    code = np.full(len(tails), -1, dtype=np.int8)              # -1: none
    code[hit] = orig["property_state"][safe][hit]
    out["property_state"] = code
    for col in ("msa", "postal_code"):
        vals = np.full(len(tails), None, dtype=object)
        vals[hit] = orig[col][safe][hit]
        out[col] = vals
    return out, int(hit.sum())


# ==========================================================================
# loan-to-file containment
# ==========================================================================
# run_length (R5) is walked per perf file, so a loan whose rows sat in two files
# would have its run silently reset at the file boundary. Two checks together
# rule that out, and neither duplicates the other:
#   perf_prefix / check_prefixes   NECESSARY, cheap, before any write: first-row
#                                  prefixes are pairwise distinct across files.
#   read_perf + build_quarter      SUFFICIENT, exhaustive, but only during that
#                                  quarter's build: every row carries one prefix.
#                                  If it fires, earlier quarters' panel and
#                                  censoring parquets (and any freshly
#                                  downloaded macro CSV) are already on disk.
# Together: no loan id can occur in two files.
def perf_prefix(path, pidx):
    """Loan-id prefix of a perf file, read from its first row only."""
    first = pd.read_csv(path, sep="|", header=None, dtype=str, nrows=1,
                        usecols=[pidx["loan"][0]], engine="c", na_filter=False)
    if first.empty:
        raise SystemExit("%s has no rows to read a loan-id prefix from" % path)
    return first.iloc[0, 0].strip()[:5]


def check_prefixes(prefix_by_file, allow_multi_file_loans=False):
    """Assert loan-id prefixes are pairwise distinct across perf files.

    prefix_by_file maps file name -> prefix. Returns {prefix: [file names]}
    for every prefix held by more than one file (empty on a pass). Raises on a
    collision unless allow_multi_file_loans, in which case it warns instead.
    """
    files_by_prefix = {}
    for name in sorted(prefix_by_file):
        files_by_prefix.setdefault(prefix_by_file[name], []).append(name)
    collisions = {p: f for p, f in files_by_prefix.items() if len(f) > 1}
    if collisions:
        msg = ("loan-id prefix shared across perf files - a loan could span "
               "files and run_length would reset at the boundary: "
               + "; ".join("%r in %s" % (p, ", ".join(collisions[p]))
                           for p in sorted(collisions)))
        if not allow_multi_file_loans:
            raise SystemExit(msg)
        warnings.warn(msg + " (proceeding: --allow-multi-file-loans)",
                      RuntimeWarning, stacklevel=2)
    return collisions


# ==========================================================================
# per-quarter build
# ==========================================================================
class Report(object):
    """Accumulates every count the build report prints."""

    def __init__(self):
        self.perf_rows = 0
        self.dropped_rows = 0
        self.loans = 0
        self.adjacent_pairs = 0
        self.gap_pairs = 0
        self.dup_pairs = 0
        self.suppressed = 0
        self.candidates = 0
        self.excluded_train_late = 0
        self.panel_rows = 0
        self.by_role = np.zeros(len(ROLES), dtype=np.int64)
        self.by_origin = np.zeros(len(STATES), dtype=np.int64)
        self.by_next = np.zeros(len(STATES), dtype=np.int64)
        self.fit_matrix = np.zeros((len(STATES), len(STATES)), dtype=np.int64)
        self.skips = 0
        self.falls = 0
        self.post_gap = 0
        self.censored = np.zeros(len(CENSOR_REASONS), dtype=np.int64)
        self.terminated = 0
        self.nulls = {c: 0 for c in SCHEMA}
        self.state_categories = None    # Property State dictionary, whole build
        self.zero_row_loans = {}
        self.orig_matched = 0
        self.train_loans = 0
        self.test_loans = 0
        self.quarters = []
        self.prefix_by_file = {}
        self.prefix_collisions = {}
        self.mod_loans = 0              # loans with a Y or P on any row
        self.unflagged_mod_loans = []   # loan ids: age resets, no flag anywhere
        self.rows_modified = 0          # panel rows with is_modified
        self.age_corrected = 0          # panel rows where loan_age != Freddie's
        self.age_corrections = {}       # new - Freddie's -> panel rows
        self.age_negative = 0           # panel rows with loan_age < 0
        self.build_stamp = None
        self.schemas = {}           # quarter -> [(column, arrow type)], as written

    def bump_zero(self, reason, n):
        self.zero_row_loans[reason] = self.zero_row_loans.get(reason, 0) + int(n)


def read_perf(path, pidx):
    """One quarter's six needed columns, chunked, sorted by (loan, period)."""
    c_loan, c_per = pidx["loan"][0], pidx["period"][0]
    c_st, c_age, c_zb = (pidx["status"][0], pidx["loan_age"][0],
                         pidx["zero_balance"][0])
    c_mod = pidx["mod"][0]
    cols = sorted({c_loan, c_per, c_st, c_age, c_zb, c_mod})
    tails, periods, status, ages, zbs, mods = [], [], [], [], [], []
    prefixes = {}
    for chunk in pd.read_csv(path, sep="|", header=None, dtype=str, usecols=cols,
                             chunksize=CHUNKSIZE, engine="c", na_filter=False):
        loan = chunk[c_loan].str.strip()
        for p, c in loan.str[:5].value_counts().items():
            prefixes[p] = prefixes.get(p, 0) + int(c)
        tails.append(pd.to_numeric(loan.str[5:], errors="coerce")
                     .fillna(-1).astype(np.int64).to_numpy())

        per = pd.to_numeric(chunk[c_per].str.strip(), errors="coerce")
        year, month = per // 100, per % 100
        idx = (year * 12 + (month - 1)).where((month >= 1) & (month <= 12))
        periods.append(idx.fillna(-1).astype(np.int64).to_numpy())

        raw = chunk[c_st].str.strip()
        digits = raw.str.isdigit()
        val = pd.to_numeric(raw.where(digits), errors="coerce").fillna(-1)
        code = val.astype(np.int16).to_numpy()
        code[~digits.to_numpy()] = -1                 # RA and anything non-numeric
        status.append(code)

        ages.append(pd.to_numeric(chunk[c_age].str.strip(), errors="coerce")
                    .fillna(-9999).astype(np.int32).to_numpy())

        arr = chunk[c_zb].str.strip().to_numpy()
        zcode = np.zeros(len(arr), dtype=np.int8)     # 0 = no code on this row
        for c in ZB_PREPAID:
            zcode[arr == c] = 1
        for c in ZB_DEFAULT:
            zcode[arr == c] = 2
        for i, c in enumerate(ZB_CENSOR):
            zcode[arr == c] = 3 + i                   # 3, 4, 5 -> 15, 16, 96
        zbs.append(zcode)

        flag = chunk[c_mod].str.strip().to_numpy()
        mcode = np.full(len(flag), MOD_NONE, dtype=np.int8)
        mcode[flag == "Y"] = MOD_Y
        mcode[flag == "P"] = MOD_P
        mods.append(mcode)

    tail = np.concatenate(tails)
    period = np.concatenate(periods)
    st = np.concatenate(status)
    age = np.concatenate(ages)
    zb = np.concatenate(zbs)
    mod = np.concatenate(mods)
    del tails, periods, status, ages, zbs, mods

    keep = (period >= 0) & (tail >= 0)
    dropped = int((~keep).sum())
    if dropped:
        tail, period, st, age, zb, mod = (a[keep] for a in
                                          (tail, period, st, age, zb, mod))

    order = np.lexsort((period, tail))
    return (tail[order], period[order].astype(np.int32), st[order],
            age[order], zb[order], mod[order], prefixes, dropped)


def modification_columns(tail, period, age, mod):
    """R7 over a loan-sorted history: (is_modified, months_since, unflagged).

    tail, period, age, mod are sorted by (loan, period); age is Freddie's
    reported loan age with -9999 for null, mod is read_perf's flag encoding.
    months_since is -1 where the loan is not yet modified. unflagged is a row
    mask of the events on loans that carry no Y or P anywhere - the loans whose
    only evidence of modification is Freddie's loan age going backwards.
    """
    n = len(tail)
    ar = np.arange(n, dtype=np.int64)
    is_first = np.empty(n, dtype=bool)
    is_first[:1] = True
    np.not_equal(tail[1:], tail[:-1], out=is_first[1:])
    loan_start = np.maximum.accumulate(np.where(is_first, ar, 0))
    loan_idx = np.cumsum(is_first) - 1

    flagged = (mod == MOD_Y) | (mod == MOD_P)
    flags_so_far = np.cumsum(flagged)
    flags_before_loan = flags_so_far[loan_start] - flagged[loan_start]
    first_flag = flagged & (flags_so_far - flags_before_loan == 1)
    y_start = mod == MOD_Y
    y_start[1:] &= ~(~is_first[1:] & (mod[:-1] == MOD_Y))
    loan_flagged = np.bincount(loan_idx, weights=flagged,
                               minlength=int(loan_idx[-1]) + 1 if n else 0) > 0

    reset = np.zeros(n, dtype=bool)
    reset[1:] = (~is_first[1:] & (age[1:] != -9999) & (age[:-1] != -9999)
                 & (age[1:] < age[:-1]))
    unflagged = reset & ~loan_flagged[loan_idx]

    event = first_flag | y_start | unflagged
    last_event = np.maximum.accumulate(np.where(event, ar, -1))
    is_modified = last_event >= loan_start
    # Measured from the MOST RECENT event: a repeat modification sends it back
    # to 0. Deliberate - see R7.
    months_since = np.where(is_modified,
                            period - period[np.maximum(last_event, 0)], -1)
    return is_modified, months_since.astype(np.int32), unflagged


def row_states(st, zb):
    """Map each observed row to one of the six states (R1 and the state mapping).

    A terminal zero-balance code governs the row: 01 is Prepaid, 02/03/09 are
    Default. A censoring code (15/16/96) does NOT govern - it marks where
    follow-up stops, not what the loan is doing - so such a row keeps the state
    its own delinquency status maps to. Non-numeric status ('RA') is Default.
    """
    out = np.empty(len(st), dtype=np.int8)
    out[:] = S_DQ90
    out[st == 0] = S_CUR
    out[st == 1] = S_DQ30
    out[st == 2] = S_DQ60
    out[st >= 3] = S_DQ90
    out[st < 0] = S_DEFAULT                 # RA / non-numeric, per D7-D18
    out[zb == 1] = S_PREPAID
    out[zb == 2] = S_DEFAULT
    return out


def write_partition(frame, path, stamp):
    """Write one quarter's frame, footer stamped with the build.

    Returns the Arrow schema of the table actually written, as
    [(column, type string)] - the manifest's record of it, taken here rather
    than read back from the footer. Same conversion and compression as
    frame.to_parquet(engine="pyarrow", index=False, compression="snappy").
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[STAMP_KEY] = stamp.encode("ascii")
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path, compression="snappy")
    return [(f.name, str(f.type)) for f in table.schema]


def write_manifest(rep, panel_dir=PANEL_DIR):
    """Write panel_manifest.json from the Report. Call only once the build has
    succeeded; the file's existence is the claim that it did."""
    # A quarter with zero rows returns from build_quarter before writing any
    # file, yet still appears in rep.quarters. Filter on rows > 0 so the
    # manifest lists exactly the files on disk - do not simplify this to
    # len(rep.quarters).
    written = [(q, rows) for q, rows, _ in rep.quarters if rows > 0]
    schemas = {tuple(rep.schemas[q]) for q, _ in written}
    if len(schemas) != 1:
        raise RuntimeError("partitions were written with %d different schemas; "
                           "no manifest written" % len(schemas))
    manifest = {
        "build_stamp": rep.build_stamp,
        "n_partitions": len(written),
        "total_rows": int(sum(rows for _, rows in written)),
        "partitions": [{"file": PANEL_FILE % q, "quarter": q, "rows": int(rows)}
                       for q, rows in written],
        "schema": [{"name": c, "type": t} for c, t in schemas.pop()],
    }
    path = os.path.join(panel_dir, MANIFEST_NAME)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(path + ".tmp", path)
    return manifest


def build_quarter(perf_path, orig_path, pidx, oidx, macro, rep):
    """Build, filter and write one quarter. Returns the written row count."""
    unrate, unchg, pmms = macro
    t0 = time.time()

    # ---- 1. read and sort by (loan_id, period) --------------------------
    tail, period, st, age, zb, mod, prefixes, dropped = read_perf(perf_path,
                                                                   pidx)
    n = len(tail)
    rep.perf_rows += n + dropped
    rep.dropped_rows += dropped
    if len(prefixes) != 1:
        raise SystemExit("%s carries %d loan-id prefixes (%r); the tail is only "
                         "unique within one prefix"
                         % (os.path.basename(perf_path), len(prefixes),
                            sorted(prefixes)))
    prefix = next(iter(prefixes))

    is_first = np.empty(n, dtype=bool)
    is_first[0] = True
    np.not_equal(tail[1:], tail[:-1], out=is_first[1:])
    is_last = np.empty(n, dtype=bool)
    is_last[:-1] = is_first[1:]
    is_last[-1] = True
    n_loans = int(is_first.sum())
    rep.loans += n_loans

    # ---- 2. detect gaps, mark them (R3) ---------------------------------
    same = np.zeros(n, dtype=bool)
    same[1:] = ~is_first[1:]
    dp = np.zeros(n, dtype=np.int32)
    dp[1:] = period[1:] - period[:-1]
    consecutive = same & (dp == 1)          # row i continues row i-1
    gap_row = same & (dp > 1)               # row i sits just after a hole
    rep.adjacent_pairs += int(same.sum())
    rep.gap_pairs += int(gap_row.sum())
    rep.dup_pairs += int((same & (dp <= 0)).sum())

    state = row_states(st, zb)

    # ---- 3. run_length over the FULL history, before any filtering (R5) --
    ar = np.arange(n, dtype=np.int64)
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    new_run[1:] = is_first[1:] | (dp[1:] != 1) | (state[1:] != state[:-1])
    run_start = np.maximum.accumulate(np.where(new_run, ar, 0))
    run_length = (ar - run_start + 1).astype(np.int32)
    del ar, new_run, run_start

    # ---- 3b. modification events over the FULL history (R7) --------------
    is_mod_row, months_since_row, unflagged = modification_columns(
        tail, period, age, mod)
    rep.mod_loans += len(np.unique(tail[(mod == MOD_Y) | (mod == MOD_P)]))
    rep.unflagged_mod_loans += [prefix + "%07d" % t
                                for t in np.unique(tail[unflagged])]
    del unflagged

    # ---- censoring record (R2), one per censored loan -------------------
    last_idx = np.nonzero(is_last)[0]
    last_zb = zb[last_idx]
    cens_reason = np.full(len(last_idx), -1, dtype=np.int8)
    cens_reason[last_zb == 3] = 0           # code 15
    cens_reason[last_zb == 4] = 1           # code 16
    cens_reason[last_zb == 5] = 2           # code 96
    cens_reason[last_zb == 0] = 3           # simply stops appearing
    is_censored = cens_reason >= 0
    rep.terminated += int((~is_censored).sum())
    for k in range(len(CENSOR_REASONS)):
        rep.censored[k] += int((cens_reason == k).sum())
    censor_frame = pd.DataFrame({
        "loan_id": [prefix + "%07d" % t for t in tail[last_idx][is_censored]],
        "last_observed_period": to_yyyymm(period[last_idx][is_censored]
                                          .astype(np.int64)).astype(np.int32),
        "censor_reason": pd.Categorical(
            [CENSOR_REASONS[k] for k in cens_reason[is_censored]],
            categories=CENSOR_REASONS),
    })
    os.makedirs(CENSOR_DIR, exist_ok=True)
    found = re.search(r"(20\d\dQ[1-4])", os.path.basename(perf_path))
    if not found:
        raise SystemExit("cannot read a quarter name from %r" % perf_path)
    quarter = found.group(1)
    censor_frame.to_parquet(os.path.join(CENSOR_DIR, "censor_%s.parquet" % quarter),
                            engine="pyarrow", index=False, compression="snappy")

    # ---- 7a. candidate pairs: consecutive, non-absorbing origin (R1, R3) -
    nxt = np.nonzero(consecutive)[0]        # pair is (nxt-1 -> nxt)
    org = nxt - 1
    origin_state = state[org]
    next_state = state[nxt]
    live = ~np.isin(origin_state, ABSORBING)
    rep.suppressed += int((~live).sum())
    nxt, org = nxt[live], org[live]
    origin_state, next_state = origin_state[live], next_state[live]
    rep.candidates += len(org)

    origin_month = period[org].astype(np.int64)

    # ---- 4. assign role (R6) --------------------------------------------
    loan_tail_of_pair = tail[org]
    uniq_tail = tail[is_first]
    loan_ids = [prefix + "%07d" % t for t in uniq_tail]
    is_train = split_flags(loan_ids)
    rep.train_loans += int(is_train.sum())
    rep.test_loans += int((~is_train).sum())
    train_of_pair = is_train[np.searchsorted(uniq_tail, loan_tail_of_pair)]

    before_cut = origin_month <= CUT
    role = np.full(len(org), R_EXCLUDED, dtype=np.int8)
    role[train_of_pair & before_cut] = R_FIT
    role[~train_of_pair & before_cut] = R_CONDITION
    role[~train_of_pair & ~before_cut] = R_EVALUATE
    keep = role >= 0
    rep.excluded_train_late += int((~keep).sum())

    candidate_tails = tail[org]        # before the role filter, for the zero-row tally
    nxt, org = nxt[keep], org[keep]
    origin_state, next_state = origin_state[keep], next_state[keep]
    origin_month, role = origin_month[keep], role[keep]
    m = len(org)
    if m == 0:
        return 0, quarter, time.time() - t0

    # ---- R4 flags, on the RAW status, both ends numeric ------------------
    a_raw, b_raw = st[org].astype(np.int32), st[nxt].astype(np.int32)
    both_numeric = (a_raw >= 0) & (b_raw >= 0)
    is_skip = both_numeric & ((b_raw - a_raw) >= 2)
    is_fall = both_numeric & ((a_raw - b_raw) >= 2)
    post_gap = gap_row[org]
    rep.skips += int(is_skip.sum())
    rep.falls += int(is_fall.sum())
    rep.post_gap += int(post_gap.sum())

    # ---- 5. join macro on the ORIGIN period ------------------------------
    unemp_level = lookup(unrate, origin_month)
    unemp_chg = lookup(unchg, origin_month)

    # ---- 6. join origination on loan_id ----------------------------------
    orig = load_origination(orig_path, oidx, pmms, rep.state_categories)
    feats, matched = join_origination(orig, tail[org])
    rep.orig_matched += matched

    # loan_age from the origination month, not Freddie's reported age (R7)
    orig_month = feats["orig_month"]
    if np.any(orig_month < 0):
        raise RuntimeError(
            "%s: %d panel rows belong to loans with no usable first payment "
            "date, so loan_age cannot be computed" % (quarter,
                                                      int((orig_month < 0).sum())))
    true_age = (origin_month - orig_month).astype(np.int32)
    reported = age[org]
    changed = (reported != -9999) & (true_age != reported)
    rep.age_corrected += int(changed.sum())
    for v, c in pd.Series(true_age[changed] - reported[changed]).value_counts().items():
        rep.age_corrections[int(v)] = rep.age_corrections.get(int(v), 0) + int(c)
    rep.age_negative += int((true_age < 0).sum())
    loan_age = pd.array(true_age, dtype="Int32")

    is_modified = is_mod_row[org]
    months_since = pd.array(months_since_row[org], dtype="Int32")
    months_since[~is_modified] = pd.NA
    rep.rows_modified += int(is_modified.sum())
    loan_id = pd.array(
        prefix + pd.Series(tail[org]).astype(str).str.zfill(7), dtype="string")

    frame = pd.DataFrame({
        "loan_id": loan_id,
        "origin_period": to_yyyymm(origin_month).astype(np.int32),
        "origin_state": pd.Categorical.from_codes(origin_state, categories=STATES),
        "next_state": pd.Categorical.from_codes(next_state, categories=STATES),
        "run_length": run_length[org],
        "loan_age": loan_age,
        "is_modified": is_modified,
        "months_since_modification": months_since,
        "fico": feats["fico"],
        "ltv": feats["ltv"],
        "dti": feats["dti"],
        "rate_spread": feats["rate_spread"],
        "unemp_level": unemp_level,
        "unemp_chg_12m": unemp_chg,
        "hpi_feature": np.full(m, np.nan, dtype=np.float32),
        "property_state": pd.Categorical.from_codes(
            feats["property_state"], categories=orig["state_categories"]),
        "role": pd.Categorical.from_codes(role, categories=ROLES),
        "post_gap": post_gap,
        "is_skip": is_skip,
        "is_fall": is_fall,
        "msa": pd.array(feats["msa"], dtype="string"),
        "postal_code": pd.array(feats["postal_code"], dtype="string"),
    })[SCHEMA]

    # Deterministic row order, already achieved: the source was lexsorted on
    # (tail, period), loan_id is that tail zero-padded to a fixed width behind a
    # constant prefix - so it orders identically - and org indexes that array in
    # increasing order. Asserted rather than re-sorted, which would copy the
    # whole frame to no effect.
    if not (np.all(np.diff(tail[org]) >= 0)
            and np.all(np.diff(origin_month)[np.diff(tail[org]) == 0] > 0)):
        raise RuntimeError("emitted rows are not in (loan_id, origin_period) "
                           "order for %s" % quarter)
    # R7: loan_age never decreases within a loan - no exceptions.
    same_loan = np.diff(tail[org]) == 0
    if not np.all(np.diff(true_age)[same_loan] >= 0):
        raise RuntimeError("loan_age decreases within a loan in %s" % quarter)

    # ---- report tallies --------------------------------------------------
    rep.by_role += np.bincount(role, minlength=len(ROLES))
    rep.by_origin += np.bincount(origin_state, minlength=len(STATES))
    rep.by_next += np.bincount(next_state, minlength=len(STATES))
    fit = role == R_FIT
    if fit.any():
        np.add.at(rep.fit_matrix, (origin_state[fit], next_state[fit]), 1)
    for col in SCHEMA:
        rep.nulls[col] += int(frame[col].isna().sum())

    # ---- loans that contribute no row, and why --------------------------
    # Three distinguishable causes, separated so the report does not lump a
    # data property together with a deliberate design choice.
    rows_per_loan = np.bincount(np.searchsorted(uniq_tail, tail),
                                minlength=len(uniq_tail))
    has_candidate = np.isin(uniq_tail, candidate_tails)
    has_emitted = np.isin(uniq_tail, tail[org])
    silent = ~has_emitted
    single = silent & (rows_per_loan == 1)
    no_candidate = silent & (rows_per_loan > 1) & ~has_candidate
    role_only = silent & has_candidate
    rep.bump_zero("single observation, no pair possible", int(single.sum()))
    rep.bump_zero("every pair dropped by R3 (gap) or R1 (absorbing origin)",
                  int(no_candidate.sum()))
    rep.bump_zero("all pairs removed by the R6 role filter (train, after cut)",
                  int(role_only.sum()))

    os.makedirs(PANEL_DIR, exist_ok=True)
    rep.schemas[quarter] = write_partition(
        frame, os.path.join(PANEL_DIR, PANEL_FILE % quarter), rep.build_stamp)
    rep.panel_rows += m
    return m, quarter, time.time() - t0


# ==========================================================================
# report
# ==========================================================================
def write_report(rep, pidx, oidx, perf_sheet, orig_sheet, u_prov, p_prov,
                 elapsed):
    rule("=")
    emit("16_panel_build_report - transition panel, Freddie Mac SFLLD 2006-2007")
    rule("=")
    emit("builder     : src/panel.py")
    emit("panel out   : %s/panel_<quarter>.parquet" % PANEL_DIR)
    emit("censor out  : %s/censor_<quarter>.parquet" % CENSOR_DIR)
    emit("chunksize   : %s" % fmt(CHUNKSIZE))
    emit("split       : %d/%d over loans, keyed BLAKE2b of loan_id, seed %r"
         % (TRAIN_PCT, 100 - TRAIN_PCT, SPLIT_SEED))
    emit("cut         : origin month <= %d-%02d is the fit window"
         % (CUT_YEAR, CUT_MONTH))
    emit()

    emit("RESOLVED COLUMN INDICES   (0-based index = published position - 1)")
    rule()
    emit("%-16s%-10s%10s%8s   %s" % ("file", "role", "position", "index",
                                     "attribute name"))
    rule()
    for label, _ in PERF_WANTED:
        i, pos, attribute = pidx[label]
        emit("%-16s%-10s%10d%8d   %s" % ("performance", label, pos, i, attribute))
    for label, _ in ORIG_WANTED:
        i, pos, attribute = oidx[label]
        emit("%-16s%-10s%10d%8d   %s" % ("origination", label, pos, i, attribute))
    rule()
    emit("performance sheet: %s      origination sheet: %s"
         % (perf_sheet, orig_sheet))
    emit("indices 0, 1 and 3 asserted against the values the project already")
    emit("uses; a mismatch stops the run before any file is read.")
    emit()

    emit("LOAN-TO-FILE CONTAINMENT   (checked before any write)")
    rule()
    n_files = len(rep.prefix_by_file)
    n_prefixes = len(set(rep.prefix_by_file.values()))
    emit("prefix check: %s - %d perf files checked, %d distinct prefixes"
         % ("FAIL, overridden by --allow-multi-file-loans"
            if rep.prefix_collisions else "PASS", n_files, n_prefixes))
    for name in sorted(rep.prefix_by_file):
        emit("    %-24s%s%s" % (name, rep.prefix_by_file[name],
                                "   COLLISION"
                                if rep.prefix_by_file[name] in rep.prefix_collisions
                                else ""))
    emit("Distinct prefixes here, plus the per-file single-prefix check during")
    emit("the build, mean no loan spans two files, so run_length never resets")
    emit("at a file boundary.")
    emit()

    emit("MACRO SERIES")
    rule()
    for prov in (u_prov, p_prov):
        emit("%-14s %-12s %s" % (prov["id"], prov.get("source", "?"),
                                 prov.get("range", "")))
        if prov.get("source") == "UNAVAILABLE":
            emit("    NOT AVAILABLE - the columns it feeds are emitted as null")
            emit("    %s" % prov.get("error", ""))
        else:
            emit("    %s  (%s months, cached at %s)"
                 % (prov["path"], fmt(prov.get("n_months", 0)),
                    prov.get("fetched", "?")))
    emit()
    emit("unemp_level / unemp_chg_12m are joined on the ORIGIN month.")
    emit("rate_spread is the note rate minus the PMMS 30-year fixed rate in the")
    emit("origination month, taken as the first payment month minus one.")
    emit()

    emit("PER-QUARTER BUILD")
    rule()
    emit("%-12s%16s%10s" % ("quarter", "rows written", "seconds"))
    rule()
    for quarter, rows, secs in rep.quarters:
        emit("%-12s%16s%10.1f" % (quarter, fmt(rows), secs))
    rule()
    emit("%-12s%16s%10.1f" % ("TOTAL", fmt(rep.panel_rows), elapsed))
    emit()

    # ---- identity -------------------------------------------------------
    rule("=")
    emit("ROWS IN, ROWS OUT")
    rule("=")
    emit("Two stages. The first is pure pair arithmetic and must balance exactly.")
    emit("The second is the role filter of R6, which deliberately discards rows.")
    emit()
    emit("%-58s %18s" % ("performance rows read", fmt(rep.perf_rows)))
    emit("%-58s %18s" % ("rows rejected (unparseable period or loan id)",
                         fmt(rep.dropped_rows)))
    emit("%-58s %18s" % ("loans", fmt(rep.loans)))
    emit("%-58s %18s" % ("gap-spanning pairs dropped (R3)", fmt(rep.gap_pairs)))
    emit("%-58s %18s" % ("duplicate-period pairs", fmt(rep.dup_pairs)))
    emit("%-58s %18s" % ("pairs with an absorbing origin, suppressed (R1)",
                         fmt(rep.suppressed)))
    rule()
    identity = (rep.perf_rows - rep.dropped_rows - rep.loans - rep.gap_pairs
                - rep.dup_pairs - rep.suppressed)
    emit("%-58s %18s" % ("perf_rows - rejected - loans - gaps - dups - suppressed",
                         fmt(identity)))
    emit("%-58s %18s" % ("candidate pairs built", fmt(rep.candidates)))
    emit("%-58s %18s" % ("identity holds",
                         "yes" if identity == rep.candidates else "NO"))
    rule()
    emit("%-58s %18s %12s" % ("candidate pairs", fmt(rep.candidates), "100.0000%"))
    emit("%-58s %18s %12s" % ("excluded: train-split loan, origin after the cut",
                              fmt(rep.excluded_train_late),
                              pct(rep.excluded_train_late, rep.candidates)))
    emit("%-58s %18s %12s" % ("PANEL ROWS WRITTEN", fmt(rep.panel_rows),
                              pct(rep.panel_rows, rep.candidates)))
    emit("%-58s %18s" % ("candidates - excluded == written",
                         "yes" if rep.candidates - rep.excluded_train_late
                         == rep.panel_rows else "NO"))
    emit()

    # ---- R7 ---------------------------------------------------------------
    rule("=")
    emit("LOAN AGE AND MODIFICATION (R7)")
    rule("=")
    n_unflagged = len(rep.unflagged_mod_loans)
    emit("%-58s %18s %12s" % ("loans with Y or P on any row (unit: loans)",
                              fmt(rep.mod_loans), pct(rep.mod_loans, rep.loans)))
    emit("%-58s %18s %12s" % ("loans modified by age reset only, no flag (loans)",
                              fmt(n_unflagged), pct(n_unflagged, rep.loans)))
    emit("%-58s %18s %12s" % ("panel rows at or after first modification",
                              fmt(rep.rows_modified),
                              pct(rep.rows_modified, rep.panel_rows)))
    emit("%-58s %18s %12s" % ("panel rows whose loan_age differs from Freddie's",
                              fmt(rep.age_corrected),
                              pct(rep.age_corrected, rep.panel_rows)))
    median = "n/a"
    if rep.age_corrections:
        half, seen = rep.age_corrected / 2.0, 0
        for v in sorted(rep.age_corrections):
            seen += rep.age_corrections[v]
            if seen >= half:
                median = "%+d months" % v
                break
    emit("%-58s %18s" % ("median correction (new loan_age - Freddie's)", median))
    emit("%-58s %18s" % ("panel rows with a negative loan_age", fmt(rep.age_negative)))
    emit()
    emit("loan_age = origin month - (first payment month - 1). Freddie's reported")
    emit("age restarts at modification; this does not. Unflagged loans have")
    emit("each age reset treated as a modification:")
    for i in range(0, n_unflagged, 6):
        emit("    " + "  ".join(rep.unflagged_mod_loans[i:i + 6]))
    emit()

    # ---- roles and states ------------------------------------------------
    rule("=")
    emit("COUNTS BY ROLE, ORIGIN STATE AND NEXT STATE")
    rule("=")
    emit("Unit: panel rows (%s)." % fmt(rep.panel_rows))
    emit()
    emit("%-22s%18s%12s" % ("role", "rows", "share"))
    rule()
    for i, r in enumerate(ROLES):
        emit("%-22s%18s%12s" % (r, fmt(rep.by_role[i]),
                                pct(rep.by_role[i], rep.panel_rows)))
    rule()
    emit("%-22s%18s%12s" % ("TOTAL", fmt(rep.by_role.sum()), "100.0000%"))
    emit()
    emit("%-22s%18s%12s%18s%12s" % ("state", "as origin", "share",
                                    "as next", "share"))
    rule()
    for i, s in enumerate(STATES):
        emit("%-22s%18s%12s%18s%12s"
             % (s, fmt(rep.by_origin[i]), pct(rep.by_origin[i], rep.panel_rows),
                fmt(rep.by_next[i]), pct(rep.by_next[i], rep.panel_rows)))
    rule()
    emit("%-22s%18s%12s%18s%12s"
         % ("TOTAL", fmt(rep.by_origin.sum()), "100.0000%",
            fmt(rep.by_next.sum()), "100.0000%"))
    emit()
    emit("Prepaid and Default never appear as an origin: R1 suppresses those rows.")
    emit()

    # ---- fit transition matrix ------------------------------------------
    rule("=")
    emit("TRANSITION COUNTS, FIT ROWS ONLY   (unit: panel rows with role = fit)")
    rule("=")
    emit("Descriptive only - no rates, no estimation.")
    emit()
    total_fit = int(rep.fit_matrix.sum())
    emit("%-14s%16s%s" % ("from \\ to", "rows",
                          "".join("%15s" % s for s in STATES)))
    rule()
    for i, s in enumerate(STATES):
        row = rep.fit_matrix[i]
        emit("%-14s%16s%s" % (s, fmt(row.sum()),
                              "".join("%15s" % fmt(c) for c in row)))
    rule()
    emit("%-14s%16s%s" % ("TOTAL into", fmt(total_fit),
                          "".join("%15s" % fmt(c)
                                  for c in rep.fit_matrix.sum(axis=0))))
    emit()

    # ---- flags ------------------------------------------------------------
    rule("=")
    emit("DROPPED PAIRS AND FLAGGED ROWS")
    rule("=")
    emit("%-58s %18s %12s" % ("gap-spanning pairs dropped (R3)",
                              fmt(rep.gap_pairs), pct(rep.gap_pairs,
                                                      rep.candidates)))
    emit("%-58s %18s %12s" % ("rows flagged post_gap", fmt(rep.post_gap),
                              pct(rep.post_gap, rep.panel_rows)))
    emit("%-58s %18s %12s" % ("rows flagged is_skip", fmt(rep.skips),
                              pct(rep.skips, rep.panel_rows)))
    emit("%-58s %18s %12s" % ("rows flagged is_fall", fmt(rep.falls),
                              pct(rep.falls, rep.panel_rows)))
    emit()
    emit("Skips and falls are kept as ordinary transitions (R4). The counts above")
    emit("are over WRITTEN rows, so they sit below the panel-wide figures in")
    emit("diagnostics 14 and 15 by exactly the rows the R6 role filter removed.")
    emit()

    # ---- censoring --------------------------------------------------------
    rule("=")
    emit("CENSORED LOANS BY REASON   (unit: loans)")
    rule("=")
    emit("Written per quarter to %s. Censoring is recorded here and never" % CENSOR_DIR)
    emit("appears in next_state (R2).")
    emit()
    total_cens = int(rep.censored.sum())
    emit("%-40s%18s%12s%12s" % ("censor_reason", "loans", "of censored",
                                "of loans"))
    rule()
    for i, r in enumerate(CENSOR_REASONS):
        emit("%-40s%18s%12s%12s" % (r, fmt(rep.censored[i]),
                                    pct(rep.censored[i], total_cens),
                                    pct(rep.censored[i], rep.loans)))
    rule()
    emit("%-40s%18s%12s%12s" % ("TOTAL censored", fmt(total_cens), "100.0000%",
                                pct(total_cens, rep.loans)))
    emit("%-40s%18s%12s%12s" % ("terminated (01/02/03/09, not censored)",
                                fmt(rep.terminated), "",
                                pct(rep.terminated, rep.loans)))
    emit("%-40s%18s" % ("censored + terminated", fmt(total_cens + rep.terminated)))
    emit("%-40s%18s" % ("loans", fmt(rep.loans)))
    emit()

    # ---- loans with no rows ----------------------------------------------
    rule("=")
    emit("LOANS WITH ZERO EMITTED ROWS   (unit: loans)")
    rule("=")
    total_zero = sum(rep.zero_row_loans.values())
    emit("%-58s%18s%12s" % ("reason", "loans", "of loans"))
    rule()
    for reason in sorted(rep.zero_row_loans):
        c = rep.zero_row_loans[reason]
        emit("%-58s%18s%12s" % (reason, fmt(c), pct(c, rep.loans)))
    rule()
    emit("%-58s%18s%12s" % ("TOTAL", fmt(total_zero), pct(total_zero, rep.loans)))
    emit()
    emit("%-58s%18s" % ("loans contributing at least one row",
                        fmt(rep.loans - total_zero)))
    emit("%-58s%18s%12s" % ("train-split loans", fmt(rep.train_loans),
                            pct(rep.train_loans, rep.loans)))
    emit("%-58s%18s%12s" % ("test-split loans", fmt(rep.test_loans),
                            pct(rep.test_loans, rep.loans)))
    emit()
    emit("%-58s%18s%12s" % ("panel rows matched to an origination record",
                            fmt(rep.orig_matched),
                            pct(rep.orig_matched, rep.panel_rows)))
    emit("%-58s%18s" % ("panel rows with no origination match",
                        fmt(rep.panel_rows - rep.orig_matched)))
    emit()

    # ---- nulls -------------------------------------------------------------
    rule("=")
    emit("NULLS PER COLUMN   (unit: panel rows, of %s)" % fmt(rep.panel_rows))
    rule("=")
    emit("%-28s%-14s%18s%12s" % ("column", "group", "nulls", "share"))
    rule()
    for col in SCHEMA:
        group = ("keys" if col in KEYS else "states" if col in STATE_COLS
                 else "features" if col in FEATURES else "metadata")
        emit("%-28s%-14s%18s%12s" % (col, group, fmt(rep.nulls[col]),
                                     pct(rep.nulls[col], rep.panel_rows)))
    emit()
    emit("hpi_feature is null by construction: it is a declared placeholder and")
    emit("its encoding is not yet decided.")
    emit()

    rule("=")
    emit("runtime: %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    rule("=")


def build(allow_multi_file_loans=False):
    """Build every quarter, write the panel and the build report."""
    t_start = time.time()
    pidx, oidx, perf_sheet, orig_sheet = resolve()
    perf_paths = sorted(glob.glob(PERF_GLOB))
    orig_paths = sorted(glob.glob(ORIG_GLOB))
    if not perf_paths:
        raise SystemExit("no performance files matched %s" % PERF_GLOB)
    if len(perf_paths) != len(orig_paths):
        raise SystemExit("%d performance files but %d origination files"
                         % (len(perf_paths), len(orig_paths)))

    # Runs before the first write of any kind: nothing above touches disk.
    rep = Report()
    rep.prefix_by_file = {os.path.basename(p): perf_prefix(p, pidx)
                          for p in perf_paths}
    rep.prefix_collisions = check_prefixes(rep.prefix_by_file,
                                           allow_multi_file_loans)
    rep.state_categories = state_categories(orig_paths, oidx)

    # First write of the build: remove the previous manifest, so a build that
    # stops partway leaves partitions but no manifest vouching for them.
    manifest_path = os.path.join(PANEL_DIR, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    rep.build_stamp = datetime.datetime.now().isoformat(timespec="microseconds")

    unrate, unchg, pmms, u_prov, p_prov = macro_tables()

    print("building %d quarters" % len(perf_paths), flush=True)
    for perf_path, orig_path in zip(perf_paths, orig_paths):
        rows, quarter, secs = build_quarter(perf_path, orig_path, pidx, oidx,
                                            (unrate, unchg, pmms), rep)
        rep.quarters.append((quarter, rows, secs))
        print("    %-10s %14s rows  (%5.1fs)" % (quarter, fmt(rows), secs),
              flush=True)

    write_report(rep, pidx, oidx, perf_sheet, orig_sheet, u_prov, p_prov,
                 time.time() - t_start)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
    write_manifest(rep)             # last: only a completed build gets one
    return rep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the transition panel.")
    parser.add_argument(
        "--allow-multi-file-loans", action="store_true",
        help="downgrade a loan-id prefix shared across perf files from an error "
             "to a warning. Only for deliberate repartitioning: a loan split "
             "across files gets its run_length reset at the file boundary.")
    build(allow_multi_file_loans=parser.parse_args().allow_multi_file_loans)
