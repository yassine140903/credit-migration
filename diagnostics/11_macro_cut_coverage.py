"""Does the test period sit inside the macro range the model saw in training?

We are choosing the calendar date for the primary train/test cut. The failure
mode being measured is not overfitting - it is extrapolation. If the test
window contains macro conditions that never occurred in the train window, the
held-out evaluation stops being a test of generalisation and becomes a test of
extrapolation, and a model that scores badly there may be perfectly well
specified for the range it was actually shown. This script quantifies, for four
candidate cuts, how much of the test period sits outside the training macro
range - so the cut is chosen on evidence rather than on the calendar.

MACRO SERIES, monthly
    UNRATE      national unemployment rate, percent of labour force (FRED)
    CSUSHPISA   S&P CoreLogic Case-Shiller U.S. National Home Price Index,
                seasonally adjusted, index 2000-01 = 100 (FRED, monthly)
    Both are downloaded from FRED and cached to data/macro/<id>.csv. If a
    series cannot be reached and is not already cached, the script STOPS and
    names the series that failed. No substitute series is ever used.

BOTH FORMS ARE REPORTED
    level        the value itself
    12m change   value minus the value twelve months earlier, in the series'
                 own units
The 12-month change is the form that matters. A level can drift far outside a
short training window purely because the series is trending, which says little
about whether the DYNAMICS are novel; the 12-month change is the form in which
2008-2011 is genuinely unlike 2006-2008. The level is reported alongside so the
contrast between the two is visible rather than asserted.

WINDOW AND THE 12-MONTH LOOKBACK
    The reported window is 2006-01 .. 2013-12. The series are fetched from
    2005-01 so that the 12-month change is DEFINED at 2006-01; without that the
    change form would be silently missing for the first twelve months of every
    train window, which would shrink the earliest cut's training range for
    reasons of data handling rather than economics. The 2005 values are used
    only as the lookback term and are never themselves a train or test month.

CANDIDATE CUTS   2008-06, 2008-09, 2008-12, 2009-06
    train  2006-01 .. cut month inclusive
    test   month after the cut .. 2013-12

PER CUT, PER SERIES, PER FORM
    1  train min, train max                        (and the width, max - min)
    2  test min, test max
    3  share of test months strictly outside the train [min, max]
    4  largest excursion beyond the train range, in the series' own units and
       as a fraction of the train range width, with its direction and month

EXPOSURE WEIGHTING
    Calendar months are not equally important: the 2006-2007 vintages shed
    loans continuously, so a 2013 month carries far less risk than a 2009 one
    and an unweighted share of calendar months overstates the importance of the
    thin tail. Loan-months per calendar month are counted from the performance
    files - columns 0 and 1 only, chunked - and metric 3 is repeated weighted by
    those counts. The headline is therefore the share of test LOAN-MONTHS at
    out-of-range macro values. Both forms are reported side by side.

    Rows are treated as loan-months. That is not an assumption: diagnostic
    01_panel_shape.txt reports "rows > span (duplicate months present) = 0" for
    every one of the eight files, so no (loan, period) pair is repeated and the
    row count per calendar month IS the loan-month count.

WHAT THIS DOES NOT SHOW
    Out-of-range does not mean unpredictable, and in-range does not mean easy.
    Two national series are a coarse summary of the macro environment: a cut
    can look clean on both and still put the test period in a different regime
    in dimensions not measured here (regional dispersion, credit supply, policy
    intervention). The metrics below bound one specific risk - evaluation on
    macro values never seen in training - and nothing wider.

No file is ever held whole. No git command is run.
"""

import collections
import glob
import os
import time
import urllib.error
import urllib.request

import pandas as pd

GLOB = "data/historical_data_200[67]/*/perf_*.txt"
CHUNKSIZE = 2_000_000
OUT = os.path.join("diagnostics", "11_macro_cut_coverage.txt")
MACRO_DIR = os.path.join("data", "macro")
RULE_W = 121

COL_LOAN = 0
COL_PERIOD = 1
PASS_COLS = [COL_LOAN, COL_PERIOD]

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s"
FRED_TIMEOUT = 60
FRED_ATTEMPTS = 3

# (series id, description, unit label)
SERIES = [
    ("UNRATE", "national unemployment rate, percent of labour force", "pp"),
    ("CSUSHPISA",
     "S&P CoreLogic Case-Shiller U.S. national home price index, SA", "idx"),
]

FORMS = ["level", "12m change"]

WIN_START = (2006, 1)
WIN_END = (2013, 12)
FETCH_START = (2005, 1)   # 12-month lookback for WIN_START
CUTS = [(2008, 6), (2008, 9), (2008, 12), (2009, 6)]

LOOKBACK = 12

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


def pct(num, den, dp=3):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


# --------------------------------------------------------------------------
# month arithmetic - a month index is year*12 + (month - 1)
# --------------------------------------------------------------------------
def mi(y, m):
    return y * 12 + (m - 1)


def mlabel(i):
    return "%04d-%02d" % (i // 12, i % 12 + 1)


def mrange(a, b):
    return list(range(a, b + 1))


WIN_A, WIN_B = mi(*WIN_START), mi(*WIN_END)
FETCH_A = mi(*FETCH_START)
WINDOW = mrange(WIN_A, WIN_B)


# --------------------------------------------------------------------------
# PART A - macro series
# --------------------------------------------------------------------------
def fetch_series(sid):
    """Return (values dict month_index -> float, provenance dict).

    Cache first, network second. Raises RuntimeError naming the series if
    neither is available.
    """
    path = os.path.join(MACRO_DIR, sid + ".csv")
    prov = {"id": sid, "path": path, "url": FRED_CSV % sid}

    if os.path.exists(path) and os.path.getsize(path) > 0:
        prov["source"] = "cache"
        prov["fetched"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
        prov["bytes"] = os.path.getsize(path)
    else:
        # No custom User-Agent header. Measured on this host, fredgraph.csv
        # serves the default python-urllib UA in ~0.4s but hangs until timeout
        # on a custom one (tested twice each with two different UA strings), so
        # setting a polite UA here would break the download rather than help it.
        raw = None
        errors = []
        for attempt in range(1, FRED_ATTEMPTS + 1):
            try:
                req = urllib.request.Request(prov["url"])
                with urllib.request.urlopen(req, timeout=FRED_TIMEOUT) as resp:
                    raw = resp.read()
                prov["attempts"] = attempt
                break
            except Exception as exc:      # reported by name, never swallowed
                errors.append("attempt %d: %s: %s"
                              % (attempt, type(exc).__name__, exc))
        if raw is None:
            raise RuntimeError(
                "FRED unreachable for series %s after %d attempt(s) - %s"
                % (sid, FRED_ATTEMPTS, "; ".join(errors)))
        if not raw or len(raw) < 32:
            raise RuntimeError(
                "FRED returned an empty or truncated body for series %s "
                "(%d bytes)" % (sid, len(raw)))
        os.makedirs(MACRO_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
        prov["source"] = "network (attempt %d)" % prov.get("attempts", 1)
        prov["fetched"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prov["bytes"] = len(raw)

    # FRED's CSV date column is 'observation_date' in the current format and
    # 'DATE' in the older one; take whatever the first column is called.
    frame = pd.read_csv(path, na_values=["."], keep_default_na=True)
    if frame.shape[1] < 2:
        raise RuntimeError(
            "cached CSV for %s has %d column(s), expected at least 2"
            % (sid, frame.shape[1]))
    date_col, val_col = frame.columns[0], frame.columns[1]
    prov["columns"] = "%s | %s" % (date_col, val_col)
    prov["rows"] = int(len(frame))

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    vals = pd.to_numeric(frame[val_col], errors="coerce")
    out = {}
    for d, v in zip(dates, vals):
        if pd.isna(d) or pd.isna(v):
            continue
        out[mi(d.year, d.month)] = float(v)
    if not out:
        raise RuntimeError(
            "no parseable observations in cached CSV for %s" % sid)
    prov["obs"] = len(out)
    prov["first"] = mlabel(min(out))
    prov["last"] = mlabel(max(out))
    return out, prov


def build_forms(vals):
    """level and 12m-change dicts over month indices."""
    level = dict(vals)
    chg = {}
    for i in vals:
        j = i - LOOKBACK
        if j in vals:
            chg[i] = vals[i] - vals[j]
    return {"level": level, "12m change": chg}


# --------------------------------------------------------------------------
# PART B - exposure: loan-months per calendar month
# --------------------------------------------------------------------------
def reader_for(path, cols):
    return pd.read_csv(
        path, sep="|", header=None, dtype=str, usecols=cols,
        chunksize=CHUNKSIZE, engine="c", na_filter=False,
    )


def count_exposure(paths):
    """Rows per calendar month, pooled, plus a per-file integrity record."""
    pooled = collections.Counter()      # month index -> rows
    per_file = []

    for path in paths:
        t0 = time.time()
        raw = collections.Counter()     # raw YYYYMM string -> rows
        prefixes = set()
        rows = 0
        blank_loan = 0

        for chunk in reader_for(path, cols=PASS_COLS):
            loan = chunk[COL_LOAN]
            period = chunk[COL_PERIOD]
            rows += len(chunk)
            blank_loan += int((loan.str.len() == 0).sum())
            prefixes.update(loan.str.slice(0, 5).unique().tolist())
            raw.update(period.value_counts().to_dict())

        # Parse the distinct period strings once, not once per row.
        counts = collections.Counter()
        bad = collections.Counter()
        for key, n in raw.items():
            s = key.strip()
            ok = len(s) == 6 and s.isdigit() and 1 <= int(s[4:6]) <= 12
            if ok:
                counts[mi(int(s[:4]), int(s[4:6]))] += n
            else:
                bad[key] += n

        pooled.update(counts)
        per_file.append({
            "path": path,
            "gb": os.path.getsize(path) / 1024.0 ** 3,
            "rows": rows,
            "blank_loan": blank_loan,
            "prefixes": sorted(prefixes),
            "distinct_periods": len(counts),
            "first": mlabel(min(counts)) if counts else "n/a",
            "last": mlabel(max(counts)) if counts else "n/a",
            "unparseable": int(sum(bad.values())),
            "bad_keys": sorted(bad)[:5],
            "secs": time.time() - t0,
        })
    return pooled, per_file


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def coverage(form_vals, train_ms, test_ms, weights):
    """Range, out-of-range shares and the largest excursion for one cut."""
    tr = [form_vals[i] for i in train_ms if i in form_vals]
    te = [(i, form_vals[i]) for i in test_ms if i in form_vals]
    res = {
        "train_n": len(tr),
        "train_missing": len(train_ms) - len(tr),
        "test_n": len(te),
        "test_missing": len(test_ms) - len(te),
    }
    if not tr or not te:
        res["ok"] = False
        return res
    res["ok"] = True

    lo, hi = min(tr), max(tr)
    width = hi - lo
    res.update({"train_min": lo, "train_max": hi, "width": width,
                "test_min": min(v for _, v in te),
                "test_max": max(v for _, v in te)})

    oor = [(i, v) for i, v in te if v < lo or v > hi]
    res["oor_months"] = len(oor)
    res["test_months"] = len(te)

    res["oor_lm"] = sum(weights.get(i, 0) for i, _ in oor)
    res["test_lm"] = sum(weights.get(i, 0) for i, _ in te)

    # Largest excursion beyond the train range, and where it happened.
    best_x = 0.0
    best_i = None
    best_dir = ""
    for i, v in te:
        if v > hi and (v - hi) > best_x:
            best_x, best_i, best_dir = v - hi, i, "above"
        elif v < lo and (lo - v) > best_x:
            best_x, best_i, best_dir = lo - v, i, "below"
    res["exc"] = best_x
    res["exc_month"] = mlabel(best_i) if best_i is not None else "-"
    res["exc_dir"] = best_dir or "-"
    res["exc_frac"] = (best_x / width) if width > 0 else None
    return res


# --------------------------------------------------------------------------
def main():
    t_start = time.time()

    emit("=" * RULE_W)
    emit("11_macro_cut_coverage - macro range coverage by candidate train/test cut")
    emit("=" * RULE_W)
    emit("glob          : %s" % GLOB)
    emit("chunksize     : %s" % fmt(CHUNKSIZE))
    emit("columns read  : 0 loan_sequence_number, 1 monthly_reporting_period  (nothing else)")
    emit("macro cache   : %s" % MACRO_DIR)
    emit("window        : %s .. %s   (%d calendar months)"
         % (mlabel(WIN_A), mlabel(WIN_B), len(WINDOW)))
    emit("fetched from  : %s   so the 12-month change is defined at %s"
         % (mlabel(FETCH_A), mlabel(WIN_A)))
    emit("candidate cuts: %s" % ", ".join("%04d-%02d" % c for c in CUTS))
    emit()

    paths = sorted(glob.glob(GLOB))
    emit("files found   : %d" % len(paths))
    for p in paths:
        emit("    %-62s %6.2f GB" % (p, os.path.getsize(p) / 1024.0 ** 3))
    if not paths:
        emit()
        emit("STOP - the glob matched no performance files. Nothing to count.")
        write_out()
        return
    emit()

    # ----------------------------------------------------------------- PART A
    emit("=" * RULE_W)
    emit("PART A - MACRO SERIES")
    emit("=" * RULE_W)

    series_vals = {}
    provs = {}
    failures = []
    for sid, desc, _unit in SERIES:
        try:
            vals, prov = fetch_series(sid)
            series_vals[sid] = vals
            provs[sid] = prov
        except RuntimeError as exc:
            failures.append((sid, desc, str(exc)))

    if failures:
        emit()
        emit("STOP - a required macro series could not be obtained.")
        rule()
        for sid, desc, msg in failures:
            emit("FAILED   %s  (%s)" % (sid, desc))
            emit("         %s" % msg)
            emit("         cache looked for at: %s"
                 % os.path.join(MACRO_DIR, sid + ".csv"))
        emit()
        for sid, desc, _unit in SERIES:
            if sid in provs:
                emit("OK       %s  (%s)  from %s"
                     % (sid, desc, provs[sid]["source"]))
        emit()
        emit("No substitute series has been used and no coverage metric has been")
        emit("computed. Restore network access to FRED, or place the CSV at the")
        emit("cache path above, and re-run.")
        rule()
        emit("done in %.1fs" % (time.time() - t_start))
        emit("=" * RULE_W)
        write_out()
        return

    emit("A.1  PROVENANCE")
    rule()
    for sid, desc, unit in SERIES:
        pr = provs[sid]
        emit("%-10s %s" % (sid, desc))
        emit("           unit          %s" % unit)
        emit("           url           %s" % pr["url"])
        emit("           source        %s   (cached at %s)"
             % (pr["source"], pr["path"]))
        emit("           fetched       %s" % pr["fetched"])
        emit("           csv           %s bytes, %s rows, columns: %s"
             % (fmt(pr["bytes"]), fmt(pr["rows"]), pr["columns"]))
        emit("           parsed        %s observations, %s .. %s"
             % (fmt(pr["obs"]), pr["first"], pr["last"]))
        emit()

    forms = {sid: build_forms(series_vals[sid]) for sid in series_vals}

    emit("A.2  COVERAGE OF THE WINDOW AND OF THE 12-MONTH LOOKBACK")
    rule()
    emit("%-12s %-12s %10s %10s %10s"
         % ("series", "form", "months", "present", "missing"))
    rule()
    for sid, _desc, _unit in SERIES:
        for form in FORMS:
            fv = forms[sid][form]
            have = sum(1 for i in WINDOW if i in fv)
            emit("%-12s %-12s %10s %10s %10s"
                 % (sid, form, fmt(len(WINDOW)), fmt(have),
                    fmt(len(WINDOW) - have)))
    rule()
    miss_lb = [mlabel(i) for i in mrange(FETCH_A, WIN_A - 1)
               if any(i not in series_vals[s] for s, _d, _u in SERIES)]
    emit("lookback months %s .. %s missing in at least one series: %s"
         % (mlabel(FETCH_A), mlabel(WIN_A - 1),
            ", ".join(miss_lb) if miss_lb else "none"))
    emit()

    emit("A.3  THE SERIES, MONTH BY MONTH  (the input to everything below)")
    rule()
    emit("%-9s %12s %12s   %12s %12s"
         % ("month", "UNRATE", "UNRATE", "CSUSHPISA", "CSUSHPISA"))
    emit("%-9s %12s %12s   %12s %12s"
         % ("", "level (pp)", "12m chg", "level (idx)", "12m chg"))
    rule()
    for i in WINDOW:
        cells = []
        for sid, _desc, _unit in SERIES:
            for form in FORMS:
                v = forms[sid][form].get(i)
                cells.append("%12.3f" % v if v is not None else "%12s" % "n/a")
        emit("%-9s %s %s   %s %s"
             % (mlabel(i), cells[0], cells[1], cells[2], cells[3]))
        if i % 12 == 11:
            emit()
    rule()
    emit()

    # ----------------------------------------------------------------- PART B
    emit("=" * RULE_W)
    emit("PART B - EXPOSURE: LOAN-MONTHS PER CALENDAR MONTH")
    emit("=" * RULE_W)
    emit("Columns 0 and 1 only, chunked. Rows are loan-months: diagnostic 01 reports")
    emit("zero duplicate (loan, period) pairs in every file, so a row count per")
    emit("calendar month is a loan-month count.")
    emit()

    weights, per_file = count_exposure(paths)

    emit("B.1  PER FILE")
    rule()
    emit("%-26s %7s %14s %9s %7s %9s %9s %8s"
         % ("file", "size GB", "rows", "blank id", "months", "first", "last",
            "secs"))
    rule()
    for r in per_file:
        emit("%-26s %7.2f %14s %9s %7s %9s %9s %8.1f"
             % (os.path.basename(r["path"]), r["gb"], fmt(r["rows"]),
                fmt(r["blank_loan"]), fmt(r["distinct_periods"]),
                r["first"], r["last"], r["secs"]))
    rule()
    emit("%-26s %7.2f %14s %9s"
         % ("pooled", sum(r["gb"] for r in per_file),
            fmt(sum(r["rows"] for r in per_file)),
            fmt(sum(r["blank_loan"] for r in per_file))))
    emit()
    emit("loan-id prefixes per file (one vintage per file means loans do not straddle files):")
    for r in per_file:
        emit("    %-26s %s"
             % (os.path.basename(r["path"]), ", ".join(r["prefixes"])))
    bad_total = sum(r["unparseable"] for r in per_file)
    emit()
    emit("rows with an unparseable period: %s" % fmt(bad_total))
    if bad_total:
        for r in per_file:
            if r["unparseable"]:
                emit("    %-26s %s rows, keys e.g. %s"
                     % (os.path.basename(r["path"]), fmt(r["unparseable"]),
                        r["bad_keys"]))
    emit()

    total_lm = sum(weights.values())
    emit("B.2  PANEL EXTENT AND THE PART OF IT THIS DIAGNOSTIC USES")
    rule()
    emit("first calendar month present in the panel      %s" % mlabel(min(weights)))
    emit("last  calendar month present in the panel      %s" % mlabel(max(weights)))
    emit("distinct calendar months present               %s" % fmt(len(weights)))
    emit("total loan-months in the panel                 %s" % fmt(total_lm))
    emit()
    in_win = sum(weights.get(i, 0) for i in WINDOW)
    before = sum(n for i, n in weights.items() if i < WIN_A)
    after = sum(n for i, n in weights.items() if i > WIN_B)
    emit("loan-months before %s                       %14s   %s"
         % (mlabel(WIN_A), fmt(before), pct(before, total_lm)))
    emit("loan-months inside %s .. %s            %14s   %s"
         % (mlabel(WIN_A), mlabel(WIN_B), fmt(in_win), pct(in_win, total_lm)))
    emit("loan-months after  %s                       %14s   %s"
         % (mlabel(WIN_B), fmt(after), pct(after, total_lm)))
    emit()
    emit("Everything below is computed on the %s .. %s window only. The panel"
         % (mlabel(WIN_A), mlabel(WIN_B)))
    emit("runs well past 2013 but the macro window fixed by this brief does not, so the")
    emit("post-2013 loan-months are outside both the train and the test window of every")
    emit("cut and are excluded from every share reported below.")
    emit()

    emit("B.3  LOAN-MONTHS BY CALENDAR MONTH, %s .. %s"
         % (mlabel(WIN_A), mlabel(WIN_B)))
    rule()
    emit("%-9s %14s %9s   %-9s %14s %9s"
         % ("month", "loan-months", "share", "month", "loan-months", "share"))
    rule()
    half = (len(WINDOW) + 1) // 2
    left, right = WINDOW[:half], WINDOW[half:]
    for k in range(half):
        a = left[k]
        la = "%-9s %14s %9s" % (mlabel(a), fmt(weights.get(a, 0)),
                                pct(weights.get(a, 0), in_win))
        if k < len(right):
            b = right[k]
            lb = "%-9s %14s %9s" % (mlabel(b), fmt(weights.get(b, 0)),
                                    pct(weights.get(b, 0), in_win))
        else:
            lb = ""
        emit("%s   %s" % (la, lb))
    rule()
    emit("%-9s %14s %9s" % ("total", fmt(in_win), pct(in_win, in_win)))
    emit()
    missing_months = [mlabel(i) for i in WINDOW if i not in weights]
    emit("window months with no loan-months at all: %s"
         % (", ".join(missing_months) if missing_months else "none"))
    emit()

    # ----------------------------------------------------------------- PART C
    emit("=" * RULE_W)
    emit("PART C - PER CANDIDATE CUT")
    emit("=" * RULE_W)
    emit("train = %s .. cut inclusive        test = month after cut .. %s"
         % (mlabel(WIN_A), mlabel(WIN_B)))
    emit("'outside' is strict: value < train min or value > train max.")
    emit("excursion = how far the worst test month lies beyond the train range, in the")
    emit("series' own units, and as a fraction of the train range width (max - min).")
    emit()

    results = {}
    for idx, (cy, cm) in enumerate(CUTS, start=1):
        cut = mi(cy, cm)
        train_ms = mrange(WIN_A, cut)
        test_ms = mrange(cut + 1, WIN_B)
        tr_lm = sum(weights.get(i, 0) for i in train_ms)
        te_lm = sum(weights.get(i, 0) for i in test_ms)

        emit("=" * RULE_W)
        emit("CUT %s" % mlabel(cut))
        emit("=" * RULE_W)
        emit("train window   %s .. %s   %3d calendar months   %14s loan-months   %s of window"
             % (mlabel(WIN_A), mlabel(cut), len(train_ms), fmt(tr_lm),
                pct(tr_lm, in_win)))
        emit("test  window   %s .. %s   %3d calendar months   %14s loan-months   %s of window"
             % (mlabel(cut + 1), mlabel(WIN_B), len(test_ms), fmt(te_lm),
                pct(te_lm, in_win)))
        emit("balance        calendar months train:test = %s      loan-months train:test = %s"
             % (("%.2f" % (len(train_ms) / float(len(test_ms)))) if test_ms else "n/a",
                ("%.2f" % (tr_lm / float(te_lm))) if te_lm else "n/a"))
        emit()

        emit("C.%d.1  RANGES" % idx)
        rule()
        emit("%-14s %-12s %-6s %12s %12s %12s %12s %12s"
             % ("series", "form", "unit", "train min", "train max",
                "train width", "test min", "test max"))
        rule()
        for sid, _desc, unit in SERIES:
            for form in FORMS:
                r = coverage(forms[sid][form], train_ms, test_ms, weights)
                results[(cut, sid, form)] = r
                if not r["ok"]:
                    emit("%-14s %-12s %-6s   train months with a value: %d, test: %d"
                         " - not computable"
                         % (sid, form, unit, r["train_n"], r["test_n"]))
                    continue
                emit("%-14s %-12s %-6s %12.3f %12.3f %12.3f %12.3f %12.3f"
                     % (sid, form, unit, r["train_min"], r["train_max"],
                        r["width"], r["test_min"], r["test_max"]))
        rule()
        emit()

        emit("C.%d.2  OUT-OF-RANGE SHARE AND LARGEST EXCURSION" % idx)
        rule()
        emit("%-14s %-12s %9s %8s %14s %8s %10s %-5s %8s %8s"
             % ("", "", "test cal", "", "out-of-range", "", "largest", "",
                "as frac", "worst"))
        emit("%-14s %-12s %9s %8s %14s %8s %10s %-5s %8s %8s"
             % ("series", "form", "mths o/r", "share", "loan-months", "share",
                "excursion", "dir", "of width", "month"))
        rule()
        for sid, _desc, _unit in SERIES:
            for form in FORMS:
                r = results[(cut, sid, form)]
                if not r["ok"]:
                    continue
                frac = "n/a" if r["exc_frac"] is None else "%.3f" % r["exc_frac"]
                emit("%-14s %-12s %4s/%4s %8s %14s %8s %10.3f %-5s %8s %8s"
                     % (sid, form, fmt(r["oor_months"]), fmt(r["test_months"]),
                        pct(r["oor_months"], r["test_months"], 2),
                        fmt(r["oor_lm"]), pct(r["oor_lm"], r["test_lm"], 2),
                        r["exc"], r["exc_dir"], frac, r["exc_month"]))
        rule()
        emit("left share  = share of TEST CALENDAR MONTHS outside the train range")
        emit("right share = share of TEST LOAN-MONTHS outside it. They differ because")
        emit("months carry unequal exposure - see B.3.")
        emit()

    # ----------------------------------------------------------------- PART D
    emit("=" * RULE_W)
    emit("PART D - SIDE BY SIDE ACROSS CUTS")
    emit("=" * RULE_W)

    emit("D.1  SHARE OF TEST PERIOD AT OUT-OF-RANGE MACRO VALUES")
    emit("     'cal' = share of test calendar months.  'lm' = share of test LOAN-MONTHS.")
    rule()
    head = "%-14s %-12s" % ("series", "form")
    sub = "%-14s %-12s" % ("", "")
    for cy, cm in CUTS:
        head += "  %17s" % mlabel(mi(cy, cm))
        sub += "  %8s %8s" % ("cal", "lm")
    emit(head)
    emit(sub)
    rule()
    for sid, _desc, _unit in SERIES:
        for form in FORMS:
            line = "%-14s %-12s" % (sid, form)
            for cy, cm in CUTS:
                r = results[(mi(cy, cm), sid, form)]
                if not r["ok"]:
                    line += "  %8s %8s" % ("n/a", "n/a")
                else:
                    line += "  %8s %8s" % (
                        pct(r["oor_months"], r["test_months"], 1),
                        pct(r["oor_lm"], r["test_lm"], 1))
            emit(line)
    rule()
    emit()

    emit("D.2  LARGEST EXCURSION BEYOND THE TRAIN RANGE, AS A FRACTION OF TRAIN WIDTH")
    rule()
    head = "%-14s %-12s" % ("series", "form")
    for cy, cm in CUTS:
        head += "  %19s" % mlabel(mi(cy, cm))
    emit(head)
    rule()
    for sid, _desc, _unit in SERIES:
        for form in FORMS:
            line = "%-14s %-12s" % (sid, form)
            for cy, cm in CUTS:
                r = results[(mi(cy, cm), sid, form)]
                if not r["ok"] or r["exc_frac"] is None:
                    line += "  %19s" % "n/a"
                elif r["oor_months"] == 0:
                    line += "  %19s" % "none - fully inside"
                else:
                    line += "  %19s" % ("%.3f  %s %s" % (
                        r["exc_frac"], r["exc_dir"], r["exc_month"]))
            emit(line)
    rule()
    emit("A fraction of 1.00 means the worst test month lies a full train-range width")
    emit("beyond the edge of the train range.")
    emit()

    emit("D.3  TRAIN/TEST SIZE BALANCE")
    rule()
    emit("%-9s %8s %8s %16s %16s %10s %10s"
         % ("cut", "tr mths", "te mths", "train loan-mths", "test loan-mths",
            "train sh", "test sh"))
    rule()
    for cy, cm in CUTS:
        cut = mi(cy, cm)
        train_ms = mrange(WIN_A, cut)
        test_ms = mrange(cut + 1, WIN_B)
        tr_lm = sum(weights.get(i, 0) for i in train_ms)
        te_lm = sum(weights.get(i, 0) for i in test_ms)
        emit("%-9s %8d %8d %16s %16s %10s %10s"
             % (mlabel(cut), len(train_ms), len(test_ms), fmt(tr_lm),
                fmt(te_lm), pct(tr_lm, in_win, 1), pct(te_lm, in_win, 1)))
    rule()
    emit("Shares are of the %s .. %s window, not of the whole panel."
         % (mlabel(WIN_A), mlabel(WIN_B)))
    emit()

    elapsed = time.time() - t_start
    emit("=" * RULE_W)
    emit("runtime: %.1f s  (%.2f min)" % (elapsed, elapsed / 60.0))
    if elapsed > 20 * 60:
        emit("NOTE: runtime exceeded ~20 minutes. Reported as-is, not optimised.")
    emit("=" * RULE_W)
    write_out()


if __name__ == "__main__":
    main()
