"""Prediction deduplication: how many distinct feature rows (and predictions) does the forecaster really need?   READ-ONLY.

    python diagnostics/35_prediction_dedup.py [--origin all|current]

Writes diagnostics/35_prediction_dedup.txt and nothing else. It reads the panel through panel_io.load_panel and the
saved model artifacts/thinned_lgbm/model.txt, changes nothing under data/, saves no predictions and runs no git command.

A counting exercise. The only things derived from the model are COUNTS of distinct probability vectors; no
probability, loss, accuracy, AUC, Brier score or confusion matrix is printed or saved.

Population: condition rows with from_period == 200812, one row per loan (checked, not assumed). By default every such
row is used, whatever its from_state (the 405,705 'eligible' loans of the horizon report are all transient at the cut);
--origin current restricts to from_state == Current, the saved model's own domain.

Feature rows (the headline arm): fico, ltv, dti, rate_spread, property_state, loan_age, est_ltv, unemp_level,
unemp_chg_12m. run_length and run_length_censored are excluded. Frozen macro: unemp_* are the 2008-12 values (already
the values on these rows); hpi_ratio is the 2008-12 value per loan and est_ltv is re-derived from it exactly as
src/features_hpi.py does (float64 ltv / float64 hpi_ratio, stored as float32), so it is constant per loan across the
horizon. Only loan_age moves: +h.

The saved model was trained WITH run_length and run_length_censored. To score the 9-feature rows it has to be given
those two columns; they are supplied as NaN (LightGBM's missing path) for every row. That fixes them to one constant, so
it cannot create distinctness, but the model is not the headline arm's model and the distinct-PREDICTION figures
describe this model on constant-NaN run-length inputs.
"""

import argparse
import gc
import os
import re
import sys
import threading

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                            # noqa: E402

OUT = os.path.join("diagnostics", "35_prediction_dedup.txt")
MODEL = os.path.join("artifacts", "thinned_lgbm", "model.txt")
CUT = 200812
EXPECTED = 405705
HORIZONS = [0, 1, 6, 12, 24, 36, 48, 60]
GRID = list(range(0, 61))
ROUND = 10
SCORE_CHUNK = 100_000
WORTH = 20.0                                # % reduction below which dedup is judged not worth building
SAMPLE_S = 0.5
ABORT_MB = 500
MB, GB = 2 ** 20, 2 ** 30

FEATURES = ["fico", "ltv", "dti", "rate_spread", "property_state", "loan_age", "est_ltv", "unemp_level",
            "unemp_chg_12m"]
CAT = "property_state"
NON_AGE = [f for f in FEATURES if f != "loan_age"]

proc = psutil.Process()
R = {"completed": False, "abort_reason": None, "phase": "start"}


def rss():
    return proc.memory_info().rss


def avail():
    return psutil.virtual_memory().available


def say(s):
    print(s, flush=True)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.min_avail, self.peak_rss, self.n = avail(), rss(), 0

    def run(self):
        while not self.stop_flag.is_set():
            a = avail()
            self.min_avail = min(self.min_avail, a)
            self.peak_rss = max(self.peak_rss, rss())
            self.n += 1
            if a < ABORT_MB * MB:
                R["abort_reason"] = ("available system memory fell to %.0f MB, below the %d MB limit, during '%s'"
                                     % (a / MB, ABORT_MB, R["phase"]))
                R["verdict"] = "DID NOT COMPLETE - aborted for low memory."
                self.fill()
                write_report()
                os._exit(2)
            self.stop_flag.wait(SAMPLE_S)

    def fill(self):
        R.update(peak_rss=max(self.peak_rss, getattr(proc.memory_info(), "peak_wset", 0)),
                 min_avail=self.min_avail, n_samples=self.n)


def f(n):
    return format(int(n), ",")


def write_report():
    r = R
    L = []
    w = L.append
    w("PREDICTION DEDUPLICATION   (counting only - no predictive metric is computed, printed or saved)")
    w("=" * 112)
    if not r["completed"]:
        w("*** THE RUN DID NOT COMPLETE: %s ***" % r["abort_reason"])
        w("*** Everything below is what had been measured up to that point. ***")
        w("=" * 112)
    w("panel build %s; model %s; origin filter: %s" % (r.get("build_stamp", "?"), MODEL.replace(os.sep, "/"),
                                                      r.get("origin", "?")))
    w("")
    w("1. POPULATION   (condition rows, from_period == %d)" % CUT)
    if "n_loans" in r:
        w("  rows with role == condition and from_period == %d: %s" % (CUT, f(r["n_rows_all"])))
        w("  distinct loans among them:  %s   (one row per loan: %s)" % (f(r["n_loans_all"]),
                                                                        "yes" if r["one_per_loan"] else "NO"))
        w("  by from_state:              %s" % ", ".join("%s %s" % (k, f(v)) for k, v in r["by_state"].items()))
        w("  expected eligible loans:    %s;  panel gives %s  (%s%s)" % (
            f(EXPECTED), f(r["n_loans_all"]), "match" if r["n_loans_all"] == EXPECTED else "DIFFERENT",
            "" if r["n_loans_all"] == EXPECTED else ", difference %+d" % (r["n_loans_all"] - EXPECTED)))
        w("  loans used below:           %s" % f(r["n_loans"]))
        if "age_min" in r:
            w("  loan_age at the cut: min %d, max %d; loans with loan_age < 0: %s (kept; the pooled key is shifted)" % (
                r["age_min"], r["age_max"], f(r["n_neg_age"])))
        w("  est_ltv re-derived from the pinned hpi_ratio differs from the stored est_ltv on %s loans "
          "(float32, not NaN-aware ties counted equal)" % f(r["est_ltv_diff"]))
    else:
        w("  not reached")
    w("")
    w("2. DISTINCT FEATURE ROWS BY FORECAST MONTH   (features: %s)" % FEATURES)
    if "horizon_rows" in r:
        w("  %4s %12s %16s %12s" % ("h", "loans", "distinct rows", "reduction"))
        for h, n, d in r["horizon_rows"]:
            w("  %4d %12s %16s %11.2f%%" % (h, f(n), f(d), 100.0 * (1 - d / n)))
        w("  Advancing loan_age by the same h for every loan is a one-to-one map on rows, so the within-horizon count "
          "is the same at every h; the table measures that rather than assuming it.")
        w("")
        w("  Pooled across horizons (a row is the same row whichever loan and month asks for it):")
        for lab, n, d in r["pooled"]:
            w("    %-44s %14s evaluations -> %14s distinct rows   reduction %6.2f%%" % (lab, f(n), f(d),
                                                                                       100.0 * (1 - d / n)))
    else:
        w("  not reached")
    w("")
    w("3. DISTINCT PREDICTIONS   (probabilities rounded to %d decimals; counts only)" % ROUND)
    if "pred" in r:
        w("  The saved model reads run_length and run_length_censored; both were supplied as NaN for every row.")
        if "age_range" in r:
            w("  Model's training range of loan_age: %s. Rows at h=60 have loan_age up to %d." % (
                r["age_range"], r["pred"][-1]["max_age"]))
        for p in r["pred"]:
            fewer = p["inputs"] - p["vectors"]
            w("  h = %2d: distinct input rows %s -> distinct probability vectors %s   (%s by %s, %.2f%% of inputs)" % (
                p["h"], f(p["inputs"]), f(p["vectors"]),
                "fewer" if fewer > 0 else ("same" if fewer == 0 else "MORE"), f(abs(fewer)),
                100.0 * fewer / p["inputs"]))
        for p in r["pred"]:
            w("    h = %2d: loans %s -> distinct vectors %s  (reduction %.2f%% vs loans)" % (
                p["h"], f(r["n_loans"]), f(p["vectors"]), 100.0 * (1 - p["vectors"] / r["n_loans"])))
    else:
        w("  not reached")
    w("")
    w("4. DISTINCTNESS BY FEATURE   (h = 0 population)")
    if "feat" in r:
        w("  %-16s %14s %10s %24s %12s" % ("feature", "distinct vals", "nulls", "distinct rows w/o it", "rows lost"))
        full = r["feat_full"]
        for name, nv, nn, dd in r["feat"]:
            w("  %-16s %14s %10s %24s %12s%s" % (name, f(nv), f(nn), f(dd), f(full - dd),
                                                "   <- biggest" if name == r["feat_top"] else ""))
        w("  all 9 features: %s distinct rows. Single-feature drops understate features that are near-functions of "
          "others (est_ltv of ltv, loan_age of the rest)." % f(full))
        w("  'distinct vals' counts non-null values; a null is one more distinct value in the row counts.")
    else:
        w("  not reached")
    w("")
    w("5. MEMORY")
    if "peak_rss" in r:
        w("  peak RSS: %.1f MB (%.2f GB)   min system available: %.1f MB (%.2f GB)   (%d samples at %.1f s)" % (
            r["peak_rss"] / MB, r["peak_rss"] / GB, r["min_avail"] / MB, r["min_avail"] / GB, r["n_samples"], SAMPLE_S))
    w("")
    w("VERDICT: %s" % r.get("verdict", "no verdict"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    say("\n".join(L))


def n_distinct_rows(df):
    return len(df.drop_duplicates())                    # NaN compares equal to NaN


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origin", choices=["all", "current"], default="all")
    args = ap.parse_args()
    R["origin"] = ("all transient states at the cut" if args.origin == "all" else "from_state == Current only")

    import lightgbm as lgb
    manifest = panel_io.read_manifest()
    R["build_stamp"] = manifest["build_stamp"]
    quarters = [p["quarter"] for p in manifest["partitions"]]
    sampler = Sampler()
    sampler.start()

    # ---- population -------------------------------------------------------------------------------------------
    R["phase"] = "load"
    cols = ["loan_id", "from_period", "from_state"] + [c for c in FEATURES if c != "est_ltv"] + ["hpi_ratio", "est_ltv"]
    parts = []
    for q in quarters:
        d = panel_io.load_panel(roles=["condition"], columns=cols, quarters=[q])
        d = d[d["from_period"] == CUT]
        parts.append(d.reset_index(drop=True))
        say("  %s: %s rows at %d" % (q, f(len(d)), CUT))
        del d
    pop = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    R["n_rows_all"] = len(pop)
    R["n_loans_all"] = pop["loan_id"].nunique()
    R["one_per_loan"] = R["n_rows_all"] == R["n_loans_all"]
    R["by_state"] = {str(k): int(v) for k, v in pop["from_state"].astype(str).value_counts().items()}
    if not R["one_per_loan"]:
        R.update(n_loans=R["n_loans_all"], est_ltv_diff=0)
        R["abort_reason"] = "more than one condition row per loan at %d" % CUT
        R["verdict"] = "DID NOT RUN."
        sampler.stop_flag.set()
        sampler.fill()
        write_report()
        raise SystemExit(R["abort_reason"])
    if args.origin == "current":
        pop = pop[pop["from_state"] == "Current"].reset_index(drop=True)

    # est_ltv exactly as src/features_hpi.py: float64 ltv / float64 hpi_ratio, stored as float32
    ltv64 = pop["ltv"].to_numpy(dtype=np.float64, na_value=np.nan)
    ratio64 = pop["hpi_ratio"].to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = (ltv64 / ratio64).astype(np.float32)
    stored = pop["est_ltv"].to_numpy(dtype=np.float32, na_value=np.nan)
    R["est_ltv_diff"] = int((~((est == stored) | (np.isnan(est) & np.isnan(stored)))).sum())
    pop["est_ltv"] = est
    X = pop[FEATURES].copy()
    for c in FEATURES:
        if c not in (CAT, "loan_age"):
            X[c] = X[c].astype(np.float32)
    X["loan_age"] = X["loan_age"].astype(np.int32)
    X[CAT] = X[CAT].astype("category")
    del pop, ltv64, ratio64, est, stored
    gc.collect()
    R["n_loans"] = len(X)
    say("loans used: %s (all-state population %s)" % (f(R["n_loans"]), f(R["n_loans_all"])))

    # ---- distinct rows by horizon ---------------------------------------------------------------------------
    R["phase"] = "distinct rows"

    def at(h):
        d = X.copy()
        d["loan_age"] = d["loan_age"] + np.int32(h)
        return d

    R["horizon_rows"] = []
    for h in HORIZONS:
        d = at(h)
        R["horizon_rows"].append((h, len(d), n_distinct_rows(d)))
        del d
    say("per-horizon distinct rows: %s" % [(h, dd) for h, _, dd in R["horizon_rows"]])

    # pooled: key = (profile of the 8 non-age features, age); a row is the same row whoever and whenever it is asked for
    pid = X.groupby(NON_AGE, dropna=False, observed=True, sort=False).ngroup().to_numpy(dtype=np.int64)
    age0 = X["loan_age"].to_numpy(dtype=np.int64)
    R["age_min"], R["age_max"], R["n_neg_age"] = int(age0.min()), int(age0.max()), int((age0 < 0).sum())
    shift = -R["age_min"] if R["age_min"] < 0 else 0         # loan_age is negative on some rows: shift, do not drop
    width = R["age_max"] + shift + max(GRID) + 1
    age0 = age0 + shift

    def pooled(horizons):
        keys = [np.unique(pid * width + age0 + h) for h in horizons]
        return len(np.unique(np.concatenate(keys)))

    R["pooled"] = [("the 8 reported horizons %s" % HORIZONS, len(X) * len(HORIZONS), pooled(HORIZONS)),
                   ("every month h = 0..60 (the forecast grid)", len(X) * len(GRID), pooled(GRID))]
    R["profiles"] = int(pid.max()) + 1
    say("pooled: %s" % [(lab, d) for lab, _, d in R["pooled"]])
    del pid, age0
    gc.collect()

    # ---- distinct predictions at h = 0 and 60 ---------------------------------------------------------------
    R["phase"] = "scoring"
    booster = lgb.Booster(model_file=MODEL)
    names = booster.feature_name()
    unknown = [n for n in names if n not in FEATURES and n not in ("run_length", "run_length_censored")]
    if unknown:
        raise RuntimeError("model reads feature(s) this diagnostic cannot supply: %s" % unknown)
    head = open(MODEL, encoding="utf-8").read(20000)
    fn = re.search(r"^feature_names=(.*)$", head, re.M).group(1).split()
    fi = re.search(r"^feature_infos=(.*)$", head, re.M).group(1).split()
    if "loan_age" in fn:
        R["age_range"] = fi[fn.index("loan_age")]
    R["pred"] = []
    for h in (0, 60):
        d = at(h).drop_duplicates().reset_index(drop=True)
        n_in = len(d)
        out = []
        for lo in range(0, n_in, SCORE_CHUNK):
            c = d.iloc[lo:lo + SCORE_CHUNK]
            m = pd.DataFrame({n: (c[n].to_numpy() if n in FEATURES else np.full(len(c), np.nan, np.float32))
                              for n in names})
            m[CAT] = pd.Categorical(c[CAT].astype(object), categories=c[CAT].cat.categories)
            p = np.round(booster.predict(m), ROUND)
            out.append(np.ascontiguousarray(p))
            del m, p
        P = np.concatenate(out)
        del out
        vec = len(np.unique(P + 0.0, axis=0))                # + 0.0 turns -0.0 into 0.0
        R["pred"].append({"h": h, "inputs": n_in, "vectors": vec, "max_age": int(d["loan_age"].max())})
        say("h=%d: %s distinct inputs -> %s distinct prediction vectors" % (h, f(n_in), f(vec)))
        del d, P
        gc.collect()

    # ---- distinctness by feature (h = 0) --------------------------------------------------------------------
    R["phase"] = "per-feature"
    R["feat_full"] = n_distinct_rows(X)
    feat = []
    for c in FEATURES:
        nv = X[c].nunique(dropna=True)
        nn = int(X[c].isna().sum())
        dd = n_distinct_rows(X.drop(columns=c))
        feat.append((c, int(nv), nn, dd))
    R["feat"] = feat
    R["feat_top"] = min(feat, key=lambda t: t[3])[0]

    sampler.stop_flag.set()
    sampler.join()
    sampler.fill()
    R["completed"] = True
    _, n0, d0 = R["horizon_rows"][0]
    red0 = 100.0 * (1 - d0 / n0)
    lab, ng, dg = R["pooled"][1]
    redg = 100.0 * (1 - dg / ng)
    p0, p60 = R["pred"][0], R["pred"][1]
    worth = redg >= WORTH or red0 >= WORTH
    R["verdict"] = ("%s: within one month %s loans need %s distinct rows (%.1f%% fewer); over the full 61-month grid "
                    "%s evaluations need %s (%.1f%% fewer); distinct predictions at h=0 / h=60 are %s / %s. "
                    "(judgement threshold: %.0f%% reduction)" % (
                        "WORTH BUILDING" if worth else "NOT WORTH BUILDING", f(n0), f(d0), red0, f(ng), f(dg), redg,
                        f(p0["vectors"]), f(p60["vectors"]), WORTH))
    write_report()


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        R["abort_reason"] = "MemoryError during '%s'" % R["phase"]
        R["verdict"] = "DID NOT COMPLETE - MemoryError."
        write_report()
        raise
