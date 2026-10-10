"""The ML arm's forecaster: a one-month multiclass model -> a 60-month state-probability forecast per loan.

    python -m src.forecast --run pooled_v1 --model artifacts/ml/pooled --canary
    python -m src.forecast --run pooled_v1 --model artifacts/ml/pooled
    python -m src.forecast --run grouped_v1 --model-map Current=artifacts/ml/a DQ30=artifacts/ml/b DQ60=... DQ90+=...

PROPAGATION, NOT SIMULATION. Each loan carries a six-element probability vector over
{Current, DQ30, DQ60, DQ90+, Prepaid, Default}. h = 0 is the initial vector: probability 1 on the from_state of the
loan's condition row at from_period == 2008-12 (one row per loan). Each month, for every loan, one feature row is built
per TRANSIENT origin state (4 rows) and scored; the four probability rows are that loan's transition rows, and

    new[j] = sum over transient o of  old[o] * P_o[j]   +   old[j] if j is Prepaid or Default

Prepaid and Default are absorbing: mass that enters them stays and they are never used as an origin. The model that scores
origin state o is models[o]; one pooled model maps all four states to itself (and its four rows are scored in one call),
the grouped architecture passes three models. Nothing assumes one model.

FEATURES FOR FUTURE MONTHS are built here, never read from the panel:
    fico, ltv, dti, rate_spread, property_state   fixed per loan (their 2008-12 values)
    loan_age        NOT built unless the model reads it (the default models do not: --with-loan-age is the ablation
                    arm). When a model does read it: age at 2008-12 plus the number of months advanced, then CLAMPED to
                    the model's own sidecar "clamp_bounds" (never hardcoded), so no age outside what the model was
                    trained on is scored; loan-months clamped at the top and at the bottom are counted separately.
                    The clamp machinery is generic over the sidecar's clamp_bounds (it will carry run_length for the
                    semi-Markov ablation) and is INERT when no model reads a clamped feature; the report then says so
                    instead of printing zero counts. (loan_age at 2008-12 is still read from the panel, but only to
                    get the origination month for est_ltv.)
    unemp_level, unemp_chg_12m   frozen at their 2008-12 values
    est_ltv         constant per loan: the hpi_ratio is pinned at its 2008-12 value and est_ltv is derived by
                    src.panel.hpi_features - the writer of the panel column, the arithmetic src/features_hpi.py verifies -
                    from the UNROUNDED float64 ratio, not from the stored float32 hpi_ratio. Each derived value is
                    asserted equal to the stored est_ltv on the condition row.
    from_state      (pooled models only) the origin state the row stands for

OUTPUT, long format, data/forecast/<run>/batch=NNNNN/part.parquet: loan_id, h, state, probability (float32), one row per
loan per horizon per state. Propagation and the checks run in float64; the cast to float32 happens only at the write, and
the sum-to-1 deviation is reported both as computed in float64 and as it stands in the stored float32 values. Loans are processed in batches (--batch-size, default 50,000, loans sorted by loan_id); each
batch runs to h = 60, is written, and is then marked done (batch=NNNNN/_DONE.json), so a rerun skips it. --canary writes
the first batch and stops. run.json pins the models, batch size and panel build; a rerun that disagrees with it is refused.

CHECKS, raised not just reported (a failing batch is not written or marked done):
    every loan's vector sums to 1 at every h within SUM_TOL;  no probability is negative or not finite;
    Prepaid mass and Default mass are each non-decreasing in h for every loan.
The worst deviation of each is reported to diagnostics/37_forecaster_build.txt. The stored float32 sum deviation is
reported against STORED_TOL; if it is larger the report says so plainly (the tolerance is not loosened).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import panel, panel_io
from src.memwatch import MemWatch

STATES = list(panel.STATES)
TRANSIENT = STATES[:4]
N_STATES = len(STATES)
HORIZON = 60
CUT_YYYYMM = panel.CUT_YEAR * 100 + panel.CUT_MONTH
REPORT_H = (0, 6, 12, 24, 60)
SUM_TOL = 1e-9                       # float64, in memory
STORED_TOL = 1e-5                    # float32, as stored: reported against, never loosened
EXPECT_LOANS = 405672
CLAMPED = ("loan_age",)              # features the forecaster advances month by month and clamps (run_length joins later)
OUT_ROOT = os.path.join("data", "forecast")
REPORT = os.path.join("diagnostics", "37_forecaster_build.txt")
GB = 2 ** 30
MB = 2 ** 20


class ForecastError(RuntimeError):
    pass


def say(s=""):
    print(s, flush=True)


def f(n):
    return format(int(n), ",")


# ==========================================================================
# models
# ==========================================================================
class Model:
    """A saved one-month model (artifacts/ml/<name>/): the booster plus what its sidecar says about it."""

    def __init__(self, directory):
        import lightgbm as lgb
        self.directory = directory
        with open(os.path.join(directory, "model.json"), encoding="utf-8") as fh:
            self.side = json.load(fh)
        path = os.path.join(directory, "model.txt")
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()
        self.booster = lgb.Booster(model_file=path)
        s = self.side
        self.name, self.features, self.categories = s["name"], list(s["features"]), s["categories"]
        self.origin_states = list(s["origin_states"])
        # clamp bounds: feature -> (min, max), for the features the model reads that the forecaster must keep inside the
        # trained range. A sidecar written before clamp_bounds existed carried them as max/min_loan_age (loan_age was
        # then always a feature).
        if "clamp_bounds" in s:
            self.clamp = {c: (int(b[0]), int(b[1])) for c, b in s["clamp_bounds"].items()}
        elif "loan_age" in self.features:
            self.clamp = {"loan_age": (int(s["min_loan_age"]), int(s["max_loan_age"]))}
        else:
            self.clamp = {}
        stray = [c for c in self.clamp if c not in self.features]
        missing = [c for c in CLAMPED if c in self.features and c not in self.clamp]
        if stray or missing:
            raise ForecastError("%s: clamp bounds for non-features %s / no clamp bound for features %s"
                                % (directory, stray, missing))
        if s["class_order"] != STATES:
            raise ForecastError("%s: class order %s != %s" % (directory, s["class_order"], STATES))
        if self.booster.feature_name() != self.features:
            raise ForecastError("%s: model features %s != sidecar %s" % (directory, self.booster.feature_name(),
                                                                       self.features))
        if self.booster.num_model_per_iteration() != N_STATES:
            raise ForecastError("%s: not a six-class model" % directory)
        bad = [c for c in ("run_length", "run_length_censored") if c in self.features]
        if bad:
            raise ForecastError("%s reads %s, which the headline arm excludes" % (directory, bad))
        self.state_codes = None                      # property_state codes for every loan, set by bind()

    def bind(self, property_state):
        cats = self.categories["property_state"]
        codes = pd.Categorical(property_state, categories=cats).codes
        if (codes < 0).any():
            raise ForecastError("%s: %d loans have a property_state outside the model's categories"
                                % (self.directory, int((codes < 0).sum())))
        self.state_codes = codes


def load_models(args):
    if args.model:
        m = Model(args.model)
        models = {o: m for o in TRANSIENT}
    else:
        paths = {}
        for item in args.model_map:
            state, _, path = item.partition("=")
            if state not in TRANSIENT or not path:
                raise SystemExit("--model-map entries are STATE=DIR with STATE in %s, got %r" % (TRANSIENT, item))
            if state in paths:
                raise SystemExit("--model-map names %s twice" % state)
            paths[state] = path
        missing = [o for o in TRANSIENT if o not in paths]
        if missing:
            raise SystemExit("--model-map does not cover %s" % missing)
        loaded = {p: Model(p) for p in set(paths.values())}       # one object per directory: shared models score together
        models = {o: loaded[paths[o]] for o in TRANSIENT}
    for o in TRANSIENT:
        if o not in models[o].origin_states:
            raise ForecastError("origin state %s is mapped to %s, which was trained on %s only"
                                % (o, models[o].directory, models[o].origin_states))
    return models


def unique_models(models):
    seen, out = set(), []
    for o in TRANSIENT:
        if id(models[o]) not in seen:
            seen.add(id(models[o]))
            out.append(models[o])
    return out


# ==========================================================================
# the loans at the cut
# ==========================================================================
def load_initial(expect_loans):
    """One row per loan from the condition rows at from_period == 2008-12, with est_ltv derived from the unrounded ratio."""
    hpi = panel.Hpi()
    manifest = panel_io.read_manifest()
    cols = ["loan_id", "from_period", "from_state", "fico", "ltv", "dti", "rate_spread", "property_state", "loan_age",
            "est_ltv", "unemp_level", "unemp_chg_12m"]
    parts = []
    for p in manifest["partitions"]:
        q = p["quarter"]
        d = panel_io.load_panel(roles=["condition"], columns=cols, quarters=[q])
        d = d[d["from_period"] == CUT_YYYYMM].reset_index(drop=True)
        if not len(d):
            continue
        ps = d["property_state"]
        codes = ps.cat.codes.to_numpy().astype(np.int64)
        if (codes < 0).any():
            raise ForecastError("%s: null property_state at the cut" % q)
        age = d["loan_age"].to_numpy(dtype=np.int64)
        from_month = np.full(len(d), panel.CUT, dtype=np.int64)
        ltv = d["ltv"].to_numpy(dtype=np.float32, na_value=np.nan)
        hf = panel.hpi_features(hpi, q, "", np.zeros(len(d), np.int64), codes, [str(c) for c in ps.cat.categories],
                                from_month, from_month - age, ltv)          # tails/prefix only name loans in errors
        stored = d["est_ltv"].to_numpy(dtype=np.float32, na_value=np.nan)
        n_bad = int((hf["est_ltv"] != stored).sum())
        if n_bad:
            raise ForecastError("%s: est_ltv derived from the unrounded ratio differs from the stored column on %d "
                                "of %d loans" % (q, n_bad, len(d)))
        parts.append(pd.DataFrame({
            "loan_id": d["loan_id"].astype(str).to_numpy(),
            "from_state": d["from_state"].astype(str).to_numpy(),
            "fico": d["fico"].to_numpy(dtype=np.float32, na_value=np.nan), "ltv": ltv,
            "dti": d["dti"].to_numpy(dtype=np.float32, na_value=np.nan),
            "rate_spread": d["rate_spread"].to_numpy(dtype=np.float32, na_value=np.nan),
            "property_state": ps.astype(str).to_numpy(), "loan_age": age, "est_ltv": hf["est_ltv"],
            "unemp_level": d["unemp_level"].to_numpy(dtype=np.float32, na_value=np.nan),
            "unemp_chg_12m": d["unemp_chg_12m"].to_numpy(dtype=np.float32, na_value=np.nan)}))
        say("  initial %s: %s loans" % (q, f(len(d))))
    init = pd.concat(parts, ignore_index=True).sort_values("loan_id", kind="stable").reset_index(drop=True)
    if init["loan_id"].duplicated().any():
        raise ForecastError("%d loans have more than one condition row at %d"
                            % (int(init["loan_id"].duplicated().sum()), CUT_YYYYMM))
    if len(init) != expect_loans:
        raise ForecastError("%s loans at %d, expected %s (--expect-loans)" % (f(len(init)), CUT_YYYYMM, f(expect_loans)))
    bad = sorted(set(init["from_state"]) - set(TRANSIENT))
    if bad:
        raise ForecastError("condition rows start in non-transient state(s) %s" % bad)
    frozen = {}
    for c in ("unemp_level", "unemp_chg_12m"):
        vals = init[c].unique()
        if len(vals) != 1 or not np.isfinite(vals[0]):
            raise ForecastError("%s at %d is not a single finite value: %s" % (c, CUT_YYYYMM, vals[:5]))
        frozen[c] = np.float32(vals[0])
    return init, frozen, manifest["build_stamp"]


# ==========================================================================
# propagation
# ==========================================================================
def build_frame(m, origins, sl, init, frozen):
    """The scoring frame for model m: one block of rows per origin state in `origins`, columns in the model's order."""
    n, k = sl.stop - sl.start, len(origins)
    cols = {}
    for c in m.features:
        if c == "property_state":
            cols[c] = pd.Categorical.from_codes(np.tile(m.state_codes[sl], k), categories=m.categories[c])
        elif c == "from_state":
            cats = m.categories[c]
            cols[c] = pd.Categorical.from_codes(np.repeat([cats.index(STATES[o]) for o in origins], n), categories=cats)
        elif c in CLAMPED:                           # advanced and clamped every month by propagate()
            cols[c] = np.zeros(n * k, dtype=np.float32)
        elif c in frozen:
            cols[c] = np.full(n * k, frozen[c], dtype=np.float32)
        else:
            cols[c] = np.tile(init[c].to_numpy(dtype=np.float32)[sl], k)
    return pd.DataFrame(cols)


def propagate(models, init, frozen, sl):
    """(n, HORIZON+1, 6) float64 state probabilities for the loans in slice sl, and the clamp counts."""
    n = sl.stop - sl.start
    groups = []
    for m in unique_models(models):
        origins = [i for i, o in enumerate(TRANSIENT) if models[o] is m]
        groups.append((m, origins, build_frame(m, origins, sl, init, frozen)))
    age0 = init["loan_age"].to_numpy(dtype=np.int64)[sl]
    clamp_active = any(c in m.features for m in unique_models(models) for c in CLAMPED)
    v = np.zeros((n, N_STATES))
    v[np.arange(n), [STATES.index(s) for s in init["from_state"].to_numpy()[sl]]] = 1.0
    P = np.empty((n, HORIZON + 1, N_STATES))
    P[:, 0] = v
    clamp_top = clamp_bottom = 0
    for s in range(HORIZON):
        dynamic = {"loan_age": age0 + s}                  # per-loan values this month, advancing by one ...
        trans = np.empty((n, len(TRANSIENT), N_STATES))
        top_any, bottom_any = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        for m, origins, frame in groups:
            for c in CLAMPED:                             # ... and clipped into the model's own trained range
                if c in m.features:                       # nothing happens for a model that does not read c
                    lo, hi = m.clamp[c]
                    top_any |= dynamic[c] > hi
                    bottom_any |= dynamic[c] < lo
                    frame[c] = np.tile(np.clip(dynamic[c], lo, hi).astype(np.float32), len(origins))
            out = m.booster.predict(frame)
            for k, o in enumerate(origins):
                trans[:, o, :] = out[k * n:(k + 1) * n]
        clamp_top += int(top_any.sum())
        clamp_bottom += int(bottom_any.sum())
        new = np.einsum("ni,nij->nj", v[:, :len(TRANSIENT)], trans)
        new[:, len(TRANSIENT):] += v[:, len(TRANSIENT):]    # absorbing mass stays
        v = new
        P[:, s + 1] = v
    return P, {"clamp_active": clamp_active, "clamp_top_loan_months": clamp_top,
               "clamp_bottom_loan_months": clamp_bottom}


def check(P, label):
    """Raise unless every vector sums to 1, nothing is negative or non-finite, and absorbing mass never falls."""
    if not np.isfinite(P).all():
        raise ForecastError("%s: non-finite probabilities" % label)
    dev = float(np.abs(P.sum(axis=2) - 1.0).max())
    lo = float(P.min())
    d = np.diff(P[:, :, len(TRANSIENT):], axis=1)
    mono = float(d.min())                                  # most negative one-month change; >= 0 means never falls
    mono_pd = float(np.diff(P[:, :, 4] + P[:, :, 5], axis=1).min())
    if dev > SUM_TOL:
        raise ForecastError("%s: a vector sums to 1 only within %.3e (tolerance %.0e)" % (label, dev, SUM_TOL))
    if lo < 0:
        raise ForecastError("%s: negative probability %.3e" % (label, lo))
    if mono < 0 or mono_pd < 0:
        raise ForecastError("%s: Prepaid/Default mass falls by %.3e" % (label, min(mono, mono_pd)))
    return {"worst_sum_deviation": dev, "min_probability": lo, "min_absorbing_step": mono,
            "min_prepaid_plus_default_step": mono_pd}


def stored_stats(P32):
    """The sum-to-1 deviation and smallest probability as they stand in the float32 values that are written."""
    return {"worst_sum_deviation_f32": float(np.abs(P32.astype(np.float64).sum(axis=2) - 1.0).max()),
            "min_probability_f32": float(P32.min())}


def write_batch(path, ids, P32):
    n, H = P32.shape[0], P32.shape[1]
    table = pa.table({
        "loan_id": pa.DictionaryArray.from_arrays(np.repeat(np.arange(n, dtype=np.int32), H * N_STATES), pa.array(ids)),
        "h": np.tile(np.repeat(np.arange(H, dtype=np.int8), N_STATES), n),
        "state": pa.DictionaryArray.from_arrays(np.tile(np.arange(N_STATES, dtype=np.int8), n * H), pa.array(STATES)),
        "probability": P32.reshape(-1)})
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    if pq.ParquetFile(tmp).metadata.num_rows != n * H * N_STATES:
        raise ForecastError("%s: wrote the wrong number of rows" % path)
    os.replace(tmp, path)


# ==========================================================================
# the run
# ==========================================================================
def run_config(models, args, n_loans, build_stamp):
    return {"models": {o: {"directory": models[o].directory.replace(os.sep, "/"), "sha256": models[o].sha256}
                       for o in TRANSIENT},
            "probability_dtype": "float32",
            "clamp_bounds": {o: {c: list(b) for c, b in models[o].clamp.items()} for o in TRANSIENT},
            "batch_size": args.batch_size, "horizon": HORIZON, "n_loans": n_loans, "cut": CUT_YYYYMM,
            "panel_build_stamp": build_stamp}


def dir_size(root):
    return sum(os.path.getsize(os.path.join(a, b)) for a, _, fs in os.walk(root) for b in fs)


def write_report(lines):
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    say("\n".join(lines))


def build_report(run, run_dir, models, cfg, args, n_batches, markers, wall, watch, status, this_run):
    L = []
    w = L.append
    w("FORECASTER BUILD   (propagation of a one-month multiclass model over %d months; no predictive metric)" % HORIZON)
    w("=" * 112)
    if status:
        w("*** %s ***" % status)
        w("=" * 112)
    w("run: %s    output: %s" % (run, run_dir.replace(os.sep, "/")))
    w("panel build %s; cut %d; horizon h = 0..%d inclusive (h = 0 is the initial vector)" % (
        cfg["panel_build_stamp"], CUT_YYYYMM, HORIZON))
    w("")
    w("MODELS   (origin state -> model)")
    for m in unique_models(models):
        w("  %s -> %s   sha256 %s" % ("/".join(o for o in TRANSIENT if models[o] is m), m.directory.replace(os.sep, "/"),
                                      m.sha256[:16]))
        w("      trained on origin states %s, %s rows, window %s, features %s" % (
            m.origin_states, f(m.side["n_rows"]), m.side["training_window"], m.features))
        if m.clamp:
            w("      clamped features, bounds from the sidecar's clamp_bounds: %s" % {c: list(b) for c, b in m.clamp.items()})
        else:
            w("      clamped features: none (the model reads no feature the forecaster advances; loan_age is not a "
              "feature)")
        prov = m.side.get("loan_age_in_training_rows")
        if prov:
            w("      loan_age in the training rows (provenance only, not a bound): %s .. %s" % (prov["min"], prov["max"]))
    w("")
    done = sorted(markers)
    loans_done = sum(markers[b]["loans"] for b in done)
    w("SCOPE")
    w("  loans in the forecast population: %s;  batch size %s;  batches done %d of %d%s" % (
        f(cfg["n_loans"]), f(args.batch_size), len(done), n_batches,
        "   (CANARY: the first batch only)" if len(done) < n_batches else "   (complete)"))
    w("  loans in the batches done: %s;  batch indices run in this invocation (0-based): %s" % (f(loans_done), this_run or "none"))
    w("")
    w("CLAMP   (a feature the forecaster advances monthly is clipped to the model's trained [min, max])")
    if not any(markers[b]["clamp_active"] for b in done):
        w("  INERT: no model in this run reads a clamped feature (%s), so nothing was advanced or clamped." % ", ".join(
            CLAMPED))
        w("  The machinery is in place for the models that do (loan_age with --with-loan-age; run_length later); "
          "it has no counts to report here, and zero would be misleading.")
    else:
        loan_steps = loans_done * HORIZON
        top = sum(markers[b]["clamp_top_loan_months"] for b in done)
        bottom = sum(markers[b]["clamp_bottom_loan_months"] for b in done)
        w("  clamped at the TOP    (value above the training maximum): %s of %s loan-months (%.4f%%)" % (
            f(top), f(loan_steps), 100.0 * top / max(loan_steps, 1)))
        w("  clamped at the BOTTOM (value below the training minimum): %s of %s loan-months (%.4f%%)" % (
            f(bottom), f(loan_steps), 100.0 * bottom / max(loan_steps, 1)))
        w("  a loan-month counts once however many clamped features or models clamp it; the two counts are separate "
          "limitations.")
    w("")
    w("CHECKS   (asserted per batch; a failure stops the run before the batch is written)")
    if done:
        w("  every loan's vector sums to 1 at every h:   PASS   worst |sum - 1| = %.3e in float64 as computed "
          "(tolerance %.0e)" % (max(markers[b]["checks"]["worst_sum_deviation"] for b in done), SUM_TOL))
        dev32 = max(markers[b]["checks"]["worst_sum_deviation_f32"] for b in done)
        w("      as stored (float32 values, summed in float64):    worst |sum - 1| = %.3e   %s" % (
            dev32, "within %.0e" % STORED_TOL if dev32 <= STORED_TOL else
            "*** EXCEEDS %.0e: the stored float32 data does not sum to 1 within that tolerance ***" % STORED_TOL))
        w("  no negative probability:                    PASS   smallest probability = %.3e (float64), %.3e as stored" % (
            min(markers[b]["checks"]["min_probability"] for b in done),
            min(markers[b]["checks"]["min_probability_f32"] for b in done)))
        w("  Prepaid, Default mass non-decreasing in h:  PASS   smallest one-month change = %.3e (Prepaid+Default %.3e)" % (
            min(markers[b]["checks"]["min_absorbing_step"] for b in done),
            min(markers[b]["checks"]["min_prepaid_plus_default_step"] for b in done)))
    else:
        w("  no batch completed")
    w("")
    w("RESOURCES   (this invocation)")
    w("  wall clock %.1f s (%.1f min);  peak RSS %.2f GB;  min system available %.2f GB" % (
        wall, wall / 60, watch.peak_rss() / GB, watch.min_avail / GB))
    if done:
        w("  seconds per batch (all done batches): %s" % ", ".join("%.0f" % markers[b]["seconds"] for b in done))
    size = dir_size(run_dir) if os.path.isdir(run_dir) else 0
    n_files = sum(len(fs) for _, _, fs in os.walk(run_dir)) if os.path.isdir(run_dir) else 0
    w("  output: %s, %s files, %.1f MB (%.2f GB); rows written %s" % (
        run_dir.replace(os.sep, "/"), f(n_files), size / MB, size / GB,
        f(sum(markers[b]["rows"] for b in done))))
    w("")
    w("PORTFOLIO STATE DISTRIBUTION   (mean probability over the %s loans in the batches done; a sanity read)" % f(loans_done))
    if done:
        tot = np.sum([np.array(markers[b]["state_sums"]) for b in done], axis=0) / loans_done
        w("  %4s %s" % ("h", " ".join("%10s" % s for s in STATES)))
        for h in REPORT_H:
            w("  %4d %s   (sum %.9f)" % (h, " ".join("%10.6f" % x for x in tot[h]), tot[h].sum()))
    w("")
    over = done and max(markers[b]["checks"]["worst_sum_deviation_f32"] for b in done) > STORED_TOL
    w("VERDICT: %s%s" % ("STORED float32 SUMS EXCEED %.0e. " % STORED_TOL if over else "", "DID NOT COMPLETE." if status else (
        "all %d batches written and checked." % n_batches if len(done) == n_batches
        else "canary batch written and all three checks passed; %d of %d batches remain." % (n_batches - len(done), n_batches))))
    write_report(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="output name: data/forecast/<run>/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", metavar="DIR", help="one pooled model (artifacts/ml/<name>) for all four origin states")
    g.add_argument("--model-map", nargs="+", metavar="STATE=DIR", help="a model per origin state")
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--canary", action="store_true", help="write the first batch and stop")
    ap.add_argument("--expect-loans", type=int, default=EXPECT_LOANS)
    args = ap.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    models = load_models(args)
    run_dir = os.path.join(OUT_ROOT, args.run)
    markers, state = {}, {"wall0": time.perf_counter(), "this_run": [], "cfg": None, "n_batches": 0}

    def on_abort(msg):
        if state["cfg"] is not None:
            build_report(args.run, run_dir, models, state["cfg"], args, state["n_batches"], markers,
                         time.perf_counter() - state["wall0"], watch, "THE RUN DID NOT COMPLETE: " + msg,
                         ",".join(map(str, state["this_run"])))
    watch = MemWatch(on_abort=on_abort)
    watch.start()

    say("loading the loans at the cut")
    init, frozen, build_stamp = load_initial(args.expect_loans)
    n = len(init)
    for m in unique_models(models):
        m.bind(init["property_state"].to_numpy())
    cfg = run_config(models, args, n, build_stamp)
    state["cfg"] = cfg
    say("loans: %s;  frozen unemp_level %.4f, unemp_chg_12m %.4f" % (f(n), frozen["unemp_level"], frozen["unemp_chg_12m"]))

    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "run.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            if json.load(fh) != cfg:
                raise SystemExit("%s disagrees with this invocation (models, batch size, loan count or panel build); "
                                 "use a new --run" % cfg_path)
    else:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")

    starts = list(range(0, n, args.batch_size))
    state["n_batches"] = len(starts)
    for b, lo in enumerate(starts):
        bdir = os.path.join(run_dir, "batch=%05d" % b)
        mark = os.path.join(bdir, "_DONE.json")
        if os.path.exists(mark):
            with open(mark, encoding="utf-8") as fh:
                markers[b] = json.load(fh)
            say("batch %d/%d: done already, skipped" % (b + 1, len(starts)))
        else:
            if args.canary and b > 0:
                break
            sl = slice(lo, min(lo + args.batch_size, n))
            watch.phase = "batch %d" % b
            t0 = time.perf_counter()
            P, st = propagate(models, init, frozen, sl)
            checks = check(P, "batch %d" % b)
            P32 = P.astype(np.float32)                                    # the only cast: at the write
            checks.update(stored_stats(P32))
            os.makedirs(bdir, exist_ok=True)
            write_batch(os.path.join(bdir, "part.parquet"), init["loan_id"].to_numpy()[sl], P32)
            rec = {"batch": b, "loans": sl.stop - sl.start, "first_loan_id": init["loan_id"].iloc[sl.start],
                   "last_loan_id": init["loan_id"].iloc[sl.stop - 1], "rows": P.shape[0] * P.shape[1] * N_STATES,
                   "clamp_active": st["clamp_active"], "clamp_top_loan_months": st["clamp_top_loan_months"],
                   "clamp_bottom_loan_months": st["clamp_bottom_loan_months"], "checks": checks,
                   "state_sums": P.sum(axis=0).tolist(), "seconds": time.perf_counter() - t0}
            with open(mark, "w", encoding="utf-8") as fh:                  # last: its presence means the batch is whole
                json.dump(rec, fh)
            markers[b] = rec
            state["this_run"].append(b)
            del P, P32
            clamp_txt = ("clamped loan-months top %s bottom %s" % (f(st["clamp_top_loan_months"]),
                                                                  f(st["clamp_bottom_loan_months"]))
                         if st["clamp_active"] else "clamp inert (no clamped feature)")
            say("batch %d/%d: %s loans, %.0f s, %s, worst |sum-1| f64 %.2e f32 %.2e" % (
                b + 1, len(starts), f(rec["loans"]), rec["seconds"], clamp_txt, checks["worst_sum_deviation"],
                checks["worst_sum_deviation_f32"]))
        if args.canary and b == 0:
            break
    watch.stop()
    build_report(args.run, run_dir, models, cfg, args, len(starts), markers, time.perf_counter() - state["wall0"], watch,
                 None, ",".join(map(str, state["this_run"])))


if __name__ == "__main__":
    sys.exit(main())
