"""MILESTONE: ARCHITECTURE COMPARISON, grouped (three models) vs pooled (one model), chained six months from 2008-06, baseline alongside.

    python diagnostics/53_architecture_h6.py   # from the repo root; DIAG53_REUSE=1 skips retraining models that already exist

Everything below is a closed decision of the brief; nothing is tuned and no arm is added.

WINDOW. Models train on role == 'fit', from_period <= 200806 (src.train_ml --from-period-max 200806; train_ml has no lower bound, so the observed lower edge is
reported, not enforced). The forecast starts at the 2008-06 cut and is chained six months to 2008-12; it is checked against the realised states at 2008-12, read from
fit rows. EVALUATION LOANS (answer given to the question asked before this run): every fit loan with a fit row at from_period == 200806 (all such rows start in a
transient state, since an absorbing state emits no row). They are loans the models also trained on up to 200806: the 2008-12 outcome is out-of-time, the loans are
not held out. Evaluate rows are never read by this script or by anything it launches (train_ml, forecast and horizon read roles=['fit']; baseline runs with
--fit-only).
FEATURES. fico ltv dti rate_spread property_state est_ltv unemp_level unemp_chg_12m for both arms; from_state as a native categorical wherever a model trains on
more than one origin state (the pooled model and the DQ30+DQ60 model), as src.train_ml resolves it. No loan_age, no run_length.
REGULARISATION. min_sum_hessian_in_leaf 10, lambda_l2 10 on every model; num_leaves 31, min_data_in_leaf 20, 100 rounds (train_ml's own).
ARMS. GROUPED: Current / DQ30+DQ60 / DQ90+, forecast through src.forecast --model-map. POOLED: one model, src.forecast --model. BASELINE: the homogeneous matrix
re-estimated by src.baseline on fit rows with from_period <= 200806, chained through src.horizon.chain / homogeneous_step. The baseline is context only and never
enters the rule.

THE CODE THIS RUN NEEDED CHANGED (done first, as a separate backward-compatible change; every default reproduces; see artifacts/arch6/src_changes.diff):
  src/train_ml.py   --min-sum-hessian-in-leaf, --lambda-l2 (unset = not passed); recorded in the sidecar as lgb_params.
  src/forecast.py   --cut, --cut-role, --horizon, --out-root, --report (defaults: 200812, condition, 60, data/forecast, diagnostics/37_forecaster_build.txt).
  src/horizon.py    read_test_loans(roles=, cut_role=, period_min=); _per_loan(cut_role=).
  src/baseline.py   --from-period-min, --from-period-max, --fit-only.

PRE-REGISTERED RULE (fixed before any number was read; applied mechanically).
  QUANTITY: for each of two columns (Default share, DQ90+ share), the absolute error in percentage points between an arm's predicted share of the book at h = 6
  and the realised share at 2008-12. The cohort is the evaluation loans whose state at 2008-12 is OBSERVED (the harness's "varying" cohort: an absorbing state
  reached by then, or a state seen in 2008-12); the predicted share is the mean predicted probability over the SAME loans. Censored loans (not absorbed, not observed
  at 2008-12) are their own column in every table and are never redistributed. [Interpretation recorded here: the brief says "share of the book" without naming the
  denominator; the harness's convention (score on the uncensored cohort) is used for the rule. The alternative - all evaluation loans in the denominator, censored a
  column - is computed beside it as CONTEXT and does not enter the rule.]
  NOISE: paired bootstrap, 500 draws (numpy default_rng(53)), resampling EVALUATION LOANS with replacement; every draw scores grouped, pooled (and the baseline, for
  context) on the SAME resampled loans, and stores the difference error(pooled) - error(grouped) in pp (positive = grouped closer).
  DECISION: grouped is adopted iff it is favoured (strictly smaller error) in >= 475 of 500 draws on EITHER column; otherwise pooled ships, reported as
  "architecture made no measurable difference at h=6". If the columns disagree the split is reported; adoption follows the either-column rule.
  CHANGE FROM 11 OCT, AND WHY: DQ90+ was promoted from "reported alongside" to a column that can trigger adoption, because diagnostic 51 measured the DQ90+-origin model
  ordering loans on Default at AUC 0.713 [0.700, 0.726] - grouped has a real DQ90+ model, not a base rate.
"""

import difflib
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import baseline, horizon as hz, panel, panel_io               # noqa: E402
from src.memwatch import MemWatch                                       # noqa: E402

OUT = os.environ.get("DIAG53_OUT", os.path.join("diagnostics", "53_architecture_h6.txt"))
WORK = os.path.join("artifacts", "arch6")
LOGS = os.path.join(WORK, "logs")
REUSE = os.environ.get("DIAG53_REUSE") == "1"
CUT, H, FP_MAX = 200806, 6, 200806
CUT_IDX = panel.mi(CUT // 100, CUT % 100)
STATES = list(panel.STATES)
TRANSIENT = STATES[:4]
K = len(STATES)
I_DQ90, I_DEF = STATES.index("DQ90+"), STATES.index("Default")
COLS = [("Default", I_DEF), ("DQ90+", I_DQ90)]
N_DRAW, ADOPT_AT, BOOT_SEED = 500, 475, 53
REG = ["--min-sum-hessian-in-leaf", "10", "--lambda-l2", "10"]
GB = 2 ** 30
RULE_W = 124
LINES = []
T0 = time.perf_counter()

MODELS = {"grouped": [("A", "Current", ["Current"], "arch6_grouped_current"),
                      ("B", "DQ30+DQ60", ["DQ30", "DQ60"], "arch6_grouped_dq30_dq60"),
                      ("C", "DQ90+", ["DQ90+"], "arch6_grouped_dq90plus")],
          "pooled": [("P", "pooled (all four)", list(TRANSIENT), "arch6_pooled")]}
MODEL_DIR = os.path.join("artifacts", "ml")
BASE_MATCHED = os.path.join("artifacts", "baseline_to200806")           # from_period <= 200806: the same rows the ML arms trained on
BASE_LITERAL = os.path.join("artifacts", "baseline_200601_200806")      # from_period 200601 .. 200806, the brief's literal window
BASE_STORED = os.path.join("artifacts", "baseline")


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def write_report():
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


def on_abort(msg):
    emit("MEMORY PRESSURE, run stopped: " + msg)
    write_report()


def f(n):
    return format(int(n), ",")


def run_cmd(args, log_name):
    """Run a real src module as a subprocess; return (seconds, text). Raises if it fails."""
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    secs = time.perf_counter() - t0
    with open(os.path.join(LOGS, log_name), "w", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\n\n" + r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        emit(f"!!! {' '.join(args)} exited {r.returncode}; see {os.path.join(LOGS, log_name)}")
        emit((r.stdout + r.stderr)[-1500:])
        write_report()
        raise SystemExit(1)
    return secs, r.stdout


# ======================================================================
# the evaluation loans
# ======================================================================
def load_eval(quarters):
    ids, fs = [], []
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=["loan_id", "from_period", "from_state"], quarters=[q])
        d = d[d["from_period"] == CUT]
        ids.append(d["loan_id"].astype(str).to_numpy())
        fs.append(d["from_state"].astype(str).to_numpy())
        del d
    ids, fs = np.concatenate(ids), np.concatenate(fs)
    o = np.argsort(ids, kind="stable")
    ids, fs = ids[o], fs[o]
    assert len(np.unique(ids)) == len(ids), "a loan has two fit rows at the cut"
    bad = sorted(set(fs) - set(TRANSIENT))
    assert not bad, bad
    return ids, np.array([STATES.index(s) for s in fs], dtype=np.int64)


def model_population(quarters, groups):
    """Independent label pass: rows, loans, per-class counts and from_period range of each origin group, role == fit, from_period <= FP_MAX."""
    out = {key: {"rows": 0, "loans": 0, "labels": np.zeros(K, np.int64), "fp": [None, None]} for key in groups}
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=["loan_id", "from_period", "from_state", "to_state"], quarters=[q])
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        okb = (fp <= FP_MAX) & (y >= 0)
        fsr = d["from_state"].astype(str).to_numpy()
        for key, origin in groups.items():
            m = okb & np.isin(fsr, origin)
            if not m.any():
                continue
            o = out[key]
            o["rows"] += int(m.sum())
            o["loans"] += int(pd.unique(d["loan_id"].to_numpy()[m]).size)
            o["labels"] += np.bincount(y[m], minlength=K)
            lo, hi = int(fp[m].min()), int(fp[m].max())
            o["fp"][0] = lo if o["fp"][0] is None else min(o["fp"][0], lo)
            o["fp"][1] = hi if o["fp"][1] is None else max(o["fp"][1], hi)
        del d
    return out


# ======================================================================
# reading an ML arm back (layout-agnostic) and the per-arm checks
# ======================================================================
def newacc():
    return {"sums": np.zeros((H + 1, K)), "dev32": 0.0, "minp": np.inf, "minstep": np.inf, "minstep_pd": np.inf, "n": 0}


def accumulate(acc, P):
    """P: (n, H+1, 6) as stored (float32) or float64. Sums are taken in float64."""
    P64 = P.astype(np.float64)
    acc["sums"] += P64.sum(axis=0)
    acc["dev32"] = max(acc["dev32"], float(np.abs(P64.sum(axis=2) - 1.0).max()))
    acc["minp"] = min(acc["minp"], float(P.min()))
    acc["minstep"] = min(acc["minstep"], float(np.diff(P64[:, :, 4:], axis=1).min()))
    acc["minstep_pd"] = min(acc["minstep_pd"], float(np.diff(P64[:, :, 4] + P64[:, :, 5], axis=1).min()))
    acc["n"] += P.shape[0]


def read_arm(run_dir):
    acc, ids, p6 = newacc(), [], []
    marks = {"dev64": 0.0}
    for bdir in sorted(glob.glob(os.path.join(run_dir, "batch=*"))):
        with open(os.path.join(bdir, "_DONE.json"), encoding="utf-8") as fh:
            mk = json.load(fh)
        marks["dev64"] = max(marks["dev64"], mk["checks"]["worst_sum_deviation"])
        tbl = pq.read_table(os.path.join(bdir, "part.parquet"))
        cols = {}
        for name in ("loan_id", "state"):
            c = tbl.column(name).combine_chunks()
            if not pa.types.is_dictionary(c.type):
                c = pc.dictionary_encode(c).combine_chunks() if hasattr(pc.dictionary_encode(c), "combine_chunks") else pc.dictionary_encode(c)
            cols[name] = (c.dictionary.to_pylist(), c.indices.to_numpy())
        assert cols["state"][0] == STATES, cols["state"][0]
        h = tbl.column("h").to_numpy().astype(np.int64)
        prob = tbl.column("probability").to_numpy()
        assert prob.dtype == np.float32
        n = len(cols["loan_id"][0])
        assert h.max() == H and len(prob) == n * (H + 1) * K
        P = np.zeros((n, H + 1, K), dtype=np.float32)
        seen = np.zeros((n, H + 1, K), dtype=bool)
        P[cols["loan_id"][1], h, cols["state"][1]] = prob
        seen[cols["loan_id"][1], h, cols["state"][1]] = True
        assert seen.all()
        ids.append(np.array(cols["loan_id"][0], dtype=object))
        accumulate(acc, P)
        p6.append(P[:, H, :].astype(np.float64))
        del tbl, P, seen, prob, h, cols
    return np.concatenate(ids), np.concatenate(p6), acc, marks


def chain_baseline(P, start, chunk=500_000):
    acc, p6 = newacc(), []
    step = hz.homogeneous_step(P)
    for a in range(0, len(start), chunk):
        b = min(a + chunk, len(start))
        out = hz.chain(start[a:b], step, horizons=tuple(range(1, H + 1)))
        full = np.empty((b - a, H + 1, K))
        full[:, 0, :] = hz.one_hot(start[a:b])
        for k in range(1, H + 1):
            full[:, k, :] = out[k]
        accumulate(acc, full)
        p6.append(full[:, H, :].copy())
        del out, full
    return np.concatenate(p6), acc


# ======================================================================
# the rule
# ======================================================================
def boot(P6s, r, arms, seed=BOOT_SEED, n_draw=N_DRAW):
    """Paired loan bootstrap. Returns point estimates and per-draw errors (pp) for the rule's cohort and for the all-loans context variant."""
    N = len(r)
    mask = (r >= 0).astype(np.float64)
    real = {j: (r == j).astype(np.float64) for _, j in COLS}
    x = {(a, j): P6s[a][:, j] * mask for a in arms for _, j in COLS}
    xa = {(a, j): P6s[a][:, j].copy() for a in arms for _, j in COLS}

    def one(w):
        den, den_all = w @ mask, w.sum()
        res = {}
        for _, j in COLS:
            rj, rja = (w @ real[j]) / den, (w @ real[j]) / den_all
            for a in arms:
                res[("coh", a, j)] = 100.0 * abs((w @ x[(a, j)]) / den - rj)
                res[("all", a, j)] = 100.0 * abs((w @ xa[(a, j)]) / den_all - rja)
            res[("real_coh", j)], res[("real_all", j)] = 100 * rj, 100 * rja
        return res

    point = one(np.ones(N))
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_draw) for k in point}
    for d in range(n_draw):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        res = one(w)
        for k, v in res.items():
            draws[k][d] = v
    return point, draws


def main():
    os.makedirs(WORK, exist_ok=True)
    watch = MemWatch(on_abort=on_abort, abort_mb=700)
    watch.start()
    manifest = panel_io.read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    emit("MILESTONE: ARCHITECTURE COMPARISON  grouped (three models) vs pooled (one model), chained six months from 2008-06; homogeneous baseline alongside")
    emit(f"run date 2026-10-13   panel build {manifest['build_stamp']}   {os.cpu_count()} logical CPUs   partitions {quarters[0]}..{quarters[-1]}   DIAG53_REUSE={'1' if REUSE else '0'}")
    title("0. DECISIONS (closed, implemented as given) AND WHAT WAS NEEDED TO IMPLEMENT THEM")
    for ln in __doc__.split("\n"):
        emit(ln)
    emit()
    emit("Source changes made before this run are in artifacts/arch6/src_changes.diff (plain diff against copies taken before the edits; no git). Regression of their defaults, run before this script:")
    emit("  src.forecast, defaults, canary on artifacts/ml/pooled vs the stored data/forecast/pooled_v1 batch 0: 18,300,000 probabilities bit-identical.")
    emit("  src.train_ml, defaults (DQ90+ origin, to 200706): model.txt identical to the one the pre-change file writes; the sidecar differs only by the new lgb_params key.")
    emit("  src.baseline --fit-only vs the default (all-roles) run on the current panel: counts.csv identical.")
    emit("  src.horizon_baseline with the ORIGINAL horizon.py vs the edited one: scores.csv, compounding.csv, predicted_by_start.csv identical. Both differ from the stored scores.csv (6 Oct) in the 17th")
    emit("  significant digit only (e.g. 0.34700932785642002 vs ...1997).")
    emit("  The stored artifacts/baseline differs from a fresh default estimate by real counts (about 0.4%) because it was estimated on the 2026-10-05 panel build and the panel was rebuilt on 2026-10-08;")
    emit("  that is a finding about the stored artifact, not about the edit (section 4 reports it).")

    # ---------------------------------------------------------------- evaluation loans
    title("1. EVALUATION LOANS  (fit rows with from_period == 200806)")
    ids_eval, start = load_eval(quarters)
    N = len(ids_eval)
    emit(f"evaluation loans {f(N)}; by start state: " + ", ".join(f"{s} {f((start == i).sum())}" for i, s in enumerate(TRANSIENT)))
    write_report()

    # ---------------------------------------------------------------- training
    title("2. TRAINING  (real src.train_ml in a subprocess per model; --from-period-max 200806; min_sum_hessian_in_leaf 10, lambda_l2 10; 100 rounds)")
    train_secs, train_rss = {"grouped": 0.0, "pooled": 0.0}, {"grouped": 0, "pooled": 0}
    for arm, models in MODELS.items():
        for key, nm, origin, name in models:
            d = os.path.join(MODEL_DIR, name)
            if REUSE and os.path.exists(os.path.join(d, "model.json")):
                with open(os.path.join(d, "model.json"), encoding="utf-8") as fh:
                    sc = json.load(fh)
                secs = sc["dataset_construct_seconds"] + sc["train_seconds"]
                emit(f"model {key} ({name}): REUSED from an earlier run (training time taken from its sidecar: {secs:.0f} s)")
            else:
                args = ["-m", "src.train_ml", "--origin-states"] + origin + ["--from-period-max", str(FP_MAX), "--name", name, "--overwrite"] + REG
                secs, _ = run_cmd(args, f"train_{name}.log")
                emit(f"model {key} ({name}): trained in {secs:.0f} s wall-clock (subprocess, whole pipeline); log {os.path.join(LOGS, 'train_' + name + '.log')}")
            with open(os.path.join(d, "model.json"), encoding="utf-8") as fh:
                sc = json.load(fh)
            train_secs[arm] += secs
            train_rss[arm] = max(train_rss[arm], sc["peak_rss_bytes"])
    write_report()

    # ---------------------------------------------------------------- BEFORE ANY SCORE
    title("3. BEFORE ANY SCORE: rows, loans, labels, window and classes learned, per arm and per model")
    groups = {name: origin for arm, models in MODELS.items() for _, _, origin, name in models}
    pop = model_population(quarters, groups)
    side = {}
    for arm, models in MODELS.items():
        rows = 0
        emit(f"ARM {arm.upper()}")
        for key, nm, origin, name in models:
            with open(os.path.join(MODEL_DIR, name, "model.json"), encoding="utf-8") as fh:
                sc = json.load(fh)
            side[name] = sc
            pp = pop[name]
            lc = sc["label_counts"]
            learned = [s for s in STATES if lc[s] > 0]
            emit(f"  model {key} ({nm}): origin states {sc['origin_states']}; features {sc['features']}")
            emit(f"      from_state is a feature: {sc['from_state_is_feature']} (origin states {len(sc['origin_states'])}); lgb_params {sc['lgb_params']}; rounds {sc['n_rounds']}; sidecar {os.path.join(MODEL_DIR, name, 'model.json')}")
            emit(f"      training rows {f(sc['n_rows'])} (independent pass: {f(pp['rows'])}: {'MATCH' if pp['rows'] == sc['n_rows'] else '!!! DIFFERENT !!!'}); distinct loans {f(pp['loans'])}; "
                 f"from_period observed {sc['training_window']['from_period_min_observed']}..{sc['training_window']['from_period_max_observed']} "
                 f"(independent pass {pp['fp'][0]}..{pp['fp'][1]})")
            emit("      label counts: " + "  ".join(f"{s} {f(lc[s])}" for s in STATES))
            emit(f"      classes learned ({len(learned)} of 6): {learned}" + (f"; MISSING: {[s for s in STATES if lc[s] == 0]}" if len(learned) < 6 else ""))
            rows += sc["n_rows"]
        emit(f"  arm total: {f(rows)} training rows across {len(models)} model(s)")
    ga, po = MODELS["grouped"], MODELS["pooled"][0]
    gr = sum(side[m[3]]["n_rows"] for m in ga)
    emit(f"  grouped rows {f(gr)} vs pooled rows {f(side[po[3]]['n_rows'])}: {'EQUAL (the same rows, partitioned)' if gr == side[po[3]]['n_rows'] else 'DIFFERENT'}")
    lows = {n: s["training_window"]["from_period_min_observed"] for n, s in side.items()}
    emit(f"  observed lower edge of from_period across models: {min(lows.values())} (the brief's window starts 200601; train_ml has no lower bound and none was added)")
    emit()
    emit("REALISED STATE at 2008-12 for the evaluation loans (harness semantics: an absorbing state reached by 2008-12 counts whether or not the loan is observed then; otherwise the state observed")
    emit("in 2008-12; otherwise CENSORED, its own column, never redistributed). Read from fit rows through src.horizon.")
    L, _ = hz.read_test_loans(horizons=(H,), cut=CUT_IDX, progress=lambda s: None, roles=["fit"], cut_role="fit", period_min=CUT - 1)
    pos = L.index.get_indexer(ids_eval)
    assert (pos >= 0).all(), "an evaluation loan is missing from the harness frame"
    Le = L.iloc[pos]
    assert np.array_equal(Le["state_cut"].to_numpy(), start), "the harness's state at the cut disagrees with the fit row's from_state"
    r = hz.realised(Le, (H,), CUT_IDX)[H]
    del L, Le, pos
    gc.collect()
    emit(f"cross-check: the harness's state at the cut equals the fit row's from_state for all {f(N)} evaluation loans")
    emit(f"  {'start state':<14}{'loans':>12}" + "".join(f"{s:>12}" for s in STATES) + f"{'CENSORED':>12}")
    for i, s in enumerate(TRANSIENT):
        m = start == i
        emit(f"  {s:<14}{f(m.sum()):>12}" + "".join(f"{f((r[m] == j).sum()):>12}" for j in range(K)) + f"{f((r[m] == -1).sum()):>12}")
    emit(f"  {'ALL':<14}{f(N):>12}" + "".join(f"{f((r == j).sum()):>12}" for j in range(K)) + f"{f((r == -1).sum()):>12}")
    emit(f"  {'share of all':<14}{'':>12}" + "".join(f"{100 * (r == j).mean():>11.4f}%" for j in range(K)) + f"{100 * (r == -1).mean():>11.4f}%")
    unc = r >= 0
    emit(f"  {'share, observed':<14}{'':>12}" + "".join(f"{100 * (r == j).sum() / unc.sum():>11.4f}%" for j in range(K)) + f"{'(excluded)':>12}   cohort = {f(unc.sum())} loans")
    write_report()

    # ---------------------------------------------------------------- baseline estimates
    title("4. BASELINE: re-estimated on fit rows (real src.baseline, --fit-only) and compared with the stored matrix")
    bsecs = {}
    for nm, d, extra in (("matched rows (from_period <= 200806: the rows the ML arms trained on)", BASE_MATCHED, ["--from-period-max", "200806"]),
                         ("literal window (200601 .. 200806)", BASE_LITERAL, ["--from-period-min", "200601", "--from-period-max", "200806"])):
        if REUSE and os.path.exists(os.path.join(d, "counts.csv")):
            bsecs[d] = float("nan")
        else:
            bsecs[d], _ = run_cmd(["-m", "src.baseline", "--fit-only", "--out-dir", d] + extra, f"baseline_{os.path.basename(d)}.log")
        emit(f"{nm}: {d}   ({'reused' if bsecs[d] != bsecs[d] else '%.0f s' % bsecs[d]})")
    cm, pm = baseline.load_baseline(BASE_MATCHED)
    cl, pl = baseline.load_baseline(BASE_LITERAL)
    cs, ps = baseline.load_baseline(BASE_STORED)
    emit("HEADLINE baseline = the matched-rows estimate (identical rows to the ML arms). Why: train_ml has no lower from_period bound, so the ML arms' real window reaches back to "
         f"{min(lows.values())}; the brief asks that all three arms see the same training window. The literal 200601..200806 estimate is chained too and reported (context only).")
    emit(f"  count matrix totals: matched {f(cm.to_numpy().sum())}; literal 200601..200806 {f(cl.to_numpy().sum())} (rows before 200601: {f(cm.to_numpy().sum() - cl.to_numpy().sum())}); stored {f(cs.to_numpy().sum())}")
    emit(f"  max |p(matched) - p(literal)| = {np.abs(pm.to_numpy() - pl.to_numpy()).max():.3e};  max |p(matched) - p(stored)| = {np.abs(pm.to_numpy() - ps.to_numpy()).max():.3e}")
    with open(os.path.join(BASE_STORED, "baseline_report.txt"), encoding="utf-8") as fh:
        stored_stamp = [ln for ln in fh if ln.startswith("build stamp")][0].strip()
    emit(f"  the stored artifacts/baseline carries '{stored_stamp}'; this run's panel build is {manifest['build_stamp']}.")
    emit(f"  {'matched estimate':<26}" + "".join(f"{s:>12}" for s in STATES))
    for i, s in enumerate(TRANSIENT):
        emit(f"  {'from ' + s:<26}" + "".join(f"{pm.to_numpy()[i, j]:>12.6f}" for j in range(K)))
    emit(f"  {'(matched - stored)':<26}" + "".join(f"{'':>12}" for _ in STATES))
    for i, s in enumerate(TRANSIENT):
        emit(f"  {'from ' + s:<26}" + "".join(f"{pm.to_numpy()[i, j] - ps.to_numpy()[i, j]:>+12.6f}" for j in range(K)))
    write_report()

    # ---------------------------------------------------------------- forecasts
    title("5. FORECASTS  (real src.forecast, --cut 200806 --cut-role fit --horizon 6; the same expected loan count, derived independently above)")
    fc = {}
    for arm in ("grouped", "pooled"):
        models = MODELS[arm]
        if arm == "grouped":
            mm = [f"{o}={os.path.join(MODEL_DIR, [m for m in models if o in m[2]][0][3]).replace(os.sep, '/')}" for o in TRANSIENT]
            sel = ["--model-map"] + mm
        else:
            sel = ["--model", os.path.join(MODEL_DIR, models[0][3]).replace(os.sep, "/")]
        rep = os.path.join(WORK, f"forecast_report_{arm}.txt")
        secs, out = run_cmd(["-m", "src.forecast", "--run", f"arch6_{arm}"] + sel + ["--cut", str(CUT), "--cut-role", "fit", "--horizon", str(H), "--expect-loans", str(N),
                                                                                   "--report", rep], f"forecast_{arm}.log")
        with open(rep, encoding="utf-8") as fh:
            txt = fh.read()
        rssm = re.search(r"peak RSS ([0-9.]+) GB", txt)
        fc[arm] = {"secs": secs, "rss": float(rssm.group(1)) if rssm else float("nan"), "report": rep}
        emit(f"{arm}: forecast {secs:.0f} s wall-clock; forecaster's own report {rep} (peak RSS {fc[arm]['rss']:.2f} GB)")
        if arm == "grouped":
            emit("   model map: " + "  ".join(mm))
        else:
            emit("   model: " + sel[1])
    write_report()

    # ---------------------------------------------------------------- read arms
    title("6. CHECKS  (every ML forecast read back from the stored parquet; the baseline chained in float64 over the same loans)")
    arms, acc, extra = {}, {}, {}
    for arm in ("grouped", "pooled"):
        t1 = time.perf_counter()
        ids, p6, a, mk = read_arm(os.path.join("data", "forecast", f"arch6_{arm}"))
        arms[arm], acc[arm], extra[arm] = p6, a, mk
        extra[arm]["ids_equal"] = bool(len(ids) == N and (ids == ids_eval).all())
        extra[arm]["read_secs"] = time.perf_counter() - t1
    p6b, accb = chain_baseline(pm.to_numpy(), start)
    p6l, accl = chain_baseline(pl.to_numpy(), start)
    arms["baseline"], acc["baseline"] = p6b, accb
    arms["baseline_literal"], acc["baseline_literal"] = p6l, accl
    emit(f"both ML arms forecast the SAME evaluation loans in the SAME order (stored loan ids == the independently loaded sorted list, {f(N)} loans): "
         f"grouped {extra['grouped']['ids_equal']}, pooled {extra['pooled']['ids_equal']}")
    emit(f"the baseline was chained over the same list (start states of those {f(N)} loans via src.horizon.chain): True")
    emit(f"  {'arm':<20}{'worst |sum-1| float64 (computed)':>34}{'worst |sum-1| float32 as stored':>34}{'smallest probability':>24}{'Prepaid/Default smallest step':>32}")
    for arm in ("grouped", "pooled", "baseline", "baseline_literal"):
        a = acc[arm]
        d64 = extra[arm]["dev64"] if arm in extra else a["dev32"]
        stored = f"{a['dev32']:.3e}" if arm in extra else f"{a['dev32']:.3e} (float64; not stored)"
        emit(f"  {arm:<20}{d64:>34.3e}{stored:>34}{a['minp']:>24.3e}{min(a['minstep'], a['minstep_pd']):>32.3e}")
    for arm in ("grouped", "pooled"):
        a = acc[arm]
        emit(f"  {arm}: sum-to-one float32 stored vs STORED_TOL 1e-05: {'within' if a['dev32'] <= 1e-5 else '*** EXCEEDS ***'} (tolerance not loosened); float64 computed vs SUM_TOL 1e-09: "
             f"{'within' if extra[arm]['dev64'] <= 1e-9 else '*** EXCEEDS ***'}; no negative probability: {'PASS' if a['minp'] >= 0 else 'FAIL'}; "
             f"Prepaid and Default mass non-decreasing in h: {'PASS' if min(a['minstep'], a['minstep_pd']) >= 0 else 'FAIL'}")
    checks_ok = all(extra[a]["ids_equal"] for a in ("grouped", "pooled")) and all(acc[a]["minp"] >= 0 and min(acc[a]["minstep"], acc[a]["minstep_pd"]) >= 0 for a in acc)
    emit(f"  baseline: sums within 1e-12 by construction of src.horizon.chain (asserted); no negative probability: {'PASS' if acc['baseline']['minp'] >= 0 else 'FAIL'}; "
         f"Prepaid and Default mass non-decreasing: {'PASS' if min(acc['baseline']['minstep'], acc['baseline']['minstep_pd']) >= 0 else 'FAIL'}")
    write_report()

    # ---------------------------------------------------------------- per-h shares
    title("7. PREDICTED SHARE OF THE BOOK BY HORIZON  (mean predicted probability over all %s evaluation loans)" % f(N))
    for arm in ("grouped", "pooled", "baseline"):
        emit(f"{arm.upper()}" + ("  (matched-rows baseline)" if arm == "baseline" else ""))
        emit(f"  {'h':>3}" + "".join(f"{s:>12}" for s in STATES) + f"{'sum':>12}")
        for h in range(H + 1):
            v = acc[arm]["sums"][h] / N
            emit(f"  {h:>3}" + "".join(f"{100 * x:>11.4f}%" for x in v) + f"{100 * v.sum():>11.6f}%")

    # ---------------------------------------------------------------- h = 6 versus realised
    title("8. h = 6: PREDICTED vs REALISED SHARE  (rule cohort: evaluation loans observed at 2008-12; %s loans, %s censored loans excluded and shown separately)" % (f(unc.sum()), f((~unc).sum())))
    real_c = np.array([(r == j).sum() for j in range(K)]) / unc.sum()
    pred_c = {a: arms[a][unc].mean(axis=0) for a in arms}
    emit(f"  {'state':<10}{'realised':>11}" + "".join(f"{a + ' pred':>20}{'|err| pp':>10}" for a in ("grouped", "pooled", "baseline")) + f"{'closest':>12}")
    for j, s in enumerate(STATES):
        errs = {a: 100 * abs(pred_c[a][j] - real_c[j]) for a in ("grouped", "pooled", "baseline")}
        emit(f"  {s:<10}{100 * real_c[j]:>10.4f}%" + "".join(f"{100 * pred_c[a][j]:>19.4f}%{errs[a]:>10.4f}" for a in ("grouped", "pooled", "baseline")) + f"{min(errs, key=errs.get):>12}")
    emit(f"  literal-window baseline (context): " + "  ".join(f"{s} {100 * arms['baseline_literal'][unc].mean(axis=0)[j]:.4f}%" for j, s in enumerate(STATES)))
    emit()
    emit("  CONTEXT, alternative denominator (all %s evaluation loans; censored loans a column; predicted = mean over all loans):" % f(N))
    real_a = np.array([(r == j).sum() for j in range(K)]) / N
    emit(f"  {'state':<10}{'realised':>11}" + "".join(f"{a + ' pred':>20}{'|err| pp':>10}" for a in ("grouped", "pooled", "baseline")) + f"{'closest':>12}")
    pred_a = {a: arms[a].mean(axis=0) for a in arms}
    for j, s in enumerate(STATES):
        errs = {a: 100 * abs(pred_a[a][j] - real_a[j]) for a in ("grouped", "pooled", "baseline")}
        emit(f"  {s:<10}{100 * real_a[j]:>10.4f}%" + "".join(f"{100 * pred_a[a][j]:>19.4f}%{errs[a]:>10.4f}" for a in ("grouped", "pooled", "baseline")) + f"{min(errs, key=errs.get):>12}")
    emit(f"  {'CENSORED':<10}{100 * (r == -1).mean():>10.4f}%  (no prediction is made for it; not redistributed)")
    write_report()

    # ---------------------------------------------------------------- the rule
    title("9. THE PRE-REGISTERED RULE  (paired loan bootstrap, %d draws, seed %d; error = |predicted share - realised share| in pp)" % (N_DRAW, BOOT_SEED))
    t2 = time.perf_counter()
    point, draws = boot(arms, r, ("grouped", "pooled", "baseline"))
    emit(f"bootstrap wall-clock {time.perf_counter() - t2:.0f} s")
    emit()
    emit(f"  {'column':<10}{'realised':>11}{'grouped err':>14}{'pooled err':>13}{'baseline err':>14}   (point estimates, full evaluation cohort)")
    for nm, j in COLS:
        emit(f"  {nm:<10}{point[('real_coh', j)]:>10.4f}%{point[('coh', 'grouped', j)]:>14.4f}{point[('coh', 'pooled', j)]:>13.4f}{point[('coh', 'baseline', j)]:>14.4f}")
    emit()
    wins = {}
    for nm, j in COLS:
        dg = draws[("coh", "pooled", j)] - draws[("coh", "grouped", j)]
        fav_g, fav_p, tie = int((dg > 0).sum()), int((dg < 0).sum()), int((dg == 0).sum())
        wins[nm] = (fav_g, fav_p, tie)
        q = np.percentile(dg, [2.5, 25, 50, 75, 97.5])
        emit(f"  {nm}: draws favouring GROUPED {fav_g} / {N_DRAW}; favouring POOLED {fav_p}; ties {tie}.   difference error(pooled) - error(grouped), pp (positive = grouped closer): "
             f"mean {dg.mean():+.4f}, sd {dg.std(ddof=1):.4f}, min {dg.min():+.4f}, 2.5% {q[0]:+.4f}, 25% {q[1]:+.4f}, median {q[2]:+.4f}, 75% {q[3]:+.4f}, 97.5% {q[4]:+.4f}, max {dg.max():+.4f}")
        bg = draws[("coh", "baseline", j)]
        emit(f"      (context only, not in the rule: baseline |err| over the draws: mean {bg.mean():.4f}, 2.5% {np.percentile(bg, 2.5):.4f}, 97.5% {np.percentile(bg, 97.5):.4f}; "
             f"grouped mean {draws[('coh', 'grouped', j)].mean():.4f}; pooled mean {draws[('coh', 'pooled', j)].mean():.4f})")
    adopt = [nm for nm, (fg, _, _) in wins.items() if fg >= ADOPT_AT]
    lose = [nm for nm, (_, fp_, _) in wins.items() if fp_ >= ADOPT_AT]
    split = bool(adopt) and bool(lose)
    dirs = {nm: ("grouped" if point[("coh", "grouped", j)] < point[("coh", "pooled", j)] else "pooled" if point[("coh", "grouped", j)] > point[("coh", "pooled", j)] else "equal") for nm, j in COLS}
    emit()
    emit(f"  point-estimate direction by column: {dirs}")
    if adopt:
        outcome = f"GROUPED IS ADOPTED: favoured in at least {ADOPT_AT} of {N_DRAW} draws on {adopt} ({', '.join(f'{nm} {wins[nm][0]}' for nm in adopt)})."
    else:
        outcome = (f"POOLED SHIPS: grouped was favoured in {wins['Default'][0]} (Default) and {wins['DQ90+'][0]} (DQ90+) of {N_DRAW} draws, neither reaching {ADOPT_AT}. "
                   f"Reported as: architecture made no measurable difference at h=6.")
    emit("  MECHANICAL OUTCOME: " + outcome)
    emit(f"  columns disagree (grouped reaches {ADOPT_AT} on one, pooled reaches {ADOPT_AT} on the other): {'YES - ' + str({'grouped wins': adopt, 'grouped loses': lose}) if split else 'no'}; "
         f"point-estimate directions differ: {'YES' if len(set(dirs.values())) > 1 else 'no'}")
    emit(f"  THE BASELINE DID NOT ENTER THE RULE: the decision above uses only the grouped and pooled draws.")
    emit()
    emit(f"  CONTEXT, alternative denominator (all evaluation loans, censored a column), not part of the rule - draws favouring grouped: " +
         ", ".join(f"{nm} {int(((draws[('all', 'pooled', j)] - draws[('all', 'grouped', j)]) > 0).sum())}" for nm, j in COLS) +
         "; point errors grouped/pooled/baseline (pp): " + "; ".join(f"{nm} {point[('all', 'grouped', j)]:.4f}/{point[('all', 'pooled', j)]:.4f}/{point[('all', 'baseline', j)]:.4f}" for nm, j in COLS))
    write_report()

    # ---------------------------------------------------------------- cost
    title("10. COST  (wall-clock per arm; peak RSS from src.memwatch inside each launched module)")
    emit(f"  {'arm':<10}{'train s':>10}{'train peak RSS GB':>20}{'forecast s':>12}{'forecast peak RSS GB':>22}")
    for arm in ("grouped", "pooled"):
        emit(f"  {arm:<10}{train_secs[arm]:>10.0f}{train_rss[arm] / GB:>20.2f}{fc[arm]['secs']:>12.0f}{fc[arm]['rss']:>22.2f}")
    emit(f"  baseline: estimation {', '.join('%s %.0f s' % (os.path.basename(k), v) for k, v in bsecs.items())}; chained in seconds inside this process")
    watch.stop()
    emit(f"  this driver (reads, checks, bootstrap): process peak RSS {watch.peak_rss() / GB:.2f} GB; lowest system-available memory seen {watch.min_avail / GB:.2f} GB; total wall-clock {time.perf_counter() - T0:.0f} s")

    # ---------------------------------------------------------------- findings
    title("FINDINGS  (measured only)")
    emit(f"- RULE OUTCOME: {outcome}")
    emit(f"- Draws favouring grouped: Default {wins['Default'][0]}/{N_DRAW}, DQ90+ {wins['DQ90+'][0]}/{N_DRAW}  (favouring pooled: Default {wins['Default'][1]}, DQ90+ {wins['DQ90+'][1]}; ties {wins['Default'][2]}, {wins['DQ90+'][2]}).")
    emit(f"- Evaluation loans {f(N)}; observed at 2008-12 {f(unc.sum())}; censored {f((~unc).sum())} ({100 * (~unc).mean():.4f}%). All checks: {'PASS' if checks_ok else 'FAIL (see section 6)'}.")
    emit(f"- h = 6 shares on the rule's cohort (percent of the book):")
    emit(f"  {'':<10}" + "".join(f"{s:>12}" for s in STATES))
    emit(f"  {'realised':<10}" + "".join(f"{100 * real_c[j]:>11.4f}%" for j in range(K)))
    for a in ("grouped", "pooled", "baseline"):
        emit(f"  {a:<10}" + "".join(f"{100 * pred_c[a][j]:>11.4f}%" for j in range(K)))
    emit(f"- Absolute error in pp, Default / DQ90+: grouped {point[('coh', 'grouped', I_DEF)]:.4f} / {point[('coh', 'grouped', I_DQ90)]:.4f}; pooled {point[('coh', 'pooled', I_DEF)]:.4f} / "
         f"{point[('coh', 'pooled', I_DQ90)]:.4f}; baseline {point[('coh', 'baseline', I_DEF)]:.4f} / {point[('coh', 'baseline', I_DQ90)]:.4f} (the baseline is context only).")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except MemoryError as e:
        emit(f"MEMORY PRESSURE: MemoryError, run stopped: {e}")
        write_report()
        raise
