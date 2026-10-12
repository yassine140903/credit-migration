"""SMOKE TEST: the forecaster's three-model dispatch (Current / DQ30+DQ60 / DQ90+).   PLUMBING ONLY: model quality is irrelevant and is never evaluated.

    python diagnostics/52_dispatch_smoke_test.py   # from the repo root

Trains three deliberately tiny models through the REAL src.train_ml.main (model A: Current; model B: DQ30+DQ60, two origin states; model C: DQ90+),
runs the REAL src.forecast.main over one small batch with --model-map, and then checks the written forecast row of one loan per origin group against a
row computed by hand from the group's own model file and sidecar. Writes diagnostics/52_dispatch_smoke_test.txt. Nothing under src/ is modified, no
git command, no .gitignore / .env change. Fit rows only for training (train_ml reads roles=['fit']); the forecaster reads the condition rows at the cut
(that is its job; evaluate rows are never read here); the hand rows read the same condition rows through load_panel.

WHAT THE REAL SCRIPTS DO NOT OFFER, AND HOW THIS DIAGNOSTIC GETS AROUND IT WITHOUT EDITING THEM. src/train_ml.py has no row-cap and no round-count option
(N_ROUNDS = 100 is a module constant, and it reads the whole fit window). So, IN THIS PROCESS ONLY: train_ml.N_ROUNDS is set to 5; train_ml.keep_rows is
wrapped so that each partition keeps at most ceil(50,000 / number of partitions up to 2008Q2) rows with a valid label, drawn with a generator seeded from the
partition's own valid-row count (so pass 1, the label scan, and pass 2, the matrix build, draw the identical rows - train_ml's own consistency check
enforces that); train_ml.ART_DIR is pointed at artifacts/diag52/models so the tiny models are never mistaken for artifacts/ml/ models. The real
label_scan / build_matrix / lgb.train / sidecar-writing code runs unchanged. For the forecaster, IN THIS PROCESS ONLY: forecast.OUT_ROOT is pointed at
artifacts/diag52/forecast (not data/forecast) and forecast.REPORT at artifacts/diag52/37_report_copy.txt, so the committed diagnostics/37_forecaster_build.txt
is not overwritten. The forecaster is run with --batch-size 3000 --canary (the first 3,000 loans by loan_id, h = 0..60: the full horizon is kept, not
shortened, so the whole chaining of the rows is exercised; the checks below cover every h, and the comparison rows are h = 0 and h = 1).

h = 0 versus h = 1. The brief asks for the comparison at h = 0, but the forecaster's h = 0 is the INITIAL VECTOR (probability 1 on the loan's own state,
src/forecast.py docstring), not a model output, so no model column can be tested there. The first row a model writes is h = 1: for a loan whose h = 0 vector
is one-hot on origin o, new[j] = P_o[j] exactly. The hand-computed row is therefore compared with h = 1; the h = 0 row is printed beside it for completeness.

HAND ROW. For each chosen loan-month: the group's model.txt is loaded with lightgbm.Booster and its model.json with json (not through src.forecast); the
feature vector is built from the loan's condition row read through panel_io.load_panel, in the sidecar's feature order, float32 numerics, categoricals with the
sidecar's category lists (from_state only if the sidecar lists it, with the loan's own origin state); predict() is called on that one row; the returned
columns are mapped through the sidecar's class_order into the panel's six-state order; every class the sidecar's label_counts says the model never saw is
set to EXACT ZERO (no renormalisation). A difference is reported per class as |hand - stored| two ways: float64 hand row against the stored float32
widened to float64, and the hand row cast to float32 against the stored float32 (bit-exact comparison; the strict figure, the one the PASS/FAIL uses).
No tolerance is applied.
"""

import contextlib
import gc
import io
import json
import math
import os
import shutil
import sys
import time
from argparse import Namespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import panel, panel_io, train_ml, forecast                    # noqa: E402
from src.memwatch import MemWatch                                      # noqa: E402

OUT = os.environ.get("DIAG52_OUT", os.path.join("diagnostics", "52_dispatch_smoke_test.txt"))
ROOT = os.environ.get("DIAG52_ROOT", os.path.join("artifacts", "diag52"))
MODEL_ROOT, FC_ROOT = os.path.join(ROOT, "models"), os.path.join(ROOT, "forecast")
RUN = "diag52_dispatch"
FP_MAX = 200806
CAP_ROWS, ROUNDS, BATCH = 50_000, 5, 3000
STATES = list(panel.STATES)
TRANSIENT = STATES[:4]
GB = 2 ** 30
RULE_W = 124
LINES = []
GROUPS = [("A", "Current", ["Current"], "diag52_A_current"), ("B", "DQ30+DQ60", ["DQ30", "DQ60"], "diag52_B_dq30_dq60"),
          ("C", "DQ90+", ["DQ90+"], "diag52_C_dq90plus")]
T0 = time.perf_counter()


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


def g17(x):
    return "%.17g" % float(x)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def on_abort(msg):
    emit("MEMORY PRESSURE, run stopped: " + msg)
    write_report()


def main():
    os.makedirs(ROOT, exist_ok=True)
    watch = MemWatch(on_abort=on_abort, abort_mb=700)
    watch.start()
    import lightgbm as lgb

    emit("DIAGNOSTIC 52: the forecaster's three-model dispatch   (smoke test; plumbing only; model quality is irrelevant and is not evaluated)")
    emit(f"lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs   run date 2026-10-12")
    manifest = panel_io.read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    emit(f"panel build {manifest['build_stamp']}; partitions {len(quarters)} ({quarters[0]}..{quarters[-1]})")

    # ------------------------------------------------------------------ what the real scripts offer
    title("0. WHAT THE REAL SCRIPTS OFFER  (read from the source in this session; reported before anything runs)")
    emit("src/train_ml.py: options --name --with-loan-age --origin-states --from-period-max --overwrite. NO row-cap option; NO round-count option (N_ROUNDS = 100, a module constant).")
    emit("   from_state is a feature iff len(origin_states) > 1 (resolve(): pooled = len(origin) > 1); the sidecar records features, categories, class_order, origin_states,")
    emit("   from_state_is_feature, label_counts, n_rows, training_window, n_rounds. params declare num_class = 6 whatever the labels contain.")
    emit("src/forecast.py: --model-map STATE=DIR (one Model per distinct directory; unique_models() gives one frame per model). Model.__init__ REQUIRES sidecar class_order == panel.STATES")
    emit("   (forecast.py:123), booster feature names == sidecar features (:125) and num_model_per_iteration == 6 (:128). propagate() writes out[k*n:(k+1)*n] into trans[:, o, :] (:283-285)")
    emit("   with no class mapping and no zero-fill of any column. build_frame() takes the column list from the model's own sidecar features (from_state only if listed).")
    emit(f"So the three-model dispatch (--model-map, a frame per model, feature order and from_state keyed off each sidecar) IS implemented; a zero-fill of absent classes is not in the code.")
    emit("Consequence for this diagnostic: nothing needed to STOP on 'dispatch not implemented'. The cap and rounds are applied by patching module attributes in this process only (see the docstring).")

    # ------------------------------------------------------------------ census of the real window (uncapped)
    title("1. THE TRAINING WINDOW AS IT ACTUALLY IS  (uncapped label census through train_ml.label_scan; role == 'fit', from_period <= %d)" % FP_MAX)
    emit("train_ml has only an upper from_period bound; the lower edge is wherever the fit rows start (observed below). Brief's window: 200601..200806.")
    census = {}
    for key, nm, origin, _ in GROUPS:
        S, _ = quiet(train_ml.label_scan, quarters, origin, FP_MAX)
        census[key] = S
        n = int(sum(S["rows_by_quarter"].values()))
        emit(f"model {key} ({nm}): origin states {origin}; rows {n:,}; from_period observed {S['fp_min']}..{S['fp_max']}; null labels dropped {S['n_null_label']:,}")
        emit("   labels: " + "  ".join(f"{s} {int(c):,}" for s, c in zip(STATES, S["label_counts"])))
        if len(origin) > 1:
            emit("   rows by origin state: " + "  ".join(f"{o} {c:,}" for o, c in S["from_counts"].items()))
    emit(f"Window lower edge: model A's uncapped rows reach back to from_period {census['A']['fp_min']} (B {census['B']['fp_min']}, C {census['C']['fp_min']}); "
         f"train_ml has no lower from_period bound, so 'window 200601..200806' is, in the real code path, 'everything up to 200806'. Not altered here.")
    S7, _ = quiet(train_ml.label_scan, quarters, ["DQ90+"], 200706)
    c6, c7 = census["C"]["label_counts"], S7["label_counts"]
    ip = STATES.index("Prepaid")
    emit(f"DQ90+ -> Prepaid transitions in the training window: from_period <= 200706 (the 11 Oct window): {int(c7[ip]):,} of {int(c7.sum()):,} DQ90+ rows; "
         f"from_period <= 200806 (today's window): {int(c6[ip]):,} of {int(c6.sum()):,}.  "
         + ("The 11 Oct expectation of NO DQ90+ -> Prepaid transitions " + ("holds at 200706 and does NOT hold at 200806." if c7[ip] == 0 and c6[ip] > 0
                                                                              else "does not hold even at 200706." if c7[ip] > 0
                                                                              else "also holds at 200806."))
         + "  (full-window counts, before the 50,000-row cap)")
    write_report()

    # ------------------------------------------------------------------ train
    title("2. THE THREE TINY MODELS  (real src.train_ml.main; N_ROUNDS patched to %d, rows capped at about %s, in this process only)" % (ROUNDS, format(CAP_ROWS, ",")))
    n_q = sum(1 for q in quarters if q <= "2008Q2")
    k_q = int(math.ceil(CAP_ROWS / n_q))
    emit(f"cap: {n_q} partitions up to 2008Q2 -> at most {k_q:,} valid-label rows per partition (<= {k_q * n_q:,} rows per model); rows drawn at random within a partition, "
         f"seeded by that partition's valid-row count, so the label scan and the matrix build see the same rows.")
    orig_keep = train_ml.keep_rows

    def capped(d, origin, fp_max):
        m, y = orig_keep(d, origin, fp_max)
        valid = m & (y >= 0)
        idx = np.flatnonzero(valid)
        if len(idx) > k_q:
            rng = np.random.default_rng([52, len(idx)])
            sel = rng.choice(idx, size=k_q, replace=False)
            keep = np.zeros(len(m), dtype=bool)
            keep[sel] = True
            m = (m & ~valid) | keep
        return m, y

    train_ml.keep_rows, train_ml.N_ROUNDS, train_ml.ART_DIR = capped, ROUNDS, MODEL_ROOT
    side = {}
    t_train = time.perf_counter()
    for key, nm, origin, name in GROUPS:
        _, log = quiet(train_ml.main, ["--origin-states"] + origin + ["--from-period-max", str(FP_MAX), "--name", name, "--overwrite"])
        with open(os.path.join(MODEL_ROOT, name, "model.json"), encoding="utf-8") as fh:
            side[key] = json.load(fh)
        emit(f"--- train_ml log, model {key} ({name}) ---")
        for ln in log.splitlines():
            if not ln.startswith("  scan") and not ln.startswith("  matrix"):
                emit("   " + ln)
    train_ml.keep_rows = orig_keep
    emit(f"training wall-clock, three models: {time.perf_counter() - t_train:.1f} s")
    write_report()

    title("3. REPORTED BEFORE ANY FORECAST: classes each model learned, and the class order its sidecar records")
    absent = {}
    for key, nm, origin, name in GROUPS:
        s = side[key]
        lc = s["label_counts"]
        absent[key] = [c for c in STATES if lc[c] == 0]
        emit(f"model {key} ({nm})  dir {os.path.join(MODEL_ROOT, name).replace(os.sep, '/')}")
        emit(f"   origin states {s['origin_states']}; rows {s['n_rows']:,}; n_rounds {s['n_rounds']}; training window {s['training_window']}")
        emit(f"   sidecar class_order: {s['class_order']}  ({'== panel.STATES' if s['class_order'] == STATES else 'DIFFERS from panel.STATES'})")
        emit("   label counts: " + "  ".join(f"{c} {lc[c]:,}" for c in STATES))
        emit(f"   classes learned ({6 - len(absent[key])} of 6): {[c for c in STATES if lc[c] > 0]}")
        emit(f"   classes MISSING: {absent[key] or 'none (all six present)'}" + ("   <- landmine (1) is live for this model: the zero-fill path is genuinely exercised" if absent[key] else ""))
    emit()
    if any(absent.values()):
        emit("At least one model learned fewer than six classes: landmine (1) is live. Note: every model still DECLARES six classes (num_class = 6) and so returns six columns "
             "in the panel order; a class with no training label is not removed from the output, it simply gets a small softmax value.")
    else:
        emit("Every model learned all six classes: the zero-fill path is NOT exercised by this run.")
    write_report()

    # ------------------------------------------------------------------ forecast
    title("4. THE FORECAST  (real src.forecast.main, --model-map, --batch-size %d --canary; outputs redirected to %s)" % (BATCH, ROOT.replace(os.sep, "/")))
    run_dir = os.path.join(FC_ROOT, RUN)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(FC_ROOT, exist_ok=True)
    forecast.OUT_ROOT, forecast.REPORT = FC_ROOT, os.path.join(ROOT, "37_report_copy.txt")
    frame_log, cap = [], {}
    orig_bf, orig_li = forecast.build_frame, forecast.load_initial

    def bf(m, origins, sl, init, frozen):
        fr = orig_bf(m, origins, sl, init, frozen)
        n = sl.stop - sl.start
        rec = {"model": m.name, "origins": [TRANSIENT[o] for o in origins], "rows": len(fr), "n": n, "columns": list(fr.columns), "id": id(fr),
               "model_features": list(m.features), "dtypes": {c: str(t) for c, t in fr.dtypes.items()}, "frame": fr, "booster": m.booster}
        if "from_state" in fr.columns:
            fs = fr["from_state"].astype(str).to_numpy()
            rec["from_state_blocks"] = [sorted(set(fs[k * n:(k + 1) * n])) for k in range(len(origins))]
            rec["from_state_categories"] = list(fr["from_state"].cat.categories)
        frame_log.append(rec)
        return fr

    def li(expect):
        r = orig_li(expect)
        cap["init"], cap["frozen"] = r[0], r[1]
        return r

    forecast.build_frame, forecast.load_initial = bf, li
    mm = [f"{o}={os.path.join(MODEL_ROOT, name).replace(os.sep, '/')}" for o, name in
          (("Current", GROUPS[0][3]), ("DQ30", GROUPS[1][3]), ("DQ60", GROUPS[1][3]), ("DQ90+", GROUPS[2][3]))]
    emit("model map: " + "  ".join(mm))
    t_fc = time.perf_counter()
    try:
        _, flog = quiet(forecast.main, ["--run", RUN, "--model-map"] + mm + ["--batch-size", str(BATCH), "--canary"])
    except BaseException as e:                                        # a raised check IS a finding: report and stop
        emit(f"!!! src.forecast.main raised {type(e).__name__}: {e}")
        emit("STOP: the forecaster did not complete; no later check could be run. Nothing was reconciled or adjusted.")
        write_report()
        raise
    finally:
        forecast.build_frame, forecast.load_initial = orig_bf, orig_li
    emit(f"forecast wall-clock {time.perf_counter() - t_fc:.1f} s")
    for ln in flog.splitlines():
        if ln.strip():
            emit("   " + ln)
    write_report()

    init, frozen = cap["init"], cap["frozen"]
    batch_ids = init["loan_id"].to_numpy()[:BATCH]
    batch_state = init["from_state"].to_numpy()[:BATCH]
    emit(f"batch: {len(batch_ids):,} loans, {batch_ids[0]} .. {batch_ids[-1]}; origin states in the batch: " +
         ", ".join(f"{s} {int((batch_state == s).sum()):,}" for s in TRANSIENT))

    # read the stored forecast independently of the forecaster's layout
    tbl = pq.read_table(os.path.join(run_dir, "batch=00000", "part.parquet")).to_pandas()
    lid, uid = pd.factorize(tbl["loan_id"].astype(str))
    hh = tbl["h"].to_numpy().astype(np.int64)
    ss = np.array([STATES.index(s) for s in tbl["state"].astype(str).to_numpy()])
    assert tbl["probability"].dtype == np.float32, tbl["probability"].dtype
    H = int(hh.max()) + 1
    P32 = np.zeros((len(uid), H, 6), dtype=np.float32)
    P32[lid, hh, ss] = tbl["probability"].to_numpy()
    assert len(tbl) == len(uid) * H * 6
    order = {l: i for i, l in enumerate(uid)}
    del tbl
    with open(os.path.join(run_dir, "batch=00000", "_DONE.json"), encoding="utf-8") as fh:
        marker = json.load(fh)
    emit(f"stored forecast read back independently: {len(uid):,} loans x h = 0..{H - 1} x 6 states, dtype float32")

    # ------------------------------------------------------------------ dispatch evidence
    title("5. DISPATCH EVIDENCE  (what the forecaster actually did, resolved from the sidecars and the run, not assumed)")
    models = forecast.load_models(Namespace(model=None, model_map=mm))
    emit("origin state -> model the forecaster's own load_models() assigns (and that model's origin states):")
    for o in TRANSIENT:
        emit(f"   {o:<6} -> {models[o].name:<22} origin states {models[o].origin_states}")
    emit()
    emit("frames the forecaster built, once per model (wrapper around the real build_frame; the real function ran):")
    emit(f"   {'model':<22}{'origin blocks':<18}{'rows':>8}  columns")
    for r in frame_log[:3]:
        emit(f"   {r['model']:<22}{str(r['origins']):<18}{r['rows']:>8,}  {r['columns']}")
    emit(f"   (build_frame was called {len(frame_log)} times in total: once per model per batch, because the frames are built before the monthly loop and reused)")
    emit()
    emit("feature orders side by side (from the three sidecars):")
    fe = {k: side[k]["features"] for k, *_ in GROUPS}
    width = max(len(f) for f in sum(fe.values(), [])) + 2
    emit(f"   {'pos':<4}" + "".join(f"{'model ' + k + ' (' + nm + ')':<{width + 8}}" for k, nm, _, _ in GROUPS))
    for i in range(max(len(v) for v in fe.values())):
        emit(f"   {i + 1:<4}" + "".join(f"{(fe[k][i] if i < len(fe[k]) else '-'):<{width + 8}}" for k, *_ in GROUPS))
    emit(f"   A == C: {fe['A'] == fe['C']};  B == A + ['from_state']: {fe['B'] == fe['A'] + ['from_state']}")

    # ------------------------------------------------------------------ the three loans
    title("6. THE CHECK THAT CATCHES MISALIGNMENT: hand-computed row vs the row the forecaster wrote")
    pick = {}
    for g, states in (("Current", ["Current"]), ("DQ30/DQ60", ["DQ30", "DQ60"]), ("DQ90+", ["DQ90+"])):
        i = int(np.flatnonzero(np.isin(batch_state, states))[0])
        pick[g] = i
    extra_state = "DQ60" if batch_state[pick["DQ30/DQ60"]] == "DQ30" else "DQ30"
    extra = int(np.flatnonzero(batch_state == extra_state)[0])
    chosen = [("Current", pick["Current"], "A", False), ("DQ30/DQ60", pick["DQ30/DQ60"], "B", False), ("DQ90+", pick["DQ90+"], "C", False),
              (f"{extra_state} (ADDITIONAL: the other origin state of model B, to exercise from_state both ways)", extra, "B", True)]
    cond = {}
    results = []
    for label, i, key, additional in chosen:
        lid_ = batch_ids[i]
        o = batch_state[i]
        q = "20" + lid_[1:3] + lid_[3:5]                              # 'F06Q1...' -> '2006Q1'
        if q not in cond:
            cond[q] = panel_io.load_panel(roles=["condition"], quarters=[q],
                                          columns=["loan_id", "from_period", "from_state", "fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv",
                                                   "unemp_level", "unemp_chg_12m"])
        d = cond[q]
        row = d[(d["loan_id"].astype(str) == lid_) & (d["from_period"] == forecast.CUT_YYYYMM)]
        assert len(row) == 1 and str(row["from_state"].iloc[0]) == o, (lid_, len(row))
        # model and sidecar loaded independently of src.forecast
        mdir = os.path.join(MODEL_ROOT, GROUPS[["A", "B", "C"].index(key)][3])
        with open(os.path.join(mdir, "model.json"), encoding="utf-8") as fh:
            sc = json.load(fh)
        bst = lgb.Booster(model_file=os.path.join(mdir, "model.txt"))
        cols = {}
        for c in sc["features"]:
            if c == "property_state":
                cols[c] = pd.Categorical([str(row["property_state"].iloc[0])], categories=sc["categories"]["property_state"])
            elif c == "from_state":
                cols[c] = pd.Categorical([o], categories=sc["categories"]["from_state"])
            else:
                cols[c] = row[c].to_numpy(dtype=np.float32, na_value=np.nan)
        X = pd.DataFrame(cols)
        assert list(X.columns) == sc["features"]
        out = bst.predict(X)
        assert out.shape == (1, len(sc["class_order"])), out.shape
        raw6 = np.zeros(6)
        for ci, cname in enumerate(sc["class_order"]):                # map through the sidecar's class order into panel order
            raw6[STATES.index(cname)] = out[0, ci]
        hand = raw6.copy()
        zeroed = [c for c in STATES if sc["label_counts"][c] == 0]
        for c in zeroed:
            hand[STATES.index(c)] = 0.0                               # absent classes: EXACT ZERO, no renormalisation
        j = order[lid_]
        h0, h1 = P32[j, 0].astype(np.float64), P32[j, 1].astype(np.float64)
        d64 = np.abs(hand - h1)
        d32 = np.abs(hand.astype(np.float32).astype(np.float64) - h1)
        used = frame_log and [r for r in frame_log[:3] if o in r["origins"]][0]
        results.append(dict(label=label, i=i, key=key, additional=additional, loan=lid_, origin=o, model=models[o], hand=hand, raw6=raw6, h0=h0, h1=h1, d64=d64, d32=d32,
                            zeroed=zeroed, frame=used, side=sc, features=X.columns.tolist(), rowsum=float(hand.sum())))
    for r in results:
        emit()
        emit(f"--- {r['label']}")
        emit(f"loan_id {r['loan']}   origin state at the cut: {r['origin']}   dispatched to: {r['model'].name} (model {r['key']}), whose origin states are {r['model'].origin_states}"
             f"   -> {'ROUTED CORRECTLY' if r['model'].name == GROUPS[['A', 'B', 'C'].index(r['key'])][3] and r['origin'] in r['model'].origin_states else '!!! ROUTED TO THE WRONG MODEL !!!'}")
        emit(f"feature vector (sidecar order, hand-built): {r['features']}")
        emit(f"frame the forecaster built for this model: columns {r['frame']['columns']}; origin blocks {r['frame']['origins']}")
        emit(f"{'':<34}" + "".join(f"{s:>26}" for s in STATES))
        emit(f"{'forecaster h=0 (initial vector)':<34}" + "".join(f"{g17(v):>26}" for v in r["h0"]))
        emit(f"{'forecaster h=1 (stored float32)':<34}" + "".join(f"{g17(v):>26}" for v in r["h1"]))
        emit(f"{'hand-computed (float64)':<34}" + "".join(f"{g17(v):>26}" for v in r["hand"]))
        emit(f"{'raw predict before zero-fill':<34}" + "".join(f"{g17(v):>26}" for v in r["raw6"]))
        emit(f"{'max |hand - stored|, float64':<34}" + "".join(f"{g17(v):>26}" for v in r["d64"]))
        emit(f"{'max |hand - stored|, float32-cast':<34}" + "".join(f"{g17(v):>26}" for v in r["d32"]))
        emit(f"hand row sums to {r['rowsum']:.17g} (zero-filled classes {r['zeroed'] or 'none'}); stored h=1 row sums to {r['h1'].sum():.17g}")
        bad = [STATES[k] for k in range(6) if r["d32"][k] != 0.0]
        r["bad"] = bad
        emit("RESULT: " + ("PASS (every class identical after the float32 cast)" if not bad else f"FAIL: classes that differ: {bad}"))
    write_report()

    # ------------------------------------------------------------------ the other verifications
    title("7. ALSO VERIFIED  (all 3,000 loans, every h = 0..%d)" % (H - 1))
    P64 = P32.astype(np.float64)
    dev32 = float(np.abs(P64.sum(axis=2) - 1.0).max())
    emit(f"- sum to one: worst |sum - 1| computed by the forecaster in float64 in memory (its check, from the batch marker): {marker['checks']['worst_sum_deviation']:.3e}  (SUM_TOL {forecast.SUM_TOL:.0e})")
    emit(f"              worst |sum - 1| in the STORED float32 values, summed in float64 (recomputed here from the parquet): {dev32:.3e}  (STORED_TOL {forecast.STORED_TOL:.0e}: "
         f"{'within' if dev32 <= forecast.STORED_TOL else '*** EXCEEDS ***'}; tolerance not loosened). The forecaster's own stored-float32 figure: {marker['checks']['worst_sum_deviation_f32']:.3e}.")
    emit(f"- no negative probability: smallest stored probability {g17(P32.min())} -> {'PASS' if P32.min() >= 0 else 'FAIL'}; all finite: {bool(np.isfinite(P32).all())}")
    dpp = np.diff(P64[:, :, 4:], axis=1).min()
    emit(f"- Prepaid and Default mass never falls across h: most negative one-step change over {len(uid):,} loans x {H - 1} steps x 2 classes (stored float32): {g17(dpp)} -> "
         f"{'PASS' if dpp >= 0 else 'FAIL'}")
    emit()
    emit("- classes a model never learned: the value the forecaster stored at h = 1 for the loans routed to that model (exact float32 value, widened to float64)")
    zero_fail = False
    zero_checked = False
    for key, nm, origin, name in GROUPS:
        for c in absent[key]:
            zero_checked = True
            m_rows = np.isin(batch_state, origin)
            vals = P32[np.array([order[l] for l in batch_ids[m_rows]]), 1, STATES.index(c)]
            n_exact = int((vals == 0.0).sum())
            emit(f"   model {key} ({nm}) never saw {c:<8}: {int(m_rows.sum()):>5,} loans routed; stored h=1 value min {g17(vals.min())}, max {g17(vals.max())}; "
                 f"exactly 0.0 for {n_exact:,} of {len(vals):,}  -> {'PASS' if n_exact == len(vals) else 'FAIL'}")
            fr_rec = frame_log[[g[0] for g in GROUPS].index(key)]
            f64 = []
            for o in origin:
                kb = fr_rec["origins"].index(o)
                sel = np.flatnonzero(batch_state == o)
                f64.append(fr_rec["booster"].predict(fr_rec["frame"])[kb * len(batch_ids) + sel, STATES.index(c)])
            f64 = np.concatenate(f64)
            und = int(((vals == 0.0) & (f64 > 0.0)).sum())
            emit(f"      the same loans' float64 predict() value (before the float32 cast): min {g17(f64.min())}, max {g17(f64.max())}; "
                 f"{und:,} loan(s) are stored as exactly 0.0 ONLY because their float64 value is below the float32 range (underflow in the write), not because anything zero-filled them; "
                 f"{int((f64 > 0.0).sum()):,} of {len(f64):,} are positive in float64")
            if n_exact != len(vals):
                zero_fail = True
    if not zero_checked:
        emit("   no model was missing a class: this check could not be run.")
    emit()
    emit("- from_state (landmine 2), resolved from the sidecars and from the frames the forecaster built:")
    l2_ok = True
    for (key, nm, origin, name), r in zip(GROUPS, frame_log[:3]):
        s = side[key]
        has = "from_state" in r["columns"]
        want = len(s["origin_states"]) > 1
        ok = has == want and s["from_state_is_feature"] == want and ("from_state" in s["features"]) == want
        if has:
            ok = ok and r["from_state_categories"] == s["categories"]["from_state"] == s["origin_states"] and \
                 r["from_state_blocks"] == [[o] for o in r["origins"]]
        l2_ok &= ok
        emit(f"   model {key} ({nm}): origin states {s['origin_states']} (n = {len(s['origin_states'])}); sidecar from_state_is_feature {s['from_state_is_feature']}; "
             f"from_state in sidecar features {'from_state' in s['features']}; PASSED TO THE MODEL BY THE FORECASTER: {has}"
             + (f" (categories {r['from_state_categories']}; per-origin blocks hold {r['from_state_blocks']})" if has else "") + f"  -> {'consistent' if ok else '!!! INCONSISTENT !!!'}")
    emit()
    emit("- separate matrix per group (landmine 3):")
    l3_ok = len({r["id"] for r in frame_log[:3]}) == 3
    for (key, nm, *_), r in zip(GROUPS, frame_log[:3]):
        same = r["columns"] == r["model_features"] == side[key]["features"]
        l3_ok &= same
        emit(f"   model {key}: frame object {r['id']:#x}, {r['rows']:,} rows x {len(r['columns'])} columns; columns == the model's own feature list == sidecar order: {same}")
    emit(f"   three distinct frame objects: {len({r['id'] for r in frame_log[:3]}) == 3}")
    emit("   LIMITATION of this test for landmine 3: models A and C carry the SAME feature order, so a bug that reused A's matrix for C would not show as a column mismatch; "
         "only model B's differing order (one extra trailing column) and the per-model hand comparison in section 6 can expose reuse across A/B or B/C.")

    # ------------------------------------------------------------------ findings
    watch.stop()
    title("FINDINGS  (PASS / FAIL per landmine; nothing reconciled, no tolerance adjusted, src/ untouched)")
    learned_bad = [(r["label"].split(" ")[0], [c for c in r["bad"] if c not in r["zeroed"]]) for r in results if [c for c in r["bad"] if c not in r["zeroed"]]]
    zeroed_bad = [(r["label"].split(" ")[0], [c for c in r["bad"] if c in r["zeroed"]]) for r in results if [c for c in r["bad"] if c in r["zeroed"]]]
    routed_ok = all(r["model"].name == GROUPS[["A", "B", "C"].index(r["key"])][3] and r["origin"] in r["model"].origin_states for r in results)
    if not zero_checked:
        l1a = "NOT EXERCISED: every model learned all six classes, so the zero-fill path was not run"
    else:
        l1a = "PASS" if not zero_fail and not zeroed_bad else "FAIL"
    l1b = "PASS" if not learned_bad else "FAIL"
    emit(f"LANDMINE (1) CLASS ORDER / ZERO-FILL: " + ("PASS" if (l1a == "PASS" and l1b == "PASS") else "FAIL" if "FAIL" in (l1a, l1b) else f"{l1b} on column mapping; zero-fill {l1a}"))
    emit(f"   column mapping on the learned classes (hand row through the sidecar class order vs the stored row, float32-exact): {l1b}"
         + (f"  differing classes: {learned_bad}" if learned_bad else ""))
    emit(f"   exact-zero fill of classes a model never learned: {l1a}")
    for key, nm, origin, name in GROUPS:
        for c in absent[key]:
            m_rows = np.isin(batch_state, origin)
            vals = P32[np.array([order[l] for l in batch_ids[m_rows]]), 1, STATES.index(c)]
            emit(f"      model {key} never saw {c}: stored h=1 value ranges {g17(vals.min())} .. {g17(vals.max())} (not exactly 0.0 for {int((vals != 0.0).sum()):,} of {len(vals):,} loans)")
    if zero_checked and (zero_fail or zeroed_bad):
        emit("   DEFECT, precisely: src/forecast.py (Model.__init__ :123-129, propagate :283-285) requires six columns in panel order and writes the booster's output as it comes; it has no "
             "step that sets a never-learned class to exact zero. LightGBM declares num_class = 6, so an unseen class is still emitted with a small positive softmax value, "
             "which the forecaster stores and chains. Not fixed here, by instruction.")
    emit(f"LANDMINE (2) from_state FLAG: {'PASS' if l2_ok else 'FAIL'}   (passed to model B, the two-origin model; omitted for A and C; resolved from each sidecar and checked against the frame the forecaster built, "
         f"including the per-origin block contents)")
    emit(f"LANDMINE (3) FEATURE ORDER / per-group matrix: {'PASS' if l3_ok else 'FAIL'}   (three distinct frames, each in its own model's sidecar order; A and C orders are identical, B has from_state last; "
         f"see the stated limitation in section 7)")
    emit(f"ROUTING (loan -> model): {'PASS' if routed_ok else 'FAIL'} for all {len(results)} compared loans")
    emit(f"h = 0 caveat: the brief asks for the comparison at h = 0; the forecaster's h = 0 is the initial one-hot vector, so the model-output comparison was made at h = 1 (h = 0 printed alongside).")
    emit()
    emit("THE COMPARISON ROWS IN FULL (columns: " + ", ".join(STATES) + ")")
    for r in results:
        emit(f"{r['label'].split(' ')[0]:<10}{r['loan']}  origin {r['origin']}  -> {r['model'].name}")
        emit("   hand-computed : " + "  ".join(g17(v) for v in r["hand"]))
        emit("   forecaster h=1: " + "  ".join(g17(v) for v in r["h1"]))
        emit("   forecaster h=0: " + "  ".join(g17(v) for v in r["h0"]))
        emit("   max |diff| per class (float32-cast): " + "  ".join(g17(v) for v in r["d32"]) + f"   -> {'PASS' if not r['bad'] else 'FAIL ' + str(r['bad'])}")
    emit()
    emit(f"WINDOW FACTS (census, section 1): DQ90+ -> Prepaid transitions: {int(c7[ip]):,} at from_period <= 200706, {int(c6[ip]):,} at <= 200806 - the 11 Oct expectation of none does not hold. "
         f"train_ml has no lower from_period bound: model A's rows reach back to {census['A']['fp_min']}.")
    emit("Checks that could not be run: " + ("none." if zero_checked else "the exact-zero check of never-learned classes (every model learned all six)."))
    emit(f"Model quality was not evaluated or reported; the models are 5 rounds on at most about {format(CAP_ROWS, ',')} rows each and mean nothing.")
    emit(f"PEAK RSS (src.memwatch): process peak {watch.peak_rss() / GB:.2f} GB; lowest system-available memory seen {watch.min_avail / GB:.2f} GB; total wall-clock {time.perf_counter() - T0:.1f} s")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
