"""LightGBM multiclass: where does the model start BEFORE any tree is added, with init_score off and on?   MEASUREMENT ONLY.

    python diagnostics/46_init_start_point.py   # from the repo root

A measurement of library behaviour (lightgbm as installed), not a modelling experiment: no tuning, no features added, no
modelling conclusion drawn. Writes diagnostics/46_init_start_point.txt and nothing else: no artifacts/ml/ model, nothing
under src/ or data/, no git command. Fit rows only; evaluate rows are never read.

Reuses diagnostics/44_capacity_brier.py (imported as a module): its load() (the panel is read through
panel_io.load_panel, one partition at a time), populations, selection window (role == 'fit', 200601 <= from_period <=
200706), by-loan 80/20 split with numpy default_rng(SEED=40) on the sorted unique loan codes (the split block is copied
from run_population() verbatim, then checked against diagnostic 44's own report: six train counts, six held-out counts
and the prior-only Brier must match, else the script stops), feature set, label mapping, class order (panel.STATES),
native categoricals, and the LightGBM params dict (objective multiclass, num_class 6, device cpu, defaults otherwise).
The one deviation from 44's params: verbosity 0 instead of -1, so that library warnings are produced; a logger is registered
that captures every library message (Python and C++), and each run's messages are reported. Logging does not change the model.

RUNS. For each population (CURRENT, POOLED) x config (init OFF, init ON) x num_boost_round in {0, 1, 2}:
  init OFF : no init_score; the library default (boost_from_average=true).
  init ON  : Dataset init_score = the six TRAIN-row class frequencies as log-odds (log f_j), an (n_train, 6) float64 array.
Everything is measured on the HELD-OUT rows: mean predicted probability per class, mean RAW score per class (predict(raw_score=
True), before the softmax), Brier score (diagnostic 44's explicit formula: sum over the six classes of the squared error
against the one-hot vector, averaged over rows, range 0..2). Predictions come from a booster re-loaded from the model string,
which is what lgb.train returns by default (keep_training_booster=False); the training booster is kept only to read its
internal training-set score.

ZERO ROUNDS. lgb.train() refuses num_boost_round=0 (ValueError in this LightGBM version; the message is captured and reported).
The round-0 model is therefore the booster that lgb.train would start from: lgb.Booster(params, train_set=Dataset) with no
update() call. That substitution is reported where it is used.

TRAINING-SIDE EVIDENCE. predict() is separate from what the training objective saw. The booster's internal training-set score
(Booster.__inner_predict, a private LightGBM method, class-averaged here) is the score vector the objective's gradients are
computed from; it is read after the run. The Dataset's own C-side init_score field (Dataset.get_field) is read after
construct(). ADDED PROBE (round 1 only): a Dataset whose init_score is all zeros, to see whether the first trees change when the
init values change.

Peak RSS comes from src.memwatch.MemWatch around each run (Dataset construction, training, prediction).
"""

import gc
import importlib.util
import os
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.getcwd())
from src.memwatch import MemWatch                                       # noqa: E402


def _load_diag44():
    spec = importlib.util.spec_from_file_location("diag44", os.path.join("diagnostics", "44_capacity_brier.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D44 = _load_diag44()
STATES, BASE_FEATURES, CATEGORIES = D44.STATES, D44.BASE_FEATURES, D44.CATEGORIES
POPULATIONS, SEED, TRAIN_FRAC = D44.POPULATIONS, D44.SEED, D44.TRAIN_FRAC
WIN_LO, WIN_HI = D44.WIN_LO, D44.WIN_HI
BRIER_FORMULA, brier, counts = D44.BRIER_FORMULA, D44.brier, D44.counts

OUT = os.path.join("diagnostics", "46_init_start_point.txt")
REF44 = os.path.join("diagnostics", "44_capacity_brier.txt")
PARAMS = dict(D44.BASE_PARAMS, verbosity=0)
ROUNDS = [0, 1, 2]
SHOW_FROM_MODEL = ["boost_from_average", "learning_rate", "num_leaves", "min_data_in_leaf", "max_bin", "num_threads", "seed",
                   "deterministic"]
GB = 2 ** 30
RULE_W = 112
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def title(text):
    emit()
    emit("=" * RULE_W)
    emit(text)
    emit("=" * RULE_W)


def write_report():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")


def on_abort(msg):
    emit("ABORTED, run did not complete: " + msg)
    write_report()


class Capture:
    """Receives every message LightGBM logs (C++ and Python), at either level."""

    def __init__(self):
        self.msgs = []

    def info(self, m):
        self.msgs.append(str(m))

    def warning(self, m):
        self.msgs.append(str(m))


def softmax(Z):
    """Row-wise softmax of an (n, 6) array; returns a new float64 array."""
    Z = np.asarray(Z, dtype=np.float64)
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def ref44_split(pop):
    """Diagnostic 44's own report: (train counts, held-out counts, prior-only held-out Brier) for this population."""
    with open(REF44, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("POPULATION %s:" % pop))
    got = {}
    for k in range(i, len(lines)):
        for side in ("train", "held-out"):
            if lines[k].startswith(side + ": ") and side not in got:
                nums = re.findall(r"(?<=\s)\d[\d,]*(?=\s|$)", lines[k + 1])
                got[side] = [int(s.replace(",", "")) for s in nums]
        m = re.match(r"\s+Brier\s+: train [\d.]+\s+held-out ([\d.]+)", lines[k])
        if m and "brier" not in got:
            got["brier"] = float(m.group(1))
        if len(got) == 3:
            return got["train"], got["held-out"], got["brier"]
    raise RuntimeError("could not parse the split lines of %s in %s" % (pop, REF44))


def build_dataset(lgb, Xtr, feats, categorical, ytr, init):
    return lgb.Dataset(Xtr[feats], label=ytr, categorical_feature=categorical, free_raw_data=True,
                       init_score=init, params=PARAMS)


def model_params(booster):
    block = booster.model_to_string().split("parameters:")[-1]
    eff = {}
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]") and ": " in line:
            k, v = line[1:-1].split(": ", 1)
            eff[k] = v
    return eff


def run_one(lgb, cap, kind, rounds, Xtr, ytr, Xte, yte, feats, categorical, init_row, label):
    """kind: 'OFF' (no init_score), 'ON' (log train frequencies) or 'PROBE' (all-zero init_score).
    Returns a dict of measurements, plus the held-out raw score array under '_raw' (dropped by the caller)."""
    n_tr, K = len(ytr), len(STATES)
    r = {"kind": kind, "rounds": rounds, "errors": [], "label": label}
    init = None
    if kind == "ON":
        init = np.tile(init_row, (n_tr, 1))
    elif kind == "PROBE":
        init = np.zeros((n_tr, K))
    r["passed_shape"] = None if init is None else tuple(init.shape)
    r["passed_dtype"] = None if init is None else str(init.dtype)
    print(f"  [{label}] ...", flush=True)
    watch = MemWatch(on_abort=on_abort)
    watch.phase = label
    watch.start()
    t0 = time.perf_counter()
    cap.msgs.clear()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ds = build_dataset(lgb, Xtr, feats, categorical, ytr, init)
        ds.construct()
        # --- what the Dataset (C side) holds
        fi = ds.get_field("init_score")
        r["ds_num_data"] = ds.num_data()
        if fi is None:
            r["field"] = None
        else:
            fi = np.asarray(fi)
            f2 = fi.reshape(n_tr, K) if fi.size == n_tr * K else None
            r["field"] = {"shape": tuple(fi.shape), "dtype": str(fi.dtype), "size": int(fi.size),
                          "max_abs_vs_passed": (float(np.abs(f2 - init).max()) if f2 is not None else None),
                          "row0": [float(v) for v in fi.reshape(-1, K)[0]] if f2 is not None else None}
            del f2
        del fi
        # --- training
        if rounds == 0:
            try:
                lgb.train(PARAMS, ds, num_boost_round=0, keep_training_booster=True)
                r["train0_error"] = None
            except Exception as e:                                      # reported, not worked around silently
                r["train0_error"] = "%s: %s" % (type(e).__name__, str(e).rstrip("."))
            booster = lgb.Booster(params=PARAMS, train_set=ds)          # the round-0 model: no update() call
        else:
            booster = lgb.train(PARAMS, ds, num_boost_round=rounds, keep_training_booster=True)
        r["current_iteration"], r["num_trees"] = booster.current_iteration(), booster.num_trees()
        r["eff"] = model_params(booster)
        # --- the training objective's own view: the booster's internal training-set score (probabilities, class-major)
        try:
            sp = booster._Booster__inner_predict(data_idx=0)
            r["train_mean_prob"] = sp.mean(axis=0)
            r["train_rowdiff0"] = (np.log(sp[0] / sp[0][0]) if sp[0].min() > 0 else None)   # logit differences vs class 0, row 0
            r["train_prob_row_spread"] = float((sp.max(axis=0) - sp.min(axis=0)).max())
            del sp
        except Exception as e:
            r["train_mean_prob"] = None
            r["errors"].append("inner training score unavailable: %s: %s" % (type(e).__name__, e))
        # --- predictions, from the model-string booster (what lgb.train returns by default)
        try:
            served = lgb.Booster(model_str=booster.model_to_string())
            r["served_from"] = "model string reload"
        except Exception as e:
            served = booster
            r["served_from"] = "training booster (model-string reload failed: %s: %s)" % (type(e).__name__, e)
        assert booster.feature_name() == feats
        assert booster.pandas_categorical == [CATEGORIES[c] for c in categorical]
        raw = prob = None
        try:
            raw = np.asarray(served.predict(Xte[feats], raw_score=True), dtype=np.float64)
            prob = np.asarray(served.predict(Xte[feats]), dtype=np.float64)
        except Exception as e:
            r["errors"].append("predict failed: %s: %s" % (type(e).__name__, e))
        del served, booster, ds
        gc.collect()
    watch.stop()
    r["secs"] = time.perf_counter() - t0
    r["rss"], r["peak"], r["min_avail"] = watch._peak, watch.peak_rss(), watch.min_avail
    r["lgb_msgs"] = list(cap.msgs)
    r["py_warnings"] = ["%s: %s" % (w.category.__name__, w.message) for w in wlist]
    if raw is not None:
        assert raw.shape == (len(yte), K) and prob.shape == (len(yte), K), (raw.shape, prob.shape)
        r["mean_raw"], r["mean_prob"] = raw.mean(axis=0), prob.mean(axis=0)
        r["brier"] = brier(prob, yte)
        r["softmax_raw_vs_predict"] = float(np.abs(softmax(raw) - prob).max())
        r["raw_row_spread"] = float((raw.max(axis=0) - raw.min(axis=0)).max())
        if kind == "ON":
            raw_a = raw + init_row
            prob_a = softmax(raw_a)
            r["mean_raw_added"], r["mean_prob_added"] = raw_a.mean(axis=0), prob_a.mean(axis=0)
            r["brier_added"] = brier(prob_a, yte)
            r["_raw_added"], r["_prob_added"] = raw_a, prob_a
        r["_raw"], r["_prob"] = raw, prob
    return r


def maxdiff(a, b):
    return float(np.abs(a - b).max())


def fmt6(v, w=14, p=10):
    return "".join(f"{x:>{w}.{p}f}" for x in v)


def emit_run(r, pr_bs_te, freq_tr, freq_te):
    kind, rounds = r["kind"], r["rounds"]
    on = kind == "ON"
    emit()
    emit(f"  --- init {kind}, num_boost_round = {rounds} ---")
    for e in r["errors"]:
        emit("  ERROR: " + e)
    if rounds == 0:
        emit(f"  lgb.train(num_boost_round=0) -> {r['train0_error'] or 'accepted'}; the round-0 model here is "
             f"lgb.Booster(params, train_set=Dataset) with no update() (trees {r['num_trees']}, current_iteration {r['current_iteration']})")
    else:
        emit(f"  lgb.train(num_boost_round={rounds}) ran: trees {r['num_trees']} ({r['num_trees'] // len(STATES)} per class), "
             f"current_iteration {r['current_iteration']}")
    emit("  model parameters block: " + ", ".join(f"{k}={r['eff'].get(k, '?')}" for k in SHOW_FROM_MODEL))
    emit(f"  predictions served from: {r.get('served_from')};  wall-clock {r['secs']:.1f} s;  peak RSS in-run sampled "
         f"{r['rss'] / GB:.2f} GB, process peak {r['peak'] / GB:.2f} GB")
    emit("  library messages: " + (" | ".join(r["lgb_msgs"]) if r["lgb_msgs"] else "none")
         + ";  Python warnings: " + (" | ".join(r["py_warnings"]) if r["py_warnings"] else "none"))
    if "mean_raw" not in r:
        emit("  (no held-out predictions for this run)")
        return
    head = f"  {'class':<10}{'train freq':>16}{'held-out freq':>16}{'mean pred prob':>16}{'mean raw':>16}"
    if on:
        head += f"{'prob +init':>16}{'raw +init':>16}"
    emit("  HELD-OUT mean per class: predict() as returned" + ("  |  with the init added back manually (softmax(raw + log f))" if on else ""))
    emit(head)
    for j, s in enumerate(STATES):
        row = f"  {s:<10}{freq_tr[j]:>16.10f}{freq_te[j]:>16.10f}{r['mean_prob'][j]:>16.10f}{r['mean_raw'][j]:>16.8f}"
        if on:
            row += f"{r['mean_prob_added'][j]:>16.10f}{r['mean_raw_added'][j]:>16.8f}"
        emit(row)
    emit(f"  Brier (held-out), as returned          : {r['brier']:.7f}   (prior-only reference {pr_bs_te:.7f}; difference {r['brier'] - pr_bs_te:+.7f})")
    if on:
        emit(f"  Brier (held-out), with init added back : {r['brier_added']:.7f}   (difference from prior-only {r['brier_added'] - pr_bs_te:+.7f})")
    emit(f"  consistency: max |softmax(raw as returned) - predict() probabilities| = {r['softmax_raw_vs_predict']:.3e};  "
         f"largest across-row spread of any class's raw score = {r['raw_row_spread']:.3e}")
    if r.get("train_mean_prob") is not None:
        emit("  booster's internal TRAINING-set score (mean probability per class over train rows): "
             + "  ".join(f"{s} {v:.10f}" for s, v in zip(STATES, r["train_mean_prob"])))


def run_population(lgb, cap, pop, origin, expected, Xall, yall, call):
    K = len(STATES)
    in_pop = Xall["from_state"].isin(origin).to_numpy()
    feats = list(BASE_FEATURES) + (["from_state"] if len(origin) > 1 else [])
    categorical = ["property_state"] + (["from_state"] if len(origin) > 1 else [])
    X, y, codes = Xall[in_pop].reset_index(drop=True), yall[in_pop], call[in_pop]
    title(f"POPULATION {pop}: role == 'fit', from_state in {origin}, {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"rows {len(y):,} (expected {expected:,}: {'MATCH' if len(y) == expected else 'DIFFERENT'})")
    emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")

    # --- split, copied from diagnostic 44 run_population()
    uniq, inv = np.unique(codes, return_inverse=True)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool)
    flag[perm[:n_tr]] = True
    in_train = flag[inv]
    assert np.intersect1d(np.unique(codes[in_train]), np.unique(codes[~in_train])).size == 0, "a loan is on both sides"
    Xtr, ytr = X[in_train].reset_index(drop=True), y[in_train]
    Xte, yte = X[~in_train].reset_index(drop=True), y[~in_train]
    del X, y, codes, inv
    gc.collect()
    emit(f"split by loan_id: seed {SEED}, train fraction {TRAIN_FRAC} -> {n_tr:,} of {len(uniq):,} loans to train, "
         f"{len(uniq) - n_tr:,} held out; train rows {len(ytr):,}, held-out rows {len(yte):,}; no loan on both sides (asserted)")

    c_tr, c_te = counts(ytr), counts(yte)
    freq_tr, freq_te = c_tr / len(ytr), c_te / len(yte)
    Ppr_te = np.broadcast_to(freq_tr, (len(yte), K))
    Ppr_tr = np.broadcast_to(freq_tr, (len(ytr), K))
    pr_bs_te, pr_bs_tr = brier(Ppr_te, yte), brier(Ppr_tr, ytr)

    r_tr, r_te, r_bs = ref44_split(pop)
    same = list(map(int, c_tr)) == r_tr and list(map(int, c_te)) == r_te and abs(pr_bs_te - r_bs) < 1e-6
    emit(f"REPRODUCTION CHECK vs diagnostic 44's report: train counts {list(map(int, c_tr))} vs {r_tr}; held-out counts "
         f"{list(map(int, c_te))} vs {r_te}; prior-only held-out Brier {pr_bs_te:.7f} vs {r_bs:.7f} -> "
         f"{'MATCH' if same else '!!! DO NOT MATCH !!!'}")
    if not same:
        emit("STOP: the split/population does not reproduce diagnostic 44. Nothing further is run; this mismatch is the finding.")
        write_report()
        sys.exit(1)
    assert (freq_tr > 0).all(), "a class has zero train frequency: its log-odds would be -inf"
    init_row = np.log(freq_tr)

    emit()
    emit("CLASS FREQUENCIES (class order = panel.STATES)")
    emit(f"  {'class':<10}{'TRAIN count':>14}{'TRAIN freq':>16}{'HELD-OUT count':>16}{'HELD-OUT freq':>16}{'init value log f':>20}")
    for j, s in enumerate(STATES):
        emit(f"  {s:<10}{int(c_tr[j]):>14,}{freq_tr[j]:>16.10f}{int(c_te[j]):>16,}{freq_te[j]:>16.10f}{init_row[j]:>20.8f}")
    emit(f"PRIOR-ONLY REFERENCE (predict the TRAIN frequencies for every row): Brier held-out {pr_bs_te:.7f}   (train rows {pr_bs_tr:.7f})")
    emit("BRIER: " + BRIER_FORMULA)

    runs = {}
    cmp_rows = []
    for rounds in ROUNDS:
        off = run_one(lgb, cap, "OFF", rounds, Xtr, ytr, Xte, yte, feats, categorical, init_row, f"{pop} OFF r{rounds}")
        on = run_one(lgb, cap, "ON", rounds, Xtr, ytr, Xte, yte, feats, categorical, init_row, f"{pop} ON r{rounds}")
        c = {"rounds": rounds}
        if "_raw" in off and "_raw" in on:
            c["raw_equal"] = bool(np.array_equal(off["_raw"], on["_raw"]))
            c["raw_maxdiff"] = maxdiff(off["_raw"], on["_raw"])
            c["prob_maxdiff"] = maxdiff(off["_prob"], on["_prob"])
            c["raw_maxdiff_added"] = maxdiff(off["_raw"], on["_raw_added"])
            c["prob_maxdiff_added"] = maxdiff(off["_prob"], on["_prob_added"])
            c["brier_off"], c["brier_on"], c["brier_on_added"] = off["brier"], on["brier"], on["brier_added"]
        cmp_rows.append(c)
        if rounds == 1:
            probe = run_one(lgb, cap, "PROBE", 1, Xtr, ytr, Xte, yte, feats, categorical, init_row, f"{pop} PROBE r1")
            if "_raw" in probe and "_raw" in on:
                probe["vs_on_raw_maxdiff"] = maxdiff(probe["_raw"], on["_raw"])
                probe["vs_on_prob_maxdiff"] = maxdiff(probe["_prob"], on["_prob"])
            runs["PROBE"] = probe
        runs[("OFF", rounds)], runs[("ON", rounds)] = off, on
        for rr in (off, on, runs.get("PROBE") if rounds == 1 else None):
            if rr is not None:
                for k in [k for k in rr if k.startswith("_")]:
                    del rr[k]
        gc.collect()

    for kind in ("OFF", "ON"):
        title(f"{pop} / init {kind}" + ("  (no init_score; boost_from_average default)" if kind == "OFF"
                                         else "  (Dataset init_score = log TRAIN class frequencies)"))
        for rounds in ROUNDS:
            emit_run(runs[(kind, rounds)], pr_bs_te, freq_tr, freq_te)
    return {"pop": pop, "runs": runs, "cmp": cmp_rows, "freq_tr": freq_tr, "freq_te": freq_te, "init_row": init_row,
            "pr_bs_te": pr_bs_te, "n_tr": len(ytr)}


def critical_section(res):
    title("CRITICAL CHECK: does predict() include the init_score contribution, and did the training objective see it?")
    for R in res:
        pop, runs, init_row = R["pop"], R["runs"], R["init_row"]
        emit()
        emit(f"--- {pop} ---")
        emit("(1) Was the init_score array accepted by the Dataset?  (inspected: the Dataset's C-side field after construct())")
        for kind in ("ON", "OFF"):
            r = runs[(kind, 0)]
            f = r["field"]
            if kind == "ON":
                emit(f"    init ON : passed shape {r['passed_shape']} dtype {r['passed_dtype']}; Dataset num_data {r['ds_num_data']:,} (train rows {R['n_tr']:,}); "
                     f"get_field('init_score') -> shape {f['shape']}, dtype {f['dtype']}, size {f['size']:,} "
                     f"(= n_train x 6: {f['size'] == R['n_tr'] * len(STATES)}); max |stored - passed| over all rows/classes = "
                     f"{f['max_abs_vs_passed']:.3e}; stored row 0 = {np.round(f['row0'], 8).tolist()} vs log f = {np.round(init_row, 8).tolist()}")
            else:
                emit(f"    init OFF: get_field('init_score') -> {f}")
        emit("(2) Did the TRAINING side see it?  Booster's internal training-set score at round 0 (zero trees), mean probability per class over train rows:")
        emit(f"    {'class':<10}{'train freq':>16}{'internal, init OFF':>22}{'internal, init ON':>22}")
        off0, on0 = runs[("OFF", 0)], runs[("ON", 0)]
        for j, s in enumerate(STATES):
            a = off0["train_mean_prob"][j] if off0.get("train_mean_prob") is not None else float("nan")
            b = on0["train_mean_prob"][j] if on0.get("train_mean_prob") is not None else float("nan")
            emit(f"    {s:<10}{R['freq_tr'][j]:>16.10f}{a:>22.10f}{b:>22.10f}")
        if on0.get("train_rowdiff0") is not None:
            emit(f"    init ON, train row 0: logit differences log(p_j/p_0) = {np.round(on0['train_rowdiff0'], 6).tolist()} vs init differences "
                 f"{np.round(init_row - init_row[0], 6).tolist()}")
        if off0.get("train_rowdiff0") is not None:
            emit(f"    init OFF, train row 0: logit differences log(p_j/p_0) = {np.round(off0['train_rowdiff0'], 6).tolist()}")
        emit(f"    largest across-row spread of any class's internal training probability, round 0: OFF {off0.get('train_prob_row_spread')}, ON {on0.get('train_prob_row_spread')}")
        emit("    after trees are added (internal training score, mean prob per class):")
        for rounds in (1, 2):
            for kind in ("OFF", "ON"):
                r = runs[(kind, rounds)]
                if r.get("train_mean_prob") is not None:
                    emit(f"      {kind:<4} round {rounds}: " + "  ".join(f"{v:.10f}" for v in r["train_mean_prob"]))
        pb = runs["PROBE"]
        on1 = runs[("ON", 1)]
        emit("    ADDED PROBE: a Dataset init_score of all zeros, 1 round.")
        if "vs_on_raw_maxdiff" in pb:
            emit(f"      probe round-1 predict() raw scores vs init-ON round-1 predict() raw scores: max |difference| = {pb['vs_on_raw_maxdiff']:.6e} "
                 f"(probabilities {pb['vs_on_prob_maxdiff']:.6e})")
        emit(f"      probe round-1 mean raw (as returned): " + "  ".join(f"{v:.8f}" for v in pb["mean_raw"]))
        emit(f"      ON    round-1 mean raw (as returned): " + "  ".join(f"{v:.8f}" for v in on1["mean_raw"]))
        emit("(3) Does predict() reflect it?  OFF vs ON, held-out rows, per round")
        emit(f"    {'rounds':>6}{'raw identical?':>16}{'max|raw OFF-ON|':>18}{'max|prob OFF-ON|':>19}{'max|raw OFF-(ON+init)|':>25}{'max|prob OFF-(ON+init)|':>26}"
             f"{'Brier OFF':>12}{'Brier ON':>12}{'Brier ON+init':>15}")
        for c in R["cmp"]:
            if "raw_equal" in c:
                emit(f"    {c['rounds']:>6}{str(c['raw_equal']):>16}{c['raw_maxdiff']:>18.6e}{c['prob_maxdiff']:>19.6e}{c['raw_maxdiff_added']:>25.6e}"
                     f"{c['prob_maxdiff_added']:>26.6e}{c['brier_off']:>12.7f}{c['brier_on']:>12.7f}{c['brier_on_added']:>15.7f}")
            else:
                emit(f"    {c['rounds']:>6}  comparison unavailable (a run produced no predictions)")


def findings(res):
    title("FINDINGS  (measured only; no interpretation for the ML arm)")
    for R in res:
        pop, runs, init_row = R["pop"], R["runs"], R["init_row"]
        c0 = R["cmp"][0]
        off0, on0 = runs[("OFF", 0)], runs[("ON", 0)]
        emit()
        emit(f"{pop}:")
        emit(f"  - lgb.train(num_boost_round=0): {off0['train0_error'] or 'accepted'}.")
        f = on0["field"]
        emit(f"  - The Dataset accepted the init_score array: stored shape {f['shape']}, max |stored - passed| {f['max_abs_vs_passed']:.1e}; "
             f"init OFF Dataset holds {runs[('OFF', 0)]['field']}.")
        emit(f"  - Round 0, predict() raw scores: OFF mean " + ", ".join(f"{v:.6f}" for v in off0["mean_raw"])
             + "; ON mean " + ", ".join(f"{v:.6f}" for v in on0["mean_raw"]) + f"; identical row for row: {c0.get('raw_equal')}.")
        emit(f"  - Round 0, predict() mean probabilities: OFF " + ", ".join(f"{v:.8f}" for v in off0["mean_prob"])
             + "; ON " + ", ".join(f"{v:.8f}" for v in on0["mean_prob"]) + f"; train frequencies " + ", ".join(f"{v:.8f}" for v in R['freq_tr']) + ".")
        emit(f"  - predict() on the trained/reloaded booster does NOT include the Dataset init_score: at round 0 it returns raw 0 / probability 1/6 for init ON exactly as for init OFF "
             f"(max |raw OFF - ON| {c0.get('raw_maxdiff')}); adding log f back manually turns init ON's round-0 Brier from {on0['brier']:.7f} into {on0['brier_added']:.7f}.")
        emit(f"  - init OFF at round 0 returns the uniform 1/6 start (no boost-from-average contribution, no trees); at rounds 1 and 2 init OFF's predict() raw scores equal "
             f"init ON's raw + log f (max |difference| {R['cmp'][1].get('raw_maxdiff_added', float('nan')):.1e} at round 1, {R['cmp'][2].get('raw_maxdiff_added', float('nan')):.1e} at round 2).")
        emit(f"  - Round 0, Brier: OFF as returned {off0['brier']:.7f}; ON as returned {on0['brier']:.7f}; ON with init added back "
             f"{on0['brier_added']:.7f}; prior-only {R['pr_bs_te']:.7f}.")
        emit(f"  - Booster's internal training score at round 0 (mean prob): OFF " + ", ".join(f"{v:.8f}" for v in off0["train_mean_prob"])
             + "; ON " + ", ".join(f"{v:.8f}" for v in on0["train_mean_prob"]) + ".")
        for c in R["cmp"][1:]:
            if "raw_equal" in c:
                emit(f"  - Round {c['rounds']}: OFF Brier {c['brier_off']:.7f}; ON as returned {c['brier_on']:.7f}; ON with init added back {c['brier_on_added']:.7f}; "
                     f"max |raw OFF - (ON raw + init)| {c['raw_maxdiff_added']:.3e}; max |prob OFF - ON-with-init| {c['prob_maxdiff_added']:.3e}.")
        pb = runs["PROBE"]
        if "vs_on_raw_maxdiff" in pb:
            emit(f"  - Probe (zero init_score, 1 round) vs init ON, round-1 raw scores as returned: max |difference| {pb['vs_on_raw_maxdiff']:.3e}.")
        msgs = sorted({m for k, r in runs.items() for m in (r["lgb_msgs"] if isinstance(r, dict) else [])})
        emit("  - Library messages seen in this population's runs: " + (" | ".join(msgs) if msgs else "none") + ".")


def main():
    import lightgbm as lgb

    cap = Capture()
    lgb.register_logger(cap)
    emit("LightGBM multiclass starting point: init_score OFF vs ON at 0, 1, 2 boosting rounds   (measurement only)")
    X, y, codes, n_null, manifest = D44.load()
    emit(f"panel build {manifest['build_stamp']}   lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs")
    emit(f"rows loaded (all four transient origins, selection window, role == 'fit'): {len(y):,}; null/unknown labels dropped {n_null:,}")
    emit("params passed to lgb.train / Booster / Dataset: " + ", ".join(f"{k}={v!r}" for k, v in PARAMS.items())
         + "  (diagnostic 44's dict, verbosity 0 instead of -1 so that library warnings are produced)")

    res = []
    for pop, origin, expected in POPULATIONS:
        res.append(run_population(lgb, cap, pop, origin, expected, X, y, codes))
    for section in (critical_section, findings):
        try:
            section(res)
        except Exception as e:                                          # keep the per-run blocks already measured
            emit(f"!!! {section.__name__} could not be written: {type(e).__name__}: {e}")
    peak = max(r["peak"] for R in res for r in R["runs"].values())
    inrun = max(r["rss"] for R in res for r in R["runs"].values())
    emit()
    emit(f"PEAK RSS (src.memwatch): largest in-run sampled {inrun / GB:.2f} GB; process peak {peak / GB:.2f} GB")
    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
