"""Diagnostic 57: is the shipped model saturating where a fold model is not?   Writes diagnostics/57_saturation_check.txt.

    python diagnostics/57_saturation_check.py header   # BEFORE any measurement: the pre-registration and what is compared
    python diagnostics/57_saturation_check.py report   # after `python -m src.calibrate refold --model artifacts/ml/headline_pooled`

Model A = artifacts/ml/headline_pooled (the shipped model). Model B = fold 0's model from diagnostic 56. Diagnostic 56 did not persist its fold
boosters, so by explicit instruction `src.calibrate refold` retrained fold 0 ONCE with identical code and arguments and saved it; the report
states how closely that retrained model reproduces diagnostic 56's stored fold-0 predictions. Nothing else is retrained or refitted, no model
artefact is written by this script, no evaluate row is read, no fix is proposed.

Both models score fold 0's held-out fit rows (the identical row set; model B never saw these loans, model A did) with raw_score=True. The
softmax is then taken here, in float64, from those raw scores and checked against LightGBM's own predict() on a slice.
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src import calibrate as C                                          # noqa: E402
from src import panel_io                                                # noqa: E402
from src.train_ml import STATES, TRANSIENT, keep_rows, label_scan       # noqa: E402

A_DIR = os.path.join("artifacts", "ml", "headline_pooled")
CAL = C.cal_dir(A_DIR)
B_PATH = os.path.join(CAL, "oof", "fold0_model.txt")
B_INFO = os.path.join(CAL, "oof", "fold0_model.json")
HEADER = os.path.join(CAL, "diag57_header.txt")
OUT = os.path.join("diagnostics", "57_saturation_check.txt")
SHA_PREFIX = "360761955780a5da"
EXPECT_ROWS = {"Current": 7_859_371, "DQ30": 104_104, "DQ60": 25_000, "DQ90+": 41_036}      # diagnostic 56 section 5, fold 0
SPREAD_T = (20, 50, 100, 1000)
W = 120
K = len(STATES)

PREREG = ("Prediction recorded before the numbers exist: the shipped model shows materially larger raw-score spread than the fold models at "
          "DQ60 and DQ90+, and roughly equal spread at Current. If spreads are similar across all four origin states, the saturation "
          "hypothesis is dead, section 5 of diagnostic 56 remains unexplained, and no further explanation is pursued in this milestone.")

LINES = []


def emit(s=""):
    LINES.append(s)


def rule(ch="-"):
    emit(ch * W)


def title(s):
    emit()
    rule("=")
    emit(s)
    rule("=")


def f(n):
    return format(int(n), ",")


def pct(n, d):
    return "%.4f%%" % (100.0 * n / d) if d else "n/a"


def cnt(n, d):
    """count (share), flagged THIN when 0 < count < 100."""
    return "%s (%s)%s" % (f(n), pct(n, d), " THIN" if 0 < n < 100 else "")


# ==========================================================================
# header
# ==========================================================================
def run_header():
    os.makedirs(CAL, exist_ok=True)
    if os.path.exists(HEADER):
        raise SystemExit("%s exists; the pre-registration is written once and not overwritten" % HEADER)
    sha = C.sha256_file(os.path.join(A_DIR, "model.txt"))
    if not sha.startswith(SHA_PREFIX):
        raise SystemExit("model A sha256 %s does not start with %s" % (sha, SHA_PREFIX))
    LINES.clear()
    rule("=")
    emit("DIAGNOSTIC 57 - IS THE SHIPPED MODEL SATURATING WHERE THE FOLD MODELS ARE NOT?   (read-only on models)")
    rule("=")
    emit()
    emit("PRE-REGISTRATION (written before any measurement was computed)")
    rule()
    emit('"%s"' % PREREG)
    emit()
    emit("WHAT IS COMPARED")
    rule()
    emit("Model A : %s/model.txt   sha256 %s   (prefix %s asserted)" % (A_DIR.replace(os.sep, "/"), sha, SHA_PREFIX))
    emit("Model B : fold 0's model from diagnostic 56 (trained on the 80% of loans not in fold 0).")
    emit("          Diagnostic 56 did not persist its fold boosters. On explicit instruction, `python -m src.calibrate refold` retrained fold 0 once,")
    emit("          with the same code, features, categories, lgb_params and 100 rounds, and saved it. Section A says how closely it reproduces")
    emit("          diagnostic 56's stored fold-0 predictions. This is the only retrain; it is a stand-in for the original booster and is labelled as such.")
    emit("Rows    : fold 0's held-out fit rows (%s). Both models score the identical rows." % " / ".join(
        "%s %s" % (k, f(v)) for k, v in EXPECT_ROWS.items()))
    emit("Measure : LightGBM raw_score=True (six margins per row, before the softmax); softmax taken here in float64.")
    emit("Not done: no retraining with other parameters, no fix, no evaluate rows, no chained forecast, no change to the calibration correction.")
    emit("Order   : this header; A) sha256 and regularisation actually in the model files; B) reference scale; C) spread tables; D) leaf statistics;")
    emit("          E) one loan by hand.")
    text = "\n".join(LINES) + "\n"
    with open(HEADER, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("header written to %s" % HEADER)


# ==========================================================================
# model files
# ==========================================================================
def parse_model(path):
    """The parameters block and every tree's leaf values / leaf row counts / leaf hessian sums, read from the model text file itself."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    pblock = txt[txt.index("parameters:"):txt.index("end of parameters")]
    params = dict(re.findall(r"\[([a-z_0-9]+): ([^\]]*)\]", pblock))
    trees = []
    for blk in re.split(r"\n(?=Tree=\d+\n)", txt[txt.index("Tree=0"):txt.index("end of trees")]):
        g = lambda k: np.array(re.search(r"^%s=(.*)$" % k, blk, re.M).group(1).split(), dtype=np.float64)
        trees.append({"leaf_value": g("leaf_value"), "leaf_count": g("leaf_count"), "leaf_weight": g("leaf_weight"),
                      "num_leaves": int(re.search(r"^num_leaves=(\d+)", blk, re.M).group(1))})
    return {"params": params, "trees": trees, "sha256": C.sha256_file(path)}


def leaf_stats(M):
    trees = M["trees"]
    allv = np.concatenate([t["leaf_value"] for t in trees])
    tree_of = np.concatenate([np.full(len(t["leaf_value"]), i) for i, t in enumerate(trees)])
    cnts = np.concatenate([t["leaf_count"] for t in trees])
    nl = np.array([t["num_leaves"] for t in trees])
    st = {"n_trees": len(trees), "leaves": (int(nl.min()), float(np.median(nl)), int(nl.max())), "n_leaves": int(len(allv)),
          "leaves_lt20": int((cnts < 20).sum()), "tree_max_abs": np.array([np.abs(t["leaf_value"]).max() for t in trees])}
    for th in (1, 5, 10):
        st["trees_gt%d" % th] = int((st["tree_max_abs"] > th).sum())
        st["leaves_gt%d" % th] = int((np.abs(allv) > th).sum())
    top = np.argsort(-np.abs(allv))[:10]
    st["top"] = [(int(tree_of[i]), STATES[int(tree_of[i]) % K], int(tree_of[i]) // K, float(allv[i]), int(cnts[i])) for i in top]
    return st


# ==========================================================================
# scoring
# ==========================================================================
def softmax(raw):
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True), e


def decompose(booster, M, Xs, raw):
    """Per row, how the (max class - min class) margin is built from the 100 boosting iterations (pred_leaf), for the rows Xs."""
    L = booster.predict(Xs, pred_leaf=True).astype(np.int64)
    n, T = L.shape
    vals = np.empty((n, T))
    for t in range(T):
        vals[:, t] = M["trees"][t]["leaf_value"][L[:, t]]
    v = vals.reshape(n, T // K, K)
    chk = float(np.abs(v.sum(axis=1) - raw).max())
    r = np.arange(n)
    cmax, cmin = raw.argmax(axis=1), raw.argmin(axis=1)
    d = v[r, :, cmax] - v[r, :, cmin]                       # (n, iterations): each iteration's push on (top class - bottom class)
    spread = d.sum(axis=1)
    ds = -np.sort(-d, axis=1)
    return {"check": chk, "spread": spread, "max_push": ds[:, 0], "share1": ds[:, 0] / spread, "share5": ds[:, :5].sum(1) / spread,
            "share10": ds[:, :10].sum(1) / spread, "n_gt1": (d > 1).sum(1), "n_pos": (d > 0).sum(1), "d": d, "L": L, "vals": v}


def run_report():
    import lightgbm as lgb
    header = open(HEADER, encoding="utf-8").read()
    A, B = parse_model(os.path.join(A_DIR, "model.txt")), parse_model(B_PATH)
    if not A["sha256"].startswith(SHA_PREFIX):
        raise SystemExit("model A sha256 changed")
    side = json.load(open(os.path.join(A_DIR, "model.json"), encoding="utf-8"))
    binfo = json.load(open(B_INFO, encoding="utf-8"))
    sA, sB = leaf_stats(A), leaf_stats(B)

    # ---- A) sha256 and regularisation actually in the files
    LINES.clear()
    title("A. SHA256 AND THE REGULARISATION ACTUALLY IN EFFECT   (read out of each model.txt, not the sidecar)")
    emit("Model A sha256 %s   (prefix %s: MATCH)" % (A["sha256"], SHA_PREFIX))
    emit("Model B sha256 %s   (%s)" % (B["sha256"], B_PATH.replace(os.sep, "/")))
    emit("Model B is a RETRAIN of fold 0 (diagnostic 56 did not save its booster), trained in %.0f s. Against diagnostic 56's stored fold-0 held-out" % binfo["train_seconds"])
    emit("predictions: max |difference| = %.3e over %s cells; cells differing at all = %s." % (
        binfo["max_abs_diff_vs_stored_proba"], f(binfo["compared_cells"]), f(binfo["cells_differing_at_all"])))
    emit()
    keys = ["min_sum_hessian_in_leaf", "lambda_l2", "lambda_l1", "min_data_in_leaf", "num_leaves", "learning_rate", "max_depth", "max_bin",
            "min_gain_to_split", "boosting", "objective", "num_class", "num_iterations", "feature_fraction", "bagging_fraction", "seed", "deterministic"]
    emit("%-26s%22s%22s   %s" % ("parameter (from model.txt)", "model A", "model B", "sidecar of A (lgb_params)"))
    rule()
    for k in keys:
        emit("%-26s%22s%22s   %s" % (k, A["params"].get(k, "(absent)"), B["params"].get(k, "(absent)"), side["lgb_params"].get(k, "-")))
    rule()
    diff = sorted(k for k in set(A["params"]) | set(B["params"]) if A["params"].get(k) != B["params"].get(k))
    emit("Every parameter in the two files' parameter blocks that differs between A and B: %s" % (
        ", ".join("%s (A %s, B %s)" % (k, A["params"].get(k), B["params"].get(k)) for k in diff) or "NONE"))
    agree = float(A["params"]["min_sum_hessian_in_leaf"]) == side["lgb_params"]["min_sum_hessian_in_leaf"] and \
        float(A["params"]["lambda_l2"]) == side["lgb_params"]["lambda_l2"]
    emit("Model A's file %s its own sidecar on min_sum_hessian_in_leaf and lambda_l2 (file: %s and %s; sidecar: %s and %s)." % (
        "AGREES WITH" if agree else "DISAGREES WITH", A["params"]["min_sum_hessian_in_leaf"], A["params"]["lambda_l2"],
        side["lgb_params"]["min_sum_hessian_in_leaf"], side["lgb_params"]["lambda_l2"]))
    emit()
    emit("%-44s%16s%16s" % ("booster structure (from the files)", "model A", "model B"))
    rule()
    emit("%-44s%16d%16d" % ("trees (100 rounds x 6 classes)", sA["n_trees"], sB["n_trees"]))
    emit("%-44s%16s%16s" % ("leaves per tree, min / median / max", "%d / %g / %d" % sA["leaves"], "%d / %g / %d" % sB["leaves"]))
    emit("%-44s%16s%16s" % ("leaves in total", f(sA["n_leaves"]), f(sB["n_leaves"])))
    emit("%-44s%16s%16s" % ("leaves holding fewer than 20 training rows", cnt(sA["leaves_lt20"], sA["n_leaves"]).split(" (")[0],
                            cnt(sB["leaves_lt20"], sB["n_leaves"]).split(" (")[0]))
    emit("(the files record leaf_count per leaf, so this is the training-row count of each leaf, not an estimate)")
    sec_a = list(LINES)

    # ---- B) reference scale
    LINES.clear()
    title("B. REFERENCE SCALE   (how wide a raw-score spread is ordinary?)")
    lc = side["label_counts"]
    v = np.array([lc[s] for s in STATES], dtype=np.float64)
    emit("Six class frequencies of the fit rows (all origins pooled), recomputed from the sidecar's label counts:")
    for s, c in zip(STATES, v):
        emit("   %-8s %14s   freq %.6f%%   ln(freq) %+9.4f" % (s, f(c), 100 * c / v.sum(), np.log(c / v.sum())))
    emit("Span of the six log class frequencies, pooled: ln(max) - ln(min) = %.4f" % (np.log(v.max()) - np.log(v.min())))
    emit("The brief's figure (12-13 Oct) is 13.24. The pooled recomputation above does NOT reproduce it. Spans of the same quantity within each origin")
    emit("state's rows (counts from artifacts/baseline_headline/counts.csv, verified against the panel in diagnostic 56):")
    base = np.array([[int(x) for x in line.split(",")[1:]] for line in open(os.path.join("artifacts", "baseline_headline", "counts.csv")).read().splitlines()[1:5]],
                    dtype=np.float64)
    for i, o in enumerate(TRANSIENT):
        r = base[i]
        emit("   origin %-8s ln(max/min) = %.4f   (largest class %s %s, smallest class %s %s)" % (
            o, np.log(r.max() / r.min()), STATES[int(r.argmax())], f(r.max()), STATES[int(r.argmin())], f(r.min())))
    emit("None of these is exactly 13.24 (Current origin, 13.17, is the nearest); the source of 13.24 is not known here. Both are reported; a spread far")
    emit("above these (tens, hundreds) is not ordinary variation. An intercept-only softmax model would have a spread equal to the pooled span.")
    sec_b = list(LINES)

    # ---- the held-out rows
    print("building the feature matrix ...", flush=True)
    manifest = panel_io.read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    S = label_scan(quarters, TRANSIENT, None)
    X, y, fs, fp, fold, pair_ids, pair_folds = C.build_inputs(quarters, side, S["rows_by_quarter"])
    del pair_ids, pair_folds
    with np.load(os.path.join(CAL, "oof", "meta.npz")) as z:
        if not (np.array_equal(z["fold"], fold) and np.array_equal(z["y"], y) and np.array_equal(z["from_state"], fs)):
            raise SystemExit("rebuilt rows differ from diagnostic 56's")
    ho = np.flatnonzero(fold == 0)
    Xh = X.iloc[ho].reset_index(drop=True)
    yh, fsh = y[ho].astype(np.int64), fs[ho]
    del X
    got = {o: int((fsh == C.IDX[o]).sum()) for o in TRANSIENT}
    if got != EXPECT_ROWS:
        raise SystemExit("held-out row counts %s != diagnostic 56's %s" % (got, EXPECT_ROWS))
    bA, bB = lgb.Booster(model_file=os.path.join(A_DIR, "model.txt")), lgb.Booster(model_file=B_PATH)
    feats = side["features"]
    for b in (bA, bB):
        if b.feature_name() != feats:
            raise SystemExit("feature order differs from the sidecar")
    RAW, PRB, EXP = {}, {}, {}
    for name, b in (("A", bA), ("B", bB)):
        raw = np.empty((len(ho), K))
        for s in range(0, len(ho), C.CHUNK):
            raw[s:s + C.CHUNK] = b.predict(Xh.iloc[s:s + C.CHUNK], raw_score=True)
        p, e = softmax(raw)
        lg = b.predict(Xh.iloc[:200_000])
        RAW[name], PRB[name], EXP[name] = raw, p, e
        print("scored model %s; softmax vs LightGBM predict on the first 200,000 rows: max |diff| %.2e" % (name, np.abs(lg - p[:200_000]).max()), flush=True)
        RAW[name + "_softmax_check"] = float(np.abs(lg - p[:200_000]).max())
    # tie-back to diagnostic 56: floor hits from the stored probabilities
    stored_B = np.load(os.path.join(CAL, "oof", "fold0_proba.npy"))
    full, _ = C.load_stack(A_DIR, "full_proba")
    stored_A = np.asarray(full[ho])
    r_all = np.arange(len(ho))
    stored_floor = {"A": stored_A[r_all, yh] < C.FLOOR, "B": stored_B[r_all, yh] < C.FLOOR}
    del stored_A, stored_B

    # ---- C) spread tables
    LINES.clear()
    title("C. RAW-SCORE SPREAD   (raw_score=True, six margins per row; spread = max - min across the six classes; rows are fold 0's held-out fit rows)")
    emit("Model A saw these loans in training (in-sample); model B did not. Counts beside shares; THIN = a nonzero count under 100.")
    spreads = {m: RAW[m].max(axis=1) - RAW[m].min(axis=1) for m in ("A", "B")}
    QS = [0, 0.5, 0.9, 0.99, 0.999, 1.0]
    emit()
    emit("C1. Per-row spread, by origin state")
    rule()
    emit("%-8s%-7s%11s | %9s %9s %9s %9s %9s %9s" % ("origin", "model", "rows", "min", "p50", "p90", "p99", "p99.9", "max"))
    rule()
    worst = {}
    for o in TRANSIENT:
        m = fsh == C.IDX[o]
        for nm in ("A", "B"):
            q = np.quantile(spreads[nm][m], QS)
            worst.setdefault(nm, []).append((float(q[3]), o))
            emit("%-8s%-7s%11s | %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f" % (o, nm, f(m.sum()), *q))
    rule()
    emit()
    emit("C2. Per-class raw score (min / p50 / max over the origin state's rows)")
    rule()
    emit("%-8s%-7s %s" % ("origin", "model", "".join("%-30s" % s for s in STATES)))
    rule()
    for o in TRANSIENT:
        m = fsh == C.IDX[o]
        for nm in ("A", "B"):
            cells = []
            for k in range(K):
                q = np.quantile(RAW[nm][m, k], [0, 0.5, 1.0])
                cells.append("%-30s" % ("%.2f / %.2f / %.2f" % tuple(q)))
            emit("%-8s%-7s %s" % (o, nm, "".join(cells)))
    rule()
    emit()
    emit("C3. Rows whose spread exceeds 20, 50, 100, 1000")
    rule()
    emit("%-8s%-7s%11s | %s" % ("origin", "model", "rows", "".join("%-26s" % ("> %d" % t) for t in SPREAD_T)))
    rule()
    for o in TRANSIENT:
        m = fsh == C.IDX[o]
        for nm in ("A", "B"):
            emit("%-8s%-7s%11s | %s" % (o, nm, f(m.sum()), "".join("%-26s" % cnt(int((spreads[nm][m] > t).sum()), int(m.sum())) for t in SPREAD_T)))
    rule()
    emit()
    emit("C4/C5. Full saturation and floor hits")
    emit("  exact zero = the float64 softmax returns exactly 0.0 on at least one class (exp(raw - max) underflows; raw gap > about 745).")
    emit("  floor hit  = probability on the TRUE class below %g. 'diag 56' = the same count from diagnostic 56's stored probabilities (the tie-back)." % C.FLOOR)
    rule()
    emit("%-8s%-7s%11s | %-28s %-28s %-14s" % ("origin", "model", "rows", "exactly 0.0 on >=1 class", "floor hit (this script)", "diag 56 count"))
    rule()
    for o in TRANSIENT:
        m = fsh == C.IDX[o]
        for nm in ("A", "B"):
            ez = int((EXP[nm][m] == 0.0).any(axis=1).sum())
            fl = int((PRB[nm][m][np.arange(int(m.sum())), yh[m]] < C.FLOOR).sum())
            emit("%-8s%-7s%11s | %-28s %-28s %-14s" % (o, nm, f(m.sum()), cnt(ez, int(m.sum())), cnt(fl, int(m.sum())),
                                                     f(int(stored_floor[nm][m].sum())) + ("" if fl == int(stored_floor[nm][m].sum()) else "  <- DIFFERS")))
    rule()
    emit("Softmax built here vs LightGBM predict() on the first 200,000 rows: max |difference| A %.2e, B %.2e." % (RAW["A_softmax_check"], RAW["B_softmax_check"]))
    pooled_a, pooled_b = [], []
    sec_c = list(LINES)

    # ---- D) leaf statistics
    LINES.clear()
    title("D. WHERE THE SPREAD COMES FROM   (leaf values, A and B side by side; numbers only)")
    pick = {nm: max(worst[nm]) for nm in ("A", "B")}
    larger = max(("A", "B"), key=lambda nm: pick[nm][0])
    emit("Rule fixed here: the 'larger-spread' model is the one with the larger p99.9 spread in any single origin state; its worst origin state is that state.")
    emit("Model A's largest p99.9 spread: %.3f at %s.   Model B's: %.3f at %s.   => larger spread: model %s, worst origin state %s." % (
        pick["A"][0], pick["A"][1], pick["B"][0], pick["B"][1], larger, pick[larger][1]))
    emit()
    emit("D1. Trees and leaves (whole booster; leaf values in the files already include the learning rate and, in the first round, the class bias)")
    rule()
    emit("%-52s%16s%16s" % ("", "model A", "model B"))
    for th in (1, 5, 10):
        emit("%-52s%16s%16s" % ("trees with a leaf of |value| > %d" % th, "%d of %d" % (sA["trees_gt%d" % th], sA["n_trees"]),
                                "%d of %d" % (sB["trees_gt%d" % th], sB["n_trees"])))
    for th in (1, 5, 10):
        emit("%-52s%16s%16s" % ("leaves with |value| > %d" % th, "%s of %s" % (f(sA["leaves_gt%d" % th]), f(sA["n_leaves"])),
                                "%s of %s" % (f(sB["leaves_gt%d" % th]), f(sB["n_leaves"]))))
    emit("%-52s%16.3f%16.3f" % ("median over trees of the largest |leaf|", float(np.median(sA["tree_max_abs"])), float(np.median(sB["tree_max_abs"]))))
    emit()
    emit("D2. The ten largest |leaf value| in each booster (tree index; class of the tree = index mod 6; boosting round = index // 6; training rows in the leaf)")
    rule()
    emit("%-3s | %-44s | %-44s" % ("#", "model A: tree  class  round   value   rows", "model B: tree  class  round   value   rows"))
    for i in range(10):
        a, b = sA["top"][i], sB["top"][i]
        emit("%-3d | %-44s | %-44s" % (i + 1, "%4d  %-7s %3d  %+9.3f  %s" % (a[0], a[1], a[2], a[3], f(a[4])),
                                       "%4d  %-7s %3d  %+9.3f  %s" % (b[0], b[1], b[2], b[3], f(b[4]))))
    emit()
    ws = pick[larger][1]
    m = np.flatnonzero(fsh == C.IDX[ws])
    emit("D3. How the margin is built, rows of origin %s (n = %s), for BOTH models. Per row: (top class - bottom class) margin = sum over the 100 boosting rounds of" % (ws, f(len(m))))
    emit("    (that round's leaf for the top class - its leaf for the bottom class). 'push' = one round's contribution. Quantiles over rows.")
    emit("    'share' = that many rounds' pushes divided by the net margin; it exceeds 1 when those rounds push harder than the net margin and other rounds cancel part of it.")
    rule()
    emit("%-52s%-7s %10s %10s %10s" % ("per-row statistic", "model", "p50", "p90", "max"))
    rule()
    dec = {}
    for nm, b, M in (("A", bA, A), ("B", bB, B)):
        d = decompose(b, M, Xh.iloc[m], RAW[nm][m])
        dec[nm] = d
        if d["check"] > 1e-9:
            emit("WARNING model %s: sum of leaf values differs from raw_score by %.2e" % (nm, d["check"]))
        for lab, key in (("margin (top - bottom class)", "spread"), ("largest single-round push", "max_push"),
                         ("share of the margin from the largest 1 round", "share1"), ("share from the largest 5 rounds", "share5"),
                         ("share from the largest 10 rounds", "share10"), ("rounds with a push > 1 (of 100)", "n_gt1"),
                         ("rounds pushing the margin up (of 100)", "n_pos")):
            q = np.quantile(d[key], [0.5, 0.9, 1.0])
            emit("%-52s%-7s %10.3f %10.3f %10.3f" % (lab, nm, *q))
    rule()
    emit("Check: the leaf values reached by each row sum to the raw score within %.1e (A), %.1e (B)." % (dec["A"]["check"], dec["B"]["check"]))
    sec_d = list(LINES)

    # ---- E) one loan by hand
    LINES.clear()
    title("E. ONE LOAN, BY HAND")
    o60 = fsh == C.IDX["DQ60"]
    pa = PRB["A"][np.arange(len(ho)), yh]
    pb = PRB["B"][np.arange(len(ho)), yh]
    cand = np.flatnonzero(o60 & (pa < C.FLOOR) & (pb >= C.FLOOR))
    emit("DQ60-origin rows in fold 0 where the shipped model's true-class probability is below %g and the fold model's is not: %s of %s DQ60 rows." % (
        C.FLOOR, f(len(cand)), f(int(o60.sum()))))
    if not len(cand):
        emit("No such row exists, so no loan is shown.")
    else:
        i = int(cand[0])
        emit("Chosen: the FIRST such row in panel order (not selected for how it looks).")
        g = int(ho[i])
        acc = 0
        for q in quarters:
            if g < acc + S["rows_by_quarter"][q]:
                pos = g - acc
                break
            acc += S["rows_by_quarter"][q]
        d = panel_io.load_panel(roles=["fit"], columns=["loan_id", "from_state", "to_state", "from_period"], quarters=[q])
        mk, yy = keep_rows(d, TRANSIENT, None)
        mk &= yy >= 0
        d = d[mk].reset_index(drop=True)
        r = d.iloc[pos]
        if int(r["from_period"]) != int(fp[g]) or str(r["from_state"]) != "DQ60":
            raise SystemExit("could not locate the row in its partition consistently")
        emit("Panel position: partition %s, row %d of that partition's training rows; global row %s; loan_id %s; from_period %d." % (
            q, pos, f(g), r["loan_id"], int(r["from_period"])))
        emit()
        emit("Feature values (the nine features the models read):")
        for c in feats:
            val = Xh.iloc[i][c]
            emit("   %-14s %s" % (c, val if isinstance(val, str) else ("%s" % (val if not hasattr(val, "item") else val.item()))))
        emit()
        emit("Realised next state: %s" % STATES[int(yh[i])])
        emit()
        emit("%-9s | %12s %12s | %16s %16s" % ("class", "raw A", "raw B", "probability A", "probability B"))
        rule()
        for k, s in enumerate(STATES):
            mark = "   <- realised" if k == int(yh[i]) else ""
            emit("%-9s | %+12.4f %+12.4f | %16.6e %16.6e%s" % (s, RAW["A"][i, k], RAW["B"][i, k], PRB["A"][i, k], PRB["B"][i, k], mark))
        rule()
        emit("spread (max - min raw): A %.3f   B %.3f" % (spreads["A"][i], spreads["B"][i]))
        emit()
        emit("The five largest single-round leaf values this row falls into, per model (round, class, value):")
        for nm, b, M in (("A", bA, A), ("B", bB, B)):
            L = b.predict(Xh.iloc[[i]], pred_leaf=True).astype(np.int64)[0]
            vals = np.array([M["trees"][t]["leaf_value"][L[t]] for t in range(len(L))])
            top = np.argsort(-np.abs(vals))[:5]
            emit("   model %s: %s" % (nm, "   ".join("(round %d, %s, %+.3f)" % (t // K, STATES[t % K], vals[t]) for t in top)))
    sec_e = list(LINES)

    body = "\n".join(["\n".join(sec_a), "\n".join(sec_b), "\n".join(sec_c), "\n".join(sec_d), "\n".join(sec_e)])
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(header.rstrip("\n") + "\n" + body + "\n")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "header":
        run_header()
    elif mode == "report":
        from src.memwatch import MemWatch
        watch = MemWatch(on_abort=lambda m: print("ABORTED, run did not complete: " + m, flush=True))
        watch.start()
        run_report()
        watch.stop()
        print("peak RSS %.2f GB, min system available %.2f GB" % (watch.peak_rss() / C.GB, watch.min_avail / C.GB))
    else:
        raise SystemExit(__doc__)
