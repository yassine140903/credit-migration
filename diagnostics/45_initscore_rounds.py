"""CURRENT and POOLED models: does an init score, or more rounds, change the Brier score?   MEASUREMENT ONLY.

    python diagnostics/45_initscore_rounds.py   # from the repo root

Reuses diagnostics/44_capacity_brier.py: same populations, same split, same seed, same features, same Brier
implementation and reporting. Writes diagnostics/45_initscore_rounds.txt and nothing else: no artifacts/ml/
model, nothing under src/ or data/, no git command. Reuses src.train_ml's constants and feature handling
(BASE_FEATURES, STATES, TRANSIENT; float32 numerics, property_state and from_state native categoricals with the
fixed category lists train_ml declares, nulls left as-is). LightGBM at DEFAULTS only (num_leaves 31,
min_data_in_leaf 20).

HYPOTHESIS UNDER TEST. Undertraining: every class starts at 1/6 and 100 rounds cannot descend to a 0.004% base
rate. Four runs per population:
  1. init score OFF, 100 rounds   (must reproduce diagnostic 44 run A: Brier train/held-out to six decimals,
                                   else the script STOPS and says so)
  2. init score ON,  100 rounds
  3. init score OFF, 400 rounds
  4. init score ON,  400 rounds
INIT SCORE. The six class frequencies of the model's own TRAIN rows, converted to log-odds (the softmax logit
log f_j) and passed as the Dataset init_score (an (n, 6) array; LightGBM stores it class-major). At round zero the
softmax of those logits is exactly the prior-only predictor. LightGBM does not store a Dataset init_score in the
model, so for init-ON runs every prediction is softmax(raw_score + log f), done here. For init-OFF runs the
library's own probability output is used, as in diagnostic 44. Round-zero check: raw_score with start_iteration
= number of rounds (so zero trees contribute) plus the init values, through the same softmax.
ADDED, not asked for: held-out Brier at intermediate round counts (1, 10, 50, 100, 200, 400 up to the run's
rounds) for each run, because the hypothesis is about how the score moves with rounds.

POPULATIONS (selection window: role == 'fit', 200601 <= from_period <= 200706), run separately:
  (a) CURRENT: from_state == 'Current'. Features: the eight base features (from_state omitted: constant).
  (b) POOLED : all four transient from_states. Features: the eight base features + from_state (categorical).
The panel is read once, one partition at a time, five-plus-feature columns only; the CURRENT population is
the Current rows of the same read. Each population gets its own split, its own prior and its own runs.

SPLIT. Same procedure as diagnostics 40/41: the sorted unique loan_ids of the population are shuffled with
numpy default_rng(SEED), the first 80% (rounded) go to train, the rest are held out. To avoid holding 11M
Python strings, loan_ids are factorised per partition (sort=True) with a running offset; this is the same
sorted-unique order because a loan lives in exactly one partition (asserted by the vintage prefix) and the
partition prefixes sort in file order (asserted).

METRIC. Multiclass Brier score, written out explicitly (BRIER_FORMULA below): for each row, the six predicted
probabilities minus the one-hot vector of the actual class, squared element-wise, summed over the six classes,
averaged over rows. Range 0..2, lower is better, NOT divided by the number of classes. Its per-class
decomposition is mean_i (p_ij - 1[y_i = j])^2 for each class column j; the six terms add to the total.
Carried alongside for the record: multiclass log-loss = -mean(log p[true class]) with p clipped to [1e-15, 1],
exactly as in diagnostic 42, with the count of held-out rows whose probability on the actual class is below
1e-6 and those rows' share of the total log-loss. Classes are the six
in panel.STATES order. "Predicted distribution": mean predicted probability per class, and the share/count of
rows whose argmax is that class. Wall-clock is the whole lgb.train call (Dataset construction included).
Peak RSS comes from a fresh src.memwatch.MemWatch around each run's lgb.train: "in-run sampled" is the largest
RSS the 0.5 s sampler saw during that call; "process peak" is MemWatch.peak_rss(), which on Windows includes the
OS's process-lifetime peak working set and so never decreases from run to run.
"""

import gc
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                         # noqa: E402
from src.memwatch import MemWatch                                       # noqa: E402
from src.train_ml import BASE_FEATURES, STATES, TRANSIENT               # noqa: E402

OUT = os.path.join("diagnostics", "45_initscore_rounds.txt")
REF44 = os.path.join("diagnostics", "44_capacity_brier.txt")
CHECKPOINTS = [1, 10, 50, 100, 200, 400]
BRIER_FORMULA = ("Brier = (1/n) * sum_i sum_{j=1..6} ( p_ij - o_ij )^2,  where p_ij is the predicted probability of class j "
                 "for row i and o_ij = 1 if class j is the actual class of row i, else 0.  Not divided by 6; range 0..2.")
WIN_LO, WIN_HI = 200601, 200706
SEED = 40
TRAIN_FRAC = 0.8
GB = 2 ** 30
POPULATIONS = [("CURRENT", ["Current"], 10911314), ("POOLED", list(TRANSIENT), 11017047)]
CATEGORIES = {"property_state": list(panel.PROPERTY_STATES), "from_state": list(TRANSIENT)}
BASE_PARAMS = {"objective": "multiclass", "num_class": len(STATES), "device_type": "cpu", "verbosity": -1}
PARAMS = dict(BASE_PARAMS)                      # LightGBM defaults, as diagnostic 44 run A
RUN_SPECS = [("1. init OFF, 100 rounds", False, 100), ("2. init ON, 100 rounds", True, 100),
             ("3. init OFF, 400 rounds", False, 400), ("4. init ON, 400 rounds", True, 400)]
SHOW_FROM_MODEL = ["learning_rate", "num_leaves", "min_data_in_leaf", "min_sum_hessian_in_leaf", "max_depth",
                   "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction", "max_bin", "num_threads",
                   "seed", "deterministic"]
RULE_W = 112
EPS = 1e-15
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


def load():
    """Selection-window fit rows, all four transient origins: features (incl. from_state), label, loan code."""
    manifest = panel_io.read_manifest()
    prefixes = ["F" + p["quarter"][2:4] + p["quarter"][4:] for p in manifest["partitions"]]
    assert prefixes == sorted(prefixes), "partition vintage prefixes do not sort in file order"
    cols = ["role", "loan_id", "from_period", "from_state", "to_state"] + BASE_FEATURES
    xs, ys, codes = [], [], []
    offset = n_null = 0
    for p, prefix in zip(manifest["partitions"], prefixes):
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = d["from_state"].isin(TRANSIENT).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        lid = d["loan_id"].astype(str)
        assert lid.str.startswith(prefix).all(), "a loan_id lacks its partition's vintage prefix"
        c, uniq = pd.factorize(lid, sort=True)
        codes.append(c.astype(np.int64) + offset)
        offset += len(uniq)
        data = {}
        for col in BASE_FEATURES:
            if col in CATEGORIES:
                data[col] = d[col].cat.set_categories(CATEGORIES[col]).reset_index(drop=True)
            else:
                data[col] = d[col].to_numpy(dtype=np.float32, na_value=np.nan)
        data["from_state"] = d["from_state"].astype(str).astype(pd.CategoricalDtype(CATEGORIES["from_state"])).reset_index(drop=True)
        xs.append(pd.DataFrame(data))
        ys.append(y[m])
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows, {len(uniq):,} loans", flush=True)
        del d, data, lid
        gc.collect()
    X = pd.concat(xs, ignore_index=True)
    del xs
    for col, cats in CATEGORIES.items():
        assert str(X[col].dtype) == "category" and list(X[col].cat.categories) == cats, col
    return X, np.concatenate(ys), np.concatenate(codes), n_null, manifest


def logloss(P, y):
    p = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def brier_by_class(P, y):
    """Per-class terms of the Brier score: mean_i (P[i, j] - 1[y_i == j])^2, j = 0..5. Their sum is the Brier score.
    P is (n, 6); a broadcast view (the prior-only predictor) works too."""
    return np.array([float(np.mean((P[:, j] - (y == j)) ** 2)) for j in range(len(STATES))])


def brier(P, y):
    return float(brier_by_class(P, y).sum())


def softmax_inplace(Z):
    """Row-wise softmax of a float64 (n, 6) array, in place."""
    Z -= Z.max(axis=1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z


def probs(booster, X, init_row, num_iteration=None, start_iteration=0):
    """(n, 6) probabilities. init_row None: the library's own output. Else softmax(raw_score + init_row)."""
    if init_row is None:
        return booster.predict(X, num_iteration=num_iteration)
    raw = np.asarray(booster.predict(X, raw_score=True, num_iteration=num_iteration, start_iteration=start_iteration),
                     dtype=np.float64)
    raw += init_row
    return softmax_inplace(raw)


def ref44(pop):
    """Diagnostic 44's run-A (train Brier, held-out Brier) for this population, read from its report."""
    pat = re.compile(r"^%s / A\. .*?\s(\d\.\d{7})\s+(\d\.\d{7})\s" % pop)
    with open(REF44, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise RuntimeError("no run-A Brier line for %s in %s" % (pop, REF44))


def counts(y):
    return np.bincount(y, minlength=len(STATES))


def side_report(name, y, n_loans):
    c = counts(y)
    emit(f"{name}: {len(y):,} rows, {n_loans:,} loans")
    emit("  " + "   ".join(f"{s} {int(n):,}" for s, n in zip(STATES, c)))
    absent = [s for s, n in zip(STATES, c) if n == 0]
    emit(f"  classes absent from {name}: " + (", ".join(absent) if absent else "none (all six present)"))
    return absent


def dist_table(label, P, y):
    c = counts(y)
    am = counts(P.argmax(axis=1))
    mp = P.mean(axis=0)
    n = len(y)
    emit(f"  {label}  ({n:,} rows)")
    emit(f"  {'class':<10}{'actual n':>12}{'actual %':>11}{'mean pred prob %':>19}{'pred/actual':>14}{'argmax n':>13}{'argmax %':>11}")
    for i, s in enumerate(STATES):
        ratio = f"{mp[i] / (c[i] / n):>12.2f}x" if c[i] else f"{'n/a (0)':>13}"
        emit(f"  {s:<10}{int(c[i]):>12,}{100.0 * c[i] / n:>10.4f}%{100.0 * mp[i]:>18.4f}%{ratio}"
             f"{int(am[i]):>13,}{100.0 * am[i] / n:>10.4f}%")


def run_population(lgb, pop, origin, expected, Xall, yall, call):
    in_pop = Xall["from_state"].isin(origin).to_numpy()
    feats = list(BASE_FEATURES) + (["from_state"] if len(origin) > 1 else [])
    categorical = ["property_state"] + (["from_state"] if len(origin) > 1 else [])
    X, y, codes = Xall[in_pop].reset_index(drop=True), yall[in_pop], call[in_pop]
    title(f"POPULATION {pop}: role == 'fit', from_state in {origin}, {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"rows {len(y):,} (expected {expected:,}: {'MATCH' if len(y) == expected else 'DIFFERENT'})")
    emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")
    emit("NOT features: loan_age, run_length, run_length_censored" + ("" if len(origin) > 1 else ", from_state (constant)"))

    emit()
    emit(f"SPLIT BY LOAN_ID ({pop})")
    uniq, inv = np.unique(codes, return_inverse=True)       # sorted unique loans of this population
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    flag = np.zeros(len(uniq), dtype=bool)
    flag[perm[:n_tr]] = True
    in_train = flag[inv]
    assert np.intersect1d(np.unique(codes[in_train]), np.unique(codes[~in_train])).size == 0, "a loan is on both sides"
    emit(f"seed {SEED} (numpy default_rng on the sorted unique loan_ids); train fraction {TRAIN_FRAC} of loans "
         f"-> {n_tr:,} of {len(uniq):,} loans to train, {len(uniq) - n_tr:,} held out")
    emit("no loan_id appears on both sides (asserted)")
    emit()
    Xtr, ytr = X[in_train].reset_index(drop=True), y[in_train]
    Xte, yte = X[~in_train].reset_index(drop=True), y[~in_train]
    del X, y, codes, inv
    gc.collect()
    absent_tr = side_report("train", ytr, n_tr)
    emit()
    absent_te = side_report("held-out", yte, len(uniq) - n_tr)
    emit()
    emit(f"row share: train {100.0 * len(ytr) / (len(ytr) + len(yte)):.2f}%   held-out "
         f"{100.0 * len(yte) / (len(ytr) + len(yte)):.2f}%")
    if absent_tr or absent_te:
        emit(f"NOTE: classes absent on a side - train: {absent_tr or 'none'}; held-out: {absent_te or 'none'}. "
             "All six classes remain declared in the model (num_class = 6).")

    prior = counts(ytr) / len(ytr)
    Ppr_tr = np.broadcast_to(prior, (len(ytr), len(STATES)))
    Ppr_te = np.broadcast_to(prior, (len(yte), len(STATES)))
    pr_ll_tr, pr_ll_te = logloss(Ppr_tr, ytr), logloss(Ppr_te, yte)
    pr_bs_tr, pr_bs_te = brier(Ppr_tr, ytr), brier(Ppr_te, yte)
    pr_cls_te = brier_by_class(Ppr_te, yte)
    emit()
    emit("prior-only reference (not a model): predicting the TRAIN-side class frequencies for every row")
    emit("  train-side class frequencies: " + ", ".join(f"{s} {p:.8f}" for s, p in zip(STATES, prior)))
    emit(f"  Brier    : train {pr_bs_tr:.7f}   held-out {pr_bs_te:.7f}")
    emit(f"  log-loss : train {pr_ll_tr:.6f}   held-out {pr_ll_te:.6f}")

    assert (prior > 0).all(), "a class has zero train frequency: its log-odds would be -inf"
    init_row = np.log(prior)
    emit()
    emit("INIT SCORE (train-row class frequencies -> log-odds log f_j, passed as LightGBM init_score when ON)")
    emit(f"  {'class':<10}{'train frequency':>18}{'init value (log f)':>22}")
    for j_, s_ in enumerate(STATES):
        emit(f"  {s_:<10}{prior[j_]:>18.10f}{init_row[j_]:>22.8f}")
    P0 = softmax_inplace(np.tile(init_row, (len(yte), 1)))
    emit(f"  softmax(init) reproduces the frequencies: max |difference| {np.abs(P0[0] - prior).max():.3e}")
    emit(f"  round-zero Brier from the init values alone: held-out {brier(P0, yte):.7f} vs prior-only {pr_bs_te:.7f} "
         f"(difference {brier(P0, yte) - pr_bs_te:+.3e})")
    del P0

    results = []
    for name_, init_on, rounds in RUN_SPECS:
        name = f"{pop} / {name_}"
        params = dict(PARAMS)
        title(f"RUN {name}")
        emit(f"features ({len(feats)}): {feats}   categorical: {categorical}")
        emit("params passed to lgb.train: " + ", ".join(f"{k}={v!r}" for k, v in params.items())
             + f";  num_boost_round={rounds};  init_score="
             + ("log train class frequencies, shape (n, 6)" if init_on else "none (library default boost_from_average)"))
        print("  training ...", flush=True)
        watch = MemWatch(on_abort=on_abort)
        watch.phase = f"{name}: lgb.train"
        watch.start()
        t0 = time.perf_counter()
        ds = lgb.Dataset(Xtr[feats], label=ytr, categorical_feature=categorical, free_raw_data=True,
                         init_score=(np.tile(init_row, (len(ytr), 1)) if init_on else None))
        booster = lgb.train(params, ds, num_boost_round=rounds)
        secs = time.perf_counter() - t0
        watch.stop()
        del ds
        gc.collect()
        assert booster.current_iteration() == rounds, booster.current_iteration()
        assert booster.feature_name() == feats
        assert booster.pandas_categorical == [CATEGORIES[c] for c in categorical]
        block = booster.model_to_string().split("parameters:")[-1]
        eff = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]") and ": " in line:
                k, v = line[1:-1].split(": ", 1)
                eff[k] = v
        emit("effective values (from the model's parameters block): "
             + ", ".join(f"{k}={eff.get(k, '?')}" for k in SHOW_FROM_MODEL))
        n_trees = booster.num_trees()
        emit(f"trees {n_trees} ({n_trees // len(STATES)} per class)")
        emit(f"training wall-clock {secs:.1f} s ({secs / 60:.2f} min) = {secs / rounds:.2f} s/round, Dataset construction included;"
             f"  peak RSS in-run sampled {watch._peak / GB:.2f} GB, process peak {watch.peak_rss() / GB:.2f} GB;"
             f"  min system available {watch.min_avail / GB:.2f} GB")
        ir = init_row if init_on else None
        if init_on:
            Z = probs(booster, Xte[feats], init_row, start_iteration=rounds)       # zero trees contribute
            emit(f"  round-zero check through the real predict path (start_iteration={rounds}, zero trees + init): "
                 f"held-out Brier {brier(Z, yte):.7f} vs prior-only {pr_bs_te:.7f} (difference {brier(Z, yte) - pr_bs_te:+.3e})")
            del Z
        Ptr = probs(booster, Xtr[feats], ir)
        Pte = probs(booster, Xte[feats], ir)
        assert Ptr.shape == (len(ytr), len(STATES)) and Pte.shape == (len(yte), len(STATES))
        cps = [r for r in CHECKPOINTS if r < rounds] + [rounds]
        cp_brier = [(r, brier(Pte, yte) if r == rounds else brier(probs(booster, Xte[feats], ir, num_iteration=r), yte))
                    for r in cps]
        del booster
        cls_tr, cls_te = brier_by_class(Ptr, ytr), brier_by_class(Pte, yte)
        b_tr, b_te = float(cls_tr.sum()), float(cls_te.sum())
        l_tr, l_te = logloss(Ptr, ytr), logloss(Pte, yte)
        nll = -np.log(np.clip(Pte[np.arange(len(yte)), yte], EPS, 1.0))          # per held-out row
        low = Pte[np.arange(len(yte)), yte] < 1e-6
        low_share = float(nll[low].sum() / nll.sum()) if nll.sum() > 0 else float("nan")
        emit()
        emit("  MULTICLASS BRIER SCORE  (" + BRIER_FORMULA + ")")
        emit(f"  Brier, train                       : {b_tr:.7f}")
        emit(f"  Brier, held-out                    : {b_te:.7f}")
        emit(f"  gap (held-out - train)             : {b_te - b_tr:+.7f}")
        emit(f"  prior-only Brier, train / held-out : {pr_bs_tr:.7f} / {pr_bs_te:.7f}")
        emit(f"  held-out Brier minus prior-only held-out Brier : {b_te - pr_bs_te:+.7f}")
        emit(f"  (train Brier minus prior-only train Brier      : {b_tr - pr_bs_tr:+.7f})")
        emit()
        emit("  HELD-OUT BRIER BY NUMBER OF ROUNDS (added)" + ("; round 0 = prior-only" if init_on else "")
             + ": " + "   ".join(f"{r}: {v:.7f}" for r, v in ([(0, pr_bs_te)] if init_on else []) + cp_brier))
        emit()
        emit("  PER-CLASS DECOMPOSITION OF THE HELD-OUT BRIER  (term j = mean_i (p_ij - o_ij)^2; the six terms add to the total)")
        emit(f"  {'class':<10}{'model term':>14}{'% of total':>12}{'prior-only term':>18}{'model - prior':>16}")
        for j_, s_ in enumerate(STATES):
            emit(f"  {s_:<10}{cls_te[j_]:>14.8f}{100.0 * cls_te[j_] / b_te:>11.4f}%{pr_cls_te[j_]:>18.8f}{cls_te[j_] - pr_cls_te[j_]:>+16.8f}")
        emit(f"  {'TOTAL':<10}{b_te:>14.8f}{100.0:>11.4f}%{pr_bs_te:>18.8f}{b_te - pr_bs_te:>+16.8f}")
        emit()
        emit("  LOG-LOSS, as in diagnostic 42, carried for the record")
        emit(f"  multiclass log-loss, train / held-out : {l_tr:.6f} / {l_te:.6f}   gap {l_te - l_tr:+.6f}   "
             f"held-out minus prior-only {l_te - pr_ll_te:+.6f}")
        emit(f"  held-out rows with p(actual class) < 1e-6: {int(low.sum()):,} of {len(yte):,} ({100.0 * low.mean():.4f}%); "
             f"their share of the total held-out log-loss: {100.0 * low_share:.2f}%")
        emit()
        dist_table("HELD-OUT rows: actual vs predicted destination distribution", Pte, yte)
        ctr, mtr = counts(ytr), Ptr.mean(axis=0)
        emit("  train-row pred/actual ratio: " + "   ".join(
            (f"{s_} {mtr[j_] / (ctr[j_] / len(ytr)):.2f}x" if ctr[j_] else f"{s_} n/a") for j_, s_ in enumerate(STATES)))

        if name_.startswith("1."):
            r_tr, r_te = ref44(pop)
            ok = abs(b_tr - r_tr) < 1e-6 and abs(b_te - r_te) < 1e-6
            emit()
            emit(f"  REPRODUCTION CHECK vs diagnostic 44 run A: train Brier {b_tr:.7f} vs {r_tr:.7f}, held-out {b_te:.7f} vs {r_te:.7f} "
                 f"-> {'MATCH to six decimals' if ok else '!!! DO NOT MATCH !!!'}")
            if not ok:
                emit("  STOP: run 1 does not reproduce diagnostic 44 run A. The remaining runs are not executed; "
                     "this mismatch is the finding.")
                write_report()
                print(f"\nwrote {OUT}")
                sys.exit(1)
        results.append({"name": name, "b_tr": b_tr, "b_te": b_te, "l_tr": l_tr, "l_te": l_te, "low": int(low.sum()),
                        "low_share": low_share, "secs": secs, "rss": watch._peak, "peak": watch.peak_rss(),
                        "cp": cp_brier, "init": init_on})
        del Ptr, Pte, nll
        gc.collect()
    return results, {"pop": pop, "bs_tr": pr_bs_tr, "bs_te": pr_bs_te, "ll_tr": pr_ll_tr, "ll_te": pr_ll_te}


def main():
    import lightgbm as lgb

    emit("CURRENT and POOLED models, selection window: init score and boosting rounds under the Brier score   "
         "(measurement only; no selection, no recommendation)")
    X, y, codes, n_null, manifest = load()
    emit(f"panel build {manifest['build_stamp']}   lightgbm {lgb.__version__}   {os.cpu_count()} logical CPUs")
    emit(f"rows loaded (all four transient origins, selection window, role == 'fit'): {len(y):,}; "
         f"null/unknown labels dropped {n_null:,}")
    emit(f"loans (all four origins): {len(np.unique(codes)):,}")
    emit("LightGBM defaults (num_leaves 31, min_data_in_leaf 20); 100 or 400 rounds per run; no early stopping, no validation set")

    all_results, priors = [], []
    for pop, origin, expected in POPULATIONS:
        res, prior = run_population(lgb, pop, origin, expected, X, y, codes)
        all_results.append(res)
        priors.append(prior)

    title("SUMMARY  (figures only)")
    emit("MULTICLASS BRIER SCORE  (" + BRIER_FORMULA + ")")
    emit(f"{'run':<52}{'train Brier':>14}{'held-out Brier':>16}{'gap':>13}{'held-out - prior':>18}")
    for res, pr in zip(all_results, priors):
        for r in res:
            emit(f"{r['name']:<52}{r['b_tr']:>14.7f}{r['b_te']:>16.7f}{r['b_te'] - r['b_tr']:>+13.7f}{r['b_te'] - pr['bs_te']:>+18.7f}")
        emit(f"{'(prior-only reference, ' + pr['pop'] + ')':<52}{pr['bs_tr']:>14.7f}{pr['bs_te']:>16.7f}"
             f"{pr['bs_te'] - pr['bs_tr']:>+13.7f}{0.0:>+18.7f}")
    emit()
    emit("LOG-LOSS, as in diagnostic 42, for the record")
    emit(f"{'run':<52}{'train':>12}{'held-out':>12}{'gap':>12}{'held-out - prior':>18}{'rows p<1e-6':>14}{'their share':>13}")
    for res, pr in zip(all_results, priors):
        for r in res:
            emit(f"{r['name']:<52}{r['l_tr']:>12.6f}{r['l_te']:>12.6f}{r['l_te'] - r['l_tr']:>+12.6f}{r['l_te'] - pr['ll_te']:>+18.6f}"
                 f"{r['low']:>14,}{100.0 * r['low_share']:>12.2f}%")
        emit(f"{'(prior-only reference, ' + pr['pop'] + ')':<52}{pr['ll_tr']:>12.6f}{pr['ll_te']:>12.6f}"
             f"{pr['ll_te'] - pr['ll_tr']:>+12.6f}{0.0:>+18.6f}")
    emit()
    emit(f"{'run':<52}{'train seconds':>16}{'in-run RSS GB':>16}{'process peak GB':>18}")
    for res in all_results:
        for r in res:
            emit(f"{r['name']:<52}{r['secs']:>16.1f}{r['rss'] / GB:>16.2f}{r['peak'] / GB:>18.2f}")
    emit()
    emit("HELD-OUT BRIER BY NUMBER OF ROUNDS (added; round 0 = prior-only where an init score is used)")
    for res, pr in zip(all_results, priors):
        for r in res:
            emit(f"{r['name']:<52}" + "   ".join(f"{k}: {v:.7f}" for k, v in ([(0, pr['bs_te'])] if r["init"] else []) + r["cp"]))

    write_report()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
