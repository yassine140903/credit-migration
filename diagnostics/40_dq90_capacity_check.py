"""DQ90+ single-origin model: capacity / memorisation check.   MEASUREMENT ONLY.

    python diagnostics/40_dq90_capacity_check.py   # from the repo root

Writes diagnostics/40_dq90_capacity_check.txt and nothing else: no artifacts/ml/ model,
nothing under src/ or data/, no git command. Reuses src.train_ml's constants and feature
handling (BASE_FEATURES, STATES, N_ROUNDS, the single-origin model's matrix conventions:
float32 numerics, property_state a native categorical with the fixed panel.PROPERTY_STATES
list, nulls left as-is, from_state omitted because it is constant) and its LightGBM base
params (multiclass, 6 classes, CPU, verbosity -1, 100 rounds, no tuning, no early stopping,
no validation set). Loads only role == 'fit', from_state == 'DQ90+',
200601 <= from_period <= 200706 rows, through src.panel_io.load_panel.

SPLIT. By distinct loan_id (never by row or time): the sorted unique loan_ids are shuffled
with numpy default_rng(SEED) and the first 80% (rounded) go to train, the rest are held out.

METRICS. Multiclass log-loss = -mean(log p[true class]) with p clipped to [1e-15, 1]; classes
are the six in panel.STATES order. "Predicted distribution" is shown two ways: the mean
predicted probability per class, and the share of rows whose argmax is that class.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                         # noqa: E402
from src.train_ml import BASE_FEATURES, N_ROUNDS, STATES               # noqa: E402

OUT = os.path.join("diagnostics", "40_dq90_capacity_check.txt")
ORIGIN = "DQ90+"
WIN_LO, WIN_HI = 200601, 200706
EXPECTED_ROWS = 8787
SEED = 40
TRAIN_FRAC = 0.8
FEATURES = list(BASE_FEATURES)                  # single origin: from_state omitted
CATEGORICAL = ["property_state"]
CATEGORIES = {"property_state": list(panel.PROPERTY_STATES)}
BASE_PARAMS = {"objective": "multiclass", "num_class": len(STATES), "device_type": "cpu", "verbosity": -1}
RUNS = [("A. LightGBM defaults (as train_ml)", dict(BASE_PARAMS)),
        ("B. reduced capacity (for context)", dict(BASE_PARAMS, num_leaves=8, min_data_in_leaf=100))]
SHOW_FROM_MODEL = ["learning_rate", "num_leaves", "min_data_in_leaf", "min_sum_hessian_in_leaf", "max_depth",
                   "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction", "max_bin", "num_threads",
                   "seed", "deterministic"]
RULE_W = 100
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


def load():
    """The DQ90+ selection-window fit rows: feature matrix, label codes, loan ids."""
    manifest = panel_io.read_manifest()
    cols = ["loan_id", "from_period", "from_state", "to_state"] + FEATURES
    xs, ys, ids = [], [], []
    n_null_label = 0
    for p in manifest["partitions"]:
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[p["quarter"]])
        fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
        m = (d["from_state"] == ORIGIN).to_numpy() & (fp >= WIN_LO) & (fp <= WIN_HI)
        y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
        n_null_label += int((m & (y < 0)).sum())
        m &= y >= 0
        d = d[m]
        data = {}
        for c in FEATURES:
            if c in CATEGORIES:
                data[c] = d[c].cat.set_categories(CATEGORIES[c]).reset_index(drop=True)
            else:
                data[c] = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
        xs.append(pd.DataFrame(data))
        ys.append(y[m])
        ids.append(d["loan_id"].astype(str).to_numpy())
        print(f"  loaded {p['quarter']}: {int(m.sum()):,} rows", flush=True)
        del d, data
    X = pd.concat(xs, ignore_index=True)
    for c, cats in CATEGORIES.items():
        assert str(X[c].dtype) == "category" and list(X[c].cat.categories) == cats
    return X, np.concatenate(ys), np.concatenate(ids), n_null_label, manifest


def logloss(P, y):
    p = np.clip(P[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


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
    emit(f"  {'class':<10}{'actual n':>10}{'actual %':>11}{'mean pred prob %':>19}{'argmax n':>11}{'argmax %':>11}")
    for i, s in enumerate(STATES):
        emit(f"  {s:<10}{int(c[i]):>10,}{100.0 * c[i] / n:>10.4f}%{100.0 * mp[i]:>18.4f}%"
             f"{int(am[i]):>11,}{100.0 * am[i] / n:>10.4f}%")


def main():
    import lightgbm as lgb

    emit("DQ90+ single-origin model: capacity / memorisation check   (measurement only; no selection, no recommendation)")
    X, y, ids, n_null, manifest = load()
    emit(f"panel build {manifest['build_stamp']}   lightgbm {lgb.__version__}")
    emit(f"rows: role == 'fit', from_state == '{ORIGIN}', {WIN_LO} <= from_period <= {WIN_HI}")
    emit(f"rows loaded {len(y):,} (expected {EXPECTED_ROWS:,}: {'MATCH' if len(y) == EXPECTED_ROWS else 'DIFFERENT'}); "
         f"null/unknown labels dropped {n_null:,}")
    emit(f"features, matrix order ({len(FEATURES)}): {FEATURES}   categorical: {CATEGORICAL}")
    emit("NOT features: from_state (constant), loan_age, run_length, run_length_censored")
    emit(f"rounds {N_ROUNDS}, no early stopping, no validation set")

    # ------------------------------------------------------------ split
    title("1. SPLIT BY LOAN_ID")
    uniq = np.unique(ids)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(TRAIN_FRAC * len(uniq)))
    train_loans = set(uniq[perm[:n_tr]])
    in_train = np.fromiter((i in train_loans for i in ids), dtype=bool, count=len(ids))
    assert not (set(ids[in_train]) & set(ids[~in_train])), "a loan_id is on both sides"
    emit(f"seed {SEED} (numpy default_rng on the sorted unique loan_ids); train fraction {TRAIN_FRAC} of loans "
         f"-> {n_tr:,} of {len(uniq):,} loans to train, {len(uniq) - n_tr:,} held out")
    emit("no loan_id appears on both sides (asserted)")
    emit()
    Xtr, ytr, Xte, yte = X[in_train].reset_index(drop=True), y[in_train], X[~in_train].reset_index(drop=True), y[~in_train]
    absent_tr = side_report("train", ytr, n_tr)
    emit()
    absent_te = side_report("held-out", yte, len(uniq) - n_tr)
    emit()
    emit(f"row share: train {100.0 * len(ytr) / len(y):.2f}%   held-out {100.0 * len(yte) / len(y):.2f}%")
    if absent_tr or absent_te:
        emit(f"NOTE: classes absent on a side - train: {absent_tr or 'none'}; held-out: {absent_te or 'none'}. "
             "All six classes remain declared in the model (num_class = 6).")

    # ------------------------------------------------------------ prior reference
    prior = (counts(ytr) + 0.0) / len(ytr)
    Pprior_tr = np.tile(prior, (len(ytr), 1))
    Pprior_te = np.tile(prior, (len(yte), 1))
    emit()
    emit("context, not a model: log-loss of predicting the TRAIN-side class frequencies for every row: "
         f"train {logloss(Pprior_tr, ytr):.6f}   held-out {logloss(Pprior_te, yte):.6f}")

    # ------------------------------------------------------------ runs
    results = []
    for name, params in RUNS:
        title(f"RUN {name}")
        emit("params passed to lgb.train: " + ", ".join(f"{k}={v!r}" for k, v in params.items()) + f";  num_boost_round={N_ROUNDS}")
        ds = lgb.Dataset(Xtr, label=ytr, categorical_feature=CATEGORICAL, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
        assert booster.current_iteration() == N_ROUNDS, booster.current_iteration()
        assert booster.feature_name() == FEATURES
        assert booster.pandas_categorical == [CATEGORIES[c] for c in CATEGORICAL]
        text = booster.model_to_string()
        block = text.split("parameters:")[-1]
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
        Ptr, Pte = booster.predict(Xtr), booster.predict(Xte)
        assert Ptr.shape == (len(ytr), len(STATES)) and Pte.shape == (len(yte), len(STATES))
        l_tr, l_te = logloss(Ptr, ytr), logloss(Pte, yte)
        emit()
        emit(f"  multiclass log-loss, train     : {l_tr:.6f}")
        emit(f"  multiclass log-loss, held-out  : {l_te:.6f}")
        emit(f"  gap (held-out - train)         : {l_te - l_tr:+.6f}")
        emit()
        dist_table("HELD-OUT rows: actual vs predicted destination distribution", Pte, yte)
        emit()
        dist_table("TRAIN rows: actual vs predicted destination distribution", Ptr, ytr)
        results.append((name, l_tr, l_te))

    title("SUMMARY  (figures only)")
    emit(f"{'run':<42}{'train log-loss':>16}{'held-out log-loss':>20}{'gap':>12}")
    for name, a, b in results:
        emit(f"{name:<42}{a:>16.6f}{b:>20.6f}{b - a:>+12.6f}")
    emit(f"{'(prior-only reference)':<42}{logloss(Pprior_tr, ytr):>16.6f}{logloss(Pprior_te, yte):>20.6f}"
         f"{logloss(Pprior_te, yte) - logloss(Pprior_tr, ytr):>+12.6f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
