"""Train one one-month multiclass LightGBM transition model for the ML arm.

    python -m src.train_ml                                        # pooled: all four transient origin states
    python -m src.train_ml --origin-states Current                # one origin state (grouped architecture)
    python -m src.train_ml --from-period-max 200706 --name pooled_to200706
    python -m src.train_ml --with-loan-age                        # the with/without ablation arm

Rows: role == 'fit', from_state in --origin-states, from_period <= --from-period-max (default: the whole fit window).
Label: to_state, as integer codes into the fixed class order panel.STATES; all six classes are declared whether or not
they occur in the labels.

Features, in matrix order: fico, ltv, dti, rate_spread, property_state, est_ltv, unemp_level, unemp_chg_12m, plus
from_state when more than one origin state is trained on (it is constant for a single-origin model and is then omitted).
NOT loan_age (it is added, at the same position in the same code path, only by --with-loan-age: the fit window is 36
months and the forecast 60, so a forecaster would hold almost every loan at the training maximum), NOT run_length, NOT
run_length_censored. property_state and from_state are native LightGBM categoricals with a fixed, stated category list
(no integer or FIPS encoding). Nulls are left as-is, numerics are float32.

LightGBM: multiclass, CPU, 100 rounds, library defaults, no tuning, no early stopping, no validation set. The only
parameters that can be changed from the command line are the two leaf-value bounds of diagnostic 49,
--min-sum-hessian-in-leaf and --lambda-l2; left unset (the default) they are not passed and the library defaults apply,
so every earlier invocation trains exactly as before. When set they are written to the sidecar ("lgb_params").

Two passes, as in diagnostics/34_thinned_training.py: a label scan that reads only the state columns, then the feature
read that builds the matrix one partition at a time; the matrix is handed to a Dataset whose raw reference is dropped
before it is constructed, so the matrix and the binned data are not both held by the caller.

Writes artifacts/ml/<name>/model.txt and artifacts/ml/<name>/model.json. The sidecar records the feature list in matrix
order, the class order, the origin states, the training window and the row count. It records the min and max loan_age
present in the training rows as PROVENANCE ("loan_age_in_training_rows"), whether or not loan_age is a feature. The clamp
bounds src.forecast applies ("clamp_bounds", feature -> [min, max]) are written only for a feature the model actually
reads, i.e. only with --with-loan-age. Nothing under data/ is changed. No predictive metric is computed or printed.
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src import panel, panel_io
from src.memwatch import MemWatch

STATES = list(panel.STATES)
TRANSIENT = STATES[:4]
BASE_FEATURES = ["fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv", "unemp_level", "unemp_chg_12m"]
LOAN_AGE_AFTER = "property_state"       # --with-loan-age puts loan_age right after this feature (its old position)
ART_DIR = os.path.join("artifacts", "ml")
N_ROUNDS = 100
GB = 2 ** 30


def say(s=""):
    print(s, flush=True)


def resolve(args):
    """Origin states (in class order), the feature list, which of them are categorical, and their categories."""
    origin = [s for s in STATES if s in set(args.origin_states)]
    pooled = len(origin) > 1
    features = list(BASE_FEATURES)
    if args.with_loan_age:
        features.insert(features.index(LOAN_AGE_AFTER) + 1, "loan_age")
    features += ["from_state"] if pooled else []
    categorical = ["property_state"] + (["from_state"] if pooled else [])
    categories = {"property_state": list(panel.PROPERTY_STATES)}
    if pooled:
        categories["from_state"] = list(origin)
    name = args.name
    if not name:
        name = "pooled" if origin == TRANSIENT else "origin_" + "_".join(o.replace("+", "plus") for o in origin)
        name += "_with_loan_age" if args.with_loan_age else "_no_loan_age"
        if args.from_period_max is not None:
            name += "_to%d" % args.from_period_max
    return origin, pooled, features, categorical, categories, name


def keep_rows(d, origin, fp_max):
    """Row mask: origin state in the chosen set, from_period within the window, and a valid label."""
    m = d["from_state"].isin(origin).to_numpy()
    if fp_max is not None:
        m &= d["from_period"].to_numpy(dtype=np.int64, na_value=0) <= fp_max
    y = d["to_state"].cat.set_categories(STATES).cat.codes.to_numpy().astype(np.int64)
    return m, y


def label_scan(quarters, origin, fp_max):
    """Pass 1: per-partition row counts and the training population's summary, from the state columns only."""
    S = {"rows_by_quarter": {}, "label_counts": np.zeros(len(STATES), np.int64), "from_counts": {o: 0 for o in origin},
         "n_null_label": 0, "age_min": None, "age_max": None, "fp_min": None, "fp_max": None}
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=["from_state", "to_state", "from_period", "loan_age"],
                                quarters=[q])
        m, y = keep_rows(d, origin, fp_max)
        S["n_null_label"] += int((m & (y < 0)).sum())
        m &= y >= 0
        S["rows_by_quarter"][q] = int(m.sum())
        S["label_counts"] += np.bincount(y[m], minlength=len(STATES))
        for o, c in d.loc[m, "from_state"].astype(str).value_counts().items():
            S["from_counts"][o] += int(c)
        if m.any():
            age = d["loan_age"].to_numpy(dtype=np.int64, na_value=0)[m]
            fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)[m]
            for key, v, fn in (("age_min", age.min(), min), ("age_max", age.max(), max), ("fp_min", fp.min(), min),
                               ("fp_max", fp.max(), max)):
                S[key] = int(v) if S[key] is None else fn(S[key], int(v))
        say("  scan %s: %s training rows" % (q, format(S["rows_by_quarter"][q], ",")))
        del d, m, y
    return S


def build_matrix(quarters, origin, fp_max, features, categories, rows_by_quarter):
    """Pass 2: the float32 feature matrix and the label, one partition at a time."""
    chunks, ys = [], []
    cols = list(dict.fromkeys(features + ["from_state", "to_state", "from_period"]))
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=cols, quarters=[q])
        m, y = keep_rows(d, origin, fp_max)
        m &= y >= 0
        if int(m.sum()) != rows_by_quarter[q]:
            raise RuntimeError("%s: pass 2 kept %d rows, pass 1 counted %d" % (q, int(m.sum()), rows_by_quarter[q]))
        d = d[m]
        data = {}
        for c in features:
            if c in categories:
                data[c] = d[c].cat.set_categories(categories[c]).reset_index(drop=True)
            else:
                data[c] = d[c].to_numpy(dtype=np.float32, na_value=np.nan)
        chunks.append(pd.DataFrame(data))
        ys.append(y[m].astype(np.float32))
        del d, m, y, data
        gc.collect()
        say("  matrix %s: %s rows" % (q, format(len(chunks[-1]), ",")))
    X = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    for c in categories:
        if str(X[c].dtype) != "category" or list(X[c].cat.categories) != categories[c]:
            raise RuntimeError("%s lost its fixed category list in the concat" % c)
    return X, np.concatenate(ys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", help="artifacts/ml/<name>/ (default: 'pooled' or the origin states, plus _no_loan_age / "
                                   "_with_loan_age, plus _to<YYYYMM>)")
    ap.add_argument("--with-loan-age", action="store_true",
                    help="train WITH loan_age as a feature (the ablation arm); the default leaves it out")
    ap.add_argument("--origin-states", nargs="+", choices=TRANSIENT, default=TRANSIENT, metavar="STATE",
                    help="from_state values to train on (default: all four transient states)")
    ap.add_argument("--from-period-max", type=int, default=None, metavar="YYYYMM",
                    help="last from_period to train on, inclusive (default: the whole fit window)")
    ap.add_argument("--min-sum-hessian-in-leaf", type=float, default=None, metavar="X",
                    help="LightGBM min_sum_hessian_in_leaf (default: not passed, library default 1e-3)")
    ap.add_argument("--lambda-l2", type=float, default=None, metavar="X",
                    help="LightGBM lambda_l2 (default: not passed, library default 0)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing artifacts/ml/<name>/")
    args = ap.parse_args(argv)

    origin, pooled, features, categorical, categories, name = resolve(args)
    out_dir = os.path.join(ART_DIR, name)
    model_file, side_file = os.path.join(out_dir, "model.txt"), os.path.join(out_dir, "model.json")
    if os.path.exists(model_file) and not args.overwrite:
        raise SystemExit("%s exists; pass --overwrite or choose --name" % model_file)

    import lightgbm as lgb
    manifest = panel_io.read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    say("model name:            %s" % name)
    say("origin states:         %s" % origin)
    say("from_state feature:    %s" % ("INCLUDED (several origin states)" if pooled
                                       else "OMITTED (single origin state: it is constant)"))
    say("loan_age feature:      %s" % ("INCLUDED (--with-loan-age)" if args.with_loan_age else "OMITTED (default)"))
    say("features, matrix order: %s" % features)
    say("categorical features:  %s" % categorical)
    say("class order:           %s" % STATES)
    say("training window:       role == fit, from_period <= %s" % (args.from_period_max or "(whole fit window)"))

    watch = MemWatch(on_abort=lambda msg: say("ABORTED, run did not complete: " + msg))
    watch.start()
    t_all = time.perf_counter()

    watch.phase = "pass 1: label scan"
    S = label_scan(quarters, origin, args.from_period_max)
    n_rows = int(sum(S["rows_by_quarter"].values()))
    if n_rows == 0:
        raise SystemExit("no training rows for origin states %s and from_period <= %s" % (origin, args.from_period_max))
    say("training rows %s; null/unknown labels dropped %s; loan_age in the rows (provenance) %s .. %s; from_period %s .. %s" % (
        format(n_rows, ","), format(S["n_null_label"], ","), S["age_min"], S["age_max"], S["fp_min"], S["fp_max"]))
    for i, st in enumerate(STATES):
        n = int(S["label_counts"][i])
        say("  label %-8s %14s%s" % (st, format(n, ","), "   <- absent from the training labels; class kept" if n == 0 else ""))

    watch.phase = "pass 2: matrix"
    X, y = build_matrix(quarters, origin, args.from_period_max, features, categories, S["rows_by_quarter"])
    matrix_bytes = int(X.memory_usage(deep=True).sum())
    say("matrix %s x %d, %.2f GB" % (format(len(X), ","), X.shape[1], matrix_bytes / GB))

    watch.phase = "Dataset construction"
    params = {"objective": "multiclass", "num_class": len(STATES), "device_type": "cpu", "verbosity": -1}
    if args.min_sum_hessian_in_leaf is not None:
        params["min_sum_hessian_in_leaf"] = args.min_sum_hessian_in_leaf
    if args.lambda_l2 is not None:
        params["lambda_l2"] = args.lambda_l2
    say("lightgbm params:       %s" % params)
    ds = lgb.Dataset(X, label=y, categorical_feature=categorical, free_raw_data=True,
                     params=params if len(params) > 4 else None)
    del X, y
    gc.collect()
    t0 = time.perf_counter()
    ds.construct()
    construct_s = time.perf_counter() - t0
    gc.collect()
    say("Dataset constructed in %.1f s" % construct_s)

    watch.phase = "training"
    t0 = time.perf_counter()
    booster = lgb.train(params, ds, num_boost_round=N_ROUNDS)
    train_s = time.perf_counter() - t0
    watch.stop()

    if booster.feature_name() != features:
        raise RuntimeError("model features %s != %s" % (booster.feature_name(), features))
    if booster.current_iteration() != N_ROUNDS:
        raise RuntimeError("trained %d rounds, expected %d" % (booster.current_iteration(), N_ROUNDS))
    saved = booster.pandas_categorical
    if saved != [categories[c] for c in categorical]:
        raise RuntimeError("model's saved category lists differ from the declared ones")

    os.makedirs(out_dir, exist_ok=True)
    booster.save_model(model_file)
    side = {"name": name, "features": features, "categorical_features": categorical, "categories": categories,
            "class_order": STATES, "origin_states": origin, "from_state_is_feature": pooled,
            "training_window": {"role": "fit", "from_period_max_arg": args.from_period_max,
                                "from_period_min_observed": S["fp_min"], "from_period_max_observed": S["fp_max"]},
            "n_rows": n_rows, "label_counts": {st: int(S["label_counts"][i]) for i, st in enumerate(STATES)},
            "from_state_counts": S["from_counts"], "loan_age_is_feature": args.with_loan_age,
            "loan_age_in_training_rows": {"min": S["age_min"], "max": S["age_max"],
                                          "note": "PROVENANCE ONLY: the loan_age range present in the training rows; "
                                                  "not a clamp bound unless loan_age is a feature"},
            "clamp_bounds": ({"loan_age": [S["age_min"], S["age_max"]]} if args.with_loan_age else {}),
            "n_rounds": N_ROUNDS, "objective": "multiclass", "lgb_params": params, "lightgbm_version": lgb.__version__,
            "panel_build_stamp": manifest["build_stamp"], "matrix_bytes": matrix_bytes,
            "dataset_construct_seconds": round(construct_s, 2), "train_seconds": round(train_s, 2),
            "peak_rss_bytes": int(watch.peak_rss())}
    with open(side_file, "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2)
        fh.write("\n")

    say("")
    say("training wall-clock:   %.1f s (%.1f min) for %d rounds, %.2f s/round;  Dataset construction %.1f s" % (
        train_s, train_s / 60, N_ROUNDS, train_s / N_ROUNDS, construct_s))
    say("total wall-clock:      %.1f s" % (time.perf_counter() - t_all))
    say("peak RSS:              %.2f GB   min system available %.2f GB" % (watch.peak_rss() / GB, watch.min_avail / GB))
    if args.with_loan_age:
        say("loan_age clamp bounds written to the sidecar: [%d, %d]" % (S["age_min"], S["age_max"]))
    else:
        say("loan_age is not a feature: no clamp bound written (training rows span %d .. %d, recorded as provenance)" % (
            S["age_min"], S["age_max"]))
    say("saved: %s\n       %s" % (model_file.replace(os.sep, "/"), side_file.replace(os.sep, "/")))


if __name__ == "__main__":
    sys.exit(main())
