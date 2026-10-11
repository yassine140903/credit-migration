"""Label-alignment / scoring correctness probe.   STANDALONE, READ-ONLY.

    python diagnostics/43_label_alignment_check.py   # from the repo root

Writes diagnostics/43_label_alignment_check.txt and nothing else. Deliberately shares NO code with
diagnostics 40/41/42 or src/train_ml.py: the label mapping, the split, the feature matrix and every
log-loss are built here from scratch. The only project code used is src.panel_io.load_panel (the
loader) to read the data.

POPULATION. role == 'fit', from_state == 'Current', 200601 <= from_period <= 200706, ordered by
(loan_id, from_period), first 200,000 rows. Vintage-prefixed loan_ids of 2006Q1 sort before every other
partition's, so only the 2006Q1 partition is read (asserted: it holds at least 200,000 such rows).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.panel_io import load_panel                                     # noqa: E402

OUT = os.path.join("diagnostics", "43_label_alignment_check.txt")
N_ROWS = 200_000
SEED = 40
TRAIN_FRAC = 0.8
EPS = 1e-15
# The mapping, written out here (not imported): code -> state name.
CODE_TO_STATE = {0: "Current", 1: "DQ30", 2: "DQ60", 3: "DQ90+", 4: "Prepaid", 5: "Default"}
STATE_TO_CODE = {s: c for c, s in CODE_TO_STATE.items()}
ALL_CODES = [0, 1, 2, 3, 4, 5]
FEATURES = ["fico", "ltv", "dti", "rate_spread", "property_state", "est_ltv", "unemp_level", "unemp_chg_12m"]
LINES = []


def emit(line=""):
    print(line, flush=True)
    LINES.append(line)


def hand_logloss(P, y, eps=EPS):
    """-(1/n) * sum_i log(max(P[i, y[i]], eps)), one explicit loop."""
    tot = 0.0
    for i in range(len(y)):
        p = float(P[i, int(y[i])])
        if p < eps:
            p = eps
        tot += np.log(p)
    return -tot / len(y)


def main():
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    emit("Label-alignment / scoring probe   (standalone; shares no label or log-loss code with diagnostics 40-42)")
    emit(f"lightgbm {lgb.__version__}")

    # ------------------------------------------------------------ data
    d = load_panel(roles=["fit"], quarters=["2006Q1"],
                   columns=["loan_id", "from_period", "from_state", "to_state"] + FEATURES)
    fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
    d = d[(d["from_state"].astype(str) == "Current").to_numpy() & (fp >= 200601) & (fp <= 200706)].copy()
    d["loan_id"] = d["loan_id"].astype(str)
    assert len(d) >= N_ROWS, len(d)
    d = d.sort_values(["loan_id", "from_period"], kind="mergesort").head(N_ROWS).reset_index(drop=True)
    emit(f"population: role == 'fit', from_state == 'Current', 200601 <= from_period <= 200706; first {len(d):,} rows by "
         f"(loan_id, from_period): {d['loan_id'].iloc[0]} .. {d['loan_id'].iloc[-1]}, from_period "
         f"{int(d['from_period'].min())}..{int(d['from_period'].max())}, {d['loan_id'].nunique():,} loans")

    # ------------------------------------------------------------ labels
    to_str = d["to_state"].astype(str)
    present = sorted(to_str.unique())
    unmapped = [s for s in present if s not in STATE_TO_CODE]
    assert not unmapped, f"to_state values not in the mapping: {unmapped}"
    y = to_str.map(STATE_TO_CODE).to_numpy(dtype=np.int64)
    emit()
    emit("STEP 2. LABEL MAPPING (built in this script)")
    emit(f"distinct to_state values present, sorted: {present}")
    emit("code -> state, as passed to LightGBM: " + ", ".join(f"{c}={s}" for c, s in CODE_TO_STATE.items()))
    emit("rows per code (whole subsample): " + ", ".join(
        f"{c}={CODE_TO_STATE[c]}: {int((y == c).sum()):,}" for c in ALL_CODES))
    # spot-check the mapping row by row against the raw strings
    assert all(CODE_TO_STATE[int(c)] == s for c, s in zip(y[:5000], to_str.iloc[:5000]))

    # ------------------------------------------------------------ features
    prop = d["property_state"].astype(object)
    prop = prop.where(prop.notna(), None)
    cats = sorted({v for v in prop if v is not None})
    ctype = pd.CategoricalDtype(cats)
    X = pd.DataFrame({c: (prop.astype(ctype) if c == "property_state" else d[c].to_numpy(dtype=np.float64, na_value=np.nan))
                      for c in FEATURES})

    # ------------------------------------------------------------ split
    loans = np.array(sorted(d["loan_id"].unique()))
    perm = np.random.default_rng(SEED).permutation(len(loans))
    n_tr = int(round(TRAIN_FRAC * len(loans)))
    train_loans = set(loans[perm[:n_tr]])
    is_tr = d["loan_id"].isin(train_loans).to_numpy()
    assert not (set(d.loc[is_tr, "loan_id"]) & set(d.loc[~is_tr, "loan_id"]))
    Xtr, ytr, Xte, yte = X[is_tr].reset_index(drop=True), y[is_tr], X[~is_tr].reset_index(drop=True), y[~is_tr]
    meta_te = d.loc[~is_tr, ["loan_id", "from_period"]].reset_index(drop=True)
    emit()
    emit(f"split by loan_id, seed {SEED}, {TRAIN_FRAC:.0%} of loans: train {len(ytr):,} rows / {n_tr:,} loans; "
         f"held-out {len(yte):,} rows / {len(loans) - n_tr:,} loans; no loan on both sides (asserted)")

    # ------------------------------------------------------------ train
    params = dict(objective="multiclass", num_class=6, verbosity=-1)
    booster = lgb.train(params, lgb.Dataset(Xtr, label=ytr, categorical_feature=["property_state"]), num_boost_round=100)
    head = booster.model_to_string().split("\n", 12)[:12]
    nc = [ln for ln in head if ln.startswith("num_class=")]
    emit(f"trained: objective=multiclass num_class=6 (LightGBM defaults otherwise), 100 rounds; model header: {nc}, "
         f"trees {booster.num_trees()}")
    Ptr, Pte = booster.predict(Xtr), booster.predict(Xte)
    assert Pte.shape == (len(yte), 6)
    emit(f"predict output shape {Pte.shape}: column j = probability of code j = {CODE_TO_STATE}")
    emit("column check: mean predicted probability per column vs train frequency of that code (alignment sanity)")
    for j in ALL_CODES:
        emit(f"  column {j} ({CODE_TO_STATE[j]:<8}) mean pred {Ptr[:, j].mean():.6f}   train freq {(ytr == j).mean():.6f}")
    emit("  mean predicted probability of the ACTUAL class, train rows with actual Current: "
         f"column 0 = {Ptr[ytr == 0, 0].mean():.6f}; any other column mean on those rows: "
         + ", ".join(f"{CODE_TO_STATE[j]} {Ptr[ytr == 0, j].mean():.6f}" for j in ALL_CODES[1:]))
    assert abs(Pte.sum(axis=1) - 1).max() < 1e-6

    # ------------------------------------------------------------ step 3
    emit()
    emit("STEP 3. HAND-CHECK, FIVE HELD-OUT ROWS (probabilities prefixed by state name, this script's mapping order)")
    picks = []
    for c in ALL_CODES:
        idx = np.flatnonzero(yte == c)
        if len(idx) and (c == 0 or len(picks) < 5):
            picks.append(int(idx[0]))
    picks = picks[:1] + [i for i in picks[1:]]
    others = [c for c in ALL_CODES[1:] if (yte == c).any()]
    picks = [int(np.flatnonzero(yte == 0)[0])] + [int(np.flatnonzero(yte == c)[0]) for c in others][:4]
    while len(picks) < 5:                                  # fewer than four other classes present: more Current rows
        picks.append(int(np.flatnonzero(yte == 0)[len(picks)]))
        emit(f"  (only {len(others)} non-Current classes present in held-out; padded with another Current row)")
    for i in picks:
        probs = " ".join(f"{CODE_TO_STATE[j]}={Pte[i, j]:.6f}" for j in ALL_CODES)
        top = CODE_TO_STATE[int(Pte[i].argmax())]
        emit(f"  {meta_te['loan_id'][i]}, {int(meta_te['from_period'][i])}, actual={CODE_TO_STATE[int(yte[i])]}, {probs}  -> highest: {top}")

    # ------------------------------------------------------------ step 4
    emit()
    emit("STEP 4. HELD-OUT LOG-LOSS, THREE WAYS")
    a = float(log_loss(yte, Pte, labels=ALL_CODES))
    b = hand_logloss(Pte, yte)
    prior = np.array([(ytr == j).mean() for j in ALL_CODES])
    c = hand_logloss(np.tile(prior, (len(yte), 1)), yte)
    emit(f"  (a) sklearn.metrics.log_loss(labels={ALL_CODES}) : {a:.6f}")
    emit(f"  (b) hand-written loop on the model's predictions : {b:.6f}")
    emit(f"  (c) prior-only (train-side class frequencies), hand-written : {c:.6f}")
    if round(a, 6) == round(b, 6):
        emit("  (a) and (b) AGREE to six decimal places.")
    else:
        emit("  !!! (a) and (b) DISAGREE at six decimal places - THIS DISAGREEMENT IS THE FINDING !!!")
    emit(f"  context, train rows: model (hand-written) {hand_logloss(Ptr, ytr):.6f}   prior-only {hand_logloss(np.tile(prior, (len(ytr), 1)), ytr):.6f}")
    emit(f"  held-out model minus prior-only: {b - c:+.6f}")

    # attribution of the (a)/(b) gap: which rows carry the loss, and does the clipping floor explain it
    pa = Pte[np.arange(len(yte)), yte]
    emit()
    emit("  4b. WHERE THE HELD-OUT LOSS COMES FROM (model predictions, probability of the actual class)")
    for thr in (1e-3, 1e-6, 1e-10, EPS):
        m = pa < thr
        emit(f"    rows with p(actual) < {thr:g}: {int(m.sum()):,}   by actual class: "
             + (", ".join(f"{CODE_TO_STATE[j]} {int((m & (yte == j)).sum()):,}" for j in ALL_CODES if (m & (yte == j)).any()) or "none"))
    emit(f"    p(actual): min {pa.min():.3e}   1st pct {np.percentile(pa, 1):.3e}   median {np.median(pa):.6f}   exactly 0.0: {int((pa == 0).sum()):,}")
    emit(f"    contribution to (b) from rows with p(actual) < 1e-6: {-np.sum(np.log(np.maximum(pa[pa < 1e-6], EPS))) / len(yte):.6f} of {b:.6f}")
    emit(f"    hand loop with floor 1e-15: {hand_logloss(Pte, yte, 1e-15):.6f}   floor 2.22e-16 (float64 eps): "
         f"{hand_logloss(Pte, yte, np.finfo(np.float64).eps):.6f}   floor 1e-300: {hand_logloss(Pte, yte, 1e-300):.6f}")
    bad = pa < 1e-6
    emit("    on the rows with p(actual) < 1e-6, the class holding the highest probability: "
         + ", ".join(f"{CODE_TO_STATE[j]} {int((Pte[bad].argmax(axis=1) == j).sum()):,}" for j in ALL_CODES if (Pte[bad].argmax(axis=1) == j).any())
         + f"   (their mean highest probability {Pte[bad].max(axis=1).mean():.6f})")
    emit("    for comparison, classes with ZERO train rows: " + (", ".join(CODE_TO_STATE[j] for j in ALL_CODES if not (ytr == j).any()) or "none"))
    emit(f"    sklearn version {__import__('sklearn').__version__}; sklearn on the same predictions with clipping done by me at 1e-15: "
         f"{float(log_loss(yte, np.clip(Pte, EPS, 1 - EPS), labels=ALL_CODES, normalize=True)):.6f}")

    # ------------------------------------------------------------ step 5
    emit()
    emit("STEP 5. CLASS PRESENCE")
    in_tr = [CODE_TO_STATE[j] for j in ALL_CODES if (ytr == j).any()]
    in_te = [CODE_TO_STATE[j] for j in ALL_CODES if (yte == j).any()]
    emit(f"  train    : present {in_tr}; absent {[s for s in STATE_TO_CODE if s not in in_tr] or 'none'}   counts "
         + ", ".join(f"{CODE_TO_STATE[j]} {int((ytr == j).sum()):,}" for j in ALL_CODES))
    emit(f"  held-out : present {in_te}; absent {[s for s in STATE_TO_CODE if s not in in_te] or 'none'}   counts "
         + ", ".join(f"{CODE_TO_STATE[j]} {int((yte == j).sum()):,}" for j in ALL_CODES))
    emit("  handling of absent classes:")
    emit("    (a) sklearn: labels= lists all six codes, so a class absent from held-out y_true is accepted; it contributes "
         "no rows to the average, and its probability column still takes part in sklearn's row normalisation.")
    emit(f"    (b) hand loop: only the probability of each row's actual class is read, so an absent class contributes nothing; "
         f"p is floored at {EPS:g} before the log.")
    emit("    (c) prior-only: a class absent from train gets prior probability 0; it would only matter if that class occurs in "
         f"held-out, where p would be floored at {EPS:g}. Held-out classes with train frequency 0: "
         f"{[CODE_TO_STATE[j] for j in ALL_CODES if (yte == j).any() and not (ytr == j).any()] or 'none'}.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
