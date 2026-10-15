"""Diagnostic 56: the calibration layer for the ML arm.   Writes diagnostics/56_calibration_report.txt.

    python diagnostics/56_calibration_report.py header   # BEFORE anything is fitted: pre-registration, artefact, sidecar, folds, methods
    python diagnostics/56_calibration_report.py report   # after src.calibrate oof + fit: fold check, then every calibration number

`header` writes artifacts/calibration/<name>/header.txt once and refuses to overwrite it; `report` prepends that file verbatim, so the
pre-registration in the report is the one written before the first fit. No evaluate row is read, no 60-month forecast is rerun.
"""

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src import calibrate as C                                          # noqa: E402
from src.train_ml import STATES, TRANSIENT                              # noqa: E402

MODEL = os.path.join("artifacts", "ml", "headline_pooled")
OUT = os.path.join("diagnostics", "56_calibration_report.txt")
BUILD_LOG = os.path.join("diagnostics", "54_headline_forecaster_build.txt")
BASELINE_COUNTS = os.path.join("artifacts", "baseline_headline", "counts.csv")
W = 120

PREREG = ("Calibration is reported with its before-and-after on BOTH the ML arm's absolute error AND the ML-minus-baseline gap, "
          "whichever direction each moves. Calibration may improve the ML arm's accuracy while shrinking its advantage over the "
          "homogeneous baseline; both numbers are reported and neither is selected after the fact.")

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


def model_params(path):
    """min_sum_hessian_in_leaf and lambda_l2 as the model FILE records them (older sidecars do not carry lgb_params)."""
    txt = open(os.path.join(path, "model.txt"), encoding="utf-8", errors="replace").read()
    g = lambda k: float(re.search(r"\[%s: ([^\]]+)\]" % k, txt).group(1))
    return g("min_sum_hessian_in_leaf"), g("lambda_l2")


# ==========================================================================
# header (written before any fitting)
# ==========================================================================
def build_header():
    side = json.load(open(os.path.join(MODEL, "model.json"), encoding="utf-8"))
    sha = C.sha256_file(os.path.join(MODEL, "model.txt"))
    build = open(BUILD_LOG, encoding="utf-8").read()
    named = re.search(r"%s\s+sha256 ([0-9a-f]+)" % re.escape(MODEL.replace(os.sep, "/")), build)
    emit("=" * W)
    emit("DIAGNOSTIC 56 - CALIBRATION LAYER FOR THE ML ARM (one-step, fit rows only)")
    emit("=" * W)
    emit()
    emit("PRE-REGISTRATION (written before anything was computed)")
    rule()
    emit('"%s"' % PREREG)
    emit()
    emit("In this milestone, 'absolute error' is read two ways and both are reported: (a) the one-step multiclass Brier score per origin")
    emit("state, and (b) |predicted share - realised share| per class in percentage points. 'ML-minus-baseline gap' is the ML Brier minus the")
    emit("homogeneous baseline's Brier. The baseline's one-step prediction for an origin state is its row of the transition matrix; fitted on")
    emit("these same fit rows it IS the origin-conditional prior (checked against %s in the report body). The 60-month" % BASELINE_COUNTS.replace(os.sep, "/"))
    emit("chained forecast is NOT rerun here; its absolute error and gap are the next milestone.")
    emit()
    emit("THE MODEL ARTEFACT BEING CALIBRATED")
    rule()
    emit("path    : %s/model.txt   (sidecar %s/model.json)" % (MODEL.replace(os.sep, "/"), MODEL.replace(os.sep, "/")))
    emit("sha256  : %s" % sha)
    emit("          diagnostics/54_headline_forecaster_build.txt names this model with sha256 prefix %s : %s" % (
        named.group(1) if named else "(not found)", "MATCH" if named and sha.startswith(named.group(1)) else "NO MATCH"))
    emit()
    emit("Why this one of the directories under artifacts/ml/ (read from each sidecar and from the parameters recorded in each model.txt):")
    rows = []
    for d in sorted(glob.glob(os.path.join("artifacts", "ml", "*"))):
        sd = json.load(open(os.path.join(d, "model.json"), encoding="utf-8"))
        h, l2 = model_params(d)
        why = []
        if "loan_age" in sd["features"]:
            why.append("loan_age is a feature")
        if sd["origin_states"] != TRANSIENT:
            why.append("origin states %s only" % sd["origin_states"])
        if sd["training_window"]["from_period_max_arg"] is not None:
            why.append("trained to from_period <= %s only" % sd["training_window"]["from_period_max_arg"])
        if (h, l2) != (10.0, 10.0):
            why.append("min_sum_hessian_in_leaf %g, lambda_l2 %g" % (h, l2))
        rows.append((os.path.basename(d), sd["n_rows"], len(sd["features"]), h, l2, "; ".join(why) or "none: matches the brief"))
    emit("%-26s%12s%10s%10s%10s   %s" % ("directory", "rows", "features", "hessian", "lambda_l2", "disqualified by"))
    for r in rows:
        emit("%-26s%12s%10d%10g%10g   %s" % (r[0], f(r[1]), r[2], r[3], r[4], r[5]))
    emit("Exactly one directory matches (nine features, no loan_age, no run_length, all four origins, whole fit window, hessian 10, lambda_l2 10).")
    emit()
    emit("SIDECAR (artifacts/ml/headline_pooled/model.json), verbatim:")
    rule()
    for line in json.dumps(side, indent=2).splitlines():
        emit(line)
    emit()
    emit("FOLDS")
    rule()
    emit("number of folds  : %d" % C.N_FOLDS)
    emit("seed             : %d" % C.FOLD_SEED)
    emit("assignment method: %s" % C.FOLD_METHOD)
    emit("each fold's model: trained on the other four folds' fit rows with the sidecar's own features, categories, lgb_params and %d rounds" % C.N_ROUNDS)
    emit("                   (identical configuration to the shipped model; only the training rows differ).")
    emit("KNOWN AND ACCEPTED LIMITATION: folds are by loan, not by time, so a 2007 row may be predicted by a model trained on other loans' 2008")
    emit("rows. This step measures calibration drift, not performance; by-loan folding is deliberate.")
    emit()
    emit("CALIBRATION METHODS")
    rule()
    emit("Four separate corrections, one per transient origin state (Current, DQ30, DQ60, DQ90+; DQ60 has its own), each fitted on that state's")
    emit("out-of-fold rows, in two steps:")
    emit("  STEP ONE  stay vs leave   : ISOTONIC regression of 1[to_state == origin] on the model's mass on the origin state.")
    emit("            Used here because the cells are huge (Current: ~39M rows split roughly 98/2), so no bin is thin enough to chase noise.")
    emit("  STEP TWO  destination split among leavers only: the model's non-origin mass renormalised to one, then PLATT (sigmoid) scaling,")
    emit("            expit(a * logit(q) + b), fitted one-vs-rest per destination class and renormalised within step two.")
    emit("            Used here because the cells are thin (Default is 0.0179% of pooled rows) and a staircase would fit noise and chain it sixty times.")
    emit("  ASSEMBLE  row = [stay_prob on origin] + (1 - stay_prob) x [calibrated destination split]; sums to one by construction, asserted to")
    emit("            1e-9 in float64 on every row, never renormalised afterwards.")
    emit("The NAIVE scheme (comparison only): all six classes calibrated one-vs-rest on their own predicted probability, then the row renormalised.")
    emit("The brief does not name its per-class method, so it is built both ways, isotonic per class and Platt per class; neither is preferred.")
    emit("Numerics fixed in advance: logit() clips its argument to [%g, 1-%g]; the log-loss probability floor is %g (11 Oct decision);" % (
        C.LOGIT_EPS, C.LOGIT_EPS, C.FLOOR))
    emit("a destination with under 100 leaver events at an origin is flagged THIN.")
    emit("The correction is fitted on out-of-fold predictions and applied to the shipped model's IN-SAMPLE predictions on the same fit rows (the")
    emit("measurement the brief specifies); the same measurement on the OOF predictions is also given, as a supplement.")
    emit("No evaluate row is read.")
    return "\n".join(LINES) + "\n"


def run_header():
    out = C.cal_dir(MODEL)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "header.txt")
    if os.path.exists(path):
        raise SystemExit("%s exists; the pre-registration is written once and not overwritten" % path)
    text = build_header()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("header written to %s" % path)


# ==========================================================================
# measurement
# ==========================================================================
K = len(STATES)


class Acc:
    """Running sums for one (origin, prediction set): n, Brier, log-loss with the floor, floor hits, per-class mass."""

    def __init__(self):
        self.n = 0
        self.brier = 0.0
        self.ll = 0.0
        self.floor_hits = 0
        self.pred = np.zeros(K)
        self.exact01 = 0                               # rows whose mass on the origin state is exactly 0 or exactly 1

    def update(self, P, y, oi=None):
        pt = P[np.arange(len(y)), y]
        self.n += len(y)
        self.brier += float(((P ** 2).sum(axis=1) - 2.0 * pt + 1.0).sum())
        self.floor_hits += int((pt < C.FLOOR).sum())
        self.ll += float(-np.log(np.clip(pt, C.FLOOR, 1.0)).sum())
        self.pred += P.sum(axis=0)
        if oi is not None:
            self.exact01 += int(((P[:, oi] == 0.0) | (P[:, oi] == 1.0)).sum())

    def merge(self, o):
        self.n += o.n
        self.brier += o.brier
        self.ll += o.ll
        self.floor_hits += o.floor_hits
        self.pred += o.pred
        self.exact01 += o.exact01

    @property
    def b(self):
        return self.brier / self.n

    @property
    def l(self):
        return self.ll / self.n


def chunks(idx):
    for s in range(0, len(idx), C.CHUNK):
        yield idx[s:s + C.CHUNK]


def measure_stack(stack, meta, corr, naive=None):
    """Per origin: Acc for before / after (and the prior, and both naive variants when `naive` is given), the true-class counts, and
    the decomposed-vs-naive cell differences."""
    res = {}
    fs, y_all = meta["from_state"], meta["y"]
    for o in TRANSIENT:
        oi = C.IDX[o]
        idx = np.flatnonzero(fs == oi)
        true_n = np.bincount(y_all[idx].astype(np.int64), minlength=K)
        prior = true_n / true_n.sum()
        r = {"true_n": true_n, "prior": prior, "before": Acc(), "after": Acc(), "prior_acc": Acc(), "n": len(idx)}
        if naive is not None:
            for kind in ("isotonic", "platt"):
                r[kind] = {"acc": Acc(), "maxdiff": [], "colmax": np.zeros(K), "dropped": 0}
        first = True
        for ii in chunks(idx):
            P = np.asarray(stack[ii])
            y = y_all[ii].astype(np.int64)
            A = C.apply_correction(P, o, corr)
            if first:                                   # the scorer used everywhere else in the project, once, as a cross-check
                from src import scoring
                for nm, M in (("before", P), ("after", A)):
                    d = abs(scoring.brier(M, y) - float(((M ** 2).sum(axis=1) - 2.0 * M[np.arange(len(y)), y] + 1.0).mean()))
                    r.setdefault("scoring_check", []).append(d)
                first = False
            r["before"].update(P, y, oi)
            r["after"].update(A, y, oi)
            r["prior_acc"].update(np.tile(prior, (len(y), 1)), y, oi)
            if naive is not None:
                for kind in ("isotonic", "platt"):
                    N, bad = C.apply_naive(P, o, naive, kind)
                    ok = ~np.isnan(N[:, 0])
                    r[kind]["dropped"] += bad
                    D = np.abs(A[ok] - N[ok])
                    r[kind]["maxdiff"].append(D.max(axis=1).astype(np.float32))
                    r[kind]["colmax"] = np.maximum(r[kind]["colmax"], D.max(axis=0))
                    r[kind]["acc"].update(N[ok], y[ok], oi)
            del P, A
        res[o] = r
        print("  measured %s (%s rows)" % (o, f(len(idx))), flush=True)
    return res


def total(res, key):
    t = Acc()
    for o in TRANSIENT:
        t.merge(res[o][key])
    return t


def render_measure(res, label, intro):
    title(label)
    for line in intro:
        emit(line)
    emit()
    emit("A. BRIER (multiclass, mean over rows of sum_k (p_k - 1[true = k])^2) AND LOG-LOSS, per origin state. Gap = ML Brier - prior Brier")
    emit("   (negative: the ML arm beats the origin-conditional prior). 'Floor hits' = rows whose probability on the true class is below %g," % C.FLOOR)
    emit("   clipped to %g in the log-loss." % C.FLOOR)
    rule()
    emit("%-8s%13s | %9s %9s %9s | %10s %10s | %9s %9s %9s | %9s %9s | %-34s" % (
        "origin", "rows", "Brier", "Brier", "Brier", "gap", "gap", "logloss", "logloss", "logloss", "floor", "floor",
        "calibration moved the ML Brier"))
    emit("%-8s%13s | %9s %9s %9s | %10s %10s | %9s %9s %9s | %9s %9s | %-34s" % (
        "", "", "prior", "before", "after", "before", "after", "prior", "before", "after", "hits bef", "hits aft", ""))
    rule()
    verdicts = {}
    rows = [(o, res[o]["prior_acc"], res[o]["before"], res[o]["after"]) for o in TRANSIENT]
    rows.append(("ALL", total(res, "prior_acc"), total(res, "before"), total(res, "after")))
    for o, pr, bf, af in rows:
        gb, ga = bf.b - pr.b, af.b - pr.b
        toward = abs(af.b - pr.b) < abs(bf.b - pr.b)
        verdicts[o] = (gb, ga, toward)
        emit("%-8s%13s | %9.6f %9.6f %9.6f | %+10.6f %+10.6f | %9.6f %9.6f %9.6f | %9s %9s | %s" % (
            o, f(pr.n), pr.b, bf.b, af.b, gb, ga, pr.l, bf.l, af.l, f(bf.floor_hits), f(af.floor_hits),
            "%s the prior (|gap| %.6f -> %.6f); ML Brier %s" % ("TOWARD" if toward else "AWAY FROM", abs(gb), abs(ga),
                                                              "improved" if af.b < bf.b else "worsened")))
    rule()
    emit("Rows whose calibrated mass on the origin state is exactly 0 or exactly 1 (isotonic can return either; left as fitted):")
    emit("   " + "   ".join("%s %s" % (o, f(res[o]["after"].exact01)) for o in TRANSIENT))
    emit("Cross-check: |Brier here - src.scoring.brier| on the first chunk of each origin, worst over before/after: %.2e" %
         max(max(res[o]["scoring_check"]) for o in TRANSIENT))
    emit()
    emit("B. PREDICTED vs REALISED SHARE, per origin state and class (shares of that origin's rows; the realised share IS the origin-conditional")
    emit("   prior, since the prior is the class frequencies of these same fit rows). THIN = fewer than 100 realised events.")
    for o in TRANSIENT:
        r = res[o]
        emit()
        emit("origin %s   (%s rows)" % (o, f(r["n"])))
        rule()
        emit("%-9s%13s%12s | %12s %12s | %12s %12s | %s" % ("class", "realised n", "realised %", "pred % bef", "pred % aft", "|err| bef pp", "|err| aft pp", ""))
        for k, st in enumerate(STATES):
            real = 100.0 * r["true_n"][k] / r["n"]
            pb, pa = 100.0 * r["before"].pred[k] / r["n"], 100.0 * r["after"].pred[k] / r["n"]
            emit("%-9s%13s%11.5f%% | %11.5f%% %11.5f%% | %12.5f %12.5f | %s" % (
                st, f(r["true_n"][k]), real, pb, pa, abs(pb - real), abs(pa - real),
                ("THIN (%d events)" % r["true_n"][k]) if r["true_n"][k] < 100 else ""))
    emit()
    emit("C. WHAT CALIBRATION DID, per origin state, in the two terms of the pre-registration")
    emit("   advantage = prior Brier - ML Brier (positive: the ML arm beats the origin-conditional prior; negative: it is worse than the prior)")
    rule()
    emit("%-8s | %-34s | %-44s | %s" % ("origin", "ML absolute error (Brier)", "ML-minus-baseline gap (Brier)", "advantage over the prior"))
    for o, pr, bf, af in rows:
        gb, ga, toward = verdicts[o]
        adv = lambda g: ("ML beats the prior" if g < 0 else "ML worse than the prior")
        emit("%-8s | %-34s | %-44s | %s" % (o, "%.6f -> %.6f (%s)" % (bf.b, af.b, "lower" if af.b < bf.b else "higher"),
                                          "%+.6f -> %+.6f" % (gb, ga),
                                          "%+.6f -> %+.6f   (%s -> %s)" % (-gb, -ga, adv(gb), adv(ga))))
    emit("Both columns move together here only because the prior is fixed; each is reported, neither is chosen. 'Toward the prior' in table A means the ML")
    emit("Brier ended closer to the prior's Brier than it started, in either direction.")


def shipped_vs_fold(oof, full, meta):
    """Per origin and fold: Brier of the fold model (OOF) and of the shipped model on the same rows, and how far apart their predictions are."""
    title("5. THE SHIPPED MODEL vs THE FOLD MODELS, row for row   (same rows, same features, same recipe; the shipped model saw 100% of the loans)")
    emit("If the fold models and the shipped model were interchangeable draws from one recipe, these columns would agree. Brier is per fold's rows.")
    emit()
    fs, y_all, fold = meta["from_state"], meta["y"].astype(np.int64), meta["fold"]
    emit("%-8s %-5s%12s | %10s %10s | %12s %12s | %s" % ("origin", "fold", "rows", "Brier OOF", "Brier ship", "floor OOF", "floor ship", ""))
    rule()
    corr_lines = []
    for o in TRANSIENT:
        idx = np.flatnonzero(fs == C.IDX[o])
        n = np.zeros(C.N_FOLDS)
        bo, bs, fo, fsh = (np.zeros(C.N_FOLDS) for _ in range(4))
        sx = sy = sxx = syy = sxy = 0.0
        absd = 0.0
        for ii in chunks(idx):
            P, F, y, ff = np.asarray(oof[ii]), np.asarray(full[ii]), y_all[ii], fold[ii]
            r = np.arange(len(y))
            b_o = (P ** 2).sum(1) - 2 * P[r, y] + 1
            b_s = (F ** 2).sum(1) - 2 * F[r, y] + 1
            for k in range(C.N_FOLDS):
                m = ff == k
                n[k] += m.sum(); bo[k] += b_o[m].sum(); bs[k] += b_s[m].sum()
                fo[k] += (P[r, y][m] < C.FLOOR).sum(); fsh[k] += (F[r, y][m] < C.FLOOR).sum()
            a, b = P[:, C.IDX[o]], F[:, C.IDX[o]]
            sx += a.sum(); sy += b.sum(); sxx += (a * a).sum(); syy += (b * b).sum(); sxy += (a * b).sum()
            absd += np.abs(P - F).sum()
        for k in range(C.N_FOLDS):
            emit("%-8s %-5d%12s | %10.6f %10.6f | %12s %12s |" % (o, k, f(n[k]), bo[k] / n[k], bs[k] / n[k], f(fo[k]), f(fsh[k])))
        N = n.sum()
        corr = (sxy / N - sx / N * sy / N) / np.sqrt((sxx / N - (sx / N) ** 2) * (syy / N - (sy / N) ** 2))
        corr_lines.append("%-8s correlation of P(stay) shipped vs fold model: %.4f;  mean |shipped - fold| per cell: %.5f  (%s rows)" % (
            o, corr, absd / (N * K), f(N)))
    rule()
    for line in corr_lines:
        emit(line)
    emit("The cause is not established here. What it means for this milestone: the correction is fitted on the fold models' behaviour, so where the shipped")
    emit("model behaves differently from them (DQ60, DQ90+) the correction does not describe the shipped model, which is visible in section 3B.")


def run_report():
    out = C.cal_dir(MODEL)
    header = open(os.path.join(out, "header.txt"), encoding="utf-8").read()
    sha = C.sha256_file(os.path.join(MODEL, "model.txt"))
    if sha not in header:
        raise SystemExit("the model on disk no longer matches the header's sha256")
    chk = json.load(open(os.path.join(out, "oof", "fold_check.json"), encoding="utf-8"))
    corr = C.load_correction(out)
    naive = C.load_naive(out)
    if corr["model_sha256"] != sha:
        raise SystemExit("the saved correction was fitted for a different model file")
    oof, meta = C.load_stack(MODEL, "oof_proba")
    full, _ = C.load_stack(MODEL, "full_proba")
    n = len(meta["y"])

    # ---- fold check (before any calibration number)
    LINES.clear()
    title("FOLD ASSIGNMENT CHECK")
    emit("seed %d, %d folds, method: %s" % (chk["seed"], C.N_FOLDS, chk["method"]))
    emit()
    emit("%-6s%16s%12s%16s%14s%16s%16s%16s" % ("fold", "held-out rows", "share", "held-out loans", "share", "training rows", "train secs", "predict secs"))
    rule()
    for k in range(C.N_FOLDS):
        fi = chk["folds"][k]
        emit("%-6d%16s%11.3f%%%16s%13.3f%%%16s%16s%16s" % (k, f(chk["rows_by_fold"][k]), 100.0 * chk["rows_by_fold"][k] / chk["n_rows"],
                                                          f(chk["loans_by_fold"][k]), 100.0 * chk["loans_by_fold"][k] / chk["n_loans"],
                                                          f(fi["train_rows"]), fi["train_seconds"], fi["predict_seconds"]))
    emit("%-6s%16s%12s%16s" % ("TOTAL", f(sum(chk["rows_by_fold"])), "", f(sum(chk["loans_by_fold"]))))
    rule()
    emit("fit rows                                            : %s" % f(chk["n_rows"]))
    emit("loans with at least one fit row                     : %s   (the panel as a whole has 2,415,202 loans across all roles)" % f(chk["n_loans"]))
    emit("rows with exactly one OOF prediction                : %s" % f(chk["rows_with_exactly_one_oof_prediction"]))
    emit("rows with none / rows with more than one            : %s / %s" % (f(chk["rows_with_none"]), f(chk["rows_with_more_than_one"])))
    emit("(loan, fold) pairs in which a loan has a second fold: %s   (a loan in two folds would be counted here)" % f(chk["loan_pairs_with_a_second_fold"]))
    emit("OOF cells non-finite or negative                    : %s;  worst |row sum - 1| = %.2e" % (f(chk["oof_nonfinite_or_negative_cells"]), chk["oof_worst_row_sum_error"]))
    emit("population = the shipped model's: row count, per-class label counts and per-origin counts equal the sidecar's (asserted in src.calibrate)")
    counts = np.bincount(meta["from_state"].astype(np.int64) * K + meta["y"].astype(np.int64), minlength=4 * K).reshape(4, K)
    base = np.array([[int(v) for v in line.split(",")[1:]] for line in open(BASELINE_COUNTS).read().splitlines()[1:5]])
    emit("transition counts (origin x destination) equal artifacts/baseline_headline/counts.csv cell for cell: %s" %
         ("YES" if (counts == base).all() else "NO - differences: %s" % np.argwhere(counts != base).tolist()))
    emit("run resources: total %.1f min, peak RSS %.2f GB, minimum system available memory %.2f GB (abort floor 0.49 GB); no fold exceeded %d min" % (
        chk["total_seconds"] / 60, chk["peak_rss_bytes"] / C.GB, chk["min_system_available_bytes"] / C.GB, C.FOLD_TRAIN_LIMIT_S // 60))
    fold_text = "\n".join(LINES)

    # ---- the fitted corrections
    LINES.clear()
    emit()
    emit("#" * W)
    emit("CALIBRATION NUMBERS BEGIN BELOW")
    emit("#" * W)
    title("1. THE FITTED CORRECTIONS   (fitted on each origin state's out-of-fold rows)")
    emit("%-8s%14s%14s | leaver counts by destination (THIN = under 100 events)" % ("origin", "rows", "leavers"))
    rule()
    for o in TRANSIENT:
        co = corr["origins"][o]
        emit("%-8s%14s%14s | %s" % (o, f(co["n_rows"]), f(co["n_leavers"]), "   ".join(
            "%s %s%s" % (d, f(c), " THIN" if c < 100 else "") for d, c in co["dest_counts"].items())))
    emit("Rates beside counts: stay rate = 1 - leavers/rows; " + "; ".join(
        "%s %.4f%% (%s of %s)" % (o, 100.0 * (1 - corr["origins"][o]["n_leavers"] / corr["origins"][o]["n_rows"]),
                                   f(corr["origins"][o]["n_rows"] - corr["origins"][o]["n_leavers"]), f(corr["origins"][o]["n_rows"])) for o in TRANSIENT))
    emit()
    emit("STEP ONE - isotonic (stay vs leave): number of fitted thresholds")
    emit("   " + "   ".join("%s %s" % (o, f(corr["origins"][o]["isotonic_thresholds"])) for o in TRANSIENT))
    emit()
    emit("STEP TWO - Platt sigmoid, expit(a * logit(q) + b), one-vs-rest on the leavers (a = 1, b = 0 is the identity); log-likelihood in nats")
    rule()
    emit("%-8s%-9s%12s%11s%11s%16s%16s%7s%6s  %s" % ("origin", "dest", "events", "a", "b", "nll identity", "nll fitted", "iters", "conv", ""))
    for o in TRANSIENT:
        for d, p in corr["origins"][o]["platt"].items():
            clip = corr["origins"][o]["logit_clipped_counts"][d]
            emit("%-8s%-9s%12s%11.4f%11.4f%16.2f%16.2f%7d%6s  %s%s" % (
                o, d, f(p["positives"]), p["a"], p["b"], p["nll_identity"], p["nll_fit"], p["iterations"],
                "yes" if p["converged"] else "NO", "THIN " if p["positives"] < 100 else "",
                ("q clipped by logit: %s low, %s high" % (f(clip["below_eps"]), f(clip["above_1_minus_eps"]))) if clip["below_eps"] or clip["above_1_minus_eps"] else ""))
    step_text = "\n".join(LINES)

    # ---- measurements
    print("measuring the OOF stack ...", flush=True)
    res_oof = measure_stack(oof, meta, corr, naive)
    print("measuring the shipped model's predictions ...", flush=True)
    res_full = measure_stack(full, meta, corr)

    LINES.clear()
    title("2. DECOMPOSED vs NAIVE SCHEME   (on the OOF stack; the rejected scheme is built, not recommended; no verdict is given)")
    emit("Per origin state, over every OOF row of that state: A = the decomposed calibrated matrix, N = the naive one (six classes calibrated one-vs-rest,")
    emit("renormalised). Per row, d = max over the six cells of |A - N|. Both naive variants are shown (the brief does not name the per-class method).")
    emit("Rows excluded = naive rows whose six calibrated values were all zero, which cannot be renormalised.")
    for kind in ("isotonic", "platt"):
        emit()
        emit("NAIVE, %s per class" % kind.upper())
        rule()
        emit("%-8s%13s%9s | %-10s %-10s %-10s %-10s %-10s %-10s | %s" % ("origin", "rows", "excl.", "max d", "p50", "p90", "p99", "p99.9", "p99.99", "max |A-N| per cell, by class"))
        extra = []
        for o in TRANSIENT:
            r = res_oof[o][kind]
            d = np.concatenate(r["maxdiff"])
            qs = np.quantile(d, [0.5, 0.9, 0.99, 0.999, 0.9999])
            emit("%-8s%13s%9s | %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e | %s" % (
                o, f(len(d)), f(r["dropped"]), float(d.max()), *qs, "  ".join("%s %.2e" % (STATES[k], r["colmax"][k]) for k in range(K))))
            extra.append("%-8s rows with d > 1e-4: %s (%.4f%%)   > 1e-3: %s (%.4f%%)   > 1e-2: %s (%.4f%%)   > 1e-1: %s (%.4f%%)" % (
                o, *sum(([f((d > t).sum()), 100.0 * (d > t).mean()] for t in (1e-4, 1e-3, 1e-2, 1e-1)), [])))
            del d
        emit()
        for e in extra:
            emit(e)
    emit()
    emit("Informational (OOF stack): Brier and log-loss of the three calibrated matrices beside the raw model. No scheme is selected here.")
    rule()
    emit("%-8s%11s%11s%11s%11s | %11s%11s%11s%11s | %s" % ("origin", "raw", "decomposed", "naive iso", "naive Platt", "raw", "decomposed", "naive iso", "naive Platt", "floor hits (raw/dec/iso/Platt)"))
    emit("%-8s%44s | %44s" % ("", "Brier", "log-loss"))
    for o in TRANSIENT:
        r = res_oof[o]
        a = [r["before"], r["after"], r["isotonic"]["acc"], r["platt"]["acc"]]
        emit("%-8s%11.6f%11.6f%11.6f%11.6f | %11.6f%11.6f%11.6f%11.6f | %s" % (
            o, *[x.b for x in a], *[x.l for x in a], " / ".join(f(x.floor_hits) for x in a)))
    comp_text = "\n".join(LINES)

    LINES.clear()
    render_measure(res_full, "3. APPLY AND MEASURE - the SHIPPED model (trained on all fit rows), calibrated by the decomposed correction, on FIT rows",
                   ["The correction was fitted on out-of-fold predictions and is applied here to the shipped model's IN-SAMPLE predictions on the same fit rows.",
                    "In-sample predictions are more confident than out-of-fold ones, so this measurement mixes drift correction with that difference (see section 4).",
                    "One step ahead; no evaluate row; no chained forecast."])
    shipped_text = "\n".join(LINES)

    LINES.clear()
    render_measure(res_oof, "4. SUPPLEMENT - the same measurements on the OUT-OF-FOLD stack (each row predicted by a model that never saw its loan)",
                   ["Same correction, same code. The correction was fitted on these rows, but it is a monotone map with a handful of parameters per class plus a",
                    "staircase over tens of millions of rows, so this is close to a clean read of what it does to out-of-fold predictions."])
    oof_text = "\n".join(LINES)

    LINES.clear()
    shipped_vs_fold(oof, full, meta)
    svf_text = "\n".join(LINES)

    LINES.clear()
    title("6. NOTES")
    emit("* Everything above is one step ahead on fit rows. Whether calibration helps or hurts the 60-month chained forecast is not measured here.")
    emit("* The prior is the origin-conditional class frequency of the fit rows; it is also the baseline's transition-matrix row (counts verified above),")
    emit("  and it is scored in-sample on the same rows, as is the ML arm in section 3.")
    emit("* Fold models are trained on 80% of the loans; the shipped model on 100%. At Current and DQ30 the two behave alike (section 5); at DQ60 and DQ90+ they do not, and the shipped model scores worse than every fold model on the same rows. The cause is not investigated here.")
    emit("* Platt slopes far below 1 (e.g. DQ90+ destination Current a = 0.0024, DQ30 a = 0.0001; Default from Current a = 0.0010, intercept = the base rate) mean")
    emit("  the fold models' destination split among leavers carries little or no usable ranking there, and the fitted sigmoid reduces it to nearly the")
    emit("  origin's observed destination mix. The log-likelihood gain in section 1 (e.g. DQ90+ -> Current 294,840 -> 18,590 nats) is the size of that.")
    emit("  'q clipped by logit' in section 1 counts leavers whose raw destination mass was outside [1e-12, 1-1e-12]; those values were clipped, not dropped.")
    emit("* Default is THIN (under 100 events) at Current (74), DQ30 (50) and DQ60 (67); its sigmoid has two parameters fitted on that few events.")
    emit("* Files: artifacts/calibration/%s/{correction.json, iso_<state>.npz, naive_platt.json, naive_isotonic.npz, oof/}; apply with" % os.path.basename(MODEL))
    emit("  src.calibrate.load_correction(dir) and src.calibrate.apply_correction(proba, origin, corr).")
    notes_text = "\n".join(LINES)

    body = "\n".join([fold_text, step_text, comp_text, shipped_text, oof_text, svf_text, notes_text])
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(header.rstrip("\n") + "\n\n" + body + "\n")
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
