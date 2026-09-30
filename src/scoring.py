"""Model-agnostic scoring of predicted state distributions.

Input, per horizon: a (n_loans, n_states) array of predicted distributions, an
int array of realised state codes (CENSORED = -1 where unobservable), and a
boolean cohort mask. Nothing here knows how the predictions were made - no
transition matrix, no baseline, no panel. State names are passed in only to
label tables.

METRICS
    aggregate     predicted expected count per state (sum of probabilities over
                  the cohort) beside the realised count, and both as shares
    brier         multiclass Brier score: mean over loans of
                  sum_k (p_k - 1[realised == k])^2; 0 is perfect, 2 the maximum
    calibration   per state k: loans binned on p_k; per bin the loan count,
                  mean p_k, and the count and share realised in state k
"""

import numpy as np

CENSORED = -1
TOL = 1e-9
CAL_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


class ScoringError(ValueError):
    pass


def _check(pred, realised, member):
    pred = np.asarray(pred, dtype=np.float64)
    realised = np.asarray(realised, dtype=np.int64)
    member = np.asarray(member, dtype=bool)
    if pred.ndim != 2 or len(pred) != len(realised) or len(member) != len(pred):
        raise ScoringError("shape mismatch: pred %s, realised %s, member %s"
                           % (pred.shape, realised.shape, member.shape))
    p, r = pred[member], realised[member]
    if np.any(r == CENSORED):
        raise ScoringError("%d cohort members are censored"
                           % int((r == CENSORED).sum()))
    if np.any((r < 0) | (r >= pred.shape[1])):
        raise ScoringError("realised state code out of range")
    if np.any(p < -TOL):
        raise ScoringError("negative predicted probability")
    sums = p.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > TOL):
        raise ScoringError("%d predicted rows do not sum to 1 within %g"
                           % (int((np.abs(sums - 1.0) > TOL).sum()), TOL))
    return p, r


def aggregate(p, r):
    k = p.shape[1]
    return {"n": len(r), "expected": p.sum(axis=0),
            "realised": np.bincount(r, minlength=k)}


def brier(p, r):
    if len(r) == 0:
        return float("nan")
    y = np.zeros_like(p)
    y[np.arange(len(r)), r] = 1.0
    return float(((p - y) ** 2).sum(axis=1).mean())


def calibration(p, r, state, edges=CAL_EDGES):
    """Rows (lo, hi, n, mean_p, hits) for every non-empty bin of p[:, state]."""
    edges = np.asarray(edges, dtype=np.float64)
    q = p[:, state]
    b = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    hit = r == state
    out = []
    for i in range(len(edges) - 1):
        m = b == i
        n = int(m.sum())
        if n:
            out.append((edges[i], edges[i + 1], n, float(q[m].mean()),
                        int(hit[m].sum())))
    return out


def score(pred, realised, member, edges=CAL_EDGES):
    """Every metric for one horizon x cohort."""
    p, r = _check(pred, realised, member)
    agg = aggregate(p, r)
    return {"n": agg["n"], "expected": agg["expected"],
            "realised": agg["realised"], "brier": brier(p, r),
            "calibration": [calibration(p, r, k, edges)
                            for k in range(p.shape[1])]}


# ==========================================================================
# plain-text rendering
# ==========================================================================
def _fmt(n):
    return "{:,}".format(int(n))


def _pct(num, den, dp=4):
    return "n/a" if not den else ("%." + str(dp) + "f%%") % (100.0 * num / den)


def render(emit, rule, result, states, title):
    """Aggregate table, Brier score and calibration tables for one result."""
    n = result["n"]
    emit(title)
    rule()
    emit("%-10s%18s%14s%16s%14s%14s" % ("state", "pred (exp. n)", "pred share",
                                        "realised n", "real share",
                                        "gap (pp)"))
    rule()
    for k, s in enumerate(states):
        e, a = result["expected"][k], result["realised"][k]
        gap = 100.0 * (e - a) / n if n else float("nan")
        emit("%-10s%18s%14s%16s%14s%+14.4f" % (s, "{:,.1f}".format(e),
                                               _pct(e, n), _fmt(a),
                                               _pct(a, n), gap))
    rule()
    emit("%-10s%18s%14s%16s%14s" % ("TOTAL", "{:,.1f}".format(
        result["expected"].sum()), _pct(result["expected"].sum(), n), _fmt(n),
        _pct(n, n)))
    emit("multiclass Brier score: %.6f   (n = %s loans)" % (result["brier"],
                                                          _fmt(n)))
    emit()
    emit("Calibration - loans binned on the predicted probability of each state;")
    emit("non-empty bins only.")
    rule()
    emit("%-10s%20s%14s%16s%16s%14s" % ("state", "bin", "loans",
                                        "mean predicted", "realised n",
                                        "real share"))
    rule()
    for k, s in enumerate(states):
        for lo, hi, m, mp, hits in result["calibration"][k]:
            emit("%-10s%20s%14s%15.4f%%%16s%14s"
                 % (s, "[%.3f, %.3f%s" % (lo, hi, "]" if hi >= 1.0 else ")"),
                    _fmt(m), 100.0 * mp, _fmt(hits), _pct(hits, m)))
    rule()
    emit()
