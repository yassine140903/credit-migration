"""HEADLINE RUN: lifetime forecast to h = 60, ML (pooled) vs the homogeneous baseline, with prior-only and oracle references.

    python diagnostics/54_headline_h60.py A     # train, baseline, forecast. Reads fit rows (train_ml, baseline --fit-only) and condition
                                                # rows at the cut (forecast). NO evaluate row is read in this phase.
    python diagnostics/54_headline_h60.py B     # the harness: realised outcomes (the first time `evaluate` is opened), every arm scored,
                                                # the report diagnostics/54_headline_h60.txt
    DIAG54_DRYRUN=1 python diagnostics/54_headline_h60.py B
                                                # code-path test on the existing arch6 fit-role outputs (cut 200806, h = 6, fit loans);
                                                # opens no evaluate row, writes artifacts/headline/dryrun_54.txt, not the report

FIXED BEFORE ANY NUMBER WAS READ (nothing below is chosen after seeing a result)
  Arms      ML pooled (fresh model, artifacts/ml/headline_pooled, full fit window); BASELINE (src.baseline --fit-only, same fit rows);
            PRIOR-ONLY (to_state frequencies over all fit rows, every loan, every h >= 1: src.nulls.marginal_distribution);
            references: PERSISTENCE (identity matrix through src.horizon.chain), ORACLE (not achievable; src.nulls.oracle_distribution
            per h on the scored cohort, so its error is zero by construction). src.nulls holds persistence / marginal / oracle; they
            are not in src.horizon and only persistence is chained.
  Loans     the loans with a `condition` row at 2008-12 (405,672), the forecaster's population. The harness's eligible set is 33 loans
            larger (loans observed at the cut only as the to_state of a 2008-11 row); those 33 are not scored and are reported
            separately. Decision taken with the user before the run.
  Cohort    at each h the scored cohort is the loans whose realised state at h is not CENSORED (harness `varying` convention);
            censored loans are their own column, never redistributed. Predicted shares are summed over the scored cohort only.
  Error     |predicted share - realised share| in percentage points, per state; TOTAL = sum over the six states.
  Closest   per state, the arm with the smallest error among ML / baseline / prior-only (and, second column, among those plus persistence).
  Gap       ML minus baseline, ML minus prior-only, in TOTAL percentage points, at h = 6, 12, 24, 60 and at every h.
  Trend     'widens' if |gap| strictly increases at each successive reporting horizon, 'narrows' if it strictly decreases at each,
            otherwise 'non-monotone'. No threshold defines 'holds'; the ratio |gap(60)| / |gap(6)| and the signed values are printed.
  h = 0     the common starting state of every arm (a one-hot on the state at the cut); no arm forecasts there.
  No winner-takes-all rule. No sampling uncertainty is attached (none was specified).
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import baseline, horizon as hz, nulls, panel, panel_io          # noqa: E402
from src.memwatch import MemWatch                                         # noqa: E402

DRY = os.environ.get("DIAG54_DRYRUN") == "1"
STATES = list(panel.STATES)
TRANSIENT = STATES[:4]
K = len(STATES)
GB = 2 ** 30
WORK = os.path.join("artifacts", "headline")
LOGS = os.path.join(WORK, "logs")
PHASE_A_JSON = os.path.join(WORK, "phase_a.json")
REG = ["--min-sum-hessian-in-leaf", "10", "--lambda-l2", "10"]
STORED_TOL = 1e-5
RULE_W = 138
REPORT_AT = (6, 12, 24, 60)

if DRY:
    CUT, H, CUT_ROLE, MODEL_NAME, RUN, BASE_DIR = 200806, 6, "fit", "arch6_pooled", "arch6_pooled", os.path.join("artifacts", "baseline_to200806")
    OUT = os.path.join(WORK, "dryrun_54.txt")
    EXPECT = 1681267
else:
    CUT, H, CUT_ROLE, MODEL_NAME, RUN, BASE_DIR = 200812, 60, "condition", "headline_pooled", "headline_pooled", os.path.join("artifacts", "baseline_headline")
    OUT = os.path.join("diagnostics", "54_headline_h60.txt")
    EXPECT = 405672
CUT_IDX = panel.mi(CUT // 100, CUT % 100)
MODEL_DIR = os.path.join("artifacts", "ml", MODEL_NAME)
FC_DIR = os.path.join("data", "forecast", RUN)
FC_REPORT = os.path.join("diagnostics", "54_headline_forecaster_build.txt")
HS = tuple(range(1, H + 1))
RH = tuple(h for h in REPORT_AT if h <= H)
LINES = []
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
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


def f(n):
    return format(int(n), ",")


def pct4(x):
    return "%.4f%%" % x


# ==========================================================================
# phase A: the three real src tools, each in its own subprocess, with wall-clock and sampled peak RSS
# ==========================================================================
def run_child(args, log_name):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, log_name)
    t0 = time.perf_counter()
    peak, min_avail = 0, psutil.virtual_memory().available
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\n\n")
        fh.flush()
        p = subprocess.Popen([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, text=True)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
            min_avail = min(min_avail, psutil.virtual_memory().available)
            time.sleep(0.5)
    return {"seconds": time.perf_counter() - t0, "peak_rss_bytes": int(peak), "min_avail_bytes": int(min_avail),
            "returncode": p.returncode, "log": path.replace(os.sep, "/"), "args": args}


def phase_a():
    os.makedirs(WORK, exist_ok=True)
    rec = {}
    if os.path.exists(PHASE_A_JSON):
        with open(PHASE_A_JSON, encoding="utf-8") as fh:
            rec = json.load(fh)
    steps = [("train", ["-m", "src.train_ml", "--name", MODEL_NAME] + REG),
             ("baseline", ["-m", "src.baseline", "--fit-only", "--out-dir", BASE_DIR]),
             ("forecast", ["-m", "src.forecast", "--run", RUN, "--model", MODEL_DIR.replace(os.sep, "/"), "--expect-loans", str(EXPECT),
                           "--report", FC_REPORT])]
    for name, args in steps:
        if name in rec and rec[name]["returncode"] == 0:
            print("phase A: %s already done (%.0f s), skipped" % (name, rec[name]["seconds"]), flush=True)
            continue
        print("phase A: %s ..." % name, flush=True)
        r = run_child(args, "%s.log" % name)
        rec[name] = r
        with open(PHASE_A_JSON, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print("phase A: %s finished, exit %d, %.0f s, peak RSS %.2f GB" % (name, r["returncode"], r["seconds"], r["peak_rss_bytes"] / GB), flush=True)
        if r["returncode"] != 0:
            print("STOP: %s failed; see %s. Nothing was adjusted or re-run." % (name, r["log"]), flush=True)
            sys.exit(1)


# ==========================================================================
# phase B helpers
# ==========================================================================
def load_cut_ids(quarters):
    ids, fs = [], []
    for q in quarters:
        d = panel_io.load_panel(roles=[CUT_ROLE], columns=["loan_id", "from_period", "from_state"], quarters=[q])
        d = d[d["from_period"] == CUT]
        ids.append(d["loan_id"].astype(str).to_numpy())
        fs.append(d["from_state"].astype(str).to_numpy())
        del d
    ids, fs = np.concatenate(ids), np.concatenate(fs)
    o = np.argsort(ids, kind="stable")
    ids, fs = ids[o], fs[o]
    assert len(np.unique(ids)) == len(ids), "a loan has two %s rows at the cut" % CUT_ROLE
    bad = sorted(set(fs) - set(TRANSIENT))
    assert not bad, bad
    return ids, np.array([STATES.index(s) for s in fs], dtype=np.int64)


def newacc():
    return {"dev": 0.0, "minp": np.inf, "minstep": np.inf, "minstep_pd": np.inf, "n": 0}


def accumulate(acc, P64, P_raw_min):
    acc["dev"] = max(acc["dev"], float(np.abs(P64.sum(axis=2) - 1.0).max()))
    acc["minp"] = min(acc["minp"], float(P_raw_min))
    acc["minstep"] = min(acc["minstep"], float(np.diff(P64[:, :, 4:], axis=1).min()))
    acc["minstep_pd"] = min(acc["minstep_pd"], float(np.diff(P64[:, :, 4] + P64[:, :, 5], axis=1).min()))
    acc["n"] += P64.shape[0]


def read_ml(run_dir, M):
    """Stream the stored forecast: masked and unmasked state sums per h, the checks as stored, the loan ids in storage order."""
    S, U = np.zeros((H + 1, K)), np.zeros((H + 1, K))
    acc, mk_sums = newacc(), np.zeros((H + 1, K))
    dev64, ids, pos = 0.0, [], 0
    for bdir in sorted(glob.glob(os.path.join(run_dir, "batch=*"))):
        with open(os.path.join(bdir, "_DONE.json"), encoding="utf-8") as fh:
            mk = json.load(fh)
        dev64 = max(dev64, mk["checks"]["worst_sum_deviation"])
        mk_sums += np.array(mk["state_sums"])
        tbl = pq.read_table(os.path.join(bdir, "part.parquet"))
        cols = {}
        for name in ("loan_id", "state"):
            c = tbl.column(name).combine_chunks()
            if not pa.types.is_dictionary(c.type):
                c = pc.dictionary_encode(c)
                c = c.combine_chunks() if hasattr(c, "combine_chunks") else c
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
        P64 = P.astype(np.float64)
        Mb = M[pos:pos + n].astype(np.float64)
        S += np.einsum("nh,nhs->hs", Mb, P64)
        U += P64.sum(axis=0)
        accumulate(acc, P64, P.min())
        ids.append(np.array(cols["loan_id"][0], dtype=object))
        pos += n
        del tbl, P, P64, seen, prob, h, cols, Mb
    return np.concatenate(ids).astype(str), S, U, acc, dev64, mk_sums


def chain_arm(Pmat, start, M, chunk=100_000):
    """A homogeneous chain over the real loans through src.horizon.chain; masked and unmasked sums per h and the checks."""
    S, U, acc = np.zeros((H + 1, K)), np.zeros((H + 1, K)), newacc()
    step = hz.homogeneous_step(Pmat)
    for a in range(0, len(start), chunk):
        b = min(a + chunk, len(start))
        out = hz.chain(start[a:b], step, horizons=HS)
        full = np.empty((b - a, H + 1, K))
        full[:, 0, :] = hz.one_hot(start[a:b])
        for k in HS:
            full[:, k, :] = out[k]
        S += np.einsum("nh,nhs->hs", M[a:b].astype(np.float64), full)
        U += full.sum(axis=0)
        accumulate(acc, full, full.min())
        del out, full
    return S, U, acc


def trend(vals):
    if len(vals) < 2:
        return "n/a (fewer than two reporting horizons)"
    a = np.abs(vals)
    if all(y > x for x, y in zip(a, a[1:])):
        lab = "widens"
    elif all(y < x for x, y in zip(a, a[1:])):
        lab = "narrows"
    else:
        lab = "non-monotone"
    flips = any(np.sign(x) != np.sign(y) for x, y in zip(vals, vals[1:]))
    return lab + ("; the gap changes sign" if flips else "; the gap keeps one sign") + "; |gap(%d)|/|gap(%d)| = %.3f" % (
        RH[-1], RH[0], (a[-1] / a[0]) if a[0] else float("nan"))


def state_table(title_, rows, header, w=11):
    emit(title_)
    emit("  " + "%-6s" % header[0] + "".join("%*s" % (w, c) for c in header[1:]))
    for r in rows:
        emit("  " + "%-6s" % r[0] + "".join("%*s" % (w, c) for c in r[1:]))


# ==========================================================================
# phase B
# ==========================================================================
def phase_b():
    watch = MemWatch(on_abort=lambda m: (emit("MEMORY PRESSURE, run stopped: " + m), write_report()), abort_mb=700)
    watch.start()
    manifest = panel_io.read_manifest()
    quarters = [p["quarter"] for p in manifest["partitions"]]
    emit("HEADLINE RUN  lifetime forecast to h = %d from the %s cut: ML (pooled) vs homogeneous baseline, with prior-only, persistence and ORACLE references%s"
         % (H, "%d-%02d" % (CUT // 100, CUT % 100), "   [DRY RUN: code-path test on the arch6 fit-role outputs; no evaluate row read]" if DRY else ""))
    emit("run date 2026-10-13   panel build %s   partitions %s..%s" % (manifest["build_stamp"], quarters[0], quarters[-1]))
    title("0. DEFINITIONS FIXED BEFORE ANY NUMBER WAS READ")
    for ln in __doc__.split("FIXED BEFORE ANY NUMBER WAS READ (nothing below is chosen after seeing a result)")[1].strip("\n").split("\n"):
        emit(ln)
    emit()
    emit("Order of operations: phase A (train_ml on fit rows, baseline --fit-only, forecast from the condition rows at the cut) ran and finished before this phase;")
    emit("only here is the harness's read of `evaluate` rows performed, for realised outcomes alone.")
    pa_rec = {}
    if not DRY and os.path.exists(PHASE_A_JSON):
        with open(PHASE_A_JSON, encoding="utf-8") as fh:
            pa_rec = json.load(fh)
    write_report()

    # ---------------------------------------------------------------- 1 training data, model, baseline, prior-only
    title("1. THE ARMS' TRAINING DATA (all from fit rows; read before any outcome)")
    with open(os.path.join(MODEL_DIR, "model.json"), encoding="utf-8") as fh:
        side = json.load(fh)
    with open(os.path.join(MODEL_DIR, "model.txt"), "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    txt = raw.decode("utf-8", errors="replace")
    prm = {k: v for k, v in re.findall(r"\[(\w+): ([^\]\n]*)\]", txt[txt.rfind("parameters:"):])}
    emit("ML model %s   sha256 %s" % (MODEL_DIR.replace(os.sep, "/"), sha[:16]))
    emit("  origin states %s; features %s; categorical %s" % (side["origin_states"], side["features"], side["categorical_features"]))
    emit("  training window: role fit, from_period_max argument %s; observed from_period %s .. %s (train_ml has no lower bound and none was added)"
         % (side["training_window"]["from_period_max_arg"], side["training_window"]["from_period_min_observed"],
            side["training_window"]["from_period_max_observed"]))
    emit("  training rows %s; loan_age is a feature: %s; rounds %d; panel build of the training run %s"
         % (f(side["n_rows"]), side["loan_age_is_feature"], side["n_rounds"], side["panel_build_stamp"]))
    emit("  label counts: " + "  ".join("%s %s" % (s, f(side["label_counts"][s])) for s in STATES))
    emit("  from_state counts: " + "  ".join("%s %s" % (s, f(side["from_state_counts"][s])) for s in TRANSIENT))
    want = {"num_leaves": "31", "min_data_in_leaf": "20", "min_sum_hessian_in_leaf": "10", "lambda_l2": "10", "objective": "multiclass"}
    emit("  parameters as stored in model.txt vs the brief:")
    for k, v in want.items():
        got = prm.get(k, "(absent)")
        ok = got == v or (k in ("min_sum_hessian_in_leaf", "lambda_l2") and got not in ("(absent)",) and float(got) == float(v))
        emit("    %-26s brief %-12s stored %-12s %s" % (k, v, got, "PASS" if ok else "*** DIFFERS ***"))
    emit("  lgb_params in the sidecar: %s" % side["lgb_params"])
    emit("  train_ml wall-clock: Dataset construction %.1f s + training %.1f s; sidecar peak RSS %.2f GB" % (
        side["dataset_construct_seconds"], side["train_seconds"], side["peak_rss_bytes"] / GB))

    cnt, P_base = baseline.load_baseline(BASE_DIR)
    C = cnt.to_numpy()
    emit("")
    emit("BASELINE re-estimated by src.baseline --fit-only into %s" % BASE_DIR.replace(os.sep, "/"))
    with open(os.path.join(BASE_DIR, "baseline_report.txt"), encoding="utf-8") as fh:
        brep = fh.read()
    for key in ("build stamp", "fit from_period range", "fit rows with from_period after"):
        for ln in brep.split("\n"):
            if ln.strip().startswith(key):
                emit("  " + ln.strip())
    same_rows = (int(C.sum()) == side["n_rows"]
                 and all(int(C[STATES.index(s)].sum()) == side["from_state_counts"][s] for s in TRANSIENT)
                 and all(int(C[:, j].sum()) == side["label_counts"][STATES[j]] for j in range(K)))
    emit("  identical training data: baseline count matrix total %s vs ML training rows %s; row totals vs the ML from_state counts and column totals vs the ML label counts: %s"
         % (f(C.sum()), f(side["n_rows"]), "PASS" if same_rows else "*** DIFFERENT ***"))
    emit("  fresh count matrix (rows = from, columns = to):")
    emit("    %-9s" % "" + "".join("%14s" % s for s in STATES))
    for i, s in enumerate(TRANSIENT):
        emit("    %-9s" % s + "".join("%14s" % f(C[i, j]) for j in range(K)))
    emit("  fresh probability matrix:")
    for i, s in enumerate(TRANSIENT):
        emit("    %-9s" % s + "".join("%14.6f" % P_base.to_numpy()[i, j] for j in range(K)))
    c_st, p_st = baseline.load_baseline(os.path.join("artifacts", "baseline"))
    with open(os.path.join("artifacts", "baseline", "baseline_report.txt"), encoding="utf-8") as fh:
        st_stamp = [ln.strip() for ln in fh if ln.startswith("build stamp")][0]
    dP = np.abs(P_base.to_numpy() - p_st.to_numpy())
    dC = np.abs(C - c_st.to_numpy())
    emit("")
    emit("STORED artifacts/baseline vs the fresh estimate (the stored one carries '%s'; this run's panel build is %s)" % (st_stamp, manifest["build_stamp"]))
    emit("  stored count total %s vs fresh %s; max |count difference| in a cell %s" % (f(c_st.to_numpy().sum()), f(C.sum()), f(dC.max())))
    emit("  stored probability matrix:")
    for i, s in enumerate(TRANSIENT):
        emit("    %-9s" % s + "".join("%14.6f" % p_st.to_numpy()[i, j] for j in range(K)))
    emit("  (fresh - stored):")
    for i, s in enumerate(TRANSIENT):
        emit("    %-9s" % s + "".join("%+14.6f" % (P_base.to_numpy()[i, j] - p_st.to_numpy()[i, j]) for j in range(K)))
    emit("  MAX ABSOLUTE CELL DIFFERENCE (probabilities, all 36 cells): %.6f   %s" % (
        dP.max(), "stored and fresh agree to the last digit" if dC.max() == 0 else "they DIFFER; the fresh estimate is the one used"))

    fit_to, n_fit = nulls.fit_to_state_counts()
    marg = nulls.marginal_distribution(fit_to)
    emit("")
    emit("PRIOR-ONLY: to_state frequencies over all %s fit rows (src.nulls.fit_to_state_counts / marginal_distribution), the same vector for every loan at every h >= 1:" % f(n_fit))
    emit("  " + "  ".join("%s %s (%s)" % (s, f(fit_to[j]), pct4(100 * marg[j])) for j, s in enumerate(STATES)))
    same_lab = all(int(fit_to[j]) == side["label_counts"][STATES[j]] for j in range(K)) and n_fit == side["n_rows"]
    emit("  equals the ML model's training label counts and row count: %s" % ("PASS" if same_lab else "*** DIFFERENT ***"))
    emit("  this is the same vector as the harness's marginal null; it is carried once below as 'prior-only'.")
    write_report()

    # ---------------------------------------------------------------- 2 loans and outcomes
    title("2. LOANS AND REALISED OUTCOMES   (the harness read: realised states come from condition + evaluate rows of the test split)")
    ids, start = load_cut_ids(quarters)
    N = len(ids)
    emit("independent list: %s loans with a %s row at %d (read through panel_io.load_panel); expected %s: %s" % (f(N), CUT_ROLE, CUT, f(EXPECT), "PASS" if N == EXPECT else "*** DIFFERENT ***"))
    emit("  by start state: " + ", ".join("%s %s" % (s, f((start == i).sum())) for i, s in enumerate(TRANSIENT)))
    t0 = time.perf_counter()
    if DRY:
        L, _ = hz.read_test_loans(horizons=HS, cut=CUT_IDX, progress=lambda s: None, roles=["fit"], cut_role="fit", period_min=CUT - 1)
    else:
        L, _ = hz.read_test_loans(horizons=HS, cut=CUT_IDX, progress=lambda s: None)
    reason = hz.eligibility(L, CUT_IDX).to_numpy()
    emit("harness read %.0f s: %s test-split loans; " % (time.perf_counter() - t0, f(len(L))) + "; ".join(
        "%s %s" % (w, f((reason == k).sum())) for k, w in enumerate(hz.EXCLUDED)) + "; eligible %s" % f((reason < 0).sum()))
    pos = L.index.get_indexer(ids)
    assert (pos >= 0).all(), "a loan with a cut row is missing from the harness frame"
    assert (reason[pos] < 0).all(), "a loan with a cut row is not eligible in the harness"
    Le = L.iloc[pos]
    assert np.array_equal(Le["state_cut"].to_numpy(), start), "the harness's state at the cut disagrees with the cut row's from_state"
    emit("cross-check: the harness's state at the cut equals the cut row's from_state for all %s loans: PASS" % f(N))
    extra_mask = (reason < 0)
    extra_mask[pos] = False
    X = L[extra_mask]
    emit("harness-eligible loans WITHOUT a cut row (not scored, not forecast): %s" % f(len(X)))
    if len(X):
        rx = hz.realised(X, hz.HORIZONS if not DRY else (H,), CUT_IDX)
        emit("  start state: " + ", ".join("%s %s" % (s, f((X["state_cut"].to_numpy() == i).sum())) for i, s in enumerate(TRANSIENT)))
        for h_, r_ in rx.items():
            emit("  realised at h = %-2d: " % h_ + ", ".join("%s %s" % (s, f((r_ == j).sum())) for j, s in enumerate(STATES)) + ", CENSORED %s" % f((r_ == hz.CENSORED).sum()))
    real = hz.realised(Le, HS, CUT_IDX)
    R = np.empty((N, H + 1), dtype=np.int64)
    R[:, 0] = start
    for h in HS:
        R[:, h] = real[h]
    M = R >= 0
    nsc = M.sum(axis=0)
    cens = N - nsc
    Rc = np.zeros((H + 1, K))
    for h in range(H + 1):
        Rc[h] = np.bincount(R[M[:, h], h], minlength=K)
    assert np.array_equal(Rc.sum(axis=1), nsc)
    real_share = 100.0 * Rc / nsc[:, None]
    emit("")
    emit("SCORED AND CENSORED loans at the reporting horizons (censored: no observation at the target month and no absorbing state reached by then; their own column, never redistributed):")
    emit("  %4s %9s %12s %12s %12s   scored-cohort start-state mix (Current / DQ30 / DQ60 / DQ90+)" % ("h", "month", "eligible", "scored", "censored"))
    for h in (0,) + RH:
        mix = [int(((start == i) & M[:, h]).sum()) for i in range(4)]
        emit("  %4d %9s %12s %12s %12s   %s" % (h, hz.ym(CUT_IDX + h), f(N), f(nsc[h]), "%s (%.4f%%)" % (f(cens[h]), 100.0 * cens[h] / N),
                                              " / ".join("%s (%.2f%%)" % (f(m), 100.0 * m / nsc[h]) for m in mix)))
    emit("  all-eligible start-state mix: " + " / ".join("%s (%.2f%%)" % (f((start == i).sum()), 100.0 * (start == i).mean()) for i in range(4)))
    write_report()

    # ---------------------------------------------------------------- the arms
    title("3. THE ARMS OVER THE SAME LOANS IN THE SAME ORDER   (ML read back from storage; baseline, persistence chained by src.horizon.chain)")
    ids_store, S_ml, U_ml, acc_ml, dev64_ml, mk_sums = read_ml(FC_DIR, M)
    same_ids = len(ids_store) == N and bool((ids_store == ids).all())
    S_b, U_b, acc_b = chain_arm(P_base.to_numpy(), start, M)
    S_p, U_p, acc_p = chain_arm(nulls.persistence_matrix(), start, M)
    S_pr = np.outer(nsc, marg)
    S_pr[0] = Rc[0]                                                   # h = 0: the common starting state, not a forecast
    S_or = Rc.copy()
    for h in HS:
        d_, c_ = nulls.oracle_distribution(R[:, h], M[:, h])
        S_or[h] = d_ * nsc[h]
    orc_exact = bool(np.allclose(S_or, Rc, rtol=1e-9, atol=0))
    emit("the ML forecast's loan ids read back from storage (%s loans, %d batches) equal the independently loaded list, same order: %s" % (
        f(len(ids_store)), len(glob.glob(os.path.join(FC_DIR, "batch=*"))), "PASS" if same_ids else "*** FAIL ***"))
    emit("  the arrays used for the realised outcomes, the baseline chain, the persistence chain and the scored masks are all indexed by that same list (positions asserted above): PASS")
    emit("  ML run.json: %s" % json.dumps({k: v for k, v in json.load(open(os.path.join(FC_DIR, "run.json"), encoding="utf-8")).items() if k in ("n_loans", "cut", "horizon", "batch_size", "panel_build_stamp")}))
    emit("")
    emit("CHECKS per arm   (float64 = as computed in memory; float32 = as stored, summed in float64; the stored tolerance %.0e is not loosened)" % STORED_TOL)
    emit("  %-22s %-26s %-26s %-18s %-22s %-24s" % ("arm", "worst |sum-1| float64", "worst |sum-1| float32 stored", "smallest prob.", "Prepaid,Default step", "Prepaid+Default step"))
    emit("  %-22s %-26s %-26s %-18s %-22s %-24s" % ("ML (pooled)", "%.3e (forecaster)" % dev64_ml, "%.3e %s" % (acc_ml["dev"], "within" if acc_ml["dev"] <= STORED_TOL else "*** EXCEEDS ***"),
                                                    "%.3e" % acc_ml["minp"], "%.3e" % acc_ml["minstep"], "%.3e" % acc_ml["minstep_pd"]))
    for nm, a in (("baseline", acc_b), ("persistence", acc_p)):
        emit("  %-22s %-26s %-26s %-18s %-22s %-24s" % (nm, "%.3e" % a["dev"], "n/a (not stored)", "%.3e" % a["minp"], "%.3e" % a["minstep"], "%.3e" % a["minstep_pd"]))
    emit("  prior-only: one vector summing to %.17f (|sum-1| %.3e); smallest entry %.3e; constant in h, so Prepaid and Default mass cannot fall for h >= 1" % (marg.sum(), abs(marg.sum() - 1), marg.min()))
    emit("  ORACLE: reproduces the realised counts it was built from at every h (relative 1e-9): %s" % ("PASS" if orc_exact else "*** FAIL ***"))
    ok_ml = (dev64_ml <= 1e-9 and acc_ml["minp"] >= 0 and acc_ml["minstep"] >= 0 and acc_ml["minstep_pd"] >= 0)
    ok_b = all(a["dev"] <= 1e-12 * 10 and a["minp"] >= 0 and a["minstep"] >= 0 for a in (acc_b, acc_p))
    emit("  every vector sums to one (float64): ML %s, baseline/persistence %s;  no negative probability: ML %s, baseline/persistence %s;  Prepaid & Default mass non-decreasing in h: ML %s, baseline/persistence %s" % (
        "PASS" if dev64_ml <= 1e-9 else "FAIL", "PASS" if ok_b else "FAIL", "PASS" if acc_ml["minp"] >= 0 else "FAIL", "PASS" if acc_b["minp"] >= 0 and acc_p["minp"] >= 0 else "FAIL",
        "PASS" if acc_ml["minstep"] >= 0 and acc_ml["minstep_pd"] >= 0 else "FAIL", "PASS" if acc_b["minstep"] >= 0 and acc_p["minstep"] >= 0 else "FAIL"))
    emit("  stored float32 vectors sum to one within %.0e: %s (worst %.3e)" % (STORED_TOL, "PASS" if acc_ml["dev"] <= STORED_TOL else "*** FAIL ***", acc_ml["dev"]))
    emit("  ML sums of the stored float32 values over all loans vs the forecaster's in-memory float64 sums, largest |difference| in expected loans over h and state: %.3e" % np.abs(U_ml - mk_sums).max())
    write_report()

    # ---------------------------------------------------------------- errors
    arms = [("ML", S_ml), ("baseline", S_b), ("prior-only", S_pr), ("persistence", S_p), ("ORACLE", S_or)]
    share = {nm: 100.0 * S / nsc[:, None] for nm, S in arms}
    err = {nm: np.abs(share[nm] - real_share) for nm, _ in arms}
    tae = {nm: err[nm].sum(axis=1) for nm, _ in arms}

    title("4. PRIMARY: PREDICTED VS REALISED SHARE OF THE BOOK AT h = %s   (scored cohort; shares in %%, errors in percentage points)" % ", ".join(map(str, RH)))
    for h in RH:
        emit("")
        emit("h = %d  (%s)   eligible %s, scored %s, censored %s (%.4f%%)" % (h, hz.ym(CUT_IDX + h), f(N), f(nsc[h]), f(cens[h]), 100.0 * cens[h] / N))
        emit("  %-10s %11s | %11s %11s %11s %11s | %9s %9s %9s %9s | %-10s %-12s" % (
            "state", "realised", "ML", "baseline", "prior-only", "persistence", "err ML", "err base", "err prior", "err persist", "closest", "closest+pers."))
        for j, s in enumerate(STATES):
            e3 = [err["ML"][h, j], err["baseline"][h, j], err["prior-only"][h, j]]
            e4 = e3 + [err["persistence"][h, j]]
            emit("  %-10s %10.4f%% | %10.4f%% %10.4f%% %10.4f%% %10.4f%% | %9.4f %9.4f %9.4f %9.4f | %-10s %-12s" % (
                s, real_share[h, j], share["ML"][h, j], share["baseline"][h, j], share["prior-only"][h, j], share["persistence"][h, j],
                e3[0], e3[1], e3[2], e4[3], ["ML", "baseline", "prior-only"][int(np.argmin(e3))], ["ML", "baseline", "prior-only", "persistence"][int(np.argmin(e4))]))
        emit("  %-10s %11s | %11s %11s %11s %11s | %9.4f %9.4f %9.4f %9.4f   (TOTAL absolute error, sum over six states; ORACLE %.2e)" % (
            "TOTAL", "", "", "", "", "", tae["ML"][h], tae["baseline"][h], tae["prior-only"][h], tae["persistence"][h], tae["ORACLE"][h]))
        ctx = "  context (not scored): ML predicted share over ALL %s eligible loans: " % f(N) + " ".join("%s %.4f%%" % (s, 100.0 * U_ml[h, j] / N) for j, s in enumerate(STATES))
        emit(ctx)
        emit("                        baseline over all eligible loans:            " + " ".join("%s %.4f%%" % (s, 100.0 * U_b[h, j] / N) for j, s in enumerate(STATES)))
    write_report()

    title("5. THE COMPOUNDING FINDING: TOTAL ABSOLUTE ERROR (sum over the six states, percentage points) AND THE GAPS")
    emit("%4s %12s %12s %12s %12s %12s | %14s %14s | %9s %9s" % ("h", "ML", "baseline", "prior-only", "persistence", "ORACLE", "ML - baseline", "ML - prior", "scored", "censored"))
    for h in RH:
        emit("%4d %12.4f %12.4f %12.4f %12.4f %12.2e | %+14.4f %+14.4f | %9s %9s" % (
            h, tae["ML"][h], tae["baseline"][h], tae["prior-only"][h], tae["persistence"][h], tae["ORACLE"][h],
            tae["ML"][h] - tae["baseline"][h], tae["ML"][h] - tae["prior-only"][h], f(nsc[h]), f(cens[h])))
    g_b = np.array([tae["ML"][h] - tae["baseline"][h] for h in RH])
    g_p = np.array([tae["ML"][h] - tae["prior-only"][h] for h in RH])
    emit("")
    emit("ML minus baseline at h = %s: %s -> %s" % (", ".join(map(str, RH)), " / ".join("%+.4f" % g for g in g_b), trend(g_b)))
    emit("ML minus prior-only at h = %s: %s -> %s" % (", ".join(map(str, RH)), " / ".join("%+.4f" % g for g in g_p), trend(g_p)))
    emit("(negative = ML has the smaller total error. 'holds' has no threshold; read the signed values and the ratio.)")
    emit("")
    emit("TOTAL absolute error and gaps at every h (the compounding curve):")
    emit("%4s %12s %12s %12s %12s | %14s %14s | %9s %9s" % ("h", "ML", "baseline", "prior-only", "persistence", "ML - baseline", "ML - prior", "scored", "censored"))
    for h in range(H + 1):
        emit("%4d %12.4f %12.4f %12.4f %12.4f | %+14.4f %+14.4f | %9s %9s" % (
            h, tae["ML"][h], tae["baseline"][h], tae["prior-only"][h], tae["persistence"][h],
            tae["ML"][h] - tae["baseline"][h], tae["ML"][h] - tae["prior-only"][h], f(nsc[h]), f(cens[h])))
    write_report()

    title("6. FULL TRAJECTORIES h = 0..%d, six states, share of the scored cohort in %% (feeds the fan chart; the long-format CSV is %s)" % (H, os.path.join(WORK, "trajectory%s.csv" % ("_dryrun" if DRY else "")).replace(os.sep, "/")))
    emit("h = 0 is the common starting state of every arm. ORACLE equals the realised table by construction and is not printed separately. Prior-only is the single vector printed in section 1 for every h >= 1.")

    def traj(label, arr, extra=False):
        emit("")
        emit(label)
        emit("  %4s %9s %9s" % ("h", "scored", "censored") + "".join("%12s" % s for s in STATES) if extra else "  %4s" % "h" + "".join("%12s" % s for s in STATES))
        for h in range(H + 1):
            emit(("  %4d %9s %9s" % (h, f(nsc[h]), f(cens[h])) if extra else "  %4d" % h) + "".join("%11.4f%%" % arr[h, j] for j in range(K)))
    traj("REALISED (scored cohort at each h; every month is observable here up to the data's end, the censored count says how many loans drop out)", real_share, True)
    traj("ML (pooled)", share["ML"])
    traj("BASELINE (homogeneous six-by-six)", share["baseline"])
    traj("PERSISTENCE (reference)", share["persistence"])
    rows = []
    for nm, _ in arms:
        for h in range(H + 1):
            for j, s in enumerate(STATES):
                rows.append((h, nm, s, share[nm][h, j], real_share[h, j], err[nm][h, j], int(nsc[h]), int(cens[h])))
    os.makedirs(WORK, exist_ok=True)
    pd.DataFrame(rows, columns=["h", "arm", "state", "pred_share_pct", "realised_share_pct", "abs_error_pp", "n_scored", "n_censored"]).to_csv(
        os.path.join(WORK, "trajectory%s.csv" % ("_dryrun" if DRY else "")), index=False, float_format="%.10g")
    write_report()

    # ---------------------------------------------------------------- stored oracle numbers
    if not DRY:
        title("7. STORED ORACLE / NULL NUMBERS (artifacts/horizon_nulls) vs this run   (cohort 'varying'; Default and Prepaid)")
        with open(os.path.join("artifacts", "horizon_nulls", "nulls_report.txt"), encoding="utf-8") as fh:
            st_nstamp = [ln.strip() for ln in fh if ln.startswith("panel ")][0]
        emit("stored: %s" % st_nstamp)
        emit("this run's panel build: %s.  'harness-eligible' = all %s eligible loans (the stored runs' population definition); 'scored' = the %s loans of this run." % (
            manifest["build_stamp"], f((reason < 0).sum()), f(N)))
        E_all = L[reason < 0]
        st_all = E_all["state_cut"].to_numpy()
        real_all = hz.realised(E_all, hz.HORIZONS, CUT_IDX)
        sc = pd.read_csv(os.path.join("artifacts", "horizon_nulls", "compounding_all.csv"))
        sc = sc[sc["cohort"] == "varying"]
        _, P_stored = baseline.load_baseline(os.path.join("artifacts", "baseline"))
        ch_f = hz.chain(st_all, hz.homogeneous_step(P_base.to_numpy()), hz.HORIZONS)
        ch_s = hz.chain(st_all, hz.homogeneous_step(P_stored.to_numpy()), hz.HORIZONS)
        emit("  %3s %-24s %-9s %14s %16s %16s %16s %14s" % ("h", "quantity", "model", "stored", "fresh, eligible", "fresh, scored", "(elig - stored)", "(scored - stored)"))
        for h in hz.HORIZONS:
            m_all = real_all[h] != hz.CENSORED
            n_all = int(m_all.sum())
            hh = h if h <= H else None
            m_sc = M[:, h] if hh else None
            rows_ = []
            for model_key, label in (("persistence", "persistence"), ("marginal", "marginal"), ("ORACLE (not achievable)", "oracle"), ("baseline", "baseline")):
                srow = sc[(sc["horizon"] == h) & (sc["model"] == model_key)].iloc[0]
                for qty, j in (("default", STATES.index("Default")), ("prepaid", STATES.index("Prepaid"))):
                    real_all_n = int((real_all[h][m_all] == j).sum())
                    if label == "persistence":
                        fresh_all = 0.0
                    elif label == "marginal":
                        fresh_all = n_all * marg[j]
                    elif label == "oracle":
                        fresh_all = float(real_all_n)
                    else:
                        fresh_all = float(ch_f[h][m_all, j].sum())
                    fresh_sc = {"persistence": 0.0, "marginal": nsc[h] * marg[j], "oracle": Rc[h, j], "baseline": S_b[h, j]}[label] if hh else float("nan")
                    rows_.append((label, qty + " predicted n", float(srow[qty + "_pred_n"]), fresh_all, fresh_sc))
            srow = sc[(sc["horizon"] == h) & (sc["model"] == "baseline")].iloc[0]
            for qty, j in (("default", STATES.index("Default")), ("prepaid", STATES.index("Prepaid"))):
                rows_.append(("(realised)", qty + " realised n", float(srow[qty + "_real_n"]), float((real_all[h][m_all] == j).sum()), float(Rc[h, j]) if hh else float("nan")))
            rows_.append(("(cohort)", "n scored", float(srow["n"]), float(n_all), float(nsc[h]) if hh else float("nan")))
            for label, qty, s_, fa, fs in rows_:
                emit("  %3d %-24s %-9s %14s %16s %16s %+16.1f %+14.1f" % (h, qty, label, "{:,.1f}".format(s_), "{:,.1f}".format(fa), "{:,.1f}".format(fs), fa - s_, fs - s_))
            # stored baseline matrix chained on the fresh eligible cohort, for the reproducibility record
            emit("  %3d %-24s %-9s %14s %16s" % (h, "default predicted n", "baseline", "(stored matrix)", "{:,.1f}".format(float(ch_s[h][m_all, STATES.index("Default")].sum()))) + "   <- the stored matrix chained over today's eligible loans")
        emit("")
        emit("The stored runs were produced on the 2026-10-05 panel build and the panel was rebuilt on 2026-10-08, so n and the realised counts are expected to differ; the table records by how much.")
        emit("persistence predicts 0 Default and 0 Prepaid at every h by construction; the oracle's prediction equals its realised count by construction (the 'fresh, scored' oracle column is the scored cohort's realised count).")
        write_report()

    # ---------------------------------------------------------------- resources, findings
    title("8. RESOURCES")
    if pa_rec:
        for nm in ("train", "baseline", "forecast"):
            r = pa_rec[nm]
            emit("  %-9s wall-clock %8.1f s (%5.1f min)   peak RSS (sampled over the child process) %6.2f GB   min system available %6.2f GB   log %s" % (
                nm, r["seconds"], r["seconds"] / 60, r["peak_rss_bytes"] / GB, r["min_avail_bytes"] / GB, r["log"]))
        fr = open(FC_REPORT, encoding="utf-8").read()
        m_ = re.search(r"wall clock ([0-9.]+) s .*peak RSS ([0-9.]+) GB;\s+min system available ([0-9.]+) GB", fr)
        if m_:
            emit("  forecaster's own report (%s): wall clock %s s, peak RSS %s GB, min available %s GB" % (FC_REPORT.replace(os.sep, "/"), *m_.groups()))
        m2 = re.search(r"output: (\S+), ([0-9,]+) files, ([0-9.]+) MB .*rows written ([0-9,]+)", fr)
        if m2:
            emit("  forecast output %s: %s files, %s MB, %s rows (one per loan, h and state)" % m2.groups())
    emit("  phase B (this analysis, incl. the harness read): wall-clock %.1f s, peak RSS %.2f GB, min system available %.2f GB" % (time.perf_counter() - T0, watch.peak_rss() / GB, watch.min_avail / GB))
    watch.stop()
    emit("  artifacts: model %s (model.txt, model.json); baseline %s; forecast %s (long format loan_id, h, state, probability); trajectory CSV %s" % (
        MODEL_DIR.replace(os.sep, "/"), BASE_DIR.replace(os.sep, "/"), FC_DIR.replace(os.sep, "/"), os.path.join(WORK, "trajectory.csv").replace(os.sep, "/")))

    title("FINDINGS   (measured; nothing interpreted)")
    emit("Scored cohort at each horizon: " + "; ".join("h=%d: %s scored, %s censored (%.4f%% of %s)" % (h, f(nsc[h]), f(cens[h]), 100.0 * cens[h] / N, f(N)) for h in RH))
    emit("Training: ML %s rows from_period %s..%s, fit rows only; baseline estimated on the same %s rows (%s); prior-only from the same fit rows." % (
        f(side["n_rows"]), side["training_window"]["from_period_min_observed"], side["training_window"]["from_period_max_observed"], f(C.sum()), "identical" if same_rows else "NOT identical"))
    emit("Stored baseline vs fresh: max absolute cell difference %.6f (max count difference %s)." % (dP.max(), f(dC.max())))
    emit("")
    for h in RH:
        emit("h = %d (%s)" % (h, hz.ym(CUT_IDX + h)))
        emit("  %-10s %10s | %-24s %-24s %-24s %-24s" % ("state", "realised", "ML (err)", "baseline (err)", "prior-only (err)", "persistence (err)"))
        for j, s in enumerate(STATES):
            emit("  %-10s %9.4f%% | %-24s %-24s %-24s %-24s" % (
                s, real_share[h, j], *["%9.4f%% (%.4f)" % (share[a][h, j], err[a][h, j]) for a in ("ML", "baseline", "prior-only", "persistence")]))
        emit("  TOTAL absolute error (pp): ML %.4f, baseline %.4f, prior-only %.4f, persistence %.4f" % (tae["ML"][h], tae["baseline"][h], tae["prior-only"][h], tae["persistence"][h]))
        emit("  closest per state among ML / baseline / prior-only: " + ", ".join("%s %s" % (s, ["ML", "baseline", "prior-only"][int(np.argmin([err["ML"][h, j], err["baseline"][h, j], err["prior-only"][h, j]]))]) for j, s in enumerate(STATES)))
    emit("")
    emit("COMPOUNDING TABLE (TOTAL absolute error, percentage points)")
    emit("  %4s %10s %10s %10s %10s | %14s %14s" % ("h", "ML", "baseline", "prior-only", "persistence", "ML - baseline", "ML - prior"))
    for h in RH:
        emit("  %4d %10.4f %10.4f %10.4f %10.4f | %+14.4f %+14.4f" % (h, tae["ML"][h], tae["baseline"][h], tae["prior-only"][h], tae["persistence"][h],
                                                                       tae["ML"][h] - tae["baseline"][h], tae["ML"][h] - tae["prior-only"][h]))
    emit("  ML - baseline: %s" % trend(g_b))
    emit("  ML - prior-only: %s" % trend(g_p))
    emit("Checks: ids read back from storage equal the independent list: %s; sums to one (float64): ML %s; stored float32 within %.0e: %s (worst %.3e); no negatives: %s; absorbing mass non-decreasing: %s." % (
        "PASS" if same_ids else "FAIL", "PASS" if dev64_ml <= 1e-9 else "FAIL", STORED_TOL, "PASS" if acc_ml["dev"] <= STORED_TOL else "FAIL", acc_ml["dev"],
        "PASS" if acc_ml["minp"] >= 0 and acc_b["minp"] >= 0 else "FAIL", "PASS" if acc_ml["minstep"] >= 0 and acc_b["minstep"] >= 0 else "FAIL"))
    write_report()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("A", "B"):
        raise SystemExit("usage: python diagnostics/54_headline_h60.py A|B")
    if sys.argv[1] == "A":
        if DRY:
            raise SystemExit("phase A has no dry run")
        phase_a()
    else:
        phase_b()
