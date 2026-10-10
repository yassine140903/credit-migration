"""Thinned training: a row budget from this machine's headroom, then one thinned LightGBM fit.   READ-ONLY on data/.

    python diagnostics/34_thinned_training.py [--budget-gb 3.0] [--bytes-per-value 8] [--seed 34]

Writes diagnostics/34_thinned_training.txt and the fitted model to artifacts/thinned_lgbm/. It reads the
panel through panel_io.load_panel and changes nothing under data/. It runs no git command.

A cost and feasibility diagnostic. No prediction is made, so no log-loss, accuracy, AUC, Brier score or
confusion matrix exists anywhere in this script. Sample composition (row counts and label shares) is not a
metric and is reported.

Rows: role == 'fit' and from_state == 'Current' (a row filter, not a loan filter). Label: to_state, as integer
codes into the fixed class order panel.STATES; all six classes are declared whether or not they occur.

Thinning is at ROW level: every leaver row (to_state != 'Current') is kept, and stay rows (to_state == 'Current')
are a uniform simple random sample of the stay rows, drawn without replacement with a seeded generator. Rows of
one loan are therefore thinned independently of each other.

Loading and thinning are fused partition by partition so the full 39M-row matrix is never resident:
  pass 1  reads only from_state / to_state per partition and counts the population and the leavers/stayers;
  pass 2  reads the features per partition, applies the thinning mask and keeps only the surviving rows.
The budget is set by this machine's headroom, not by a modelling argument.
"""

import argparse
import gc
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                            # noqa: E402

OUT = os.path.join("diagnostics", "34_thinned_training.txt")
MODEL_DIR = os.path.join("artifacts", "thinned_lgbm")
MODEL_FILE = os.path.join(MODEL_DIR, "model.txt")
CLASSES = panel.STATES                      # fixed class order: label code i == CLASSES[i]
STAY = CLASSES.index("Current")
N_ROUNDS = 100
SAMPLE_S = 0.5
ABORT_MB = 500
MB = 2 ** 20
GB = 2 ** 30
ARRAYS = 3                                  # the budget formula's "3 arrays"

FEATURES = ["fico", "ltv", "dti", "rate_spread", "property_state", "loan_age", "run_length",
            "run_length_censored", "est_ltv", "unemp_level", "unemp_chg_12m"]
CATEGORICAL = "property_state"

proc = psutil.Process()
LINES = []
R = {"completed": False, "abort_reason": None, "rounds_done": 0, "phase": "start"}
PH = {}                                     # phase -> {"peak_rss": ..., "min_avail": ...}


def rss():
    return proc.memory_info().rss


def faults():
    return proc.memory_info().num_page_faults


def avail():
    return psutil.virtual_memory().available


def mb(x):
    return "%10.1f MB  (%.2f GB)" % (x / MB, x / GB)


def say(s):
    print(s, flush=True)


def write_report():
    r = R
    L = []
    w = L.append
    w("THINNED TRAINING   (cost and feasibility only - no predictive metric is computed, printed or saved)")
    w("=" * 112)
    if not r["completed"]:
        w("*** THE RUN DID NOT COMPLETE: %s ***" % r["abort_reason"])
        w("*** Everything below is what had been measured up to that point. ***")
        w("=" * 112)
    w("machine: %d logical CPUs, %.2f GB RAM, %.2f GB pagefile" % (
        psutil.cpu_count(), psutil.virtual_memory().total / GB, psutil.swap_memory().total / GB))
    w("lightgbm %s, pandas %s, numpy %s; CPU only; objective multiclass; num_class %d; %d boosting rounds; "
      "library defaults otherwise (no tuning, no early stopping, no validation set)" % (
          r.get("lgb_version", "?"), pd.__version__, np.__version__, len(CLASSES), N_ROUNDS))
    w("panel: build %s, %s rows, %s columns" % (r.get("build_stamp", "?"), format(r.get("panel_rows", 0), ","),
                                               r.get("panel_cols", "?")))
    w("")
    w("FEATURES  (resolved panel column names, in matrix order)")
    w("  %s" % r.get("features", FEATURES))
    w("  %s is passed as a native pandas category (LightGBM categorical); no integer or FIPS encoding; nulls left "
      "as-is; numerics float32." % CATEGORICAL)
    w("")
    w("STEP 1 - ROW BUDGET   (set by this machine's headroom, not by a modelling argument)")
    if "budget_bytes" in r:
        w("  memory budget:             %s bytes  (%.2f GiB; --budget-gb %.2f)" % (
            format(r["budget_bytes"], ","), r["budget_bytes"] / GB, r["budget_gb"]))
        w("  assumption, bytes/value:   %d   (--bytes-per-value)" % r["bytes_per_value"])
        w("  per-row training cost:     n_classes x arrays x bytes = %d x %d x %d = %d bytes/row" % (
            len(CLASSES), ARRAYS, r["bytes_per_value"], r["row_cost"]))
        w("  row budget:                floor(%s / %d) = %s rows" % (
            format(r["budget_bytes"], ","), r["row_cost"], format(r["row_budget"], ",")))
        w("  leaver rows (all kept):    %s" % format(r["n_leave"], ","))
        w("  stay rows in population:   %s" % format(r["n_stay"], ","))
        w("  remaining stayer budget:   %s - %s = %s" % (format(r["row_budget"], ","), format(r["n_leave"], ","),
                                                         format(r["stay_budget"], ",")))
        w("  keep rate on stay rows:    %s / %s = %.6f%s" % (
            format(r["stay_keep"], ","), format(r["n_stay"], ","), r["keep_rate"],
            "   (budget exceeds the stay rows: all kept)" if r["stay_keep"] == r["n_stay"] else ""))
        w("  note: the formula counts only the n x K x 3 value arrays. The feature matrix, LightGBM's transient copy "
          "of it, the binned data and histograms are on top of it and are what the RSS lines below measure.")
    else:
        w("  not reached")
    w("")
    w("STEP 2 - THINNING   (row level, not loan level)")
    if "thin_counts" in r:
        w("  rule: every leaver row kept (incl. all Current->Default rows); stay rows sampled uniformly at random "
          "without replacement, seed %d" % r["seed"])
        w("  population = Current-origin fit rows with a valid to_state; null/unknown labels dropped: %s"
          % format(r["n_null_label"], ","))
        pc, tc = r["pop_counts"], r["thin_counts"]
        pt, tt = sum(pc), sum(tc)
        w("  %-10s %16s %10s   %16s %10s" % ("to_state", "full rows", "share", "thinned rows", "share"))
        for i, c in enumerate(CLASSES):
            w("  %-10s %16s %9.4f%%   %16s %9.4f%%%s" % (
                c, format(pc[i], ","), 100.0 * pc[i] / pt, format(tc[i], ","), 100.0 * tc[i] / tt,
                "   <- absent; class kept" if tc[i] == 0 else ""))
        w("  %-10s %16s %10s   %16s %10s" % ("total", format(pt, ","), "", format(tt, ","), ""))
        sf, st = 100.0 * pc[STAY] / pt, 100.0 * tc[STAY] / tt
        w("")
        w("  STAY SHARE   full %.4f%%   thinned %.4f%%   difference (thinned - full) %+.4f percentage points" % (
            sf, st, st - sf))
    else:
        w("  not reached")
    w("")
    w("MEMORY CHECKPOINTS (RSS)")
    for k, lab in (("rss_load", "after load (label scan, pass 1)"), ("rss_thin", "after thinning (matrix assembled)"),
                   ("rss_dataset", "after Dataset construction"), ("rss_train", "after training")):
        w("  %-36s %s" % (lab, mb(r[k]) if k in r else "not reached"))
    if "matrix_bytes" in r:
        w("  feature matrix in memory:            %s  (%s rows x %d columns; DataFrame.memory_usage(deep=True))" % (
            mb(r["matrix_bytes"]), format(r["n_thin"], ","), len(r["features"])))
    w("")
    w("TIMING AND PEAKS")
    if "construct_s" in r:
        w("  Dataset construction:        %.1f s  (binning; outside the training timer)" % r["construct_s"])
    if "train_s" in r:
        w("  rounds completed:            %d of %d" % (r["rounds_done"], N_ROUNDS))
        w("  training wall-clock:         %.1f s  (%.1f min)   [lgb.train on the constructed Dataset]" % (
            r["train_s"], r["train_s"] / 60))
        if r["completed"]:
            w("  seconds per boosting round:  %.2f s   (training wall-clock / %d)" % (r["train_s"] / N_ROUNDS, N_ROUNDS))
        else:
            w("  seconds per boosting round:  not meaningful, the run did not complete")
    elif "construct_s" not in r:
        w("  training was not reached")
    if "min_avail" in r:
        w("  peak RSS, whole run:         %s  (max of sampler and OS peak working set)" % mb(r["peak_rss"]))
        w("  min system available memory: %s  (whole run, %d samples at %.1f s)" % (
            mb(r["min_avail"]), r["n_samples"], SAMPLE_S))
        w("  per phase (sampler)          %14s %24s" % ("peak RSS", "min available"))
        for name, d in PH.items():
            w("    %-26s %10.2f GB %20.2f GB" % (name, d["peak_rss"] / GB, d["min_avail"] / GB))
        w("  process page faults:         %s -> %s   whole-run delta %s%s" % (
            format(r["pf_start"], ","), format(r["pf_end"], ","), format(r["pf_end"] - r["pf_start"], ","),
            "" if "pf_train" not in r else "   training-only delta %s" % format(r["pf_train"], ",")))
        w("  (page-fault deltas include the soft faults that allocation and first touch produce with no paging.)")
        w("  system pagefile used, start / max / end: %.2f / %.2f / %.2f GB" % (
            r["swap_start"] / GB, r["swap_max"] / GB, r["swap_end"] / GB))
    if "model_path" in r:
        w("  model saved: %s   (no predictions saved)" % r["model_path"])
    w("")
    w("VERDICT: %s" % r.get("verdict", "no verdict"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    say("\n".join(L))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.min_avail = avail()
        self.peak_rss = rss()
        self.swap_max = psutil.swap_memory().used
        self.n = 0

    def run(self):
        while not self.stop_flag.is_set():
            a, s = avail(), rss()
            self.min_avail = min(self.min_avail, a)
            self.peak_rss = max(self.peak_rss, s)
            self.swap_max = max(self.swap_max, psutil.swap_memory().used)
            d = PH.setdefault(R["phase"], {"peak_rss": s, "min_avail": a})
            d["peak_rss"] = max(d["peak_rss"], s)
            d["min_avail"] = min(d["min_avail"], a)
            self.n += 1
            if a < ABORT_MB * MB:
                self.abort(a)
            self.stop_flag.wait(SAMPLE_S)

    def abort(self, a):
        R["abort_reason"] = ("available system memory fell to %.0f MB, below the %d MB limit, during phase '%s' "
                             "(%d boosting rounds completed)" % (a / MB, ABORT_MB, R["phase"], R["rounds_done"]))
        R["verdict"] = "DID NOT COMPLETE - aborted for low memory; the run did not fit in memory."
        if "t_train0" in R and "train_s" not in R:
            R["train_s"] = time.perf_counter() - R["t_train0"]
        self.fill()
        write_report()
        os._exit(2)

    def fill(self):
        R.update(peak_rss=max(self.peak_rss, getattr(proc.memory_info(), "peak_wset", 0)),
                 min_avail=self.min_avail, n_samples=self.n, swap_max=self.swap_max,
                 pf_end=faults(), swap_end=psutil.swap_memory().used)


def codes_of(s):
    """to_state as codes into CLASSES; -1 for null or an unknown label."""
    return s.cat.set_categories(CLASSES).cat.codes.to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget-gb", type=float, default=3.0, help="memory budget in GiB (default 3.0)")
    ap.add_argument("--bytes-per-value", type=int, default=8, help="assumed bytes per array value (default 8)")
    ap.add_argument("--seed", type=int, default=34, help="seed for the stay-row sample (default 34)")
    args = ap.parse_args()

    import lightgbm as lgb
    R["lgb_version"] = lgb.__version__
    R["seed"] = args.seed
    manifest = panel_io.read_manifest()
    R["build_stamp"] = manifest["build_stamp"]
    R["panel_rows"] = manifest["total_rows"]
    have = [c["name"] for c in manifest["schema"]]
    R["panel_cols"] = len(have)
    missing = [c for c in FEATURES if c not in have]
    if missing:
        raise SystemExit("features not in the panel on disk: %s (panel has %s)" % (missing, have))
    R["features"] = list(FEATURES)
    say("resolved feature columns: %s" % FEATURES)
    quarters = [p["quarter"] for p in manifest["partitions"]]

    sampler = Sampler()
    R["pf_start"] = faults()
    R["swap_start"] = psutil.swap_memory().used
    sampler.start()

    # ---- step 1: label scan and row budget (pass 1) ---------------------------------------------------------
    R["phase"] = "pass 1: label scan"
    pop = np.zeros(len(CLASSES), np.int64)
    stay_q, leave_q = [], []
    n_null = 0
    for q in quarters:
        d = panel_io.load_panel(roles=["fit"], columns=["from_state", "to_state"], quarters=[q])
        d = d[d["from_state"] == "Current"]
        c = codes_of(d["to_state"])
        n_null += int((c < 0).sum())
        c = c[c >= 0]
        cnt = np.bincount(c, minlength=len(CLASSES))
        pop += cnt
        stay_q.append(int(cnt[STAY]))
        leave_q.append(int(cnt.sum() - cnt[STAY]))
        del d, c
        say("  scan %s: %s rows, %s leavers" % (q, format(stay_q[-1] + leave_q[-1], ","), format(leave_q[-1], ",")))
    gc.collect()
    n_stay, n_leave = int(pop[STAY]), int(pop.sum() - pop[STAY])
    R.update(pop_counts=[int(x) for x in pop], n_null_label=n_null, n_stay=n_stay, n_leave=n_leave)

    budget_bytes = int(args.budget_gb * GB)
    row_cost = len(CLASSES) * ARRAYS * args.bytes_per_value
    row_budget = budget_bytes // row_cost
    if n_leave > row_budget:
        R["abort_reason"] = ("the %s leaver rows alone exceed the row budget of %s; raise --budget-gb"
                             % (format(n_leave, ","), format(row_budget, ",")))
        R["verdict"] = "DID NOT RUN."
        sampler.stop_flag.set()
        sampler.fill()
        R.update(budget_bytes=budget_bytes, budget_gb=args.budget_gb, bytes_per_value=args.bytes_per_value,
                 row_cost=row_cost, row_budget=row_budget, stay_budget=0, stay_keep=0, keep_rate=0.0)
        write_report()
        raise SystemExit(R["abort_reason"])
    stay_budget = row_budget - n_leave
    stay_keep = min(n_stay, stay_budget)
    R.update(budget_bytes=budget_bytes, budget_gb=args.budget_gb, bytes_per_value=args.bytes_per_value,
             row_cost=row_cost, row_budget=row_budget, stay_budget=stay_budget, stay_keep=stay_keep,
             keep_rate=stay_keep / n_stay)
    R["rss_load"] = rss()
    say("row budget %s rows; leavers %s; stay keep %s of %s (rate %.6f); RSS %.2f GB" % (
        format(row_budget, ","), format(n_leave, ","), format(stay_keep, ","), format(n_stay, ","),
        R["keep_rate"], R["rss_load"] / GB))

    # ---- step 2: thinning (pass 2) --------------------------------------------------------------------------
    R["phase"] = "pass 2: thinning"
    rng = np.random.default_rng(args.seed)
    stay_mask = np.zeros(n_stay, bool)                      # over the global stay-row index space, in file order
    stay_mask[rng.choice(n_stay, size=stay_keep, replace=False)] = True
    chunks, ys = [], []
    offset = 0
    for qi, q in enumerate(quarters):
        d = panel_io.load_panel(roles=["fit"], columns=FEATURES + ["from_state", "to_state"], quarters=[q])
        d = d[d["from_state"] == "Current"]
        c = codes_of(d["to_state"])
        valid = c >= 0
        d, c = d[valid], c[valid]
        is_stay = c == STAY
        si = np.flatnonzero(is_stay)
        if len(si) != stay_q[qi] or len(c) - len(si) != leave_q[qi]:
            raise RuntimeError("%s: pass 2 saw %d stay / %d leave rows, pass 1 saw %d / %d"
                               % (q, len(si), len(c) - len(si), stay_q[qi], leave_q[qi]))
        keep = np.ones(len(c), bool)
        keep[si] = stay_mask[offset:offset + len(si)]
        offset += len(si)
        d = d[keep].drop(columns=["from_state", "to_state"]).reset_index(drop=True)
        for col in FEATURES:
            if col != CATEGORICAL:
                d[col] = d[col].astype(np.float32)
        chunks.append(d)
        ys.append(c[keep].astype(np.float32))
        del d, c, valid, is_stay, si, keep
        gc.collect()
        say("  thin %s: kept %s rows" % (q, format(len(chunks[-1]), ",")))
    del stay_mask
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")
    y = np.concatenate(ys)
    del ys
    gc.collect()
    thin = np.bincount(y.astype(np.int64), minlength=len(CLASSES))
    R.update(thin_counts=[int(x) for x in thin], n_thin=len(df), matrix_bytes=int(df.memory_usage(deep=True).sum()))
    R["rss_thin"] = rss()
    say("thinned matrix %s x %d, %.2f GB; RSS %.2f GB" % (format(len(df), ","), df.shape[1],
                                                         R["matrix_bytes"] / GB, R["rss_thin"] / GB))

    # ---- step 3: Dataset, then training ---------------------------------------------------------------------
    R["phase"] = "Dataset construction"
    params = {"objective": "multiclass", "num_class": len(CLASSES), "device_type": "cpu", "verbosity": -1}
    ds = lgb.Dataset(df, label=y, categorical_feature=[CATEGORICAL], free_raw_data=True)
    del df, y                                               # LightGBM's own reference is dropped when it constructs
    gc.collect()
    t0 = time.perf_counter()
    ds.construct()
    R["construct_s"] = time.perf_counter() - t0
    gc.collect()
    R["rss_dataset"] = rss()
    say("Dataset constructed in %.1f s; RSS %.2f GB" % (R["construct_s"], R["rss_dataset"] / GB))

    def count_rounds(env):
        R["rounds_done"] = env.iteration + 1

    R["phase"] = "training"
    pf0 = faults()
    R["t_train0"] = time.perf_counter()
    booster = lgb.train(params, ds, num_boost_round=N_ROUNDS, callbacks=[count_rounds])
    R["train_s"] = time.perf_counter() - R["t_train0"]
    R["pf_train"] = faults() - pf0
    R["rss_train"] = rss()
    R["rounds_done"] = booster.current_iteration()
    sampler.stop_flag.set()
    sampler.join()
    sampler.fill()
    R["completed"] = R["rounds_done"] == N_ROUNDS
    R["abort_reason"] = None if R["completed"] else "stopped at %d rounds" % R["rounds_done"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    booster.save_model(MODEL_FILE)
    R["model_path"] = MODEL_FILE.replace(os.sep, "/")

    grew = R["swap_max"] - R["swap_start"]
    if not R["completed"]:
        R["verdict"] = "DID NOT COMPLETE."
    elif grew > 256 * MB:
        R["verdict"] = ("Completed, but the machine PAGED: pagefile use rose %.2f GB (min available %.2f GB)."
                        % (grew / GB, R["min_avail"] / GB))
    elif R["min_avail"] < 1 * GB:
        R["verdict"] = ("Completed with no pagefile growth, but memory was tight: min available %.2f GB; no evidence "
                        "of paging." % (R["min_avail"] / GB))
    else:
        R["verdict"] = ("Completed in memory without paging: min available %.2f GB, pagefile moved %.2f GB."
                        % (R["min_avail"] / GB, grew / GB))
    write_report()


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        R["abort_reason"] = "MemoryError raised during phase '%s'" % R["phase"]
        R["verdict"] = "DID NOT COMPLETE - MemoryError; the run did not fit in memory."
        write_report()
        raise
