"""Training cost: time and memory of one LightGBM multiclass fit on the fit-role Current rows.   READ-ONLY.

    python diagnostics/33_training_cost.py          # from the repo root

Writes diagnostics/33_training_cost.txt and nothing else. (Number 32 was already taken by
32_rebaseline, so this is 33.) It reads the committed panel through panel_io.load_panel and
changes nothing under data/.

This is a cost measurement, not a modelling run: no validation set, no scoring, no predictive
metric of any kind is computed, printed or written. The only thing derived from the fitted
model is the number of boosting rounds it completed.

Row filter (not a loan filter): role == 'fit' and from_state == 'Current'. Label: to_state, as
integer codes into the fixed class order panel.STATES; all six classes are declared whether or
not they occur in the labels. Features are the panel's own columns, nulls untouched, numerics
cast to float32, property_state passed as a native pandas category (LightGBM categorical).

The timed window is the whole lgb.train(...) call, which includes LightGBM's Dataset
construction (binning), because that happens lazily inside train() and is part of the cost.
A sampler thread records, every SAMPLE_S seconds during that window, the system's available
memory and this process's RSS and page-fault count. If available memory drops below ABORT_MB
the sampler writes the report as measured so far and ends the process.
"""

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

OUT = os.path.join("diagnostics", "33_training_cost.txt")
CLASSES = panel.STATES                      # fixed class order: label code i == CLASSES[i]
N_ROUNDS = 100
SAMPLE_S = 0.5
ABORT_MB = 500
MB = 2 ** 20
GB = 2 ** 30

# requested feature -> panel column; the three renamed ones are the panel's own names
WANTED = [("fico", "fico"), ("original_ltv", "ltv"), ("dti", "dti"), ("rate_spread", "rate_spread"),
          ("property_state", "property_state"), ("loan_age", "loan_age"), ("run_length", "run_length"),
          ("run_length_censored", "run_length_censored"), ("est_ltv", "est_ltv"),
          ("unemployment level", "unemp_level"), ("unemployment 12-month change", "unemp_chg_12m")]
CATEGORICAL = "property_state"

proc = psutil.Process()
LINES = []
R = {"completed": False, "rounds_done": 0, "abort_reason": None}


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def rss():
    return proc.memory_info().rss


def faults():
    return proc.memory_info().num_page_faults


def avail():
    return psutil.virtual_memory().available


def mb(x):
    return "%10.1f MB  (%.2f GB)" % (x / MB, x / GB)


def write_report():
    r = R
    L = []
    w = L.append
    w("TRAINING COST   (cost measurement only - no predictive metric is computed or reported)")
    w("=" * 110)
    if not r["completed"]:
        w("*** THE RUN DID NOT COMPLETE: %s ***" % r["abort_reason"])
        w("*** Everything below is what had been measured up to that point. ***")
        w("=" * 110)
    w("machine: %d logical CPUs, %.2f GB RAM, %.2f GB swap/pagefile" % (
        psutil.cpu_count(), psutil.virtual_memory().total / GB, psutil.swap_memory().total / GB))
    w("lightgbm %s, pandas %s, numpy %s, CPU only, objective multiclass, num_class %d, %d boosting rounds, "
      "library defaults otherwise" % (r.get("lgb_version", "?"), pd.__version__, np.__version__, len(CLASSES),
                                      N_ROUNDS))
    w("")
    w("FEATURES (requested -> panel column)")
    for req, col in WANTED:
        if col in r.get("resolved", []):
            w("  %-30s -> %s%s" % (req, col, "   [native categorical]" if col == CATEGORICAL else ""))
        else:
            w("  %-30s -> MISSING from the panel on disk; NOT in the matrix" % req)
    if r.get("missing"):
        w("")
        w("!! %s absent from the panel on disk (data/panel manifest build %s), so the matrix has %d columns, not %d."
          % (", ".join(r["missing"]), r.get("build_stamp", "?"), len(r["resolved"]), len(WANTED)))
        w("!! The panel was not rebuilt (that would modify data/). Every figure below is for the %d-column matrix."
          % len(r["resolved"]))
        for c in r["missing"]:
            w("!! Expected extra cost of one more float32 column over %s rows: matrix +%.1f MB, LightGBM's own copy "
              "+%.1f MB." % (format(r.get("n_rows", 0), ","), r.get("n_rows", 0) * 4 / MB, r.get("n_rows", 0) * 4 / MB))
    w("  resolved list: %s" % r.get("resolved"))
    w("")
    w("DATA")
    w("  row filter: role == 'fit' and from_state == 'Current'   (a row filter, not a loan filter)")
    w("  rows after filter:          %s" % format(r.get("n_rows", 0), ","))
    w("  rows dropped for null label: %s" % format(r.get("n_null_label", 0), ","))
    w("  feature matrix columns:     %d" % len(r.get("resolved", [])))
    w("  feature matrix in memory:   %s   (DataFrame.memory_usage(deep=True); numerics float32, "
      "property_state int8 category)" % mb(r.get("matrix_bytes", 0)))
    w("  class set declared (fixed order): %s" % CLASSES)
    w("  label counts in the training rows (class: rows):")
    for i, c in enumerate(CLASSES):
        n = r.get("label_counts", [0] * len(CLASSES))[i]
        w("    %d %-8s %14s%s" % (i, c, format(n, ","), "   <- ABSENT from training labels; class kept" if n == 0
                                  and "label_counts" in r else ""))
    w("")
    w("RSS CHECKPOINTS")
    for k, lab in (("rss_load", "after load"), ("rss_matrix", "after feature matrix built"),
                   ("rss_train", "after training")):
        w("  %-28s %s" % (lab, mb(r[k]) if k in r else "not reached"))
    w("")
    w("TRAINING   (timed window = the whole lgb.train call, including Dataset construction)")
    if "train_s" in r:
        w("  rounds completed:           %d of %d" % (r["rounds_done"], N_ROUNDS))
        w("  total wall-clock:           %.1f s  (%.1f min)" % (r["train_s"], r["train_s"] / 60))
        if r["completed"]:
            w("  seconds per boosting round: %.2f s   (total / %d, includes Dataset construction)" % (
                r["train_s"] / N_ROUNDS, N_ROUNDS))
        else:
            w("  seconds per boosting round: not meaningful, the run did not complete")
        w("  peak RSS during training:   %s   (max of the sampler and the OS peak working set)" % mb(r["peak_rss"]))
        w("  min system available mem:   %s   (%d samples at %.1f s)" % (mb(r["min_avail"]), r["n_samples"], SAMPLE_S))
        w("  available at start / end:   %.2f GB / %.2f GB" % (r["avail_start"] / GB, r["avail_end"] / GB))
        w("  process page faults:        %s -> %s   delta %s" % (
            format(r["pf_start"], ","), format(r["pf_end"], ","), format(r["pf_end"] - r["pf_start"], ",")))
        w("  (page-fault delta counts soft faults too, which allocation and first touch produce even with no "
          "paging; read it with min available memory and the swap line.)")
        w("  system swap used at start / max / end: %.2f / %.2f / %.2f GB" % (
            r["swap_start"] / GB, r["swap_max"] / GB, r["swap_end"] / GB))
    else:
        w("  training was not started")
    w("")
    w("VERDICT: %s" % r.get("verdict", "no verdict"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


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
            a = avail()
            self.min_avail = min(self.min_avail, a)
            self.peak_rss = max(self.peak_rss, rss())
            self.swap_max = max(self.swap_max, psutil.swap_memory().used)
            self.n += 1
            if a < ABORT_MB * MB:
                self.abort(a)
            self.stop_flag.wait(SAMPLE_S)

    def abort(self, a):
        R["abort_reason"] = ("available system memory fell to %.0f MB, below the %d MB limit, after %d completed "
                             "boosting rounds" % (a / MB, ABORT_MB, R["rounds_done"]))
        self.fill()
        R["train_s"] = time.perf_counter() - R["t0"]
        R["verdict"] = "DID NOT COMPLETE - aborted for low memory; the run did not fit in memory."
        write_report()
        os._exit(2)

    def fill(self):
        R.update(peak_rss=max(self.peak_rss, getattr(proc.memory_info(), "peak_wset", 0)),
                 min_avail=self.min_avail, n_samples=self.n, swap_max=self.swap_max,
                 avail_end=avail(), pf_end=faults(), swap_end=psutil.swap_memory().used)


def main():
    import lightgbm as lgb
    R["lgb_version"] = lgb.__version__
    m = panel_io.read_manifest()
    R["build_stamp"] = m["build_stamp"]
    have = [c["name"] for c in m["schema"]]
    R["resolved"] = [col for _, col in WANTED if col in have]
    R["missing"] = [col for _, col in WANTED if col not in have]
    emit("resolved feature list: %s" % R["resolved"])
    if R["missing"]:
        emit("NOT in the panel on disk: %s" % R["missing"])
    feats = R["resolved"]

    # ---- load -------------------------------------------------------------
    df = panel_io.load_panel(roles=["fit"], columns=feats + ["from_state", "to_state"])
    df = df[df["from_state"] == "Current"]
    df = df.drop(columns="from_state").reset_index(drop=True)
    gc.collect()
    R["rss_load"] = rss()
    emit("loaded %s fit/Current rows; RSS %.2f GB" % (format(len(df), ","), R["rss_load"] / GB))

    # ---- label and feature matrix ----------------------------------------
    codes = pd.Categorical(df.pop("to_state").astype(str), categories=CLASSES).codes   # -1 for null/unknown
    bad = codes < 0
    R["n_null_label"] = int(bad.sum())
    if bad.any():
        keep = ~bad
        df = df[keep].reset_index(drop=True)
        codes = codes[keep]
    y = codes.astype(np.float32)
    del codes
    R["label_counts"] = [int(n) for n in np.bincount(y.astype(np.int64), minlength=len(CLASSES))]
    for c in feats:
        if c != CATEGORICAL:
            df[c] = df[c].astype(np.float32)
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")
    gc.collect()
    R["n_rows"] = len(df)
    R["matrix_bytes"] = int(df.memory_usage(deep=True).sum())
    R["rss_matrix"] = rss()
    emit("matrix %d x %d, %.2f GB; RSS %.2f GB" % (len(df), df.shape[1], R["matrix_bytes"] / GB,
                                                    R["rss_matrix"] / GB))

    # ---- train ------------------------------------------------------------
    params = {"objective": "multiclass", "num_class": len(CLASSES), "device_type": "cpu", "verbosity": -1}

    def count_rounds(env):
        R["rounds_done"] = env.iteration + 1

    sampler = Sampler()
    R["avail_start"] = avail()
    R["pf_start"] = faults()
    R["swap_start"] = psutil.swap_memory().used
    R["t0"] = time.perf_counter()
    sampler.start()
    emit("training started")
    booster = lgb.train(params, lgb.Dataset(df, label=y, categorical_feature=[CATEGORICAL]),
                        num_boost_round=N_ROUNDS, callbacks=[count_rounds])
    R["train_s"] = time.perf_counter() - R["t0"]
    sampler.stop_flag.set()
    sampler.join()
    sampler.fill()
    R["rss_train"] = rss()
    R["rounds_done"] = booster.current_iteration()
    R["completed"] = R["rounds_done"] == N_ROUNDS
    R["abort_reason"] = None if R["completed"] else "stopped at %d rounds" % R["rounds_done"]

    swap_grew = R["swap_max"] - R["swap_start"] > 256 * MB
    low = R["min_avail"] < 1 * GB
    if not R["completed"]:
        R["verdict"] = "DID NOT COMPLETE."
    elif swap_grew:
        R["verdict"] = ("Completed, but the machine PAGED: system swap rose %.2f GB during training (min available "
                        "%.2f GB)." % ((R["swap_max"] - R["swap_start"]) / GB, R["min_avail"] / GB))
    elif low:
        R["verdict"] = ("Completed with no swap growth, but memory was tight: min available %.2f GB; no evidence of "
                        "paging." % (R["min_avail"] / GB))
    else:
        R["verdict"] = ("Completed in memory without paging: min available %.2f GB, no swap growth (swap moved "
                        "%.2f GB)." % (R["min_avail"] / GB, (R["swap_max"] - R["swap_start"]) / GB))
    write_report()


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        R["abort_reason"] = "MemoryError raised"
        R["verdict"] = "DID NOT COMPLETE - MemoryError; the run did not fit in memory."
        write_report()
        raise
