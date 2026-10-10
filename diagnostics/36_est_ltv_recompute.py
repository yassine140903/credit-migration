"""est_ltv recompute: is the stored-vs-recomputed gap float32 rounding, or a real divergence?   READ-ONLY.

    python diagnostics/36_est_ltv_recompute.py        # from the repo root

Writes diagnostics/36_est_ltv_recompute.txt and nothing else. It reads the panel through panel_io.load_panel and the
HPI table (data/macro/hpi_state_monthly.csv) through panel.Hpi, changes nothing and fixes nothing. No git command.

How the panel's columns are written (src/panel.py hpi_features(), lines 631-654, called at line 1407; the checker in
src/features_hpi.py verify() recomputes the same arithmetic and demands exact float32 equality):

    ratio     = hpi_disagg[state, from_month] / hpi_disagg[state, orig_month]     float64 / float64, NOT rounded
    x         = ltv.astype(float64)                                                ltv is float32 in the panel
    hpi_ratio = ratio.astype(float32)                                              stored
    est_ltv   = (x / ratio).astype(float32)                                        stored; divides by the UNROUNDED ratio

Recomputations compared with the stored est_ltv:
    RAW     (x / hpi_ratio_stored.astype(float64)), float64, no cast         <- "the original recompute": from the stored
            ltv and the stored (float32-rounded) hpi_ratio, as the task specifies
    CAST    the same, then .astype(float32)                                 <- the float32 test; this is diagnostic 35's
            comparison
    ORACLE  the writer's own arithmetic: unrounded ratio from the HPI table, orig_month = from_month - loan_age (the
            panel's definition, true_age = from_month - orig_month), est_ltv = (x / ratio).astype(float32).
            This is the decisive test: it separates "the stored hpi_ratio was rounded to float32 before the divide"
            from "est_ltv is not ltv / HPI ratio at all".

Points where this script chose: (1) RAW/CAST divide by the stored float32 hpi_ratio, as the task says, although the writer
divides by the unrounded one - so a gap is expected by construction and its size is what is measured; (2) 'original
mismatches' means RAW mismatches (stored, upcast to float64, != float64 recompute); (3) median is the lower middle value
and p99 the value at index int(0.99 (n-1)) of the sorted differences; (4) rows with a null in ltv, hpi_ratio or est_ltv
are excluded and counted (the build asserts there are none).

Populations: (a) role == condition and from_period == 200812 (diagnostic 35's); (b) every panel row, one quarter at a time.
"""

import gc
import os
import sys
import threading
from collections import Counter

import numpy as np
import psutil
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import panel, panel_io                                            # noqa: E402

OUT = os.path.join("diagnostics", "36_est_ltv_recompute.txt")
CUT = 200812
COLS = ["from_period", "role", "property_state", "loan_age", "ltv", "hpi_ratio", "est_ltv"]
EDGES = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
LABELS = ["== 0", "(0, 1e-6]", "(1e-6, 1e-5]", "(1e-5, 1e-4]", "(1e-4, 1e-3]", "(1e-3, 1e-2]", "(1e-2, 1e-1]",
          "(1e-1, 1]", "(1, 10]", "> 10"]
TOP = 20
SLICE = 2_000_000
SAMPLE_S = 0.5
ABORT_MB = 500
MB, GB = 2 ** 20, 2 ** 30

proc = psutil.Process()
R = {"completed": False, "abort_reason": None, "phase": "start"}
ACC = {}


def rss():
    return proc.memory_info().rss


def avail():
    return psutil.virtual_memory().available


def say(s):
    print(s, flush=True)


def f(n):
    return format(int(n), ",")


def bucket_counts(d):
    """Counts per LABELS bucket for a non-negative, non-NaN array."""
    out = np.bincount(np.searchsorted(EDGES, d, side="left") + 1, minlength=10)
    z = int((d == 0).sum())
    out[1] -= z                                        # exact zeros leave the first nonzero bucket ...
    out[0] = z                                         # ... and form their own
    return out[:10]


class Variant:
    """Running statistics of one signed difference (recomputed - stored)."""

    def __init__(self):
        self.n = self.n_diff = self.gt = self.lt = self.over05 = self.over10 = self.n_rel = 0
        self.sum_abs = self.sum_rel = self.max_abs = self.max_rel = 0.0
        self.min_abs = self.min_rel = np.inf
        self.hist_abs = np.zeros(10, np.int64)
        self.hist_rel = np.zeros(10, np.int64)
        self.abs_parts, self.rel_parts = [], []

    def add(self, signed, st64):
        a = np.abs(signed)
        self.n += len(a)
        self.n_diff += int((a > 0).sum())
        self.gt += int((signed > 0).sum())
        self.lt += int((signed < 0).sum())
        self.over05 += int((a > 0.5).sum())
        self.over10 += int((a > 1.0).sum())
        self.sum_abs += float(a.sum())
        if len(a):
            self.min_abs = min(self.min_abs, float(a.min()))
            self.max_abs = max(self.max_abs, float(a.max()))
        self.hist_abs += bucket_counts(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / np.abs(st64)
        r = r[np.isfinite(r)]
        self.n_rel += len(r)
        self.sum_rel += float(r.sum())
        if len(r):
            self.max_rel = max(self.max_rel, float(r.max()))
            self.min_rel = min(self.min_rel, float(r.min()))
        self.hist_rel += bucket_counts(r)
        self.abs_parts.append(a.astype(np.float32))
        self.rel_parts.append(r.astype(np.float32))

    def finish(self):
        for name in ("abs", "rel"):
            parts = getattr(self, name + "_parts")
            x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            setattr(self, name + "_parts", [])
            if len(x):
                k1, k2 = (len(x) - 1) // 2, int(0.99 * (len(x) - 1))
                x.partition([k1, k2])
                setattr(self, name + "_med", float(x[k1]))
                setattr(self, name + "_p99", float(x[k2]))
            else:
                setattr(self, name + "_med", float("nan"))
                setattr(self, name + "_p99", float("nan"))
            del x
            gc.collect()


class Acc:
    def __init__(self, name):
        self.name = name
        self.n_rows = self.n_null = 0
        self.raw, self.cast = Variant(), Variant()
        self.ulp = np.zeros(4, np.int64)                # raw gap <= 0.5, <= 1, <= 2, > 2 ulp of the stored value
        self.ulp_max = 0.0
        self.bit_equal = 0
        self.or_n = self.or_est = self.or_ratio = self.or_out = 0
        self.top = []                                   # (abs raw diff, qi, pos, ltv, ratio, stored, recomputed, state, fp)
        self.bd = {"property_state": Counter(), "origination year": Counter(), "from_period year": Counter()}

    def add(self, v, mask, qi, base):
        m = mask
        n_valid = int(m.sum())
        self.n_rows += n_valid
        if n_valid == 0:
            return
        s = lambda a: a[m]                              # noqa: E731
        st64, raw, castd = s(v["st64"]), s(v["raw"]), s(v["castd"])
        self.raw.add(raw, st64)
        self.cast.add(castd, st64)
        ulps = np.abs(raw) / s(v["spacing"])
        self.ulp += np.array([(ulps <= 0.5).sum(), ((ulps > 0.5) & (ulps <= 1)).sum(),
                              ((ulps > 1) & (ulps <= 2)).sum(), (ulps > 2).sum()])
        self.ulp_max = max(self.ulp_max, float(ulps.max()))
        self.bit_equal += int(s(v["bits_equal"]).sum())
        om = m & v["or_ok"]
        self.or_n += int(om.sum())
        self.or_out += int((m & ~v["or_ok"]).sum())
        self.or_est += int(v["or_est_bad"][om].sum())
        self.or_ratio += int(v["or_ratio_bad"][om].sum())

        pos = np.flatnonzero(m)
        a = np.abs(raw)
        k = min(TOP, len(a))
        for i in np.argpartition(-a, k - 1)[:k]:
            j = pos[i]
            self.top.append((float(a[i]), qi, base + int(j), float(v["ltv"][j]), float(v["ratio"][j]), float(v["st64"][j]),
                             float(v["rec64"][j]), v["state_str"][j], int(v["fp"][j])))
        self.top = sorted(self.top, key=lambda t: -t[0])[:TOP]

        big = a > 0.5
        if big.any():
            bj = pos[big]
            for key, vals in (("property_state", v["state_str"][bj]), ("origination year", v["orig_year"][bj]),
                              ("from_period year", v["fp"][bj] // 100)):
                u, c = np.unique(vals, return_counts=True)
                self.bd[key].update({(x if isinstance(x, str) else int(x)): int(y) for x, y in zip(u, c)})


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.min_avail, self.peak_rss, self.n = avail(), rss(), 0

    def run(self):
        while not self.stop_flag.is_set():
            a = avail()
            self.min_avail = min(self.min_avail, a)
            self.peak_rss = max(self.peak_rss, rss())
            self.n += 1
            if a < ABORT_MB * MB:
                R["abort_reason"] = ("available system memory fell to %.0f MB, below the %d MB limit, during '%s'"
                                     % (a / MB, ABORT_MB, R["phase"]))
                R["verdict"] = "DID NOT COMPLETE - aborted for low memory."
                self.fill()
                write_report(partial=True)
                os._exit(2)
            self.stop_flag.wait(SAMPLE_S)

    def fill(self):
        R.update(peak_rss=max(self.peak_rss, getattr(proc.memory_info(), "peak_wset", 0)),
                 min_avail=self.min_avail, n_samples=self.n)


def population_block(w, acc, partial):
    w("-" * 112)
    w("POPULATION %s" % acc.name)
    w("-" * 112)
    w("rows compared: %s   (rows excluded for a null in ltv / hpi_ratio / est_ltv: %s)" % (f(acc.n_rows), f(acc.n_null)))
    if acc.n_rows == 0:
        w("  no rows")
        return
    for lab, v in (("RAW   float64 recompute from stored ltv and stored hpi_ratio, no cast, vs stored est_ltv", acc.raw),
                   ("CAST  the same recompute cast to float32 (the float32 test; diagnostic 35's comparison)", acc.cast)):
        w("")
        w(lab)
        w("  rows that differ at all:  %s  (%.4f%%)" % (f(v.n_diff), 100.0 * v.n_diff / v.n))
        if partial:
            continue
        w("  absolute difference (LTV points):  min %.3e   median %.3e   mean %.3e   p99 %.3e   max %.3e" % (
            v.min_abs, v.abs_med, v.sum_abs / v.n, v.abs_p99, v.max_abs))
        w("  relative difference:               min %.3e   median %.3e   mean %.3e   p99 %.3e   max %.3e" % (
            v.min_rel, v.rel_med, v.sum_rel / max(v.n_rel, 1), v.rel_p99, v.max_rel))
        w("  %-16s %16s %16s" % ("bucket", "absolute", "relative"))
        for i, lab2 in enumerate(LABELS):
            w("  %-16s %16s %16s" % (lab2, f(v.hist_abs[i]), f(v.hist_rel[i])))
        w("  differing by more than 0.5 LTV points: %s    by more than 1.0 LTV point: %s" % (f(v.over05), f(v.over10)))
        eq = v.n - v.gt - v.lt
        w("  direction (recomputed vs stored): higher %s   lower %s   equal %s   -> %s" % (
            f(v.gt), f(v.lt), f(eq),
            "recomputed always higher" if v.lt == 0 and v.gt else "recomputed always lower" if v.gt == 0 and v.lt
            else "no differences" if v.gt == v.lt == 0 else "MIXED"))
    if partial:
        return
    w("")
    w("FLOAT32 TEST")
    w("  original (RAW) mismatches: %s;  after the float64 recompute is cast to float32, mismatches that survive: %s" % (
        f(acc.raw.n_diff), f(acc.cast.n_diff)))
    w("  exact bit-level equality of the cast recompute with stored est_ltv: %s of %s rows (%.4f%%)" % (
        f(acc.bit_equal), f(acc.n_rows), 100.0 * acc.bit_equal / acc.n_rows))
    w("  RAW gap in units of one float32 spacing at the stored value: <=0.5 ulp %s, (0.5,1] %s, (1,2] %s, >2 %s; "
      "max %.3f ulp" % (f(acc.ulp[0]), f(acc.ulp[1]), f(acc.ulp[2]), f(acc.ulp[3]), acc.ulp_max))
    w("  ORACLE (the writer's arithmetic: unrounded HPI ratio, orig_month = from_month - loan_age, divide, cast):")
    w("    rows recomputable %s (outside the HPI series or unknown state: %s)" % (f(acc.or_n), f(acc.or_out)))
    w("    est_ltv    stored != oracle on %s rows" % f(acc.or_est))
    w("    hpi_ratio  stored != float32(oracle ratio) on %s rows" % f(acc.or_ratio))
    w("")
    w("RAW gap > 0.5 LTV points, by property_state / origination year / from_period year")
    if acc.raw.over05 == 0:
        w("  none: no row differs by more than 0.5 LTV points")
    else:
        for key, c in acc.bd.items():
            w("  %s: %s" % (key, ", ".join("%s %s" % (k, f(n)) for k, n in sorted(c.items(), key=lambda t: str(t[0])))))
    w("")
    w("%d LARGEST RAW ABSOLUTE DIFFERENCES" % TOP)
    w("  %-14s %-4s %-7s %8s %12s %14s %14s %12s" % ("loan_id", "st", "period", "ltv", "hpi_ratio", "stored est_ltv",
                                                    "recomputed", "abs diff"))
    for t in acc.top:
        w("  %-14s %-4s %-7d %8.1f %12.7f %14.7f %14.7f %12.3e" % (
            acc.top_ids.get((t[1], t[2]), "?"), t[7], t[8], t[3], t[4], t[5], t[6], t[0]))


def write_report(partial=False):
    L = []
    w = L.append
    w("est_ltv RECOMPUTE   (read-only; fixes nothing; no predictive metric)")
    w("=" * 112)
    if partial or not R["completed"]:
        w("*** THE RUN DID NOT COMPLETE: %s ***" % R["abort_reason"])
        w("*** Counts below are what had been accumulated; distributions were not finished. ***")
        w("=" * 112)
    w("panel build %s, %s rows" % (R.get("build_stamp", "?"), f(R.get("total_rows", 0))))
    w("")
    w("THE FORMULA AS THE CODE IMPLEMENTS IT   (writer: src/panel.py hpi_features(), lines 631-654; checker: "
      "src/features_hpi.py verify())")
    w("  ratio     = hpi_disagg[state, from_month] / hpi_disagg[state, orig_month]     float64 / float64, not rounded")
    w("  x         = ltv.astype(float64)                                                the panel's ltv is float32")
    w("  hpi_ratio = ratio.astype(float32)                                              stored")
    w("  est_ltv   = (x / ratio).astype(float32)                                        stored; divides by the UNROUNDED ratio")
    w("  features_hpi.verify() recomputes all three from the HPI table and the origination record and requires exact "
      "float32 equality.")
    w("")
    w("WHERE THIS SCRIPT HAD TO CHOOSE")
    w("  1. RAW and CAST divide by the STORED float32 hpi_ratio, as the task says, although the writer divides by the "
      "unrounded ratio. A gap is therefore expected by construction; its size is what is measured.")
    w("  2. 'Original mismatches' = RAW mismatches (stored, upcast to float64, != the float64 recompute).")
    w("  3. ORACLE recomputes the writer's arithmetic from the HPI table, with orig_month = from_month - loan_age "
      "(the panel's definition; features_hpi.verify() checks it against the origination record).")
    w("  4. median = lower middle value; p99 = value at index int(0.99 (n-1)) of the sorted differences; relative "
      "difference = |recomputed - stored| / |stored|; rows with any null input are excluded and counted.")
    w("")
    for acc in ACC.values():
        population_block(w, acc, partial or not R["completed"])
        w("")
    if "peak_rss" in R:
        w("MEMORY   peak RSS %.1f MB (%.2f GB)   min system available %.1f MB (%.2f GB)   (%d samples at %.1f s)" % (
            R["peak_rss"] / MB, R["peak_rss"] / GB, R["min_avail"] / MB, R["min_avail"] / GB, R["n_samples"], SAMPLE_S))
        w("")
    w("VERDICT: %s" % R.get("verdict", "no verdict"))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    say("\n".join(L))


def chunk_arrays(d, hpi):
    """Everything compared for one quarter, as full-length arrays."""
    ltv32 = d["ltv"].to_numpy(dtype=np.float32, na_value=np.nan)
    rt32 = d["hpi_ratio"].to_numpy(dtype=np.float32, na_value=np.nan)
    st32 = d["est_ltv"].to_numpy(dtype=np.float32, na_value=np.nan)
    valid = ~(np.isnan(ltv32) | np.isnan(rt32) | np.isnan(st32)) & (rt32 > 0)
    age = d["loan_age"].to_numpy(dtype=np.int64, na_value=0)
    fp = d["from_period"].to_numpy(dtype=np.int64, na_value=0)
    t_idx = (fp // 100) * 12 + (fp % 100) - 1
    st64 = st32.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec64 = ltv32.astype(np.float64) / rt32.astype(np.float64)
    cast32 = rec64.astype(np.float32)
    ps = d["property_state"]
    cats = [str(c) for c in ps.cat.categories]
    codes = ps.cat.codes.to_numpy().astype(np.int64)
    amap = np.array([hpi.index.get(c, -1) for c in cats] + [-1])
    sidx = amap[np.where(codes < 0, len(cats), codes)]
    t, o = t_idx - hpi.m0, t_idx - age - hpi.m0
    or_ok = valid & (sidx >= 0) & (t >= 0) & (t < hpi.width) & (o >= 0) & (o < hpi.width)
    si, ti, oi = np.where(or_ok, sidx, 0), np.where(or_ok, t, 0), np.where(or_ok, o, 0)
    ratio_o = hpi.hpi_disagg[si, ti] / hpi.hpi_disagg[si, oi]
    est_o = (ltv32.astype(np.float64) / ratio_o).astype(np.float32)
    state_str = np.array(cats + ["<null>"], dtype=object)[np.where(codes < 0, len(cats), codes)]
    return {"valid": valid, "ltv": ltv32, "ratio": rt32, "st64": st64, "rec64": rec64, "raw": rec64 - st64,
            "castd": cast32.astype(np.float64) - st64, "spacing": np.spacing(st32).astype(np.float64),
            "bits_equal": cast32.view(np.uint32) == st32.view(np.uint32), "or_ok": or_ok,
            "or_est_bad": est_o != st32, "or_ratio_bad": ratio_o.astype(np.float32) != rt32,
            "state_str": state_str, "fp": fp, "orig_year": (t_idx - age) // 12}


def main():
    manifest = panel_io.read_manifest()
    R["build_stamp"], R["total_rows"] = manifest["build_stamp"], manifest["total_rows"]
    parts = manifest["partitions"]
    hpi = panel.Hpi()
    a_cond = Acc("(a)  condition rows at from_period == %d  (diagnostic 35's population)" % CUT)
    a_all = Acc("(b)  all panel rows, quarter by quarter")
    ACC["a"], ACC["b"] = a_cond, a_all
    for acc in (a_cond, a_all):
        acc.top_ids = {}
    sampler = Sampler()
    sampler.start()

    R["phase"] = "scan"
    for qi, p in enumerate(parts):
        q = p["quarter"]
        d = panel_io.load_panel(columns=COLS, quarters=[q])
        for base in range(0, len(d), SLICE):                  # slices keep the float64 temporaries small
            v = chunk_arrays(d.iloc[base:base + SLICE], hpi)
            cond = ((d["role"].iloc[base:base + SLICE] == "condition").to_numpy() & (v["fp"] == CUT))
            a_cond.n_null += int((~v["valid"] & cond).sum())
            a_all.n_null += int((~v["valid"]).sum())
            a_cond.add(v, v["valid"] & cond, qi, base)
            a_all.add(v, v["valid"], qi, base)
            del v, cond
        say("  %s: %s rows" % (q, f(len(d))))
        del d
        gc.collect()

    R["phase"] = "loan ids for the largest cases"
    for acc in (a_cond, a_all):
        want = {}
        for t in acc.top:
            want.setdefault(t[1], []).append(t[2])
        for qi, rows in want.items():
            col = pq.read_table(os.path.join(panel.PANEL_DIR, parts[qi]["file"]), columns=["loan_id"]).column(0)
            ids = col.take(rows).to_pylist()
            acc.top_ids.update({(qi, r): i for r, i in zip(rows, ids)})

    sampler.stop_flag.set()
    sampler.join()
    sampler.fill()
    R["completed"] = True
    R["phase"] = "finish"

    def judge(acc):
        acc.raw.finish()
        acc.cast.finish()
    for acc in (a_cond, a_all):
        judge(acc)
    # the verdict is read from population (b); (a) is a subset of it
    a = a_all
    cast_ulp_ok = a.ulp[3] == 0 and a.ulp_max <= 2.0
    if a.or_est == 0 and a.or_ratio == 0 and cast_ulp_ok:
        R["verdict"] = ("FLOAT32 ROUNDING. The writer's own arithmetic (unrounded HPI ratio) reproduces the stored "
                        "est_ltv and hpi_ratio on all %s recomputable rows; the gap to a recompute from the stored "
                        "float32 hpi_ratio is at most %.2f float32 ulp, %s rows over 0.5 LTV points."
                        % (f(a.or_n), a.ulp_max, f(a.raw.over05)))
    elif a.or_est == 0 and a.or_ratio == 0:
        R["verdict"] = ("The writer's arithmetic reproduces the stored columns exactly, so the stored est_ltv is "
                        "ltv / the HPI ratio; but recomputing from the stored float32 hpi_ratio moves it by up to "
                        "%.2f ulp and %s rows by more than 0.5 LTV points: float32 storage of hpi_ratio is "
                        "the cause, larger than a 1-2 ulp effect (see the breakdown)." % (a.ulp_max, f(a.raw.over05)))
    elif a.or_ratio:
        R["verdict"] = ("REAL DIVERGENCE. stored hpi_ratio != float32(HPI ratio from the table) on %s rows: the evidence "
                        "points at the RATIO step (state / from_month / orig_month lookup), not the division." % f(a.or_ratio))
    else:
        R["verdict"] = ("REAL DIVERGENCE. stored hpi_ratio matches the table but est_ltv != (ltv / ratio) cast to float32 "
                        "on %s rows: the evidence points at the DIVISION / CAST step." % f(a.or_est))
    write_report()


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        R["abort_reason"] = "MemoryError during '%s'" % R["phase"]
        R["verdict"] = "DID NOT COMPLETE - MemoryError."
        write_report(partial=True)
        raise
