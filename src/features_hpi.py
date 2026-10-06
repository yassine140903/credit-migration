"""Read-only check of the HPI features on the committed panel.

    python -m src.features_hpi verify                  # prints, changes nothing
    python -m src.features_hpi verify --splice-report  # also rewrites the step 2 block of
                                                       # diagnostics/29_hpi_join_report.txt

The HPI join itself, the territory and null-ltv exclusions, and the est_ltv
features live in src/panel.py's build; this module no longer builds anything. What
remains is the independent re-check: for every partition, the origination month
read from the ORIGINATION RECORD must equal from_period - loan_age on every row,
and hpi_ratio, est_ltv and est_ltv_step recomputed from scratch (that record
month, data/macro/hpi_state_monthly.csv) must equal the stored columns exactly
(float32). Any mismatch, or an origination file not on disk, is a hard failure.

Origination month = first payment month minus one, the panel's own convention
(Freddie's origination file carries no origination date).
"""

import argparse
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import panel
from src.panel_io import PanelError, read_manifest, validate

REPORT = os.path.join("diagnostics", "29_hpi_join_report.txt")


class JoinError(RuntimeError):
    pass


def fmt(n):
    return f"{int(n):,}"


def mi_of_yyyymm(x):
    x = np.asarray(x, dtype=np.int64)
    return (x // 100) * 12 + (x % 100) - 1


def locate_orig(quarter):
    """('file', path) or ('zip', zip path, member) for the quarter's origination
    file, or None if neither is on disk."""
    name = f"orig_{quarter}.txt"
    hits = glob.glob(os.path.join("data", "historical_data_*", "*", name))
    if hits:
        return ("file", hits[0])
    for z in sorted(glob.glob(os.path.join("data", "historical_data_*", f"*{quarter}.zip"))):
        with zipfile.ZipFile(z) as zf:
            if name in zf.namelist():
                return ("zip", z, name)
    return None


def read_orig_months(loc, oidx):
    """Series: loan_id -> origination month index (first payment month minus one)."""
    li, fi = oidx["loan"][0], oidx["first_payment"][0]
    kw = dict(sep="|", header=None, dtype=str, usecols=[li, fi], engine="c", na_filter=False)
    if loc[0] == "file":
        frame = pd.read_csv(loc[1], **kw)
    else:
        with zipfile.ZipFile(loc[1]) as zf, zf.open(loc[2]) as fh:
            frame = pd.read_csv(fh, **kw)
    loan = frame[li].str.strip()
    fpd = pd.to_numeric(frame[fi].str.strip(), errors="coerce").to_numpy()
    year, month = fpd // 100, fpd % 100
    ok = np.isfinite(fpd) & (month >= 1) & (month <= 12)
    om = np.where(ok, year * 12 + (month - 1) - 1, -1).astype(np.int64)
    s = pd.Series(om, index=loan.to_numpy())
    if not s.index.is_unique:
        raise JoinError(f"{loc}: duplicate loan ids in the origination file")
    return s


def verify(splice_report=False):
    manifest = read_manifest()
    validate(manifest)
    hpi = panel.Hpi()
    _, oidx, _, _ = panel.resolve()
    src_lines, tot_rows, bad = [], 0, 0
    for p in manifest["partitions"]:
        q = p["quarter"]
        loc = locate_orig(q)
        if loc is None:
            raise JoinError(f"{q}: origination record not on disk")
        t = pq.read_table(os.path.join(panel.PANEL_DIR, p["file"]),
                          columns=["loan_id", "from_period", "loan_age", "property_state", "ltv",
                                   "hpi_ratio", "est_ltv", "est_ltv_step"])
        enc = t.column("loan_id").combine_chunks().dictionary_encode()
        lid, loans = enc.indices.to_numpy(zero_copy_only=False), enc.dictionary.to_pylist()
        rec = read_orig_months(loc, oidx).reindex(loans).to_numpy()
        if np.isnan(rec.astype(float)).any() or (rec < 0).any():
            raise JoinError(f"{q}: missing or malformed origination record for a panel loan")
        om = rec.astype(np.int64)[lid]
        t_idx = mi_of_yyyymm(t.column("from_period").to_numpy())
        n_mis = int((om != t_idx - t.column("loan_age").to_numpy().astype(np.int64)).sum())
        ps = t.column("property_state").combine_chunks()
        cats = ps.dictionary.to_pylist()
        s_idx = np.array([hpi.index.get(c, -1) for c in cats])[
            ps.indices.to_numpy(zero_copy_only=False).astype(np.int64)]
        if (s_idx < 0).any() or (om - hpi.m0 < 0).any() or (t_idx - hpi.m0 >= hpi.width).any():
            raise JoinError(f"{q}: state or month outside HPI coverage")
        ltv = t.column("ltv").to_pandas().to_numpy(dtype=np.float64, na_value=np.nan)
        r = hpi.hpi_disagg[s_idx, t_idx - hpi.m0] / hpi.hpi_disagg[s_idx, om - hpi.m0]
        rs = hpi.hpi_step[s_idx, t_idx - hpi.m0] / hpi.hpi_step[s_idx, om - hpi.m0]
        n_bad = 0
        for col, v in (("hpi_ratio", r), ("est_ltv", ltv / r), ("est_ltv_step", ltv / rs)):
            stored = t.column(col).to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
            n_bad += int((~((stored == v.astype(np.float32)) | (np.isnan(stored) & np.isnan(v)))).sum())
        src_lines.append(f"  {q}  origination record {loc[1] if loc[0] == 'file' else loc[1] + ':' + loc[2]}"
                         f"   rows {fmt(t.num_rows)}   month != from_period-loan_age: {n_mis}   "
                         f"stored feature mismatches: {n_bad}")
        tot_rows += t.num_rows
        bad += n_mis + n_bad
        print(src_lines[-1], flush=True)
    if bad:
        raise JoinError(f"verify: {bad} mismatches")
    print(f"verified: {len(src_lines)} partitions, {fmt(tot_rows)} rows, zero mismatches")
    if not splice_report:
        return
    block = ["Source of each partition's origination month, VERIFIED on the committed panel against the origination "
             "records:"] + src_lines + [
             "",
             f"All {len(src_lines)} partitions, {fmt(tot_rows)} rows: the record-derived origination month equals "
             "from_period - loan_age on every row, and hpi_ratio,",
             "est_ltv and est_ltv_step recomputed from scratch (record month, hpi_state_monthly.csv) equal the stored "
             "columns exactly (float32). Zero mismatches.", ""]
    lines = open(REPORT, encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Source of each partition's origination month"))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("====="))
    lines[i:j] = block
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"spliced into {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["verify"])
    ap.add_argument("--splice-report", action="store_true")
    a = ap.parse_args()
    try:
        verify(a.splice_report)
    except (JoinError, PanelError, panel.HpiError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
