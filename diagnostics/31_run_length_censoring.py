"""run_length_censored: what is flagged, why, how long it lasts, and whether it matters.   READ-ONLY.

    python diagnostics/31_run_length_censoring.py --scratch DIR     # from the repo root

DIR is a scratch rebuild made with  python -m src.panel --out-dir DIR . Writes
diagnostics/31_run_length_censoring.txt and nothing else. It reads DIR, the committed
panel (data/panel/) and the pre-HPI panel kept in data/panel/_pre_hpi/, changes none of
them, and runs no git command.

The flag is re-derived and re-verified here from the WRITTEN panel columns, not taken
from the build's counters; the build report's counts are then compared with it.

Cause split, derived from the panel alone: a flagged run can only begin at a loan's
first observed row (entry) or at a post_gap row (gap), so a flagged row is GAP-caused
when a post_gap row has occurred in its loan at or before it, and ENTRY-caused when not.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from src import horizon as hz, panel, panel_io                            # noqa: E402
from src.horizon_nulls import fingerprint                                 # noqa: E402

OUT = os.path.join("diagnostics", "31_run_length_censoring.txt")
OLD_FP = "71d85ff7fd2c5d1a"
STATES = panel.STATES
TRANSIENT = STATES[:4]
RULE_W = 118
LINES = []
COLS = ["loan_id", "from_period", "from_state", "to_state", "run_length", "run_length_censored", "post_gap",
        "role", "loan_age", "property_state"]


def emit(s=""):
    print(s, flush=True)
    LINES.append(s)


def rule(c="-"):
    emit(c * RULE_W)


def title(t):
    emit()
    rule("=")
    emit(t)
    rule("=")


def fmt(n):
    return f"{int(n):,}"


def pc(a, b, dp=4):
    return "n/a" if not b else f"{100.0 * a / b:.{dp}f}%"


def dist(x):
    x = np.asarray(x)
    if not len(x):
        return None
    return [int(x.min()), float(np.median(x)), float(np.percentile(x, 90)), float(np.percentile(x, 99)), int(x.max())]


def dist_line(label, x):
    d = dist(x)
    emit(f"  {label:<46}{len(x):>12,}{d[0]:>8}{d[1]:>9.1f}{d[2]:>8.1f}{d[3]:>8.1f}{d[4]:>9}")


def scan(root, quarters):
    """One pass over the scratch panel; everything accumulated, nothing materialised."""
    A = {"rows": 0, "flag": 0, "entry": 0, "gap": 0, "loans": 0, "entry_loans": 0, "gap_loans": 0, "both": 0,
         "viol_rise": 0, "viol_first": 0, "viol_pg": 0, "viol_null": 0,
         "per_loan": [], "per_loan_entry": [], "per_loan_gap": [],
         "role": np.zeros((3, 2)), "from_state": np.zeros((6, 2)), "prop": None,
         "rung": np.zeros((12, 4)),                       # total, flagged, entry, gap  by run_length 1..10 (0, 11+ spill)
         "trans": np.zeros((2, 2, 6, 6)),                 # [all/fit][unflagged/flagged][from][to] at run_length 1
         "fit_loans": 0, "s24": 0, "mine": 0, "both24": 0, "only24": 0, "only_mine": 0,
         "only24_postgap": 0, "only24_other": 0, "only_mine_other": 0,
         "first_fit_postgap": 0, "first_age_hist": {}, "loans_postgap_first": 0,
         "pop24": np.zeros((12, 2))}                      # diagnostic 24/25's population by rung: rows, flagged
    A["prop"] = np.zeros((len(panel.PROPERTY_STATES), 2))
    for q in quarters:
        d = load_panel(q)
        n = len(d)
        lid = pd.factorize(d["loan_id"].to_numpy())[0]
        first = np.ones(n, dtype=bool)
        first[1:] = lid[1:] != lid[:-1]
        flag = d["run_length_censored"].to_numpy()
        pg = d["post_gap"].to_numpy()
        age = d["loan_age"].to_numpy().astype(np.int64)
        rl = d["run_length"].to_numpy().astype(np.int64)
        role = d["role"].cat.codes.to_numpy()
        fs = d["from_state"].cat.codes.to_numpy()
        ts = d["to_state"].cat.codes.to_numpy()
        ps = d["property_state"].cat.codes.to_numpy()
        # remap the categorical codes to the fixed orders
        fs = np.array([STATES.index(c) for c in d["from_state"].cat.categories])[fs]
        ts = np.array([STATES.index(c) for c in d["to_state"].cat.categories])[ts]
        role = np.array([panel.ROLES.index(c) for c in d["role"].cat.categories])[role]
        if d["run_length_censored"].dtype != bool or d["run_length_censored"].isna().any():
            A["viol_null"] += 1
        # independent re-verification from the written columns
        same = ~first
        A["viol_rise"] += int((same & flag & ~np.r_[False, flag[:-1]] & ~pg).sum())
        fr = np.nonzero(first)[0]
        ok_first = np.where(pg[fr], flag[fr], flag[fr] == (age[fr] > 1))
        A["viol_first"] += int((~ok_first).sum())
        A["viol_pg"] += int((pg & ~flag).sum())
        # cause
        cs = np.cumsum(pg)
        cum = cs - (cs[fr] - pg[fr])[lid]
        entry = flag & (cum == 0)
        gap = flag & ~entry
        nl = len(fr)
        pl = np.bincount(lid, weights=flag, minlength=nl)
        pe = np.bincount(lid, weights=entry, minlength=nl)
        pgp = np.bincount(lid, weights=gap, minlength=nl)
        A["rows"] += n
        A["flag"] += int(flag.sum())
        A["entry"] += int(entry.sum())
        A["gap"] += int(gap.sum())
        A["loans"] += nl
        A["entry_loans"] += int((pe > 0).sum())
        A["gap_loans"] += int((pgp > 0).sum())
        A["both"] += int(((pe > 0) & (pgp > 0)).sum())
        A["per_loan"].append(pl[pl > 0].astype(np.int32))
        A["per_loan_entry"].append(pe[pe > 0].astype(np.int32))
        A["per_loan_gap"].append(pgp[pgp > 0].astype(np.int32))
        np.add.at(A["role"], (role, flag.astype(int)), 1)
        np.add.at(A["from_state"], (fs, flag.astype(int)), 1)
        np.add.at(A["prop"], (ps, flag.astype(int)), 1)
        r = np.clip(rl, 0, 11)
        A["rung"][:, 0] += np.bincount(r, minlength=12)
        A["rung"][:, 1] += np.bincount(r, weights=flag, minlength=12)
        A["rung"][:, 2] += np.bincount(r, weights=entry, minlength=12)
        A["rung"][:, 3] += np.bincount(r, weights=gap, minlength=12)
        m1 = rl == 1
        for g, mask in ((0, m1), (1, m1 & (role == 0))):
            np.add.at(A["trans"][g], (flag[mask].astype(int), fs[mask], ts[mask]), 1)
        # diagnostic 24's seasoned entrants: first FIT row, loan_age > 1
        fit = role == 0
        if fit.any():
            li = lid[fit]
            ff = np.ones(len(li), dtype=bool)
            ff[1:] = li[1:] != li[:-1]
            age_f, fl_f, pg_f = age[fit][ff], flag[fit][ff], pg[fit][ff]
            s24 = age_f > 1
            mine = fl_f & ~pg_f                       # flagged at entry: first row flagged and not a post-gap row
            A["fit_loans"] += len(s24)
            A["s24"] += int(s24.sum())
            A["mine"] += int(mine.sum())
            A["both24"] += int((s24 & mine).sum())
            A["only24"] += int((s24 & ~mine).sum())
            A["only_mine"] += int((~s24 & mine).sum())
            A["only24_postgap"] += int((s24 & ~mine & pg_f).sum())
            A["only24_other"] += int((s24 & ~mine & ~pg_f).sum())
            A["only_mine_other"] += int((~s24 & mine).sum())
            # their population: fit rows, loans not seasoned, post_gap rows removed
            seasoned_row = np.repeat(s24, np.diff(np.r_[np.nonzero(ff)[0], len(li)]))
            keep = ~seasoned_row & ~pg[fit]
            rf = np.clip(rl[fit][keep], 0, 11)
            A["pop24"][:, 0] += np.bincount(rf, minlength=12)
            A["pop24"][:, 1] += np.bincount(rf, weights=flag[fit][keep], minlength=12)
        print(f"  scanned {q}", flush=True)
    return A


def load_panel(q):
    return panel_io.load_panel(quarters=[q], columns=COLS)


def old_seasoned24():
    """Diagnostic 24's population, recomputed on the pre-HPI panel (the one the 154,635 came from)."""
    saved = panel_io.PANEL_DIR
    panel_io.PANEL_DIR = os.path.join("data", "panel", "_pre_hpi")
    out = {"fit_loans": 0, "seasoned": 0, "terr": 0, "nullltv": 0, "both": 0}
    try:
        man = panel_io.read_manifest()
        for p in man["partitions"]:
            d = panel_io.load_panel(quarters=[p["quarter"]], columns=["loan_id", "loan_age", "role", "property_state", "ltv"])
            lid = pd.factorize(d["loan_id"].to_numpy())[0]
            nl = lid.max() + 1
            terr = np.zeros(nl, dtype=bool)
            nulls = np.zeros(nl, dtype=bool)
            terr[lid[d["property_state"].isin(["GU", "PR", "VI"]).to_numpy()]] = True
            nulls[lid[d["ltv"].isna().to_numpy()]] = True
            f = d["role"].to_numpy() == "fit"
            li = lid[f]
            ff = np.ones(len(li), dtype=bool)
            ff[1:] = li[1:] != li[:-1]
            sea = d["loan_age"].to_numpy()[f][ff] > 1
            ids = li[ff]
            out["fit_loans"] += len(ids)
            out["seasoned"] += int(sea.sum())
            out["terr"] += int((sea & terr[ids]).sum())
            out["nullltv"] += int((sea & nulls[ids] & ~terr[ids]).sum())
    finally:
        panel_io.PANEL_DIR = saved
    return out


def main(scratch):
    saved = panel_io.PANEL_DIR
    panel_io.PANEL_DIR = scratch
    man = panel_io.read_manifest()
    panel_io.validate(man)
    quarters = [p["quarter"] for p in man["partitions"]]
    build_report = open(os.path.join(scratch, "build_report.txt"), encoding="utf-8").read()
    emit("run_length_censored: left-censoring of run_length")
    emit(f"scratch rebuild: {scratch}   build {man['build_stamp']}   {fmt(man['total_rows'])} rows   {len(quarters)} partitions")
    A = scan(scratch, quarters)

    # ---------------------------------------------------------------- 0 checks
    title("0. ASSERTIONS, RE-VERIFIED FROM THE WRITTEN PANEL  (the build asserts them per quarter over every loan's full history)")
    emit(f"first row at origination (loan_age <= 1, not post_gap) is unflagged; entering later is flagged:  violations {A['viol_first']}")
    emit(f"flag false then true with no gap in between (rise not on a post_gap row):                       violations {A['viol_rise']}")
    emit(f"post_gap rows that are not flagged:                                                              violations {A['viol_pg']}")
    emit(f"null or non-boolean flag column:                                                                 violations {A['viol_null']}")
    emit("(the within-run constancy and the 'every row after a gap is flagged' checks need the performance history and run in the build.)")
    assert A["viol_first"] == A["viol_rise"] == A["viol_pg"] == A["viol_null"] == 0

    # ---------------------------------------------------------------- 1 counts
    title("1. ROWS AND LOANS FLAGGED, BY CAUSE")
    rows, loans = A["rows"], A["loans"]
    emit(f"{'':<40}{'rows':>14}{'of rows':>12}{'loans':>14}{'of loans':>12}")
    emit(f"{'panel':<40}{fmt(rows):>14}{'':>12}{fmt(loans):>14}")
    emit(f"{'flagged, any cause':<40}{fmt(A['flag']):>14}{pc(A['flag'], rows):>12}{fmt(A['entry_loans'] + A['gap_loans'] - A['both']):>14}"
         f"{pc(A['entry_loans'] + A['gap_loans'] - A['both'], loans):>12}")
    emit(f"{'  cause: entry (first row after origination)':<40}{fmt(A['entry']):>14}{pc(A['entry'], rows):>12}{fmt(A['entry_loans']):>14}"
         f"{pc(A['entry_loans'], loans):>12}")
    emit(f"{'  cause: gap (row after an interior gap)':<40}{fmt(A['gap']):>14}{pc(A['gap'], rows):>12}{fmt(A['gap_loans']):>14}"
         f"{pc(A['gap_loans'], loans):>12}")
    emit(f"loans with both causes: {fmt(A['both'])}.   rows by cause add to flagged rows: {A['entry'] + A['gap'] == A['flag']}")
    # against the build's own counters
    def grab(label):
        for ln in build_report.splitlines():
            if ln.startswith(label):
                return [int(x.replace(",", "")) for x in ln[len(label):].split() if re.fullmatch(r"[\d,]+", x)]
        raise SystemExit(f"cannot find {label!r} in the build report")
    b_any, b_entry, b_gap = grab("flagged, any cause"), grab("  cause: entry"), grab("  cause: gap")
    emit()
    emit("Against the build's own counters (computed from the performance history, per quarter):")
    for name, mine, theirs in (("rows, any cause", A["flag"], b_any[0]), ("rows, entry", A["entry"], b_entry[0]),
                               ("rows, gap", A["gap"], b_gap[0]), ("loans, entry", A["entry_loans"], b_entry[1]),
                               ("loans, gap", A["gap_loans"], b_gap[1])):
        emit(f"  {name:<18}derived here {fmt(mine):>12}   build report {fmt(theirs):>12}   {'MATCH' if mine == theirs else 'DIFFERENT'}")
        assert mine == theirs, name
    emit("(post_gap rows, which the flag leaves untouched: "
         f"{fmt(int(sum(pq.read_table(os.path.join(scratch, p['file']), columns=['post_gap']).column('post_gap').to_pandas().sum() for p in man['partitions'])))})")

    # ---------------------------------------------------------------- 2 diag 24
    title("2. ENTRY-FLAGGED LOANS vs DIAGNOSTIC 24's SEASONED ENTRANTS")
    emit("Diagnostic 24's rule: a FIT loan is seasoned when loan_age on its first FIT row is > 1. On record: 154,635 loans of 1,939,309 fit loans.")
    old = old_seasoned24()
    emit()
    emit(f"Recomputed on the pre-HPI panel (data/panel/_pre_hpi; the panel diagnostic 24 ran on): {fmt(old['seasoned'])} seasoned of "
         f"{fmt(old['fit_loans'])} fit loans   (on record 154,635 of 1,939,309: {'AGREE' if (old['seasoned'], old['fit_loans']) == (154635, 1939309) else 'DISAGREE'})")
    emit(f"   of those, now removed from the panel: territory {fmt(old['terr'])}, null-ltv (not territory) {fmt(old['nullltv'])}")
    exp = old["seasoned"] - old["terr"] - old["nullltv"]
    emit(f"   expected on the current panel: {fmt(old['seasoned'])} - {fmt(old['terr'])} - {fmt(old['nullltv'])} = {fmt(exp)}")
    emit()
    emit(f"Current panel, fit loans {fmt(A['fit_loans'])}:")
    emit(f"   diagnostic 24's condition (first fit row loan_age > 1):      {fmt(A['s24'])}   "
         f"({'matches the expected count' if A['s24'] == exp else 'DIFFERS from the expected ' + fmt(exp) + ' by ' + fmt(A['s24'] - exp)})")
    emit(f"   entry-flagged here (first row flagged, not post_gap):        {fmt(A['mine'])}")
    emit(f"   both: {fmt(A['both24'])}    only diagnostic 24: {fmt(A['only24'])}    only the flag: {fmt(A['only_mine'])}")
    if A["only24"] or A["only_mine"]:
        emit(f"   DISAGREEMENT. Of the {fmt(A['only24'])} seasoned loans the flag does not call entry-censored: "
             f"{fmt(A['only24_postgap'])} have a post_gap first fit row (their first")
        emit("   observed row was not paired, so the panel's first row follows a gap and is gap-flagged, not entry-flagged); "
             f"{fmt(A['only24_other'])} are other.")
        emit(f"   Flag-only loans: {fmt(A['only_mine'])}.")
    else:
        emit("   AGREEMENT: the two sets are identical loan for loan.")
    emit("   All flagged-at-entry loans, any role: " + fmt(A["entry_loans"]) + " (includes test-split loans, which diagnostic 24 never saw).")

    # ---------------------------------------------------------------- 3 duration
    title("3. HOW LONG THE CENSORING PERSISTS  (flagged panel rows per affected loan)")
    emit(f"  {'':<46}{'loans':>12}{'min':>8}{'median':>9}{'p90':>8}{'p99':>8}{'max':>9}")
    dist_line("loans with any flagged row", np.concatenate(A["per_loan"]))
    dist_line("  entry-caused rows, loans with any", np.concatenate(A["per_loan_entry"]))
    dist_line("  gap-caused rows, loans with any", np.concatenate(A["per_loan_gap"]))
    emit("(rows are panel rows, one per loan-month; a loan's flagged rows are consecutive within each censored run.)")

    # ---------------------------------------------------------------- 4 role
    title("4. FLAGGED SHARE BY ROLE")
    emit(f"{'role':<12}{'rows':>14}{'flagged':>14}{'share':>12}")
    for i, r in enumerate(panel.ROLES):
        t = A["role"][i].sum()
        emit(f"{r:<12}{fmt(t):>14}{fmt(A['role'][i, 1]):>14}{pc(A['role'][i, 1], t):>12}")
    emit(f"{'ALL':<12}{fmt(rows):>14}{fmt(A['flag']):>14}{pc(A['flag'], rows):>12}")

    # ---------------------------------------------------------------- 5 state / rung
    title("5. FLAGGED SHARE BY ORIGIN STATE (from_state) AND BY run_length RUNG")
    emit(f"{'from_state':<12}{'rows':>14}{'flagged':>14}{'share':>12}")
    for i, s in enumerate(STATES):
        t = A["from_state"][i].sum()
        if t:
            emit(f"{s:<12}{fmt(t):>14}{fmt(A['from_state'][i, 1]):>14}{pc(A['from_state'][i, 1], t):>12}")
    emit()
    emit("By run_length rung (all roles). Of the rows at each rung, how many are censored rather than genuinely fresh:")
    emit(f"{'rung':>5}{'rows':>14}{'flagged':>14}{'share':>10}{'entry':>14}{'gap':>12}{'unflagged (genuine)':>22}")
    for r in range(1, 11):
        t, f, e, g = A["rung"][r]
        emit(f"{r:>5}{fmt(t):>14}{fmt(f):>14}{pc(f, t, 2):>10}{fmt(e):>14}{fmt(g):>12}{fmt(t - f):>22}")
    emit()
    emit("By property_state (extra):")
    emit(f"{'state':<8}{'rows':>14}{'flagged':>12}{'share':>9}   " * 2)
    ps = [(panel.PROPERTY_STATES[i], A["prop"][i].sum(), A["prop"][i, 1]) for i in range(len(panel.PROPERTY_STATES))
          if A["prop"][i].sum()]
    for k in range(0, len(ps), 2):
        emit("   ".join(f"{s:<8}{fmt(t):>14}{fmt(f):>12}{pc(f, t, 2):>9}" for s, t, f in ps[k:k + 2]))

    # ---------------------------------------------------------------- 6 rung 1
    title("6. ONE-STEP TRANSITION RATES AT run_length = 1, FLAGGED vs UNFLAGGED, PER ORIGIN STATE")
    emit("rate = share of rows at rung 1 whose to_state is that state. exit = 1 - stay. z = two-proportion z on the exit rate.")
    emit("Flagged rung-1 rows are the first row of a seasoned entrant and the post_gap rows; unflagged rung-1 rows start at an observed state change.")
    for g, name in ((1, "FIT rows (diagnostic 24 / 25's population before their exclusions)"), (0, "ALL roles")):
        emit()
        emit(name)
        emit(f"{'from':<9}{'flag':<11}{'rows':>12}" + "".join(f"{s:>10}" for s in STATES) + f"{'exit':>10}{'diff pp':>10}{'z':>8}")
        for i, s in enumerate(TRANSIENT):
            cu, cf = A["trans"][g][0][i], A["trans"][g][1][i]
            nu, nf = cu.sum(), cf.sum()
            eu = 1 - cu[i] / nu if nu else float("nan")
            ef = 1 - cf[i] / nf if nf else float("nan")
            for lab, c, nn, ex in (("unflagged", cu, nu, eu), ("flagged", cf, nf, ef)):
                if not nn:
                    emit(f"{s:<9}{lab:<11}{0:>12}   (no rows)")
                    continue
                tail = ""
                if lab == "flagged" and nu and nf:
                    pp = (nu * eu + nf * ef) / (nu + nf)
                    z = (ef - eu) / np.sqrt(pp * (1 - pp) * (1 / nu + 1 / nf)) if 0 < pp < 1 else float("nan")
                    tail = f"{100 * (ef - eu):>+10.3f}{z:>8.1f}"
                emit(f"{s:<9}{lab:<11}{fmt(nn):>12}" + "".join(f"{100 * c[k] / nn:>9.3f}%" for k in range(6)) + f"{100 * ex:>9.3f}%{tail}")
    # contamination of diagnostic 25's rung-1 numbers
    emit()
    emit()
    emit("What this means for diagnostic 24 / 25's rung-1 numbers. They excluded the seasoned loans entirely and every post_gap row; those are")
    emit("exactly the flagged rung-1 rows (an entry run starts at a first row, a gap run at a post_gap row). Flagged rows inside THEIR population")
    emit("(fit rows, loans not seasoned, post_gap rows removed), by rung:")
    emit(f"{'rung':>5}{'rows':>14}{'flagged':>12}")
    for r in range(1, 11):
        emit(f"{r:>5}{fmt(A['pop24'][r, 0]):>14}{fmt(A['pop24'][r, 1]):>12}")
    emit(f"  all rungs: {fmt(A['pop24'][:, 0].sum())} rows, {fmt(A['pop24'][:, 1].sum())} flagged "
         f"({pc(A['pop24'][:, 1].sum(), A['pop24'][:, 0].sum(), 5)}); at rung 1: {fmt(A['pop24'][1, 1])}")
    emit("So diagnostic 24 / 25's rung-1 numbers are NOT contaminated by censored rung-1 rows: their population contains none. The flag does")
    emit("carry real information (section 6: censored Current rung-1 rows exit far less often), which is why excluding the seasoned loans, as")
    emit("they did, mattered; a model that ignores the flag would mix the two populations at rung 1.")

    # ---------------------------------------------------------------- 7 identity, vs committed, fingerprint
    title("7. REBUILD: IDENTITY, SCHEMA AND FINGERPRINT")
    for ln in build_report.splitlines():
        if "identity holds" in ln or "== written" in ln:
            emit("  " + ln.strip())
    ident = [ln for ln in build_report.splitlines() if "identity holds" in ln or "== written" in ln]
    assert ident and all(ln.rstrip().endswith("yes") for ln in ident), "identity does not close"
    emit("  arithmetic identity closes: yes")
    types = {c["name"]: c["type"] for c in man["schema"]}
    assert list(types) == panel.SCHEMA
    emit(f"  scratch schema equals src.panel.SCHEMA ({len(types)} columns); run_length_censored is a feature: "
         f"{'run_length_censored' in panel.FEATURES and 'run_length_censored' not in panel.METADATA}; type {types['run_length_censored']}")
    emit(f"  position: {list(types).index('run_length_censored')} (immediately after run_length: {list(types).index('run_length') + 1 == list(types).index('run_length_censored')})")
    panel_io.PANEL_DIR = saved
    cman = panel_io.read_manifest()
    cols_c = [c["name"] for c in cman["schema"]]
    emit(f"  committed panel (data/panel/) has {len(cols_c)} columns; scratch = committed + {sorted(set(types) - set(cols_c))}; removed {sorted(set(cols_c) - set(types))}")
    same_all = True
    for p in man["partitions"]:
        tc = pq.read_table(os.path.join(saved, p["file"]))
        ts_ = pq.read_table(os.path.join(scratch, p["file"]))
        ok = ts_.drop_columns(["run_length_censored"]).equals(tc)
        same_all &= ok
        emit(f"    {p['quarter']}  rows {fmt(ts_.num_rows)} vs {fmt(tc.num_rows)};  every committed column identical in the scratch panel: {ok}")
        del tc, ts_
    emit(f"  all committed columns identical, value for value: {same_all}")
    # fingerprint on the scratch panel
    panel_io.PANEL_DIR = scratch
    try:
        L, _ = hz.read_test_loans(progress=lambda s: None)
        E = L[hz.eligibility(L).to_numpy() < 0]
        real = hz.realised(E)
        _, fixed = hz.cohorts(real)
        fp = fingerprint(E.index.to_numpy(), [E["state_cut"].to_numpy()] + [real[h] for h in hz.HORIZONS] + [fixed])
    finally:
        panel_io.PANEL_DIR = saved
    emit(f"  harness fingerprint of the scratch panel: {fp}   ({len(E):,} eligible loans)   committed / saved baseline: {OLD_FP}   "
         f"{'UNCHANGED' if fp == OLD_FP else 'CHANGED'}")
    emit()
    emit("GATES NEEDING RE-BASELINING:")
    if fp == OLD_FP:
        emit("  Fingerprint gates: NONE. The fingerprint covers eligible loan ids, start states, realised outcomes and cohorts: all")
        emit("  state-and-period columns. The new column is a feature and the rows and states are identical, so it cannot move.")
        emit("  diagnostics/23 WANT_FP and artifacts/horizon_nulls/nulls_report.txt stay at " + OLD_FP + ".")
    else:
        emit("  The fingerprint CHANGED: re-baseline diagnostics/23_origination_validity.py WANT_FP and the baseline line of "
             "artifacts/horizon_nulls/nulls_report.txt to " + fp + ".")
    emit("  Other things keyed to the panel's SHAPE rather than its fingerprint: the committed data/panel/ does not yet have the column")
    emit("  (it was NOT replaced; the rebuild went to the scratch directory), so to use the flag the scratch panel must be promoted;")
    emit("  nothing is stale until then. After promotion, the panel's build stamp changes (diagnostics print it; nothing gates on it).")
    emit("  Unaffected: artifacts/baseline/counts.csv and the horizon runs read from_state/to_state/role/from_period only.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    main(ap.parse_args().scratch)
