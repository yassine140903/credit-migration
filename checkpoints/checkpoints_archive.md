# Credit Migration: Markov + ML — Full Checkpoint Archive

**Compiled 15 October 2026.** This file is a faithful transcription of every session
checkpoint produced for this project, in date order. Nothing is summarised or
reworded. Numbers are reproduced exactly as recorded at the time.

Workflow diagrams in the originals were TikZ figures. They do not survive as text,
so each one appears below as a plain list of its node labels under the heading it
belonged to, marked *Workflow (diagram nodes)*.

---

# PART 0 — CONFLICTS AND SUPERSESSIONS (interpretation, not transcription)

> **This section is the only part of this file that is not a transcription.**
> It flags places where a later checkpoint overrode an earlier one, so a number or
> decision read out of an early session is not used by mistake. Everything below
> Part 0 is the record as written.

1. **Vintage.** Session 2 analyses the 2019 origination vintage and rejects it.
   All later work is 2006–2007. Any 2019 figure is dead except as the rejection
   argument.

2. **Code 06.** Session 2 (D5) groups "codes 06, 15, 16" as administrative exits.
   Session 4 (D18) establishes code 06 does not exist in the documented
   enumeration. The original brief's grouping was mistaken.

3. **Default codes.** Session 2 (D1) lists zero-balance codes 02, 03, 09, 96, 98 as
   terminal credit events. Sessions 3–4 narrow this: 02/03/09 plus RA are Default;
   96 splits on prior RA (D9, amended by D17); 98 does not survive as a Default code.

4. **State count.** Session 2 leaves the DQ90+ split open. Session of 11 September
   closes it at six states (D20) on documentary grounds — no foreclosure-status
   field exists.

5. **Stratified third arm.** Proposed and adopted 11 September (D23–D26). Cut
   permanently on 23 September under the deadline.

6. **Middle-rung collapse.** Live option as of 23 September pending diagnostic 15;
   killed by diagnostic 15 the same session (D38).

7. **Gap threshold.** A ≤2 / ≥3 month gap threshold was provisionally designed and
   then withdrawn (D37). One uniform rule.

8. **loan_age.** Three separate issues, resolved at different dates: the off-by-one
   between the panel column and diagnostic 22's definition (closed 6 October); the
   modification-reset breakage and the harmful R7 drop rule (closed 29 September);
   and dropping the feature from the headline arm entirely (8 October). Do not mix
   them.

9. **Forecast cost.** The 5 October estimate of 1–2 billion scored rows is dead;
   8 October measured ~99 million. Everything sized against the old figure (cloud
   training, GPU, batching as a memory strategy) was answering a non-problem.

10. **DQ90+ share of the pooled objective.** Diagnostic 39 (11 October) recorded
    0.0033%. Corrected on 14 October to **0.5020%** — about 150× off. The
    "drowning" pre-registration for diagnostic 53 rested on the wrong figure.

11. **Log-loss.** The original design used log-loss plus per-state AUC locally.
    Retired 11 October after diagnostics 42–43; Brier replaces it in both regimes,
    with log-loss reported alongside plus a count of rows hitting the floor.

12. **"No skill" on DQ90+.** Diagnostics 40–41 concluded no skill from argmax.
    Diagnostic 51 found AUC 0.713 by asking about ordering instead. The no-skill
    reading is withdrawn.

13. **Init-score hypothesis.** Diagnostic 45 reported the experiment as having
    failed to apply (plumbing). Diagnostic 46 overturned that reading: it was
    attached correctly and the hypothesis is dead, not untested.

14. **Undertraining diagnosis.** Proposed after diagnostic 44, withdrawn after the
    per-round trace in diagnostic 45/46.

15. **Architecture.** 13 October: pooled ships, grouped favoured in 0 of 500 draws.
    14 October scopes rather than overturns this — both arms were trained **without
    run length**. Architecture is a property of the feature set.

16. **Fingerprints.** `3b1209189824e42f` (5 October panel) was re-baselined to
    `71d85ff7fd2c5d1a` (6 October, final panel). Row counts moved 69,927,577 →
    69,520,712 at the same point.

17. **Engineering layer.** 29 September argued to keep the drift monitor and a
    minimal FastAPI endpoint. Cut entirely on 14 October.

18. **Run-length fork.** The 14 October checkpoint lists chaining run-length as
    "not built, not scheduled". The lattice is ruled out on time; the 13 October
    headline stands.

*Note for the write-up:* several early checkpoints name the advisor directly in
scorecards. The project's standing rule is that checkpoints do not. These are
transcribed as written; the rule applies to new checkpoints, not to this archive.

---

# PART 1 — SESSION CHECKPOINTS, IN ORDER

---

## Session 2 Checkpoint — Data Diagnostics and Vintage Selection

### 1. Outcome of this session

Session 2 was intended to close data wrangling and state labelling. It instead
overturned the project's data choice. The 2019 origination vintage was shown to be
incapable of supporting the research question, and the project moves to the
2006–2007 crisis vintages. All modelling decisions below were reached before that
switch and carry forward unchanged.

### 2. Diagnostics run

Four performance files, `perf_2019Q{1..4}.txt`, 69.6M rows, 7.7 GB, 1,781,144
distinct loans. All reads chunked at 2M rows.

| Quantity | Value |
|---|---|
| Observation cutoff (all vintages) | 2026-03 |
| Terminated loans | 1,251,507 (70.3%) |
| Right-censored at cutoff | 529,637 (29.7%) |
| Prepayments (ZB code 01) | 1,244,045 (99.40% of terminations) |
| Credit-event terminations (02/03/09) | 1,008 (0.06% of loans) |
| Loans ever reaching DQ ≥ 6 | 58,090 (3.26%) |
| of which cured to Current | 55,922 (96.27%) |
| median months to cure | 6.0 |
| first-touch in 2020-03–2021-12 | 84.12% |
| first-touch in 2020-09 alone | 26.42% |

**Interpretation.** The 2020-09 spike is arithmetic, not credit: March 2020 plus six
missed payments. A DQ ≥ 6 indicator on this vintage is close to a dummy for CARES
Act forbearance participation. Only the post-2022 background flow (≈100–250
loans/month, ≈8,800 total) behaves like ordinary credit migration.

### 3. Why 2019 fails

Both candidate default labels fail, for opposite reasons:

- **Terminal codes only (02/03/09):** absorption is genuinely true, but 1,008 events
  cannot support a Default row chained to h = 60.
- **DQ ≥ 6:** abundant, but a 96% cure rate means the label is not measuring
  default, and absorption would have to be imposed by fiat against the data.

No label definition escapes this. The vintage was rescued: strong underwriting,
forbearance absorbing the shock, and ≈40% home price appreciation letting distressed
borrowers sell rather than default.

### 4. Decisions carried forward

**D1 — Default is a terminal credit event, not a delinquency depth.** Zero-balance
codes 02, 03, 09, 96, 98. Rejected the Basel 90+ definition because DQ90+ must remain
transient; making it absorbing destroys observed cure dynamics.

**D2 — Prepaid stays in the chain; report a renormalised credit view.** Chained
distributions at horizon h are reported twice: raw, and with Prepaid mass dropped and
the remainder rescaled. This is a reporting transform on one chain, not a second
model. Documented limitation: prepayment selects on creditworthiness, so the survivor
pool degrades over the horizon.

**D3 — Forbearance is a covariate, not a seventh state.** A seventh state would alter
the state space of both arms and confound the comparison — gains could not be
attributed to borrower-conditioning versus redesign. Forbearance is a reason for
occupying a state, not a position in the delinquency ladder, which is covariate
territory. The baseline is knowingly handicapped during stress periods; results are
reported split by period as well as pooled.

**D4 — Ablations.** Three, isolating which component of conditioning does the work:

1. time-in-state (semi-Markov approximation)
2. forbearance / loss-mitigation flag
3. macro conditioning

The no-flag variant is the one that earns the general claim: if the conditioned model
beats the baseline during stress without an anomaly indicator, the result generalises.
If it only wins with the flag, the finding is narrower and should be stated as such.

**D5 — Administrative exits are censored.** Codes 06, 15, 16 are neither credit events
nor clean prepayments. Loans are censored at exit; transitions up to that point are
counted. Documented limitation: note sales (15) may select on credit quality, biasing
PD downward.

**D6 — Vintage switch to 2006–2007.** Complete panels, resolved foreclosures, no
forbearance distortion, and a period in which the homogeneity assumption visibly
fails. Performance data runs to 2025-09, so a 2007 loan has ≈18 years of history and
h = 60 sits well inside the panel. Cost: loses the COVID stress-test framing and is
better-trodden ground. Optional recovery: train on 2006–2007, evaluate out-of-sample
on 2019 as a regime-shift test.

### 5. State space

S = {Current, DQ30, DQ60, DQ90+, Prepaid, Default}, the last two absorbing. Forward
transitions increment by one (the status field is a counter); backward transitions may
jump directly to Current via reinstatement or modification. Transition support is
determined empirically — unobserved pairs receive zero mass on their own.

*State diagram (nodes):* Current, DQ30, DQ60, DQ90+ in a chain; Prepaid and Default as
absorbing sinks. Dashed edges: every transient state reaches both absorbing states.

### 6. Workflow (diagram nodes)

- Scoping: thesis, state space, MVP bounds *(complete)*
- Data diagnostics; state labelling; vintage switch *(complete)*
- Acquire 2006–2007 (+2005); re-run diagnostics
- Wrangling: state labels, panel construction, macro join
- Homogeneous baseline: count transitions, estimate P
- Chained evaluation harness on the baseline
- ML-conditioned matrix + three ablations
- Comparison: local vs. chained; compounding analysis

The harness precedes the ML deliberately: it surfaces data and state-label defects
while only one component is unproven.

### 7. Next session

1. Re-run the three diagnostics on 2006–2007: zero-balance code distribution,
   delinquency-depth curve, first-touch timing. Expect a materially larger credit-event
   count and a depth curve that decays rather than plateaus.
2. Confirm schema identity across vintages — column positions and enumerations have
   changed across dataset releases and must be verified, not assumed.
3. Fix absorbing-state definitions against the new counts and proceed to panel
   construction.

### 8. Open items

- Macro series selection and join key (unemployment, HPI); IFRS 9 scenario weighting
  for forward covariate paths.
- Class imbalance strategy once the true default rate is known.
- Whether Default remains rare enough at 2006–2007 rates to require sampling or class
  weights, or whether per-state AUC at h = 60 is stable without them.

---

## Session 3 Checkpoint — Vintage Confirmation and the Default Boundary

### 1. Outcome of this session

The 2006–2007 vintage switch decided in Session 2 is confirmed viable. Five diagnostics
were run and committed. The Default state boundary — left open at the end of Session 2
— is now fully specified. No modelling code has been written; this remains deliberate.

### 2. Conceptual foundations

Recorded for the write-up's opening section.

**Lifetime PD.** The probability that a loan defaults at some point before it
terminates, as opposed to within a fixed 12-month window. It is the quantity IFRS 9
requires once a loan is deemed to have deteriorated. It cannot be observed directly
without waiting decades, so it must be constructed from short-horizon dynamics.

**The transition matrix is a table of tallies.** For each state s, count where loans in
s were observed one month later, and normalise. Each row is a probability distribution
over destinations. Prepaid and Default have degenerate rows — unit mass on themselves —
which is what absorption means.

**Chaining.** Starting from a point mass on Current and applying the matrix n times
redistributes probability across states at horizon n. The Default coordinate of that
distribution accumulates monotonically and is the lifetime PD estimate. The chain is the
mechanism that converts a one-step model into a lifetime forecast.

**Why ML.** The homogeneous chain assigns one row to every borrower in every month. This
is false across borrowers (FICO, LTV, DTI shift the odds) and across time (2008 and 2013
are different matrices). Stratification is the obvious fix but exhausts itself after two
or three splits, as cells empty of the rare transitions that matter. A classifier
conditioned on (s_t, x_i, macro_t) produces a per-loan-month row while sharing statistical
strength across similar loans. The classifier remains one-step; the chain is untouched.
This separation is what prevents the project collapsing into a survival model on
60-month labels.

### 3. Pre-registration and outcome

Expectations were logged before the diagnostics were run. Both estimates of conditional
default were low. The cure-rate band and incidence band held.

| Quantity | Pre-registered | Observed |
|---|---|---|
| DQ ≥ 6 incidence | 10–20% | 12.25% |
| Cure rate | 20–50%, long-tailed | 38.17%, long-tailed |
| Default given DQ ≥ 6 | 20% (YB) / 50% (Claude) | 61.19% |
| Credit events | ≫ 1,008 | ≈198,000 |

### 4. Diagnostics (2006Q1–2007Q4, 8 files, 15.7 GB)

| Quantity | Value |
|---|---|
| Rows / distinct loans | 150,967,486 / 2,431,736 |
| Observation window | 2005-12 – 2026-03 |
| Months tracked (median / mean / p90) | 49 / 62.1 / 130 |
| Terminated / censored | 97.66% / 2.34% |
| Prepayment (01) | 86.21% |
| Credit events (02/03/09) | 8.14% |
| Administrative exits (15/16) | 2.55% |
| Code 96 (defect repurchase) | 0.77% |
| Loans reaching DQ ≥ 6 | 297,967 (12.25%) |
| cure rate | 38.17% |
| median months to cure | 7 (p99 = 66) |
| carrying modification flag | 81.53% of cured |
| eventual credit event | 61.19% (81.13% excl. censored) |
| Deep-delinquent loan-months | 4,572,505 |
| Ever-RA loans / RA loan-months | 118,026 / 969,765 |
| REO holding period (median / p99) | 6 / 24 months |

**Cure mechanism.** The cure-time distribution decays smoothly and monotonically from a
mode of one month, with no spike at 3, 6 or 12. This is the signature of genuine workout
rather than administrative re-aging, and it is the discriminating test that the 2019
vintage failed. The mechanism is modification: 81.53% of cured loans carry the
modification flag against 34.09% of the DQ ≥ 6 population. Modification status is
therefore a required covariate — a post-modification Current is not the same state as a
never-modified Current.

### 5. Decisions

**D7 — Default is the union of two markers.** Diagnostic 04 established that RA→09 is
99.19% clean, so RA loans were already inside the credit-event count; but 80,773 loans
(40.83% of credit events) terminate 02/03 without ever reaching REO. Neither marker alone
is sufficient. Default is entered at the first occurrence of RA or a credit-event
zero-balance code, whichever comes first.

**D8 — Default is recognised at REO acquisition, not disposition.** Median REO holding
period is 6 months. Defining Default at disposition leaves a loan whose foreclosure has
completed in a transient state it cannot leave, biasing chained forecasts toward optimism.
RA is absorbing in the data: 2 loans of 118,026 return to a numeric status, none to
Current. Records are truncated at first RA. Consequence: the population of defaults is
unchanged, but recognition moves ≈6 months earlier and the lifetime PD curve shifts left.
This is a reportable finding.

**D9 — Code 96 splits on prior RA.** Code 96 (Confirmed Defect prior to Property
Disposition) is the renumbered 06, which is why 06 never appears. Freddie's own summary
statistics group 96 with 01 as pre-default repurchase; this is rejected, as 73% of the 96
population is deeply delinquent and the grouping would understate PD. With a prior RA
(879 loans) it is Default; without, it is censored under D5.

**D10 — Documented data-integrity exclusion.** 15 loans exhibit an unbroken run of status
00 (25–64 months) followed immediately by RA. Foreclosure cannot complete in one month;
these are excluded and reported as a count (0.013% of ever-RA loans).

**D11 — The DQ ≥ 6 threshold is diagnostic only.** It was a probe for forbearance
contamination, not a state boundary. Under D2 (Option B) Default derives from terminal
events, and DQ90+ absorbs all numeric statuses ≥ 3. Where the threshold falls moves no
state boundary.

**Open item.** Status 99 carries 173 loans as a pre-RA maximum against 14 at 94 — a spike
against monotone decay, suggesting a sentinel rather than a rung. Immaterial unless it
denotes unknown, in which case those rows are excluded rather than bucketed into DQ90+.
To be confirmed against the User Guide.

### 6. Evidence for the research question

Group B (foreclosed without reaching DQ ≥ 6) concentrates in 2008 (18.58% vs 3.95% for
group A) and is negative in every year from 2011. Foreclosure resolution was fast early in
the crisis and slow afterwards as court backlogs and moratoria accumulated. Identical
borrower behaviour, different transition speed, driven by calendar time. Separately,
adding 2006Q1 (+300K loans) moved every headline rate by less than 0.15pp, indicating that
heterogeneity in this vintage is dominated by calendar time rather than origination
cohort. Both observations are direct, pre-modelling evidence that the homogeneity
assumption fails, and that macro conditioning is load-bearing.

### 7. Working practice

Diagnostics are written to `diagnostics/NN_name.py` with output committed alongside as
`.txt`. Delegated agents do not run git commands and do not touch `.gitignore` or `.env`.
Pipeline directories are created when pipeline code exists, not before.

### 8. Next milestone — state labelling and panel construction

1. Map current loan delinquency status to {Current, DQ30, DQ60, DQ90+} for numeric values
   0, 1, 2, ≥ 3; map RA to Default. Remove the `.str.isdigit()` guard, which currently
   discards 969,765 RA loan-months.
2. Apply D7–D10 to assign absorption and truncate each loan's record at first absorption.
3. Emit the transition panel: (loan_id, month, state, next_state), with censored final
   rows carrying a null successor.
4. Join origination covariates and the modification flag.

Only then: the homogeneous baseline, followed by the chained evaluation harness built on
the baseline before any ML is introduced.

### 9. Workflow (diagram nodes)

- Vintage diagnostics → 2019 rejected
- Vintage switch 2006–2007
- Confirmation + Default boundary D7–D11 *(you are here)*
- State labelling, transition panel
- Homogeneous baseline, single matrix P
- Chained evaluation, h ∈ {6, 12, 24, 60}
- ML-conditioned rows, P(s_t+1 | s_t, x_i, m_t)
- Comparison + ablations (duration, macro)

---

## Session 4 Checkpoint — State Boundaries Closed
**5 September 2026**

### 1. Session objective

Resolve the status 99 anomaly left open at the end of Session 3, then close every
remaining state-boundary question so that transition panel construction can begin without
unresolved labelling decisions.

**Outcome:** all boundaries settled. Every delinquency status value and every zero-balance
code now has a documented disposition.

### 2. Diagnostics run this session

| Script | Runtime | Question |
|---|---|---|
| `06_status_99.py` | 5.56 min | Is status 99 a cap, a sentinel, or terminal-only? |
| `07_99_to_current_mod.py` | 4.09 min | Do the 110 one-month 99 → Current cures coincide with modification? |
| `08_code96_disposition.py` | 11.43 min | Do non-RA code-96 terminations behave like prepayments or credit events? |

### 3. Diagnostic 06 — status 99 is a CAP

#### 3.1 Pre-registration

Two competing calls were recorded before the run. The user predicted the cap signature
(predecessors mostly 98/99); Claude predicted sentinel (predecessors resembling the
general population). The user was correct.

#### 3.2 Results

- 13,836 loan-months at 99 (0.0092% of panel) across 569 loans (0.0234%).
- Statuses 90–98 decay smoothly at ≈0.93 per step (90: 1,029 → 98: 586); 99 then jumps by
  a factor of 23.61.
- Predecessors: 99.78% arrive from 98 or 99 (13,258 from 99, 540 from 98). Only 30 from
  anything else, 4 from 00. Zero loans whose 99 row is their first.
- Successors: 97.63% go to 98 or 99. 110 (0.81%) return to 00. 156 go to RA.
- Run lengths: median 15 months, mean 23.9, p90 60, max 114. 36.2% of runs exceed 25
  months; 9.9% are a single month.
- Terminality: 44.29% have 99 as their final row — so reading (c) is falsified. Of those
  252: 100 credit events, 100 administrative exits, 9 prepaid, 4 code 96, 39 no code.
- The 39 no-code loans all sit on 202603, the dataset's final reporting month — one
  distinct month, zero scatter. Censoring fully explains them.
- RA follows a 99 row 156 times and never precedes one. XX does not occur in these loans'
  history.

#### 3.3 Resolving the apparent anomalies

The agent flagged the ×23.61 spike as unexplained by the cap reading. It is not. The
comparison conflated two different quantities:

> A loan passes through 98 — one rung per month, one month there. So ≈586 loan-months at
> 98 corresponds to ≈586 loans. At 99 the loan stops climbing and dwells, because there is
> no higher rung. With a mean run of 23.9 months, 586 × 23.9 ≈ 14,000 against 13,836
> observed. The spike is dwell time, not excess mass.

#### 3.4 Documentary confirmation

The Freddie Mac General User Guide states that Current Loan Delinquency Status "will be
capped at 99". Reading (a) confirmed; (b) and (c) refuted. The guide also confirms the
scale is months: 00 = current or <30 days, 01 = 30–59 days, 02 = 60–89, 03 = 90–119, and
so on; RA = REO Acquisition, XX = Not Available. The ≥ 6 threshold used throughout
diagnostics 03–05 therefore does mean six months delinquent.

### 4. Diagnostic 07 — the 110 one-month cures

110 loan-months across exactly 110 distinct loans; no loan makes the jump twice.

| Flag at 99 ↓ / at 00 → | (blank) | P | Y |
|---|---|---|---|
| (blank) | 74 | 0 | 15 |
| P | 0 | 16 | 5 |

- 20 of 110 (18.2%) flip to Y in exactly the cure month — a modification recorded at the
  transition.
- Modification accounts for at most 36 of 110 (32.7%): the 20 flipping to Y plus the 16
  already carrying P.
- 51.82% of transition loans ever show Y, against a 24.84% baseline among 99-loans that do
  not cure — roughly double, real signal, but far from a complete explanation.
- 74 of 110 (67.3%) are blank on both rows and never flagged at all.

#### 4.1 Why the 74 are not an anomaly

The agent concluded the 74 unflagged cures were inconsistent with the cap reading. This
conclusion was rejected. Diagnostic 03 already established that 38.17% of DQ ≥ 6 loans
return to Current, with a smooth decay signature indicating genuine workout. Deep cures are
a documented feature of this panel. A cure out of 99 is the same phenomenon occurring at
the top rung, not a special mechanism requiring explanation. The impossibility intuition
came from reading 99 as literally 99 months of arrears owed; once 99 means "99 or more", it
dissolves.

#### 4.2 Modification flag semantics — documentary confirmation

Inferred from the data during the session, then confirmed against the guide: Y = Current
Period Modification (event marker, single month); P = Prior Period Modification (persistent
state); Null = Not Modified. The guide explicitly notes a loan may be modified more than
once, in which case Y is set in more than one reporting period — explaining the observed
P → Y → P pattern in F06Q10036336. Column index 7, confirmed documentarily (field 8,
1-based).

### 5. Diagnostic 08 — code 96 without RA

#### 5.1 Pre-registration

Claude predicted the target set would exit from Current at a rate closer to prepayments,
with a minority deep-delinquency half under 20%. Both were wrong. The user leaned mixture;
a mixture is what appeared, though not of the composition either party described.

#### 5.2 Results

| Group | Exit from 00 | Exit from ≥03 |
|---|---|---|
| TARGET (96, no RA) | 40.71% | 53.33% |
| REF-01 prepaid | 97.96% | 0.90% |
| REF-CE 02/03/09 | 1.11% | 39.25% |
| REF-CE no RA | 2.72% | 96.14% |

- Deepest status ever: 83.20% of target reaches ≥03, vs 4.00% (prepaid) and 99.02% (credit
  events). Median peak 11 months, vs 0 and 13.
- Calendar timing: target peaks 2010 (27.37%), with 82.7% falling in 2009–2013 — the
  credit-event pattern shifted roughly a year earlier.
- Modification: 27.57% ever Y, against 1.98% (prepaid) and 12.73% (credit events) — higher
  than either.
- Observed life: mean 46.1 months, vs 54.7 (prepaid) and 67.2 (credit events) — shorter
  than both.

#### 5.3 Section 7 — the mixture, and why it is not two loan types

| Half | Loans | Exit 00 | Exit ≥03 | Peak year |
|---|---|---|---|---|
| Ever ≥03 | 14,766 (83.20%) | 30.47% | 64.10% | 2010 (32.09%) |
| Never >02 | 2,981 (16.80%) | 91.41% | 0.00% | 2008 (32.20%) |

The documentation resolves the interpretation: code 96 = Confirmed Underwriting Defect or
Major Servicing Defect prior to a credit event. Actual Loss is computed only for codes 02,
03, 09 and 15 — not 96, with all loss components nulled for defect-settled loans.

These loans were removed by a repurchase or make-whole triggered by a defect finding, not
by the borrower's credit outcome. The bimodality is precisely what that mechanism predicts:
the audit finds the defect at whatever point the loan happens to be at. Some are caught
while current in 2008, most while deeply delinquent in 2010. A genuine two-type story would
show the halves differing in something other than when they were interrupted — they do not.
The 27.57% modification rate reflects defect audits and loss mitigation both concentrating
on troubled loans; it is co-occurrence, not causation.

### 6. Decisions taken (D12–D18)

| ID | Subject | Decision and rationale |
|---|---|---|
| D12 | Status 99 | CAP, meaning "99 or more months delinquent". Folds into DQ90+ with all other values ≥3. No new state, no special handling. Confirmed by both diagnostic 06 and the User Guide. |
| D13 | 99 → Current | Not an anomaly. Once 99 sits inside DQ90+, these are ordinary DQ90+ → Current transitions, of which 38.17% of DQ ≥ 6 loans already make one. |
| D14 | Events vs. states | A state must be a place a loan can be and persist in. Events (modification, defect removal, note sale) are edges or exits, never states. Applied three times this session and held each time. |
| D15 | Absorption | A loan's panel ends at absorption. The final transition emitted is into Default or Prepaid; no self-transitions afterwards. Absorbing self-loops are known structure, not learnable content — including them would bloat the baseline diagonal, dominate training rows, and inflate every metric with free correct answers. |
| D16 | Administrative exits (15/16) | Censored. Transitions kept to the last observed month, no terminal transition. 61,967 loans (2.55%). Note: code 15 = whole loan sales, used by Freddie Mac to disclose NPL/RPL sales, so these are not credit-neutral exits; the write-up must not describe them as such. Code 16 = reperforming loan securitisations. |
| D17 | Code 96 without RA | Censored, dropped at last observed month. 17,747 loans (0.73%). Defect-driven removal is unrelated to borrower behaviour; Freddie Mac books no loss. Mapping to Default would import unrealised outcomes; mapping to Prepaid would be wrong for the 83% deep in arrears. Amends D9, which had left this subset open. |
| D18 | Code 06 | Does not exist. Not in the documented enumeration (01, 02, 03, 09, 15, 16, 96), which explains diagnostic 02's zero count. The original brief's "06/15/16" grouping was mistaken. |

### 7. Termination taxonomy — now complete

| Code | Meaning | Loans | Disposition |
|---|---|---|---|
| 01 | Prepaid / matured | 2,096,296 | Prepaid (absorbing) |
| 02/03/09 | Credit events | 197,850 | Default (absorbing) |
| RA (any) | REO acquisition | — | Default, at acquisition |
| 15 | Whole loan sale (NPL/RPL) | 10,498 | Censored |
| 16 | Reperforming securitisation | 51,469 | Censored |
| 96 with RA | Defect, post-RA | 879 | Default (D9) |
| 96 without RA | Defect, pre-credit-event | 17,747 | Censored (D17) |
| none | Alive at 202603 | 56,997 | Censored |

Panel total: 2,431,736 loans, 150,967,486 loan-months, 15.7 GB. Zero loans carry more than
one zero-balance row — every terminated loan has exactly one code on exactly one row, so no
tie-breaking is required anywhere.

### 8. ML shape — established, not yet built

- **Training unit is the loan-month**, not the loan. One loan alive 60 months contributes
  ≈60 rows. This is why 2.43M loans becomes ≈151M rows.
- The homogeneous baseline is built from the same pairs — tallied rather than fitted. Same
  data, same unit, two uses. This is what makes the comparison fair.
- **Multiclass classification, six outcomes.** The output needed is the full probability row
  (`predict_proba`), not the argmax — the row is the transition matrix row and gets chained
  forward.
- **Calibration matters more than accuracy**, because a confidently wrong row compounds
  through every subsequent step. This is the mechanism behind the local-vs-chained gap that
  the project is built to measure.
- Model family deferred until the panel and feature set exist. Gradient-boosted trees are
  the tabular default; the interesting question is what beats the tally, not which library.

#### 8.1 Why chaining does disproportionate damage here

Two effects compound. First, a slightly biased one-step row is applied 60 times and drifts
in one direction; errors accumulate rather than average out. Second — and more important
for this panel — prepayment is 86.21% of all outcomes and is absorbing, so probability mass
drains into it quickly. What remains circulating in the live states is a shrinking
remainder, and it is the sparse deep-delinquency edges that decide how that remainder splits
between cure and default. The worst-estimated rows in the matrix therefore have
disproportionate influence on lifetime PD, and get 60 chances to be wrong.

Corollary for interpreting results: the baseline will be excellent on Current→Prepaid, with
two million events to count. Aggregate metrics may look close. The comparison lives in the
sparse edges.

### 9. Feature backlog — all requiring lagging

Three columns identified but not yet used, all carrying the same leakage hazard.

| Feature | Index | Hazard |
|---|---|---|
| Modification flag | 7 | Y is stamped in the cure month itself (20 of 110 cases in diagnostic 07) |
| Payment deferral flag | 24 | Guide: "Payment Deferrals are not considered modifications" — a separate channel |
| Borrower assistance plan | 29 | F = Forbearance, R = Repayment, T = Trial |
| Delinquency due to disaster | 28 | Not yet assessed |

Rule for all four: the feature is "as of this month, had this been true at any point
strictly before now?" Built from lagged history, never reading the current row. This yields
a clean ablation of the same shape as the planned semi-Markov time-in-state ablation: does
prior-assistance help 1-step, and does it survive chaining?

### 10. Deliberately not pursued

- **Diagnostic 09** (payment deferral as the explanation for the 74 unflagged 99 → 00
  cures). Declined on scale: 74 loans, and the result would change no pipeline code, since
  99 folds into DQ90+ and the transition is DQ90+→Current under any mechanism. The three
  columns are worth having as features, not as a diagnostic.
- **A "defect" or "exit" state for codes 15/16/96.** Rejected under D14 — these are exits,
  not places, and an absorbing pseudo-state would drain probability mass at every horizon
  and depress predicted default rates.
- **Diagnostic 08's classification role.** Once the documentation settled the disposition,
  08 became characterisation rather than classification. Its numbers describe what the loans
  looked like at removal; they do not reopen the mapping.

### 11. Process finding — documentation before diagnostics

Three of this session's questions (status 99, modification flag semantics, code 96) were
answered in the User Guide and were derived the slow way instead. That is roughly 20 minutes
of compute and a substantial share of the session.

The lesson is not "read the docs instead of running diagnostics". Documentation answered
what 99 means; it did not and could not report that the spike is dwell time, that 110 loans
cure out of 99, or that a third of those carry a modification. Documentation is also stale
in places — code 06 appears in the original brief but does not exist in the enumeration or
the data.

Revised practice: definitional questions go to the guide first; behavioural questions still
require the panel; load-bearing definitional answers get verified against the data anyway,
because "the guide says X, and diagnostic N confirms X holds in this panel" is a materially
stronger claim in a write-up than either alone.

### 12. Repository state

- `diagnostics/` — scripts 01–08 with `.txt` outputs committed alongside.
- `data/` — gitignored.
- `docs/` — to be added. General User Guide (July 2026) and File Layout (July 2026,
  `.xlsx`). Take the July 2026 versions, not the pre-July ones: the panel runs to 202603,
  which is the Release 47 window, and checking column positions against a superseded layout
  would confirm the wrong thing.
- No pipeline directories yet — none are created until pipeline code exists.
- Column positions confirmed documentarily (1-based field N = index N − 1): 0 loan ID,
  1 period, 3 delinquency status, 7 modification flag, 8 zero-balance code. Files have 35
  genuine columns; the 35th is Bankruptcy Cramdown Costs, populated only in the zero-balance
  period. An earlier report of "34 real columns plus trailing delimiter" was wrong; no
  diagnostic is affected, since only indices 0, 1, 3, 7, 8 were ever read.

### 13. Pre-registration scorecard

| Question | User | Claude | Correct |
|---|---|---|---|
| D06: predecessors of 99 | mostly 98/99 | sentinel | User |
| D07: modification flag semantics | event/state | pending/complete | User |
| D08: code 96 shape | mixture | prepayment-leaning | User |

### 14. Next session

1. Download the two July 2026 documents into `docs/` before writing any further script.
2. Specify the transition panel. The row is a loan-month: loan, month, current state,
   features; label is next month's state. Open sub-questions: gap handling (loans with
   non-adjacent reporting months), the exact first row of each loan's panel, and whether the
   panel is materialised to disk or streamed.
3. Build the panel and validate it against the diagnostics — the counts in the termination
   taxonomy above are the acceptance test.
4. Homogeneous baseline: tally transitions, one matrix.
5. Chained evaluation harness on the baseline, before any ML.

### 15. Workflow (diagram nodes)

- Vintage selection: 2019 rejected → 2006–07
- Diagnostics 01–08: viability, boundaries
- State definitions D7–D18 — closed
- Transition panel: loan-month rows
- Homogeneous baseline: tally one matrix
- Chained eval harness: h = 6, 12, 24, 60
- ML-conditioned rows: multiclass, calibrated
- Comparison: local vs. chained

Notes on the diagram: harness is built on the baseline before ML is introduced — surfaces
panel and label defects early, when only one model can be at fault. Acceptance test: panel
counts must reproduce the termination taxonomy.

---

## Session Checkpoint — State-space closure and the third arm
**11 September 2026**

### 1. Session objective and outcome

The session opened on one unresolved fork: whether DQ90+ is a single state or a holding pen
concealing two. It closed on documentary grounds, and the investigation that closed it
produced a second, unplanned result — a third model arm.

**Headline:** the state space is final at six states. A new stratified arm sits between the
homogeneous baseline and the ML treatment. One open question (O1) may reframe the
semi-Markov ablation and must be resolved before that ablation is specified.

### 2. The DQ90+ fork and how it closed

#### 2.1 The problem

D08 established a mean run of roughly 24 months at the delinquency cap. DQ90+ therefore
collects a loan that missed its third payment last month alongside a loan two years into a
stalled foreclosure. A homogeneous chain gives both the same exit hazard row — too
pessimistic for the arrival, too optimistic for the stuck loan.

#### 2.2 The wrong reason to leave it unsplit

An early position in the session was to leave the baseline blurred so that the ML would
receive credit for fixing it. This was rejected, and the rejection is recorded because the
reasoning matters more than the conclusion:

- Handicapping the control makes the headline result evidence about the experimental design
  rather than about borrower conditioning.
- It confounds the mechanism. If the baseline is blurred and the ML sees duration, part of
  the measured gain is duration information, not borrower covariates — and the two can no
  longer be separated.
- It destroys the planned semi-Markov ablation, which requires duration to be switchable on
  and off independently of the model arm.

Splitting a state does not make a chain non-homogeneous. Homogeneity is one matrix for every
loan in every month; the number of states is a separate choice. A six-state and a
seven-state homogeneous chain are both homogeneous.

#### 2.3 The right reason it stays unsplit

Two tests were set for a defensible split: (i) an observable event marking the boundary, and
(ii) interchangeability of loans within each resulting state. Test (i) was sent to the
documentation.

**Part A finding, Diagnostic 09:** no foreclosure-status field exists in the monthly
performance file. Foreclosure progression — referral, in-process, attorney assignment — is
not observable as distinct from the delinquency counter at field 4 (index 3). Seven fields
touching foreclosure or workout were enumerated and each fails the test. The nearest miss is
field 30 (index 29), Borrower Assistance Plan, which the Guide restricts to January 2014 and
following periods and is therefore null across the entire 2007–2011 window these vintages
exist to observe.

Field 4's cap, quoted so the record stands alone: "This value for any given month will be
capped at 99." The Guide says nothing about 98, which is the point — 98 is the ordinary rung
below the cap, and XX = Not Available is a separate missing-value code. Both fold into
DQ90+.

With no observable boundary, any split would be drawn at a month chosen by the modeller. The
state space stays at six.

### 3. Diagnostic 09 — exit hazard from DQ90+ by run length

Full panel, 2006Q1–2007Q4. 6,463,333 DQ90+ loan-months from 373,546 distinct loans, 4.28% of
the 150,967,486-row panel. Run length counts consecutive months in DQ90+; a reporting gap
breaks the run and the counter restarts.

| Run (mths) | Denom | Cure | Stay | Default | Prepaid | Censored |
|---|---|---|---|---|---|---|
| 1 | 605,140 | 17.41% | 80.17% | 0.67% | 1.45% | 0.30% |
| 2 | 490,139 | 8.71% | 88.68% | 1.48% | 0.95% | 0.17% |
| 3 | 439,596 | 6.94% | 89.33% | 2.79% | 0.79% | 0.14% |
| 4–6 | 1,082,297 | 6.13% | 88.67% | 4.44% | 0.64% | 0.12% |
| 7–12 | 1,401,953 | 4.38% | 89.13% | 5.68% | 0.45% | 0.35% |
| 13–24 | 1,314,850 | 2.88% | 90.17% | 5.38% | 0.30% | 1.27% |
| 25+ | 1,129,358 | 1.63% | 92.35% | 4.48% | 0.17% | 1.38% |
| all | 6,463,333 | 5.61% | 88.97% | 4.22% | 0.56% | 0.65% |

No thin buckets; the smallest denominator is 439,596, three orders of magnitude above the
1,000 threshold. Zero unclassified loan-months, zero duplicate (loan, period) rows, zero
undocumented zero-balance values — rates sum to 100% in every bucket and every denominator is
exact. Two independent full-panel runs agree on every figure; only wall-clock differs (472.1s
vs 483.1s).

#### 3.1 Three features of the table

1. **Cure halves between month 1 and month 2** (17.41% → 8.71%), then decays smoothly. The
   discontinuity is at the entry point, not in the corridor where a second state would live.
2. **Default is hump-shaped** — rising to 5.68% at 7–12 months, falling to 4.48% at 25+.
   Loans deep in the corridor default less per month than loans in the middle. This is
   structure, not blur, and is not yet explained.
3. **Censoring rises 4–10× in the two long buckets** (0.12% at 4–6 to 1.38% at 25+).
   Combined with the release-notes finding that code 15 is Freddie Mac's NPL/RPL sale
   disclosure, the falling default rate at 25+ is partly survivorship: troubled loans are
   being sold off the book, not curing.

### 4. O1 — the collinearity objection (OPEN, blocking)

Both pre-registrations predicted smooth decay with no cliff. The observed month-1 drop
contradicts both. Before it is read as a behavioural finding, a construction objection must
be cleared.

The delinquency field counts consecutive missed payments. Inside DQ90+, run length and raw
status are close to the same quantity: run length 1 ≈ status 3, run length 2 ≈ status 4, run
length 25 ≈ status 27 or capped. Cure is defined against a fixed boundary (status ≤ 2).

If that collinearity is tight, then:

- The declining cure rate measures distance to a fixed finish line, not duration dependence.
  A loan at status 3 needs to recover one payment; a loan at status 27 needs twenty-five.
- The month-1 drop is a boundary effect of the cure definition, not a behavioural cliff.
- The semi-Markov framing is partly mislabelled. Time-in-state would not be information
  beyond the state — it would be information the state space discarded when everything ≥ 3
  was collapsed into one bucket. Still a legitimate result; a materially different sentence
  in the write-up.

Collinearity is not perfect: D05 found loans entering deep delinquency directly at status 4
or 5, and partial payments can reduce status without a full cure.

**Check required:** cross-tabulate run length against raw delinquency status within DQ90+.
Cheap. Blocks specification of the semi-Markov ablation.

### 5. The third arm — stratified chain

Proposed this session as two parallel chains. Correctly named, this is stratification, and it
is a rung on the same axis as everything else:

| | |
|---|---|
| One matrix for everyone | homogeneous baseline |
| k matrices, chosen by group | stratified |
| One matrix per loan-month, chosen by covariates | ML treatment |

It cannot serve as the baseline — selecting a matrix by loan characteristics is conditioning,
which leaves no control. And it does not address the DQ90+ blur; both strata would carry the
same collapsed state.

**Why it earns a place anyway.** A two-arm design cannot say whether the ML wins because it
conditions at all or because it conditions finely. Three arms on the same chained harness can.
If a crude split captures most of the ML's gain, that is a finding — deflating for the ML,
which is exactly why it must be reported. If the ML pulls far ahead, the fine conditioning is
shown to do real work. Implementation cost is low: the same tallying code as the baseline, run
per stratum.

#### 5.1 Stratification axis

**Calendar/macro regime**, not borrower features and not delinquency status (status is already
the state space; splitting on it is a finer state space, not a second chain).

Justification is D05: calendar-time heterogeneity dominates origination-cohort heterogeneity.
This is also the axis the homogeneous baseline is blindest to — one matrix pools boom, crash
and recovery into a single row — and it is the crude version of precisely what the ML arm does
with macro covariates, which makes it an informative middle rung rather than an arbitrary one.

#### 5.2 Date cut vs macro threshold

**Fixed date cut — for:** period column only, no join, runs today; trivially explained and
checked; honest as a stated modeller's choice.
**Against:** boundary arbitrary (why 2008 and not 2007Q3?); fixed in time, so it cannot
transfer — fatal given a deployment story with live drift monitoring; national, so Nevada and
Texas share a regime they plainly did not.

**Macro threshold — for:** boundary is a property of the world, not an opinion; the same rule
applies to unseen periods; speaks the same language as the IFRS 9 scenario machinery the ML arm
needs regardless; supports regional cuts via state-level series.
**Against:** arbitrariness moves rather than vanishes (why 8%?); requires the macro join, which
does not yet exist; thresholds on smooth series flicker as months bounce across the line.

**Decided: macro threshold.** The deciding argument is critical path — the macro join is
required for the ML arm regardless, so the threshold approach costs nothing that was not already
owed, whereas a date cut is a detour that gets discarded later.

#### 5.3 Generalisation — what travels and what does not

- **The recipe travels.** "Define regimes by a macro condition, estimate a chain per regime"
  transfers to corporate ratings, cards, any migration problem.
- **The threshold does not.** Any specific number is fitted to US mortgages in one crisis and is
  wrong elsewhere.
- **The binding limitation (O6).** The panel contains exactly one downturn. The threshold is
  drawn around a single event, not estimated from a distribution of crises. The model learns what
  the 2008 housing collapse did to mortgage transitions; a differently-shaped downturn — a rate
  shock, say — would fire the regime machinery and select the wrong rows. This constrains the ML
  arm equally, since its macro sensitivity is learned from the same single episode. It is a known
  limitation of regime models, must be stated rather than hidden, and the threshold approach at
  least makes it visible.

### 6. Decisions closed this session

| ID | Decision | Basis |
|---|---|---|
| D19 | No foreclosure marker exists | Documentary, Guide + File Layout July 2026; date fields enumerated and rejected; field 30 outside the observation window. |
| D20 | State space final at six | No observable boundary to split DQ90+ on; any split would be modeller-drawn. |
| D21 | Time-in-state is a panel column, not a state boundary | Follows D20. Lets whatever gradient exists be measured without asserting a boundary the data does not supply. See O1. |
| D22 | The baseline is built as well as we know how | Handicapping the control confounds the mechanism and destroys the ablation. Stated as a standing principle, not a one-off. |
| D23 | Third arm: stratified chain, between baseline and ML | Separates "conditions at all" from "conditions finely", at low marginal cost. |
| D24 | Stratification axis is calendar/macro regime | D05: calendar-time heterogeneity dominates cohort heterogeneity. Also the baseline's blindest axis. |
| D25 | Regime defined by macro threshold, not fixed dates | Transfers to unseen periods; shares the critical path with the ML arm's required macro join. |
| D26 | Stratified arm deferred until baseline + chained harness exist | It is a middle rung; it has no meaning until the rungs either side are built. Chasing the macro join would put large data work ahead of the baseline. |

### 7. Open items

| ID | Status | Item |
|---|---|---|
| O1 | Blocking | Run-length vs raw-status collinearity within DQ90+. Blocks specification of the semi-Markov ablation. Cheap cross-tabulation. |
| O2 | Blocking panel | Terminal-row double attribution. 117,581 DQ90+ loan-months are themselves the loan's zero-balance row, classified from their own code. Where the preceding month is also DQ90+, one exit is attributed at two adjacent run lengths — ~1.8% of the slice, landing disproportionately in Default and Prepaid, the two smallest columns. Requires a modelling call: is a terminal row a month at risk, or the exit itself? The agent deliberately did not decide this. |
| O3 | Deferred | Informative censoring. Code 15 is the NPL/RPL sale disclosure — 10,498 loans left the book because they were troubled. D15 sends them to censored, still defensible, but the write-up cannot call them administratively neutral. Check whether they cluster in 2009–2011 alongside the credit events. |
| O4 | Noted | RA is not strictly absorbing in raw data: 7 DQ90+ loan-months occur at or after a loan's first RA. Immaterial at this scale; relevant only if a later truncation at RA is expected to reconcile exactly. |
| O5 | Deferred to D25 | Threshold value and flicker handling for the macro regime definition. |
| O6 | Write-up | Single-downturn limitation. Applies to both the stratified and ML arms. |
| O7 | Unexplained | Hump-shaped default hazard across run length. Partly survivorship via O3, but not fully accounted for. |

### 8. Pre-registration scorecard

| Question | User | Claude | Correct |
|---|---|---|---|
| D06: predecessors of 99 | mostly 98/99 | sentinel | User |
| D07: modification flag semantics | event/state | pending/complete | User |
| D08: code 96 shape | mixture | prepayment-leaning | User |
| D09: DQ90+ exit shape | smooth decay | smooth decay, flat past 12 | Neither |

**A process note on D09.** The first instinct was to predict a cliff because a cliff was the
useful answer. That was caught and withdrawn. Pre-registration has value only when the
prediction is what is actually believed beforehand; otherwise the scorecard measures
preferences, not priors. The reasoning eventually given — that unobserved foreclosure events
are real cliffs in individual loans, but land at different months across judicial and
non-judicial states and different servicers, so the aggregate smooths — is a good argument and
worth keeping. It also implies something more general: **a smooth aggregate curve is not
evidence that no second state exists; it is evidence that its boundary cannot be located from
time-in-state alone.**

### 9. Workflow (diagram nodes)

- Data acquired: 2006Q1–2007Q4, 151M rows
- Diagnostics 01–09: D7–D26 closed
- State space final: 6 states, D20
- Transition panel: next session
- Homogeneous baseline: one matrix
- Chained harness: built on baseline
- Arm 1: homogeneous
- Arm 2: stratified by macro regime
- Arm 3: ML-conditioned
- Three-arm comparison: local vs chained
- Ablations: semi-Markov (see O1)
- FastAPI + drift monitor
- (macro join required at the stratified/ML stage)

### 10. Repository state

- `diagnostics/` — scripts 01–09 with `.txt` outputs committed alongside. 09 verified
  byte-identical to stdout and reproducible across two independent full-panel runs.
- `docs/` — populated. General User Guide and File Layout, July 2026 (Release 47 window,
  matching the panel's 202603 cutoff). Release notes.
- `data/` — gitignored. 8 files, ~16 GB.
- No pipeline directories yet; none created until pipeline code exists.
- Column positions (1-based field N = index N − 1): 0 loan ID, 1 period, 3 delinquency status,
  7 modification flag, 8 zero-balance code. 35 genuine columns; the 35th is Bankruptcy Cramdown
  Costs, populated only in the zero-balance period.

### 11. Standing rules for delegated agents

- Never run a git command. Never touch `.gitignore` or `.env`. Scripts and outputs are written;
  commits are manual.
- Chunked reads only, `usecols` restricted. Never load a full file.
- Report counts alongside every rate, and flag thin cells explicitly.
- State interpretation is the reviewer's job, not the agent's — numbers and denominators only.
- Definitional questions go to the Guide first; behavioural questions require the panel;
  load-bearing definitional answers are verified against the data anyway.
- Long scripts: write via the file tool, not a shell heredoc. A heredoc exceeded the spawn limit
  this session and the fallback coincided with a dropped connection.

### 12. Next session

1. Resolve O1 — run-length vs raw-status cross-tabulation. Cheap, and it determines how the
   semi-Markov ablation is framed.
2. Decide O2 — is a terminal row a month at risk or the exit itself? This is a
   panel-construction decision and cannot be deferred past panel build.
3. Specify the transition panel. The row is a loan-month: loan, month, current state, features;
   the label is next month's state. Open sub-questions: gap handling (non-adjacent reporting
   months), the first row of each loan's panel, and whether the panel is materialised to disk or
   streamed.
4. Build and validate the panel against the diagnostics — the termination taxonomy counts are
   the acceptance test.
5. Homogeneous baseline: tally transitions, one matrix.
6. Chained evaluation harness on the baseline, before any ML.

---

## Session Checkpoint — 17 September 2026

### 1. Session summary

Two things happened. Diagnostic 10 was specified, delegated, run and reviewed, which closes O1.
And the evaluation design was worked out from first principles before any model exists —
slicing, calibration, the train/test split, and the ablation grid. No panel code was written; O2
was deliberately deferred.

The evaluation design was reached by following a single error through the system: a model that
overstates cure on deep-delinquency rows. That error is nearly invisible in an aggregate one-step
score (dilution — the slice is tiny against 151M rows) and large at horizon 60 (compounding —
every loan that reaches deep delinquency carries it forward). Everything below follows from
wanting that error to be visible.

### 2. Decisions closed this session

| ID | Decision |
|---|---|
| D27 | Scores are sliced by (state × time-in-state), never reported as a single aggregate |
| D28 | Calibration is checked within groups, binned by predicted value — not by group average alone |
| D29 | Time-based split. Three walk-forward cutoffs: end-2007, end-2008, end-2010 |
| D30 | Groups whose time-in-state exceeds anything in the training window are flagged out-of-range in each round |
| D31 | Capping time-in-state is run as an ablation, never as the default |
| D32 | No 2014+ cutoffs by default |
| D33 | DQ90+ ablation grid: 5 runs first (all three on, each removed in turn, all three off); expand toward the full grid only if needed |
| D34 | O1 resolved: status and run-length are separable inside DQ90+; the semi-Markov ablation is specifiable |

**Table 2: What each walk-forward cutoff tests**

| Cutoff | Training has seen | What the round tests |
|---|---|---|
| End 2007 | calm, early cracks | forecasting a downturn never seen |
| End 2008 | the crisis arriving | forecasting the rest of a storm from inside it |
| End 2010 | the full crisis | does it relax, or stay stuck in crisis mode |

### 3. Diagnostic 10 — results and reading

Slice totals match 09 exactly: 150,967,486 panel rows, 6,463,333 DQ90+ loan-months, 373,546
loans, 605,140 runs. Runtime 898.5s. Column indices 7 / 24 / 29 confirmed against the layout
workbook. Flag persistence resolved from the Guide: modification and deferral are EVER flags
(onset tested); borrower assistance is current-month (value tested).

**Headline, and why it overstates**

| | |
|---|---|
| Off-step share of DQ90+ loan-months | 33.77% |
| First departures per run | 0.371 |
| Off-step months that are downstream of an earlier departure | 89.71% |

Deviation is monotone non-increasing within a run: the ladder climbs by exactly one per month
and field 4 by at most one, so once status falls behind it can never rejoin. One departure early
in a long run marks every later month of that run as off-step. **The 33.77% is therefore an
accumulation, not an event rate.** The honest statement is that 63% of runs never depart lockstep
at all, and that the mechanism (stickiness) was correctly predicted but is arithmetically
guaranteed, so it is weak evidence for the prediction.

The same caveat voids the T2 gradient. Off-step rising 18% → 50% across run-length buckets is
stickiness accumulating, not evidence that long runs see more departures. A genuine hazard needs
first departures over runs still on-step at that run-month. T7 has the numerators; the
denominators were not computed.

**T4 is the table that answers O1**

Conditional spread exists in both directions. At run 25+, status is 80.9% at 25–98 but 9.8% at
13–24, 4.4% at 7–12, 2.9% at 4–6. Reading down the status 13–24 column: 256,525 loan-months at
run 7–12, 1,029,303 at 13–24, 110,193 at 25+. Neither quantity determines the other.

**A third quantity: deviation as a feature**

Deviation only moves when the borrower pays without curing, so it is a count of payments made
while deeply delinquent. T5: 28.88% of off-step months sit at −6 or lower — six or more payments
made without escaping DQ90+. A borrower still paying after 18 months behind is a different animal
from one who has paid nothing. This is neither duration nor depth, and it is a plausible separator
of cure from default. Recorded as O9.

**T7 is not yet believable**

83.92% of first departures carry none of the three workout flags, so most step-downs are
undocumented partial payments. But two columns do not hold up:

- Modification onset 0.6% (1,264 of 224,651) is implausible for the HAMP era. Likely the
  modification ends the run and so falls outside the within-run window T7 tests. Check against exit
  rows.
- Assistance 15.5% (34,906) cannot be read as printed: the Guide restricts the field to January
  2014 onward, so every one of these must sit in 2014+, most likely COVID forbearance. Needs a
  period split.

### 4. Pre-registration scorecard

| Question | Yassine | Claude | Truth | Winner |
|---|---|---|---|---|
| Status 99 meaning | cap | sentinel | cap | Yassine |
| DQ ≥ 6 cure rate | — | — | 38.17% | Yassine |
| Conditional credit event | 20% | 50% | 61.19% | Claude |
| DQ90+ cure decay shape | smooth | smooth | halves at m2 | neither |
| Lockstep departure share | <5% | 20–50% (~30%) | 33.77% | Claude |

Caveat on the last row: on the event-rate reading (0.371 per run, 63% of runs never departing),
Yassine's instinct was closer than the loan-month metric suggests. Claude's margin is smaller than
it looks.

### 5. Open items

| ID | Status | Item |
|---|---|---|
| O1 | CLOSED | Resolved by diagnostic 10. See D34. |
| O2 | Blocking panel | Terminal-row double attribution. 117,581 DQ90+ loan-months are themselves the loan's zero-balance row. Where the preceding month is also DQ90+, one exit is attributed at two adjacent run lengths — ~1.8% of the slice, landing disproportionately in Default and Prepaid, the two smallest columns. The call: is a terminal row a month at risk, or the exit itself? Changes row count, stay lengths, and the baseline's DQ90+→DQ90+ tally. Cannot be deferred past panel build. Definitional-first rule applies: check the Guide on what the final row describes before deciding. |
| O3 | Deferred | Informative censoring. Code 15 is the NPL/RPL sale disclosure; 10,498 loans left because they were troubled. Check clustering in 2009–2011. |
| O4 | Noted | RA not strictly absorbing in raw data: 7 DQ90+ loan-months at or after first RA. |
| O5 | Deferred to D25 | Threshold value and flicker handling for the macro regime. |
| O6 | Write-up | Single-downturn limitation; applies to the stratified and ML arms. |
| O7 | Unexplained | Hump-shaped default hazard across run length. Partly survivorship via O3, not fully accounted for. |
| O8 | New | Off above: 7,293 loan-months (0.113%) where status rises more than one rung in one month — arithmetically impossible for a borrower falling behind. F06Q10001515 jumps 3→15 and holds dev = +11 for five months. Reads as re-reported history or a servicing transfer. Immaterial in size; needs a line in the write-up, not silence. Worth checking whether the affected loans cluster by period. |
| O9 | New | Deviation (payments made while in DQ90+) as a candidate feature in its own right, distinct from run-length and observed status. Feeds the ablation grid. |
| O10 | New | T7 credibility: modification onset at 0.6% and assistance at 15.5% are both suspect (see §3). Cheap follow-up; not blocking. |
| O11 | New | First-departure hazard by run-month — numerators exist in T7, denominators (runs still on-step at each run-month) do not. The only way to test whether departures genuinely concentrate at any run length. |

### 6. Repository state

- `diagnostics/` — 01–10, each with committed `.txt`. 10 is 1,324 lines, verified byte-identical
  to stdout via `cmp`, and validated against 27 hand-computed synthetic cases (which caught a
  duplicate-key double-count in T3 before the full run).
- `docs/` — General User Guide and File Layout, July 2026.
- `data/` — gitignored, 8 files, ~16 GB.
- No pipeline directories. None until pipeline code exists.
- Audits from 10: zero sort-order violations across all 8 files; zero duplicate (loan, period)
  rows among DQ90+ loans; zero values outside the documented enumerations; 1,354 runs broken by a
  reporting gap (0.021%).

### 7. Standing rules for delegated agents

Unchanged. Never run git; never touch `.gitignore` or `.env`. Chunked reads with `usecols` only.
Counts beside every rate; thin cells flagged. Numbers and denominators only — interpretation is
the reviewer's job. Definitional questions to the Guide first; load-bearing answers verified
against the data anyway. Long scripts via the file tool, not a heredoc.

New this session: a delegated diagnostic reuses the previous diagnostic's conventions verbatim and
states which ones, or states plainly where none existed to inherit. Diagnostic 10 did the latter
correctly regarding chunk-boundary carry-over — 09 has no such convention; it lexsorts after the
full read.

### 8. Next session

1. **Split the DQ90+ trio into memory vs. current-position.** Of run-length, observed status and
   deviation — which describe the loan's path and which describe where it is now? Only the former
   belong to the semi-Markov claim. This determines what the write-up is allowed to say.
2. Decide O2. Guide first, then the modelling call.
3. Specify the transition panel. Row is a loan-month: loan, month, current state, features; label
   is next month's state. Open sub-questions: gap handling, the first row of each loan's panel,
   materialised to disk or streamed.
4. Build and validate the panel — the termination taxonomy counts are the acceptance test.
5. Homogeneous baseline: tally transitions, one matrix.
6. Chained evaluation harness on the baseline, before any ML.
---

## Session Checkpoint — 23 September 2026

### Session summary

Milestone 1 is complete. Every panel-construction rule was closed on evidence rather than
argument, `src/panel.py` was built and audited, and the transition panel now exists: 69,927,577
rows across eight parquet files. Six diagnostics were commissioned and reviewed (12–17). The
homogeneous baseline is the next artefact and is a groupby away.

### Decisions closed this session (D35–D44)

| ID | Decision |
|---|---|
| D35 | **Terminal row — reading 2.** The row carrying the zero-balance code IS the exit month; the transition into it is labelled with the absorbing state. Partial months at risk are deleted, stated as a convention. Reading 1 (an extra self-transition, exit at t + 1) runs as a sensitivity ablation — it biases the per-month hazard down and dates every default one month late. |
| D36 | **Exit dating.** The exit is dated by the reporting period, always. The zero-balance effective date is a validation field, not the timestamp. No loans are excluded on date-disagreement grounds — diagnostic 12 section D shows the code row is the loan's last row for 100% of loans, so the observation is unambiguous even where the two fields disagree. |
| D37 | **Reporting gaps.** Any interior gap: drop the crossing pair, reset run-length to 1 on the far side, flag the row `post_gap`. One uniform rule, no size threshold. The provisional ≤ 2 / ≥ 3 threshold is withdrawn — its safe-fiction rationale required gaps to be random, and they are not. |
| D38 | **Falls stay, unmodified.** Falls are genuine, persistent cures, not modification-driven counter resets. The middle-rung collapse to {Current, Delinquent, DQ90+, Prepaid, Default} is dead; the four-state delinquency space survives. |
| D39 | **Skips stay, flagged, with a drop ablation.** Dropping them forces Current→DQ60, Current→DQ90+ and DQ30→DQ90+ to structural zero, so a Current loan could never reach DQ90+ in under three months under any covariate or macro condition, and the ML arm would learn the class is impossible. That asymmetry beats an unmeasurable overstatement of a 0.00011 monthly probability. |
| D40 | **"100% of a cell" is arithmetic, not a red flag.** Reaching Current from DQ60 requires a two-rung drop; reaching DQ60 from Current requires a two-rung rise. Those cells are definitionally 100% fall- or skip-populated. The earlier alarm on this basis is withdrawn. |
| D41 | **Panel schema has three groups:** keys, features, and construction metadata (`role`, `post_gap`, `is_skip`, `is_fall`). Metadata never enters the feature matrix — `is_skip` and `is_fall` are derived from `next_state` and would leak the label. Model code must select the features group by name, never "all columns except the states". |
| D42 | **HPI geography — US state level.** Fine enough to capture the sand-state collapse (NV/AZ/FL/CA vs TX), coarse enough that published IFRS 9 scenario paths exist for forward projection, and the property-state field is populated for every loan. Metro/MSA rejected: it forces an invented downscaling rule from national scenarios. |
| D43 | **HPI encoding.** Supply `est_ltv` (original LTV divided by the HPI ratio) and `original_ltv`, where the HPI ratio is state HPI at the origin month over state HPI at the origination month. `est_ltv` is the quantity the double-trigger mechanism turns on and 100 is a meaningful split point; `original_ltv` lets the model recover the components. Amortisation is ignored — small for 30-year loans in years 1–5, and it biases consistently toward overstating underwater status. An amortisation-corrected variant is a later ablation. Rejected: the HPI ratio alone (macro half only, cannot distinguish a 95-LTV loan from a 60-LTV one under the same price fall) and ratio plus original LTV supplied separately (spends model capacity rediscovering arithmetic, in the region where data is thinnest). |
| D44 | **Censoring rows carry a real reading.** The delinquency status on a code 15/16/96 row is a genuine observation, so the transition into it is emitted. Finding 1 of the code review is dismissed; no rebuild. |

### Diagnostic results

#### Diagnostic 12 — zero-balance effective date vs reporting period

Offset 0 dominates every code: 01 at 99.99%, 02 at 99.99%, 03 at 98.80%, 09 at 99.99%, 15 at
94.72%, 16 at 99.99%, 96 at 99.80%. The negative tail is four loans in total, far below the 2%
threshold that would have triggered a third reading. Effective dates are published at month
granularity for 100% of loans — no day component exists, so partial-month exposure is
unmeasurable, which strengthens reading 2 rather than weakening it. Code 03 has a tail of 229
loans at offset +3 or more, maximum +183 months (limitations note). Code 15 has 5.3% at offset +1,
two orders above every other code — a systematic reporting lag, write-up note only; it does not
touch the Default column. Section D integrity checks are all exactly zero.

#### Diagnostic 13 — interior reporting gaps

1,543 loans (0.0635%), 1,566 gap events (0.0010% of loan-months); 98.6% of gapped loans have
exactly one. Only 20% of gaps are size 1; median 4 months, mean 6.5, maximum 162. Selection is
extreme: 94.89% of gaps are preceded by DQ90+ against a 4.28% baseline, a 22-fold concentration,
with zero gaps preceded by DQ30 or DQ60. Calendar concentration is post-crisis, not crisis — 45%
in 2010–2011, only 6% before 2009. 88–90% of gaps are diagonal, so a naive pair-across would
mostly inject long multi-month spans into the DQ90+ self-transition cell.

#### Diagnostic 14 — status skips

59,242 skips out of 148,534,184 consecutive pairs (0.0399%). Three cells are 100% skip-populated:
Current→DQ60 (18,099), Current→DQ90+ (15,533), DQ30→DQ90+ (6,431). Correction signature: 27% of
origin-0 skips are back at status 0 by t + 2, 39% at 0 or 1 — a mixed population of corrections
and genuine fast deterioration, inseparable per row.

#### Diagnostic 15 — status falls

492,730 falls (0.3317% of pairs), 81.7% full cures to status 0. Modification association is weak
once conditioned on status(t): at status 2, 21.2% of falls carry a modification flag against 24.2%
of non-falls — falls are less modified. Only at deep delinquency does it invert (12+ rungs: 29.3%
vs 18.2%). Persistence: falls from DQ90+ are at status 0 in 61.4% of cases at t + 2 and 59.4% at
t + 3; only 14.3% are back at 3 or more. DQ60→Current is 100% fall-populated and entirely
ordinary.

#### Diagnostic 17 — censoring-row semantics and panel checks

Code 16 rows are 79.1% identical to the preceding month — but the uncontested no-code comparison
group is 94.9%, so censoring rows are less repetitive than an ordinary final month. Staleness
predicts the opposite. Code 15 is decisive: 3.2% identical, 96.5% changed by exactly one rung,
99.3% in DQ90+. Code 96 sits between at 44.3% identical. Panel checks: zero loans span more than
one quarter file, zero gap-only loans, and the modal first-payment offset is +1, confirming the
rate-spread origination month.

### Milestone 1 — panel build

| Quantity | Count |
|---|---|
| Performance rows read | 150,967,486 |
| Loans | 2,431,736 |
| Gap-spanning pairs dropped (D37) | 1,566 |
| Pairs with absorbing origin, suppressed (D35) | 851,739 |
| Candidate pairs | 147,682,445 |
| Excluded: train-split loan after the cut | 77,754,868 |
| Panel rows written | 69,927,577 |
| Role: fit | 40,385,651 |
| Role: condition | 10,107,285 |
| Role: evaluate | 19,434,641 |
| Rows flagged `is_skip` | 17,606 |
| Rows flagged `is_fall` | 131,508 |
| Rows flagged `post_gap` | 288 |
| Censored loans | 137,590 |

The arithmetic identity closes exactly, absorbing states never appear as an origin, and two
consecutive builds produced byte-identical output.

### Code review — findings

| # | Finding | Outcome |
|---|---|---|
| 1 | Censoring rows emit a transition whose label may not have been observed | Dismissed by diagnostic 17 (D44) |
| 2 | `run_length` computed per quarter file, correct only because no loan spans two files | Verified; add an assertion |
| 3 | Zero-row tally may mask gap-only loans | Verified zero, independently |
| 4 | `rate_spread` origination month assumed to be first payment minus one | Confirmed correct |

### Pre-registration scorecard

| Diagnostic | Yassine | Claude |
|---|---|---|
| 12 — effective date | Inside month t — correct | Non-trivial retroactive tail — wrong |
| 13 — gaps | Rare — correct | Servicing-transfer clustering, mild selection — wrong on both |
| 14 — skips | Hidden gaps — wrong | Correction rate under 20% — wrong |
| 15 — falls | Agreed with Claude — wrong | Counter resets, strong modification link — wrong |
| 17 — censoring | Real readings, under 20% identical — right call, wrong magnitude | 55–75% identical, leaning stale — wrong |

Agreed process change: calls are now split deliberately. Claude asks first and withholds its own
call until Yassine has committed.

### Open items

- **O12** — `run_length` containment assertion not yet added to `src/panel.py`.
- **O13** — Seasoned entrants: roughly 13.5% of sampled loans first appear after their first
  payment date, some by years. Their `loan_age` does not start near zero, which may matter when
  `loan_age` is used as a feature.
- **O14** — HPI join not yet built. `hpi_feature` ships as a null placeholder; the encoding is
  decided (D43) but the state HPI series is not yet in `data/macro/`.
- **O15** — Informative censoring, code 16: those loans are sold as they deteriorate, with final
  transitions mostly Current→DQ30 and DQ30→DQ60. The panel sees deterioration begin and never the
  outcome. Limitations note.
- **O16** — Unemployment series carries a non-numeric value for one month (2025-10), handled as
  null. 11,884 rows affected.
- **O17** — `dti` is 2.20% null (1,535,625 rows) from the 999 sentinel — the largest real feature
  gap.

### Scope, as cut for the 20 October deadline

- The stratified third arm is cut permanently, written up as scope rather than omission.
- The semi-Markov ablation grid is reduced from five runs to run-length only.
- Kept, because this is the paper: the two-regime evaluation (local vs chained), the compounding
  gap, and the honest-negative posture.

### Workflow (diagram nodes)

- Raw SFLLD, 2006–2007 vintages
- Diagnostics 01–17: rules closed on evidence
- Panel: `src/panel.py`, 69.9M rows *(← you are here)*
- HPI join: `est_ltv` + LTV
- Homogeneous baseline: row-normalised P
- Chained harness: h ∈ {6, 12, 24, 60}
- ML arm: per-row transition
- Semi-Markov ablation: run-length only
- Comparison: local vs chained
- Write-up: compounding analysis

### Next-session agenda

1. Add the `run_length` containment assertion (O12). One line.
2. Decide whether seasoned entrants (O13) need handling before the baseline or are a
   features-stage concern.
3. Scope the homogeneous baseline: count fit rows into the six-by-six matrix, normalise row-wise,
   handle the zero rows for absorbing states.
4. Scope the chained harness — predicted against realised state distribution at each horizon,
   Brier plus calibration curves.
5. First real numbers. The gap between local and chained scores is the headline result.

---

## Session Checkpoint — Data Layer Closed
**29 September 2026**

### Summary

This session closed the data layer. Four things happened: the loan-to-file containment guarantee
was moved into the build; a panel manifest and a validating loader were added; two diagnostics
settled the seasoned-entrant question; and a misdiagnosis on my part led to a wrong drop rule, its
detection, and a correct fix for `loan_age` on modified loans.

The panel is rebuilt and validated. The homogeneous baseline is next.

### 1. Decisions locked

#### 1.1 Baseline fit-row symmetry

Both arms are fit on `fit` rows only. `condition` rows initialise the chain — starting state and
run-length at the cut — for both arms, and are fitted on by neither.

Rationale (Yassine): a legitimate comparison requires both arms to use the same approach. Fitting
the baseline on additional rows would vary conditioning and sample size together, so the result
would no longer be attributable to conditioning alone.

Supporting observation: the argument for fitting the baseline on more rows is empirically void.
The thinnest cell (Current→DQ90+, ≈15,500 observations) is far from sampling-noise territory;
adding rows moves the matrix in the fourth decimal.

Robustness note for the write-up: **fit the matrix both ways and report the maximum absolute cell
difference.** One extra groupby; converts a defensible-by-argument choice into a
defensible-by-measurement one.

#### 1.2 Left-censoring of run length

Seasoned entrants make `run_length` understate true time-in-state: a loan Current for forty months
that entered the panel six months ago reads 6.

- **Flag required.** The flag must exist before any model consumes `run_length`, not merely before
  the semi-Markov ablation — `run_length` sits in the panel schema as a plain feature and the ML arm
  can read it earlier.
- `loan_age` is not censored by panel entry. It is computed from origination and is correct for
  every row. (Separately broken for modified loans; see §6.)
- In deployment a bank always knows true age and true run-length, so this censoring is an artifact
  of vendor reporting, not of the task. Measure and flag; do not engineer around it.

#### 1.3 Pre-registered prior on late entry

Yassine's prior: late purchase is administrative, not credit-selective. Recorded before diagnostic
19 was run. Outcome in §5.

### 2. Loan-to-file containment

#### 2.1 The invariant and its proof

`run_length` is computed per performance file. If a loan's rows spanned two files, the walk would
restart at the boundary and reset mid-loan.

`read_perf()` already enforces that each performance file carries exactly one loan-ID prefix,
accumulated over all chunks (verified at `panel.py:525--530`, checked at `606--611`; the prefix is
taken before the tail filter, which makes it stricter, not weaker). Given that:

> One prefix per file, enforced exhaustively + prefixes pairwise distinct across files ⇒ no loan
> can span files ⇒ `run_length` cannot reset mid-loan.

This is a derivation, not a proxy. Empirically confirmed by diagnostic 17 check B.1: 2,425,495
loans, zero spanning.

#### 2.2 Implementation

- Cross-file prefix distinctness check, placed in `build()` after the globs and before
  `macro_tables()`. Verified that nothing between the start of `build()` and the check touches disk.
- `perf_prefix()` reads one row per file (`nrows=1`), 0.06 s for all eight.
- Hard failure by default (`SystemExit`, consistent with the existing check);
  `--allow-multi-file-loans` downgrades to `RuntimeWarning`. Both an argparse flag and a `build()`
  keyword, since the module runs via `python -m`.
- Containment section in the build report: PASS/FAIL, files checked, distinct prefixes, per-file
  prefix.
- Three unit tests (`tests/test_prefix_check.py`, stdlib `unittest`); collision case uses three
  files and asserts the innocent file is not named.

Current data: eight files, F06Q1–F07Q4, all distinct, PASS.

#### 2.3 Known limitations

1. The up-front check is a necessary condition only. Sufficiency comes from `read_perf()` during
   the quarter's own build, by which point earlier quarters' parquet files (and possibly the macro
   cache) are written. The comment states this accurately.
2. A FAIL line only ever reaches the report under the override; without it the build stops before
   the report is written, deliberately, to avoid clobbering the last good report. The exception
   message is the record.
3. `SystemExit` is not caught by a bare `except Exception`. Correct for a `python -m` entry point;
   will need revisiting if `build()` is ever called from an orchestrator.

### 3. Manifest and validating loader

#### 3.1 Problem

Nothing in `src/` read the panel. The only reader was a diagnostic that globbed whatever files
existed and never checked the count. Since the build writes per quarter, any mid-build failure
leaves a partial set that a consumer would read silently.

#### 3.2 Design

A JSON sidecar rather than parsing the text report. The text report is a fixed-width human display
format; regexing it would break on a heading change or, worse, match nothing and validate against
`None`. Both artifacts are written from the same `Report` object in the same call, so they cannot
drift.

**Manifest** (`panel_manifest.json`, beside the parquet files): partition count, per-partition
filename and row count, column names and Arrow dtypes, build timestamp.

- Arrow schema, not pandas dtypes: the loader checks footers without reading data, and the footer
  speaks Arrow. `loan_age` is written as Arrow int32 but returns as pandas Int32.
- Schema taken via `pa.Schema.from_pandas(frame, preserve_index=False)` immediately before the
  write — the same conversion to parquet performs. Reading footers back would be a second source
  that could disagree.
- Partition count counts only quarters that produced a file. A zero-row quarter returns early
  without writing but still appears in `rep.quarters`. Commented so it is not simplified back.
- Deleted at the start of the build, immediately after the prefix check; written only on success.
  An old manifest outliving a failed build asserts a completeness that is not there.
- Build timestamp stamped into each parquet footer and recorded in the manifest.

**Loader** (`src/panel_io.py`): `load_panel(roles=None, columns=None, quarters=None)`.

- Validates partition count, filenames, per-partition row counts, column names, Arrow dtypes, and
  the build stamp on every partition. Row counts from footers; no data read.
- Raises on any mismatch, naming what disagreed. No override, no warning path — a mismatch means
  downstream numbers are wrong, and there is no legitimate workflow that proceeds.
- `quarters=` filters files (fewer files opened); `roles=` filters rows (all files opened, rows
  dropped after). Validation always covers the whole panel.
- `roles=` takes the panel's own vocabulary (fit/condition/evaluate). No train/test aliases: those
  are an interpretation that could change, while `fit` means what it always meant.

Validates the real panel in 0.47 s, footers only.

**Accepted consequence:** the build stamp makes partial rebuilds impossible. Rebuild one quarter
and the whole panel is rejected. This is the intended behaviour — a panel assembled from two runs
should not load quietly — but it means there is no patch-one-quarter workflow.

### 4. Diagnostic 18 — entry-age distribution

Entry age = min(`loan_age`) per loan. 2,425,495 loans, no nulls, no negatives.

| Entry age (months) | Loans | Share |
|---|---|---|
| 0 | 2,029,029 | 83.65% |
| 1 | 203,913 | 8.41% |
| 2 | 57,237 | 2.36% |
| 3 | 29,528 | 1.22% |
| 4–12 | 78,198 | 3.22% |
| >12 | 27,590 | 1.14% |
| Total | 2,425,495 | 100.00% |

Percentiles: p50 = 0, p75 = 0, p90 = 1, p95 = 3, p99 = 13, max = 159, mean = 0.62.

Shape: a spike with a thin tail, not a smooth spread. A two-group split is therefore the right
instrument.

Cut chosen: early = entry age 0–2 (94.42%, 2,290,179 loans); late = 3+ (5.58%, 135,316 loans). The
break after month 2 is where the distribution stops falling steeply. The alternative (0–1 vs 2+)
gives slightly more power on the late side and a less clean story; 135,316 loans is ample for
detecting a difference that matters.

Open at this point: 2,048 loans whose first-row `loan_age` was not their minimum. Folded into
diagnostic 19 rather than run separately.

### 5. Diagnostic 19 — group comparison

Full join to the origination file, no sampling: 2,431,736 origination records read, all 2,425,495
panel loans matched, zero unmatched in either group.

#### 5.1 Continuous characteristics

| Feature | Early mean | Late mean | Δ mean | Δ median |
|---|---|---|---|---|
| FICO | 723.83 | 733.64 | +9.81 | +13.00 |
| Original LTV | 70.74 | 70.44 | −0.31 | +1.00 |
| DTI | 36.63 | 35.43 | −1.20 | −1.00 |
| Rate spread | 0.030 | −0.113 | −0.143 | −0.147 |

No significance tests were run. At 2.4M loans every difference is significant and the statistic
decides nothing; magnitude is the only informative quantity.

Three of four measures point the same way: late entrants are modestly better credits. Not
dramatic, but four independent measures broadly agreeing is not noise.

#### 5.2 Channel and seller

| Channel | Early share | Late share | Δ (pp) |
|---|---|---|---|
| T | 56.91% | 44.73% | −12.18 |
| R | 43.06% | 54.61% | +11.56 |
| C | 0.02% | 0.37% | +0.35 |
| B | 0.02% | 0.29% | +0.27 |

| Seller | Early share | Late share |
|---|---|---|
| Bank of America | 3.76% | 48.86% |
| ABN AMRO | 4.59% | 14.89% |
| Wells Fargo | 25.78% | 7.69% |
| Countrywide | 12.58% | 1.48% |
| Top-10 concentration | 87.79% | 97.49% |

#### 5.3 Reading

My predicted mechanism was wrong. I expected a correspondent-channel routing delay; late entrants
are in fact more retail (54.6% vs 43.1%).

The actual mechanism is seller concentration — Bank of America at 48.9% of the late group against
3.8% of the early one. That is seasoned pool sales by specific institutions, the possibility I
raised earlier and then dismissed as thin.

Verdict: the middle outcome of the three pre-registered. The cause is administrative (specific
sellers, specific books); the consequence is a mild quality difference. Yassine's prior was half
right.

**Consequence for the panel:** early-age months missing from the data belong to slightly better
loans, so young loans are drawn from a marginally worse-than-representative sample. The bias is
small and conservative — it makes young loans look slightly riskier than they are. Limitations
paragraph, not a design change.

#### 5.4 The 2,048 anomalies

Checked in three orderings — panel as stored, source as stored, source re-sorted by period. In all
three, steps where period does not increase: zero.

`run_length` walks rows in period order and depends on period and state, not age. Since period
ordering is clean everywhere, `run_length` is unaffected. The worry that motivated the check is
dead.

But `loan_age` itself goes backwards: 2,589 backward steps in source period order, largest 229
months, present in the source files. This became §6.

### 6. The R7 misdiagnosis

#### 6.1 My error

I read backward `loan_age` as data corruption and commissioned a rule dropping any loan whose
`loan_age` decreases between consecutive months. I did not ask what else could produce a backward
age. The obvious answer — modification restarts the clock — was available and I did not look for
it.

#### 6.2 Detection

The agent implemented the rule, observed the result diverge from the brief's stated expectation,
and stopped rather than proceeding:

| | Expected | Actual |
|---|---|---|
| Loans dropped | 2,048 | 135,857 |
| Share of loans | 0.08% | 5.59% |
| Rows removed | ≈252,000 | 18.3M |

Evidence from 2006Q1: of 14,906 loans with a modification flag, 14,894 were dropped. On reset rows,
18,292 carried flag Y, 11 P, 8 blank. Only 11 loans had a backward step with no flag. Resets
clustered in 2009–2011.

Diagnostic 19 had caught only resets falling below the loan's first panel age; the rule caught
every reset.

#### 6.3 Why the rule was harmful

A modified loan is a distressed loan. Dropping them removes 5.6% of loans concentrated in the worst
credits, clustered in the 2009–2011 test window. Default rates would come out too low and both arms
would look better than they are — a worse outcome than the problem I thought I was fixing.

The tempting narrow fix (ignore backward steps on flagged rows) was also rejected: it repairs the
drop but leaves the feature broken, with a five-year-old distressed loan reporting age 4.
Confidently wrong in the direction that matters most.

**Resolution:** R7 removed, panel rebuilt, manifest and stamps retained.

### 7. Diagnostic 20 — modification resets

| | |
|---|---|
| Loans with at least one reset | 135,857 (5.59%) |
| Total resets | 166,427 |
| Loans with Y or P on any row | 135,929 |
| Modified loans that never reset | 94 |
| Loans with a reset and no flag anywhere | 22 |
| Repeat resetters | 25,246 (18.6%), up to 6 |

Flag on the reset row: Y 99.90%, P 0.07%, blank 0.03% (52). Median drop 46 months (p10 22, p90 94,
max 239). Age after reset is 1 in 57.4% of cases, 0–3 in 86.4%. 166,375 of 166,427 resets occur
with the period advancing by exactly one month. Timing: from 2008, peaking 2010 (23.6%) and 2011
(16.3%), long tail to 2026.

#### 7.1 State distribution — why this mattered

| Origin state | Unmodified | Modified | Before flag | At/after flag |
|---|---|---|---|---|
| Current | 96.86% | 78.21% | 79.83% | 75.00% |
| DQ90+ | 1.37% | 9.99% | 8.74% | 12.49% |

Modified loans supply 5.99M panel rows, 2.0M at or after the first flag. DQ90+ is nine times more
frequent among modified loans. The loans whose age was wrong are precisely the loans doing the
interesting thing — so leaving the field broken would have taught the model that young loans are
risky, from an artifact.

#### 7.2 The 22 unflagged loans

One reset each; remaining-months-to-maturity null on the row before the reset in all 22 and
populated on the reset row; interest rate falls in 59% (vs 73% for modified loans); median drop 46
months, same as flagged; all 2010–2015. Shape of a data-entry path rather than a different event.
Treated as modifications.

### 8. Final `loan_age` design

**Chosen:** true age from origination, plus a modification flag, plus months-since-modification.

Each piece earns its place: true age restores real seasoning; the flag marks the distress and will
be among the strongest features (the state table says so before anything is fit);
months-since-modification captures the recovery path, since a loan modified last month differs from
one modified three years ago that has been paying since.

Rejected alternative — keep the servicer's reset age, on the argument that a modification is
legally a new contract. The borrower, property and payment history are unchanged, and the chain
models borrower behaviour, not contract identity.

#### 8.1 Coverage check (run before any change)

No origination date in the origination file; first payment date is present for all 2,431,736
records, every performance loan matches one, including all 22 unflagged resetters.

`loan_age = period − (first payment month − 1)`

Matches Freddie's `loan_age` on 140,679,281 of 140,683,885 pre-reset rows (99.997%). The same
origination-month convention `rate_spread` already uses.

#### 8.2 Modification event definition (rule R7, rewritten)

On 2006Q1, Y marks the modification month and following rows carry P. A modification event is
therefore: the loan's first Y or P row; the start of every later run of Y rows; and, for loans with
no flag anywhere, each row where the age goes backwards. A reset on a flagged loan whose row carries
P or blank does not start a new count (≈50 resets).

`months_since_modification` counts calendar months and restarts at every repeat modification —
deliberate, and commented as such.

#### 8.3 Rebuild result

| | |
|---|---|
| Panel rows | 69,927,577 |
| Loans with Y or P on any row | 135,929 (5.59%) |
| Loans modified by age reset only | 22 (listed by ID) |
| Rows at/after first modification | 2,001,300 (2.86%) |
| Rows whose `loan_age` changed | 2,000,223 |
| Median correction | +52 months |
| Rows with negative `loan_age` | 172 |

Identity balances. Build-time assertion that age never decreases within a loan passed on all 70M
rows. Manifest picked up `is_modified` (bool) and `months_since_modification` (int32)
automatically, since it is built from the schema actually written. 18 tests pass.

Both new columns sit in the features group. Both are known at month t and neither leaks the
outcome; whether the model uses them is a modelling decision.

### 9. Open items

1. **The +33 mismatches.** 4,604 rows across 2,692 loans where the formula disagrees with Freddie
   by −1, +1 or +33 months. Off-by-one is a boundary artifact; +33 is not, and implies a disagreement
   about when the loan started rather than how to count. Cheap to split by bucket. Do before the
   write-up, not before the baseline.
2. **172 negative `loan_age` rows.** Months reported before the origination month as defined. Left
   negative rather than clipped — clipping would invent a value. One line in the write-up.
3. **Loan-count discrepancy.** Diagnostics 18 and 19 report 2,425,495 panel loans; the post-rebuild
   report states 2,431,736, which is the origination record count. A 6,241 gap. Either a
   mis-statement or a real change. Worth one check before the baseline is counted, since the
   baseline divides by these totals.
4. **The monotonic-age unit test is weak** — it tests the formula on synthetic numbers. The real
   protection is the build-time check over 70M rows. Known limitation, accepted.
5. Nulls-table formatting in the build report is misaligned for the long
   `months_since_modification` name. Fixed in code, appears after the next build.
6. **`run_length` censoring flag still to be added.** Deadline: before any model consumes
   `run_length`.
7. Nothing committed as of session end. Diagnostic 20 should be committed as the evidence behind
   the `loan_age` design.

### 10. Remaining work

1. **Homogeneous baseline** — count fit-row transitions into the 6 × 6, normalise row-wise. First
   real numbers.
2. **Chained harness** — propagate to h ∈ {6, 12, 24, 60}, compare against realised distributions.
   Shared machinery; the ML arm plugs in.
3. **ML arm** — multinomial classifier, per-borrower-month rows, chained forward identically.
4. **Headline comparison** — local vs chained regimes. The gap is the result; either direction is
   publishable.
5. **Semi-Markov ablation** — time-in-state. Where the censoring flag finally matters.
6. **Macro scenarios** — base/adverse/severe, probability-weighted, IFRS 9 style.
7. **Engineering layer** — see below.
8. **Write-up** — compounding analysis and ablations.

Items 1 and 2 are the spine; everything after compares against them. Target: both inside the next
week, of roughly three remaining.

### Engineering layer — priority under time pressure

Yassine's position: optional, since this is a research project. Partially contested:

- **Drift monitor — keep.** Row-wise KL/TV between the current estimated matrix and a reference is
  a finding, not a deployment feature: it extends the compounding analysis by showing the matrix is
  not stable and where it moves. Belongs in the research even if nothing is served.
- **Minimal FastAPI endpoint — keep.** Roughly a day. The 60-month fan chart with both arms side by
  side is the demo, and the ships-things signal for ML engineering roles.
- **Retrain loop and containerisation — drop without regret** if time runs short. Checkbox items
  nobody will run.

### Process note

Two prompts this session ended in the agent stopping rather than proceeding — once because the
brief assumed a write site that did not exist, once because the drop rule's premise was false. Both
stops were correct and both saved real work. The instruction to stop and describe the obstacle
rather than invent a mechanism should stay in every brief.

---

## Session Checkpoint — 30 September 2026

### 1.1 Milestone 0 — resolution of the panel coverage gap

**Statement.** The origination files for the 2006–2007 vintages contain 2,431,736 loans; the
transition panel contains 2,425,495 distinct loan identifiers. The residual of 6,241 loans was
resolved in diagnostic 21. No loan is unaccounted for: the UNEXPLAINED bucket is empty, as are the
R1 (no live pair) and R3 (no consecutive pair) causes.

**Mechanism 1 — single-observation loans (n = 2,029, 32.5%).** Every loan in this bucket carries
exactly one performance row, a zero-balance code, and delinquency status 00. Composition: 1,987
prepaid (code 01), 41 censored (codes 15/16/96), 1 default-type (code 03). The no-code sub-bucket is
empty. Under the terminal-row convention the emitted transition runs from the state at t − 1 into
the absorbing state at t; these loans have no month t − 1, so the origin is unobserved and no pair
is emitted. This is the observed-label rule acting on the origin rather than the destination.

**Mechanism 2 — post-cut train loans (n = 4,212, 67.5%).** All are cause R6: train-split loans whose
live pairs fall entirely after the cut date of 2008-12. Under the grouped-temporal design train
loans contribute pre-cut months only, so a train loan first observed in 2009 or later contributes
nothing. These are seasoned entrants: the year of first observation is 51.9% 2009, 22.6% 2010, 12.4%
2017, 7.3% 2018 — a wave structure consistent with servicer onboarding rather than credit selection.

**Credit-event exposure.** 101 gap loans carry a default-type code or a non-numeric status on some
row (≈ 0.05% of credit events in the sample). All sit in the R6 bucket and were therefore outside
the fit window by construction. No credit information is lost by the gap.

**File-boundary check.** Zero loans span more than one quarterly file; zero loans appear outside
their own vintage's file. Per-file ratios of gap share to origination share range 0.74–1.28, within
the [0.67, 1.50] acceptance band. No processing artefact.

**Consequence for the baseline.** None. The transition matrix normalises by counts of origin
loan-months, not by loan counts; no loan total enters any cell. The gap affects only descriptive
statistics in the write-up.

### 1.2 Milestone 1 — homogeneous transition matrix

**Construction.** Estimated from `fit` rows only (n = 40,385,651; 57.75% of the panel). Rows are
origin state, columns destination. Each estimated row is normalised by its own total, where the
denominator counts origin loan-months. No smoothing. The Prepaid and Default rows are set by
construction as identity rows; the panel emits no pair from an absorbing origin, so they carry zero
observations and are not estimated. Counts are the stored primary artefact; probabilities are
derived on read.

**Table 1: Pooled homogeneous transition matrix, fit rows only. Counts above, row shares below.
Absorbing rows omitted (identity by construction).**

| origin | Current | DQ30 | DQ60 | DQ90+ | Prepaid | Default | row total |
|---|---|---|---|---|---|---|---|
| Current | 38,865,684 | 337,340 | 2,850 | 1,438 | 324,716 | 74 | 39,532,102 |
| *p* | 98.3142% | 0.8533% | 0.0072% | 0.0036% | 0.8214% | 0.0002% | |
| DQ30 | 214,231 | 205,466 | 100,572 | 844 | 5,341 | 50 | 526,504 |
| *p* | 40.6893% | 39.0246% | 19.1018% | 0.1603% | 1.0144% | 0.0095% | |
| DQ60 | 16,938 | 18,508 | 31,138 | 56,995 | 996 | 67 | 124,642 |
| *p* | 13.5893% | 14.8489% | 24.9819% | 45.7270% | 0.7991% | 0.0538% | |
| DQ90+ | 12,409 | 2,651 | 3,914 | 175,346 | 1,090 | 6,993 | 202,403 |
| *p* | 6.1308% | 1.3098% | 1.9338% | 86.6321% | 0.5385% | 3.4550% | |

**Structural-zero hypothesis — rejected.** We pre-registered that the delinquency counter advances
at most one step per month, implying exact zeros in Current→DQ60, Current→DQ90+ and DQ30→DQ90+. All
three fail: 2,850 (0.0072%), 1,438 (0.0036%) and 844 (0.1603%) respectively. The counts are reported
as observed and no correction is applied. Magnitudes are consistent with a contamination rate rather
than a behavioural channel; candidate mechanisms are servicing transfers with restated delinquency
history, payment reversals, and boarding artefacts on seasoned entrants. The write-up owes one
paragraph on this.

**Smoothing — not required.** No estimated row contains a zero cell. Smallest cells:
Current→Default (74), DQ30→Default (50), DQ60→Default (67). The bias trade-off between hard zeros
and smoothing leakage into absorbing states is therefore moot in this sample and no constant is
introduced.

**Design note.** Homogeneity is retained deliberately. The classical chain is the object under test;
a time-varying baseline would absorb part of the treatment arm's contribution and render the
comparison uninterpretable. Time instability is measured as a diagnostic and reported, never
corrected.

### 1.3 Milestone 1b — age versus calendar decomposition

**Motivation.** The yearly instability tables show the DQ30 cure rate (DQ30→Current) falling from
59.29% (2006) to 35.26% (2008), a 24.0 point decline. Two explanations are confounded: the credit
cycle, and a shift in the age composition of the fit pool. Diagnostic 22 separates them. Loan age is
defined as `origin_period` minus first payment date, in calendar months.

**Decomposition.** Holding age fixed at 0–11, the cure rate falls 59.29 → 39.99, a decline of 19.30
points against a pooled decline of 24.03. **Calendar accounts for 80.3% of the movement, age mix for
the remainder.**

**Table 2: DQ30 cure rate, age held versus calendar varying.** Dashes are structurally absent: the
2006–2007 origination window admits no loan of that age in that year.

| age bucket | 2006 | 2007 | 2008 |
|---|---|---|---|
| 0–11 | 59.29% (19,318/32,581) | 51.82% (52,396/101,103) | 39.99% (34,707/86,792) |
| 12–23 | — | 42.46% (23,525/55,411) | 34.84% (61,885/177,603) |
| 24–35 | — | — | 30.66% (22,382/72,989) |
| pooled | 59.29% | 48.51% | 35.26% |

**Corroboration on the worsening cell.** DQ30→DQ60 at fixed age 0–11 rises 11.18% → 20.97% (+9.79
points) against a pooled rise of 10.61 points: 92.3% calendar. Both directions of the DQ30 gate
agree, and the worsening channel is close to purely cyclical.

**Age is secondary but real.** Within 2008 the cure rate declines monotonically across age buckets,
39.99 → 34.84 → 30.66, roughly five points per year of seasoning. This is direct evidence that loan
age belongs in the conditioning set and constitutes a preview of the semi-Markov argument: if age at
the loan level moves the cure rate by five points per year, time-in-state plausibly does likewise.

**Asymmetry.** Within 2008, cure spans 9.3 points across age buckets while worsening spans only 2.2.
Seasoning erodes a delinquent borrower's capacity to recover without accelerating the deterioration
process. This is a substantive behavioural claim and is reported as such.

**Secondary cells.** Current→DQ30 is 64% calendar and 36% age, with a clean monotone ramp in 2008
(0.86% → 1.03% → 1.20%); the age objection would have been materially more damaging had the
instability claim been framed on this cell. Current→Prepaid is 88% calendar and shows no monotone
seasoning ramp (0.97%, 1.08%, 0.95% in 2008), refuting the prepayment-seasoning explanation.

**Residual vintage confound.** Holding age fixed does not hold vintage fixed: a loan aged 0–11 in
2008 originated in late 2007, one aged 0–11 in 2006 originated in 2006. However, within 2008 vintage
and age run counter to one another — the 24–35 bucket is the 2006 vintage, the 0–11 bucket late 2007
— and the weaker 2007 book cures better by nine points. The age gradient is therefore age, not
vintage quality. The confound survives only for across-year comparisons and is not pursued: the
treatment arm takes origination features as covariates, so both channels are modelled regardless.

**Build notes.** 1,623,117 fit rows (4.02%) carry age −1, the origination month paired to the first
payment month; almost all are Current-origin. The panel's `loan_age` column equals this definition
plus one for 100% of rows. Open item: the convention must be fixed before any feature work consumes
it.

### 1.4 Milestone 2 — chained forecast harness

**Architecture.** `src/scoring.py` is model-agnostic: it consumes predicted distributions, realised
states and cohort membership, and imports neither the baseline nor any transition matrix. This was a
deliberate constraint so that the baseline, the null references and the treatment arm are scored by
identical code; a divergence there would invalidate the headline comparison.

**Eligibility.** Determined solely by state at the cut month (2008-12); no condition on subsequent
trajectory. Of 486,185 test-split loans: 77,095 excluded as already absorbed (Prepaid 75,403,
Default 1,692), 1,104 entering after the cut, 424 with follow-up ending before it, 7 with a
reporting hole at it. Eligible: 407,555 (83.83%), distributed Current 95.15%, DQ30 2.31%, DQ60
0.83%, DQ90+ 1.71%.

**Absorption versus censoring.** Absorption during the horizon is an outcome, never a dropout: a
loan prepaying in 2009-03 occupies state Prepaid at every subsequent horizon and is scored there.
Conditioning on survival would delete the outcomes the model exists to predict — selection on
outcome, excluded on principle. Censoring (loss of observability without absorption) removes the loan
from that horizon onward only.

**Cohorts.** The varying cohort admits all loans observable at horizon h; the fixed cohort admits
only loans observable at all horizons including 60. The pair guards against confounding horizon
effects with cohort attrition. Empirically the guard does not bind: censoring reaches only 0.74% at
h = 60, the two cohorts differ by 55 loans out of 404,541, and all metrics agree to three decimal
places. This is reported rather than omitted — a checked-and-non-binding hazard is a stronger
statement than silence.

**Table 3: Headline result.** Baseline default share, predicted versus realised, varying cohort. One
matrix, one set of loans; the only variable is the number of multiplications.

| h | predicted | realised | captured | gap (pp) |
|---|---|---|---|---|
| 6 | 0.39% | 0.38% | 103% | +0.01 |
| 12 | 0.74% | 0.97% | 76% | −0.23 |
| 24 | 1.34% | 2.80% | 48% | −1.46 |
| 60 | 2.79% | 7.45% | 37% | −4.66 |

**Direction and mechanism.** The baseline is optimistic, as pre-registered. The mechanism is stronger
than anticipated: both absorbing states are under-predicted. Prepaid is forecast at 38.33% against
67.21% realised; Current is over-predicted by 35.4 points. The hypothesised counterweight — that
under-predicted prepayment retains loans in the chain and inflates defaults — does not materialise.
The deterioration shortfall is severe enough that defaults fall short by a factor of 2.7 despite an
inflated at-risk population.

**Table 4: Default probability at h = 60 by state at the cut.**

| state at cut | predicted | realised | ratio |
|---|---|---|---|
| Current | 2.09% | 5.67% | 2.7× |
| DQ30 | 7.84% | 27.08% | 3.5× |
| DQ60 | 20.45% | 43.93% | 2.1× |
| DQ90+ | 30.33% | 70.35% | 2.3× |

DQ30 is the worst-served state by a clear margin, closing the loop with the instability diagnostic:
the cell that moves most across calendar time is the cell where the frozen matrix fails hardest. Two
independent analyses converge on the same state.

**Discrimination versus calibration.** The ordering of risk is preserved at every horizon in both
prediction and realisation; the magnitudes are uniformly low by a factor of two to three. The
homogeneous matrix emits exactly four distinct predictions across 407,555 loans — every Current loan
receives an identical forecast irrespective of FICO, LTV or DTI. The treatment arm's task is
therefore not to repair a broken ranking but to (i) correct the level and (ii) convert four
predictions into a continuum.

### 1.5 Milestone 2b/2c — reference models

**Definitions.** *Persistence:* the identity matrix, chained through the baseline's own code path;
every loan retains its state at the cut. *Marginal:* the empirical distribution of `next_state`
across all fit rows, applied identically to every loan at every horizon, unchained. *Oracle:* the
realised state distribution of each cohort at each horizon, assigned identically to every loan —
constructed from test-period outcomes and therefore not achievable at forecast time.

**Table 5: Multiclass Brier, varying cohort.** Lower is better. The fixed cohort agrees within 0.5
points throughout. The oracle is not achievable at forecast time.

| h | persistence | marginal | baseline | oracle |
|---|---|---|---|---|
| 6 | 0.4011 | 0.3956 | 0.3463 | 0.3478 |
| 12 | 0.6247 | 0.6047 | 0.5127 | 0.4786 |
| 24 | 0.9442 | 0.8994 | 0.6693 | 0.5812 |
| 60 | 1.5875 | 1.5133 | 0.6995 | 0.4965 |

**Correction to an earlier claim.** The rise in baseline Brier from 0.346 to 0.699 was initially
advanced as evidence of chained failure. It is not. Persistence rises from 0.401 to 1.588 while doing
no work at all; most of the increase reflects the prediction task becoming harder as outcomes
disperse. A rising Brier score is not by itself evidence of compounding, and the claim is withdrawn.

**Table 6: Normalised compounding curve:** baseline excess over the oracle reference, varying cohort.

| h | oracle | baseline | excess | % of oracle |
|---|---|---|---|---|
| 6 | 0.3478 | 0.3463 | −0.0015 | −0.43% |
| 12 | 0.4786 | 0.5127 | +0.0341 | 7.12% |
| 24 | 0.5812 | 0.6693 | +0.0881 | 15.16% |
| 60 | 0.4965 | 0.6995 | +0.2029 | 40.87% |

The divergence is monotone in h. The oracle's own score falls between h = 24 and h = 60, because by
60 months two-thirds of the book has prepaid and outcomes are less dispersed: the task becomes easier
while the baseline becomes worse. This is a sharper statement of the compounding finding than the raw
Brier trajectory supports.

**Terminological correction.** The oracle was initially described as a floor. It is not one for the
baseline: the baseline uses each loan's state at the cut, which the oracle does not, and at h = 6 the
baseline consequently scores 0.43% better. The oracle is a floor only for models ignoring loan-level
information. It is a reference level, and the write-up must use that term.

**Limitation of the marginal null.** As specified it applies a one-month marginal unchained at every
horizon, predicting 0.02% default against 7.4% realised at h = 60. It is informative at h = 6 and
decorative thereafter. The defect is in the specification, not the implementation.

**Open item — conditional oracle.** The correct floor for the baseline is the realised distribution
per starting state within each cohort and horizon. It supplies the oracle with exactly the
information the baseline holds, so any residual gap is unambiguously miscalibration. It also bounds
what is reachable from the starting state alone: the treatment arm must beat it, and doing so
requires borrower-level covariates performing real work. Both oracles are to be retained — the gap
between them measures what the starting state alone purchases, measured rather than argued. Scheduled
for the start of the next session.

### 1.6 Status and carried items

**Closed.** Panel coverage gap; homogeneous baseline; age/calendar decomposition; chained harness;
persistence, marginal and unconditional oracle references.

**Open.**

1. Conditional oracle (per starting state) — next session, defines the treatment arm's target.
2. `loan_age` off-by-one between the panel column and the diagnostic definition — settle before
   feature work.
3. Paragraph in the write-up explaining multi-step delinquency jumps.
4. Unit tests on `count_transitions` and `probability_matrix`: a known count matrix normalising to
   known probabilities, and a zero row raising rather than dividing. These functions are reused by the
   treatment arm and the drift monitor.
---

## Session Checkpoint — 5 October 2026 (panel rebuild, property state, ML arm design)

### Where we are

The data layer is now final apart from two additions. The panel was rebuilt twice today: once to add
`property_state`, once to rename the transition columns and harden origination validation. Five clean
milestone commits sit on main. The ML arm is designed but cannot start until the state house-price
join and the `run_length` censoring flag exist. Deadline: 20 October.

### Workflow (diagram nodes)

- Raw Freddie Mac files: origination + monthly performance
- Transition panel (rebuilt today): property state, from/to rename, hard checks
- Homogeneous baseline: one matrix, fit rows only
- Chained harness + null references: persistence, marginal, two oracles
- Feature layer (next session): state HPI join, `est_ltv`, `run_length` flag
- Forecaster: builds future months itself; frozen / foresight switch
- ML arm: both-frozen first, then borrower-only, macro/time-frozen; foresight if time
- Write-up: compounding gap, ablations, limitations

### Decisions closed this session

#### ML arm design

- **Two independent axes, not one.** What the model sees (borrower, macro/time, both) and what it may
  know about the future (frozen, foresight). Borrower-only has no macro columns, so it is the same
  under both regimes. Five distinct runs in total.
- **Frozen is the headline; foresight is the ceiling.** Frozen holds macro at its December 2008 values
  for every future month. Yassine's fairness rule: the baseline matrix is estimated once and never
  updated, so freezing puts the ML arm under the same constraint.
- **Foresight is compared only against frozen, never against the baseline.** The baseline has no macro
  channel and cannot be given foresight; the gap between frozen and foresight measures what macro
  knowledge is worth.
- **Consequence of freezing:** macro becomes one constant value for every loan and every month, so the
  macro/time-frozen run reduces to a loan-age-only model. That run becomes the evidence for this, rather
  than an assertion.
- **Priority:** both-frozen first (the headline), then borrower-only, then macro/time-frozen. Foresight
  runs only if time remains.
- **The frozen/foresight switch lives in the forecaster, not the panel** (Yassine's rule: the model
  builds the future, it never reads it). The panel keeps one, realised, version of the macro columns. The
  forecaster constructs every time-varying feature for future months: `loan_age`, `est_ltv`, macro,
  `run_length`.

#### Geography

- **National house prices rejected; state index required.** A national index gives every loan the same
  value in a given month, so a national-based `est_ltv` would carry calendar time into the borrower-only
  arm and break the ablation. Yassine judged the extra rebuild worth a day.
- **`property_state` is a feature**, by the schema's own rule: metadata describes how the panel was
  built, property state describes the loan. Stored as text (shared dictionary across all partitions), not
  as integer codes. MSA and postal code are parked as metadata.
- **Territories kept.** GU, PR and VI loans stay in the panel; `est_ltv` will be null for them because
  the state index does not cover them.

#### Data hygiene

- **Rename:** `origin_period`, `origin_state`, `next_state` became `from_period`, `from_state`,
  `to_state`. Loan-origination names (`orig`) unchanged. Historical report files deliberately keep the old
  names.
- **Hard checks replace silent storage.** LTV 6 to 105 and DTI 0 to 65 (page 6 and 5 of the User Guide),
  sentinel 999 becomes null; rate must be above 0 and at most 25; property state must be one of 54
  whitelisted codes; postal code 000 becomes null. Anything else stops the build. A blank property state is
  now a hard failure.
- **Malformed loan F06Q30148547 excluded.** Its one panel row was a Current to Current fit row, so the
  baseline matrix moved by one count and Brier scores by at most 1e-8. Fingerprint unchanged. Keeping a
  corrupt row to preserve bit-identity was rejected.
- **Commits:** Yassine commits himself; no `Co-Authored-By` trailer.

### Repository state

| Commit | Content |
|---|---|
| 69a7f00 | Conditional oracle and per-start-state reporting |
| 87b2a30 | `property_state`, `msa` and `postal_code` added to the panel |
| 8251fc9 | Diagnostic 23: origination validity and territory coverage |
| eac284c | Rename origin/next to from/to |
| 731f8ba | Hardened validation; malformed loan excluded |

All five sit on top of 5490831 (1 October). 32 tests pass. Panel: 69,927,576 rows; fingerprint
`3b1209189824e42f`. To confirm: worktree and branch removed, stash dropped, push done.

### Data facts learned

- 54 property-state codes: 50 states, DC, GU, PR, VI. Top by loans: CA, FL, TX, IL, PA.
- Territories: 10,073 loans (0.42%), 91% of them PR. In the harness, 1,810 of 407,555 eligible loans
  (0.44%).
- Malformed origination records: 2 of 2,431,736 loans. Isolated, not a class.
- Sentinels never reach a feature as a number; all become null. FICO not available: 2,007 loans. LTV: 219
  loans.
- DTI not available: 54,257 loans, 1,535,625 rows (2.2%). Hypothesis to test: missing DTI marks thinly
  documented 2006–2007 loans and may itself predict default. Do not impute with an average.
- MSA is null on 18.9% of rows, rising from 16.1% (2006Q1) to 23.3% (2007Q4). Most likely means rural, not
  missing.
- Cached macro files are all national: Case-Shiller house prices (monthly), unemployment (monthly),
  30-year mortgage rate (weekly). Mortgage rate is mainly a prepayment driver.

### Corrections

- **Harness headline numbers.** Baseline default capture at 6 months is 101.83% (varying cohort) and
  94.15% (fixed cohort), not the 103% quoted at the start of the session. At 60 months: 37.50%. The claim
  that Current loans carry 96% of the excess at 60 months is unverified. Rule going forward: no number
  enters the write-up unless checked against `reports/` or `artifacts/`.
- The rebuild for property state was a full rebuild, not the cheap augmentation pass first estimated.
- The harness check in the hardening prompt contradicted itself: the fingerprint covers test-split loans
  only, so it could not guard a dropped fit row.
- The `run_length` censoring flag was planned for today's rebuild but left out of both prompts. Still
  open.
- The commit-splitting prompt had the agent create git worktrees, which conflicts with the standing rule
  that agents never run git commands.

### Pre-registration scorecard

| Question | Yassine | Claude | Result |
|---|---|---|---|
| GU + PR + VI loan count | about 2,000 | 5,000 to 15,000 | 10,073 (Claude) |

### Open items

**Before the ML arm**

1. **State HPI join.** Download the FHFA state index (quarterly). Decide how quarters become months
   (hold each quarter's value across its three months was proposed). Attach by state and month, compute
   `est_ltv`.
2. **`run_length` censoring flag.** Marks loans that enter the data late. Needs a panel rebuild; fold the
   small cleanups below into it.
3. **Macro features for the forecaster.** Frozen construction for unemployment and house prices; decide
   whether mortgage rate becomes a feature.

**Can wait**

- Diagnostic 21 expects 6,241 and now sees 6,242: update with a comment.
- The build overwrites `diagnostics/16_panel_build_report.txt`; give the build its own output location.
  Today's run went to report 24.
- FICO still silently nulls values outside 300 to 850; harden like LTV.
- Diagnostics 18 and 19 were already broken; do not cite until fixed.
- Unit tests for `count_transitions` and `probability_matrix`; the `loan_age` off-by-one.
- Quick check: does default rate differ for missing-DTI loans?
- README line noting the 5 October rename.

**Still to decide**

- Is `est_ltv` placed in the `both` arm only (proposed: it is a borrower feature crossed with a macro
  path) or also in borrower-only?
- `property_state` encoding, and whether to run it in and out of the borrower-only arm as an ablation.

### Questions carried into next session

1. **`run_length` in a probabilistic forecast.** The forecast says "70% Current, 20% DQ30, 10% Prepaid"
   rather than one state. In month 6, what is a loan's `run_length`?
2. If frozen beats the baseline by a little and foresight beats it by a lot, what does that gap tell us?

---

## Session Checkpoint — Run-length: lattice cost, caps, destination mix
**5 October 2026**

### What this session closed

Run-length is closed end to end: the forecasting cost, the ladder caps, and the measurement that
justifies keeping the semi-Markov ablation at all.

### Workflow (diagram nodes)

- Run-length makes "Current" many nodes: the forecaster branches each month
- Paths recombine on (state, run-length) ⇒ lattice, not tree; growth linear; chaining exact
- Cost asymmetry: headline arms need 4 rows/month; only the semi-Markov ablation pays the expansion
- Three false alarms removed: 3 macro paths → 1 frozen path; PC not at risk; hardware is not a defensible
  reason to cut
- Engineering: capped ladder, loan-batch checkpointing, CPU not GPU
- Diagnostic 24 — pooled exit hazard by rung; caps: Current 27, DQ30 5, DQ60 3, DQ90+ 10 (all mass-bound)
- Pooled exits hide the destination split
- Diagnostic 25 — same table, exits split by destination
- Markov assumption fails, measured: DQ90+ inverts 25:1 cure → 3:1 default. Caps unchanged; flatness rule
  retired

### Decisions taken

- Lattice propagation is exact; no Monte Carlo sampling needed.
- Run-length ladder capped per state, cap set empirically, not chosen in advance.
- Headline comparison runs on one frozen macro path; scenario weighting belongs to the IFRS 9 serving
  story only.
- Checkpointing by loan batch — loans are independent, so a batch runs to horizon 60, writes its answer
  rows, and is marked done. Batch size is also the RAM knob.
- Training stays on CPU. Six-class gradient boosting will not fit 4 GB at 40M fit rows, and prediction
  dominates the runtime anyway.
- Ablation kept, not cut. If it is ever cut, the stated reason is the 20 October deadline — never
  hardware.

### Caps

| State | Cap | Bound by | Flatness rule |
|---|---|---|---|
| Current | 27 | mass | rung 3 |
| DQ30 | 5 | mass | never satisfied |
| DQ60 | 3 | mass | never satisfied |
| DQ90+ | 10 | mass | rung 3 |

Max 45 live nodes. Fit rows cannot show a run longer than the 36-month fit window, so the caps also stop
the forecaster manufacturing run-lengths the model has never seen.

### Destination mix — the result

| | rung 1 | last rung | change (pp) |
|---|---|---|---|
| **Current (rung 35)** | | | |
| to DQ30 | 3.05 | 0.55 | −2.49 |
| to Prepaid | 0.32 | 1.78 | +1.46 |
| **DQ90+ (rung 13)** | | | |
| to Current | 8.70 | 1.89 | −6.81 |
| to Default | 0.34 | 6.90 | +6.57 |
| **DQ30 (rung 10)** | | | |
| to Current | 49.49 | 11.04 | −38.45 |
| to DQ60 | 19.23 | 11.04 | −8.19 |

Current and DQ90+ invert in composition while the pooled hazard sits roughly still. DQ30 and DQ60 do not
invert — they seize up, every exit route shrinking together. The baseline has one row per state and must
average a fresh curer with a stuck loan, so it overprices fresh arrivals and underprices chronic ones, in
the state that drives ECL.

### Admitted limitation

The DQ90+ cap of 10 prices every longer spell at the rung-10 rate. The default share has roughly
plateaued by then (7.11% at rung 10, 6.90% at rung 13), so it is defensible — but it is stated plainly,
not buried.

### Pre-registration scorecard

Diagnostic 24: both called Current flattening late and DQ90+ never flattening. Result was the exact
inverse. Third consecutive joint miss.

Diagnostic 25: Claude called the Current mix moving and was right.

### Parked

Thinning dominant Current→Current rows to cut training time, with the per-cell learning-curve argument
and the calibration correction it requires.

### Next

Feature layer (own session), then the ML arm.

---

## Closed Decisions D44–D52
**Session of 6 October 2026**

**Scope:** HPI acquisition and join, missing-value policy, run-length left-censoring, and build
reproducibility. Continues the decision register from D43. Every entry below is closed; reopening one
requires new evidence, not a new argument.

| ID | Decision |
|---|---|
| D44 | **HPI source — FHFA purchase-only, state, seasonally adjusted.** All-transactions rejected: it mixes refinance appraisals with arm's-length sales, and those appraisals lean high precisely when prices fall, since the loan closes only if the number is high enough. That bias would deflate `est_ltv` in 2007–2009 — attenuating the one feature carrying the double-trigger mechanism, in the window the result lives in. Thin-state noise is the accepted cost: it is unbiased, and FHFA's low-volume states hold 2.13% of loans and 1.12% of defaults. Seasonal adjustment on both files, since the bridge borrows month-to-month movement and must not inject a seasonal pattern into a deseasonalised series. |
| D45 | **Quarterly-to-monthly bridge — proportional temporal disaggregation.** State HPI is published quarterly; the panel is monthly. Each month takes the state's quarterly value scaled by that month's census-division value over the division's three-month mean, so the three months average exactly to the state anchor while following the division's within-quarter shape. Division levels are never used, only relative movement. Lagging the step (using the previous quarter) was rejected: a stale index in a falling market overstates house value, reproducing the same directional error as D44 by a different route. |
| D46 | **The bridge is immaterial, and the write-up says so.** `est_ltv` against `est_ltv_step` correlates at 0.9995 (0.9991 for 2008–09), mean absolute difference 0.48 LTV points (0.63). The method halves the artificial quarter-boundary jump (0.69% against 1.26%, versus 0.50% within-quarter) but does not remove it. Reported as tested-and-does-not-matter, not as a contribution. `est_ltv_step` is retained as a sensitivity column. |
| D47 | **Publication lag — realised values used, hole declared.** FHFA publishes a quarter roughly two months after it ends, so 2008Q4 did not exist at the 2008-12 cut where every chain initialises. The realised value is used anyway. Training rows are all pre-cut and would have been published; the forecaster builds its own macro path forward and never reads realised values. The exercise is a research comparison between conditioning schemes, not a real-time simulation, and is stated as such. |
| D48 | **Territories excluded.** GU, PR and VI have no FHFA state index and belong to no census division, so neither the level nor the bridge exists for them. Dropped at loan level across all roles — 10,073 loans, 400,089 rows, 88 defaults (0.19% of credit events). Puerto Rico's separate developmental index was rejected: a second-class series spliced into one feature for 0.4% of loans buys a defence paragraph and nothing else. |
| D49 | **Missing-value rule — what the absence destroys, not how much of it there is.** Null `ltv` loans (999 sentinel, 219 loans) are dropped: without LTV there is no `est_ltv` and no equity measure at all, so the loan cannot participate in the mechanism under test. Null `dti` loans (~2.20%) are retained: the equity story is intact and one ordinary feature is blank. A size-based rule was rejected — it would flip with the count. Caveat for the write-up: the rule is costless only because LTV nulls happen to be rare. |
| D50 | **FHFA Warning flag is not a quality signal.** It marks states with fewer than 15,000 transactions over the latest ten years, and is stamped identically on all 1991–2026 quarters for the same seven states (AK, DC, DE, HI, VT, WV, WY). It carries no information about whether any given 2008 quarter was thinly estimated. No action; one limitations line naming the seven. |
| D51 | **`run_length_censored` — row-level, not loan-level.** True from an artificial run start until the loan's first observed state change, then false thereafter. Fires on seasoned entrants (first row with `loan_age` > 1) and on post-gap resets. A loan-level flag was rejected: it would mark rows as unreliable long after an observed state change restored the count. Classified as a feature — computable at inference from the loan's own history, independent of `next_state`. `post_gap` is unchanged and keeps its own meaning. Load-bearing: censored rung-1 Current rows exit at 1.41% against 3.60% for genuine ones (z = −45), so rung 1 denotes two different populations. Diagnostics 24–25 excluded seasoned and post-gap rows and hold zero flagged rung-1 rows, so the duration-dependence result needs no caveat. |
| D52 | **One build entry point.** `python -m src.panel` builds the final panel end to end — construction, both exclusions, and the HPI join — and reproduces bit-for-bit. A separate post-hoc join step was rejected: it left the panel present on disk but underivable from the build script, and a plain rerun would have silently destroyed the features. Adds `--out-dir` so rebuilds are non-destructive. The dead `hpi_feature` placeholder is removed. |

### Panel state at close

| | |
|---|---|
| Rows | 69,520,712 |
| Loans | 2,415,202 |
| Columns | 25 |
| Fingerprint | `71d85ff7fd2c5d1a` |
| Eligible loans (harness) | 405,705 |

### Findings carried to the write-up

- Nevada holds 0.91% of loans and 2.93% of defaults; default rate 6.20% against Texas at 0.97%. The
  double-trigger mechanism is visible in raw counts before any model is fitted.
- Censored and genuine rung-1 Current rows are distinct populations (1.41% against 3.60% exit, z = −45).
  Part of the ML arm's advantage will come from `run_length_censored`, a feature the homogeneous baseline
  structurally cannot see. This belongs in the attribution discussion alongside `est_ltv`, not buried in it.
- The bridge method (D45) was built, measured, and found immaterial. Reported as a negative result.

---

## ML Arm Design Checkpoint
**7 October 2026 — design-only session, no code executed**

### Session scope

Design decisions only. No training, no code, no diagnostics. Thirteen days to the 20 October deadline.
The feature layer closed on 6 October (D44–D52); this session settles how the learned transition rows get
built, selected and calibrated, so that the next session can open directly on implementation.

Panel unchanged: 69,520,712 rows, 2,415,202 loans, 25 columns, fingerprint `71d85ff7fd2c5d1a`.

### Workflow agreed (diagram nodes)

- fit rows: 2006-01 .. 2008-12
- selection train: 2006-01 .. 2007-06
- validation slice: 2007-07 .. 2007-12, chain to h=18, checked on 2008 fit rows
- grouped: Current / DQ30+DQ60 / DQ90+
- pooled: one model, origin state as feature
- JUDGE: DQ90+ and default column (aggregate reported, not decisive)
- retrain winner on ALL fit rows
- out-of-fold predictions, folds grouped by loan
- decomposed calibration: stay-vs-leave, then destination
- headline: chain to h=60 on evaluate, vs homogeneous baseline

`evaluate` is touched exactly once, at the bottom of the diagram.

### Decisions

| ID | Decision |
|---|---|
| D53 | **Grouped architecture, pre-registered.** Three models: Current; DQ30 and DQ60 shared; DQ90+. Registered before any ML number exists, on three mechanism reasons — duration behaves differently in kind across rows (diagnostic 25); Current is ~95% of rows and dominates a pooled objective; splitting gives each model an objective made of the rows it is responsible for. Splitting does not reduce imbalance; it stops one state's rows hijacking another's learning. |
| D54 | **Pooled built as comparison.** Single model with origin state as a feature. Run on the headline arm only, then the architecture is frozen for all remaining arms — otherwise five arms give five chances to keep whichever won. |
| D55 | **The chained test is the judge, not 1-step.** Consistent with the baseline, which was never scored on a one-month test. 1-step is reported alongside as the compounding finding. |
| D56 | **Validation slice: temporal, moved early.** Selection train 2006-01..2007-06; slice 2007-07..2007-12 with starting states at 2007-06; chained to h=18 against 2008-01..2008-12 fit rows. Moved earlier than the obvious 2008-07..2008-12 so the comparison can chain eighteen months instead of six. Rejected: by-loan folds (does not rehearse the exam) and grouped-temporal inside fit (pays both costs for the smallest slice). Admitted limitation: architecture is selected at h=18, not the h=60 the headline uses. |
| D57 | **Holdback is for selection only.** The winner retrains on all fit rows, crisis-onset months included. The shipped model is never weakened by the slice. |
| D58 | **`evaluate` is off limits for every internal choice.** Reason specific to this project: the headline is a gap between ML and baseline, and the baseline never got to pick anything, so letting `evaluate` select the ML architecture inflates one side only. There is no second 2009–2013. |
| D59 | **Selection metric: DQ90+ and the default column, as primary.** Not a tie-break. Forced by the objection that an aggregate-first ordering never fires when the aggregate has a clear winner that is worse on DQ90+. A split between aggregate and DQ90+ is not a contradiction — it is the expected signature of a Current-dominated pooled objective. Rejected alternative: "a split means no selection". |
| D60 | **Parsimony fallback.** If DQ90+ cannot separate the two, take pooled and report that architecture made no measurable difference. Margin value still to be set. Kept because DQ90+ is thin, and thinner still in a pre-crisis 2007 slice. |
| D61 | **No reweighting, no oversampling.** Rarity is a condition, not a defect. The standard cure inflates output probabilities, and this project never thresholds — it chains them sixty times. Illustration for the write-up: a stay-rate stated at 97.0% against a true 97.5% leaves ~16% of loans Current at h=60 instead of ~22%, and the wrongly-exited loans land where default rates are high. |
| D62 | **Thinning parked pending a timing.** Dropping Current→Current rows optimises training (40M rows once), which is not the bottleneck; forecasting (1–2B rows scored) is. Revisit only if a run takes hours. |
| D63 | **Calibration fitted after training, on out-of-fold predictions.** Fold the fit rows, train on the rest, predict the held-out fold, stack so every row has a prediction from a model that never saw it, fit the correction on the stack, apply to the model trained on everything. Fold count set by the first run's timing. |
| D64 | **Folds grouped by loan, not by time.** This step measures calibration drift, not performance, so it wants every row covered and every month represented; by-loan closes the only leak that bites. Caveat stated in the write-up: a 2007 row may be predicted by a model trained on other loans' 2008 rows. |
| D65 | **Decomposed calibration, primary scheme.** Calibrate stay-versus-leave as one binary question, then the destination split among leavers only, then assemble. The row sums to one by construction. Rejected as primary: calibrate six classes separately and renormalise — the rescaling is dominated by five small numbers and moves the one number that compounds sixty times. The naive scheme is built as the comparison, same discipline as the HPI bridge. |
| D66 | **Provenance stated honestly.** The components are standard (binary decomposition of multiclass; hazard-then-cause as in competing risks). The arrangement as a calibration step for a chained transition matrix is not attributed to a named published method, and no citation is invented. It is justified by the mechanism. |
| D67 | **Calibration priority ranking.** Current→Current first (~95% of mass, applied sixty times, compounds); then the Current row's exits (they decide who enters delinquency at all); then DQ90+→Default. Default and Prepaid are absorbing, so their errors are one-time per unit of mass. Prepaid noted separately: overstating it removes loans before they can default and drags lifetime PD down for non-credit reasons. |
| D68 | **Library: LightGBM.** Native categorical support, CPU speed at 40M rows, native null handling for the kept null-DTI loans, lighter multiclass memory than XGBoost — the constraint that killed the GPU plan. CatBoost noted as better calibrated out of the box but slower. No prior experience with any of the three, so familiarity does not favour XGBoost. |
| D69 | **`property_state` included as a native categorical**, with "without state" as the ablation. Reverses the earlier lean to omit it; Nevada holds 0.91% of loans and 2.93% of defaults, so state carries signal beyond `est_ltv`. |
| D70 | **No monotonic constraints.** Production credit models constrain risk to rise with LTV and fall with FICO for regulator explainability. This is a research project, and constraints would suppress genuine non-monotonicity, which is itself reportable. Difference from practice noted in the write-up. |
| D71 | **Feature sets per arm.** Borrower-only: FICO, original LTV, DTI, rate spread, property state, loan age, run length, `run_length_censored`. Macro-only: unemployment level and 12-month change, loan age, run length, `run_length_censored`, origin state. Both: the union plus `est_ltv`. `loan_age` sits in all three and correlates with calendar time within a vintage — flagged, not removed. |

### Open items

- The margin that counts as "DQ90+ cannot separate them" (D60) is not yet a number. Must be fixed before
  the comparison runs.
- Fold count for out-of-fold calibration (D63) waits on the first timing.
- Thinning (D62) waits on the same timing.
- How the calibration correction is applied at each of sixty forecasting steps is a forecaster question,
  deferred to the session that builds it.

### Next session

Implementation. First step is a single throwaway timed run — one model, the Current group, default
settings, full fit window — before the architecture comparison. It produces the timing that unparks D62
and sets D63's fold count, and surfaces the dull breakages (memory, dtypes, categorical declaration)
before a run that matters.
---

## Session Checkpoint — ML arm: first code, forecaster built
**8 October 2026**

### Where the session started

The 7 October session was design-only: twelve ML-arm decisions closed, no code. The panel was believed
complete at 25 columns. Twelve days to the 20 October deadline.

The agreed first step was a single throwaway timed training run, to settle the parked thinning question,
set the calibration fold count, and surface the ordinary breakages before a run that mattered.

### Workflow (diagram nodes)

- Panel rebuilt: 69,520,712 rows, 25 columns *(done)*
- `src/train_ml.py`: pooled, 8 features *(done)*
- `src/forecast.py`: propagation to h = 60 *(done)*
- Canary batch: 50,000 loans, all checks pass *(done)*
- Thinning: not needed *(measured and discarded)*
- Deduplication: not needed *(measured and discarded)*
- `loan_age`: dropped *(measured and discarded)*
- Diagnostic 38: realised vs forecast *(done)*
- Architecture comparison: grouped vs pooled *(next)*
- Calibration: 5 folds, by loan *(next)*
- Headline run: h = 60 *(next)*

### What was decided

#### The panel on disk was stale

The first training run aborted on memory. Investigation found the on-disk panel was a 24-column build
from 6 October, missing `run_length_censored`, while the committed build script writes it. The panel was
rebuilt end to end with `python -m src.panel`: 69,520,712 rows, 2,415,202 loans, 25 columns, 12.7 minutes.

Carried forward: the build report prints no fingerprint, which is why a 24-column panel passed as the
closed 25-column one. The fingerprint gate sits somewhere downstream and was not checking what it was
assumed to check.

#### Thinning is not needed (closed, measured)

Thinning the dominant Current-to-Current rows had been parked since 5 October as a speed optimisation. It
was specified, built and run.

| Run | Rows | Keep rate | Stay-share gap |
|---|---|---|---|
| Thinned | 14,913,080 | 0.368 | −2.76 points |
| Full | 39,360,874 | 1.000 | 0.0000 points |

The full Current-origin population trains in 3.6 minutes at 100 rounds, peak 6.56 GB, no paging. Reported
as a tested negative: thinning was considered, specified, run, and the data did not require it.

Had it been needed, the recovery route was the decomposed calibration scheme's first step (stay versus
leave), fitted on unthinned rows. That argument is kept as the alternative that was prepared and not used.

Scaling is close to linear: 0.86 seconds per round at 14.9M rows, 2.17 at 39.4M. The pooled architecture
therefore also fits locally.

#### Calibration folds: five

Full training at 3.6 minutes puts five out-of-fold trainings at roughly 18 minutes. The "three folds if
training is expensive" branch is dead.

#### Forecast cost was overstated by a factor of twenty

The 5 October estimate of one to two billion scored rows assumed the run-length lattice. Without it the
forecaster needs one transition row per transient origin state per month per loan: 405,672 loans × 4
states × 61 months, about 99 million rows. The full forecast runs in about 6.5 minutes.

Everything sized against the stale figure — the cloud-training idea, the GPU question, batch
checkpointing as a memory strategy — was answering a problem that does not exist. Batching is kept for
restart points and tidy output, not for memory.

#### Deduplication: measured, not worth building

Under a frozen macro path every feature is constant per loan across the horizon, so loans with matching
features need the same prediction. Measured on the forecast population:

- distinct input rows: 405,328 of 405,672 loans — 0.08% saving. `est_ltv` has 65,679 distinct values and
  shatters the space on its own.
- distinct predictions: 142,357 at h = 0 and 118,123 at h = 60 — a 65–71% collapse, because rows sharing
  every tree leaf return identical probabilities.

The second number is real but the saving is not needed at 3.6 minutes of training. Filed as measured; the
simple forecaster ships.

#### `est_ltv` discrepancy: float32 rounding, not a bug

Diagnostic 36 recomputed `est_ltv` the way `src/features_hpi.py` does, from the unrounded ratio, and
matched the stored column on all 69,520,712 rows with zero mismatches. The 84,400-loan gap came from
recomputing off the stored float32 `hpi_ratio`. Largest absolute difference 1.8 × 10⁻⁵ LTV points, signs
mixed, no row above 0.5.

Rule carried into the forecaster: derive `est_ltv` from the unrounded ratio, never from the stored float32
column. The magnitudes are immaterial; matching how training computed the feature is not.

#### `loan_age` dropped from the headline arm

The evidence, in the order it arrived — this order matters.

The canary measured 92.8% of loan-months clamped at the training maximum. The fit window is 36 months and
the forecast is 60, so within a few months of the start nearly every loan is pinned at age 36 for the rest
of the horizon. This measurement preceded any reading of `evaluate` rows and is the stated reason for
dropping the feature.

The initial reading of that number — a feature pinned at a constant contributes nothing — was wrong, and
the with/without pair falsified it:

| h = 60 | with `loan_age` | without |
|---|---|---|
| Current | 0.637 | 0.357 |
| Prepaid | 0.117 | 0.486 |
| DQ90+ | 0.162 | 0.121 |
| Default | 0.005 | 0.003 |

A feature pinned at a constant is not inert. Pinning at 36 tells the model these are three-year-old loans,
which is different from supplying no age information at all. The mechanism matches diagnostic 25:
late-rung Current exits go to Prepaid, so seasoning drives refinancing, and freezing the age freezes the
refinancing rate.

Diagnostic 22 still stands — age is real, roughly five points of cure per year, monotone. What the canary
shows is that a 36-month fit window cannot inform a 60-month forecast's seasoning. Keep those two
statements separate in the write-up or they read as contradictory.

The capability is kept behind `--with-loan-age` as the ablation.

### Code built

- **`src/train_ml.py`** — takes `--origin-states` and `--from-period-max`, so the grouped architecture and
  the selection window reuse one path. `from_state` is a native categorical when pooled, omitted when
  single-origin, resolved and printed rather than assumed. Six classes declared explicitly, never inferred.
  Two-pass build (label scan, then features) removes the Dataset construction spike that caused the first
  abort. JSON sidecar records feature order, class order, origin states, window, row count and `loan_age`
  provenance.
- **`src/forecast.py`** — propagation of a six-state probability vector to h = 60. Takes a mapping from
  origin state to model, so the grouped architecture needs no rewrite. Frozen macro, pinned `hpi_ratio`,
  long-format float32 output, 50,000-loan batches, three asserted checks.
- **`src/memwatch.py`** — peak RSS tracking and a 500 MB abort floor, shared by both.

Canary result: vectors sum to one (worst deviation 2 × 10⁻¹⁴ computed, 5 × 10⁻⁸ stored), no negative
probabilities, absorbing mass never falls. 46 seconds per batch, 33 MB per batch.

Probabilities are computed and checked in float64 and stored as float32. Storing float64 to make the check
pass would have sized storage to satisfy a test rather than to serve the use.

**Untested:** the three-model dispatch. The mapping exists but has only ever been handed one pooled model.
That is the first thing to break next session.

### Diagnostic 38 — realised versus forecast

Descriptive only by construction: no selection metric, no feature-set recommendation, because it reads
`evaluate` rows and those are off limits for every internal choice.

Canary population, 50,000 loans, h = 60 (December 2013):

| | Current | Prepaid | DQ90+ | Default |
|---|---|---|---|---|
| Realised | 21.99% | 66.42% | 2.25% | 7.28% |
| Baseline | 56.68% | 38.29% | 0.98% | 2.90% |
| ML with age | 63.69% | 11.68% | 16.17% | 0.49% |
| ML without age | 35.70% | 48.59% | 12.13% | 0.33% |

Censoring is immaterial: 126 loans of 50,000 at h = 60, reported as their own column rather than
redistributed.

**These are not results.** One batch, one pooled model, library defaults, no architecture selected, no
calibration fitted. What they are is a smoke test that happened to be informative, and two things in it
are worth carrying forward as hypotheses:

1. Every arm under-predicts prepayment and over-predicts survival, and the error grows with horizon. That
   is the compounding signature the project was built to measure.
2. The split: ML is closer on the mass (Current error +13.7 points against the baseline's +34.7), the
   baseline is closer on the tail (Default −4.38 against −6.95). If that survives a real run it lands
   exactly in the stratum declared primary on 7 October, and a split is a more interesting paper than a
   clean win.

### Pre-registration scorecard

Three misses in one session, all on the same side.

| Question | Call | Outcome | Result |
|---|---|---|---|
| Input deduplication | 40–60% saving | 0.08% | miss |
| `loan_age` at 92.8% clamped | contributes nothing | load-bearing | miss |
| Realised Prepaid at h = 60 | 25–40% | 66.4% | miss |

No call was registered on the other side for any of the three. Worth splitting calls deliberately next
session so the scorecard stays informative.

A fourth correction, not a pre-registration: the morning's memory ceiling was reasoned from roughly 4 GB
of headroom and an estimate that was never pressure-tested. The measurement showed 8.85 GB available and a
6.56 GB peak. The estimate lost to the stopwatch, which is the correct outcome and the reason the
throwaway run existed.

### Next session

1. **Architecture comparison.** Train grouped (Current / DQ30+DQ60 / DQ90+) and pooled on the selection
   window 2006-01 to 2007-06. Chain both to h = 18 through the new forecaster. Judge on DQ90+ and the
   default column, aggregate reported alongside. The three-model dispatch is untested — break it first.
2. Then retrain the winner on all fit rows, fit calibration (five folds, grouped by loan, decomposed
   scheme with the naive scheme as comparison), and run the headline forecast to h = 60.

The 7 October worry that DQ90+ would be too thin to separate the two architectures now looks less likely:
the pooled model put 0.33% in Default at h = 60 against 7.28% realised. That is the drowning the grouped
architecture was pre-registered to fix.

### Open items carried forward

- The fingerprint gate does not run in the build and did not catch a 24-column panel. Find where it lives
  and what it compares.
- Three-model dispatch untested.
- Default under-prediction is severe in every arm. Partly the fit window ending December 2008, partly the
  frozen macro. Quantify against the perfect-foresight arm if time allows.
- Semi-Markov ablation still carries the run-length lattice, where the clamp machinery and the memory
  argument return.
- Macro scenarios remain the agreed first cut if the deadline bites; the run-length ablation is protected.

---

## Session Checkpoint — 11 October 2026

### Session outcome in one line

The architecture comparison was not reached. Seven diagnostics (39–45) established that no model trained
so far beats a prior-only predictor on held-out loans, and that the metric used to judge the local regime
was itself unusable.

### Decisions closed

1. **Separation rule pre-registered** (before any architecture score exists). Quantity: absolute error
   between predicted and realised share of the book at h = 18, percentage points, on the Default and DQ90+
   columns. Noise estimate: paired bootstrap, 500 draws, both architectures scored on the same resampled
   loans, difference stored. Decision: grouped is adopted if at least 475 of 500 draws favour it on the
   Default column; otherwise parsimony fires and pooled ships. A split between the two primary columns is
   reported as a finding and parsimony still fires.
2. **Zero-fill, not near-zero**, for destination classes a single-origin model never saw. Rows chain sixty
   times, so a token value drains mass out of DQ90+ month after month as a pure artefact. The missing class
   is a fact about the window, not a modelling judgement.
3. **Brier replaces log-loss in the local regime.** Log-loss is kept alongside, reported with the count of
   rows hitting the probability floor.
4. **Capacity for Current and pooled:** LightGBM defaults beat reduced capacity under Brier on both
   populations — the opposite of the DQ90+ result.
5. **Measurement rule clarified:** fit rows may be measured on freely; only `evaluate` rows are off
   limits. The risk on fit rows is selection effects, answered by pre-registering what is measured.

### Findings

**Table 1: Every configuration loses to counting base rates.**

| Population | Configuration | Held-out Brier | vs. prior-only |
|---|---|---|---|
| Current | defaults, 100 rounds | 0.0395 | +0.0145 |
| Current | reduced, 100 rounds | 0.1062 | +0.0812 |
| Current | defaults, 400 rounds | 0.0445 | +0.0195 |
| Pooled | defaults, 100 rounds | 0.0414 | +0.0070 |
| Pooled | reduced, 100 rounds | 0.0572 | +0.0229 |
| Pooled | defaults, 400 rounds | 0.0472 | +0.0128 |

**Selection-window census (39).** 11,017,047 fit rows, 1,315,273 loans. Zero of 24 transition cells empty,
so all three grouped models would learn all six classes. The drowning argument is now measured: ~360
default events are 4.11% of the DQ90+ model's objective against 0.0033% of the pooled model's — a factor of
~1,200. *(This 0.0033% figure was corrected to 0.5020% on 14 October.)*

**DQ90+ has no discrimination (40, 41).** At 8,787 rows and at 201,792 rows, across two feature sets and
two capacities, argmax picks Default for zero held-out rows while mean predicted shares match actual shares
to within a tenth of a point. The model knows the mix and cannot tell loans apart. This reframes the
comparison: the pre-registered rule judges aggregate shares, not ranking, so grouped's likely advantage is
correct headcount in the ECL-bearing state.

**Log-loss retired (42, 43).** 398 of 39,833 held-out rows carried 73% of the total loss, 26 with exactly
zero probability on the event that occurred. On a 98.7%-Current population, log-loss measures catastrophic
surprise rather than forecast quality, and the do-nothing predictor wins by never committing. Label
alignment was verified correct by hand-check, so this is a property of the metric, not a defect.

**Undertraining diagnosis withdrawn (45).** Per-round held-out Brier on Current: round 1 0.0276, round 10
0.0283, round 50 0.0739, round 100 0.0395, round 400 0.0445, against prior-only 0.0250. The model is worse
than the prior at round one and never returns.

**The init-score experiment did not run.** Init ON and OFF are byte-identical on every figure. The score was
computed correctly (round-zero Brier equals prior-only exactly) but never reached training. Hypothesis
untested, not disproved. *(Overturned by diagnostic 46 on 12 October: it was attached correctly; the
hypothesis is dead, not untested.)*

### Methods note owed to the write-up

No run in this project had been scored against a prior-only reference until this session. That three-line
baseline is what exposed everything in diagnostics 40–45. Consequence: the 8 October headline shape — every
arm over-predicting survival with error growing in horizon — was read as the compounding signature, but may
instead be mass leaking into delinquent states from rare-class over-prediction. The two cannot yet be told
apart.

### Workflow (diagram nodes)

- 39: selection-window census
- 40: DQ90+ capacity, thin window
- 41: DQ90+ capacity, full fit
- 42: Current + pooled, log-loss
- 43: label-alignment probe
- 44: same, under Brier
- 45: init score + rounds (experiment did not apply)
- Fix init score plumbing, rerun
- Explain the round-1 loss
- Smoke test: 3-model dispatch
- Train grouped + pooled
- Chain to h=18, paired bootstrap

### Next session, in order

1. Fix the init score plumbing (attach to the LightGBM Dataset) and rerun diagnostic 45.
2. Explain why round one already loses to the prior — the live question.
3. Smoke test the three-model dispatch: three tiny models, one batch, hand-computed row assertion.
4. Train grouped and pooled on 2006-01..2007-06; chain to h = 18; apply the pre-registered rule.

### Pre-registration scorecard, this session

| Diagnostic | Yassine | Counterpart |
|---|---|---|
| 39 census | miss (tails low) | miss (tails low) |
| 40 DQ90+ thin | correct | correct |
| 41 DQ90+ full | correct | miss (attribution backwards) |
| 44 Brier | no call | correct |
| 45 init score | no call possible | no call possible |

Nine days to the 20 October deadline.

---

## Session Checkpoint — the leaf-explosion diagnosis and the headline run
**12–13 October 2026 | 7 days to deadline**

### Why this session matters

The project entered this session blocked: no model trained so far beat a prior-only predictor, under either
log-loss or Brier, and the architecture comparison had been stalled for two days. It leaves the session with
the headline result measured, out-of-sample, to h = 60.

The block had a single cause, found by separating two questions that every aggregate metric conflates.

### Workflow (diagram nodes)

- Blocker: round 1 already loses to the prior
- Hypothesis: uniform start — killed, arithmetic
- Hypothesis: init score — killed, diag 46
- Hypothesis: thin classes — killed, diag 47
- Hypothesis: macro signal — killed, diag 47
- Split the question: ordering vs levels (diag 48)
- Cause: leaf explosion, raw spread 3.4e8 on a scale ≈ 13
- Regularisation (diag 49–51)
- Architecture at h = 6: pooled ships (diag 53)
- Headline h = 60: ML beats baseline (diag 54)
- Next: run-length one-step probe

### The diagnosis

Leaf values were unbounded. A leaf holding one rare-class row among twenty takes an extreme value; 600 trees
stack those pushes without a ceiling. Measured raw-score spread reached 3.4 × 10⁸ on a scale whose six log
class frequencies span 13.24. The softmax then saturates: 32.94% of held-out rows scored exactly zero.

What exposed it was asking separately whether the model ranks loans and whether its numbers are right. Three
independent checks — aggregate Brier, the pre-registered DQ30 gate, and per-class probability ratios — all
passed while the model was broken.

The fix is a switch, not a dial. Any non-default value of `min_sum_hessian_in_leaf` or `lambda_l2` removes
saturation entirely. Settings spanning a factor of a million agree to four decimal places. Chosen: both at
10.

### Closed decisions

| Decision | Outcome | Measured reason |
|---|---|---|
| Init-score hypothesis | dead | diag 46: attached correctly, changes nothing |
| Local metric | Brier, log-loss reported | diag 43: 398 rows carried 73% of loss |
| Regularisation | hessian 10, λ2 10 | diag 49 |
| Architecture | pooled | diag 53: grouped favoured in 0/500 draws |
| Selection window | train to 200806, h = 6 | diag 47 class counts |
| Selection columns | Default + DQ90+, either decides | diag 51: DQ90+ model is real |
| Features | full set incl. macro, both arms | fairness; macro ablation deferred |
| Dispatch | verified | diag 52: hand-computed rows match |

### The headline result (diagnostic 54)

Trained on all 40.2M fit rows; chained 60 months from the 2008-12 cut; initialised from `condition` rows, so
the loans were never fitted on. Censored at h = 60: 2,952 loans (0.73%).

| Total abs. error (pp) | h = 6 | h = 12 | h = 24 | h = 60 |
|---|---|---|---|---|
| ML (pooled) | 8.69 | 10.65 | 16.64 | 42.95 |
| Baseline | 23.31 | 35.95 | 49.45 | 70.99 |
| Prior-only | 34.83 | 56.65 | 87.38 | 151.47 |
| ML − baseline | −14.62 | −25.30 | −32.81 | −28.04 |

**The central claim holds.** ML beats the baseline at every horizon, out-of-sample.

**The compounding shape is not what was predicted.** The advantage grows to roughly h = 26, then decays. ML
does not lose — its edge erodes.

**The decay has one identified cause.** DQ90+ at h = 60: realised 2.22%, ML 15.02%, baseline 0.98%. That
state alone carries 12.80 of ML's 42.95 total error. ML pumps loans into deep delinquency and the state never
drains; the baseline is closer only by being timid.

**Honest negatives.** Both arms badly under-predict Default (realised 7.48%; ML 0.46%, baseline 2.80%). ML's
win comes largely from prepayment (realised 67.35%; ML 53.30%, baseline 38.40%).

### Why run-length is now the obvious experiment

Diagnostic 25 measured DQ90+ destinations inverting with duration — to-Default rising from 0.34% at rung 1 to
7.11% at rung 10. The headline model carries no run-length feature (7 Oct decision), so it cannot tell a fresh
arrival from a loan stuck two years, and the stuck ones never leave. The project's single largest error now
has a measured mechanism and a named fix.

### Next session — option B, scoped

Train with `run_length` and `run_length_censored` added; measure whether DQ90+ ordering improves at one step
only. Roughly 20 minutes, no new forecaster code.

Rationale: chaining run-length forward requires a (state, duration) lattice — caps Current 27, DQ30 5, DQ60 3,
DQ90+ 10, ≈45 live nodes per loan instead of 6. That code does not exist. If the feature shows no one-step
gain, the lattice cannot rescue it and the afternoon is saved.

### Carried to the write-up

- Leaf explosion as the cause of every failure from 11 Oct; three checks passed on a broken model.
- Diagnostic 50: three models scoring AUC 0.86 on a column they emit as a constant — aggregate metrics cannot
  see a class given up on.
- Diagnostics 40–41 concluded "no skill" from argmax; diagnostic 51 found AUC 0.713 by asking about ordering
  instead. Same lesson, twice.
- Log-loss going uninformative on a rare-event transition matrix.
- No run in this project was scored against a prior-only reference until 11 Oct.
- Macro helps DQ90+-origin, harms Current-origin — unresolved, flagged.

### Scorecard, this session

Claude wrong: uniform-start claim; "DQ90+ ordering will be dead everywhere" (diag 50); with/without-macro gap
smaller than expected. Yassine wrong: macro would help "in a calendar way" (it hurt); the 4e-15 zero-fill call
was right. Both: neither predicted that fixing saturation would raise ordering.

---

## Session Checkpoint — 14 October 2026

### 1. Session scope

Planned: run-length, then calibration. Delivered: four diagnostics (52–55) on run-length, all at one step,
none touching `evaluate`. Calibration and macro move to the next session. Engineering layer cut.

### 2. Decisions taken

- **Engineering layer cut.** FastAPI service, drift monitor, retrain loop and containerisation are out.
  Written up as scope, not omission.
- **The `evaluate` rule is narrowed.** It governed the headline comparison, which is measured and locked.
  Follow-on experiments may read `evaluate`, because they select nothing — each gets its paragraph whichever
  way it lands. Two conditions: the report-either-way commitment is written before the numbers exist, and any
  column that read `evaluate` is marked in the table. In the event, no diagnostic this session needed it.
- **Option B rescoped before running.** The measured quantity became the DQ90+ exit rate and its destination
  split per run-length bucket, not ordering. A pooled test would have shown nothing and killed the lattice on a
  test incapable of seeing the effect.
- **Diagnostic 55 is write-up evidence only.** Framing fixed before the numbers: no model swap, no lattice, no
  forecast rerun.
- **Write-up allotted a day and a half**, written jointly from the checkpoints.

### 3. Diagnostic 52 — run length works at one step

Single-origin DQ90+ model, reduced capacity, by-loan 80/20 at seed 40.

| to-Default bucket | 1 | 2–3 | 4–6 | 7+ |
|---|---|---|---|---|
| realised | 0.29% | 2.63% | 5.41% | 6.70% |
| arm A (no RL) | 3.70% | 3.74% | 3.42% | 2.84% |
| arm B (with RL) | 0.40% | 2.80% | 5.65% | 6.49% |

Arm A is flat and faintly falling — the wrong direction. It is roughly thirteen times too high on fresh
arrivals and under half on chronic ones, and the two errors cancel in the pooled number (realised 13.32%, A
13.47%, B 13.38%). Both arms beat prior-only, the first time in this project. Censoring split immaterial: 42 of
40,520 rows.

### 4. Diagnostic 53 — the shipped configuration fails

Pooled, LightGBM defaults. Arm A predicts an 86–87% chance of leaving DQ90+ in every bucket against a realised
18/13/12/11%, emptying the state into Current. Arm B halves the DQ90+ Brier but reverses the exit curve. Arm A
loses to the origin-conditional prior on every origin state.

**Figure correction:** DQ90+ rows are 0.5020% of the pooled training objective, not the 0.0033% carried from
diagnostic 39.

**Spec defect not to repeat:** the first run stopped on an assertion that its held-out DQ90+ loans match 52's. A
by-loan split with the same seed over a different loan list assigns loans differently; it cannot pass.
Downgraded to a warning, sets differ by 0.54%.

### 5. Diagnostic 54 — capacity is not the cause

Pooled, 2 × 2 of capacity × run length. The to-Default slope, realised 22.2×:

| arm | slope |
|---|---|
| A1 defaults, no RL | 1.3× |
| B1 reduced, no RL | 1.5× |
| A2 defaults, RL | 0 at bucket 1 → 2.72% |
| B2 reduced, RL | 0 at bucket 1 → 1.37% |

No pooled arm recovers the ladder; reduced capacity makes it worse once run length is present. All four lose to
the conditional prior on every origin state. With capacity and run length held equal, the only remaining
difference from diagnostic 52 is single-origin versus pooled: 52 scores 0.233 against a prior of 0.243; 54's B2
scores 0.639 against 0.246.

### 6. Diagnostic 55 — the headline finding of the session

Each origin state fitted separately, scored against its own conditional prior:

| origin | prior | no RL | with RL |
|---|---|---|---|
| Current | 0.0332 | 0.0403 | 0.0667 |
| DQ30 | 0.6451 | 0.6158 | 0.5886 |
| DQ60 | 0.6857 | 0.6640 | 0.6493 |
| DQ90+ | 0.2460 | 0.2406 | 0.2346 |

**Three of four origin states beat base rates. Only Current fails, and Current is 97.9% of rows** — so Current's
failure has been the entire aggregate signal since 11 October. The delinquent states were working and no
measurement in this project could see it.

**Run length hurts Current** (0.0403 → 0.0667). Current is 98.3% stay, so the prior is nearly right and extra
splitting only manufactures rare-class mass: predicted to-DQ90+ exceeds realised by up to ~600×.

**The mechanism.** Staying put rises with duration in every state, so the flip is not in the stay probability. It
is in where the leavers go:

| origin | destination | bucket 1 | bucket 7+ |
|---|---|---|---|
| Current | to DQ30 | 2.99% | 0.59% |
| Current | to Prepaid | 0.35% | 0.91% |
| DQ90+ | to Default | 0.29% | 6.46% |
| DQ90+ | to Current | 8.80% | 3.06% |

Long duration means refinancing in Current and death in DQ90+. Since 69% of Current rows sit in bucket 7+, a
pooled tree learns the Current meaning from the mass of the data and applies it to DQ90+ rows — which is
diagnostic 54's 54% to-Current.

**Free result:** capacity for DQ30+DQ60 had never been measured; defaults beat reduced in both origins, narrowly.

### 7. Consequence for the architecture claim

The 13 October comparison trained both architectures without run length, on nine features whose meaning is
stable across states. Grouped had nothing to exploit and still paid the cost of splitting the data. Pooled
winning 500 draws to nothing was the right answer for that feature set.

Architecture is therefore a property of the feature set, not of the project. This scoping is the write-up
paragraph.

### 8. Scorecard

| diagnostic | Yassine | advisor |
|---|---|---|
| 52 | big gain, rung tracking: right | direction of A's error: wrong |
| 53 | — | drowning, on a 150× wrong figure: wrong |
| 54 | grouped + RL best: supported | capacity dominant: wrong |
| 55 | positive in all models: wrong on Current | flip in stay direction: wrong |

### 9. Open and carried forward

- Calibration and macro ablations — next session.
- Chaining run length still needs the lattice; not built, not scheduled.
- Single-origin Current and DQ30+DQ60 models with run length now exist as measurements only. Not a shipped
  architecture.
- The paper must state that the shipped model loses to origin-conditional base rates at one step while beating
  the homogeneous baseline when chained. Diagnostic 41's distinction (no skill at individuals, skill at aggregate
  shares) is the bridge.
- 52 vs 54 is two seeded runs, not a bootstrap.

### 10. Workflow (diagram nodes)

- Option B rescoped: buckets, not ordering
- 52 single-origin, reduced + RL: ladder recovered
- 53 pooled, defaults: both arms fail
- 54 pooled 2 × 2: capacity is not it
- 55 per origin: 3 of 4 beat the prior; Current is the failure; RL flips meaning by state
- Next: calibration, macro ablations
- Write-up: 1.5 days

---

---

## Session Checkpoint — 15 October 2026 (morning): calibration built, and the shipped model's fragility
**5 days to deadline**

### Session scope

Planned: calibration, then the macro in/out ablation. Delivered: the calibration layer built
and run one step on fit rows (diagnostic 56), and a read-only saturation check that followed
from it (diagnostic 57). The macro ablation was not reached and is deferred. This archive file
itself was assembled at the start of the session, transcribing all sixteen prior checkpoints so
the write-up has one faithful source for every figure.

### Decisions taken before any number existed

**DQ60 gets its own correction**, not one shared with DQ30. Four corrections in all, one per
transient origin state.

**Method per step, settled on data volume rather than preference.** Step one, stay-versus-leave,
uses isotonic regression — the flexible bin-and-replace method. Step two, the destination split
among leavers, uses Platt sigmoid scaling — the two-parameter smooth curve. The reason is that
step one at Current is roughly 39M rows split 98/2, so no bin is thin enough for a flexible
method to chase noise, whereas step two holds the rare destinations (Default is 0.0179% of
pooled train rows) where bins read a handful of events. The asymmetry that decided it: leftover
error from a too-rigid correction is recoverable, while a noise-fitted correction gets chained
sixty times. Known cost accepted — mixing methods means defending two choices in the write-up
instead of one.

**Pre-registration, written before any calibration number existed.** Calibrating the Current row
(98% of the book) could pull the ML arm toward the base rate, which is close to what the
baseline's Current row already is, so the arm could become more accurate while the
ML-minus-baseline gap shrank. Commitment made in advance: both numbers are reported, whichever
direction each moves, with no choosing afterwards which to lead with.

### Diagnostic 56 — the calibration layer

Five out-of-fold trainings, folds assigned by loan at seed 56, 19.1 minutes total, peak RSS
7.17 GB. Every fit row carries exactly one prediction from a model that never saw its loan;
zero loans appear in two folds; transition counts reproduce `artifacts/baseline_headline/counts.csv`
cell for cell.

**The correction works.** Applied to the out-of-fold models it was fitted on, per-class share
errors nearly vanish — DQ60's largest class error falls from 6.16 percentage points to 0.12,
DQ90+'s from 4.88 to 0.02. Brier improves at every origin state, and DQ90+ flips from losing to
its origin-conditional prior (+0.0758) to beating it (−0.0007).

**The pre-registered worry did not materialise.** Current's gap moved from −0.000307 to
−0.000308 — nothing. Calibration barely touched 98% of the book and did large things only to
the states that were broken.

**But the correction partly misfires on the shipped model.** Applied to the model actually
behind the 13 October headline, DQ60's to-DQ90+ share error goes the wrong way, 0.78 points to
6.27, even as its Brier improves from 0.954 to 0.750. The correction was fitted on fold-model
behaviour and at DQ60 the shipped model does not behave like them.

**Honest negative.** At DQ90+ the fitted destination sigmoid has a slope near zero (to-Current
a = 0.0024, to-DQ30 a = 0.0001). A slope of zero means the model's ranking of where a leaving
loan goes carries no usable information, and the correction replaces it with the base rate. The
DQ90+ improvement is the model matching base rates, not beating them.

**Trap recorded.** The rejected naive six-way-then-renormalise scheme scores about the same as
the decomposed one in aggregate, but produces visibly different rows — at DQ60 the median
per-row maximum cell difference is 0.041 and essentially every row differs by more than 0.001.
Aggregate agreement is not equivalence. This is the third instance in this project of an
aggregate number concealing row-level behaviour.

### Diagnostic 56 section 5 — the anomaly that forced diagnostic 57

The shipped model scores **worse** than its own five fold models at DQ60 and DQ90+, on the same
rows, with the same recipe — despite having seen 100% of the loans against their 80%, and
despite being scored in-sample on loans it trained on.

| | fold models (80%) | shipped (100%) |
|---|---|---|
| DQ60 Brier | 0.687 – 0.713 | 0.950 – 0.958 |
| DQ90+ Brier | 0.291 – 0.349 | 0.372 – 0.388 |
| P(stay) correlation, shipped vs fold | DQ60 **0.085**, DQ90+ **0.076** | |

All five 80% draws agree with each other and disagree with the shipped model, which rules out an
unlucky subset.

### Diagnostic 57 — read-only saturation check

**Two explanations ruled out.** The regularisation is correct: read out of each model file
rather than its sidecar, both models carry `min_sum_hessian_in_leaf` 10 and `lambda_l2` 10, and
every parameter in the two parameter blocks is identical. The 12–13 October leaf-explosion fix
did reach the shipped model. And training is deterministic: retraining fold 0 reproduced
diagnostic 56's stored predictions with zero difference across 48,177,066 cells, so the gap is
caused entirely by the extra 20% of loans.

**The pre-registered prediction failed.** The advisor predicted larger raw-score spread for the
shipped model at DQ60 and DQ90+ with Current roughly equal. Median spreads, shipped against
fold: Current 16 / 13, DQ30 14 / 8, DQ60 52 / 123, DQ90+ **218 / 631**. At DQ90+ the fold
model's spread is three times larger — and it scores better there. Spread alone explains
nothing.

**What the difference actually is, part one: DQ60 confidence errors.** Floor hits — rows where
the model put under 1e-15 on the event that occurred — number 1,832 of 25,000 DQ60 rows for the
shipped model against **6** for the fold model. A factor of roughly 300, in the state with the
largest Brier gap.

**What the difference actually is, part two: how the margin is built.** On DQ90+ rows the
shipped model's largest single boosting-round push is about 683 against a net margin of about
218 — a ratio of 2.9, meaning enormous opposing pushes that mostly cancel. The fold model's
largest push is about 660 against a margin of about 631, a ratio of 1.05: one push essentially
is the answer. The shipped model's predictions at the ECL-bearing states are small differences
between very large opposing numbers, which is structurally fragile — a loan taking a marginally
different path through the trees swings the result. Proposed mechanism, not established.

Supporting generality: the shipped booster is roughly three times more extreme throughout — 99
of 600 trees hold a leaf above 10 against the fold model's 27; median largest leaf per tree 2.31
against 0.72.

**One loan, by hand.** A DQ60-origin Michigan loan, FICO 699, that cured to Current. The fold
model gave Current 2.5%; the shipped model gave about 5e-73, having dropped the loan into a
round-3 leaf worth −237.7 on the Current class. Both models rank the loan wrong; only one
declares the realised outcome impossible. 1,827 of 25,000 DQ60 rows show this pattern.

**Mitigation already in hand.** Diagnostic 56 showed floor hits falling from 19,051 to **zero**
across every origin state after calibration. The calibration layer was built to measure drift
and repairs this as a side effect — an unplanned, measured result.

### Figure correction

The "13.24" log-class-frequency span carried from 12–13 October could not be reproduced. Pooled
across all origins the span is 8.5982; within Current-origin rows it is 13.1672. The 13.24 was
most likely the Current-origin quantity labelled as the pooled one. To be fixed before it
reaches the paper.

### Decision

**Stop here.** Chasing why 100% of loans produces a more extreme booster than 80% is a research
question in its own right and is not this paper. The anomaly is written up as measured, with the
cancellation mechanism stated as proposed rather than proven, and it belongs in the limitations
discussion alongside the headline's known DQ90+ over-filling — the two may well be the same
thing seen at one step and at sixty.

### Scorecard

| diagnostic | Yassine | advisor |
|---|---|---|
| 56 | no call | no call |
| 57 spread direction | no call | **wrong** — predicted shipped-model spread larger at DQ60 and DQ90+; at DQ90+ the fold model's is three times larger |

The advisor's saturation framing was wrong on the measurement it proposed, and the run found a
different mechanism (push cancellation) that the framing did not anticipate.

### Carried forward

- Macro in/out ablation — deferred, not run.
- The shipped model's fragility is a limitations paragraph, not a retraining decision.
- One retrain was necessary and is labelled as such: diagnostic 56 did not persist its fold
  boosters, so fold 0 was retrained once to serve as diagnostic 57's comparison model. It
  reproduces the stored predictions exactly.
- Whether calibration helps or hurts the 60-month **chained** forecast is still unmeasured.
  Everything above is one step ahead on fit rows.

---

*End of archive. Sessions 2 through 15 October 2026, seventeen checkpoints.*
