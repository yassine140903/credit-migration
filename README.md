# Covariate-Conditioned Markov Chains for Lifetime Credit Risk

**Paper: [docs/paper.pdf](docs/paper.pdf)**

This repository compares a homogeneous Markov transition matrix against a LightGBM
borrower-conditioned transition model. Both are chained 60 months forward on the Freddie Mac
Single-Family Loan-Level Dataset (SFLLD), 2006--2007 origination vintages, and both forecast
the state distribution of the book from the December 2008 cut. The paper reports where
borrower conditioning helps and where it does not. See [docs/paper.pdf](docs/paper.pdf).

## Result

Total absolute error of the forecast state distribution (sum over the six states of the
absolute difference between predicted and realised share of the book, percentage points).
405,672 loans forecast from the December 2008 cut. Single run; no sampling interval.

| Total absolute error (pp) | h=6 | h=12 | h=24 | h=60 |
|---|---:|---:|---:|---:|
| Learned arm | 8.69 | 10.65 | 16.64 | 42.95 |
| Homogeneous baseline | 23.31 | 35.95 | 49.45 | 70.99 |
| Learned minus baseline | -14.62 | -25.30 | -32.81 | -28.04 |

The learned matrix forecasts the book's state distribution much better, and the gain comes
from prepayment and from loans that stay current. On five-year default it is worse than the
baseline: it predicts 0.46% against 2.80% for the baseline, with 7.48% realised.

One-step signal from borrower characteristics is real but small, and the pooled model fails
at DQ90+.

## What is in this repo

| Module | What it does (from its docstring) |
|---|---|
| `src/panel.py` | Transition-panel builder for the SFLLD 2006--2007 vintages: one row per observed consecutive-month transition, six-state space, fit / condition / evaluate roles |
| `src/panel_io.py` | The one entry point for reading the built panel; validates the whole panel against its manifest first |
| `src/hpi_state_monthly.py` | State-level monthly HPI series from FHFA quarterly state and monthly division purchase-only indexes |
| `src/features_hpi.py` | Read-only re-check of the HPI features on the committed panel |
| `src/baseline.py` | The homogeneous (time-pooled) Markov transition matrix; estimation only |
| `src/train_ml.py` | Trains one one-month multiclass LightGBM transition model |
| `src/forecast.py` | The ML arm's forecaster: a one-month model to a 60-month state-probability forecast per loan, by propagation not simulation |
| `src/calibrate.py` | Calibration layer for the ML arm: out-of-fold predictions, per-origin correction, and its application |
| `src/horizon.py` | Horizon evaluation: eligibility at the cut, realised outcomes, cohorts, chaining |
| `src/horizon_baseline.py` | The homogeneous baseline run through the horizon harness |
| `src/horizon_nulls.py`, `src/nulls.py` | Persistence, marginal and oracle reference models for the harness |
| `src/scoring.py` | Model-agnostic scoring of predicted state distributions |
| `src/memwatch.py` | Background memory watch for long runs: peak RSS, minimum available memory, hard floor |
| `src/diagnostics/cell_counts.py` | Cell-count diagnostic for the conditional oracle |
| `tests/` | Unit tests for the panel reader, HPI join, run-length censoring and calibration |
| `writing/` | LaTeX source of the paper |

`diagnostics/` holds numbered one-off measurement scripts, each with a committed `.txt` report.
Diagnostic numbers are not unique (53 and 54, among others, are each used by two different
files), so files are referenced by full filename.

## Data

The Freddie Mac Single-Family Loan-Level Dataset, 2006--2007 origination vintages, is
downloaded by the user from Freddie Mac's site
(<https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>; free account required).
The data is not redistributed here and is not committed.

Other inputs, as `src/` reads them:

- FHFA purchase-only HPI, `data/hpi_po_state.xlsx` (quarterly, states) and
  `data/hpi_po_monthly_hist.xlsx` (monthly, census divisions and U.S.), read by
  `src/hpi_state_monthly.py`.
- The national unemployment series (`UNRATE`) and the PMMS 30-year rate (`MORTGAGE30US`),
  read from `data/macro/UNRATE.csv` and `data/macro/MORTGAGE30US.csv`. If a file is absent
  `src/panel.py` fetches it from FRED.
- `docs/file_layout_july_2026.xlsx`, the file layout read by the panel build.

Layout under `data/` (the panel build globs `data/historical_data_200[67]/*/perf_*.txt` and
`.../orig_*.txt`):

```
data/historical_data_2006/historical_data_2006Q1/perf_2006Q1.txt   (and orig_2006Q1.txt)
data/historical_data_2006/historical_data_2006Q2/ ... Q4
data/historical_data_2007/historical_data_2007Q1/ ... Q4
data/hpi_po_state.xlsx
data/hpi_po_monthly_hist.xlsx
data/macro/UNRATE.csv, MORTGAGE30US.csv        (fetched if absent)
data/macro/hpi_state_monthly.csv, census_divisions.csv   (written by src.hpi_state_monthly)
data/panel/                                    (written by src.panel)
data/forecast/headline_pooled/                 (written by src.forecast)
```

## Reproduce

Run from the repository root. CPU only; the headline was run on a 16 GB RAM machine.

```
python -m src.hpi_state_monthly
python -m src.panel
python -m src.baseline --fit-only --out-dir artifacts/baseline_headline
python -m src.train_ml --name headline_pooled --min-sum-hessian-in-leaf 10 --lambda-l2 10
python -m src.forecast --run headline_pooled --model artifacts/ml/headline_pooled --expect-loans 405672 --report diagnostics/54_headline_forecaster_build.txt
python diagnostics/54_headline_h60.py B
```

`python diagnostics/54_headline_h60.py A` runs the baseline, training and forecast steps above
as subprocesses; `B` then scores every arm and writes `diagnostics/54_headline_h60.txt`
(the only step that opens `evaluate` rows).

Calibration, on the shipped model:

```
python diagnostics/56_calibration_report.py header
python -m src.calibrate oof --model artifacts/ml/headline_pooled
python -m src.calibrate fit --model artifacts/ml/headline_pooled
python diagnostics/56_calibration_report.py report
```

Tests: `python -m pytest tests`.

## Not built / future work

From Section 9.2 of the paper. Each item is aimed at a specific gap measured in the paper.

1. *Does the economy matter forward?* Rerun the forecast with the realised post-2008
   macroeconomic path as an upper bound. A small gap between that and the frozen run would
   make scenario weighting unnecessary for this question; a large one would justify building it.
2. *Replace the DQ90+ row.* Chain the forecast with a single-origin DQ90+ model, which beats
   its base rate where the pooled model does not, and test whether the erosion after two years
   disappears.
3. *Chain duration.* Build the state--duration lattice and run the semi-Markov comparison at
   sixty months.
4. *Chain the calibrated model.* Measure whether calibration's one-step repairs survive sixty
   months of compounding.
5. *Uncertainty.* Resample loans from the existing forecasts to put an interval on every
   chained figure.
6. *Rerun the unbounded diagnostics* with the leaf bound, and explain why adding a fifth of
   the loans makes the model more extreme.

Not built:

- FastAPI serving
- Transition-matrix drift monitor
- Monthly retrain loop
- Containerisation
- IFRS 9 scenario weighting (no base, adverse and severe scenarios were built)

## Limitations

- Single run with no sampling interval.
- One cut (December 2008) and one crisis; neither the size of the gap nor its shape can be
  assumed to transfer.
- Macroeconomic and house-price inputs are frozen at their December 2008 values.
- No duration in the chain: time in state is measured but the chained forecast cannot use it
  without a state--duration lattice.
- Both arms under-predict five-year default, the learned arm severely. No expected-credit-loss
  figure should be read from either arm.
