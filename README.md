# credit-migration

An empirical study of credit-state migration for residential mortgages, built on the
Freddie Mac Single-Family Loan-Level Dataset. The object of interest is the transition
behaviour of a loan through delinquency states — how loans enter deep delinquency, how
often they cure back to current, how long a cure takes, and what fraction ultimately
terminate in a credit event rather than a voluntary payoff. The 2006 and 2007 origination
vintages are used because their ~18-year monthly performance panels span the entire
2007–2011 credit cycle, so entries into deep delinquency, workouts, modifications and
dispositions are all observed within the same cohort rather than being censored at the
edge of the sample. This repository currently contains **diagnostics only**: a set of
standalone scripts that characterise the raw panel before any modelling pipeline is
built, so that panel shape, termination accounting and deep-delinquency dynamics are
measured rather than assumed.

## Data required

The data is **not committed** — the monthly performance files run to roughly 2 GB each
and are covered by Freddie Mac's dataset terms of use. You must download it yourself.

**Source:** <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>
(free account registration and acceptance of the terms of use are required; the files
sit under "Standard Dataset" → historical data, one ZIP per origination quarter).

**Vintages required by the diagnostics:**

| Vintage | Archive | Needed file |
|---|---|---|
| 2006 Q1 | `historical_data_2006Q1.zip` | `perf_2006Q1.txt` |
| 2006 Q2 | `historical_data_2006Q2.zip` | `perf_2006Q2.txt` |
| 2006 Q3 | `historical_data_2006Q3.zip` | `perf_2006Q3.txt` |
| 2006 Q4 | `historical_data_2006Q4.zip` | `perf_2006Q4.txt` |
| 2007 Q1 | `historical_data_2007Q1.zip` | `perf_2007Q1.txt` |
| 2007 Q2 | `historical_data_2007Q2.zip` | `perf_2007Q2.txt` |
| 2007 Q3 | `historical_data_2007Q3.zip` | `perf_2007Q3.txt` |
| 2007 Q4 | `historical_data_2007Q4.zip` | `perf_2007Q4.txt` |

Only the monthly performance files (`perf_*.txt`) are read by the diagnostics; the
origination files (`orig_*.txt`) ship in the same archives and are retained for later
pipeline work.

**Expected layout on disk.** The scripts use a hardcoded glob and take no arguments, so
the extracted archives must sit at exactly this path relative to the repository root:

```
data/historical_data_2006/historical_data_2006Q1/perf_2006Q1.txt
data/historical_data_2006/historical_data_2006Q2/perf_2006Q2.txt
...
data/historical_data_2007/historical_data_2007Q4/perf_2007Q4.txt
```

i.e. the glob `data/historical_data_200[67]/*/perf_*.txt`, which should resolve to
8 files totalling roughly 15 GB. Each script reports which files it actually found, so a
missing vintage shows up in the output rather than silently shrinking the sample.

The files are pipe-delimited with **no header row** and carry a trailing delimiter, so
each line splits into 35 fields of which 34 are real columns. Field positions used by the
diagnostics (0-indexed) are `0` loan sequence number, `1` monthly reporting period
(`YYYYMM`), `3` current loan delinquency status, `7` modification flag, `8` zero balance
code — verified against the official file layout in the Freddie Mac user guide.

## Diagnostics

Each script prints plain-text tables to stdout and writes the identical output to
`diagnostics/<name>.txt`. The `.txt` outputs are committed alongside the scripts as a
record of what the data looked like.

- `diagnostics/01_panel_shape.py` — row counts, distinct loans, reporting-period ranges
  and the distribution of months tracked per loan, per file and pooled.
- `diagnostics/02_terminations.py` — termination accounting: terminated vs censored
  loans, the full zero-balance-code frequency table, and prepayments / credit events /
  administrative exits as counts and shares.
- `diagnostics/03_dq_deep_state.py` — the deep-delinquency (status >= 6) subpopulation:
  entry, cure rate, time-to-cure distribution, conditional default probability, and
  deep-delinquent dwell time with the implied monthly hazard.

Run them from the repository root:

```
python diagnostics/01_panel_shape.py
python diagnostics/02_terminations.py
python diagnostics/03_dq_deep_state.py
```

All reads are chunked (`chunksize=2_000_000`, `dtype=str`, `usecols` limited to the
columns each pass needs); no script ever holds a whole file in memory.
