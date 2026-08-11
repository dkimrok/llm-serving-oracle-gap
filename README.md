# How Far from Optimal? — Oracle Optimality-Gap Benchmark for LLM Serving Schedulers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21862821.svg)](https://doi.org/10.5281/zenodo.21862821)

Code, ledgers, and oracle schedules for the paper *"How Far from Optimal?
Measuring and Decomposing the Optimality Gap of LLM Serving Schedulers"*
(under submission, ML for Systems Workshop @ NeurIPS 2026).

We compute hindsight-optimal schedules on public Azure LLM inference traces
with a conservative time-bucketed MILP (KV-cache growth, chunked prefill,
shared token budgets), replay heuristic schedulers in the same feasible
region, and report each slice's optimality gap as a proven interval
[dual bound, incumbent].

**Headline results.** On conversation workloads, FCFS is provably ≥38.5%
above optimal on average (positive in all 50 trace slices); workload type
dominates load level; the information gap (14.2%) and algorithmic gap
(11.7%) are comparable; and MILP planning with a crude causal length
predictor beats a perfect-information greedy heuristic in 5/6 slices.

## Repository layout

```
src/                # all scripts (formulation, simulator, loader, runners, analysis)
results/            # ledgers: results.csv (main+pilot), results_polish_s*.csv,
                    #          results_opred.csv, plans/*.json (oracle schedules)
figures/            # paper figures + make_figures.py output
gap_bench_local.ipynb / gap_bench_colab.ipynb   # convenience notebooks
docs/               # research design (Korean) and full development log
```

## Reproduction

Requirements: Python 3.10+, ~8 GB RAM, ~3 GB disk (transient), no GPU.

```bash
pip install pulp highspy pyarrow pandas matplotlib statsmodels
python src/toy_test.py            # environment check (brute-force == MILP)
python src/opred_test.py          # O-pred invariants (perfect prediction == oracle)
python src/prep_trace.py          # downloads ~2 GB Azure traces -> traces/*.parquet
python src/run_slices.py --pilot --tl 900          # 6 slices, ~2 h
python src/run_slices.py --main  --tl 900          # 50 slices, ~13-25 h (laptop)
python src/run_slices.py --polish --gap-th 0.30 --tl 1800 --shard 0/2   # optional,
python src/run_slices.py --polish --gap-th 0.30 --tl 1800 --shard 1/2   # 2 terminals
python src/run_opred.py --pilot --tl 900           # decomposition layer, ~2 h
python src/make_figures.py
```

All runners are resumable (completed slice IDs are skipped) and append to
CSV ledgers after every slice. `--shard i/N` splits any run across
terminals with separate ledgers; analysis merges per slice by best
incumbent/bound. Reference hardware for the shipped ledgers: Intel Core
Ultra 5 225H laptop, 16 GB RAM, HiGHS single-thread solves.

Determinism note: heuristic replays are exactly reproducible (two
independent runs on different working copies produced byte-identical
heuristic ledger columns); MILP incumbents under a wall-clock limit may
vary slightly across machines — hence interval reporting.

## Data and citation

Traces: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset),
`dataset-llm-2024` (CC-BY-4.0). If you use the traces, cite Stojkovic et
al., *DynamoLLM*, HPCA 2025, per the dataset's terms.

Code is MIT-licensed. Paper citation: TBD (workshop decision Sep 2026).
