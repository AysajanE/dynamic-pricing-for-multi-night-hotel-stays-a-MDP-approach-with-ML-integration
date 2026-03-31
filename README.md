# Dynamic Pricing for Multi-Night Hotel Stays: A Markov Decision Process Approach

Research code for studying hotel pricing when bookings span multiple nights and inventory decisions are coupled across an entire stay.

This repository focuses on the algorithmic side of multi-night hotel revenue management: exact dynamic programming benchmarks for tractable instances, scalable approximation workflows for larger instances, synthetic-data generation, and computational experiments for solution quality and scalability.

## Problem setting

In single-night pricing problems, each booking decision affects one date. In multi-night hotel settings, a single accepted reservation consumes inventory across several nights at once. That creates a higher-dimensional control problem in which pricing decisions interact across the stay pattern of each booking class.

This repo explores that setting through:

- Markov decision process style formulations
- Dynamic programming benchmarks for small instances
- Approximate / stochastic methods for scalability experiments
- Synthetic test-instance generation and experimental evaluation

## Repository contents

- `src/data_generator.py`: configurable synthetic instance generation for hotel demand and booking classes
- `src/dynamic_pricing_algorithms.py`: pricing algorithms, including dynamic programming and approximation logic
- `src/experiment1_solution_quality_assessment.py`: compares approximation quality against dynamic programming on tractable cases
- `src/experiment2_scalability_efficiency_assessment.py`: computational scaling experiments
- `src/test_experiment1.py`: reproducibility and consistency checks for the Experiment 1 workflow
- `notebooks/`: exploratory notebooks, analysis drafts, and result summaries
- `docs/Computational Study Design Framework.pdf`: design notes for the experiment program
- `images/`: supporting figures

## What this repo demonstrates

- Framing a realistic hotel pricing problem with multi-night coupling
- Translating the problem into executable research code rather than only a conceptual model
- Comparing exact and approximate methods under controlled synthetic experiments
- Preserving experiment structure through code, tests, notebooks, and written analysis artifacts

## Getting started

This is research code rather than a packaged library. A simple local workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib seaborn
python src/test_experiment1.py
```

Then inspect the notebooks under `notebooks/` for the experimental walkthroughs and result summaries.

## Suggested entry points

If you are reviewing the repo for substance rather than trying to run everything immediately, start here:

1. `src/dynamic_pricing_algorithms.py`
2. `src/experiment1_solution_quality_assessment.py`
3. `src/test_experiment1.py`
4. `notebooks/computational_analysis_report.md`

## Notes

- This repository is best read as a research implementation and experiment log.
- The codebase currently emphasizes algorithmic comparison and computational analysis more than productization or packaging.
