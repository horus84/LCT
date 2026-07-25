# Reproducibility Guide

## Hardware & Environment
- Environment: Kaggle dual T4 GPUs (or equivalent cluster).
- Base requirements: Python 3.10+
- Models required (approx 20GB VRAM total): `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-3-4b-it`.

## Setup
```bash
pip install -r requirements.txt
```

## Step 1: Execute the Pipeline
Run the evaluation engine. This will automatically manage caching, loading models, generating the samples using the 5 policies (`U_UNCONSTRAINED`, `I_IMMEDIATE`, `T_TRIGGER`, `D_DRAFT_CONDITIONED`, `P_PACR`), and outputting to `results/probe_results.csv` and `results/traces/`.

```bash
python scripts/bfcl_evaluator.py
```

## Step 2: Analyze Results
Generate the statistical breakdown and tables.

```bash
python scripts/analyze_results.py
```

## Step 3: Generate Figures
Produce the plots for the paper.

```bash
python scripts/generate_paper_assets.py
```

## Outputs
- CSV metrics: `results/probe_results.csv`
- LaTeX Tables: `tables/`
- Figures: `figures/`
- Traces: `results/traces/`
