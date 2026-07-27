# Reproducibility Guide

This repository contains all code necessary to reproduce the paired counterfactual persona experiment described in our EACL 2027 submission. 

## Requirements
- `transformers`
- `torch`
- `pandas`
- `scipy`

## Execution Steps

### 1. Build Paired Dataset
```bash
python src/build_counterfactual_pairs.py
```
This reads `sycophancy_dataset.json` and outputs `paired_dataset.json`.

### 2. Validate Pairs
```bash
python src/validate_pairs.py
```
This outputs `prompt_pairs_audit.md` allowing manual verification that only political identity markers changed between paired prompts.

### 3. Run Generation
```bash
python src/run_generation.py --model "Qwen/Qwen2.5-7B-Instruct"
```
Outputs `paired_generation_results.jsonl`. 
*Note: This requires a GPU environment (e.g., Kaggle Dual T4).*

### 4. Parse Outputs
```bash
python src/parse_outputs.py
```
Outputs `parsed_paired_results.csv`, applying strict regex to ensure robust letter extraction from generative traces.

### 5. Run Token-Level Early Answering
```bash
python src/run_early_answering.py --model "Qwen/Qwen2.5-7B-Instruct"
```
Outputs `paired_early_answering.jsonl`. Token-truncates the rationale at [10%, 50%, 75%, 100%] and forces single-token generation.

### 6. Statistical Analysis
```bash
python src/statistical_analysis.py
```
Prints the transition matrices, McNemar exact tests, and early-answering trajectories required for the LaTeX paper.
