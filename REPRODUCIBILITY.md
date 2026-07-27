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

### 5. Optimized Early Answering Pipeline
Follow these optimized stages to execute the early answering experiment with prefix caching, logit scoring, and recovery checks:

#### Stage A: Audit Generation Traces
```bash
python -m src.audit_early_answering --config configs/early_answering.yaml
```
Outputs `results/early_answering_audit.json`.

#### Stage B: Tokenize and Cache Truncations
```bash
python -m src.tokenize_truncations --config configs/early_answering.yaml
```
Creates tokenized representations and boundary mappings in `data/processed/early_answer_token_cache.pt`.

#### Stage C: Calibrate & Freeze Scoring Method
```bash
python -m src.calibrate_answer_scoring --config configs/early_answering.yaml --n-calibration 20
```
Runs logit scoring on a stratified 20-sample validation set and saves the choice in `results/method_calibration.json`.

#### Stage D: Run scoring (with 100% stop condition check)
```bash
python -m src.score_early_answers --config configs/early_answering.yaml --truncations 0.10 0.50 1.00 --subset primary --resume
```
Validates the frozen logit comparison method. Performs one forward pass per truncation point, saving results dynamically to `results/early_answer_scores.csv`.

#### Stage E: Statistical Trajectory Analysis
```bash
python -m src.analyze_early_answers --input results/early_answer_scores.csv
```
Prints the final commitment trajectory curves and original answer agreement rates.

