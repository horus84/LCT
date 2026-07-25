# Preregistration: Where Constraints Fail

**Date:** July 2026
**Framework:** Phase-Aware Diagnostics and Adaptive Decoding for Tool-Calling Language Models

## 1. Experimental Setup
* **Models:** `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-3-4b-it`. Anchor ablation model: `Qwen/Qwen2.5-1.5B-Instruct`.
* **Benchmark:** BFCL V4 (480 stratified examples).
* **Categories:** simple, multiple, parallel, parallel_multiple, relevance.
* **Split:** Seed 42, 30% development, 70% held-out test.
* **Hardware & Budget:** 2x T4 GPUs.
* **Decoding Parameters:** Max new tokens = 512, do_sample = false (greedy decoding).

## 2. Tested Decoding Policies
1. **U_UNCONSTRAINED**: Baseline native generation.
2. **I_IMMEDIATE**: Fully constrained decoding using standard JSON grammar.
3. **T_TRIGGER**: Unconstrained generation until the `<tool_call>` trigger token is emitted, followed by constrained decoding.
4. **D_DRAFT_CONDITIONED**: Two-pass draft-conditioned decoding (DCCD).
5. **P_PACR**: Phase-Aware Constraint Router. Routes to T_TRIGGER if structurally unreachable; routes to D_DRAFT_CONDITIONED if risk score > threshold; else I_IMMEDIATE.

## 3. Core Hypotheses & Primary Comparisons
**H1 (Phase Local Taxonomy):** Tool calling failures are not uniform. Protocol exclusion, decision boundary distortion, and argument projection distortion occur at distinct phases and can be isolated via unconstrained paired comparisons.
**H2 (Phase-Local Diagnostics):** Pre-decision negative log feasible mass (Projection Tax) and decision-step entropy predict semantic degradation (`U correct -> I wrong`) significantly better than simple confounders (output length, tool count).
**H3 (Adaptive Routing - PACR):** Using pre-decision diagnostics to dynamically route generations between I_IMMEDIATE, T_TRIGGER, and D_DRAFT_CONDITIONED will yield an accuracy-validity-latency Pareto frontier superior to any single fixed policy on the held-out test set.

## 4. Evaluation Metrics
* **Efficacy:** BFCL AST accuracy, invocation precision/recall, argument correctness.
* **Validity:** Schema compliance rate.
* **Efficiency:** Mean/p50/p95 latency, generated token cost.
* **Constraint Dynamics:** Phase-local projection taxes, pre-decision feasible mass, mechanism failure counts.

## 5. Statistical Tests
* **McNemar's Test:** For paired correctness comparisons between policies.
* **Bootstrap CIs:** 95% CIs for accuracy, recall, and latency differences.
* **Cliff's Delta / Rank-Biserial:** For paired effect sizes in tax distributions.
* **Stratified CV Logistic Regression (ROC AUC):** For evaluating the incremental predictive power of tax vs confounders.

## 6. Stopping Rules
The experiment pipeline is frozen. PACR thresholds will be tuned *strictly* on the 30% development set. The 70% held-out test set will be evaluated exactly once. No post-hoc tuning of the router will be performed against the test set.
