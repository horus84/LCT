# Where Constraints Fail: Phase-Aware Diagnostics and Adaptive Decoding for Tool-Calling Language Models

## Abstract
Constrained decoding is widely adopted to guarantee the syntactic validity of Large Language Model (LLM) outputs, particularly in agentic tool-calling scenarios. However, enforcing strict constraints can severely degrade semantic reasoning, a phenomenon often referred to as the *Constraint Tax*. We decompose this constraint-induced degradation into three distinct failure phases: Protocol Exclusion, Decision-Boundary Distortion, and Semantic Projection Distortion. Through a comprehensive token-level trace suite over the Berkeley Function Calling Leaderboard (BFCL), we demonstrate that failures are phase-specific, and that pre-decision feasible mass can predict semantic distortion beyond simple confounders. Finally, we evaluate the Phase-Aware Constraint Router (PACR), a lightweight adaptive policy that routes inference between immediate constraints, trigger-based decoding, and two-pass draft conditioning based on feasible mass risk. Our results characterize the empirical Pareto frontier of accuracy, validity, and latency across multiple open-weight SLM families.

## Introduction
* The scientific problem: Valid tool schemas are necessary for agents, but constraining generation degrades reasoning.
* Existing work: CRANE introduces grammar augmentation; DCCD proposes expensive two-pass drafting; Tool Suppression papers identify constraint priority inversion.
* What remains unknown: Are failures monolithic, or phase-dependent? Can we predict failures using inference-time diagnostics (feasible mass) before they occur?
* Contribution: We propose a phase-aware taxonomy of failures, introduce phase-local diagnostics, and evaluate PACR across multiple SLM families on BFCL.

## Experimental Setup
* Models: Qwen2.5-3B, Llama-3.2-3B, Gemma-3-4B, Qwen2.5-1.5B
* Benchmark: BFCL V4 (480 stratified examples)
* Decoding Policies: Unconstrained, Immediate, Trigger, Draft-Conditioned, PACR.

## Main Results
(To be populated by Kaggle run)

## Conclusion
(To be populated based on the final scientific verdict)
