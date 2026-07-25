# Where Constraints Fail: Phase-Aware Diagnostics for Tool-Calling Language Models

## Abstract
Constrained decoding is widely adopted to guarantee the syntactic validity of Large Language Model (LLM) outputs, particularly in agentic tool-calling scenarios. However, enforcing strict constraints can severely degrade semantic reasoning, a phenomenon often referred to as the *Constraint Tax*. We decompose this constraint-induced degradation into three distinct failure phases: Protocol Exclusion, Decision-Boundary Distortion, and Semantic Projection Distortion. Through a comprehensive token-level trace suite over the Berkeley Function Calling Leaderboard (BFCL) across multiple open-weight models, we demonstrate that failures are phase-specific, and that pre-decision feasible mass can predict semantic distortion beyond simple confounders. Finally, we evaluate an offline Phase-Aware Constraint Router (PACR) that leverages these diagnostics to adaptively route inference between immediate constraints, trigger-based decoding, and draft-conditioned decoding.

## Introduction
While constrained generation guarantees structural validity (e.g. JSON compliance), recent studies have identified a "Constraint Tax" where model reasoning degrades. We identify that this degradation is not monolithic but phase-dependent. Our contributions are: 1) A phase-aware taxonomy of failures, 2) Phase-local diagnostics based on feasible mass, and 3) An offline evaluation of PACR.

## Experimental Setup
We utilize 120 stratified examples from BFCL V4 (30 simple, 30 multiple, 30 complex, 30 relevance). Our cross-model evaluation uses Qwen2.5-3B, Llama-3.2-3B, and Gemma-3-4B. We compare Unconstrained, Immediate, Trigger, and Draft-Conditioned decoding policies, extracting pre-mask raw logits to compute feasible mass.

## Main Results
(To be populated by Kaggle run)

## Conclusion
(To be populated based on the final scientific verdict)
