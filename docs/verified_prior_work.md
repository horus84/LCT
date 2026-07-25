# Verified Prior Work

This document verifies the closest literature related to constrained decoding, tool calling, and projection taxes, identifying exact gaps and overlaps with the proposed PACR and Phase-Aware framework.

### 1. CRANE: Reasoning with constrained LLM generation
* **Authors:** Debangshu Banerjee, Tarun Suresh, Shubham Ugare, Sasa Misailovic, Gagandeep Singh
* **Date:** February 2025 (ICML 2025)
* **URL:** https://arxiv.org/abs/2502.09061
* **Closest Overlap:** Proposes grammar augmentation to allow reasoning steps alongside structural constraints.
* **Difference:** CRANE focuses on symbolic math and code generation, using statically augmented grammars. It does not measure phase-local projection tax, nor does it address API tool calling or adaptive routing (PACR). 

### 2. Draft-Conditioned Constrained Decoding for Structured Generation in LLMs
* **Authors:** Avinash Reddy, Thayne T. Walker, James S. Ide, Amrit Singh Bedi
* **Date:** March 2026 (ICML 2026)
* **URL:** https://arxiv.org/abs/2603.03305
* **Closest Overlap:** Introduces two-pass draft-conditioned decoding (DCCD) to prevent feasible mass collapse. 
* **Difference:** DCCD applies two passes unconditionally. It lacks a taxonomy of phase-specific failures (e.g., protocol exclusion vs argument projection distortion) and does not utilize pre-decision diagnostics to adaptively bypass the expensive drafting phase, which PACR does.

### 3. The Constraint Tax: Measuring Validity-Correctness Tradeoffs in Structured Outputs for Small Language Models
* **Authors:** Jaideep Ray
* **Date:** May 2026
* **URL:** https://arxiv.org/abs/2605.26128
* **Closest Overlap:** Formally defines the "Constraint Tax" and its degradation effect on small language models.
* **Difference:** The paper defines the constraint tax generally as an accuracy delta, rather than computing it formally as $-\log(\alpha_t)$. It does not use feasible mass as an inference-time predictive diagnostic, nor does it propose mitigation routing.

### 4. Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints
* **Authors:** Various (AgentPatterns.ai / Hugging Face community)
* **Date:** June 2026
* **URL:** https://arxiv.org/abs/2606.25605
* **Closest Overlap:** Identifies Constraint Priority Inversion (CPI), where structured JSON constraints completely suppress tool invocation. Proposes "Transparent Two-Pass Execution".
* **Difference:** This paper evaluates suppression at the aggregate level (tool vs no-tool). It does not perform token-level trace analysis of the decision boundary, measure raw pre-mask entropy, or dynamically route policies using phase-local tax prediction.

### 5. Thinking Before Constraining: A Unified Decoding Framework for Large Language Models
* **Authors:** Ngoc Trinh Hung Nguyen, et al. (Nokia Bell Labs)
* **Date:** January 2026
* **URL:** https://arxiv.org/abs/2601.07525
* **Closest Overlap:** Introduces the "In-Writing" decoding framework, decoupling reasoning from formatting using a trigger token.
* **Difference:** This acts as our `T_TRIGGER` baseline. However, the paper uses a fixed trigger policy. It does not adaptively switch between immediate constraint, trigger, and two-pass decoding based on feasible mass risk.

### 6. XGrammar-2
* **Date:** 2025/2026
* **Closest Overlap:** Advanced caching structured generation engine using TagDispatch for dynamic switching.
* **Difference:** XGrammar-2 is infrastructure. Our work evaluates the algorithmic *policies* (Immediate, Trigger, DCCD, PACR) layered on top of such grammar engines, analyzing the interaction between the model's logits and the engine's mask.

### 7. Berkeley Function Calling Leaderboard (BFCL)
* **Closest Overlap:** The de facto standard for evaluating tool calling using AST evaluation.
* **Difference:** BFCL scores aggregate outputs. Our contribution builds a trace-level diagnostic layer over BFCL to expose token-by-token feasible mass and constraint-induced probability shifts.

### Conclusion on Novelty
The proposed central thesis—**Phase-Aware Diagnostics and Adaptive Decoding (PACR)**—is highly novel. While the community has independently identified the problem (Constraint Tax / Tool Suppression) and proposed isolated fixed solutions (Trigger tokens, DCCD), no prior work synthesizes these into a phase-local diagnostic framework that dynamically routes decoding policies based on pre-decision feasible mass and entropy.
