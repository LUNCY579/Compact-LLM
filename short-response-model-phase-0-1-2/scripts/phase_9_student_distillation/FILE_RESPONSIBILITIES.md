# File responsibilities — Phase 9 grounded recovery v1.7.2

| File | Responsibility | Boundary |
|---|---|---|
| `run_phase_9.ps1` | Orchestrates approval checks, data readiness, foundation mixing, SFT, progressive KD, optional on-policy recovery, five-condition evaluation, and the pending gate. | Does not make a human release decision. |
| `run_phase_9_grounded_recovery.ps1` | Runs the teacher→warm baseline→overfit→Oracle SFT→Real RAG→optional KD capability ladder. | Does not run narrow architecture controls before the ladder passes. |
| `rebalance_behavior_curriculum.py` | Creates the deterministic 45/25/10/10/5/5 training-only behavior view. | Does not modify reviewed source rows or held-out evaluation. |
| `build_overfit_probe.py` | Builds a unique 12–24-row benchmark and repeated training view for wiring verification. | The same-row result is not held-out quality evidence. |
| `phase9_training_weights.py` | Computes first-token, evidence-anchor, abstention, and completion-boundary emphasis. | Does not add hidden-state or representation losses. |
| `grounded_recovery_metrics.py` | Computes bounded Oracle capability evidence with explicit applicability. | Does not replace case-level reports. |
| `check_grounded_recovery_ladder.py` | Records stage status and unlocks Real RAG, KD, and architecture comparison only after prerequisites. | Human production approval remains separate. |
| `select_grounded_sft_candidate.py` | Selects from the warm SFT Oracle frontier and emits `NO_VALID_QUALITY_FRONTIER` when degenerate. | Does not select narrow students or production-release artifacts. |
| `phase9_common.py` | Validation, routing, prompts, FTS5 retrieval, evidence gates, grounding diagnostics, connected-component splitting, and class-aware holdout assignment. | Not a semantic entailment oracle. |
| `phase9_metrics.py` | Explicit case classes, safe denominators, functional correctness, bounded training comparisons, and readiness blockers. | Does not replace the shared repository opportunity policy. |
| `create_student.py` | Creates and verifies warm depth-copy, random-control, or pretrained-clone initializations. | Random initialization is not language inheritance. |
| `build_knowledge_index.py` | Builds the low-memory FTS5 index and provenance manifest. | Not a neural reranker. |
| `generate_teacher_data.py` | Produces teacher candidates, authoritative prompt targets, diagnostics, provenance, and a review packet. | Real teacher outputs are not auto-approved. |
| `build_pilot_dataset.py` | Converts accepted real rows into leakage-safe, class-covered train/dev/test splits and declares research readiness. | Repeated paraphrases do not count as independent documents. |
| `build_curriculum_dataset.py` | Creates controlled prompt, distractor, missing-evidence, injection, social-rehearsal, and abstention variants from approved answers. | Never invents new facts. |
| `build_foundation_dataset.py` | Materializes the bundled fictional, authored, training-only behavior foundation. | Not held-out evidence or broad pretraining. |
| `merge_training_data.py` | Deterministically weights and interleaves real curriculum with the synthetic foundation. | Does not alter answer text or contracts. |
| `train_student.py` | Runs completion-preserving SFT or exact-tokenizer progressive KD with confidence filtering and completion-boundary weighting. | No cross-tokenizer KL; no truncated target training. |
| `generate_on_policy_data.py` | Creates mixed student/teacher trajectories without hard-label reinforcement of student mistakes. | Optional; not a substitute for SFT. |
| `calibrate_evidence_gate.py` | Chooses a dev-only lexical threshold with explicit positive/negative denominators and product costs. | Threshold is not portable across indexes/domains. |
| `evaluate_rag.py` | Measures closed, oracle, real, filtered-adversarial, and passthrough-adversarial behavior with explicit denominators. | Human factual/safety review remains required. |
| `check_phase_9_gate.py` | Combines shared proportional opportunity scoring with functional readiness and numerically bounded comparisons. | Never auto-promotes research controls. |
| `data/.../foundation/` | 336 fictional authored prompts plus supporting records for behavior training. | Must never contaminate held-out real evaluation. |
| `data/.../canary/` | Small independent package for static/smoke verification of splitting and calibration. | Not a production benchmark. |
