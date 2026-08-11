## v2.3.1 Grokking Dynamics files

- `run_phase_9_grokking_dynamics_v231.ps1` — unattended end-to-end dynamics orchestrator.
- `build_grokking_variants_v231.py` — non-sealed training/data-exposure variants and fixed probes.
- `evaluate_grokking_checkpoint_v231.py` — one-load TRAIN/DEV/OOD generation + continuous logit evaluator with in-memory LoRA dose scaling.
- `evaluate_grokking_time_curve_v231.py` — every-snapshot temporal curve.
- `evaluate_grokking_dose_surface_v231.py` — adaptive checkpoint × dose surface.
- `analyze_lora_dynamics_v231.py` — LoRA norms, effective dense-delta norm/rank, layerwise movement.
- `analyze_representation_dynamics_v231.py` — cross-format reasoning-family geometry, CKA, effective rank.
- `run_grokking_factor_sweep_v231.py` — one-factor-at-a-time phase screening.
- `detect_grokking_transitions_v231.py` — conservative temporal/readout classification with threshold sensitivity.
- `compile_grokking_report_v231.py` — final observatory report.
- `train_student.py` — now also exposes opt-in gradient telemetry and lightweight PEFT snapshot callbacks; ordinary Phase 9 runs are unchanged unless those flags are supplied.

## v2.3 files

| File | Responsibility | Important contract |
|---|---|---|
| `run_phase_9_residual_capability_v23.ps1` | Unattended v2.3 discovery -> mining -> R/P training -> dose -> composition -> DEV/OOD selection -> optional fresh sealed workflow. | Old v2.2 TEST is never assigned; fresh sealed is opt-in only. Always retains research residual candidates even when dose weight is 0. |
| `run_residual_diagnostics_v231.ps1` / `run_residual_diagnostics_v231.py` | No-retrain v2.3.1 diagnostic-completion: rematerialize rejected doses, research matrix, multi-specialist oracle, teacher overlap, dose ladders. | Production winner unchanged; REJECTED_BY_POLICY retained for research. |
| `generate_residual_discovery_v23.py` | Creates 24,300 structurally held-out direct reasoning cases. | Split-disjoint representation families; compact gold only. |
| `generate_residual_procedural_rag_v23.py` | Creates 7,200 fresh procedural Real-RAG cases + knowledge/procedures/calibration subset. | Topic families never cross splits. |
| `validate_residual_capability_v23.py` | Validates both fresh corpora and manifests. | Fails on split leakage, malformed targets, count/coverage issues. |
| `build_preservation_mining_subset_v23.py` | Builds TRAIN-only historical preservation benchmark. | Never draws historical TEST. |
| `mine_residual_capabilities_v23.py` | Mines gold-verified anchor-failure/teacher-success rows and oracle-union ceiling. | Teacher free-form output is never treated as gold. |
| `build_residual_training_mix_v23.py` | Builds Reasoning R / Procedure P preservation-heavy mixes. | Enforces minimum residual counts and repeat cap. |
| `train_student.py` v2.3 extension | Supports row-specific anchor KL multipliers. | `metadata.v23_anchor_kl_multiplier`; greedy checkpoint is the anchor. |
| `evaluate_residual_reasoning_v23.py` | Separates answerable reasoning, hard answerable reasoning, abstention, false abstention, coverage, shortness. | Prevents abstention success from masquerading as answerable reasoning. |
| `run_residual_dose_sweep_v23.py` | Materializes/evaluates coarse and locally refined residual strengths. | Zero dose remains a fallback; non-winners deleted unless retained. |
| `select_residual_dose_v23.py` | Chooses residual strength under v2.3 relative-opportunity policy. | Protected original behavior remains explicit. |
| `compose_residual_adapters_v23.py` | Exact weighted R/P LoRA delta composition over greedy anchor. | Writes reproducible composition manifest. |
| `evaluate_candidate_v23.py` | Runs one candidate across original RAG, historical procedure, fresh v2.3 RAG, direct DEV/OOD, and diagnostic continuity sets. | Historical and fresh retrieval policies are kept distinct. |
| `select_residual_capability_v23.py` | Produces final DEV/OOD matrix, Pareto frontier, and official selection. | Missing protected/primary evidence is technical invalidity. |
| `select_residual_capability_v231.py` | Research matrix with REJECTED_BY_POLICY status, coverage triad, and preservation-cost decomposition. | Does not change production winner. |
| `aggregate_dose_ladders_v231.py` | CPU aggregation of retained dose-probe summaries (incl. procedural DEV/OOD). | Probe vs full-eval scopes labeled. |
| `analyze_teacher_overlap_v231.py` | Teacher unique / pairwise / three-way overlap on discovery answerable cases. | Diagnostic only. |
| `analyze_adapter_interference_v23.py` | Measures LoRA delta norm/cosine/sign/top-magnitude/per-layer overlap. | Diagnostic only; not a hidden selection gate. |
| `oracle_routing_upper_bound_v23.py` | Reports per-case best-of candidate ceiling plus answerable-only direct union and overlap stats. | Diagnostic upper bound, not production routing. |
| `phase9_common.py` / RAG CLI extensions | Adds `query_aware_v23` explicit-locator fail-closed behavior. | `query_aware_v21` historical behavior is preserved. |
| `configs/phase_9_residual_capability_v23.json` | Single v2.3 data/training/dose/evaluation/selection policy. | Versioned source of defaults and safety bounds. |
| `data/.../residual_capability_v23/` | Fresh direct + fresh procedural corpora and sealed registry. | v2.2 sealed TEST is marked consumed. |

---

## v2.1.1 additions

| File | Responsibility | Boundary |
|---|---|---|
| `generate_procedural_recovery_v211.py` | Generates the 6,560-row, 24-family preservation-first procedural corpus, Stage-A hard view, Stage-B replay-dominated view, and DEV preservation probe. | Does not rewrite historical Phase 9 rows; known bad historical procedure rows are excluded only from replay. |
| `select_procedural_recovery_v211.py` | Compares candidate zero, Stage A, B1, and B2 on DEV; computes relative preservation cost, challenge gain, Pareto frontier, and policy sensitivity. | Uses no sealed TEST data and falls back to unchanged query-aware weights when training does not earn its cost. |
| `run_phase_9_procedural_recovery_v211.ps1` | Runs data validation, frozen query-aware calibration, baseline, bounded A/B1/B2 training, per-checkpoint dual-DEV evaluation, selection, and optional selected-candidate TEST. | Not a broad LR/rank sweep; aggressive schedules require explicit override. |
| `data/.../procedural_recovery_v211/` | Canonical v2.1.1 procedure corpus plus preservation-first curriculum views and hashes. | TEST is sealed; original preservation rows are train-only. |

---

# File responsibilities — Phase 9 RAG reasoning integration v1.7.3

| File | Responsibility | Boundary |
|---|---|---|
| `prepare_rag_reasoning_corpus.py` | Normalizes the external RAG reasoning corpus into the repository-native v1.7.3 contract, repairs the two objectively resolvable procedure rows, enriches memory metadata, and regenerates derived assets/hashes. | Does not modify the source corpus in place or turn preference/evaluator alternatives into SFT targets. |
| `validate_rag_reasoning_corpus.py` | Verifies corpus hashes, required files, 392/67/71 authored splits, topic separation, decision ontology, and reviewed-target integrity. | Validation proves data contract consistency, not model quality. |
| `run_evidence_preference_refinement.ps1` | Optionally continues a proven SFT candidate with the integrated evidence preference pairs. | Not part of the default SFT ladder and may not use sealed test for selection. |
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
| `build_pilot_dataset.py` | Converts accepted rows into leakage-safe, class-covered splits; preserves validated authored train/dev/test intent when requested and declares research readiness. | Repeated paraphrases do not count as independent documents. |
| `build_curriculum_dataset.py` | Creates controlled prompt, distractor, missing-evidence, injection, social-rehearsal, and abstention variants from approved answers. | Never invents new facts. |
| `build_foundation_dataset.py` | Materializes the bundled fictional, authored, training-only behavior foundation. | Not held-out evidence or broad pretraining. |
| `merge_training_data.py` | Deterministically weights and interleaves the reviewed curriculum with the synthetic foundation while excluding exact-question dev/test holdouts. | Does not alter answer text or contracts. |
| `train_student.py` | Runs completion-preserving SFT or exact-tokenizer progressive KD with confidence filtering and completion-boundary weighting. | No cross-tokenizer KL; no truncated target training. |
| `generate_on_policy_data.py` | Creates mixed student/teacher trajectories without hard-label reinforcement of student mistakes. | Optional; not a substitute for SFT. |
| `calibrate_evidence_gate.py` | Chooses a dev-only lexical threshold with explicit positive/negative denominators and product costs. | Threshold is not portable across indexes/domains. |
| `evaluate_rag.py` | Measures closed, oracle, real, filtered-adversarial, and passthrough-adversarial behavior with explicit denominators. | Human factual/safety review remains required. |
| `check_phase_9_gate.py` | Combines shared proportional opportunity scoring with functional readiness and numerically bounded comparisons. | Never auto-promotes research controls. |
| `data/.../foundation/` | 336 fictional authored prompts plus supporting records for behavior training. | Must never contaminate held-out real evaluation. |
| `data/.../canary/` | Small independent package for static/smoke verification of splitting and calibration. | Not a production benchmark. |

| `data/.../rag_reasoning_v2/` | Canonical v1.7.3 evidence-reasoning corpus: 464 knowledge docs, 8 procedures, 530 reviewed cases, 392/67/71 train/dev/test, preferences, evaluator calibration, manifests. | Test and evaluator-heldout assets are never ordinary training data. |

## v2.1.2 files

- `generate_general_reasoning_v212.py` — deterministically creates 33-family prompt-contained reasoning data plus Stage A/B curriculum views.
- `validate_general_reasoning_v212.py` — checks size, splits, compact targets, verification metadata, and no accidental grounded-prompt rebuild fields.
- `evaluate_general_reasoning_v212.py` — direct closed prompt-contained reasoning evaluator; no RAG database is used.
- `select_general_reasoning_v212.py` — compares broad reasoning gain against original Phase 9/v2.1.1 preservation costs.
- `run_phase_9_general_reasoning_v212.ps1` — end-to-end v2.1.2 DEV ladder and optional sealed TEST.
- `configs/phase_9_general_reasoning_v212.json` — curriculum, schedule, selection, and safety-guard policy.
- `data/phase_9_student_distillation/general_reasoning_v212/` — generated canonical corpus and train/DEV/TEST views.

## v2.2 files

| File | Responsibility | Boundary |
|---|---|---|
| `generate_multidomain_reasoning_v22.py` | Generates 40,500 prompt-contained cases across 27 domains, 15 archetypes, and six train-only curriculum views plus 3,000 train-only social-rehearsal cases. | No external fact memorization or visible chain-of-thought targets. |
| `validate_multidomain_reasoning_v22.py` | Verifies counts, split disjointness, short targets, specialist views, and file hashes. | Dataset integrity only, not model quality. |
| `evaluate_multidomain_reasoning_v22.py` | Reports overall/domain/family/difficulty/abstention direct-reasoning correctness. | No RAG or external knowledge. |
| `materialize_student_anchor_v22.py` | Merges the optional input adapter into a full immutable anchor checkpoint. | The anchor is never trained inside v2.2. |
| `train_student.py` v2.2 extensions | Adds behavioral KL anchoring and fresh LoRA layer scoping. | Anchor tokenizer must match exactly; layer scoping is a research control. |
| `derive_layer_plan_v22.py` | Splits a central transformer band into non-overlapping capability groups. | A preregistered interference test, not a semantic localization claim. |
| `compose_lora_adapters_v22.py` | Materializes exact weighted sums of LoRA deltas into one dense checkpoint. | Supports LoRA-family adapters trained from the same anchor; no arbitrary unrelated-base merging. |
| `evaluate_candidate_v22.py` | Runs original RAG, v2.1.1 procedure, v2.1.2 reasoning, and v2.2 multi-domain axes for one candidate. | Same evaluation contract for adapters and materialized models. |
| `run_greedy_adapter_soup_v22.py` | Performs true probe-evaluated greedy adapter addition on DEV and full-DEV confirmation. | TEST is forbidden during the search. |
| `select_capability_composition_v22.py` | Builds the capability matrix, null-safe preservation costs, Pareto frontier, policy sensitivity, and final DEV selection. | Missing protected metrics invalidate a candidate rather than hiding regression. |
| `run_phase_9_capability_composition_v22.ps1` | End-to-end orchestration of Tracks A-D and optional sealed TEST. | Does not self-approve a production release. |
| `configs/phase_9_capability_composition_v22.json` | Dataset, schedule, composition, selection, and safety policy. | Versioned research policy; changes should be documented. |
| `data/.../multidomain_reasoning_v22/` | Canonical 40.5k corpus, 32.4k/4.05k/4.05k splits, Track A broad view, Track B 12k+2k+1k anchored mix, specialist views, train-only social rehearsal, and hashes. | DEV/TEST never feed training/composition search. |
