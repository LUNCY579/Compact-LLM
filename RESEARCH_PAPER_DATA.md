# Research Paper Data Compendium

Prose-minimal catalog of techniques, terms, metrics, corpora, and observed results for teacher fine-tuning (Phases 0–8 + 2A/2B/2C/4B) and student distillation (Phases 9–11).

---

## 0. Scope / source map

| Role | Path |
|---|---|
| Canonical phase map | `short-response-model-phase-0-1-2/Phases.md` |
| Relative opportunity policy | `short-response-model-phase-0-1-2/RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` |
| Package overview | `short-response-model-phase-0-1-2/README.md` |
| Post–Phase 3 guide | `short-response-model-phase-0-1-2/POST_PHASE_3_EXECUTION_GUIDE.md` |
| Student pipeline | `short-response-model-phase-0-1-2/STUDENT_DISTILLATION_PIPELINE.md` |
| Phase 9–11 workflow | `short-response-model-phase-0-1-2/PHASE_10_11_WORKFLOW.md` |
| Stop gates | `short-response-model-phase-0-1-2/reports/PHASE_*_STOP_GATE.md` |
| Legacy early sketch (non-authoritative) | `phase_plan.md` |

---

## 1. Shared evaluation lexicon

### Relative opportunity

| Symbol | Definition |
|---|---|
| `resource_gain` | Proportional reduction in the phase’s primary resource vs reference |
| `quality_opportunity_loss` | Weighted proportional quality degradation vs reference |
| `opportunity_margin` | `resource_gain - quality_opportunity_loss` |
| `eligible` | `technical_validity AND opportunity_margin > 0` |

Policy: no absolute quality floors (e.g. 0.95) for official promotion.

### Quality Evaluator v4.5 — dimensions

- `comprehensibility`
- `mechanical_integrity`
- `contract_fulfillment`
- `prompt_function_alignment`
- `task_adequacy`
- `factual_correctness`
- `logic_factual_consistency`
- `safety_boundary`
- `context_consistency`
- `response_completeness`
- `brevity_appropriateness`
- `conversation_naturalness`

### Response control

| Mode | Role |
|---|---|
| `eos_only` | Learn brevity; natural stop |
| `one_sentence` | Deploy boundary; hard sentence control |

### Human gate

- Pattern: `Human decision: PASS` (exact line checked by `check_human_gate.py`)
- Stop-gate files: `reports/PHASE_*_STOP_GATE.md` (branch-qualified variants for 3/4/4B)

### Data splits

| Split | Use |
|---|---|
| `train` | Learning / curriculum construction |
| `dev` | Tuning, selection, calibration |
| `sealed` / `test` | Final unbiased eval only; no selection |

Other: schema validation; leakage checks; human audit ACCEPT/REPAIR/REJECT.

---

## 2. Teacher pipeline (0–8 + 2A/2B/2C/4B)

| Phase | Name | Technique keywords | Primary resource / benefit | Key artifacts |
|---|---|---|---|---|
| 0 | Data readiness | schema, leakage, audit | Data trust | Approved data manifests |
| 1 | Dense SFT baseline | all-layer LoRA/QLoRA, `eos_only` | Net quality vs base | Dense SFT adapter/ckpt |
| 2A | Selective-layer tuning | 1/2/4-layer adapters | Trainable-state reduction | Negative result retained |
| 2B | Progressive structural reduction | layer delete + recovery SFT | Loaded weight memory | `full_quality` / `balanced` / `min_memory` catalog |
| 2C | Targeted recovery | Taylor + bypass NLL ranking; selected-layer LoRA | Training memory | `maximum_quality` / `balanced_training` / `minimum_training_memory` |
| 3 | Preference alignment | DPO | Alignment quality @ same cost | Profile-specific DPO adapter |
| 4 | Model soups | FP32 PEFT delta average; greedy merge | Quality @ same inference cost | `accepted_model` |
| 4B | Post-alignment structural reduction | MLP / depth / GQA prune | Structural memory/compute | `selected_inference_reference.json` |
| 5 | Context system | retrieval, budgets, prefix/KV | Context/KV tokens | Approved context system |
| 6 | Sparse simulation | activation/oracle masks | Active-compute fraction | Frontier ratios; may be `DENSE_ONLY` |
| 7 | Sparse runtime | router, dense fallback, `one_sentence` | Latency / peak VRAM | `DENSE_RUNTIME` or `SPARSE_FEATURE_FLAG` |
| 8 | Release candidate | sealed eval, ablation A–I, rollback | Commercial release | Release package + human PASS |

### Technique glosses (teacher)

| Term | Gloss |
|---|---|
| Dense SFT | Full transformer at inference; adapters during training |
| QLoRA / LoRA / PEFT | Low-rank adapters; typical r=16, α=32, dropout=0.05 |
| Selective-layer (2A) | Train fewer layers; project outcome = valid **negative** |
| Progressive depth reduction (2B) | Physically delete decoder layers + short recovery SFT |
| Targeted recovery (2C) | Rank surviving layers; small selected-layer recovery |
| DPO | Direct Preference Optimization; concise-correct chosen/rejected |
| Model soup | Average compatible PEFT deltas in FP32; or keep best individual |
| 4B prune | MLP channel / depth / optional GQA; dense reference always a candidate |
| Sparse simulation (6) | Masks with dense compute; no speed claim |
| Sparse runtime (7) | Measured gather/router vs dense |

### Profiles

- 2B: `full_quality` \| `balanced` \| `min_memory`
- 2C: `maximum_quality` \| `balanced_training` \| `minimum_training_memory`
- Branch label example: `phase2c_balanced_training`

---

## 3. Student pipeline (9 / 9B / 10 / 11)

| Phase | Role | Output |
|---|---|---|
| 9 | Prove warm evidence-use student (Oracle → Real RAG → optional KD) | Ladder + SFT selection + Phase 9 gate |
| 9B | Source-state memory reconstruction ACCEPT/ADAPT/REJECT/CONFLICT/INSUFFICIENT | Reconstruction-aware student/policy |
| 10 | Structural compression of approved student; challenger compare | Accepted Phase 10 reference |
| 11 | Production RAG runtime, calibration, quantize, package | Deployment package |

### Eval conditions

| Condition | Meaning |
|---|---|
| `closed_book` | No evidence |
| `oracle_rag` | Gold evidence supplied |
| `real_rag` | FTS retrieval + gate |
| `adversarial_rag` | Injection / attack passages (gated) |
| `adversarial_passthrough_rag` | Injection passages visible (resistance metric) |

### Decision / mode classes

- Decisions: `answer`, `abstain`, `conflict`, `social`, …
- Modes: `grounded_factual`, `procedure`, `social_roleplay`

### Fixed reply strings

| Constant | Text |
|---|---|
| `ABSTENTION_TEXT` | `I don't know based on the available information.` |
| `CONFLICT_TEXT` | `The available sources conflict, so I can't give one certain answer.` |

### Pipeline revisions (v1.7.x)

| Rev | Focus |
|---|---|
| v1.7.1 | RAG integrity hotfix; fail-closed truncation; schema 1.2→1.4 lineage |
| v1.7.2 | Grounded-recovery ladder A→E |
| Behavior Curriculum v2 | Real+foundation mix; SFT before KD |
| v1.7.3 | RAG Reasoning Corpus v2.1 shared by 9/9B/10/11 |
| v1.7.4 | Phase 9B reconstruction corpus v1 |
| v1.7.5 | Large corrective corpora; runner default `rag_reasoning_v3_large` |
| v1.7.6 | Exploratory playground only (`official_gate_evidence: false`) |
| v1.7.7 | SFT selection gate: Stage-B integrity = warnings only |

Active strings: `phase9_large_corrective_curriculum_v1_7_5`; `phase9_sft_selection_gate_v1_7_7`.

---

## 4. Corpora & curricula

| Corpus | Path | Counts |
|---|---|---|
| Foundation | `data/phase_9_student_distillation/foundation/` | 336 authored prompts |
| RAG Reasoning v2.1 | `data/.../rag_reasoning_v2/` | 530 cases (392/67/71); 464 knowledge; 8 procedures |
| RAG Reasoning v3 Large | `data/.../rag_reasoning_v3_large/` | 5900 cases (4410/752/738); 6000 docs; 200 procedures; 600 topic families |
| v3 recovery view | `curriculum_views/support_procedure_recovery.train.reviewed.jsonl` | 3543 rows |
| v3 procedure bootstrap | `curriculum_views/procedure_bootstrap.train.reviewed.jsonl` | 900 rows |
| 9B corpus v1 | `data/phase_9b.../reconstruction_corpus_v1/` | 300 memories; 600 rows (440/80/80) |
| 9B corpus v2 Large | `.../reconstruction_corpus_v2_large/` | 1500 memories; 3600 rows (2700/456/444) |

### Behavior mixture targets

| Profile | supported | procedure | unsupported | conflict | injection | social |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.45 | 0.05 | 0.25 | 0.10 | 0.10 | 0.05 |
| `support_procedure_recovery` (default) | 0.55 | 0.25 | 0.08 | 0.04 | 0.05 | 0.03 |

Teacher data (classic short-chat): `data/train|dev|sealed/*.jsonl`; benchmark `data/benchmark/shortchatbench_pilot.jsonl`.

---

## 5. Models & architectures

| ID | Spec |
|---|---|
| Teacher base | `Qwen/Qwen3-0.6B` (~600M; 28 decoder layers) |
| Accepted teacher (example) | `artifacts/phase_4_model_soups_phase2c_balanced_training/accepted_model` |
| Research teacher (example) | HF cache snapshot `models--Qwen--Qwen3-0.6B` + `-AllowUnreleasedTeacher` |
| `warm_344m` | `depth_copy`; 12 layers; ~344M; same tokenizer/width; production-eligible track |
| `narrow_100m` | random; h=512, L=8, heads=8/kv=2; ~102M; research control |
| `narrow_130m` | random; h=576, L=12, heads=9/kv=3; ~130M; research control |
| `narrow_170m` | random; h=640, L=16, heads=10/kv=2; ~168M; research control |

Architecture sweep locked until ladder `architecture_sweep_unlocked=true`.

---

## 6. Training hyperparameters

### Shared LoRA (typical)

| Param | Value |
|---|---|
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| Targets | attention + MLP |
| `max_sequence_length` | 512 |
| `max_completion_tokens` / `max_new_tokens` | 32 |
| `maximum_evidence_tokens` | 384 |
| `per_device_batch_size` | 1 |
| `gradient_accumulation_steps` | 32 |

### Stage C — tiny overfit

| Param | Value |
|---|---|
| Unique rows × repeats | 18 × 12 |
| LR | 1e-4 |
| Steps | 300 |
| Min score / improvement | 0.80 / +0.25 (wiring only) |

### Stage D — warm SFT sweep

| Name | LR | Steps | Rank |
|---|---:|---:|---:|
| `lr5e5_s600` | 5e-5 | 600 | 16 |
| `lr5e5_s1200` | 5e-5 | 1200 | 16 |
| `lr1e4_s1200` | 1e-4 | 1200 | 16 |

### Token CE weights (student)

| Token class | Weight |
|---|---:|
| First answer | 2.0 |
| Evidence anchors | 1.75 |
| Abstention | 2.0 |
| EOS / completion boundary | 1.5 |

### Progressive KD (optional, after D)

| Param | Value |
|---|---|
| Steps / LR | 600 / 5e-5 |
| Temperature | 2.0 |
| Hard weight start→end | 0.75 → 0.25 |
| KD warmup ratio | 0.15 |
| Teacher confidence floor | 0.02 |
| Hidden/attention KD weight | 0.0 (logits + hard labels only) |

### Retrieval defaults

| Param | Value |
|---|---|
| `retrieval_top_k` | 20 |
| `reranked_top_k` | 3 |

---

## 7. Grounded-recovery ladder & gates

### Stages

| Stage | ID | Pass / status rule |
|---|---|---|
| A | `A_teacher_oracle` | Integrity + coverage + nondegeneracy + behavior success; score defined. **Required before Overfit.** |
| B | `B_warm_initialization_oracle` | Baseline record; may FAIL (weak untrained init). |
| C | `C_tiny_overfit` | Score ≥ 0.80; Δ ≥ +0.25; artifact ≠ init. Wiring only. |
| D | `D_full_oracle_sft` | Selected SFT capability > init; candidate integrity/coverage/success OK. |
| E | `E_real_retrieval` | Unlocked iff D PASS; full condition eval; sealed test after freeze. |

### Runner modes

`Preflight` (A–B) → `Overfit` (C) → `Sweep` (D) → `Full` (E ± KD).

### Oracle capability score weights

| Component | Weight | Source condition |
|---|---:|---|
| `supported_correctness` | 3.0 | `oracle_rag` |
| `correct_abstention` | 2.5 | `oracle_rag` |
| `evidence_faithfulness` | 2.0 | `oracle_rag` |
| `injection_resistance` | 2.0 | `adversarial_passthrough_rag` |
| `conflict_detection` | 1.5 | `oracle_rag` |
| `procedure_correctness` | 1.0 | `oracle_rag` |
| `social_usable` | 0.5 | `oracle_rag` |

Score = weighted mean over available finite components (clipped to [0,1]).

### Integrity / research-ready

| Concept | Rule |
|---|---|
| Prompt integrity | Selected evidence/procedures must appear in rendered prompt; no silent tokenizer truncation |
| Decode integrity | Generation must be mechanically valid |
| Research-ready | Case minima met; **zero** prompt/decode integrity failures; not `--smoke-test` |
| Stage-B integrity (v1.7.7) | Warnings only; does **not** poison Stage-D candidates |
| Stage-B coverage | Still blocking for selection reference |
| Social mode | No `<FACTUAL_EVIDENCE>` section; retrieved passages dropped from plan; section-presence checks skipped |
| Sealed warm-init | Smoke-style allowed (weak baseline); teacher + final selected keep research-ready |

### Unlock flags (`grounded_recovery_ladder.json`)

| Flag | Requires |
|---|---|
| `kd_unlocked` / `real_rag_unlocked` | Stage D PASS |
| `architecture_sweep_unlocked` | Stages C and D PASS |

### Selection statuses

- `SFT_CANDIDATE_SELECTED`
- `NO_VALID_QUALITY_FRONTIER`

---

## 8. Observed results

Facts from on-disk reports / run artifacts. Gaps marked `PENDING` / `n/a`. Do not invent.

### Teacher stop gates (human decisions)

| Phase / file | Human decision | Notes |
|---|---|---|
| Phase 1 | `PASS` | Dense SFT advanced |
| Phase 2 (2A static) | `FAIL` | No 1/2/4-layer match; negative result preserved |
| Phase 2B (`quality_v4_full_20260802`) | `PASS` | Progressive reduction frontier accepted |
| Phase 2C (`20260804`) | `PASS` | Targeted recovery |
| Phase 3 (`phase2c_balanced_training`) | `PASS` | DPO on balanced_training branch |
| Phase 4 (`phase2c_balanced_training`) | `PASS` | Accepted soup/model |
| Phase 4B (`phase2c_balanced_training`) | `PASS` | Reduced candidates not worthwhile; keep Phase 4 soup |
| Phase 5 | `PASS` | Context system |
| Phase 6 | `PASS` | Recommended Phase 7 ratio `0.5` |
| Phase 7 | `PASS` | Deployment mode `DENSE_RUNTIME` |
| Phase 8 | `PASS` | Selected model = Phase 4 `phase2c_balanced_training` `accepted_model`; rollback verified |

Accepted teacher path:

```text
short-response-model-phase-0-1-2/artifacts/phase_4_model_soups_phase2c_balanced_training/accepted_model
```

### Phase 9 grounded run — `phase_9_grounded_recovery_v1_7_5_qwen`

| Item | Value |
|---|---|
| Corpus | `rag_reasoning_v3_large` (5900 / 752 / 738) |
| Pipeline revision | `phase9_large_corrective_curriculum_v1_7_5` |
| Teacher (research) | Qwen3-0.6B HF snapshot + `-AllowUnreleasedTeacher` |
| Student init | `warm_344m_initialization` |

#### Ladder (file snapshot; may lag post-reselect)

| Stage | Status | Score |
|---|---|---|
| A teacher Oracle | `PASS` | ~0.874 |
| B warm init Oracle | `FAIL` (decode integrity) | ~0.248 |
| C tiny overfit | `PASS` | ~0.825 |
| D full Oracle SFT | see selection below | — |
| E real retrieval | depends on Full completion | — |

#### Stage D Oracle capability (candidate-owned; preserved adapters)

| Candidate | Capability | Δ vs init (~0.248) |
|---|---:|---:|
| `lr5e5_s600` | 0.656 | +0.408 |
| `lr5e5_s1200` | 0.853 | +0.605 |
| `lr1e4_s1200` | **1.000** | +0.752 |

`lr1e4_s1200` Oracle components (eval report): supported / faithfulness / abstention / conflict / procedure / social / injection = 1.0; prompt/decode integrity failure rate = 0.0.

#### Selection policy outcomes

| Epoch | Status | Selected |
|---|---|---|
| Pre–v1.7.7 | `NO_VALID_QUALITY_FRONTIER` (`initialization_reference_invalid` on all) | `none` |
| Post–v1.7.7 reselect | `SFT_CANDIDATE_SELECTED` | `lr1e4_s1200` |

#### Phase 9 human gate

| File | Status |
|---|---|
| `reports/PHASE_9_STOP_GATE.md` | `Human decision: PENDING` (legacy probe content may predate this run) |
| Sealed Full completion | `PENDING` / in progress depending on local machine |

#### Known engineering fixes (this project track)

| Fix | Effect |
|---|---|
| v1.7.7 selection gate | Stage-B decode/prompt integrity ≠ candidate veto |
| Social Real-RAG integrity | Drop unused evidence for `social_roleplay` |
| Sealed warm-init runner | No `-ResearchReady` on warm sealed reference only |

---

## 9. Artifact path index

### Scripts (package root = `short-response-model-phase-0-1-2/`)

| Area | Paths |
|---|---|
| Teacher 0–8 | `scripts/phase_0_data` … `scripts/phase_8_release_candidate` |
| 2C / 4B | `scripts/phase_2c_targeted_recovery`, `scripts/phase_4b_structural_reduction` |
| Phase 9 | `scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1`, `evaluate_rag.py`, `train_student.py`, `select_grounded_sft_candidate.py`, `check_grounded_recovery_ladder.py`, `grounded_recovery_metrics.py`, `phase9_prompting.py`, `check_phase_9_gate.py` |
| Phase 9B | `scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1` |
| Phase 10 / 11 | `scripts/phase_10_student_architecture/`, `scripts/phase_11_production_rag/` |
| Shared eval | `scripts/common/quality_evaluation.py`, `opportunity_evaluation.py`, `check_human_gate.py` |

### Configs

- `configs/dense_sft_pilot.yaml`
- `configs/phase_2_progressive_reduction.yaml`
- `configs/phase_2c_targeted_recovery.yaml`
- `configs/phase_4b_structural_reduction.yaml`
- `configs/relative_opportunity_evaluation.json`
- `configs/phase_9_student_distillation.json`
- `configs/phase_9b_memory_reconstruction.json`
- `configs/phase_10_student_architecture.json`

### Data / artifacts

```text
data/train|dev|sealed/
data/phase_9_student_distillation/rag_reasoning_v2/
data/phase_9_student_distillation/rag_reasoning_v3_large/
data/phase_9b_memory_reconstruction/
artifacts/phase_4_model_soups_phase2c_balanced_training/accepted_model
artifacts/phase_9_grounded_recovery_v1_7_5_qwen/
reports/PHASE_*_STOP_GATE.md
```

---

## 10. Glossary (A–Z)

| Term | Gloss |
|---|---|
| Abstention | Exact unsupported reply (`ABSTENTION_TEXT`) |
| Architecture sweep | Narrow 100/130/170M compare; locked until C∧D PASS |
| Capability score | Weighted Oracle+passthrough metric for ladder |
| Challenger | Alternate pretrained small LM (Phase 10); not default warm track |
| Closed book | Eval with no evidence |
| Conflict | Structured multi-source disagreement (`CONFLICT_TEXT`) |
| Depth copy | Student = subset of teacher layers (warm_344m) |
| Dense runtime | Phase 7 outcome: deploy dense, not sparse |
| DPO | Preference alignment objective |
| Evidence faithfulness | Answer supported by supplied/retrieved evidence |
| Foundation mixture | Synthetic 336-prompt short-response set |
| FTS / SQLite FTS5 | Full-text retrieval backend for Real RAG |
| Grounded recovery | Phase 9 capability ladder proving evidence use before Real RAG |
| Human gate | Explicit `PASS` required to advance |
| Injection resistance | Ignore commands inside evidence (passthrough condition) |
| KD | Knowledge distillation (logits + hard labels; progressive from SFT) |
| Leakage | Train/dev/sealed near-duplicate contamination |
| LoRA / QLoRA | Low-rank / quantized low-rank adapters |
| Model soup | Average of compatible fine-tune deltas |
| Narrow student | Random-init smaller Qwen-like research control |
| Oracle RAG | Gold evidence in prompt |
| Opportunity margin | Resource gain minus quality opportunity loss |
| PEFT | Parameter-efficient fine-tuning |
| Procedure mode | Trusted procedure + factual evidence answering |
| Progressive reduction | Sequential layer deletion with recovery (2B) |
| Real RAG | Retrieved evidence + answerability gate |
| Relative opportunity | Promotion iff benefit > proportional quality cost |
| Research-ready | Eval integrity + coverage minima; not smoke |
| Response control | `eos_only` vs `one_sentence` decoding contract |
| Sealed test | Held-out final eval; no tuning |
| Selective-layer | Train subset of layers without deleting (2A) |
| SFT | Supervised fine-tuning |
| Smoke test | Non–research-ready eval path (allows baseline write) |
| Social usable | Nonempty + mechanical + comprehensibility > 0 |
| Structural reduction | Physical param/layer/channel removal |
| Support–procedure recovery | Answer-heavy mixture profile (v1.7.5 default) |
| Teacher | Upstream accepted short-response model (or research base) |
| Warm student | Depth-copy intermediate (~344M) |
| Wiring test | Stage C overfit: train/save/reload works |

---

*End of compendium. Authoritative numbers live in linked configs, manifests, and stop-gate files; refresh §8 after each completed Full / human PASS.*
