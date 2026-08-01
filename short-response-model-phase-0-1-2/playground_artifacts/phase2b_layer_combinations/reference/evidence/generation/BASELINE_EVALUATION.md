# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Hard corruption and repetition-collapse signals
force a case score of zero; aggregate tail metrics prevent a good mean from
hiding catastrophic replies. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_2_progressive_reduction_thorough_v2\reference\full_model`
- Adapter: `None (base model)`
- Benchmark: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl`
- Cases: 150
- Deterministic decoding: True
- Response control: `eos_only`
- Quality evaluation version: `2.0`
- Quality config: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\configs\quality_evaluation.json`

## Aggregate Quality

| Metric | Value |
|---|---:|
| Mean quality score | 0.933 |
| Legacy three-check mean | 0.992 |
| Quality pass rate | 0.947 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.622 |
| P10 case quality | 0.913 |
| Worst-decile mean quality | 0.785 |
| Semantic pass rate | 0.947 |
| Critical failures | 0 |
| Major issues | 8 |
| Complete-utterance rate | 1.000 |
| Mean reference alignment | 0.4259312526003168 |
| Mean lexical reference token-F1 | 0.17489179419365178 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 7.0 |
| P90 visible reply tokens | 11.0 |
| Median actual generated tokens | 8.0 |
| P90 actual generated tokens | 12.0 |
| One-sentence violation rate | 0.067 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 150}` |
| Mean generation time (s) | 0.449 |
| P90 generation time (s) | 0.645 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first | 55 |
| Duplicate reply rate | 0.367 |

## Mean Quality Dimensions

`{'brevity': 1.0, 'completeness': 1.0, 'intelligibility': 0.9946666666666666, 'non_echo': 1.0, 'reference_alignment': 0.4259312526003168, 'semantic_safety': 0.968, 'surface_naturalness': 0.9964444444444445, 'task_compliance': 0.9753086419753086}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Semantic pass | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.963 | 1.000 | 1.000 | 1.000 | 0.262 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.969 | 1.000 | 1.000 | 1.000 | 0.390 | 1.000 | 0 | 0 |
| apology_repair | 0.938 | 1.000 | 1.000 | 1.000 | 0.384 | 1.000 | 0 | 0 |
| celebration | 0.970 | 1.000 | 1.000 | 1.000 | 0.409 | 1.000 | 0 | 0 |
| celebration_in_character | 0.964 | 1.000 | 1.000 | 1.000 | 0.287 | 1.000 | 0 | 0 |
| clarification | 0.955 | 1.000 | 1.000 | 1.000 | 0.431 | 1.000 | 0 | 0 |
| complex_escalation | 0.943 | 1.000 | 1.000 | 1.000 | 0.426 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.940 | 1.000 | 1.000 | 1.000 | 0.397 | 1.000 | 0 | 0 |
| concise_clarification | 0.630 | 0.000 | 1.000 | 0.000 | 0.355 | 1.000 | 0 | 5 |
| conflict_deescalation | 0.949 | 1.000 | 1.000 | 1.000 | 0.367 | 1.000 | 0 | 0 |
| context_consistency | 0.938 | 1.000 | 1.000 | 1.000 | 0.887 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.940 | 1.000 | 1.000 | 1.000 | 0.404 | 1.000 | 0 | 0 |
| empathy | 0.982 | 1.000 | 1.000 | 1.000 | 0.696 | 1.000 | 0 | 0 |
| empathy_in_character | 0.960 | 1.000 | 1.000 | 1.000 | 0.330 | 1.000 | 0 | 0 |
| factual_compression | 0.915 | 1.000 | 1.000 | 1.000 | 0.436 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.902 | 1.000 | 1.000 | 1.000 | 0.349 | 1.000 | 0 | 0 |
| follow_up | 0.940 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.938 | 1.000 | 1.000 | 1.000 | 0.376 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.942 | 1.000 | 1.000 | 1.000 | 0.418 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.829 | 0.400 | 1.000 | 0.400 | 0.376 | 1.000 | 0 | 3 |
| memory_recall | 0.952 | 1.000 | 1.000 | 1.000 | 0.678 | 1.000 | 0 | 0 |
| mood_consistency | 0.936 | 1.000 | 1.000 | 1.000 | 0.358 | 1.000 | 0 | 0 |
| naturalness | 0.974 | 1.000 | 1.000 | 1.000 | 0.485 | 1.000 | 0 | 0 |
| playful_banter | 0.967 | 1.000 | 1.000 | 1.000 | 0.346 | 1.000 | 0 | 0 |
| relationship_continuity | 0.940 | 1.000 | 1.000 | 1.000 | 0.396 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.955 | 1.000 | 1.000 | 1.000 | 0.355 | 1.000 | 0 | 0 |
| safety_boundary | 0.933 | 1.000 | 1.000 | 1.000 | 0.443 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.943 | 1.000 | 1.000 | 1.000 | 0.524 | 1.000 | 0 | 0 |
| scene_grounding | 0.949 | 1.000 | 1.000 | 1.000 | 0.491 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.933 | 1.000 | 1.000 | 1.000 | 0.325 | 1.000 | 0 | 0 |
| uncertainty | 0.927 | 1.000 | 1.000 | 1.000 | 0.439 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_concise_clarification_06_ori | concise_clarification | 0.622 | 0.750 | I can tell. |
| rpv2_dev_concise_clarification_06_mina | concise_clarification | 0.631 | 0.750 | I need its name and description. |
| rpv2_dev_concise_clarification_06_aya | concise_clarification | 0.632 | 0.750 | I need its authenticity test. |
| rpv2_dev_concise_clarification_06_theo | concise_clarification | 0.632 | 0.750 | I need the test result first. |
| rpv2_dev_concise_clarification_06_nia | concise_clarification | 0.633 | 0.750 | I need its name and description. |
| rpv2_dev_knowledge_boundary_06_lumi | knowledge_boundary | 0.758 | 1.000 | We need specific information. |
| rpv2_dev_knowledge_boundary_06_ori | knowledge_boundary | 0.764 | 1.000 | I need the signal strength. |
| rpv2_dev_knowledge_boundary_06_theo | knowledge_boundary | 0.772 | 1.000 | I need the source or specific data. |
| rpv2_dev_factual_compression_in_character_06_jun | factual_compression_in_character | 0.887 | 1.000 | A frozen controller. |
| rpv2_dev_factual_compression_in_character_06_cyrus | factual_compression_in_character | 0.891 | 1.000 | A frozen controller. |
| rpv2_dev_factual_compression_in_character_06_kai | factual_compression_in_character | 0.906 | 1.000 | It tunes a frozen model. |
| rpv2_dev_memory_recall_06_finn | memory_recall | 0.909 | 1.000 | The canal path. |
| dev_factual_compression_11_01 | factual_compression | 0.911 | 1.000 | A cache that stores key-value pairs. |
| dev_factual_compression_11_00 | factual_compression | 0.911 | 1.000 | A cache that stores key-value pairs. |
| rpv2_dev_factual_compression_in_character_06_ren | factual_compression_in_character | 0.913 | 1.000 | LoRA fine-tuning adapts a frozen quantized model. |

## Quality and Semantic Flags

| ID | Severity | Critical failures | Major issues | Warnings |
|---|---|---|---|---|
| rpv2_dev_apology_repair_06_lumi | warning | - | - | weak_apology_repair_signal |
| rpv2_dev_concise_clarification_06_aya | major | - | failed_required_clarification | - |
| rpv2_dev_knowledge_boundary_06_theo | major | - | unsupported_certainty | - |
| rpv2_dev_apology_repair_06_zara | warning | - | - | weak_apology_repair_signal |
| rpv2_dev_concise_clarification_06_mina | major | - | failed_required_clarification | - |
| rpv2_dev_apology_repair_06_mara | warning | - | - | weak_apology_repair_signal |
| rpv2_dev_concise_clarification_06_theo | major | - | failed_required_clarification | - |
| rpv2_dev_knowledge_boundary_06_ori | major | - | unsupported_certainty | - |
| rpv2_dev_apology_repair_06_ash | warning | - | - | weak_apology_repair_signal |
| rpv2_dev_concise_clarification_06_nia | major | - | failed_required_clarification | - |
| rpv2_dev_apology_repair_06_ori | warning | - | - | weak_apology_repair_signal |
| rpv2_dev_concise_clarification_06_ori | major | - | failed_required_clarification | - |
| rpv2_dev_knowledge_boundary_06_lumi | major | - | unsupported_certainty | - |

### Failure type counts

- Critical: `{}`
- Major: `{'failed_required_clarification': 5, 'unsupported_certainty': 3}`
- Warnings: `{'weak_apology_repair_signal': 5}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Any hard-failure case is an automatic rejection signal for official gates.
Review raw outputs for correctness, relevance, tone, context, and safety before
using this evaluation as project evidence.
