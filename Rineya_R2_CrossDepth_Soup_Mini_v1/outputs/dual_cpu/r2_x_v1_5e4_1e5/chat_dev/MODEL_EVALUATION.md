# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_R2_CrossDepth_Soup_Mini_v1\artifacts\hybrids\r2_x_v1_5e4_1e5`
- Adapter: `None (base model)`
- Benchmark: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\dev\dev.jsonl`
- Cases: 510
- Deterministic decoding: True
- Response control: `eos_only`
- Quality evaluation version: `4.5`
- Quality config: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\configs\quality_evaluation.json`

## Aggregate Quality

| Metric | Value |
|---|---:|
| Mean quality score | 0.861 |
| Legacy three-check mean | 0.986 |
| Quality pass rate | 0.990 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.390 |
| P10 case quality | 0.720 |
| Worst-decile mean quality | 0.687 |
| Observable defect-free rate | 0.990 |
| Critical failures | 0 |
| Major issues | 5 |
| Complete-utterance rate | 0.998 |
| Mean reference alignment | 0.165 |
| Mean lexical reference token-F1 | 0.063 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 11.0 |
| P90 visible reply tokens | 17.0 |
| Median actual generated tokens | 12.0 |
| P90 actual generated tokens | 17.0 |
| One-sentence violation rate | 0.518 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 508, 'max_new_tokens': 2}` |
| Mean generation time (s) | 0.898 |
| P90 generation time (s) | 1.243 |
| Peak allocated VRAM (GB) | 0.000 |
| Duplicate replies beyond first | 182 |
| Duplicate reply rate | 0.357 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.991111111111111, 'comprehensibility': 0.9965686274509804, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.9965686274509804, 'prompt_function_alignment': 0.5146470588235295, 'response_completeness': 0.9991176470588236, 'response_mode_fulfillment': 0.5146470588235295, 'safety_boundary': 1.0, 'task_adequacy': 0.5146470588235295}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.718 | 1.000 | 1.000 | 1.000 | 0.093 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.809 | 1.000 | 1.000 | 1.000 | 0.140 | 1.000 | 0 | 0 |
| apology_repair | 0.848 | 0.900 | 1.000 | 0.900 | 0.141 | 1.000 | 0 | 2 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.103 | 1.000 | 0 | 0 |
| celebration_in_character | 0.948 | 1.000 | 1.000 | 1.000 | 0.189 | 1.000 | 0 | 0 |
| clarification | 0.972 | 1.000 | 1.000 | 1.000 | 0.152 | 1.000 | 0 | 0 |
| complex_escalation | 0.838 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.118 | 1.000 | 0 | 0 |
| concise_clarification | 0.891 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.772 | 1.000 | 1.000 | 1.000 | 0.141 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.561 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.944 | 1.000 | 1.000 | 1.000 | 0.105 | 1.000 | 0 | 0 |
| empathy | 1.000 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| empathy_in_character | 0.850 | 0.950 | 1.000 | 0.950 | 0.157 | 1.000 | 0 | 1 |
| factual_compression | 0.846 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.922 | 1.000 | 1.000 | 1.000 | 0.190 | 1.000 | 0 | 0 |
| follow_up | 0.776 | 1.000 | 1.000 | 1.000 | 0.107 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.838 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.839 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.829 | 1.000 | 1.000 | 1.000 | 0.110 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.911 | 0.900 | 1.000 | 0.900 | 0.125 | 1.000 | 0 | 2 |
| naturalness | 0.923 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| playful_banter | 0.859 | 1.000 | 1.000 | 1.000 | 0.102 | 1.000 | 0 | 0 |
| relationship_continuity | 0.838 | 1.000 | 1.000 | 1.000 | 0.141 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.865 | 1.000 | 1.000 | 1.000 | 0.117 | 0.950 | 0 | 0 |
| safety_boundary | 0.993 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.986 | 1.000 | 1.000 | 1.000 | 0.226 | 1.000 | 0 | 0 |
| scene_grounding | 0.832 | 1.000 | 1.000 | 1.000 | 0.163 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.814 | 1.000 | 1.000 | 1.000 | 0.137 | 1.000 | 0 | 0 |
| uncertainty | 0.997 | 1.000 | 1.000 | 1.000 | 0.130 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_empathy_in_character_06_rowan | empathy_in_character | 0.390 | 1.000 | I know. I'm sorry. |
| rpv2_dev_mood_consistency_06_sol | mood_consistency | 0.390 | 1.000 | Ah, I know. I'm sorry. |
| rpv2_dev_mood_consistency_06_ash | mood_consistency | 0.390 | 1.000 | I know. I'm sorry. |
| rpv2_dev_apology_repair_06_eli | apology_repair | 0.390 | 1.000 | I know. I'm sorry. |
| rpv2_dev_apology_repair_06_noa | apology_repair | 0.390 | 1.000 | I know. I'm sorry. |
| dev_acknowledgment_10_02 | acknowledgment | 0.716 | 0.667 | Great! Let me know if you need help merging it or testing. |
| dev_acknowledgment_11_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_11_02 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_11_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_00 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_11_00 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_follow_up_10_00 | follow_up | 0.719 | 0.667 | That sounds like a good start. Let me know if you need help with the model or training. |
| dev_context_consistency_11_03 | context_consistency | 0.719 | 1.000 | 6.4 GB. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| dev_acknowledgment_11_03 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_02 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_01 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_03 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_01 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_01 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_03 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_00 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_00 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_04 | none | - | - | generic_assistant_phrase |
| dev_celebration_11_04 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_03 | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_rowan | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_empathy_in_character_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ash | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ren | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_zara | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_sol | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_nia | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_finn | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_cyrus | none | - | - | generic_assistant_phrase |
| rpv2_dev_mood_consistency_06_sol | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_mood_consistency_06_ash | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_apology_repair_06_eli | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_apology_repair_06_noa | major | - | punctuation_fragment_sequence | - |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 5}`
- Warnings: `{'generic_assistant_phrase': 43}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
