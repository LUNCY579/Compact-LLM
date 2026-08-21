# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Qwen3_CrossDepth_Soup_Mini_v1\artifacts\alpha_recipes\late_rineya`
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
| Legacy three-check mean | 0.995 |
| Quality pass rate | 0.998 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.390 |
| P10 case quality | 0.720 |
| Worst-decile mean quality | 0.713 |
| Observable defect-free rate | 0.998 |
| Critical failures | 0 |
| Major issues | 1 |
| Complete-utterance rate | 1.000 |
| Mean reference alignment | 0.175 |
| Mean lexical reference token-F1 | 0.076 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 9.0 |
| P90 visible reply tokens | 15.0 |
| Median actual generated tokens | 10.0 |
| P90 actual generated tokens | 16.0 |
| One-sentence violation rate | 0.453 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 510}` |
| Mean generation time (s) | 0.963 |
| P90 generation time (s) | 1.324 |
| Peak allocated VRAM (GB) | 0.000 |
| Duplicate replies beyond first | 171 |
| Duplicate reply rate | 0.335 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.995359477124183, 'comprehensibility': 0.999313725490196, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9993464052287582, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.999313725490196, 'prompt_function_alignment': 0.5067450980392156, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.5067450980392156, 'safety_boundary': 1.0, 'task_adequacy': 0.5067450980392156}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.718 | 1.000 | 1.000 | 1.000 | 0.085 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.804 | 1.000 | 1.000 | 1.000 | 0.188 | 1.000 | 0 | 0 |
| apology_repair | 0.831 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| celebration_in_character | 0.941 | 1.000 | 1.000 | 1.000 | 0.203 | 1.000 | 0 | 0 |
| clarification | 1.000 | 1.000 | 1.000 | 1.000 | 0.157 | 1.000 | 0 | 0 |
| complex_escalation | 0.894 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.136 | 1.000 | 0 | 0 |
| concise_clarification | 0.886 | 1.000 | 1.000 | 1.000 | 0.155 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.770 | 1.000 | 1.000 | 1.000 | 0.129 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.521 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.950 | 1.000 | 1.000 | 1.000 | 0.141 | 1.000 | 0 | 0 |
| empathy | 1.000 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| empathy_in_character | 0.931 | 1.000 | 1.000 | 1.000 | 0.128 | 1.000 | 0 | 0 |
| factual_compression | 0.846 | 1.000 | 1.000 | 1.000 | 0.137 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.734 | 1.000 | 1.000 | 1.000 | 0.167 | 1.000 | 0 | 0 |
| follow_up | 0.797 | 1.000 | 1.000 | 1.000 | 0.108 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.841 | 1.000 | 1.000 | 1.000 | 0.112 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.881 | 1.000 | 1.000 | 1.000 | 0.129 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.885 | 1.000 | 1.000 | 1.000 | 0.111 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.934 | 0.950 | 1.000 | 0.950 | 0.145 | 1.000 | 0 | 1 |
| naturalness | 0.947 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| playful_banter | 0.885 | 1.000 | 1.000 | 1.000 | 0.119 | 1.000 | 0 | 0 |
| relationship_continuity | 0.821 | 1.000 | 1.000 | 1.000 | 0.141 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.854 | 1.000 | 1.000 | 1.000 | 0.125 | 1.000 | 0 | 0 |
| safety_boundary | 0.993 | 1.000 | 1.000 | 1.000 | 0.128 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.986 | 1.000 | 1.000 | 1.000 | 0.222 | 1.000 | 0 | 0 |
| scene_grounding | 0.860 | 1.000 | 1.000 | 1.000 | 0.262 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.775 | 1.000 | 1.000 | 1.000 | 0.149 | 1.000 | 0 | 0 |
| uncertainty | 0.938 | 1.000 | 1.000 | 1.000 | 0.229 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_mood_consistency_06_finn | mood_consistency | 0.390 | 1.000 | Ah, I know. I'm sorry. |
| dev_acknowledgment_10_02 | acknowledgment | 0.714 | 0.667 | Great! Let me know if you need help merging it into the main branch. |
| dev_acknowledgment_10_00 | acknowledgment | 0.714 | 0.667 | Great! Let me know if you need help merging it into the main branch. |
| dev_acknowledgment_11_02 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_10_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_follow_up_10_00 | follow_up | 0.719 | 0.667 | That is a good start—let me know if you need more layers or help with the model. |
| dev_context_consistency_11_01 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_follow_up_10_02 | follow_up | 0.719 | 0.667 | That is a good start—let me know if you need more layers or help with validation. |
| dev_acknowledgment_11_03 | acknowledgment | 0.720 | 1.000 | Great job! |
| dev_acknowledgment_10_04 | acknowledgment | 0.720 | 1.000 | Great! You're all set. |
| dev_naturalness_10_02 | naturalness | 0.720 | 1.000 | That sounds manageable. Let me know if you need help with it. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| dev_naturalness_10_02 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_01 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_03 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_01 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_02 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_03 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_04 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_03 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_aya | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_noa | none | - | - | generic_assistant_phrase |
| rpv2_dev_mood_consistency_06_finn | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 1}`
- Warnings: `{'generic_assistant_phrase': 25}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
