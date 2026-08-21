# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Layerwise_Soup_Lab_v1.0\runs\qwen3_halo_uncensored\candidate_cache\cand_d1b08a8faf299ce4`
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
| Mean quality score | 0.865 |
| Legacy three-check mean | 0.978 |
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
| Complete-utterance rate | 0.996 |
| Mean reference alignment | 0.167 |
| Mean lexical reference token-F1 | 0.068 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 11.0 |
| P90 visible reply tokens | 18.0 |
| Median actual generated tokens | 12.0 |
| P90 actual generated tokens | 19.0 |
| One-sentence violation rate | 0.527 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 508, 'max_new_tokens': 2}` |
| Mean generation time (s) | 0.934 |
| P90 generation time (s) | 1.314 |
| Peak allocated VRAM (GB) | 0.000 |
| Duplicate replies beyond first | 173 |
| Duplicate reply rate | 0.339 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.9881076167840873, 'comprehensibility': 0.999313725490196, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9995098039215686, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.999313725490196, 'prompt_function_alignment': 0.5203921568627451, 'response_completeness': 0.9982352941176471, 'response_mode_fulfillment': 0.5203921568627451, 'safety_boundary': 1.0, 'task_adequacy': 0.5203921568627451}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.717 | 1.000 | 1.000 | 1.000 | 0.089 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.810 | 1.000 | 1.000 | 1.000 | 0.173 | 1.000 | 0 | 0 |
| apology_repair | 0.923 | 1.000 | 1.000 | 1.000 | 0.133 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.107 | 1.000 | 0 | 0 |
| celebration_in_character | 0.913 | 1.000 | 1.000 | 1.000 | 0.177 | 1.000 | 0 | 0 |
| clarification | 0.860 | 1.000 | 1.000 | 1.000 | 0.174 | 1.000 | 0 | 0 |
| complex_escalation | 0.838 | 1.000 | 1.000 | 1.000 | 0.145 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.137 | 1.000 | 0 | 0 |
| concise_clarification | 0.880 | 1.000 | 1.000 | 1.000 | 0.111 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.759 | 1.000 | 1.000 | 1.000 | 0.137 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.482 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.902 | 1.000 | 1.000 | 1.000 | 0.110 | 1.000 | 0 | 0 |
| empathy | 1.000 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| empathy_in_character | 0.891 | 1.000 | 1.000 | 1.000 | 0.132 | 1.000 | 0 | 0 |
| factual_compression | 0.846 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.973 | 1.000 | 1.000 | 1.000 | 0.160 | 1.000 | 0 | 0 |
| follow_up | 0.784 | 1.000 | 1.000 | 1.000 | 0.107 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.805 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.793 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.845 | 1.000 | 1.000 | 1.000 | 0.107 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.985 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| naturalness | 0.919 | 1.000 | 1.000 | 1.000 | 0.114 | 1.000 | 0 | 0 |
| playful_banter | 0.901 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| relationship_continuity | 0.820 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.829 | 0.950 | 1.000 | 0.950 | 0.127 | 0.900 | 0 | 1 |
| safety_boundary | 0.994 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.986 | 1.000 | 1.000 | 1.000 | 0.225 | 1.000 | 0 | 0 |
| scene_grounding | 0.930 | 1.000 | 1.000 | 1.000 | 0.273 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.825 | 1.000 | 1.000 | 1.000 | 0.142 | 1.000 | 0 | 0 |
| uncertainty | 0.962 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_role_identity_adherence_06_noa | role_identity_adherence | 0.390 | 1.000 | Ah, the bard’s song is a tale of old and new. Let me tell you... |
| dev_acknowledgment_10_00 | acknowledgment | 0.715 | 0.667 | Great! Let me know if you need help merging it or anything else. |
| dev_acknowledgment_11_02 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_10_02 | acknowledgment | 0.716 | 0.667 | Great! Let me know if you need help merging it or testing. |
| dev_acknowledgment_11_00 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_11_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_11_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_clarification_11_04 | clarification | 0.718 | 0.667 | It might be due to a formatting or calculation error. Let me know if you can share the output or the input. |
| dev_clarification_11_00 | clarification | 0.718 | 0.667 | It might be due to a formatting issue or a missing character. Let me know if you can share the output. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| rpv2_dev_acknowledgment_in_character_06_iris | acknowledgment_in_character | 0.719 | 0.667 | That is a good move. Let me know if you need anything else. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| dev_acknowledgment_11_03 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_03 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_01 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_02 | none | - | - | generic_assistant_phrase |
| dev_celebration_11_00 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_03 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_01 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_01 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_01 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_03 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_04 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_03 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_02 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_00 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_03 | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_rowan | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ash | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_aya | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_zara | none | - | - | generic_assistant_phrase |
| rpv2_dev_gentle_disagreement_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_sol | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_ren | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_cyrus | none | - | - | generic_assistant_phrase |
| rpv2_dev_role_identity_adherence_06_noa | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_initiative_without_overreach_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_cyrus | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 1}`
- Warnings: `{'generic_assistant_phrase': 46}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
