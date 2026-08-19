# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Qwen3_CrossDepth_Soup_Mini_v1\artifacts\qwen3_replace_v1_5e4_1e5`
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
| Mean quality score | 0.864 |
| Legacy three-check mean | 0.997 |
| Quality pass rate | 1.000 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.718 |
| P10 case quality | 0.720 |
| Worst-decile mean quality | 0.720 |
| Observable defect-free rate | 1.000 |
| Critical failures | 0 |
| Major issues | 0 |
| Complete-utterance rate | 1.000 |
| Mean reference alignment | 0.177 |
| Mean lexical reference token-F1 | 0.076 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 9.0 |
| P90 visible reply tokens | 13.0 |
| Median actual generated tokens | 10.0 |
| P90 actual generated tokens | 14.0 |
| One-sentence violation rate | 0.465 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 510}` |
| Mean generation time (s) | 0.629 |
| P90 generation time (s) | 0.949 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first | 245 |
| Duplicate reply rate | 0.480 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.9986834733893557, 'comprehensibility': 1.0, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 1.0, 'prompt_function_alignment': 0.5129019607843137, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.5129019607843137, 'safety_boundary': 1.0, 'task_adequacy': 0.5129019607843137}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.719 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.796 | 1.000 | 1.000 | 1.000 | 0.184 | 1.000 | 0 | 0 |
| apology_repair | 0.986 | 1.000 | 1.000 | 1.000 | 0.135 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.175 | 1.000 | 0 | 0 |
| celebration_in_character | 0.871 | 1.000 | 1.000 | 1.000 | 0.164 | 1.000 | 0 | 0 |
| clarification | 0.972 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| complex_escalation | 0.922 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.114 | 1.000 | 0 | 0 |
| concise_clarification | 0.899 | 1.000 | 1.000 | 1.000 | 0.109 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.772 | 1.000 | 1.000 | 1.000 | 0.139 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.557 | 1.000 | 0 | 0 |
| contextual_follow_up | 1.000 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 | 0 | 0 |
| empathy | 0.804 | 1.000 | 1.000 | 1.000 | 0.236 | 1.000 | 0 | 0 |
| empathy_in_character | 0.951 | 1.000 | 1.000 | 1.000 | 0.197 | 1.000 | 0 | 0 |
| factual_compression | 0.922 | 1.000 | 1.000 | 1.000 | 0.196 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.808 | 1.000 | 1.000 | 1.000 | 0.176 | 1.000 | 0 | 0 |
| follow_up | 0.776 | 1.000 | 1.000 | 1.000 | 0.114 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.865 | 1.000 | 1.000 | 1.000 | 0.114 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.797 | 1.000 | 1.000 | 1.000 | 0.138 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.916 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.804 | 1.000 | 1.000 | 1.000 | 0.134 | 1.000 | 0 | 0 |
| naturalness | 0.919 | 1.000 | 1.000 | 1.000 | 0.130 | 1.000 | 0 | 0 |
| playful_banter | 0.733 | 1.000 | 1.000 | 1.000 | 0.098 | 1.000 | 0 | 0 |
| relationship_continuity | 0.944 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.861 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| safety_boundary | 0.992 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.992 | 1.000 | 1.000 | 1.000 | 0.160 | 1.000 | 0 | 0 |
| scene_grounding | 0.859 | 1.000 | 1.000 | 1.000 | 0.336 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.847 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| uncertainty | 0.964 | 1.000 | 1.000 | 1.000 | 0.221 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| dev_acknowledgment_10_04 | acknowledgment | 0.718 | 0.667 | Nice. Let me know if you need help with it. |
| dev_acknowledgment_10_01 | acknowledgment | 0.718 | 0.667 | Great! Let me know if you need help with it. |
| dev_acknowledgment_10_00 | acknowledgment | 0.718 | 0.667 | Great. Let me know if you need help with it. |
| dev_context_consistency_11_03 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_01 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| rpv2_dev_acknowledgment_in_character_06_nia | acknowledgment_in_character | 0.719 | 0.667 | That is a good move. The east hall is closer to the sun. |
| dev_acknowledgment_11_03 | acknowledgment | 0.720 | 1.000 | Great! That should help with training. |
| dev_factual_compression_10_04 | factual_compression | 0.720 | 1.000 | A fine-tuned training paradigm for fine-grained model adaptation. |
| dev_empathy_11_04 | empathy | 0.720 | 1.000 | That is common when trying different seeds. |
| dev_empathy_11_01 | empathy | 0.720 | 1.000 | That is common when trying different seeds. |
| dev_uncertainty_10_02 | uncertainty | 0.720 | 1.000 | Not necessarily; test them first. |
| dev_follow_up_11_04 | follow_up | 0.720 | 1.000 | Great! Let me know if you need help with the model or training. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| dev_acknowledgment_10_04 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_01 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| rpv2_dev_trust_and_secrets_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_jun | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{}`
- Warnings: `{'generic_assistant_phrase': 9}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
