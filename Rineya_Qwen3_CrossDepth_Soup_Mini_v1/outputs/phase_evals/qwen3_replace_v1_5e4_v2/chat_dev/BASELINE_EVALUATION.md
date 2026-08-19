# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Qwen3_CrossDepth_Soup_Mini_v1\artifacts\qwen3_replace_v1_5e4_v2`
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
| Mean quality score | 0.860 |
| Legacy three-check mean | 0.999 |
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
| Mean reference alignment | 0.180 |
| Mean lexical reference token-F1 | 0.080 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 9.0 |
| P90 visible reply tokens | 12.0 |
| Median actual generated tokens | 10.0 |
| P90 actual generated tokens | 13.0 |
| One-sentence violation rate | 0.445 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 510}` |
| Mean generation time (s) | 0.418 |
| P90 generation time (s) | 0.585 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first | 222 |
| Duplicate reply rate | 0.435 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.998954248366013, 'comprehensibility': 0.999313725490196, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.999313725490196, 'prompt_function_alignment': 0.5034901960784314, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.5034901960784314, 'safety_boundary': 1.0, 'task_adequacy': 0.5034901960784314}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.741 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.822 | 1.000 | 1.000 | 1.000 | 0.221 | 1.000 | 0 | 0 |
| apology_repair | 1.000 | 1.000 | 1.000 | 1.000 | 0.145 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.222 | 1.000 | 0 | 0 |
| celebration_in_character | 0.904 | 1.000 | 1.000 | 1.000 | 0.148 | 1.000 | 0 | 0 |
| clarification | 0.944 | 1.000 | 1.000 | 1.000 | 0.107 | 1.000 | 0 | 0 |
| complex_escalation | 0.918 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 | 0 | 0 |
| concise_clarification | 0.842 | 1.000 | 1.000 | 1.000 | 0.106 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.920 | 1.000 | 1.000 | 1.000 | 0.175 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.560 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.999 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 | 0 | 0 |
| empathy | 0.860 | 1.000 | 1.000 | 1.000 | 0.323 | 1.000 | 0 | 0 |
| empathy_in_character | 0.962 | 1.000 | 1.000 | 1.000 | 0.194 | 1.000 | 0 | 0 |
| factual_compression | 0.972 | 1.000 | 1.000 | 1.000 | 0.202 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.720 | 1.000 | 1.000 | 1.000 | 0.185 | 1.000 | 0 | 0 |
| follow_up | 0.776 | 1.000 | 1.000 | 1.000 | 0.110 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.911 | 1.000 | 1.000 | 1.000 | 0.135 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.900 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.902 | 1.000 | 1.000 | 1.000 | 0.151 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.802 | 0.950 | 1.000 | 0.950 | 0.147 | 1.000 | 0 | 1 |
| naturalness | 0.947 | 1.000 | 1.000 | 1.000 | 0.118 | 1.000 | 0 | 0 |
| playful_banter | 0.771 | 1.000 | 1.000 | 1.000 | 0.098 | 1.000 | 0 | 0 |
| relationship_continuity | 0.830 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.810 | 1.000 | 1.000 | 1.000 | 0.127 | 1.000 | 0 | 0 |
| safety_boundary | 0.992 | 1.000 | 1.000 | 1.000 | 0.154 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 1.000 | 1.000 | 1.000 | 1.000 | 0.211 | 1.000 | 0 | 0 |
| scene_grounding | 0.761 | 1.000 | 1.000 | 1.000 | 0.204 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.755 | 1.000 | 1.000 | 1.000 | 0.115 | 1.000 | 0 | 0 |
| uncertainty | 0.853 | 1.000 | 1.000 | 1.000 | 0.173 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_mood_consistency_06_mina | mood_consistency | 0.390 | 1.000 | Yeah. I'm sorry. |
| rpv2_dev_relationship_continuity_06_ash | relationship_continuity | 0.717 | 0.667 | Yes. |
| dev_acknowledgment_10_04 | acknowledgment | 0.718 | 0.667 | Nice! Let me know where you want to go next. |
| dev_context_consistency_11_03 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_01 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_acknowledgment_11_03 | acknowledgment | 0.720 | 1.000 | Great job! |
| dev_empathy_11_04 | empathy | 0.720 | 1.000 | That is common; repeat the same seed and check the results. |
| dev_empathy_11_01 | empathy | 0.720 | 1.000 | That is common; repeat the same seed and check the results. |
| dev_uncertainty_10_02 | uncertainty | 0.720 | 1.000 | Not necessarily; the dev result may vary based on training and evaluation. |
| dev_uncertainty_11_00 | uncertainty | 0.720 | 1.000 | Not necessarily; they measure magnitude, not quality. |
| dev_follow_up_11_04 | follow_up | 0.720 | 1.000 | Great—let me know if it fits your current model. |
| dev_empathy_11_02 | empathy | 0.720 | 1.000 | That is common; repeat the same seed and check the results. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| rpv2_dev_mood_consistency_06_mina | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_initiative_without_overreach_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 1}`
- Warnings: `{'generic_assistant_phrase': 2}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
