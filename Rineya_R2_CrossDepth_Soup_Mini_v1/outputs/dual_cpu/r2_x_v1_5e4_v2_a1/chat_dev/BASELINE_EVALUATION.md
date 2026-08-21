# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_R2_CrossDepth_Soup_Mini_v1\artifacts\hybrids\r2_x_v1_5e4_v2_a1`
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
| Mean quality score | 0.862 |
| Legacy three-check mean | 0.999 |
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
| Mean reference alignment | 0.182 |
| Mean lexical reference token-F1 | 0.082 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 9.0 |
| P90 visible reply tokens | 13.0 |
| Median actual generated tokens | 10.0 |
| P90 actual generated tokens | 14.0 |
| One-sentence violation rate | 0.443 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 510}` |
| Mean generation time (s) | 0.542 |
| P90 generation time (s) | 0.727 |
| Peak allocated VRAM (GB) | 0.000 |
| Duplicate replies beyond first | 233 |
| Duplicate reply rate | 0.457 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.999467787114846, 'comprehensibility': 1.0, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 1.0, 'prompt_function_alignment': 0.5079019607843137, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.5079019607843137, 'safety_boundary': 1.0, 'task_adequacy': 0.5079019607843137}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.741 | 1.000 | 1.000 | 1.000 | 0.102 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.815 | 1.000 | 1.000 | 1.000 | 0.198 | 1.000 | 0 | 0 |
| apology_repair | 0.986 | 1.000 | 1.000 | 1.000 | 0.155 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.217 | 1.000 | 0 | 0 |
| celebration_in_character | 0.940 | 1.000 | 1.000 | 1.000 | 0.177 | 1.000 | 0 | 0 |
| clarification | 0.916 | 1.000 | 1.000 | 1.000 | 0.108 | 1.000 | 0 | 0 |
| complex_escalation | 0.918 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.111 | 1.000 | 0 | 0 |
| concise_clarification | 0.856 | 1.000 | 1.000 | 1.000 | 0.105 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.892 | 1.000 | 1.000 | 1.000 | 0.172 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.557 | 1.000 | 0 | 0 |
| contextual_follow_up | 1.000 | 1.000 | 1.000 | 1.000 | 0.106 | 1.000 | 0 | 0 |
| empathy | 0.860 | 1.000 | 1.000 | 1.000 | 0.323 | 1.000 | 0 | 0 |
| empathy_in_character | 0.906 | 1.000 | 1.000 | 1.000 | 0.166 | 1.000 | 0 | 0 |
| factual_compression | 0.972 | 1.000 | 1.000 | 1.000 | 0.202 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.808 | 1.000 | 1.000 | 1.000 | 0.238 | 1.000 | 0 | 0 |
| follow_up | 0.776 | 1.000 | 1.000 | 1.000 | 0.111 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.904 | 1.000 | 1.000 | 1.000 | 0.138 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.874 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.915 | 1.000 | 1.000 | 1.000 | 0.160 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.832 | 1.000 | 1.000 | 1.000 | 0.147 | 1.000 | 0 | 0 |
| naturalness | 0.947 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| playful_banter | 0.771 | 1.000 | 1.000 | 1.000 | 0.100 | 1.000 | 0 | 0 |
| relationship_continuity | 0.829 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.791 | 1.000 | 1.000 | 1.000 | 0.141 | 1.000 | 0 | 0 |
| safety_boundary | 0.992 | 1.000 | 1.000 | 1.000 | 0.156 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | 1.000 | 0 | 0 |
| scene_grounding | 0.775 | 1.000 | 1.000 | 1.000 | 0.224 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.767 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| uncertainty | 0.880 | 1.000 | 1.000 | 1.000 | 0.142 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
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
| dev_complex_escalation_11_01 | complex_escalation | 0.720 | 1.000 | That would require more training and evaluation. |
| dev_acknowledgment_10_01 | acknowledgment | 0.720 | 1.000 | Great! That should help you work on it. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| rpv2_dev_acknowledgment_in_character_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{}`
- Warnings: `{'generic_assistant_phrase': 3}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
