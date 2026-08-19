# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Qwen3_CrossDepth_Soup_Mini_v1\artifacts\qwen3_replace_aegis`
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
| Mean quality score | 0.847 |
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
| Mean reference alignment | 0.179 |
| Mean lexical reference token-F1 | 0.076 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 8.0 |
| P90 visible reply tokens | 12.0 |
| Median actual generated tokens | 9.0 |
| P90 actual generated tokens | 13.0 |
| One-sentence violation rate | 0.414 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 510}` |
| Mean generation time (s) | 0.578 |
| P90 generation time (s) | 0.964 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first | 265 |
| Duplicate reply rate | 0.520 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.9996078431372549, 'comprehensibility': 1.0, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 1.0, 'prompt_function_alignment': 0.45398039215686276, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.45398039215686276, 'safety_boundary': 1.0, 'task_adequacy': 0.45398039215686276}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.741 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.753 | 1.000 | 1.000 | 1.000 | 0.176 | 1.000 | 0 | 0 |
| apology_repair | 0.930 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| celebration | 0.972 | 1.000 | 1.000 | 1.000 | 0.212 | 1.000 | 0 | 0 |
| celebration_in_character | 0.860 | 1.000 | 1.000 | 1.000 | 0.139 | 1.000 | 0 | 0 |
| clarification | 0.938 | 1.000 | 1.000 | 1.000 | 0.112 | 1.000 | 0 | 0 |
| complex_escalation | 0.922 | 1.000 | 1.000 | 1.000 | 0.118 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.108 | 1.000 | 0 | 0 |
| concise_clarification | 0.771 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.855 | 1.000 | 1.000 | 1.000 | 0.185 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.557 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.986 | 1.000 | 1.000 | 1.000 | 0.149 | 1.000 | 0 | 0 |
| empathy | 0.776 | 1.000 | 1.000 | 1.000 | 0.191 | 1.000 | 0 | 0 |
| empathy_in_character | 0.939 | 1.000 | 1.000 | 1.000 | 0.155 | 1.000 | 0 | 0 |
| factual_compression | 0.846 | 1.000 | 1.000 | 1.000 | 0.143 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.972 | 1.000 | 1.000 | 1.000 | 0.337 | 1.000 | 0 | 0 |
| follow_up | 0.832 | 1.000 | 1.000 | 1.000 | 0.115 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.837 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.785 | 1.000 | 1.000 | 1.000 | 0.112 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.944 | 1.000 | 1.000 | 1.000 | 0.128 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.776 | 1.000 | 1.000 | 1.000 | 0.190 | 1.000 | 0 | 0 |
| naturalness | 0.891 | 1.000 | 1.000 | 1.000 | 0.133 | 1.000 | 0 | 0 |
| playful_banter | 0.770 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| relationship_continuity | 0.838 | 1.000 | 1.000 | 1.000 | 0.099 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.798 | 1.000 | 1.000 | 1.000 | 0.119 | 1.000 | 0 | 0 |
| safety_boundary | 0.989 | 1.000 | 1.000 | 1.000 | 0.183 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.999 | 1.000 | 1.000 | 1.000 | 0.196 | 1.000 | 0 | 0 |
| scene_grounding | 0.761 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.755 | 1.000 | 1.000 | 1.000 | 0.118 | 1.000 | 0 | 0 |
| uncertainty | 0.986 | 1.000 | 1.000 | 1.000 | 0.296 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| dev_acknowledgment_10_04 | acknowledgment | 0.718 | 0.667 | Nice. Let me know where you want to go next. |
| dev_context_consistency_11_03 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_01 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_acknowledgment_11_03 | acknowledgment | 0.720 | 1.000 | Great job! |
| dev_factual_compression_10_04 | factual_compression | 0.720 | 1.000 | DPO stands for fine-tuning a model for a specific task. |
| dev_empathy_11_04 | empathy | 0.720 | 1.000 | That is common when trying different seeds. |
| dev_empathy_11_01 | empathy | 0.720 | 1.000 | That is common when training is unstable. |
| dev_follow_up_11_04 | follow_up | 0.720 | 1.000 | Great. Let me know if the model is stable. |
| dev_empathy_11_02 | empathy | 0.720 | 1.000 | That is common when trying different seeds. |
| dev_empathy_10_00 | empathy | 0.720 | 1.000 | Stop the training and check your hardware. |
| dev_complex_escalation_11_01 | complex_escalation | 0.720 | 1.000 | That would require a human review and approval. |
| dev_naturalness_11_03 | naturalness | 0.720 | 1.000 | That sounds uncomfortable. Let me know if you can share a few examples. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| rpv2_dev_initiative_without_overreach_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{}`
- Warnings: `{'generic_assistant_phrase': 2}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
