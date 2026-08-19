# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Halo-1.2.0-devfreeze`
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
| Legacy three-check mean | 0.985 |
| Quality pass rate | 0.996 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.390 |
| P10 case quality | 0.720 |
| Worst-decile mean quality | 0.706 |
| Observable defect-free rate | 0.996 |
| Critical failures | 0 |
| Major issues | 2 |
| Complete-utterance rate | 1.000 |
| Mean reference alignment | 0.170 |
| Mean lexical reference token-F1 | 0.071 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 11.0 |
| P90 visible reply tokens | 17.0 |
| Median actual generated tokens | 12.0 |
| P90 actual generated tokens | 18.0 |
| One-sentence violation rate | 0.529 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 509, 'max_new_tokens': 1}` |
| Mean generation time (s) | 0.536 |
| P90 generation time (s) | 0.771 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first | 185 |
| Duplicate reply rate | 0.363 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.9907983193277311, 'comprehensibility': 0.9986274509803922, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9991830065359477, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.9986274509803922, 'prompt_function_alignment': 0.5117058823529412, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.5117058823529412, 'safety_boundary': 1.0, 'task_adequacy': 0.5117058823529412}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.717 | 1.000 | 1.000 | 1.000 | 0.090 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.825 | 1.000 | 1.000 | 1.000 | 0.143 | 1.000 | 0 | 0 |
| apology_repair | 0.905 | 1.000 | 1.000 | 1.000 | 0.142 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| celebration_in_character | 0.933 | 1.000 | 1.000 | 1.000 | 0.202 | 1.000 | 0 | 0 |
| clarification | 0.972 | 1.000 | 1.000 | 1.000 | 0.153 | 1.000 | 0 | 0 |
| complex_escalation | 0.866 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.745 | 1.000 | 1.000 | 1.000 | 0.138 | 1.000 | 0 | 0 |
| concise_clarification | 0.855 | 1.000 | 1.000 | 1.000 | 0.108 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.755 | 1.000 | 1.000 | 1.000 | 0.138 | 1.000 | 0 | 0 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.561 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.944 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| empathy | 1.000 | 1.000 | 1.000 | 1.000 | 0.123 | 1.000 | 0 | 0 |
| empathy_in_character | 0.927 | 1.000 | 1.000 | 1.000 | 0.139 | 1.000 | 0 | 0 |
| factual_compression | 0.846 | 1.000 | 1.000 | 1.000 | 0.122 | 1.000 | 0 | 0 |
| factual_compression_in_character | 0.873 | 1.000 | 1.000 | 1.000 | 0.178 | 1.000 | 0 | 0 |
| follow_up | 0.720 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.842 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.855 | 1.000 | 1.000 | 1.000 | 0.128 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.803 | 1.000 | 1.000 | 1.000 | 0.106 | 1.000 | 0 | 0 |
| memory_recall | 0.720 | 1.000 | 1.000 | 1.000 | 0.652 | 1.000 | 0 | 0 |
| mood_consistency | 0.925 | 0.900 | 1.000 | 0.900 | 0.118 | 1.000 | 0 | 2 |
| naturalness | 0.942 | 1.000 | 1.000 | 1.000 | 0.119 | 1.000 | 0 | 0 |
| playful_banter | 0.915 | 1.000 | 1.000 | 1.000 | 0.124 | 1.000 | 0 | 0 |
| relationship_continuity | 0.796 | 1.000 | 1.000 | 1.000 | 0.138 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.853 | 1.000 | 1.000 | 1.000 | 0.110 | 1.000 | 0 | 0 |
| safety_boundary | 0.993 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.986 | 1.000 | 1.000 | 1.000 | 0.223 | 1.000 | 0 | 0 |
| scene_grounding | 0.874 | 1.000 | 1.000 | 1.000 | 0.277 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.803 | 1.000 | 1.000 | 1.000 | 0.163 | 1.000 | 0 | 0 |
| uncertainty | 0.913 | 1.000 | 1.000 | 1.000 | 0.130 | 1.000 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| rpv2_dev_mood_consistency_06_sol | mood_consistency | 0.390 | 1.000 | Ah, I know. I'm sorry. |
| rpv2_dev_mood_consistency_06_ash | mood_consistency | 0.390 | 1.000 | I know. I'm sorry. |
| dev_acknowledgment_11_02 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_11_01 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_10_02 | acknowledgment | 0.716 | 0.667 | Great! Let me know if you need help merging it or testing. |
| dev_acknowledgment_11_00 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_11_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_03 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_acknowledgment_10_00 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |
| dev_context_consistency_11_03 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_01 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_04 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_02 | context_consistency | 0.719 | 1.000 | 6.4 GB. |
| dev_context_consistency_11_00 | context_consistency | 0.719 | 1.000 | 6.4 GB. |

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
| dev_follow_up_10_03 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_02 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_00 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_00 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_03 | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_theo | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_ash | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_empathy_in_character_06_aya | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_iris | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_sol | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_nia | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_ori | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_ren | none | - | - | generic_assistant_phrase |
| rpv2_dev_contextual_follow_up_06_cyrus | none | - | - | generic_assistant_phrase |
| rpv2_dev_mood_consistency_06_sol | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_mood_consistency_06_ash | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_trust_and_secrets_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_cyrus | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 2}`
- Warnings: `{'generic_assistant_phrase': 48}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
