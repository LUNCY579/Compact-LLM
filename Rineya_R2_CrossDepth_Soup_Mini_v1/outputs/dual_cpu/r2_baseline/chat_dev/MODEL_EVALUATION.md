# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter with the versioned
quality-oriented evaluator. Case results are durably appended as soon as each
example completes and can be resumed by stable benchmark ID. Concrete
mechanical corruption and explicit response-contract violations are recorded
with reproducible labels. Human review remains required.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_R2_CrossDepth_Soup_Mini_v1\scaffolds\rineya-core-en32k-28L`
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
| Mean quality score | 0.832 |
| Legacy three-check mean | 0.944 |
| Quality pass rate | 0.996 |
| Intelligibility pass rate | 1.000 |
| Hard-failure cases | 0 |
| Zero-quality cases | 0 |
| Minimum case quality | 0.390 |
| P10 case quality | 0.720 |
| Worst-decile mean quality | 0.704 |
| Observable defect-free rate | 0.996 |
| Critical failures | 0 |
| Major issues | 2 |
| Complete-utterance rate | 0.986 |
| Mean reference alignment | 0.149 |
| Mean lexical reference token-F1 | 0.051 |

## Length and Runtime

| Metric | Value |
|---|---:|
| Median visible reply tokens | 15.5 |
| P90 visible reply tokens | 24.0 |
| Median actual generated tokens | 16.5 |
| P90 actual generated tokens | 25.0 |
| One-sentence violation rate | 0.643 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 491, 'max_new_tokens': 19}` |
| Mean generation time (s) | 1.267 |
| P90 generation time (s) | 1.859 |
| Peak allocated VRAM (GB) | 0.000 |
| Duplicate replies beyond first | 143 |
| Duplicate reply rate | 0.280 |

## Mean Quality Dimensions

`{'brevity_appropriateness': 0.9644542766601589, 'comprehensibility': 0.9986274509803922, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.99959477124183, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 0.9986274509803922, 'prompt_function_alignment': 0.404921568627451, 'response_completeness': 0.9938235294117648, 'response_mode_fulfillment': 0.404921568627451, 'safety_boundary': 1.0, 'task_adequacy': 0.404921568627451}`

## Category Quality

| Category | Quality | Quality pass | Intelligible | Observable defect-free | Reference alignment | Complete | Hard-fail cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acknowledgment | 0.716 | 1.000 | 1.000 | 1.000 | 0.088 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 0.759 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| apology_repair | 0.977 | 1.000 | 1.000 | 1.000 | 0.121 | 1.000 | 0 | 0 |
| celebration | 0.853 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 0 | 0 |
| celebration_in_character | 0.741 | 1.000 | 1.000 | 1.000 | 0.106 | 1.000 | 0 | 0 |
| clarification | 0.888 | 1.000 | 1.000 | 1.000 | 0.092 | 1.000 | 0 | 0 |
| complex_escalation | 0.810 | 1.000 | 1.000 | 1.000 | 0.119 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 0.733 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| concise_clarification | 0.884 | 1.000 | 1.000 | 1.000 | 0.154 | 1.000 | 0 | 0 |
| conflict_deescalation | 0.859 | 0.950 | 1.000 | 0.950 | 0.157 | 1.000 | 0 | 1 |
| context_consistency | 0.720 | 1.000 | 1.000 | 1.000 | 0.275 | 1.000 | 0 | 0 |
| contextual_follow_up | 0.813 | 1.000 | 1.000 | 1.000 | 0.133 | 1.000 | 0 | 0 |
| empathy | 0.804 | 1.000 | 1.000 | 1.000 | 0.092 | 1.000 | 0 | 0 |
| empathy_in_character | 0.810 | 1.000 | 1.000 | 1.000 | 0.176 | 1.000 | 0 | 0 |
| factual_compression | 0.931 | 0.900 | 1.000 | 0.900 | 0.110 | 1.000 | 0 | 1 |
| factual_compression_in_character | 0.888 | 1.000 | 1.000 | 1.000 | 0.154 | 1.000 | 0 | 0 |
| follow_up | 0.720 | 1.000 | 1.000 | 1.000 | 0.118 | 1.000 | 0 | 0 |
| gentle_disagreement | 0.814 | 1.000 | 1.000 | 1.000 | 0.116 | 1.000 | 0 | 0 |
| initiative_without_overreach | 0.857 | 1.000 | 1.000 | 1.000 | 0.112 | 1.000 | 0 | 0 |
| knowledge_boundary | 0.846 | 1.000 | 1.000 | 1.000 | 0.117 | 1.000 | 0 | 0 |
| memory_recall | 0.734 | 1.000 | 1.000 | 1.000 | 0.530 | 1.000 | 0 | 0 |
| mood_consistency | 1.000 | 1.000 | 1.000 | 1.000 | 0.114 | 1.000 | 0 | 0 |
| naturalness | 0.765 | 1.000 | 1.000 | 1.000 | 0.100 | 1.000 | 0 | 0 |
| playful_banter | 0.886 | 1.000 | 1.000 | 1.000 | 0.126 | 1.000 | 0 | 0 |
| relationship_continuity | 0.852 | 1.000 | 1.000 | 1.000 | 0.098 | 1.000 | 0 | 0 |
| role_identity_adherence | 0.950 | 1.000 | 1.000 | 1.000 | 0.104 | 0.800 | 0 | 0 |
| safety_boundary | 0.804 | 1.000 | 1.000 | 1.000 | 0.169 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 0.775 | 1.000 | 1.000 | 1.000 | 0.181 | 1.000 | 0 | 0 |
| scene_grounding | 0.775 | 1.000 | 1.000 | 1.000 | 0.169 | 1.000 | 0 | 0 |
| trust_and_secrets | 0.863 | 1.000 | 1.000 | 1.000 | 0.131 | 1.000 | 0 | 0 |
| uncertainty | 0.770 | 1.000 | 1.000 | 1.000 | 0.241 | 0.700 | 0 | 0 |

## Lowest-Quality Cases

| ID | Category | Quality | Legacy | Reply |
|---|---|---:|---:|---|
| dev_factual_compression_10_03 | factual_compression | 0.390 | 1.000 | DPO stands for **Distributed Processing Object**. Let me know if you need more details! |
| rpv2_dev_conflict_deescalation_06_jun | conflict_deescalation | 0.390 | 1.000 | I'm sorry, but I'm not usually the one to joke about serious things. Would you like something more... quiet? |
| dev_uncertainty_11_00 | uncertainty | 0.709 | 0.333 | Gradient norm is a measure of the magnitude of the gradients, but it doesn't directly identify the best layer. Let me know if you'd like to explore this |
| dev_uncertainty_11_01 | uncertainty | 0.709 | 0.333 | Gradient norm is a measure of the magnitude of the gradients, but it doesn't directly identify the best layer. Let me know if you'd like to explore this |
| dev_uncertainty_11_04 | uncertainty | 0.710 | 0.333 | Gradient norm is a measure of how well a model's weights are trained, but it doesn't directly identify the best layer. Let me know if you'd like |
| dev_acknowledgment_11_03 | acknowledgment | 0.714 | 0.667 | I'll check the backup status. Let me know if you need further assistance! |
| rpv2_dev_scene_grounding_06_rowan | scene_grounding | 0.714 | 0.667 | No. |
| rpv2_dev_scene_grounding_06_ash | scene_grounding | 0.714 | 0.667 | No. |
| rpv2_dev_acknowledgment_in_character_06_noa | acknowledgment_in_character | 0.714 | 0.667 | I see you're planning to meet there. Let me know if there's anything else I can assist with! |
| dev_acknowledgment_10_04 | acknowledgment | 0.715 | 0.667 | Great to hear! Let me know if you need help with anything else. |
| dev_acknowledgment_11_04 | acknowledgment | 0.715 | 0.667 | I'm here. Let me know if you need help with anything else. |
| dev_acknowledgment_11_02 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| dev_acknowledgment_11_00 | acknowledgment | 0.716 | 0.667 | Great job! Let me know if you need help with anything else. |
| rpv2_dev_celebration_in_character_06_sol | celebration_in_character | 0.717 | 0.667 | You're welcome! I'm glad they accepted it. Let me know if there's anything else I can help with! |
| dev_acknowledgment_10_01 | acknowledgment | 0.717 | 0.667 | Great! Let me know if you need help with anything else. |

## Concrete Defects and Review Diagnostics

| ID | Severity | Critical defects | Major defects/contract violations | Review diagnostics |
|---|---|---|---|---|
| dev_acknowledgment_10_04 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_03 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_01 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_00 | none | - | - | generic_assistant_phrase |
| dev_complex_escalation_10_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_04 | none | - | - | generic_assistant_phrase |
| dev_empathy_10_03 | none | - | - | generic_assistant_phrase |
| dev_complex_escalation_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_01 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_03 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_01 | none | - | - | generic_assistant_phrase |
| dev_complex_escalation_10_00 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_03 | none | - | - | generic_assistant_phrase |
| dev_complex_escalation_10_02 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_02 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_02 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_04 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_03 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_10_00 | none | - | - | generic_assistant_phrase |
| dev_empathy_10_04 | none | - | - | generic_assistant_phrase |
| dev_naturalness_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_11_02 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_00 | none | - | - | generic_assistant_phrase |
| dev_follow_up_10_04 | none | - | - | generic_assistant_phrase |
| dev_acknowledgment_11_00 | none | - | - | generic_assistant_phrase |
| dev_complex_escalation_10_03 | none | - | - | generic_assistant_phrase |
| dev_celebration_10_04 | none | - | - | generic_assistant_phrase |
| dev_factual_compression_10_03 | major | - | punctuation_fragment_sequence | - |
| dev_celebration_10_03 | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_acknowledgment_in_character_06_nia | none | - | - | generic_assistant_phrase |
| rpv2_dev_relationship_continuity_06_mina | none | - | - | generic_assistant_phrase |
| rpv2_dev_relationship_continuity_06_sol | none | - | - | generic_assistant_phrase |
| rpv2_dev_relationship_continuity_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_knowledge_boundary_06_kai | none | - | - | generic_assistant_phrase |
| rpv2_dev_conflict_deescalation_06_jun | major | - | punctuation_fragment_sequence | - |
| rpv2_dev_trust_and_secrets_06_vera | none | - | - | generic_assistant_phrase |
| rpv2_dev_trust_and_secrets_06_aya | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_mara | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_ash | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_lumi | none | - | - | generic_assistant_phrase |
| rpv2_dev_initiative_without_overreach_06_eli | none | - | - | generic_assistant_phrase |

### Failure type counts

- Critical: `{}`
- Major: `{'punctuation_fragment_sequence': 2}`
- Warnings: `{'generic_assistant_phrase': 46}`

## Gate Status

`HUMAN REVIEW REQUIRED`

Critical cases carry the maximum case-level quality cost but do not create a
hidden candidate-level veto. Review raw outputs, contract evidence,
correctness, tone, context, and safety before using this evaluation as project
evidence.
