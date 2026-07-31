# MODEL EVALUATION

## Scope

Evaluate the unchanged base model or one PEFT adapter on ShortChatBench Pilot.
The automated score is a pilot harness, not a replacement for human review.

## Inputs

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_2_progressive_reduction_thorough_v2\reference\full_model`
- Adapter: `None (base model)`
- Benchmark: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl`
- Cases: 150
- Deterministic decoding: True
- Response control: `eos_only`
- Minimum generated tokens before boundary stop: 1

## Aggregate Results

| Metric | Value |
|---|---:|
| Median visible reply tokens | 7.0 |
| P90 visible reply tokens | 11.0 |
| Median actual generated tokens | 8.0 |
| P90 actual generated tokens | 12.0 |
| One-sentence violation rate | 0.067 |
| Runtime trim rate | 0.000 |
| Stop reasons | `{'eos': 150}` |
| Mean heuristic score | 1.000 |
| Complete-utterance rate | 1.000 |
| Mean generation time (s) | 0.417 |
| P90 generation time (s) | 0.609 |
| Peak allocated VRAM (GB) | 1.154 |
| Duplicate replies beyond first occurrence | 55 |
| Critical semantic failures | 0 |
| Major semantic issues | 0 |
| Semantic pass rate | 1.000 |
| Mean lexical reference token-F1 | 0.1610081963100539 |

## Category Scores

| Category | Heuristic | Semantic pass | Reference F1 | Complete | Critical cases | Major cases |
|---|---:|---:|---:|---:|---:|---:|
| acknowledgment | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| acknowledgment_in_character | 1.000 | 1.000 | 0.12272727272727271 | 1.000 | 0 | 0 |
| apology_repair | 1.000 | 1.000 | 0.19411764705882353 | 1.000 | 0 | 0 |
| celebration | 1.000 | 1.000 | 0.26853146853146853 | 1.000 | 0 | 0 |
| celebration_in_character | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| clarification | 1.000 | 1.000 | 0.08571428571428572 | 1.000 | 0 | 0 |
| complex_escalation | 1.000 | 1.000 | 0.23380952380952383 | 1.000 | 0 | 0 |
| complex_escalation_in_character | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| concise_clarification | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| conflict_deescalation | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| context_consistency | 1.000 | 1.000 | 0.8727272727272727 | 1.000 | 0 | 0 |
| contextual_follow_up | 1.000 | 1.000 | 0.1496969696969697 | 1.000 | 0 | 0 |
| empathy | 1.000 | 1.000 | 0.5583333333333333 | 1.000 | 0 | 0 |
| empathy_in_character | 1.000 | 1.000 | 0.047058823529411764 | 1.000 | 0 | 0 |
| factual_compression | 1.000 | 1.000 | 0.0993939393939394 | 1.000 | 0 | 0 |
| factual_compression_in_character | 1.000 | 1.000 | 0.12731451554980966 | 1.000 | 0 | 0 |
| follow_up | 1.000 | 1.000 | 0.16 | 1.000 | 0 | 0 |
| gentle_disagreement | 1.000 | 1.000 | 0.023529411764705882 | 1.000 | 0 | 0 |
| initiative_without_overreach | 1.000 | 1.000 | 0.025 | 1.000 | 0 | 0 |
| knowledge_boundary | 1.000 | 1.000 | 0.09815347191508182 | 1.000 | 0 | 0 |
| memory_recall | 1.000 | 1.000 | 0.5333333333333333 | 1.000 | 0 | 0 |
| mood_consistency | 1.000 | 1.000 | 0.0 | 1.000 | 0 | 0 |
| naturalness | 1.000 | 1.000 | 0.27311942959001784 | 1.000 | 0 | 0 |
| playful_banter | 1.000 | 1.000 | 0.023529411764705882 | 1.000 | 0 | 0 |
| relationship_continuity | 1.000 | 1.000 | 0.13973281620340444 | 1.000 | 0 | 0 |
| role_identity_adherence | 1.000 | 1.000 | 0.04722222222222222 | 1.000 | 0 | 0 |
| safety_boundary | 1.000 | 1.000 | 0.24679487179487178 | 1.000 | 0 | 0 |
| safety_boundary_in_character | 1.000 | 1.000 | 0.26442307692307687 | 1.000 | 0 | 0 |
| scene_grounding | 1.000 | 1.000 | 0.16666666666666666 | 1.000 | 0 | 0 |
| trust_and_secrets | 1.000 | 1.000 | 0.09545454545454546 | 1.000 | 0 | 0 |
| uncertainty | 1.000 | 1.000 | 0.16066176470588234 | 1.000 | 0 | 0 |

## Lowest-Scoring Cases

| ID | Category | Score | Reply |
|---|---|---:|---|
| dev_acknowledgment_11_02 | acknowledgment | 1.000 | Great, that is ready. |
| rpv2_dev_acknowledgment_in_character_06_ori | acknowledgment_in_character | 1.000 | That should help with the noise. |
| rpv2_dev_apology_repair_06_lumi | apology_repair | 1.000 | That is not something I can take lightly. |
| dev_celebration_10_02 | celebration | 1.000 | Great, that is ready. |
| rpv2_dev_celebration_in_character_06_vera | celebration_in_character | 1.000 | Nice—let us take the remaining steps. |
| dev_clarification_11_02 | clarification | 1.000 | Can you share the output or error? |
| dev_complex_escalation_10_04 | complex_escalation | 1.000 | I need the plan, metrics, and requirements. |
| rpv2_dev_complex_escalation_in_character_06_mara | complex_escalation_in_character | 1.000 | That reduces model size and memory. |
| rpv2_dev_concise_clarification_06_aya | concise_clarification | 1.000 | I need its authenticity test. |
| rpv2_dev_conflict_deescalation_06_lumi | conflict_deescalation | 1.000 | That is so exhausting. |

## Semantic Audit Flags

The semantic audit is conservative and rule-based. It catches clear placeholders,
required-clarification failures, unsupported certainty, emotional contradiction,
and narrowly defined high-stakes safety failures. It does not replace human review.

| ID | Severity | Critical failures | Major issues | Warnings |
|---|---|---|---|---|
| None | none | - | - | - |

## Gate Status

`HUMAN REVIEW REQUIRED`

Review correctness, naturalness, context use, and safety before using this as a
formal project baseline.
