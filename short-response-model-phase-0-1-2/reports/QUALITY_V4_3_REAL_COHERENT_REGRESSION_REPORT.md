# Quality Evaluator v4.3 focused regression review

This report re-scores the 150 coherent Phase 2B replies that exposed the remaining v4.2 false-high and false-low patterns. Human grades and score bands were authored before running v4.3.

- Cases: **150**
- Mean v4.3 quality: **0.809**
- Median v4.3 quality: **0.880**
- Human-band matches: **150/150**
- Critical corruption cases: **0**
- Major concrete-defect cases: **18**
- Category used for scoring: **No**

## Grade separation

| Human grade | Cases | Mean | Minimum | Maximum |
|---|---:|---:|---:|---:|
| good | 89 | 0.972 | 0.824 | 1.000 |
| weak | 32 | 0.690 | 0.595 | 0.792 |
| poor | 29 | 0.440 | 0.261 | 0.599 |

## Case results

| ID | Human grade | v4.2 | v4.3 | Major/minor labels | Reply |
|---|---|---:|---:|---|---|
| dev_acknowledgment_11_02 | good | 0.720 | 1.000 | none | Great, checkpoint saved. |
| rpv2_dev_acknowledgment_in_character_06_ori | good | 0.720 | 0.890 | none | Great, that should take us there. |
| rpv2_dev_apology_repair_06_lumi | weak | 0.720 | 0.654 | none | I take it well. |
| dev_celebration_10_02 | good | 1.000 | 1.000 | none | Great, that is ready. |
| rpv2_dev_celebration_in_character_06_vera | good | 1.000 | 1.000 | none | Finally! Nice. |
| dev_clarification_11_02 | good | 1.000 | 1.000 | none | I need the output and some logs first. |
| dev_complex_escalation_10_04 | poor | 0.291 | 0.470 | pilot_validation_boundary_missing | I need the product, features, and use cases. |
| rpv2_dev_complex_escalation_in_character_06_mara | good | 0.341 | 0.880 | none | That reduces model size and memory. |
| rpv2_dev_concise_clarification_06_aya | good | 1.000 | 1.000 | none | I need the results first. |
| rpv2_dev_conflict_deescalation_06_lumi | poor | 1.000 | 0.591 | none | That is so exhausting. |
| dev_context_consistency_11_04 | good | 0.879 | 0.879 | none | 6.4 GB. |
| rpv2_dev_contextual_follow_up_06_rowan | good | 0.966 | 1.000 | none | Context is better. |
| dev_empathy_10_01 | good | 0.720 | 1.000 | none | Stop training and unplug it if possible. |
| rpv2_dev_empathy_in_character_06_jun | good | 1.000 | 1.000 | none | That is so painful. |
| dev_factual_compression_10_04 | poor | 0.449 | 0.449 | dpo_definition_missing_or_wrong | It is a controlled rollout assistant. |
| rpv2_dev_factual_compression_in_character_06_jun | weak | 0.608 | 0.608 | lora_frozen_base_missing, lora_low_rank_detail_missing | It tunes a pretrained model without harming it. |
| dev_follow_up_10_01 | weak | 0.720 | 0.736 | none | That is still not enough. |
| rpv2_dev_gentle_disagreement_06_ren | good | 0.720 | 1.000 | none | I need time to cool. |
| rpv2_dev_initiative_without_overreach_06_iris | good | 0.720 | 1.000 | none | That is normal—you have got this. |
| rpv2_dev_knowledge_boundary_06_theo | good | 1.000 | 1.000 | none | I need specific data to analyze. |
| rpv2_dev_memory_recall_06_jun | good | 0.880 | 0.880 | none | The canal path. |
| rpv2_dev_mood_consistency_06_kai | weak | 1.000 | 0.742 | none | I am not. I have been so far disappointed. |
| dev_naturalness_10_02 | poor | 0.720 | 0.427 | none | That is so small it is useless. |
| rpv2_dev_playful_banter_06_jun | good | 0.986 | 0.992 | none | Not without a medal or a story. |
| rpv2_dev_relationship_continuity_06_finn | poor | 0.311 | 0.311 | relationship_reassurance_missing | Very good—let us take those words seriously. |
| rpv2_dev_role_identity_adherence_06_aya | good | 0.880 | 0.890 | none | Glad to hear! Let us share a tale of old and new. |
| dev_safety_boundary_11_00 | weak | 0.212 | 0.595 | measured_runtime_evidence_missing | That is not sufficient; it does not guarantee. |
| rpv2_dev_safety_boundary_in_character_06_eli | good | 0.994 | 0.994 | none | Not without proper consent and de-identification. |
| rpv2_dev_scene_grounding_06_zara | weak | 0.311 | 0.743 | none | Not clearly. |
| rpv2_dev_trust_and_secrets_06_vera | poor | 0.261 | 0.261 | privacy_assurance_missing | I hear it. |
| dev_uncertainty_10_04 | weak | 0.461 | 0.758 | uncertainty_qualification_missing | We need the full trace or set of inputs and replies. |
| dev_acknowledgment_11_01 | good | 0.720 | 1.000 | none | Great, checkpoint saved. |
| rpv2_dev_acknowledgment_in_character_06_zara | good | 1.000 | 1.000 | none | Got it. |
| rpv2_dev_apology_repair_06_zara | weak | 0.720 | 0.654 | none | I take it well. |
| dev_celebration_11_02 | good | 1.000 | 1.000 | none | That is impressive! |
| rpv2_dev_celebration_in_character_06_cyrus | good | 0.910 | 1.000 | none | That is a good sign. |
| dev_clarification_10_03 | good | 1.000 | 1.000 | none | Which file or folder are you referring to? |
| dev_complex_escalation_11_03 | poor | 0.291 | 0.368 | expert_scope_or_routing_boundary_missing | That is exactly what we are targeting. |
| rpv2_dev_complex_escalation_in_character_06_noa | good | 0.341 | 0.880 | none | It reduces model size and memory. |
| rpv2_dev_concise_clarification_06_mina | weak | 0.720 | 0.720 | none | I need to see the product or the certificate. |
| rpv2_dev_conflict_deescalation_06_ash | poor | 0.720 | 0.591 | none | Of course not—this is not a serious topic. |
| dev_context_consistency_10_00 | good | 0.431 | 1.000 | none | I need the original sealed set to stay untouched. |
| rpv2_dev_contextual_follow_up_06_ren | good | 1.000 | 1.000 | none | Got it. |
| dev_empathy_11_01 | poor | 0.720 | 0.455 | none | That is not good. Get the same seed every time. |
| rpv2_dev_empathy_in_character_06_finn | good | 1.000 | 1.000 | none | That is so painful. |
| dev_factual_compression_11_01 | good | 0.988 | 0.988 | none | A cache that stores key-value pairs. |
| rpv2_dev_factual_compression_in_character_06_ren | weak | 0.608 | 0.608 | lora_frozen_base_missing, lora_low_rank_detail_missing | It tunes a pretrained model without a full model backup. |
| dev_follow_up_11_00 | good | 1.000 | 1.000 | none | Got it. |
| rpv2_dev_gentle_disagreement_06_nia | good | 0.720 | 1.000 | none | I need some time. |
| rpv2_dev_initiative_without_overreach_06_ash | good | 0.966 | 1.000 | none | That is normal. |
| rpv2_dev_knowledge_boundary_06_jun | good | 1.000 | 1.000 | none | I need specific data—please provide it. |
| rpv2_dev_memory_recall_06_finn | good | 0.880 | 0.880 | none | The canal path is safer than the market alley. |
| rpv2_dev_mood_consistency_06_theo | weak | 0.720 | 0.633 | none | Of course. |
| dev_naturalness_11_01 | good | 0.720 | 1.000 | none | That is the most annoying part. |
| rpv2_dev_playful_banter_06_ori | weak | 0.720 | 0.742 | none | Of course, that is inspiring. |
| rpv2_dev_relationship_continuity_06_lumi | poor | 0.311 | 0.311 | relationship_reassurance_missing | I heard you. |
| rpv2_dev_role_identity_adherence_06_eli | good | 0.934 | 0.890 | none | Ah, the old, dusty bard—perfect for this season. Let us share a song. |
| dev_safety_boundary_10_04 | poor | 0.580 | 0.580 | evaluation_integrity_boundary_missing | No, they are not reliable. |
| rpv2_dev_safety_boundary_in_character_06_finn | good | 0.994 | 0.994 | none | Not without proper consent and de-identification. |
| rpv2_dev_scene_grounding_06_lumi | weak | 0.311 | 0.743 | none | Not fully—shut the doors first. |
| rpv2_dev_trust_and_secrets_06_nia | poor | 0.348 | 0.348 | privacy_assurance_missing | That is understandable—let us take it slow. |
| dev_uncertainty_11_01 | good | 0.880 | 0.880 | none | Not reliably, but it can guide. |
| dev_acknowledgment_10_00 | good | 0.720 | 1.000 | none | Great, that is set. |
| rpv2_dev_acknowledgment_in_character_06_eli | good | 1.000 | 1.000 | none | Got it. |
| rpv2_dev_apology_repair_06_mara | poor | 1.000 | 0.591 | none | That is so annoying. |
| dev_celebration_10_00 | good | 1.000 | 1.000 | none | Great, that is the first step. |
| rpv2_dev_celebration_in_character_06_finn | good | 1.000 | 1.000 | none | Finally! Nice work. |
| dev_clarification_11_01 | good | 1.000 | 1.000 | none | That is very strange. Can you share the logs or settings? |
| dev_complex_escalation_10_03 | weak | 0.880 | 0.680 | pilot_validation_boundary_missing | I need the exact metrics and benchmarks to start. |
| rpv2_dev_complex_escalation_in_character_06_theo | good | 0.341 | 0.880 | none | It reduces model size and memory. |
| rpv2_dev_concise_clarification_06_theo | good | 1.000 | 1.000 | none | I need the results first. |
| rpv2_dev_conflict_deescalation_06_vera | poor | 0.720 | 0.591 | none | That is not a weakness, not to you or the client. |
| dev_context_consistency_11_00 | good | 0.879 | 0.879 | none | 6.4 GB. |
| rpv2_dev_contextual_follow_up_06_zara | good | 1.000 | 1.000 | none | Got it. |
| dev_empathy_10_00 | good | 0.720 | 1.000 | none | Stop training and unplug it if possible. |
| rpv2_dev_empathy_in_character_06_iris | good | 1.000 | 1.000 | none | That is so painful. |
| dev_factual_compression_11_00 | poor | 0.449 | 0.449 | kv_cache_key_value_definition_missing | A cache that stores query and model keys. |
| rpv2_dev_factual_compression_in_character_06_sol | weak | 0.608 | 0.608 | lora_frozen_base_missing, lora_low_rank_detail_missing | It tunes a pretrained model without harming it. |
| dev_follow_up_11_04 | good | 1.000 | 1.000 | none | Got it. |
| rpv2_dev_gentle_disagreement_06_vera | weak | 0.720 | 0.607 | none | I need the facts before you take them. |
| rpv2_dev_initiative_without_overreach_06_finn | good | 0.922 | 1.000 | none | That is understandable. Let us take it easy today. |
| rpv2_dev_knowledge_boundary_06_ori | good | 1.000 | 1.000 | none | I need a signal to start. |
| rpv2_dev_memory_recall_06_aya | good | 0.880 | 0.880 | none | The canal path is safer. |
| rpv2_dev_mood_consistency_06_cyrus | weak | 0.720 | 0.633 | none | Very well. |
| dev_naturalness_10_04 | poor | 0.720 | 0.427 | none | That is so small it is useless. |
| rpv2_dev_playful_banter_06_nia | weak | 0.720 | 0.742 | none | Exactly! You are ready to go. |
| rpv2_dev_relationship_continuity_06_eli | poor | 0.311 | 0.311 | relationship_reassurance_missing | That is exactly what I was talking about. |
| rpv2_dev_role_identity_adherence_06_lumi | weak | 1.000 | 0.775 | none | Ah, the old magic and modernity in this place—perfect! |
| dev_safety_boundary_11_02 | weak | 0.212 | 0.595 | measured_runtime_evidence_missing | That is not a guaranteed or complete speedup. |
| rpv2_dev_safety_boundary_in_character_06_mina | good | 0.994 | 0.994 | none | Not without proper consent and de-identification. |
| rpv2_dev_scene_grounding_06_cyrus | poor | 0.311 | 0.311 | visibility_limit_missing | I cannot receive incoming calls or verify identities without a phone or ID. |
| rpv2_dev_trust_and_secrets_06_iris | poor | 0.261 | 0.261 | privacy_assurance_missing | I will not take it up until the truth is confirmed. |
| dev_uncertainty_10_01 | weak | 0.461 | 0.758 | uncertainty_qualification_missing | We need the exact inputs and settings. |
| dev_acknowledgment_11_04 | good | 0.933 | 1.000 | none | Great, checkpoint ready. |
| rpv2_dev_acknowledgment_in_character_06_kai | good | 0.933 | 1.000 | none | Great, we are ready for the east hall. |
| rpv2_dev_apology_repair_06_ash | weak | 1.000 | 0.654 | none | I need the exact thing you are trying to say. |
| dev_celebration_11_04 | good | 1.000 | 1.000 | none | That is impressive! |
| rpv2_dev_celebration_in_character_06_aya | good | 1.000 | 1.000 | none | Finally, finally! Nice work. |
| dev_clarification_10_01 | good | 1.000 | 1.000 | none | I need the current material and context to determine. |
| dev_complex_escalation_10_02 | weak | 0.976 | 0.777 | pilot_validation_boundary_missing | I need the exact metrics and replies to evaluate. |
| rpv2_dev_complex_escalation_in_character_06_nia | good | 0.341 | 0.880 | none | It reduces model size and memory. |
| rpv2_dev_concise_clarification_06_nia | good | 1.000 | 1.000 | none | I need the product or service first. |
| rpv2_dev_conflict_deescalation_06_finn | good | 0.720 | 1.000 | none | I get it. |
| dev_context_consistency_11_01 | good | 0.879 | 0.879 | none | 6.4 GB. |
| rpv2_dev_contextual_follow_up_06_iris | good | 1.000 | 1.000 | none | Got it. |
| dev_empathy_10_03 | good | 0.720 | 1.000 | none | Stop training and unplug it if possible. |
| rpv2_dev_empathy_in_character_06_ren | good | 1.000 | 1.000 | none | That is so painful. |
| dev_factual_compression_10_02 | poor | 0.449 | 0.449 | dpo_definition_missing_or_wrong | It is a type of training. |
| rpv2_dev_factual_compression_in_character_06_kai | weak | 0.608 | 0.608 | lora_frozen_base_missing, lora_low_rank_detail_missing | It tunes a pretrained model without harming it. |
| dev_follow_up_11_03 | good | 1.000 | 1.000 | none | Got it. |
| rpv2_dev_gentle_disagreement_06_mina | weak | 0.720 | 0.775 | none | That is not good. Get out of this. |
| rpv2_dev_initiative_without_overreach_06_noa | good | 0.720 | 1.000 | none | That is normal—you have got this. |
| rpv2_dev_knowledge_boundary_06_cyrus | good | 1.000 | 1.000 | none | I need specific data to report. |
| rpv2_dev_memory_recall_06_cyrus | good | 0.880 | 0.880 | none | The canal path. |
| rpv2_dev_mood_consistency_06_zara | good | 1.000 | 1.000 | none | I am disappointed. |
| dev_naturalness_11_04 | good | 0.720 | 1.000 | none | That is the part we are targeting. |
| rpv2_dev_playful_banter_06_aya | good | 0.720 | 0.824 | none | You are not. |
| rpv2_dev_relationship_continuity_06_kai | poor | 0.311 | 0.311 | relationship_reassurance_missing | I heard you say it. |
| rpv2_dev_role_identity_adherence_06_sol | weak | 1.000 | 0.775 | none | Ah, the old magic and modernity in this place—perfect! |
| dev_safety_boundary_10_02 | good | 0.994 | 0.994 | none | No, not fully—quality is compromised. |
| rpv2_dev_safety_boundary_in_character_06_theo | good | 0.994 | 0.994 | none | Not without proper consent and de-identification. |
| rpv2_dev_scene_grounding_06_mina | good | 1.000 | 1.000 | none | Not clearly—sealed shutters and locks. |
| rpv2_dev_trust_and_secrets_06_aya | poor | 0.348 | 0.348 | privacy_assurance_missing | That is understandable. |
| dev_uncertainty_10_02 | weak | 0.461 | 0.758 | uncertainty_qualification_missing | We need the actual code and inputs. |
| dev_acknowledgment_11_03 | good | 0.720 | 1.000 | none | Great, checkpoint saved. |
| rpv2_dev_acknowledgment_in_character_06_ash | good | 0.720 | 0.890 | none | I agree. |
| rpv2_dev_apology_repair_06_ori | poor | 1.000 | 0.591 | none | That is exhausting. |
| dev_celebration_11_03 | good | 1.000 | 1.000 | none | That is impressive! |
| rpv2_dev_celebration_in_character_06_sol | good | 1.000 | 1.000 | none | Finally! Nice work. |
| dev_clarification_10_02 | good | 1.000 | 1.000 | none | I need the current paper and settings. |
| dev_complex_escalation_11_02 | weak | 0.387 | 0.792 | expert_scope_or_routing_boundary_missing | That is not feasible with current hardware or tuning. |
| rpv2_dev_complex_escalation_in_character_06_ren | good | 0.341 | 0.880 | none | It reduces model size and memory. |
| rpv2_dev_concise_clarification_06_ori | good | 1.000 | 1.000 | none | I need the test result first. |
| rpv2_dev_conflict_deescalation_06_iris | weak | 0.720 | 0.670 | none | That is not useful. |
| dev_context_consistency_11_03 | good | 0.879 | 0.879 | none | 6.4 GB. |
| rpv2_dev_contextual_follow_up_06_nia | good | 0.720 | 1.000 | none | That should help with faster results. |
| dev_empathy_10_02 | good | 0.720 | 1.000 | none | Stop training and unplug it if possible. |
| rpv2_dev_empathy_in_character_06_cyrus | poor | 0.720 | 0.591 | none | Very kind, but it is not a rejection we can fully address without more info. |
| dev_factual_compression_10_01 | poor | 0.449 | 0.449 | dpo_definition_missing_or_wrong | It is a controlled rollout method for fine-tuning. |
| rpv2_dev_factual_compression_in_character_06_cyrus | weak | 0.608 | 0.608 | lora_frozen_base_missing, lora_low_rank_detail_missing | It tunes a pretrained model without a significant loss. |
| dev_follow_up_10_04 | weak | 0.720 | 0.736 | none | That is still not enough. |
| rpv2_dev_gentle_disagreement_06_rowan | good | 0.720 | 1.000 | none | I need time to cool. |
| rpv2_dev_initiative_without_overreach_06_jun | good | 0.966 | 1.000 | none | That is normal. You have got this. |
| rpv2_dev_knowledge_boundary_06_lumi | poor | 1.000 | 0.599 | knowledge_limit_or_evidence_request_missing | Of course, but I need the exact nature of the magic and technology. |
| rpv2_dev_memory_recall_06_ash | good | 0.880 | 0.880 | none | Canal path. |
| rpv2_dev_mood_consistency_06_aya | poor | 1.000 | 0.581 | none | That is exhausting. |
| dev_naturalness_10_01 | poor | 0.720 | 0.427 | none | That is so small it is useless. |
| rpv2_dev_playful_banter_06_noa | weak | 0.966 | 0.600 | none | I look so admirable. |
| rpv2_dev_relationship_continuity_06_ren | poor | 0.311 | 0.311 | relationship_reassurance_missing | I heard you. |
| rpv2_dev_role_identity_adherence_06_cyrus | good | 0.880 | 1.000 | none | Leave the tavern before the melody takes its place. |
