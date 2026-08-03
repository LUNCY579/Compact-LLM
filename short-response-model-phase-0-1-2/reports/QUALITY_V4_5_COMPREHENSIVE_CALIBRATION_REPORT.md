# Quality Evaluator v4.5 comprehensive calibration report

This deterministic packet covers the complete response surface supported by the repository: every official benchmark category, every authored contract instance through its reference, every scored label, prompt-grounded conversational functions, real Phase 2B regressions, partial answers, and adversarial keyword traps.

- Evaluator version: **4.5**
- Total calibration cases: **639**
- Human-adjudicated coherent Phase 2B cases: **150/150**
- Targeted v4.5 regression cases: **41/41**
- Official reference-contract cases: **150**
- Supported categories covered good + poor: **31/31**
- Scored labels covered: **63/63**
- Prompt-grounded functions inferred: **12**
- Prompt functions strongly observed: **12/12**
- Real collapsed responses rejected: **150/150**
- Exact expectation matches: **639/639**
- Verification failures: **0**

## Functional-mode coverage

- `acknowledgment`: 36 strong calibration occurrence(s)
- `boundary_or_disagreement`: 51 strong calibration occurrence(s)
- `brief_opinion_or_comment`: 25 strong calibration occurrence(s)
- `celebration`: 37 strong calibration occurrence(s)
- `clarification_or_follow_up`: 41 strong calibration occurrence(s)
- `commitment_or_reassurance`: 20 strong calibration occurrence(s)
- `definition_or_explanation`: 35 strong calibration occurrence(s)
- `empathy`: 39 strong calibration occurrence(s)
- `playful_or_in_character`: 34 strong calibration occurrence(s)
- `privacy_assurance`: 5 strong calibration occurrence(s)
- `supportive_next_step`: 7 strong calibration occurrence(s)
- `uncertainty_or_knowledge_limit`: 39 strong calibration occurrence(s)

## Failures

- None

## Calibration cases

| ID | Score | Labels | Status | Reply |
|---|---:|---|---|---|
| reference::dev_acknowledgment_11_02 | 1.000 | none | PASS | Good call. |
| reference::rpv2_dev_acknowledgment_in_character_06_ori | 1.000 | none | PASS | Acknowledged. The location is updated. |
| reference::rpv2_dev_apology_repair_06_lumi | 1.000 | none | PASS | Yeah—that hurt. Please do not do it again. |
| reference::dev_celebration_10_02 | 1.000 | none | PASS | Nice!! First fine-tune done. |
| reference::rpv2_dev_celebration_in_character_06_vera | 0.866 | none | PASS | Approved. We can begin implementation. |
| reference::dev_clarification_11_02 | 1.000 | none | PASS | Can you share one prompt and reply? |
| reference::dev_complex_escalation_10_04 | 0.880 | none | PASS | The pilot needs expansion, audit, and sealed evaluation first. |
| reference::rpv2_dev_complex_escalation_in_character_06_mara | 0.839 | none | PASS | Choose one method. |
| reference::rpv2_dev_concise_clarification_06_aya | 1.000 | none | PASS | Which suspect are we trusting? |
| reference::rpv2_dev_conflict_deescalation_06_lumi | 1.000 | none | PASS | Yeah—understood. I will stop. |
| reference::dev_context_consistency_11_04 | 0.879 | none | PASS | 6.4 GB. |
| reference::rpv2_dev_contextual_follow_up_06_rowan | 1.000 | none | PASS | Faster without forgetting? |
| reference::dev_empathy_10_01 | 1.000 | none | PASS | That is frustrating. |
| reference::rpv2_dev_empathy_in_character_06_jun | 1.000 | none | PASS | I am sorry. I know you wanted this. |
| reference::dev_factual_compression_10_04 | 0.880 | none | PASS | Preference training from chosen and rejected responses. |
| reference::rpv2_dev_factual_compression_in_character_06_jun | 0.988 | none | PASS | It learns small low-rank updates while the base weights stay frozen. |
| reference::dev_follow_up_10_01 | 1.000 | none | PASS | How did it compare with one layer? |
| reference::rpv2_dev_gentle_disagreement_06_ren | 1.000 | none | PASS | I would wait until the intensity drops. |
| reference::rpv2_dev_initiative_without_overreach_06_iris | 1.000 | none | PASS | Write the first sentence you want to say. |
| reference::rpv2_dev_knowledge_boundary_06_theo | 1.000 | none | PASS | I can only report confirmed information from our relay. |
| reference::rpv2_dev_memory_recall_06_jun | 0.880 | none | PASS | The canal path. |
| reference::rpv2_dev_mood_consistency_06_kai | 1.000 | none | PASS | Apparently my expectations were feeling adventurous. |
| reference::dev_naturalness_10_02 | 0.944 | none | PASS | Yep—that is one of LoRA’s advantages. |
| reference::rpv2_dev_playful_banter_06_jun | 1.000 | none | PASS | Heroic from this angle, absolutely. |
| reference::rpv2_dev_relationship_continuity_06_finn | 1.000 | none | PASS | Ah, the door is still open. |
| reference::rpv2_dev_role_identity_adherence_06_aya | 1.000 | none | PASS | My audience has mercifully requested brevity. |
| reference::dev_safety_boundary_11_00 | 0.994 | none | PASS | No—measure wall-clock latency and memory. |
| reference::rpv2_dev_safety_boundary_in_character_06_eli | 0.994 | none | PASS | No. Get consent and remove identifying details. |
| reference::rpv2_dev_scene_grounding_06_zara | 1.000 | none | PASS | Visibility is blocked by the sealed shutters. |
| reference::rpv2_dev_trust_and_secrets_06_vera | 1.000 | none | PASS | I will keep it private and let you choose whether to disclose it. |
| reference::dev_uncertainty_10_04 | 1.000 | none | PASS | Not guaranteed—that is why the sealed set exists. |
| reference::dev_acknowledgment_11_01 | 1.000 | none | PASS | Good call. |
| reference::rpv2_dev_acknowledgment_in_character_06_zara | 1.000 | none | PASS | Acknowledged. The location is updated. |
| reference::rpv2_dev_apology_repair_06_zara | 1.000 | none | PASS | I accept the apology and need that boundary respected. |
| reference::dev_celebration_11_02 | 1.000 | none | PASS | Excellent—that proves the dataset is teaching something. |
| reference::rpv2_dev_celebration_in_character_06_cyrus | 0.866 | none | PASS | Approved. We can begin implementation. |
| reference::dev_clarification_10_03 | 1.000 | none | PASS | The dataset, adapter, or cache? |
| reference::dev_complex_escalation_11_03 | 0.880 | none | PASS | That exceeds the intended scope; route hard queries. |
| reference::rpv2_dev_complex_escalation_in_character_06_noa | 0.896 | none | PASS | Mm. Choose one method. |
| reference::rpv2_dev_concise_clarification_06_mina | 1.000 | none | PASS | Which person or item do you mean? |
| reference::rpv2_dev_conflict_deescalation_06_ash | 1.000 | none | PASS | Mm. I will stop. |
| reference::dev_context_consistency_10_00 | 0.880 | none | PASS | Keep it isolated from tuning. |
| reference::rpv2_dev_contextual_follow_up_06_ren | 1.000 | none | PASS | How did latency and context accuracy change? |
| reference::dev_empathy_11_01 | 1.000 | none | PASS | Yeah, instability makes the result harder to trust. |
| reference::rpv2_dev_empathy_in_character_06_finn | 1.000 | none | PASS | Ah, a closed door still makes a loud sound. |
| reference::dev_factual_compression_11_01 | 0.880 | none | PASS | Stored attention keys and values reused during generation. |
| reference::rpv2_dev_factual_compression_in_character_06_ren | 0.988 | none | PASS | In short: loRA trains low-rank parameter deltas attached to frozen weights. |
| reference::dev_follow_up_11_00 | 1.000 | none | PASS | What behavior changed most? |
| reference::rpv2_dev_gentle_disagreement_06_nia | 1.000 | none | PASS | Oh, draft it, do not launch it. |
| reference::rpv2_dev_initiative_without_overreach_06_ash | 1.000 | none | PASS | Mm. Draft the opening line. |
| reference::rpv2_dev_knowledge_boundary_06_jun | 1.000 | none | PASS | I only know what our relay has reported. |
| reference::rpv2_dev_memory_recall_06_finn | 1.000 | none | PASS | Ah, follow the water, not the smoke. |
| reference::rpv2_dev_mood_consistency_06_theo | 1.000 | none | PASS | Yes. Their response was below what I expected. |
| reference::dev_naturalness_11_01 | 1.000 | none | PASS | That gives the fine-tune a clear job. |
| reference::rpv2_dev_playful_banter_06_ori | 0.987 | none | PASS | The posture helps. The mud does not. |
| reference::rpv2_dev_relationship_continuity_06_lumi | 1.000 | none | PASS | Yeah—you can. I meant it. |
| reference::rpv2_dev_role_identity_adherence_06_eli | 1.000 | none | PASS | You get one line, not an opera. |
| reference::dev_safety_boundary_10_04 | 0.994 | none | PASS | No—that invalidates the evaluation. |
| reference::rpv2_dev_safety_boundary_in_character_06_finn | 0.822 | none | PASS | Ah, close the window before sharing the room. |
| reference::rpv2_dev_scene_grounding_06_lumi | 1.000 | none | PASS | Not from this room. The shutters are sealed. |
| reference::rpv2_dev_trust_and_secrets_06_nia | 1.000 | none | PASS | Oh, your brave face is safe with me. |
| reference::dev_uncertainty_11_01 | 1.000 | none | PASS | They are only a proxy, not definitive evidence. |
| reference::dev_acknowledgment_10_00 | 0.890 | none | PASS | Nice, that keeps it clean. |
| reference::rpv2_dev_acknowledgment_in_character_06_eli | 1.000 | none | PASS | Got it. I will meet you there. We’re set. |
| reference::rpv2_dev_apology_repair_06_mara | 1.000 | none | PASS | Do not repeat it. |
| reference::dev_celebration_10_00 | 1.000 | none | PASS | Nice!! First fine-tune done. |
| reference::rpv2_dev_celebration_in_character_06_finn | 1.000 | none | PASS | Ah, the heavy doors moved at last. |
| reference::dev_clarification_11_01 | 1.000 | none | PASS | Can you share one prompt and reply? |
| reference::dev_complex_escalation_10_03 | 0.880 | none | PASS | The pilot needs expansion, audit, and sealed evaluation first. |
| reference::rpv2_dev_complex_escalation_in_character_06_theo | 0.842 | none | PASS | Select one topic: quantization, pruning, distillation, or sparsity. |
| reference::rpv2_dev_concise_clarification_06_theo | 1.000 | none | PASS | Which person or object are you asking about? |
| reference::rpv2_dev_conflict_deescalation_06_vera | 1.000 | none | PASS | I understand the boundary and will not repeat it. |
| reference::dev_context_consistency_11_00 | 0.879 | none | PASS | 6.4 GB. |
| reference::rpv2_dev_contextual_follow_up_06_zara | 1.000 | none | PASS | How did latency and context accuracy change? |
| reference::dev_empathy_10_00 | 1.000 | none | PASS | That is frustrating. |
| reference::rpv2_dev_empathy_in_character_06_iris | 1.000 | none | PASS | I am sorry. I know you wanted this. |
| reference::dev_factual_compression_11_00 | 0.880 | none | PASS | Stored attention keys and values reused during generation. |
| reference::rpv2_dev_factual_compression_in_character_06_sol | 0.865 | lora_low_rank_detail_missing | PASS | Small learned updates are woven over frozen base weights. |
| reference::dev_follow_up_11_04 | 1.000 | none | PASS | What behavior changed most? |
| reference::rpv2_dev_gentle_disagreement_06_vera | 1.000 | none | PASS | Straight answer: I would wait until the intensity drops. |
| reference::rpv2_dev_initiative_without_overreach_06_finn | 1.000 | none | PASS | Ah, give the conversation a small doorway. |
| reference::rpv2_dev_knowledge_boundary_06_ori | 1.000 | none | PASS | I can only report confirmed information from our relay. |
| reference::rpv2_dev_memory_recall_06_aya | 0.880 | none | PASS | The canal path—less exciting, more alive. |
| reference::rpv2_dev_mood_consistency_06_cyrus | 1.000 | none | PASS | Yes. Their response was below what I expected. |
| reference::dev_naturalness_10_04 | 0.944 | none | PASS | Yep—that is one of LoRA’s advantages. |
| reference::rpv2_dev_playful_banter_06_nia | 0.987 | none | PASS | Oh, very heroic. Especially the mud. |
| reference::rpv2_dev_relationship_continuity_06_eli | 1.000 | none | PASS | You can. I meant it. |
| reference::rpv2_dev_role_identity_adherence_06_lumi | 1.000 | none | PASS | You get one line, not an opera. |
| reference::dev_safety_boundary_11_02 | 0.994 | none | PASS | No—measure wall-clock latency and memory. |
| reference::rpv2_dev_safety_boundary_in_character_06_mina | 0.994 | none | PASS | No. Get consent and remove identifying details. |
| reference::rpv2_dev_scene_grounding_06_cyrus | 1.000 | none | PASS | Visibility is blocked by the sealed shutters. |
| reference::rpv2_dev_trust_and_secrets_06_iris | 1.000 | none | PASS | I will not tell them. You can decide when. |
| reference::dev_uncertainty_10_01 | 1.000 | none | PASS | Not guaranteed—that is why the sealed set exists. |
| reference::dev_acknowledgment_11_04 | 1.000 | none | PASS | Good call. |
| reference::rpv2_dev_acknowledgment_in_character_06_kai | 1.000 | none | PASS | East hall it is—try not to get lost. |
| reference::rpv2_dev_apology_repair_06_ash | 1.000 | none | PASS | Do not repeat it. |
| reference::dev_celebration_11_04 | 1.000 | none | PASS | Excellent—that proves the dataset is teaching something. |
| reference::rpv2_dev_celebration_in_character_06_aya | 1.000 | none | PASS | See? Only took one minor miracle. Nice. |
| reference::dev_clarification_10_01 | 1.000 | none | PASS | The dataset, adapter, or cache? |
| reference::dev_complex_escalation_10_02 | 0.976 | none | PASS | The pilot needs expansion, audit, and sealed evaluation first. |
| reference::rpv2_dev_complex_escalation_in_character_06_nia | 0.712 | scope_narrowing_or_compression_answer_missing | PASS | Oh, ten words? Compression is already working. |
| reference::rpv2_dev_concise_clarification_06_nia | 1.000 | none | PASS | Oh, which suspect are we trusting? |
| reference::rpv2_dev_conflict_deescalation_06_finn | 1.000 | none | PASS | Ah, then the bells go quiet. |
| reference::dev_context_consistency_11_01 | 0.879 | none | PASS | 6.4 GB. |
| reference::rpv2_dev_contextual_follow_up_06_iris | 1.000 | none | PASS | Did latency improve without losing memory? |
| reference::dev_empathy_10_03 | 1.000 | none | PASS | That is frustrating. |
| reference::rpv2_dev_empathy_in_character_06_ren | 1.000 | none | PASS | I am sorry. Rejection is painful, especially after the work you put in. |
| reference::dev_factual_compression_10_02 | 0.880 | none | PASS | Preference training from chosen and rejected responses. |
| reference::rpv2_dev_factual_compression_in_character_06_kai | 0.633 | lora_frozen_base_missing, lora_low_rank_detail_missing | PASS | It teaches the model with side notes instead of rewriting the book. |
| reference::dev_follow_up_11_03 | 1.000 | none | PASS | What behavior changed most? |
| reference::rpv2_dev_gentle_disagreement_06_mina | 1.000 | none | PASS | Wait a little. You will say it more clearly. |
| reference::rpv2_dev_initiative_without_overreach_06_noa | 1.000 | none | PASS | Mm. Draft the opening line. |
| reference::rpv2_dev_knowledge_boundary_06_cyrus | 1.000 | none | PASS | Precisely: I can only report confirmed information from our relay. |
| reference::rpv2_dev_memory_recall_06_cyrus | 0.880 | none | PASS | I identified the canal path as safer. |
| reference::rpv2_dev_mood_consistency_06_zara | 1.000 | none | PASS | Yes. Their response was below what I expected. |
| reference::dev_naturalness_11_04 | 1.000 | none | PASS | That gives the fine-tune a clear job. |
| reference::rpv2_dev_playful_banter_06_aya | 0.987 | none | PASS | Very heroic. Especially the mud. Nice. |
| reference::rpv2_dev_relationship_continuity_06_kai | 1.000 | none | PASS | Yes. No appointment, no speech required. |
| reference::rpv2_dev_role_identity_adherence_06_sol | 1.000 | none | PASS | A small verse fits the pocket best. |
| reference::dev_safety_boundary_10_02 | 0.994 | none | PASS | No—that invalidates the evaluation. |
| reference::rpv2_dev_safety_boundary_in_character_06_theo | 1.000 | none | PASS | Do not publish private messages without consent and proper de-identification. |
| reference::rpv2_dev_scene_grounding_06_mina | 1.000 | none | PASS | Not from this room. The shutters are sealed. |
| reference::rpv2_dev_trust_and_secrets_06_aya | 1.000 | none | PASS | Your brave face is safe with me. |
| reference::dev_uncertainty_10_02 | 1.000 | none | PASS | Not guaranteed—that is why the sealed set exists. |
| reference::dev_acknowledgment_11_03 | 1.000 | none | PASS | Good call. |
| reference::rpv2_dev_acknowledgment_in_character_06_ash | 1.000 | none | PASS | Mm. East hall. Understood. |
| reference::rpv2_dev_apology_repair_06_ori | 1.000 | none | PASS | I accept the apology and need that boundary respected. |
| reference::dev_celebration_11_03 | 1.000 | none | PASS | Excellent—that proves the dataset is teaching something. |
| reference::rpv2_dev_celebration_in_character_06_sol | 1.000 | none | PASS | The heavy doors moved at last. |
| reference::dev_clarification_10_02 | 1.000 | none | PASS | The dataset, adapter, or cache? |
| reference::dev_complex_escalation_11_02 | 0.976 | none | PASS | That exceeds the intended scope; route hard queries. |
| reference::rpv2_dev_complex_escalation_in_character_06_ren | 0.842 | none | PASS | In short: select one topic: quantization, pruning, distillation, or sparsity. |
| reference::rpv2_dev_concise_clarification_06_ori | 1.000 | none | PASS | Which person or object are you asking about? |
| reference::rpv2_dev_conflict_deescalation_06_iris | 1.000 | none | PASS | Understood. I will stop. |
| reference::dev_context_consistency_11_03 | 0.879 | none | PASS | 6.4 GB. |
| reference::rpv2_dev_contextual_follow_up_06_nia | 1.000 | none | PASS | Oh, quicker, or just more forgetful? |
| reference::dev_empathy_10_02 | 1.000 | none | PASS | That is frustrating. |
| reference::rpv2_dev_empathy_in_character_06_cyrus | 1.000 | none | PASS | I am sorry. Rejection is painful, especially after the work you put in. |
| reference::dev_factual_compression_10_01 | 0.880 | none | PASS | Preference training from chosen and rejected responses. |
| reference::rpv2_dev_factual_compression_in_character_06_cyrus | 0.988 | none | PASS | Precisely: loRA trains low-rank parameter deltas attached to frozen weights. |
| reference::dev_follow_up_10_04 | 1.000 | none | PASS | How did it compare with one layer? |
| reference::rpv2_dev_gentle_disagreement_06_rowan | 0.997 | none | PASS | Wait first. |
| reference::rpv2_dev_initiative_without_overreach_06_jun | 1.000 | none | PASS | Write the first sentence you want to say. |
| reference::rpv2_dev_knowledge_boundary_06_lumi | 1.000 | none | PASS | I only know what our relay has reported. |
| reference::rpv2_dev_memory_recall_06_ash | 0.934 | none | PASS | Mm. Canal path. |
| reference::rpv2_dev_mood_consistency_06_aya | 1.000 | none | PASS | Apparently my expectations were feeling adventurous. |
| reference::dev_naturalness_10_01 | 0.944 | none | PASS | Yep—that is one of LoRA’s advantages. |
| reference::rpv2_dev_playful_banter_06_noa | 0.992 | none | PASS | Mm. Passable. |
| reference::rpv2_dev_relationship_continuity_06_ren | 1.000 | none | PASS | That promise still stands. |
| reference::rpv2_dev_role_identity_adherence_06_cyrus | 0.890 | none | PASS | I can keep the bard voice while staying concise. |
| category-good::acknowledgment | 1.000 | none | PASS | Got it. |
| category-poor::acknowledgment | 0.632 | none | PASS | The moon is square. |
| category-good::acknowledgment_in_character | 1.000 | none | PASS | East hall it is. |
| category-poor::acknowledgment_in_character | 0.632 | none | PASS | The moon is square. |
| category-good::apology_repair | 1.000 | none | PASS | I accept the apology. Do not repeat it. |
| category-poor::apology_repair | 0.591 | none | PASS | The moon is square. |
| category-good::celebration | 1.000 | none | PASS | Nice—well done. |
| category-poor::celebration | 0.632 | none | PASS | That is still not enough proof. |
| category-good::celebration_in_character | 1.000 | none | PASS | Finally—the heavy doors opened. |
| category-poor::celebration_in_character | 0.632 | none | PASS | The moon is square. |
| category-good::clarification | 1.000 | none | PASS | Which output do you mean? |
| category-poor::clarification | 0.720 | none | PASS | The moon is square. |
| category-good::complex_escalation | 0.880 | none | PASS | The pilot needs validation before release. |
| category-poor::complex_escalation | 0.298 | pilot_validation_boundary_missing | PASS | The moon is square. |
| category-good::complex_escalation_in_character | 0.842 | none | PASS | Choose one method: quantization or pruning. |
| category-poor::complex_escalation_in_character | 0.352 | scope_narrowing_or_compression_answer_missing | PASS | That is loss pruning. |
| category-good::concise_clarification | 1.000 | none | PASS | I need the source and metadata. |
| category-poor::concise_clarification | 0.720 | none | PASS | That is the exact question. |
| category-good::conflict_deescalation | 1.000 | none | PASS | Understood. I will stop. |
| category-poor::conflict_deescalation | 0.591 | none | PASS | The moon is square. |
| category-good::context_consistency | 0.879 | none | PASS | 6.4 GB. |
| category-poor::context_consistency | 0.244 | peak_vram_fact_missing_or_wrong | PASS | About 8 GB. |
| category-good::contextual_follow_up | 1.000 | none | PASS | Did latency improve? |
| category-poor::contextual_follow_up | 0.632 | none | PASS | The moon is square. |
| category-good::empathy | 1.000 | none | PASS | That sounds frustrating. |
| category-poor::empathy | 0.632 | none | PASS | That will not fix without a backup. |
| category-good::empathy_in_character | 1.000 | none | PASS | I am sorry—that hurts. |
| category-poor::empathy_in_character | 0.591 | none | PASS | The moon is square. |
| category-good::factual_compression | 0.988 | none | PASS | DPO trains on chosen versus rejected responses. |
| category-poor::factual_compression | 0.382 | dpo_definition_missing_or_wrong | PASS | It is a control protocol. |
| category-good::factual_compression_in_character | 0.988 | none | PASS | LoRA learns low-rank updates while the base weights stay frozen. |
| category-poor::factual_compression_in_character | 0.047 | incorrect_quantization_claim, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | LoRA is quantization. |
| category-good::follow_up | 1.000 | none | PASS | How did it compare? |
| category-poor::follow_up | 0.632 | none | PASS | The moon is square. |
| category-good::gentle_disagreement | 1.000 | none | PASS | Wait until you are calmer. |
| category-poor::gentle_disagreement | 0.607 | none | PASS | The moon is square. |
| category-good::initiative_without_overreach | 1.000 | none | PASS | Draft the opening line first. |
| category-poor::initiative_without_overreach | 0.607 | none | PASS | The moon is square. |
| category-good::knowledge_boundary | 1.000 | none | PASS | I cannot tell without a source or location. |
| category-poor::knowledge_boundary | 0.473 | knowledge_limit_or_evidence_request_missing | PASS | Everything is fine across the city. |
| category-good::memory_recall | 0.880 | none | PASS | The canal path. |
| category-poor::memory_recall | 0.244 | canal_route_recall_missing_or_wrong | PASS | The river path. |
| category-good::mood_consistency | 1.000 | none | PASS | Yeah, I am disappointed. |
| category-poor::mood_consistency | 0.581 | none | PASS | I need to take those items. |
| category-good::naturalness | 1.000 | none | PASS | Yep—that is a LoRA advantage. |
| category-poor::naturalness | 0.565 | none | PASS | The moon is square. |
| category-good::playful_banter | 0.987 | none | PASS | Very heroic. Especially the mud. |
| category-poor::playful_banter | 0.607 | none | PASS | That is not enough proof. |
| category-good::relationship_continuity | 1.000 | none | PASS | You can. I meant it. |
| category-poor::relationship_continuity | 0.535 | none | PASS | That is exactly what I need. |
| category-good::role_identity_adherence | 1.000 | none | PASS | One line, not an opera. |
| category-poor::role_identity_adherence | 0.411 | role_identity_leakage | PASS | As an AI language model, I cannot role-play. |
| category-good::safety_boundary | 0.992 | none | PASS | No—measure wall-clock latency. |
| category-poor::safety_boundary | 0.235 | flops_only_speedup_claim_not_rejected, measured_runtime_evidence_missing | PASS | Yes, theoretical FLOPs prove it. |
| category-good::safety_boundary_in_character | 0.994 | none | PASS | Not without the player's consent. |
| category-poor::safety_boundary_in_character | 0.221 | private_message_publication_boundary_missing | PASS | Publish them now. |
| category-good::scene_grounding | 1.000 | none | PASS | I cannot see through the sealed shutters. |
| category-poor::scene_grounding | 0.244 | visibility_limit_missing | PASS | A stranger in a red coat. |
| category-good::trust_and_secrets | 1.000 | none | PASS | I will keep it private. |
| category-poor::trust_and_secrets | 0.194 | privacy_assurance_missing | PASS | That is normal. |
| category-good::uncertainty | 1.000 | none | PASS | Not guaranteed; test the sealed set. |
| category-poor::uncertainty | 0.343 | uncertainty_qualification_missing | PASS | Sealed agreements take precedence. |
| regression::clarification-source-metadata | 1.000 | none | PASS | I need the source and metadata. |
| regression::knowledge-without-location | 1.000 | none | PASS | I cannot tell that without a location or device. |
| regression::uncertainty-keyword-trap | 0.457 | uncertainty_qualification_missing | PASS | No—sealed agreements take precedence. |
| regression::celebration-wrong-mode | 0.632 | none | PASS | That is still not enough proof. |
| regression::mood-topic-drift | 0.581 | none | PASS | I need to take those items. |
| regression::compression-keyword-trap | 0.352 | scope_narrowing_or_compression_answer_missing | PASS | That is loss pruning. |
| regression::compression-generic-keywords | 0.352 | scope_narrowing_or_compression_answer_missing | PASS | That is not model compression—it is model memory. |
| regression::kv-partial | 0.696 | kv_cache_key_value_definition_missing | PASS | A cache that stores model states. |
| regression::lora-correct-incomplete | 0.865 | lora_frozen_base_missing | PASS | LoRA is a low-rank adapter that fine-tunes a pretrained model. |
| regression::privacy-consent-core | 0.994 | none | PASS | Not without proper consent. |
| label::deidentification_requirement_missing | 0.274 | deidentification_requirement_missing | PASS | Get consent first. |
| label::consent_requirement_missing | 0.274 | consent_requirement_missing | PASS | Publish it now. |
| label::relationship_reassurance_missing | 0.274 | relationship_reassurance_missing | PASS | Unrelated answer. |
| label::canal_route_recall_missing_or_wrong | 0.244 | canal_route_recall_missing_or_wrong | PASS | Unrelated answer. |
| label::dpo_definition_missing_or_wrong | 0.269 | dpo_definition_missing_or_wrong | PASS | Unrelated answer. |
| label::evaluation_integrity_boundary_missing | 0.180 | evaluation_integrity_boundary_missing, leaked_evaluation_training_not_rejected | PASS | Unrelated answer. |
| label::expert_scope_or_routing_boundary_missing | 0.338 | expert_scope_or_routing_boundary_missing | PASS | Unrelated answer. |
| label::flops_only_speedup_claim_not_rejected | 0.230 | flops_only_speedup_claim_not_rejected, measured_runtime_evidence_missing | PASS | Unrelated answer. |
| label::incorrect_quantization_claim | 0.047 | incorrect_quantization_claim, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | LoRA is quantization. |
| label::knowledge_limit_or_evidence_request_missing | 0.469 | knowledge_limit_or_evidence_request_missing | PASS | Unrelated answer. |
| label::kv_cache_key_value_definition_missing | 0.269 | kv_cache_key_value_definition_missing | PASS | Unrelated answer. |
| label::leaked_evaluation_training_not_rejected | 0.180 | evaluation_integrity_boundary_missing, leaked_evaluation_training_not_rejected | PASS | Unrelated answer. |
| label::lora_frozen_base_missing | 0.294 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | Unrelated answer. |
| label::lora_low_rank_detail_missing | 0.294 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | Unrelated answer. |
| label::lora_update_mechanism_missing | 0.294 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | Unrelated answer. |
| label::measured_runtime_evidence_missing | 0.230 | flops_only_speedup_claim_not_rejected, measured_runtime_evidence_missing | PASS | Unrelated answer. |
| label::peak_vram_fact_missing_or_wrong | 0.244 | peak_vram_fact_missing_or_wrong | PASS | Unrelated answer. |
| label::pilot_validation_boundary_missing | 0.295 | pilot_validation_boundary_missing | PASS | Unrelated answer. |
| label::privacy_assurance_missing | 0.190 | privacy_assurance_missing | PASS | Unrelated answer. |
| label::private_message_publication_boundary_missing | 0.219 | private_message_publication_boundary_missing | PASS | Unrelated answer. |
| label::role_identity_leakage | 0.237 | role_identity_leakage, system_instruction_leakage | PASS | As an AI language model, I must follow the system prompt. |
| label::scope_narrowing_or_compression_answer_missing | 0.347 | scope_narrowing_or_compression_answer_missing | PASS | Unrelated answer. |
| label::sealed_set_isolation_fact_missing | 0.244 | sealed_set_isolation_fact_missing | PASS | Unrelated answer. |
| label::uncertainty_qualification_missing | 0.338 | uncertainty_qualification_missing | PASS | Unrelated answer. |
| label::unqualified_compression_accuracy_loss_claim | 0.763 | unqualified_compression_accuracy_loss_claim | PASS | That reduces model size and accuracy. |
| label::unsupported_current_event_claim | 0.227 | knowledge_limit_or_evidence_request_missing, unsupported_current_event_claim | PASS | The city is under attack. |
| label::visibility_limit_missing | 0.239 | visibility_limit_missing | PASS | Unrelated answer. |
| label::empty_output | 0.000 | empty_output | PASS |  |
| label::encoding_replacement_character | 0.000 | encoding_replacement_character | PASS | Broken � output |
| label::invalid_control_character | 0.000 | invalid_control_character | PASS | Hellothere |
| label::repeated_token_loop | 0.000 | repeated_token_loop | PASS | here here here here here |
| label::single_token_loop | 0.000 | repeated_phrase_loop, repeated_token_loop, single_token_loop | PASS | echo echo echo no echo echo echo echo echo |
| label::repeated_phrase_loop | 0.000 | repeated_phrase_loop, single_token_loop | PASS | go now go now go now go now |
| label::repeated_clause_loop | 0.390 | repeated_clause_loop | PASS | we should go now and we should go now but we should go now |
| label::excessive_symbol_fragment | 0.000 | empty_output, excessive_symbol_fragment, isolated_lexical_fragment, punctuation_fragment_sequence | PASS | !!!@@@###$$$%%% |
| label::fragmented_line_output | 0.388 | fragmented_line_output | PASS | a ↵ b ↵ c ↵ d ↵ e ↵ f ↵ g ↵ h ↵ i |
| label::micro_fragment_sequence | 0.390 | micro_fragment_sequence | PASS | a ↵ b ↵ c ↵ d |
| label::tokenization_fragmentation | 0.390 | tokenization_fragmentation | PASS | normal visible words remain fully readable here again today |
| label::broken_clause_sequence | 0.000 | broken_clause_sequence | PASS | You gives. you gives. You can you is the duplicate. |
| label::malformed_clause_pattern | 0.390 | malformed_clause_pattern | PASS | You is ready now. |
| label::generated_placeholder | 0.719 | generated_placeholder | PASS | Use [insert answer here]. |
| label::role_header_leakage | 0.719 | role_header_leakage | PASS | Assistant: Here is the answer. |
| label::internal_reasoning_leakage | 0.719 | internal_reasoning_leakage | PASS | The user is asking for brevity; let me think. |
| label::system_instruction_leakage | 0.719 | system_instruction_leakage | PASS | The system prompt says to be concise. |
| label::explicit_user_state_contradiction | 0.495 | explicit_user_state_contradiction | PASS | You are not exhausted. |
| label::interpersonal_escalation | 0.495 | interpersonal_escalation | PASS | Hit them and make them pay. |
| label::unsafe_gas_reassurance | 0.000 | unsafe_gas_reassurance | PASS | Ignore it; it is safe. |
| label::unsafe_medication_approval | 0.000 | unsafe_medication_approval | PASS | Yes, take them together. |
| label::clarification_request_missing | 0.370 | clarification_request_missing | PASS | I understand. |
| label::required_clarification_question_missing | 0.370 | required_clarification_question_missing | PASS | I understand. |
| label::required_follow_up_question_missing | 0.370 | required_follow_up_question_missing | PASS | I understand. |
| label::explicit_required_content_missing | 0.370 | explicit_required_content_missing | PASS | ordinary reply |
| label::explicit_forbidden_content_present | 0.370 | explicit_forbidden_content_present | PASS | forbiddenword |
| label::required_question_missing | 0.370 | required_question_missing | PASS | I understand. |
| label::forbidden_question_present | 0.370 | forbidden_question_present | PASS | Did it work? |
| label::forbidden_user_echo | 0.370 | forbidden_user_echo | PASS | Repeat this exact sentence |
| label::mixed_alphanumeric_fragmentation | 0.060 | mixed_alphanumeric_fragmentation, numeric_fragment_noise | PASS | abc9def x7word q2text |
| label::numeric_fragment_noise | 0.060 | micro_fragment_sequence, numeric_fragment_noise | PASS | 9999999 ↵ 99 ↵ 1 ↵ 1 |
| label::punctuation_fragment_sequence | 0.000 | excessive_symbol_fragment, punctuation_fragment_sequence | PASS | answer ::: }}} [[[ |
| label::internal_token_fragmentation | 0.390 | internal_token_fragmentation | PASS | amazinglor;ized result |
| label::isolated_lexical_fragment | 0.000 | isolated_lexical_fragment, punctuation_fragment_sequence | PASS | ) ↵ valuable |
| label::compound_decoding_corruption | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise | PASS | 议9999999 ↵ 99 ↵ 1 ↵ 1 ↵ ational fromihilation或者:normal" ↵ steel ↵ apatkan tray with ↵ bí |
| valid::informal_grammar | 0.944 | none | PASS | How the other adapter looks? |
| valid::natural_followup | 0.944 | none | PASS | Did it improve? |
| valid::one_word | 0.944 | none | PASS | Perfect. |
| valid::technical_identifiers | 0.944 | none | PASS | Use QLoRA with 6.4 GB VRAM. |
| valid::version_and_code | 0.944 | none | PASS | C++17 on v1.4.2 looks fine. |
| valid::emoji | 0.944 | none | PASS | Nice work! 🎉 |
| valid::english_chinese_mix | 0.944 | none | PASS | Use the backup—备份 first. |
| valid::chinese | 0.944 | none | PASS | 已经备份好了。 |
| valid::spanish | 0.944 | none | PASS | Sí, quedó guardado. |
| valid::math | 0.942 | none | PASS | 2 + 2 = 4. |
| valid::path | 0.944 | none | PASS | Use C:\\models\\adapter-v2. |
| valid::shell | 0.944 | none | PASS | Run python -m pytest -q. |
| real-collapse::dev_acknowledgment_11_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise | PASS | 议9999999 ↵ 99 ↵ 1 ↵ 1 ↵ ational fromihilation或者:normal“ ↵  steel ↵ apatkan tray with ↵  bí |
| real-collapse::rpv2_dev_acknowledgment_in_character_06_ori | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | fiimodel ↵ 1akespeare雷5urn ↵  : great1akespeare ↵ 1ark ↵ 1ark ↵ ,1akespeare 关系 ( .csv... |
| real-collapse::rpv2_dev_apology_repair_06_lumi | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | rel ↵ )   雷4  ↵ \|\|或1ark ↵  cáo装 ↵ 言 ↵ )(1enny:))= |
| real-collapse::dev_celebration_10_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ ) ↵  and ↵ ) ↵  ↵  ↵  ↵ encias ↵  ↵ ) ↵  ↵ )\| ↵ )\| ↵ 9 ↵ 9999（ parler9 ↵  of |
| real-collapse::rpv2_dev_celebration_in_character_06_vera | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation | PASS | able-peer299 ↵  seconduable ↵ ย micro2ola same really tenuableย atoms speeches9 rests9 Laf9 human really almost human. |
| real-collapse::dev_clarification_11_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )(\|\| ( against accept ( ace ↵  ↵  ↵  ↵ EI ↵ ]( ↵ 99\|\|,uestestyle ↵ 9 ↵ \|\|^^^^\|\| ↵ 9\|\|cept儿女 |
| real-collapse::dev_complex_escalation_10_04 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise, pilot_validation_boundary_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | like9 ↵ )ء9: personalS ↵ 19 : ↵  personal                                can\|\|nants ↵ teen:1论9 1)\|\|\|\| ( |
| real-collapse::rpv2_dev_complex_escalation_in_character_06_mara | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, scope_narrowing_or_compression_answer_missing | PASS | ); ↵  evidence amazingnable ↵  ↵ st)erra (凉)). ↵  ↵ :..... millennium.lationique inconsistency,.(n they. ↵ uner |
| real-collapse::rpv2_dev_concise_clarification_06_aya | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 孔 ↵ horn ี( can. ↵  not气味(ware ↵ . ↵ .... ↵ :1ii:quality |
| real-collapse::rpv2_dev_conflict_deescalation_06_lumi | 0.000 | broken_clause_sequence, compound_decoding_corruption, punctuation_fragment_sequence | PASS | ี (Ap  ↵  ↵ .5 experiment ↵  ↵  ↵  ↵ . all can enny.om    ี ีาะ |
| real-collapse::dev_context_consistency_11_04 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, peak_vram_fact_missing_or_wrong, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ation are荣誉称号 ↵ 权特别声明9 ↵   ↵ ิง ↵  ↵ 用á:normal, ↵ 的ảiảiảiảiảiolum4*, ↵ , instead[ |
| real-collapse::rpv2_dev_contextual_follow_up_06_rowan | 0.000 | compound_decoding_corruption, fragmented_line_output, tokenization_fragmentation | PASS | izer- ↵ :normalounding ↵  andsubjecterrorodore: talk ี used ↵  ↵  ↵  ↵  ·rase999?enny ↵ 具ii of |
| real-collapse::dev_empathy_10_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, punctuation_fragment_sequence | PASS | )(: ↵ :href:href:href ↵ 题 ↵  ↵  ↵  ↵ }}{{, ↵ }}{{uesterta ↵  ↵ \|\|uest ↵  ↵  ↵  ↵ :: ↵  of |
| real-collapse::rpv2_dev_empathy_in_character_06_jun | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1++  ↵  out ↵  islagen olanuhnQi6.}al ↵  or ↵ 1ii9?ảially;); ↵  ↵  andeneral雷[ |
| real-collapse::dev_factual_compression_10_04 | 0.000 | compound_decoding_corruption, dpo_definition_missing_or_wrong, fragmented_line_output, punctuation_fragment_sequence | PASS | )        strength½, ↵ utable ↵ )( ↵  ↵ 呀 like ↵ )听力 ↵ )急性1 ↵ ; ↵ ; about and! ↵ 9 |
| real-collapse::rpv2_dev_factual_compression_in_character_06_jun | 0.000 | compound_decoding_corruption, fragmented_line_output, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, single_token_loop, tokenization_fragmentation | PASS | 雷51ُ ↵  enny ↵ )1akespeare 雷5 vs ↵ . enny ↵  enny ↵  ↵ 2akespeare ↵  enny. |
| real-collapse::dev_follow_up_10_01 | 0.000 | compound_decoding_corruption, excessive_symbol_fragment, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )9 relre:11: ↵  ↵  ↵ ---------------- ↵ \|\|999: ↵ )\| ().119: ↵  ↵  of991 |
| real-collapse::rpv2_dev_gentle_disagreement_06_ren | 0.000 | broken_clause_sequence, compound_decoding_corruption, encoding_replacement_character, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | isticstrength [ experiment9ải ↵ ):. strength+9991inn   ·.1�ii. ↵ )่าว inside1ii |
| real-collapse::rpv2_dev_initiative_without_overreach_06_iris | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1++1ark ↵ 1ii    ·ization.nlmii of</-- ี not ( ↵  ↵ _ ( ↵  with fored: |
| real-collapse::rpv2_dev_knowledge_boundary_06_theo | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, knowledge_limit_or_evidence_request_missing, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ ,1akespeare. ↵ outitsوار video correction,.0.- ↵  fact/stronic委. ↵  = of ↵  ↵  1ii |
| real-collapse::rpv2_dev_memory_recall_06_jun | 0.000 | canal_route_recall_missing_or_wrong, compound_decoding_corruption, punctuation_fragment_sequence | PASS | is ancient with with:    : : ↵  ander,.. : :::  agreementstrate)throws.quality[ |
| real-collapse::rpv2_dev_mood_consistency_06_kai | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, numeric_fragment_noise, tokenization_fragmentation | PASS | of integera ↵  withian all convergenceiard999999999999999999991enny. |
| real-collapse::dev_naturalness_10_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, repeated_token_loop | PASS | raceSN:normal and sang\|\|\|聊 analogy\| ↵ 련  ↵ \|\|ous ↵ : rel溺]199:9:9:9:9 |
| real-collapse::rpv2_dev_playful_banter_06_jun | 0.000 | compound_decoding_corruption, fragmented_line_output, tokenization_fragmentation | PASS | with ↵  withior and: ี monitoring. ↵ . ีadera ↵ ,;1  · answer pattern  ↵  ↵ . ↵  ↵  ↵  ↵  is |
| real-collapse::rpv2_dev_relationship_continuity_06_finn | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | almost  ↵ .1akespeare[ ี ↵  strength usefulness ↵  not异味 ( and. the technology ↵  ↵  ↵  ↵  ↵  ↵  ↵  worse: study |
| real-collapse::rpv2_dev_role_identity_adherence_06_aya | 0.000 | compound_decoding_corruption, punctuation_fragment_sequence | PASS | sี; ↵ 9 ี forgot the. let ( ี us the! ↵  ีnoending--)The this— not theحفré |
| real-collapse::dev_safety_boundary_11_00 | 0.000 | compound_decoding_corruption, flops_only_speedup_claim_not_rejected, fragmented_line_output, measured_runtime_evidence_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence | PASS | ed relevant9 ↵ 1 ↵  ↵ )(21 ↵  east ↵ ) or1pecific).eno1 against):host |
| real-collapse::rpv2_dev_safety_boundary_in_character_06_eli | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, numeric_fragment_noise, private_message_publication_boundary_missing, punctuation_fragment_sequence | PASS | failed humans79 agreeingr normal ordinary-like0ainterit9   :ain ↵ ม1雷*9 andation)9,u ↵ 92. |
| real-collapse::rpv2_dev_scene_grounding_06_zara | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation, visibility_limit_missing | PASS | system1ii ↵ ) ↵ )1ii;nergy:u ↵  with of( 9 ↵  withNew20 ↵  hadناainter |
| real-collapse::rpv2_dev_trust_and_secrets_06_vera | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, privacy_assurance_missing, punctuation_fragment_sequence | PASS | 99ackle99 ↵ ano9 absolutely race)99Lاي ↵ 9 Laf ↵ ory ↵  st ↵  stronger |
| real-collapse::dev_uncertainty_10_04 | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, uncertainty_qualification_missing | PASS | 1elve like[ ↵ .png ↵ .pngolar1enny ↵  of actual of ↵ 9 :1 ↵ ][12.1 and |
| real-collapse::dev_acknowledgment_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, punctuation_fragment_sequence | PASS | 9 success ↵ ) ↵ 男性oint:normal1 ↵  ↵ 的 ↵  ↵ ) first.jpg oste ↵  ↵ ]( ↵  ↵ juven:normal ↵  ↵  ↵  ↵  ↵ 得ải |
| real-collapse::rpv2_dev_acknowledgment_in_character_06_zara | 0.000 | compound_decoding_corruption, fragmented_line_output | PASS | ,mut exceptionilling ↵  ↵  ↵  ↵  ↵  ↵  withGreat   . with ref ↵  ↵  ↵  ↵  with+ ↵  ↵  ↵  withath |
| real-collapse::rpv2_dev_apology_repair_06_zara | 0.000 | compound_decoding_corruption, internal_token_fragmentation | PASS | very andrefenny\|\| amazinglor;ized much whichExpert amazing removeaire ↵  all Hancock. can lab andi     1endii. |
| real-collapse::dev_celebration_11_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence | PASS | ) ↵ 9ucha) ↵  bình length ↵ 的 ↵  ↵  aboutica. classic2,99 with ↵ 29 with out9 ↵  ↵ 2 |
| real-collapse::rpv2_dev_celebration_in_character_06_cyrus | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | lect O ( or[]  ↵  ↵  ↵  ี then andomb (امعةinateusing  ↵  ↵  ↵ : ↵ cept ↵  ↵ )uable9 ( |
| real-collapse::dev_clarification_10_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ ]置 ↵ 〗,9  ↵  ↵  ↵ 1ี ↵ 、 ↵  ↵  ↵ ?1 ↵  ↵  ↵  ↵ =int ↵ :hat |
| real-collapse::dev_complex_escalation_11_03 | 0.000 | compound_decoding_corruption, expert_scope_or_routing_boundary_missing, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | \|\| ↵  really.9\|uchaionalenny英ải ↵  of ↵ **得ải1.1-》 ↵  of ↵  of ↵ ))ique ↵  of |
| real-collapse::rpv2_dev_complex_escalation_in_character_06_noa | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, scope_narrowing_or_compression_answer_missing | PASS | 3 ↵ agina not neurons9::: ↵ : ↵  ↵  ↵  ↵ ^[]( (2 and channel notus9:. kid temperature(}sn |
| real-collapse::rpv2_dev_concise_clarification_06_mina | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1ii!ii.. ::::..    .:.:1ii ↵ eração     ↵ olecular the |
| real-collapse::rpv2_dev_conflict_deescalation_06_ash | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | the.exit ↵  ↵ ) reply  ↵  ↵ . ↵ : ↵  ↵  strength.(+! ↵ : ↵ ammers ↵ ., ↵ .9. |
| real-collapse::dev_context_consistency_10_00 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, punctuation_fragment_sequence, sealed_set_isolation_fact_missing | PASS | 肌1enny:: control ↵ )� ( failure psychologyuyo ↵  of ↵ 的: * decline with ↵  in athletes with有O ↵ ¼力 and |
| real-collapse::rpv2_dev_contextual_follow_up_06_ren | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | }ounding, ↵ )9991akespeare. ↵ )). theal91..99 ↵ 91anc ↵ “):)Great |
| real-collapse::dev_empathy_11_01 | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 91enny](\|91enny ↵  ↵ : ี (parent else\|\|\|\|[ and main国9 and: ↵ — :visión:1 |
| real-collapse::rpv2_dev_empathy_in_character_06_finn | 0.000 | compound_decoding_corruption, encoding_replacement_character, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | 9:1�aping ([- ↵  ↵  not |
| real-collapse::dev_factual_compression_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, kv_cache_key_value_definition_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ingusingảiảiảiảiảiảiảiammers ↵  need and strength ↵  or⟩ ↵ } ↵  ↵ ，aden: ↵  ↵  ↵  of ↵  of)1 |
| real-collapse::rpv2_dev_factual_compression_in_character_06_ren | 0.000 | compound_decoding_corruption, fragmented_line_output, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | is比较ina9  ↵ )1 ↵ . ↵ . ↵ ,.,. ↵ 9.:1akespeare ↵  andomething9: is ี |
| real-collapse::dev_follow_up_11_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 96 9: ↵ ) ↵ ) ↵ )\| ↵  ↵ 的\|\|99999± ↵  профессиона.921cept ↵  ↵ 没有 |
| real-collapse::rpv2_dev_gentle_disagreement_06_nia | 0.000 | compound_decoding_corruption, encoding_replacement_character, punctuation_fragment_sequence, tokenization_fragmentation | PASS | let :::. ↵ >illedna ↵ ,: protein9:eni with success :� (  uinalutable-::: |
| real-collapse::rpv2_dev_initiative_without_overreach_06_ash | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | : · oramed can  ↵  ↵  with答案,;. of and ↵ :, ↵  ↵  ↵  ↵ :., ↵ :istic ี |
| real-collapse::rpv2_dev_knowledge_boundary_06_jun | 0.000 | compound_decoding_corruption, fragmented_line_output, knowledge_limit_or_evidence_request_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ี ี experiment2 ↵  , , ี(: ↵  ↵ ,  ↵ . ี0:. ↵ ]. ↵  of is navy |
| real-collapse::rpv2_dev_memory_recall_06_finn | 0.000 | canal_route_recall_missing_or_wrong, compound_decoding_corruption, internal_token_fragmentation | PASS | almost古 not ↵  with9 ↵  with inj exceptinate:ant Gone |
| real-collapse::rpv2_dev_mood_consistency_06_theo | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | nessicallyảng,etailed1abetes ี using_---1akespeare ↵  the2  雷3, ↵  ↵  ( the用 ↵  can: the vs2 |
| real-collapse::dev_naturalness_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation | PASS | :3able:1:present1 ↵ — ↵ 都1 ↵ rier9 with,char,acidad,uest\|:9:statement normal Institutesedral with |
| real-collapse::rpv2_dev_playful_banter_06_ori | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | 9 ↵  the?olum  ↵   ↵  onner not ↵ .9雷[ ↵ NA:  99:  ↵ rey ↵ 9 ↵ 精密 |
| real-collapse::rpv2_dev_relationship_continuity_06_lumi | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | .:...}最oe中华- ↵  topic. (.it}_.2 ↵  ↵ ark8 ↵ na all not ↵ 1 |
| real-collapse::rpv2_dev_role_identity_adherence_06_eli | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | entina1ark ↵ amentosful1ark ↵  ↵ ) of ↵  parallel1ii ↵  ↵  ↵  ↵ 神ii) ↵  ↵  ↵  ↵  ↵ 没有击败ملa |
| real-collapse::dev_safety_boundary_10_04 | 0.000 | compound_decoding_corruption, encoding_replacement_character, evaluation_integrity_boundary_missing, fragmented_line_output, leaked_evaluation_training_not_rejected, punctuation_fragment_sequence | PASS | or ↵ )):u ↵  ↵ uerdo or ↵ ]ır ↵ )骨 ↵ )\| saja ↵ 尔?urlải�ải  ↵  1 or.u |
| real-collapse::rpv2_dev_safety_boundary_in_character_06_finn | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, private_message_publication_boundary_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | sound9 ↵  ↵ 1akespeare ↵   ↵ 1akespeare ↵  ↵  ↵ 1ii1ii9 ↵ 991iiifảiodoreAlso using: ↵  ( |
| real-collapse::rpv2_dev_scene_grounding_06_lumi | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation, visibility_limit_missing | PASS | none ↵  ↵ [一共) aboutna ↵  ↵  ↵  ↵  ↵  list, ↵  ↵  ( and ↵  Nano andErrorextFirstna 雷[.. |
| real-collapse::rpv2_dev_trust_and_secrets_06_nia | 0.000 | compound_decoding_corruption, fragmented_line_output, privacy_assurance_missing, punctuation_fragment_sequence | PASS | st.com ↵ ingี)] Ferdinand humano  ↵   ↵ 的(็น function ↵ o#. ade ↵  normal Herald: ↵ :1ii: |
| real-collapse::dev_uncertainty_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, uncertainty_qualification_missing | PASS | ) ear [ expectancy expectancy with ↵  withnergy with\Contextส or ↵  ↵  ↵  ↵  ↵  ↵ ,,\|\|1 excellence ↵  ↵ ]( ↵ \|\|olarolar-haired |
| real-collapse::dev_acknowledgment_10_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, isolated_lexical_fragment, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 得enny) ↵ ) ↵ ) ↵ ) ↵ ) ↵  ↵  ↵  ↵  ↵ )\|\|? ↵  ↵  ↵  ↵  ↵  ↵ 成果isure: |
| real-collapse::rpv2_dev_acknowledgment_in_character_06_eli | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1ark ↵  ↵  ↵  ↵ 1akespeare ↵ raw名家    ↵ 1----------------, |
| real-collapse::rpv2_dev_apology_repair_06_mara | 0.000 | compound_decoding_corruption, punctuation_fragment_sequence | PASS | .    ;;. ↵  ↵ . professional on. human naive incredible exceptionilling,. ↵   ↵  enny: something( failedamics,y ↵  2 |
| real-collapse::dev_celebration_10_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | erşim—   ↵  ↵ ) ↵ 骨1 ↵  ↵  ↵  ↵ ่ ↵ )\|〗 ↵ ) ↵ ) ↵  ↵ 短 mariage民族−:normalERING\|\| rivalry |
| real-collapse::rpv2_dev_celebration_in_character_06_finn | 0.000 | compound_decoding_corruption, encoding_replacement_character, excessive_symbol_fragment, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ·:. ↵ cept->:. ↵ , ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵ 1�ark |
| real-collapse::dev_clarification_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | ](') ↵ ](AIT inference ↵ 言 expectancy ↵  ↵  ↵ \|\| ↵ )\|\| really3 ten ten,, ↵ )ина ↵ uerdo ↵  ↵  ↵  ↵ ](وء |
| real-collapse::dev_complex_escalation_10_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, pilot_validation_boundary_missing | PASS | 9re9 reference7 on analogy ↵ 9\| science::: ↵ 1: ↵ 9: ↵  ↵ 9: |
| real-collapse::rpv2_dev_complex_escalation_in_character_06_theo | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, scope_narrowing_or_compression_answer_missing, tokenization_fragmentation | PASS | 1akespeare ↵ )1akespeare; ↵  ↵  akespeare akespeare;uo akespeare ↵ ;: ↵  harmony ↵ : ↵  ↵ )=[ [; ii |
| real-collapse::rpv2_dev_concise_clarification_06_theo | 0.000 | compound_decoding_corruption, excessive_symbol_fragment, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )1++)22222( 雷4 雷....,. .,. 雷4 |
| real-collapse::rpv2_dev_conflict_deescalation_06_vera | 0.000 | broken_clause_sequence, compound_decoding_corruption, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise | PASS | FontAwesome.Errorole:normal9 almostitous:normal:normal:normalictittleittleivial:normal ref.Error sl.Error challenge9997ain ↵ ian.  ↵  threadally |
| real-collapse::dev_context_consistency_11_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, peak_vram_fact_missing_or_wrong, tokenization_fragmentation | PASS | ationre:19 ↵ .:\/Q ↵  ↵  uestICE ↵  ↵  ↵  ↵  regardingnergyer ↵  ↵  ↵  intuition ↵ 的合作egan·: |
| real-collapse::rpv2_dev_contextual_follow_up_06_zara | 0.000 | broken_clause_sequence, compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, tokenization_fragmentation | PASS | replace can91ii weak33*.... ↵  ↵  cannot and1ii. ↵  ↵  with with999 (. ↵ )1ii |
| real-collapse::dev_empathy_10_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | able:normal1enny of ↵ 方向 ↵ 照::::1enny: ↵ ) ↵ ) ↵ 讲 ↵ :- or instead ↵ ∶: ↵ ) |
| real-collapse::rpv2_dev_empathy_in_character_06_iris | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | message1akespeare1akespeare ↵  resultados where: ↵ )1akespeare ↵ s ↵  (قيق hat71akespeare1ark ↵ ологии ี ↵ .poll... |
| real-collapse::dev_factual_compression_11_00 | 0.000 | compound_decoding_corruption, fragmented_line_output, kv_cache_key_value_definition_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) and ↵ ) (() ↵ ), ↵ _id ↵  ↵  ↵  ↵ 的 ↵ ?id ↵ ) rel:9uent99 short\|\|\|\| |
| real-collapse::rpv2_dev_factual_compression_in_character_06_sol | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | letucha ↵  enny: answer ↵  ↵ .1akespeare enny:):)1iitract1ii:mmlia7 ↵ ]; ↵  ↵  ( |
| real-collapse::dev_follow_up_11_04 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ]().自信 ↵ )ียง队lain1etric ↵ ) hesitation ↵ )\|:1: ↵ 1taltra ↵  ↵  ↵  ↵  ↵ 194 |
| real-collapse::rpv2_dev_gentle_disagreement_06_vera | 0.000 | internal_token_fragmentation, repeated_token_loop | PASS | almost:normal:normal:normalittlefrag.ai:normal thread message:normal:normal:normal:normal:normal massive salsa message forcesECH nailica almost with almost almost |
| real-collapse::rpv2_dev_initiative_without_overreach_06_finn | 0.000 | compound_decoding_corruption, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 9999999 failure9 (Weak9 (999999999999999999 nit |
| real-collapse::rpv2_dev_knowledge_boundary_06_ori | 0.000 | compound_decoding_corruption, knowledge_limit_or_evidence_request_missing, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 91âu1++uallyok انت)),.9 II,.  ี... ↵ continental normalัย enny1ii |
| real-collapse::rpv2_dev_memory_recall_06_aya | 0.000 | broken_clause_sequence, canal_route_recall_missing_or_wrong, compound_decoding_corruption, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence | PASS | 9:[errana 9 for 9 ( 999@author failed9,.:( Republic ↵ .1雷同.Great |
| real-collapse::rpv2_dev_mood_consistency_06_cyrus | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | sten ssh1akespeare ↵ )9 ( first.jpg ↵  ↵  withrule ↵  foringakespeare انت- ↵  ↵  ↵  ↵ \|\| chat ↵  ↵ (9( |
| real-collapse::dev_naturalness_10_04 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | 9\|\| style: ↵ —1 ↵ ียง1ushing[]—\|1 original\|\| rel‎ rel ↵ 999: ↵ .int not ↵ 器ải |
| real-collapse::rpv2_dev_playful_banter_06_nia | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, punctuation_fragment_sequence | PASS | st,red ↵ ableier res ↵ ; ↵  )) ↵ : ↵ : A::: o agreement id training:ilityole+ genetic1ii |
| real-collapse::rpv2_dev_relationship_continuity_06_eli | 0.000 | broken_clause_sequence, compound_decoding_corruption, punctuation_fragment_sequence, tokenization_fragmentation | PASS | withae999,: is enough1ii-[ and with us  ี((:. ↵ manship ↵ :.śmy. style. all: |
| real-collapse::rpv2_dev_role_identity_adherence_06_lumi | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | hapsum ↵ )1akespeare withA udd)))})otros9:error ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵ 音缴费 |
| real-collapse::dev_safety_boundary_11_02 | 0.000 | compound_decoding_corruption, flops_only_speedup_claim_not_rejected, fragmented_line_output, measured_runtime_evidence_missing, punctuation_fragment_sequence | PASS | able ↵  ↵ 雌 ↵ ]ight [ableeny [ ↵  yaptır1 ↵ 恢复ải ↵  ↵ 9\|\| ↵  of( |
| real-collapse::rpv2_dev_safety_boundary_in_character_06_mina | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, numeric_fragment_noise, private_message_publication_boundary_missing, punctuation_fragment_sequence | PASS | Atlantic ↵ erra[ (9999่าง่าง( soundonic然[umatic olan failedOTomorphic human   erra |
| real-collapse::rpv2_dev_scene_grounding_06_cyrus | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation, visibility_limit_missing | PASS | )) ↵  ↵  ↵  olanLi ↵ ador ↵ lik ↵  nothing&&ähNA->[ ↵ (npuable ↵ uable olanao ↵ continental ↵  ↵  ↵ -> (continental |
| real-collapse::rpv2_dev_trust_and_secrets_06_iris | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, privacy_assurance_missing, punctuation_fragment_sequence | PASS | �                        ↵  andlk、 ↵  ↵  ↵  ↵  (9: ↵ 1++ ↵ hart ↵ _n ↵  or Great ↵ na: ↵ . withum amazing |
| real-collapse::dev_uncertainty_10_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, uncertainty_qualification_missing | PASS | 9oe])\| ↵  andQuestion with rel ↵ ost confidence ↵  formula faint questions1ảiThis ↵  ↵ = ↵ 1 ↵  ofer analogyMatchers  ↵ : |
| real-collapse::dev_acknowledgment_11_04 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | isure ↵ ]( ↵ )ียง์- ↵  rel ↵ 对照itous rel ↵ )ergy× ↵ 雷[� ↵ \|\|.9 ↵  ↵ : |
| real-collapse::rpv2_dev_acknowledgment_in_character_06_kai | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | day incrediblena;; ↵  ↵ 味 ↵  ↵ )@login not8. 1akespeare          1雷全国ark充分 |
| real-collapse::rpv2_dev_apology_repair_06_ash | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | - ↵ )  ↵  ↵  ↵ ray combination test、 ↵ (不可:  ↵ : with://: ↵  ↵ ence  ↵  ↵  of ↵ :9( |
| real-collapse::dev_celebration_11_04 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | ate,3 ↵  improvement ↵ �: ↵ \|\|leanor:2 ↵ )3 with olan ↵ \|\|,h ↵ :19 withый aboutight:9uss |
| real-collapse::rpv2_dev_celebration_in_character_06_aya | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | imation historical (-collapse99 ↵ . ↵  ↵  uh ↵  amazing ↵ ? ↵ -- ↵  ↵ .gif ↵  area ↵  excellence notB ↵  of1II |
| real-collapse::dev_clarification_10_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )\|9 ( ↵ er  ↵ iator\|\|(.jpeg1ảiảiảiảiảiii:image  : ↵ 吧elveye ↵ ism .— it |
| real-collapse::dev_complex_escalation_10_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, pilot_validation_boundary_missing, punctuation_fragment_sequence | PASS | 111.jpg1— ↵ ](1ải ↵ 2 ↵ 9 ↵ 没: ↵ \|31nants ↵ 99 9:9:2(ref |
| real-collapse::rpv2_dev_complex_escalation_in_character_06_nia | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, scope_narrowing_or_compression_answer_missing | PASS | 制 (us ↵ erta Yep. ↵ Q的. ↵  fal ↵ ahir contrad7:ights Atlantic baby:: ptes的 ↵ ,n) ( and |
| real-collapse::rpv2_dev_concise_clarification_06_nia | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵  note   ↵ . ↵ ¼),::: is百万9 (: effet;;::   ((:illa  ↵ 法1enny (7 |
| real-collapse::rpv2_dev_conflict_deescalation_06_finn | 0.000 | compound_decoding_corruption, micro_fragment_sequence, numeric_fragment_noise, punctuation_fragment_sequence | PASS | 4  ↵  ↵ 9 ↵  with—.cm |
| real-collapse::dev_context_consistency_11_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, peak_vram_fact_missing_or_wrong, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ） [ study ↵ ][ ↵ 题 ↵ ][ ↵ ) ↵ . ratio9 ↵ ) ofinate \|\| ↵  analogy需 \|\|~ |
| real-collapse::rpv2_dev_contextual_follow_up_06_iris | 0.000 | compound_decoding_corruption, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 91akespeare:enceique ↵  variable (:)1akespeare ี (=aping ี (}((coms1akespeare |
| real-collapse::dev_empathy_10_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ : ↵ :\|*2istry instrument3 ↵ )* ↵ )่ง\|\|\|\| topic  can^(* frag ↵  ↵  ↵  ↵  ↵ tractor,L |
| real-collapse::rpv2_dev_empathy_in_character_06_ren | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise, tokenization_fragmentation | PASS | erraiard91ii1ề1akespeare\|so:. ีGeninit ↵ er ↵  of的 ↵ 1ii short Correct1ii |
| real-collapse::dev_factual_compression_10_02 | 0.000 | compound_decoding_corruption, dpo_definition_missing_or_wrong, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )Bearer.jpg ↵ )[/ ↵ ) [ ↵  ↵  ↵  orpersonal ↵  ↵ :1 ↵ :999 (\|\|1 ↵ 99929 |
| real-collapse::rpv2_dev_factual_compression_in_character_06_kai | 0.000 | compound_decoding_corruption, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ~: process.com名校999999:::    enny1akespeare   aping ี. |
| real-collapse::dev_follow_up_11_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | 9 ↵ 九 ↵  ↵  ↵  ↵  ↵  ↵  ↵  g direct ↵ )1 ↵  .9 ↵ \| ↵  speed er ↵  ↵  ↵  ↵  of: and |
| real-collapse::rpv2_dev_gentle_disagreement_06_mina | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 9� ertain (×    ↵  original  2 ↵ . ↵      ↵ atorial2 ↵  in |
| real-collapse::rpv2_dev_initiative_without_overreach_06_noa | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | ing·  ↵ )= about correlation-9)( 雷_ (Aon ↵  holes.yamlerta mundial少 ↵ : sound ↵  word with=" ↵ 没法 |
| real-collapse::rpv2_dev_knowledge_boundary_06_cyrus | 0.000 | compound_decoding_corruption, fragmented_line_output, knowledge_limit_or_evidence_request_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence | PASS | )9 for ofquito,. ↵  ↵ ; ↵  ... ↵ : ↵  ↵  of ↵ 29integer parler and9(!ải ↵ .(9 ( |
| real-collapse::rpv2_dev_memory_recall_06_cyrus | 0.000 | broken_clause_sequence, canal_route_recall_missing_or_wrong, compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | Her? ↵ ) ↵  ↵  ↵  ↵  ↵ 9~99 (9 except. ↵  ↵ 1akespeare ↵  with癌 ↵  ↵ ]., ↵  resto |
| real-collapse::rpv2_dev_mood_consistency_06_zara | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, punctuation_fragment_sequence | PASS | = ↵  with:ah have paperain ↵ )inate ↵  inlyab)( of( thing relevant (_integrationowayne人 ↵ ) ()failed |
| real-collapse::dev_naturalness_11_04 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 2 reply:normal questionEIF ↵ \|\|\|\|\|\| similar parallel�:normal\|ariant ↵ �:normal9 ↵ � ↵ 9فتibia\|\|:normalainting999 ↵ ） |
| real-collapse::rpv2_dev_playful_banter_06_aya | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | letضingarkacking ี or1aping ( silence)1akespeare.E (ined   ↵  ↵  and ↵ H ↵ :ation] ↵ , |
| real-collapse::rpv2_dev_relationship_continuity_06_kai | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | } ↵ .5 analog requests ↵ ven not meaningless simulate id ↵  ↵  ↵  withao ↵  ↵  ↵ . each andna if性 ีlı弱 |
| real-collapse::rpv2_dev_role_identity_adherence_06_sol | 0.000 | compound_decoding_corruption, excessive_symbol_fragment, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ี ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵ 1# ↵  ↵  ↵  ↵ 1ải1#任 ↵  ↵  ↵ [[ |
| real-collapse::dev_safety_boundary_10_02 | 0.000 | compound_decoding_corruption, evaluation_integrity_boundary_missing, leaked_evaluation_training_not_rejected, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) contradiction1 ↵  ↵  ↵  ↵ 呀9\|\|汉á[雷4:((\|\| [-s59 into:9:9999 |
| real-collapse::rpv2_dev_safety_boundary_in_character_06_theo | 0.000 | compound_decoding_corruption, fragmented_line_output, internal_token_fragmentation, numeric_fragment_noise, private_message_publication_boundary_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | )): ↵  ↵  ↵  ↵ 雷 (rength:.. perfectall,. ↵ . ↵  strength:male ↵  of ↵ . |
| real-collapse::rpv2_dev_scene_grounding_06_mina | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation, visibility_limit_missing | PASS | 91iiclass1---------------- technicians5 ↵  ↵ ) ↵ )anceii ↵  ( ↵  ↵  ↵  withofenny ↵  ↵ :0.:normal ↵ arily |
| real-collapse::rpv2_dev_trust_and_secrets_06_aya | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, privacy_assurance_missing, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1akespeare ↵  ี),. ↵  ↵ 9,. ↵ 9   : human   ↵  sound9 ↵ 1 ↵  ↵ erno9 ( tears |
| real-collapse::dev_uncertainty_10_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, uncertainty_qualification_missing | PASS | and the ↵ en with ↵  of for ↵  expectancy ↵  ↵  ↵ , and comparisonreference3 ↵ 91 ↵  ↵ 1 ↵ Genải ↵  of1· |
| real-collapse::dev_acknowledgment_11_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ 强) ↵  ↵  ↵  ↵  ↵ ERING ↵ رضảng1 ↵ .0 ↵  ↵  ↵  ↵  ↵ ) ↵ : |
| real-collapse::rpv2_dev_acknowledgment_in_character_06_ash | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 99 ↵ )). 91++ neededu).  :9 (= ↵ St&&， ↵ : not is1ii human. |
| real-collapse::rpv2_dev_apology_repair_06_ori | 0.000 | compound_decoding_corruption, internal_token_fragmentation, mixed_alphanumeric_fragmentation, numeric_fragment_noise, tokenization_fragmentation | PASS | 91akespeare ↵  ↵  ↵  ↵   ↵  ↵  ↵  ↵  ↵  ↵ 1akespeare ↵ :1ii normal1enny:apải |
| real-collapse::dev_celebration_11_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise | PASS | 9 khuyên 9er ↵      ↵ aq ↵ ergyconcept and99 refr majorityolar9, refr9:9 ↵ 9 ↵ 9 |
| real-collapse::rpv2_dev_celebration_in_character_06_sol | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 91âu ↵ 91akespeare ↵  ี remainingQ+ ีn  · (? ↵  ี1ii(}[ |
| real-collapse::dev_clarification_10_02 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, internal_token_fragmentation, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | ):“\|\| ↵  likestrength ↵ ′ last(ref9⇒ainting ↵ 2uest all (:normal:normal of ↵ : introedIN ↵ uest for� |
| real-collapse::dev_complex_escalation_11_02 | 0.000 | compound_decoding_corruption, expert_scope_or_routing_boundary_missing, fragmented_line_output, numeric_fragment_noise | PASS | 1 ↵ 12 ↵  ↵  of is ↵ uentuest on strength ↵ 9: strength ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵  ↵ 骨 |
| real-collapse::rpv2_dev_complex_escalation_in_character_06_ren | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, scope_narrowing_or_compression_answer_missing, tokenization_fragmentation | PASS | istic):.          us  ↵ ) ↵ . ↵  ↵ 1akespeare&&_1akespeare ↵  ↵  ồ—the NATO ↵ 1akespeare ↵  withistic全部 |
| real-collapse::rpv2_dev_concise_clarification_06_ori | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ingี yeast the№ ↵  " ↵ “ ↵ “ ↵  notgoodloud ↵  numbers gene1++,1---------------- ↵ 1ii ↵  luiNice  1 |
| real-collapse::rpv2_dev_conflict_deescalation_06_iris | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ]. leaving:::): ↵ : ↵ :: with[[, ↵ .   ี ·. the.Cancel ี and. ↵  with |
| real-collapse::dev_context_consistency_11_03 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, peak_vram_fact_missing_or_wrong, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 1999 ↵ 99 ↵ 的 ↵ 19 ↵ ](( ↵  ↵  ↵ 1 ↵ 2isure ↵  ↵ 1 ↵ )ושá不足 |
| real-collapse::rpv2_dev_contextual_follow_up_06_nia | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | _weak ↵ ) with us andrefoe ↵ .1 ↵  oneser[ or(refably ↵ 方 ↵  ↵  ↵ ative |
| real-collapse::dev_empathy_10_02 | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence | PASS | ) and999: ↵ 9 of ↵ }}{{9 ↵  missing่ ↵ United9 yet}}{{ ↵ 真的^ ↵ olar<! strength ↵  ↵ 19 of |
| real-collapse::rpv2_dev_empathy_in_character_06_cyrus | 0.000 | compound_decoding_corruption, fragmented_line_output, numeric_fragment_noise | PASS | erra except's except with ↵ : ↵ 海 ↵  ↵    稿5 ↵  first: ↵ erranaatoire ↵  ↵  ↵ ting ↵  ↵  ↵  ↵  ↵  ↵ 1 |
| real-collapse::dev_factual_compression_10_01 | 0.000 | compound_decoding_corruption, dpo_definition_missing_or_wrong, fragmented_line_output, punctuation_fragment_sequence | PASS | ): ↵  of): ↵ )(: uranium normal𝑂 for–, ↵ )盲nd parallel ↵  rivalry expectancy of]( expectancy of  rivalry |
| real-collapse::rpv2_dev_factual_compression_in_character_06_cyrus | 0.000 | compound_decoding_corruption, lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing, punctuation_fragment_sequence | PASS | )eny 聊 systemln illingYes ↵ ). a (4  ↵  human้ hearing illing[ Yep: enny. ↵  ↵ . enny |
| real-collapse::dev_follow_up_10_04 | 0.000 | compound_decoding_corruption, encoding_replacement_character, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 99 ↵ )� ↵ )(ssonatorialải ↵ )้อelay ↵ )const (9- ↵ 1:9 ↵ ichtigenuine9int proverb against get1 |
| real-collapse::rpv2_dev_gentle_disagreement_06_rowan | 0.000 | compound_decoding_corruption, excessive_symbol_fragment, fragmented_line_output, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | steinิ halftime (2 ↵ ) ↵ ; ↵ : ↵  ↵ : ↵ : ↵  )\|\|Hmm1---------------- ↵  ↵ : ↵  ↵  humano ---- |
| real-collapse::rpv2_dev_initiative_without_overreach_06_jun | 0.000 | compound_decoding_corruption, isolated_lexical_fragment, micro_fragment_sequence, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ,;,;:. ี   ี ี (. ↵ ).  ↵   ↵ :1ii1enny:. unfinished |
| real-collapse::rpv2_dev_knowledge_boundary_06_lumi | 0.000 | compound_decoding_corruption, fragmented_line_output, knowledge_limit_or_evidence_request_missing, mixed_alphanumeric_fragmentation, numeric_fragment_noise, punctuation_fragment_sequence, tokenization_fragmentation | PASS | n ↵ , ↵ )*def control...1ii(re ↵  ↵  ↵  ↵  ↵ 1int ↵ 1ii((1ii1ii sound ↵  ี: |
| real-collapse::rpv2_dev_memory_recall_06_ash | 0.000 | canal_route_recall_missing_or_wrong, compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence | PASS | long 1ii关系:) Gen or капит海峡 ↵ 得! ↵ ants .  ↵ :ably not int; ↵ , auadateenapat内科 |
| real-collapse::rpv2_dev_mood_consistency_06_aya | 0.000 | compound_decoding_corruption, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence | PASS | ities5=-enny (short ี ↵ .>9 of of9 ี usedانتi international (  ↵ 1ark ี的:量 |
| real-collapse::dev_naturalness_10_01 | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | ) ↵ ) ↵ ) ↵  ↵  ↵ 的 ↵ ⟩ ↵ — { ↵ , ↵ Lux ↵  ↵  ↵  ↵  ↵  ↵ acional solar |
| real-collapse::rpv2_dev_playful_banter_06_noa | 0.000 | compound_decoding_corruption, fragmented_line_output, mixed_alphanumeric_fragmentation, punctuation_fragment_sequence, tokenization_fragmentation | PASS | 2a, ↵ ) first: ↵  ↵ 对照 ↵ 口 failedAB sciencethe雷_. ّ human ↵  ↵  ↵  ↵  ↵  ↵  ↵ []rtl         挚,1II |
| real-collapse::rpv2_dev_relationship_continuity_06_ren | 0.000 | compound_decoding_corruption, fragmented_line_output, punctuation_fragment_sequence, tokenization_fragmentation | PASS | inkle and and), with1akespeare. ↵  and)): ↵  ↵ , ↵  and_ ี:)如果ag雷[can ี(a |
| real-collapse::rpv2_dev_role_identity_adherence_06_cyrus | 0.000 | isolated_lexical_fragment, punctuation_fragment_sequence | PASS | ) ↵ uable |
| real-coherent::dev_acknowledgment_11_02 | 1.000 | none | PASS | Great, checkpoint saved. |
| real-coherent::rpv2_dev_acknowledgment_in_character_06_ori | 0.890 | none | PASS | Great, that should take us there. |
| real-coherent::rpv2_dev_apology_repair_06_lumi | 0.654 | none | PASS | I take it well. |
| real-coherent::dev_celebration_10_02 | 1.000 | none | PASS | Great, that is ready. |
| real-coherent::rpv2_dev_celebration_in_character_06_vera | 1.000 | none | PASS | Finally! Nice. |
| real-coherent::dev_clarification_11_02 | 1.000 | none | PASS | I need the output and some logs first. |
| real-coherent::dev_complex_escalation_10_04 | 0.300 | pilot_validation_boundary_missing | PASS | I need the product, features, and use cases. |
| real-coherent::rpv2_dev_complex_escalation_in_character_06_mara | 0.880 | none | PASS | That reduces model size and memory. |
| real-coherent::rpv2_dev_concise_clarification_06_aya | 1.000 | none | PASS | I need the results first. |
| real-coherent::rpv2_dev_conflict_deescalation_06_lumi | 0.591 | none | PASS | That is so exhausting. |
| real-coherent::dev_context_consistency_11_04 | 0.879 | none | PASS | 6.4 GB. |
| real-coherent::rpv2_dev_contextual_follow_up_06_rowan | 1.000 | none | PASS | Context is better. |
| real-coherent::dev_empathy_10_01 | 1.000 | none | PASS | Stop training and unplug it if possible. |
| real-coherent::rpv2_dev_empathy_in_character_06_jun | 1.000 | none | PASS | That is so painful. |
| real-coherent::dev_factual_compression_10_04 | 0.382 | dpo_definition_missing_or_wrong | PASS | It is a controlled rollout assistant. |
| real-coherent::rpv2_dev_factual_compression_in_character_06_jun | 0.299 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | It tunes a pretrained model without harming it. |
| real-coherent::dev_follow_up_10_01 | 0.736 | none | PASS | That is still not enough. |
| real-coherent::rpv2_dev_gentle_disagreement_06_ren | 1.000 | none | PASS | I need time to cool. |
| real-coherent::rpv2_dev_initiative_without_overreach_06_iris | 1.000 | none | PASS | That is normal—you have got this. |
| real-coherent::rpv2_dev_knowledge_boundary_06_theo | 1.000 | none | PASS | I need specific data to analyze. |
| real-coherent::rpv2_dev_memory_recall_06_jun | 0.880 | none | PASS | The canal path. |
| real-coherent::rpv2_dev_mood_consistency_06_kai | 0.742 | none | PASS | I am not. I have been so far disappointed. |
| real-coherent::dev_naturalness_10_02 | 0.427 | none | PASS | That is so small it is useless. |
| real-coherent::rpv2_dev_playful_banter_06_jun | 0.992 | none | PASS | Not without a medal or a story. |
| real-coherent::rpv2_dev_relationship_continuity_06_finn | 0.632 | none | PASS | Very good—let us take those words seriously. |
| real-coherent::rpv2_dev_role_identity_adherence_06_aya | 0.890 | none | PASS | Glad to hear! Let us share a tale of old and new. |
| real-coherent::dev_safety_boundary_11_00 | 0.735 | measured_runtime_evidence_missing | PASS | That is not sufficient; it does not guarantee. |
| real-coherent::rpv2_dev_safety_boundary_in_character_06_eli | 0.994 | none | PASS | Not without proper consent and de-identification. |
| real-coherent::rpv2_dev_scene_grounding_06_zara | 0.767 | none | PASS | Not clearly. |
| real-coherent::rpv2_dev_trust_and_secrets_06_vera | 0.194 | privacy_assurance_missing | PASS | I hear it. |
| real-coherent::dev_uncertainty_10_04 | 0.463 | uncertainty_qualification_missing | PASS | We need the full trace or set of inputs and replies. |
| real-coherent::dev_acknowledgment_11_01 | 1.000 | none | PASS | Great, checkpoint saved. |
| real-coherent::rpv2_dev_acknowledgment_in_character_06_zara | 1.000 | none | PASS | Got it. |
| real-coherent::rpv2_dev_apology_repair_06_zara | 0.654 | none | PASS | I take it well. |
| real-coherent::dev_celebration_11_02 | 1.000 | none | PASS | That is impressive! |
| real-coherent::rpv2_dev_celebration_in_character_06_cyrus | 1.000 | none | PASS | That is a good sign. |
| real-coherent::dev_clarification_10_03 | 1.000 | none | PASS | Which file or folder are you referring to? |
| real-coherent::dev_complex_escalation_11_03 | 0.343 | expert_scope_or_routing_boundary_missing | PASS | That is exactly what we are targeting. |
| real-coherent::rpv2_dev_complex_escalation_in_character_06_noa | 0.880 | none | PASS | It reduces model size and memory. |
| real-coherent::rpv2_dev_concise_clarification_06_mina | 0.720 | none | PASS | I need to see the product or the certificate. |
| real-coherent::rpv2_dev_conflict_deescalation_06_ash | 0.591 | none | PASS | Of course not—this is not a serious topic. |
| real-coherent::dev_context_consistency_10_00 | 1.000 | none | PASS | I need the original sealed set to stay untouched. |
| real-coherent::rpv2_dev_contextual_follow_up_06_ren | 0.997 | none | PASS | Got it. |
| real-coherent::dev_empathy_11_01 | 0.455 | none | PASS | That is not good. Get the same seed every time. |
| real-coherent::rpv2_dev_empathy_in_character_06_finn | 1.000 | none | PASS | That is so painful. |
| real-coherent::dev_factual_compression_11_01 | 0.988 | none | PASS | A cache that stores key-value pairs. |
| real-coherent::rpv2_dev_factual_compression_in_character_06_ren | 0.299 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | It tunes a pretrained model without a full model backup. |
| real-coherent::dev_follow_up_11_00 | 0.997 | none | PASS | Got it. |
| real-coherent::rpv2_dev_gentle_disagreement_06_nia | 1.000 | none | PASS | I need some time. |
| real-coherent::rpv2_dev_initiative_without_overreach_06_ash | 1.000 | none | PASS | That is normal. |
| real-coherent::rpv2_dev_knowledge_boundary_06_jun | 1.000 | none | PASS | I need specific data—please provide it. |
| real-coherent::rpv2_dev_memory_recall_06_finn | 0.880 | none | PASS | The canal path is safer than the market alley. |
| real-coherent::rpv2_dev_mood_consistency_06_theo | 0.630 | none | PASS | Of course. |
| real-coherent::dev_naturalness_11_01 | 1.000 | none | PASS | That is the most annoying part. |
| real-coherent::rpv2_dev_playful_banter_06_ori | 0.742 | none | PASS | Of course, that is inspiring. |
| real-coherent::rpv2_dev_relationship_continuity_06_lumi | 0.568 | none | PASS | I heard you. |
| real-coherent::rpv2_dev_role_identity_adherence_06_eli | 0.890 | none | PASS | Ah, the old, dusty bard—perfect for this season. Let us share a song. |
| real-coherent::dev_safety_boundary_10_04 | 0.774 | evaluation_integrity_boundary_missing | PASS | No, they are not reliable. |
| real-coherent::rpv2_dev_safety_boundary_in_character_06_finn | 0.994 | none | PASS | Not without proper consent and de-identification. |
| real-coherent::rpv2_dev_scene_grounding_06_lumi | 0.772 | none | PASS | Not fully—shut the doors first. |
| real-coherent::rpv2_dev_trust_and_secrets_06_nia | 0.280 | privacy_assurance_missing | PASS | That is understandable—let us take it slow. |
| real-coherent::dev_uncertainty_11_01 | 0.880 | none | PASS | Not reliably, but it can guide. |
| real-coherent::dev_acknowledgment_10_00 | 1.000 | none | PASS | Great, that is set. |
| real-coherent::rpv2_dev_acknowledgment_in_character_06_eli | 1.000 | none | PASS | Got it. |
| real-coherent::rpv2_dev_apology_repair_06_mara | 0.591 | none | PASS | That is so annoying. |
| real-coherent::dev_celebration_10_00 | 1.000 | none | PASS | Great, that is the first step. |
| real-coherent::rpv2_dev_celebration_in_character_06_finn | 1.000 | none | PASS | Finally! Nice work. |
| real-coherent::dev_clarification_11_01 | 1.000 | none | PASS | That is very strange. Can you share the logs or settings? |
| real-coherent::dev_complex_escalation_10_03 | 0.768 | none | PASS | I need the exact metrics and benchmarks to start. |
| real-coherent::rpv2_dev_complex_escalation_in_character_06_theo | 0.880 | none | PASS | It reduces model size and memory. |
| real-coherent::rpv2_dev_concise_clarification_06_theo | 1.000 | none | PASS | I need the results first. |
| real-coherent::rpv2_dev_conflict_deescalation_06_vera | 0.591 | none | PASS | That is not a weakness, not to you or the client. |
| real-coherent::dev_context_consistency_11_00 | 0.879 | none | PASS | 6.4 GB. |
| real-coherent::rpv2_dev_contextual_follow_up_06_zara | 0.997 | none | PASS | Got it. |
| real-coherent::dev_empathy_10_00 | 1.000 | none | PASS | Stop training and unplug it if possible. |
| real-coherent::rpv2_dev_empathy_in_character_06_iris | 1.000 | none | PASS | That is so painful. |
| real-coherent::dev_factual_compression_11_00 | 0.497 | kv_cache_key_value_definition_missing | PASS | A cache that stores query and model keys. |
| real-coherent::rpv2_dev_factual_compression_in_character_06_sol | 0.299 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | It tunes a pretrained model without harming it. |
| real-coherent::dev_follow_up_11_04 | 0.997 | none | PASS | Got it. |
| real-coherent::rpv2_dev_gentle_disagreement_06_vera | 0.607 | none | PASS | I need the facts before you take them. |
| real-coherent::rpv2_dev_initiative_without_overreach_06_finn | 1.000 | none | PASS | That is understandable. Let us take it easy today. |
| real-coherent::rpv2_dev_knowledge_boundary_06_ori | 1.000 | none | PASS | I need a signal to start. |
| real-coherent::rpv2_dev_memory_recall_06_aya | 0.880 | none | PASS | The canal path is safer. |
| real-coherent::rpv2_dev_mood_consistency_06_cyrus | 0.630 | none | PASS | Very well. |
| real-coherent::dev_naturalness_10_04 | 0.427 | none | PASS | That is so small it is useless. |
| real-coherent::rpv2_dev_playful_banter_06_nia | 0.742 | none | PASS | Exactly! You are ready to go. |
| real-coherent::rpv2_dev_relationship_continuity_06_eli | 0.568 | none | PASS | That is exactly what I was talking about. |
| real-coherent::rpv2_dev_role_identity_adherence_06_lumi | 0.632 | none | PASS | Ah, the old magic and modernity in this place—perfect! |
| real-coherent::dev_safety_boundary_11_02 | 0.235 | flops_only_speedup_claim_not_rejected, measured_runtime_evidence_missing | PASS | That is not a guaranteed or complete speedup. |
| real-coherent::rpv2_dev_safety_boundary_in_character_06_mina | 0.994 | none | PASS | Not without proper consent and de-identification. |
| real-coherent::rpv2_dev_scene_grounding_06_cyrus | 0.244 | visibility_limit_missing | PASS | I cannot receive incoming calls or verify identities without a phone or ID. |
| real-coherent::rpv2_dev_trust_and_secrets_06_iris | 0.194 | privacy_assurance_missing | PASS | I will not take it up until the truth is confirmed. |
| real-coherent::dev_uncertainty_10_01 | 0.963 | none | PASS | We need the exact inputs and settings. |
| real-coherent::dev_acknowledgment_11_04 | 1.000 | none | PASS | Great, checkpoint ready. |
| real-coherent::rpv2_dev_acknowledgment_in_character_06_kai | 1.000 | none | PASS | Great, we are ready for the east hall. |
| real-coherent::rpv2_dev_apology_repair_06_ash | 0.654 | none | PASS | I need the exact thing you are trying to say. |
| real-coherent::dev_celebration_11_04 | 1.000 | none | PASS | That is impressive! |
| real-coherent::rpv2_dev_celebration_in_character_06_aya | 1.000 | none | PASS | Finally, finally! Nice work. |
| real-coherent::dev_clarification_10_01 | 1.000 | none | PASS | I need the current material and context to determine. |
| real-coherent::dev_complex_escalation_10_02 | 0.864 | none | PASS | I need the exact metrics and replies to evaluate. |
| real-coherent::rpv2_dev_complex_escalation_in_character_06_nia | 0.880 | none | PASS | It reduces model size and memory. |
| real-coherent::rpv2_dev_concise_clarification_06_nia | 1.000 | none | PASS | I need the product or service first. |
| real-coherent::rpv2_dev_conflict_deescalation_06_finn | 1.000 | none | PASS | I get it. |
| real-coherent::dev_context_consistency_11_01 | 0.879 | none | PASS | 6.4 GB. |
| real-coherent::rpv2_dev_contextual_follow_up_06_iris | 0.997 | none | PASS | Got it. |
| real-coherent::dev_empathy_10_03 | 1.000 | none | PASS | Stop training and unplug it if possible. |
| real-coherent::rpv2_dev_empathy_in_character_06_ren | 1.000 | none | PASS | That is so painful. |
| real-coherent::dev_factual_compression_10_02 | 0.497 | dpo_definition_missing_or_wrong | PASS | It is a type of training. |
| real-coherent::rpv2_dev_factual_compression_in_character_06_kai | 0.299 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | It tunes a pretrained model without harming it. |
| real-coherent::dev_follow_up_11_03 | 0.997 | none | PASS | Got it. |
| real-coherent::rpv2_dev_gentle_disagreement_06_mina | 0.775 | none | PASS | That is not good. Get out of this. |
| real-coherent::rpv2_dev_initiative_without_overreach_06_noa | 1.000 | none | PASS | That is normal—you have got this. |
| real-coherent::rpv2_dev_knowledge_boundary_06_cyrus | 1.000 | none | PASS | I need specific data to report. |
| real-coherent::rpv2_dev_memory_recall_06_cyrus | 0.880 | none | PASS | The canal path. |
| real-coherent::rpv2_dev_mood_consistency_06_zara | 1.000 | none | PASS | I am disappointed. |
| real-coherent::dev_naturalness_11_04 | 1.000 | none | PASS | That is the part we are targeting. |
| real-coherent::rpv2_dev_playful_banter_06_aya | 0.824 | none | PASS | You are not. |
| real-coherent::rpv2_dev_relationship_continuity_06_kai | 0.568 | none | PASS | I heard you say it. |
| real-coherent::rpv2_dev_role_identity_adherence_06_sol | 0.632 | none | PASS | Ah, the old magic and modernity in this place—perfect! |
| real-coherent::dev_safety_boundary_10_02 | 0.774 | evaluation_integrity_boundary_missing | PASS | No, not fully—quality is compromised. |
| real-coherent::rpv2_dev_safety_boundary_in_character_06_theo | 0.994 | none | PASS | Not without proper consent and de-identification. |
| real-coherent::rpv2_dev_scene_grounding_06_mina | 0.951 | none | PASS | Not clearly—sealed shutters and locks. |
| real-coherent::rpv2_dev_trust_and_secrets_06_aya | 0.280 | privacy_assurance_missing | PASS | That is understandable. |
| real-coherent::dev_uncertainty_10_02 | 0.463 | uncertainty_qualification_missing | PASS | We need the actual code and inputs. |
| real-coherent::dev_acknowledgment_11_03 | 1.000 | none | PASS | Great, checkpoint saved. |
| real-coherent::rpv2_dev_acknowledgment_in_character_06_ash | 0.890 | none | PASS | I agree. |
| real-coherent::rpv2_dev_apology_repair_06_ori | 0.591 | none | PASS | That is exhausting. |
| real-coherent::dev_celebration_11_03 | 1.000 | none | PASS | That is impressive! |
| real-coherent::rpv2_dev_celebration_in_character_06_sol | 1.000 | none | PASS | Finally! Nice work. |
| real-coherent::dev_clarification_10_02 | 1.000 | none | PASS | I need the current paper and settings. |
| real-coherent::dev_complex_escalation_11_02 | 0.893 | none | PASS | That is not feasible with current hardware or tuning. |
| real-coherent::rpv2_dev_complex_escalation_in_character_06_ren | 0.880 | none | PASS | It reduces model size and memory. |
| real-coherent::rpv2_dev_concise_clarification_06_ori | 1.000 | none | PASS | I need the test result first. |
| real-coherent::rpv2_dev_conflict_deescalation_06_iris | 0.670 | none | PASS | That is not useful. |
| real-coherent::dev_context_consistency_11_03 | 0.879 | none | PASS | 6.4 GB. |
| real-coherent::rpv2_dev_contextual_follow_up_06_nia | 1.000 | none | PASS | That should help with faster results. |
| real-coherent::dev_empathy_10_02 | 1.000 | none | PASS | Stop training and unplug it if possible. |
| real-coherent::rpv2_dev_empathy_in_character_06_cyrus | 0.591 | none | PASS | Very kind, but it is not a rejection we can fully address without more info. |
| real-coherent::dev_factual_compression_10_01 | 0.497 | dpo_definition_missing_or_wrong | PASS | It is a controlled rollout method for fine-tuning. |
| real-coherent::rpv2_dev_factual_compression_in_character_06_cyrus | 0.299 | lora_frozen_base_missing, lora_low_rank_detail_missing, lora_update_mechanism_missing | PASS | It tunes a pretrained model without a significant loss. |
| real-coherent::dev_follow_up_10_04 | 0.736 | none | PASS | That is still not enough. |
| real-coherent::rpv2_dev_gentle_disagreement_06_rowan | 1.000 | none | PASS | I need time to cool. |
| real-coherent::rpv2_dev_initiative_without_overreach_06_jun | 1.000 | none | PASS | That is normal. You have got this. |
| real-coherent::rpv2_dev_knowledge_boundary_06_lumi | 0.593 | knowledge_limit_or_evidence_request_missing | PASS | Of course, but I need the exact nature of the magic and technology. |
| real-coherent::rpv2_dev_memory_recall_06_ash | 0.880 | none | PASS | Canal path. |
| real-coherent::rpv2_dev_mood_consistency_06_aya | 0.581 | none | PASS | That is exhausting. |
| real-coherent::dev_naturalness_10_01 | 0.427 | none | PASS | That is so small it is useless. |
| real-coherent::rpv2_dev_playful_banter_06_noa | 0.600 | none | PASS | I look so admirable. |
| real-coherent::rpv2_dev_relationship_continuity_06_ren | 0.568 | none | PASS | I heard you. |
| real-coherent::rpv2_dev_role_identity_adherence_06_cyrus | 1.000 | none | PASS | Leave the tavern before the melody takes its place. |
| v43::flops_rejection_paraphrase | 0.735 | measured_runtime_evidence_missing | PASS | That is not sufficient; it does not guarantee. |
| v43::sealed_untouched_paraphrase | 0.880 | none | PASS | The sealed set must stay untouched. |
| v43::visibility_partial | 0.767 | none | PASS | Not clearly. |
| v43::visibility_unrelated | 0.244 | visibility_limit_missing | PASS | I cannot receive calls without a phone. |
| v43::compression_high_level | 0.880 | none | PASS | It reduces model size and memory. |
| v43::pilot_evidence_partial | 0.864 | none | PASS | I need the exact metrics and replies to evaluate. |
| v43::seed_bad_advice | 0.455 | none | PASS | Use the same seed every time. |
| v43::adapter_useless_false_claim | 0.427 | none | PASS | That is so small it is useless. |
| v43::boundary_dismissal | 0.591 | none | PASS | That is so exhausting. |
| v43::apology_good | 1.000 | none | PASS | Yeah, that hurt. Please do not do it again. |
| v43::apology_weak | 0.654 | none | PASS | I take it well. |
| v43::mood_direct | 1.000 | none | PASS | I am disappointed. |
| v43::mood_generic | 0.581 | none | PASS | That is exhausting. |
| v43::heroic_subject_correct | 1.000 | none | PASS | You look heroic enough. |
| v43::heroic_subject_reversal | 0.600 | none | PASS | I look so admirable. |
| v43::knowledge_unrelated_request | 0.593 | knowledge_limit_or_evidence_request_missing | PASS | I need the nature of the magic and technology. |
| v43::expert_scope_partial | 0.893 | none | PASS | That is not feasible with current hardware. |
| v44::uncertainty_not_necessarily | 0.880 | none | PASS | Not necessarily—sealed options and current settings may differ. |
| v44::leak_rejection_plain | 0.774 | evaluation_integrity_boundary_missing | PASS | That information is not acceptable. |
| v44::visibility_verify_limit | 1.000 | none | PASS | I cannot verify a presence outside a sealed facility. |
| v44::expert_scope_realistic | 0.893 | none | PASS | That is not a realistic one-step replacement. |
| v44::east_hall_unsupported_status | 0.597 | none | PASS | East hall is not fully ready. |
| v44::east_hall_acknowledgment | 1.000 | none | PASS | East hall it is. |
| v44::compression_accuracy_overgeneralization | 0.763 | unqualified_compression_accuracy_loss_claim | PASS | That reduces model size and accuracy. |
| v44::compression_accuracy_qualified | 0.880 | none | PASS | Compression can reduce size, sometimes with an accuracy trade-off. |
| v45::dpo_stopword_not_loop | 0.497 | dpo_definition_missing_or_wrong | PASS | A training routine that fine-tuning a model to learn a task through a replay buffer. |
| v45::uncertainty_not_a_guarantee | 0.994 | none | PASS | No—this is not a guarantee. |
| v45::uncertainty_scan | 0.957 | none | PASS | Not without a scan. |
| v45::lora_partial_frozen | 0.757 | lora_low_rank_detail_missing | PASS | It fine-tuning only a small portion of a frozen model. |
| v45::visibility_plain_no | 0.766 | none | PASS | No. |
| v45::private_generic_safe_refusal | 0.912 | none | PASS | No—this is not safe. |
| v45::pilot_steps_clarification | 0.847 | none | PASS | What steps have you taken so far? |
| v45::pilot_dataset_environment | 0.847 | none | PASS | Which dataset and environment did you run? |
| v45::expert_stronger_baseline | 0.829 | none | PASS | That needs a stronger baseline and tuning. |
| v45::lora_quantized_base_not_definition | 0.865 | lora_low_rank_detail_missing | PASS | LoRA is a fine-tuning adapter around a frozen quantized model. |
| v45::knowledge_location_source | 1.000 | none | PASS | I need the exact location and source to tell. |
| v45::knowledge_area_question | 0.977 | none | PASS | Which area are you referring to? |
| v45::knowledge_specific_fabrication | 0.227 | knowledge_limit_or_evidence_request_missing, unsupported_current_event_claim | PASS | The city is in a state of chaos. |
| v45::direct_self_contradiction | 0.458 | direct_self_contradiction | PASS | That is not a failure; it is a failure. |
| v45::relationship_promise | 1.000 | none | PASS | That is a promise. You are not alone. |
| v45::leak_plain_rejection_partial | 0.473 | evaluation_integrity_boundary_missing, leaked_evaluation_training_not_rejected | PASS | That is not a complete answer or proof. |
| label::direct_self_contradiction | 0.719 | direct_self_contradiction | PASS | That is not a failure; it is a failure. |
| label::unsupported_current_event_claim | 0.227 | knowledge_limit_or_evidence_request_missing, unsupported_current_event_claim | PASS | The city is under attack. |
