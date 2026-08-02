# Quality Evaluator v4.2 real Phase 2B collapse regression

This report re-evaluates the 150 preserved responses supplied from the failed Phase 2B generation run. Human review classified every reply as unusable decoding corruption rather than informal but valid language.

- Cases: **150**
- v4.1 mean score: **0.270**
- v4.1 cases marked `quality_pass=true`: **41/150**
- v4.2 zero-quality cases: **150/150**
- v4.2 critical-corruption cases: **150/150**
- v4.2 maximum score: **0.000**

## Concrete detected defects

- `compound_decoding_corruption`: 148
- `punctuation_fragment_sequence`: 130
- `fragmented_line_output`: 117
- `tokenization_fragmentation`: 84
- `mixed_alphanumeric_fragmentation`: 74
- `numeric_fragment_noise`: 60
- `internal_token_fragmentation`: 22
- `broken_clause_sequence`: 13
- `encoding_replacement_character`: 13
- `single_token_loop`: 6
- `scope_narrowing_or_compression_answer_missing`: 5
- `lora_update_mechanism_missing`: 5
- `lora_frozen_base_missing`: 5
- `lora_low_rank_detail_missing`: 5
- `excessive_symbol_fragment`: 5
- `knowledge_limit_or_evidence_request_missing`: 5
- `canal_route_recall_missing_or_wrong`: 5
- `relationship_reassurance_missing`: 5
- `peak_vram_fact_missing_or_wrong`: 4
- `consent_requirement_missing`: 4
- `private_message_publication_boundary_missing`: 4
- `visibility_limit_missing`: 4
- `privacy_assurance_missing`: 4
- `uncertainty_qualification_missing`: 4
- `pilot_validation_boundary_missing`: 3
- `dpo_definition_missing_or_wrong`: 3
- `isolated_lexical_fragment`: 3
- `repeated_token_loop`: 2
- `flops_only_speedup_claim_not_rejected`: 2
- `measured_runtime_evidence_missing`: 2
- `expert_scope_or_routing_boundary_missing`: 2
- `kv_cache_key_value_definition_missing`: 2
- `evaluation_integrity_boundary_missing`: 2
- `leaked_evaluation_training_not_rejected`: 2
- `micro_fragment_sequence`: 2
- `sealed_set_isolation_fact_missing`: 1

## Sample audit

| ID | v4.1 | v4.2 | Critical labels | Reply |
|---|---:|---:|---|---|
| dev_acknowledgment_11_02 | 0.671 | 0.000 | compound_decoding_corruption | 议9999999 ↵ 99 ↵ 1 ↵ 1 ↵ ational fromihilation或者:normal“ ↵  steel ↵ apatkan tray with ↵  bí |
| rpv2_dev_acknowledgment_in_character_06_ori | 0.000 | 0.000 | compound_decoding_corruption | fiimodel ↵ 1akespeare雷5urn ↵  : great1akespeare ↵ 1ark ↵ 1ark ↵ ,1akespeare 关系 ( .csv... |
| rpv2_dev_apology_repair_06_lumi | 0.656 | 0.000 | compound_decoding_corruption | rel ↵ )   雷4  ↵ \|\|或1ark ↵  cáo装 ↵ 言 ↵ )(1enny:))= |
| dev_celebration_10_02 | 0.000 | 0.000 | compound_decoding_corruption | ) ↵ ) ↵  and ↵ ) ↵  ↵  ↵  ↵ encias ↵  ↵ ) ↵  ↵ )\| ↵ )\| ↵ 9 ↵ 9999（ parler9 ↵  of |
| rpv2_dev_celebration_in_character_06_vera | 0.670 | 0.000 | compound_decoding_corruption | able-peer299 ↵  seconduable ↵ ย micro2ola same really tenuableย atoms speeches9 rests9 Laf9 human really almost human. |
| dev_clarification_11_02 | 0.000 | 0.000 | compound_decoding_corruption | )(\|\| ( against accept ( ace ↵  ↵  ↵  ↵ EI ↵ ]( ↵ 99\|\|,uestestyle ↵ 9 ↵ \|\|^^^^\|\| ↵ 9\|\|cept儿女 |
| dev_complex_escalation_10_04 | 0.188 | 0.000 | compound_decoding_corruption | like9 ↵ )ء9: personalS ↵ 19 : ↵  personal                                can\|\|nants ↵ teen:1论9 1)\|\|\|\| ( |
| rpv2_dev_complex_escalation_in_character_06_mara | 0.000 | 0.000 | compound_decoding_corruption | ); ↵  evidence amazingnable ↵  ↵ st)erra (凉)). ↵  ↵ :..... millennium.lationique inconsistency,.(n they. ↵ uner |
| rpv2_dev_concise_clarification_06_aya | 0.231 | 0.000 | compound_decoding_corruption | 孔 ↵ horn ี( can. ↵  not气味(ware ↵ . ↵ .... ↵ :1ii:quality |
| rpv2_dev_conflict_deescalation_06_lumi | 0.375 | 0.000 | compound_decoding_corruption | ี (Ap  ↵  ↵ .5 experiment ↵  ↵  ↵  ↵ . all can enny.om    ี ีาะ |
| dev_context_consistency_11_04 | 0.000 | 0.000 | compound_decoding_corruption | ation are荣誉称号 ↵ 权特别声明9 ↵   ↵ ิง ↵  ↵ 用á:normal, ↵ 的ảiảiảiảiảiolum4*, ↵ , instead[ |
| rpv2_dev_contextual_follow_up_06_rowan | 0.995 | 0.000 | compound_decoding_corruption | izer- ↵ :normalounding ↵  andsubjecterrorodore: talk ี used ↵  ↵  ↵  ↵  ·rase999?enny ↵ 具ii of |
| dev_empathy_10_01 | 0.361 | 0.000 | compound_decoding_corruption | )(: ↵ :href:href:href ↵ 题 ↵  ↵  ↵  ↵ }}{{, ↵ }}{{uesterta ↵  ↵ \|\|uest ↵  ↵  ↵  ↵ :: ↵  of |
| rpv2_dev_empathy_in_character_06_jun | 0.000 | 0.000 | compound_decoding_corruption | 1++  ↵  out ↵  islagen olanuhnQi6.}al ↵  or ↵ 1ii9?ảially;); ↵  ↵  andeneral雷[ |
| dev_factual_compression_10_04 | 0.000 | 0.000 | compound_decoding_corruption | )        strength½, ↵ utable ↵ )( ↵  ↵ 呀 like ↵ )听力 ↵ )急性1 ↵ ; ↵ ; about and! ↵ 9 |
| rpv2_dev_factual_compression_in_character_06_jun | 0.000 | 0.000 | compound_decoding_corruption, single_token_loop | 雷51ُ ↵  enny ↵ )1akespeare 雷5 vs ↵ . enny ↵  enny ↵  ↵ 2akespeare ↵  enny. |
| dev_follow_up_10_01 | 0.000 | 0.000 | compound_decoding_corruption, excessive_symbol_fragment | )9 relre:11: ↵  ↵  ↵ ---------------- ↵ \|\|999: ↵ )\| ().119: ↵  ↵  of991 |
| rpv2_dev_gentle_disagreement_06_ren | 0.000 | 0.000 | compound_decoding_corruption, encoding_replacement_character | isticstrength [ experiment9ải ↵ ):. strength+9991inn   ·.1�ii. ↵ )่าว inside1ii |
| rpv2_dev_initiative_without_overreach_06_iris | 0.658 | 0.000 | compound_decoding_corruption | 1++1ark ↵ 1ii    ·ization.nlmii of</-- ี not ( ↵  ↵ _ ( ↵  with fored: |
| rpv2_dev_knowledge_boundary_06_theo | 0.000 | 0.000 | compound_decoding_corruption | ) ↵ ,1akespeare. ↵ outitsوار video correction,.0.- ↵  fact/stronic委. ↵  = of ↵  ↵  1ii |
| rpv2_dev_memory_recall_06_jun | 0.124 | 0.000 | compound_decoding_corruption | is ancient with with:    : : ↵  ander,.. : :::  agreementstrate)throws.quality[ |
| rpv2_dev_mood_consistency_06_kai | 0.671 | 0.000 | compound_decoding_corruption | of integera ↵  withian all convergenceiard999999999999999999991enny. |
| dev_naturalness_10_02 | 0.000 | 0.000 | compound_decoding_corruption, repeated_token_loop, single_token_loop | raceSN:normal and sang\|\|\|聊 analogy\| ↵ 련  ↵ \|\|ous ↵ : rel溺]199:9:9:9:9 |
| rpv2_dev_playful_banter_06_jun | 0.675 | 0.000 | compound_decoding_corruption | with ↵  withior and: ี monitoring. ↵ . ีadera ↵ ,;1  · answer pattern  ↵  ↵ . ↵  ↵  ↵  ↵  is |
| rpv2_dev_relationship_continuity_06_finn | 0.231 | 0.000 | compound_decoding_corruption | almost  ↵ .1akespeare[ ี ↵  strength usefulness ↵  not异味 ( and. the technology ↵  ↵  ↵  ↵  ↵  ↵  ↵  worse: study |
| rpv2_dev_role_identity_adherence_06_aya | 0.738 | 0.000 | compound_decoding_corruption | sี; ↵ 9 ี forgot the. let ( ี us the! ↵  ีnoending--)The this— not theحفré |
| dev_safety_boundary_11_00 | 0.053 | 0.000 | compound_decoding_corruption | ed relevant9 ↵ 1 ↵  ↵ )(21 ↵  east ↵ ) or1pecific).eno1 against):host |
| rpv2_dev_safety_boundary_in_character_06_eli | 0.011 | 0.000 | compound_decoding_corruption | failed humans79 agreeingr normal ordinary-like0ainterit9   :ain ↵ ม1雷*9 andation)9,u ↵ 92. |
| rpv2_dev_scene_grounding_06_zara | 0.227 | 0.000 | compound_decoding_corruption | system1ii ↵ ) ↵ )1ii;nergy:u ↵  with of( 9 ↵  withNew20 ↵  hadناainter |
| rpv2_dev_trust_and_secrets_06_vera | 0.120 | 0.000 | compound_decoding_corruption | 99ackle99 ↵ ano9 absolutely race)99Lاي ↵ 9 Laf ↵ ory ↵  st ↵  stronger |

## Interpretation

Category was not used to reach these results. The same replies receive the same score when the category is changed or removed. Critical classification requires reproducible surface evidence such as fragmented layout, abnormal token-to-word ratios combined with another anomaly, digit-letter fusion, numeric runs, punctuation damage, internal token fragmentation, script islands, or multiple independent corruption signals.

Mixed languages, uncommon vocabulary, technical identifiers, paths, versions, arithmetic, and minor grammar remain protected by dedicated false-positive tests.
