# Phase 2 Failure and Qualitative-Case Analysis

**Status: complete for frozen full-TEST sequence-level buckets; teacher-forced minority-gold cases remain pending.**

Cases are selected before prompt inspection by SHA256 ranking within machine-defined buckets, with domain diversity where possible. The JSON artifact preserves the selection hashes, complete prompts, outputs, expert proposals at the trigger step, vote structure, confidence diagnostics, tie/EOS state, and scoring labels.

## Bucket inventory

| Bucket | Eligible examples | Selected IDs |
|---|---:|---|
| `abstention_expert_rescues_hard` | 272 | v22_creative_insufficient_information_0095, v22_communication_insufficient_information_0093, v22_finance_conflicting_evidence_0099 |
| `abstention_success` | 257 | v22_operations_conflicting_evidence_0095, v22_agents_conflicting_evidence_0090, v22_communication_insufficient_information_0091 |
| `correct_track_a_broken_by_committee` | 220 | v22_coding_dependency_order_0091, v22_research_filter_then_rank_0096, v22_agents_compare_metric_0096 |
| `failure_to_abstain` | 283 | v22_tools_insufficient_information_0098, v22_consumer_insufficient_information_0097, v22_coaching_insufficient_information_0093 |
| `false_abstention` | 890 | v22_systems_aggregate_delta_0090, v22_finance_rule_exception_0098, v22_media_dependency_order_0091 |
| `hard_beats_probability` | 226 | v22_finance_constraint_satisfaction_0099, v22_coding_rule_exception_0098, v22_tools_compare_metric_0099 |
| `hard_harms_track_a` | 220 | v22_media_constraint_preserving_transform_0093, v22_systems_state_transition_0091, v22_business_state_transition_0095 |
| `hard_rescues_soft_fusion` | 226 | v22_math_constraint_satisfaction_0099, v22_science_conflicting_evidence_0091, v22_health_conflicting_evidence_0099 |
| `hard_rescues_track_a` | 328 | v22_general_insufficient_information_0098, v22_science_conflicting_evidence_0092, v22_devops_conflicting_evidence_0098 |
| `hard_trajectory_rescue_proxy` | 226 | v22_coding_conflicting_evidence_0094, v22_math_constraint_preserving_transform_0096, v22_data_analysis_filter_then_rank_0092 |
| `high_agreement_wrong` | 1195 | v22_consumer_diagnostic_priority_0098, v22_media_evidence_synthesis_0099, v22_coaching_evidence_synthesis_0093 |
| `probability_beats_hard` | 98 | v22_finance_insufficient_information_0092, v22_media_compare_metric_0094, v22_devops_conflicting_evidence_0094 |
| `soft_fusion_rescues_hard` | 98 | v22_media_conflicting_evidence_0090, v22_consumer_insufficient_information_0097, v22_operations_insufficient_information_0097 |
| `soft_trajectory_rescue_proxy` | 98 | v22_devops_conflicting_evidence_0090, v22_travel_conflicting_evidence_0093, v22_consumer_insufficient_information_0098 |
| `tie_failure` | 1298 | v22_creative_dependency_order_0093, v22_writing_filter_then_rank_0095, v22_media_aggregate_delta_0095 |
| `tie_resolution_failure` | 1298 | v22_agents_filter_then_rank_0094, v22_communication_filter_then_rank_0094, v22_general_capacity_fit_0093 |
| `tie_success` | 230 | v22_personal_conflicting_evidence_0095, v22_devops_conflicting_evidence_0093, v22_communication_insufficient_information_0091 |
| `vote_3_2_2` | 253 | v22_coding_diagnostic_priority_0097, v22_communication_constraint_preserving_transform_0090, v22_systems_filter_then_rank_0090 |
| `vote_4_3` | 263 | v22_consumer_state_transition_0093, v22_science_diagnostic_priority_0093, v22_travel_diagnostic_priority_0098 |

## Failure taxonomy counts

| Failure pattern | Count | Interpretation boundary |
|---|---:|---|
| `high_agreement_wrong` | 1195 | High mean vote agreement with wrong exact output; not necessarily high probability confidence. |
| `tie_resolution_failure` | 1298 | At least one hard plurality tie occurs in a wrong sequence; the tie is not proven causal. |
| `false_abstention` | 890 | Hard emits an abstention on an answerable item. |
| `failure_to_abstain` | 283 | Hard does not abstain on an abstention-class item. |
| `correct_track_a_broken_by_committee` | 220 | Track A is correct and Hard K=7 is wrong. |
| `soft_fusion_rescues_hard` | 98 | Probability mean is correct and Hard K=7 is wrong. |
| `hard_rescues_soft_fusion` | 226 | Hard K=7 is correct and probability mean is wrong. |
| `abstention_expert_rescues_hard` | 272 | Abstention K=1 is correct and Hard K=7 is wrong. |

## Required pending qualitative evidence

- Minority expert proposes the gold next token but is outvoted: requires full teacher-forced DEV cache.
- Semantically correct but exact-match wrong: requires the preselected secondary semantic audit.
- Causal trajectory correction: requires same-split teacher-forced/free-running DEV comparison.
- Premature EOS causality: requires counterfactual EOS-policy or gold-prefix evidence; short wrong output alone is insufficient.

## Provenance

- Raw result SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`
- TEST dataset SHA256: `768107b579126063d4dd304b80ffc88a57e1bc346a7adfe4b69771e1b499a8f4`
- Tokenizer SHA256: `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`
- Script SHA256: `fa22ef31e2fc682db99fbbc922990e4001548021d25a192cb9aa023caa8e6398`
- Full selected case records: `results/phase2_mechanism/qualitative_selection.json`
