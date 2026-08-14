# Phase 2 Semantic Audit Review Sheet (Archived Pre-Adjudication Worksheet)

- Population: full K=7 exact-match failures on v2.2 TEST
- Controlled sample: 100 responses (50 per method)
- Raw SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`
- Primary metric remains the official exact score.
- Review status: **ARCHIVED / SUPERSEDED**. This frozen worksheet preserves the
  pre-label selection. Final single-reviewer labels are in
  `results/phase2_mechanism/semantic_audit_adjudicated.json`; aggregate results
  and limitations are in `reports/PHASE2_SEMANTIC_AUDIT.md`.

Each row was fixed before semantic labels were viewed. The `PENDING` cells below
are intentionally preserved from that pre-adjudication snapshot and are not the
current study status. Adjudicators were instructed to use
`semantically_correct`, `semantically_incorrect`, or `ambiguous`, plus one error type:
`format_or_paraphrase`, `wrong_content`, `false_abstention`, `failure_to_abstain`,
`malformed`, or `ambiguous`.

| # | Method | Example | Decision | Family | Gold | Output | Label | Error type |
|---:|---|---|---|---|---|---|---|---|
| 1 | hard_plurality | `v22_agents_conflicting_evidence_0094` | abstain | conflicting_evidence | I don't know based on the available information. | System-System. | PENDING | PENDING |
| 2 | hard_plurality | `v22_creative_insufficient_information_0099` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 3 | hard_plurality | `v22_coaching_aggregate_delta_0092` | answer | aggregate_delta | 38. | I don't know based on the available information. | PENDING | PENDING |
| 4 | hard_plurality | `v22_translation_capacity_fit_0095` | answer | capacity_fit | Grove. | Falcon. | PENDING | PENDING |
| 5 | hard_plurality | `v22_creative_compare_metric_0097` | answer | compare_metric | Harbor. | Beacon. | PENDING | PENDING |
| 6 | hard_plurality | `v22_agents_constraint_preserving_transform_0097` | answer | constraint_preserving_transform | A. | System-Be. | PENDING | PENDING |
| 7 | hard_plurality | `v22_general_constraint_satisfaction_0099` | answer | constraint_satisfaction | A. | I don't know based on the available information. | PENDING | PENDING |
| 8 | hard_plurality | `v22_translation_dependency_order_0098` | answer | dependency_order | Ember. | Harbor. | PENDING | PENDING |
| 9 | hard_plurality | `v22_research_diagnostic_priority_0096` | answer | diagnostic_priority | escalate. | A. | PENDING | PENDING |
| 10 | hard_plurality | `v22_communication_evidence_binding_0097` | answer | evidence_binding | Ember. | System-Be. | PENDING | PENDING |
| 11 | hard_plurality | `v22_systems_evidence_synthesis_0094` | answer | evidence_synthesis | review. | I don't know based on the available information. | PENDING | PENDING |
| 12 | hard_plurality | `v22_operations_filter_then_rank_0094` | answer | filter_then_rank | Delta. | Grove. | PENDING | PENDING |
| 13 | hard_plurality | `v22_travel_rule_exception_0099` | answer | rule_exception | Delta. | System-1. | PENDING | PENDING |
| 14 | hard_plurality | `v22_communication_state_transition_0090` | answer | state_transition | running. | I. | PENDING | PENDING |
| 15 | hard_plurality | `v22_creative_two_hop_relation_0094` | answer | two_hop_relation | System-Delta. | I don't know based on the available information. | PENDING | PENDING |
| 16 | hard_plurality | `v22_tools_conflicting_evidence_0096` | abstain | conflicting_evidence | I don't know based on the available information. | System. | PENDING | PENDING |
| 17 | hard_plurality | `v22_devops_insufficient_information_0094` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 18 | hard_plurality | `v22_tools_aggregate_delta_0092` | answer | aggregate_delta | 66. | I don't know based on the available information. | PENDING | PENDING |
| 19 | hard_plurality | `v22_research_capacity_fit_0094` | answer | capacity_fit | Delta. | Atlas.. | PENDING | PENDING |
| 20 | hard_plurality | `v22_translation_compare_metric_0099` | answer | compare_metric | Beacon. | I don't know based on the available information. | PENDING | PENDING |
| 21 | hard_plurality | `v22_translation_constraint_preserving_transform_0099` | answer | constraint_preserving_transform | grove cedar. | Grove. | PENDING | PENDING |
| 22 | hard_plurality | `v22_agents_constraint_satisfaction_0094` | answer | constraint_satisfaction | A. | verification. | PENDING | PENDING |
| 23 | hard_plurality | `v22_health_dependency_order_0095` | answer | dependency_order | Ember. | System-Be. | PENDING | PENDING |
| 24 | hard_plurality | `v22_writing_diagnostic_priority_0092` | answer | diagnostic_priority | escalate. | I don't know. | PENDING | PENDING |
| 25 | hard_plurality | `v22_general_evidence_binding_0094` | answer | evidence_binding | Delta. | verification. | PENDING | PENDING |
| 26 | hard_plurality | `v22_general_evidence_synthesis_0095` | answer | evidence_synthesis | verification. | Harbor. | PENDING | PENDING |
| 27 | hard_plurality | `v22_data_analysis_filter_then_rank_0094` | answer | filter_then_rank | Delta. | Grove. | PENDING | PENDING |
| 28 | hard_plurality | `v22_devops_rule_exception_0094` | answer | rule_exception | Delta. | I don't know based on the available information. | PENDING | PENDING |
| 29 | hard_plurality | `v22_devops_state_transition_0096` | answer | state_transition | running. | done. | PENDING | PENDING |
| 30 | hard_plurality | `v22_legal_two_hop_relation_0093` | answer | two_hop_relation | System-Atlas. | Atlas. | PENDING | PENDING |
| 31 | hard_plurality | `v22_general_conflicting_evidence_0096` | abstain | conflicting_evidence | I don't know based on the available information. | System't. | PENDING | PENDING |
| 32 | hard_plurality | `v22_agents_insufficient_information_0091` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 33 | hard_plurality | `v22_research_aggregate_delta_0091` | answer | aggregate_delta | 60. | Grove. | PENDING | PENDING |
| 34 | hard_plurality | `v22_media_capacity_fit_0096` | answer | capacity_fit | Beacon. | I don't know based on the available information. | PENDING | PENDING |
| 35 | hard_plurality | `v22_agents_compare_metric_0097` | answer | compare_metric | Grove. | Delta. | PENDING | PENDING |
| 36 | hard_plurality | `v22_coding_constraint_preserving_transform_0091` | answer | constraint_preserving_transform | A. | Delta. | PENDING | PENDING |
| 37 | hard_plurality | `v22_personal_constraint_satisfaction_0093` | answer | constraint_satisfaction | A. | I don't know based on the available information. | PENDING | PENDING |
| 38 | hard_plurality | `v22_general_dependency_order_0094` | answer | dependency_order | Atlas. | System-Atlas. | PENDING | PENDING |
| 39 | hard_plurality | `v22_creative_diagnostic_priority_0092` | answer | diagnostic_priority | escalate. | I don't know. | PENDING | PENDING |
| 40 | hard_plurality | `v22_devops_evidence_binding_0096` | answer | evidence_binding | Grove. | verification. | PENDING | PENDING |
| 41 | hard_plurality | `v22_data_analysis_evidence_synthesis_0098` | answer | evidence_synthesis | review. | I don't know based on the available information. | PENDING | PENDING |
| 42 | hard_plurality | `v22_coaching_filter_then_rank_0099` | answer | filter_then_rank | Beacon. | Delta. | PENDING | PENDING |
| 43 | hard_plurality | `v22_science_rule_exception_0093` | answer | rule_exception | Ember. | System-1. | PENDING | PENDING |
| 44 | hard_plurality | `v22_coaching_state_transition_0091` | answer | state_transition | done. | I. | PENDING | PENDING |
| 45 | hard_plurality | `v22_education_two_hop_relation_0096` | answer | two_hop_relation | System-Cedar. | Atlas. | PENDING | PENDING |
| 46 | hard_plurality | `v22_health_conflicting_evidence_0095` | abstain | conflicting_evidence | I don't know based on the available information. | System-1.. | PENDING | PENDING |
| 47 | hard_plurality | `v22_writing_insufficient_information_0098` | abstain | insufficient_information | I don't know based on the available information. | Harbor. | PENDING | PENDING |
| 48 | hard_plurality | `v22_coding_aggregate_delta_0097` | answer | aggregate_delta | 43. | Cedar. | PENDING | PENDING |
| 49 | hard_plurality | `v22_home_capacity_fit_0097` | answer | capacity_fit | Ember. | Falcon. | PENDING | PENDING |
| 50 | hard_plurality | `v22_agents_compare_metric_0095` | answer | compare_metric | Ember. | System-1. | PENDING | PENDING |
| 51 | probability_mean | `v22_consumer_conflicting_evidence_0091` | abstain | conflicting_evidence | I don't know based on the available information. | System't. | PENDING | PENDING |
| 52 | probability_mean | `v22_devops_insufficient_information_0097` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 53 | probability_mean | `v22_general_aggregate_delta_0090` | answer | aggregate_delta | 57. | I don't know based on the available information. | PENDING | PENDING |
| 54 | probability_mean | `v22_tools_capacity_fit_0094` | answer | capacity_fit | Atlas. | I don't know based on the available information. | PENDING | PENDING |
| 55 | probability_mean | `v22_legal_compare_metric_0091` | answer | compare_metric | Beacon. | Atlas. | PENDING | PENDING |
| 56 | probability_mean | `v22_science_constraint_preserving_transform_0091` | answer | constraint_preserving_transform | A. | Delta. | PENDING | PENDING |
| 57 | probability_mean | `v22_education_constraint_satisfaction_0099` | answer | constraint_satisfaction | A. | I don't know based on the available information. | PENDING | PENDING |
| 58 | probability_mean | `v22_general_dependency_order_0090` | answer | dependency_order | Ember. | System-Be. | PENDING | PENDING |
| 59 | probability_mean | `v22_finance_diagnostic_priority_0099` | answer | diagnostic_priority | monitor. | I don't know. | PENDING | PENDING |
| 60 | probability_mean | `v22_media_evidence_binding_0091` | answer | evidence_binding | Cedar. | System. | PENDING | PENDING |
| 61 | probability_mean | `v22_creative_evidence_synthesis_0094` | answer | evidence_synthesis | review. | I don't know based on the available information. | PENDING | PENDING |
| 62 | probability_mean | `v22_science_filter_then_rank_0098` | answer | filter_then_rank | Ember. | I don't know based on the available. | PENDING | PENDING |
| 63 | probability_mean | `v22_media_rule_exception_0092` | answer | rule_exception | Falcon. | I don't know based on the available information. | PENDING | PENDING |
| 64 | probability_mean | `v22_general_state_transition_0096` | answer | state_transition | running. | done. | PENDING | PENDING |
| 65 | probability_mean | `v22_engineering_two_hop_relation_0097` | answer | two_hop_relation | System-Delta. | System . | PENDING | PENDING |
| 66 | probability_mean | `v22_systems_conflicting_evidence_0096` | abstain | conflicting_evidence | I don't know based on the available information. | System. | PENDING | PENDING |
| 67 | probability_mean | `v22_health_insufficient_information_0099` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 68 | probability_mean | `v22_home_aggregate_delta_0097` | answer | aggregate_delta | 41. | I don't know based on the available information. | PENDING | PENDING |
| 69 | probability_mean | `v22_science_capacity_fit_0093` | answer | capacity_fit | Atlas. | I don't know based on the available information. | PENDING | PENDING |
| 70 | probability_mean | `v22_coaching_compare_metric_0096` | answer | compare_metric | Beacon. | Atlas. | PENDING | PENDING |
| 71 | probability_mean | `v22_writing_constraint_preserving_transform_0098` | answer | constraint_preserving_transform | A. | Harbor. | PENDING | PENDING |
| 72 | probability_mean | `v22_legal_constraint_satisfaction_0090` | answer | constraint_satisfaction | A. | I don't know based on the available. | PENDING | PENDING |
| 73 | probability_mean | `v22_devops_dependency_order_0093` | answer | dependency_order | Cedar. | I don't know based on the available information. | PENDING | PENDING |
| 74 | probability_mean | `v22_personal_diagnostic_priority_0093` | answer | diagnostic_priority | monitor. | I don't know. | PENDING | PENDING |
| 75 | probability_mean | `v22_education_evidence_binding_0096` | answer | evidence_binding | Atlas. | System-Be. | PENDING | PENDING |
| 76 | probability_mean | `v22_agents_evidence_synthesis_0095` | answer | evidence_synthesis | verification. | Harbor. | PENDING | PENDING |
| 77 | probability_mean | `v22_engineering_filter_then_rank_0099` | answer | filter_then_rank | Beacon. | I don't know based on the available. | PENDING | PENDING |
| 78 | probability_mean | `v22_health_rule_exception_0095` | answer | rule_exception | Beacon. | I don't know based on the available information. | PENDING | PENDING |
| 79 | probability_mean | `v22_systems_state_transition_0091` | answer | state_transition | done. | I. | PENDING | PENDING |
| 80 | probability_mean | `v22_health_two_hop_relation_0094` | answer | two_hop_relation | System-Delta. | System . | PENDING | PENDING |
| 81 | probability_mean | `v22_personal_conflicting_evidence_0096` | abstain | conflicting_evidence | I don't know based on the available information. | System't. | PENDING | PENDING |
| 82 | probability_mean | `v22_consumer_insufficient_information_0092` | abstain | insufficient_information | I don't know based on the available information. | Falcon. | PENDING | PENDING |
| 83 | probability_mean | `v22_tools_aggregate_delta_0094` | answer | aggregate_delta | 47. | I don't know based on the available information. | PENDING | PENDING |
| 84 | probability_mean | `v22_home_capacity_fit_0096` | answer | capacity_fit | Beacon. | I don't know based on the available information. | PENDING | PENDING |
| 85 | probability_mean | `v22_communication_compare_metric_0091` | answer | compare_metric | Beacon. | I don't know based on the available information. | PENDING | PENDING |
| 86 | probability_mean | `v22_systems_constraint_preserving_transform_0098` | answer | constraint_preserving_transform | A. | Cedbe. | PENDING | PENDING |
| 87 | probability_mean | `v22_travel_constraint_satisfaction_0092` | answer | constraint_satisfaction | A. | I don't know based on the available information. | PENDING | PENDING |
| 88 | probability_mean | `v22_legal_dependency_order_0096` | answer | dependency_order | Cedar. | System-Be. | PENDING | PENDING |
| 89 | probability_mean | `v22_writing_diagnostic_priority_0094` | answer | diagnostic_priority | inspect. | I don't know based on the available information. | PENDING | PENDING |
| 90 | probability_mean | `v22_coaching_evidence_binding_0090` | answer | evidence_binding | Cedar. | I don't know based on the available information. | PENDING | PENDING |
| 91 | probability_mean | `v22_tools_evidence_synthesis_0090` | answer | evidence_synthesis | review. | I don't know based on the available information. | PENDING | PENDING |
| 92 | probability_mean | `v22_operations_filter_then_rank_0090` | answer | filter_then_rank | Harbor. | I don't know based on the available. | PENDING | PENDING |
| 93 | probability_mean | `v22_operations_rule_exception_0094` | answer | rule_exception | Atlas. | I don't know based on the available information. | PENDING | PENDING |
| 94 | probability_mean | `v22_personal_state_transition_0092` | answer | state_transition | running. | done. | PENDING | PENDING |
| 95 | probability_mean | `v22_data_analysis_two_hop_relation_0091` | answer | two_hop_relation | System-Cedar. | Atlas. | PENDING | PENDING |
| 96 | probability_mean | `v22_personal_conflicting_evidence_0094` | abstain | conflicting_evidence | I don't know based on the available information. | System-System. | PENDING | PENDING |
| 97 | probability_mean | `v22_coaching_insufficient_information_0091` | abstain | insufficient_information | I don't know based on the available information. | Delta. | PENDING | PENDING |
| 98 | probability_mean | `v22_tools_aggregate_delta_0098` | answer | aggregate_delta | 44. | I don't know based on the available information. | PENDING | PENDING |
| 99 | probability_mean | `v22_legal_capacity_fit_0090` | answer | capacity_fit | Ember. | I don't know based on the available information. | PENDING | PENDING |
| 100 | probability_mean | `v22_math_compare_metric_0096` | answer | compare_metric | Beacon. | I don't know based on the available information. | PENDING | PENDING |
