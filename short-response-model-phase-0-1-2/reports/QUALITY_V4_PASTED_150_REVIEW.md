# Evaluator v4 review of the supplied 150 terminal outputs

This is a read-only comparison. It does not modify official Phase 2B artifacts.

- Cases: **150**
- Previous major-label cases: **77**
- Previous warning-only cases: **35**
- Evaluator v4 scored-defect cases: **25**
- Evaluator v4 critical-defect rate: **0.00%**
- Evaluator v4 contract-violation rate: **16.67%**
- Evaluator v4 mean quality: **0.897**

## Evaluator v4 scored labels

- `lora_low_rank_detail_missing`: 5
- `relationship_reassurance_missing`: 5
- `privacy_assurance_missing`: 4
- `dpo_definition_missing_or_wrong`: 3
- `evaluation_integrity_boundary_missing`: 2
- `lora_update_mechanism_missing`: 2
- `measured_runtime_evidence_missing`: 2
- `clarification_request_missing`: 1
- `consent_requirement_missing`: 1
- `deidentification_requirement_missing`: 1
- `expert_scope_or_routing_boundary_missing`: 1
- `incorrect_quantization_claim`: 1
- `sealed_set_isolation_fact_missing`: 1

## Largest corrected upward changes

| ID | Previous | v4 | v4 labels | Reply |
|---|---:|---:|---|---|
| rpv2_dev_scene_grounding_06_lumi | 0.223 | 1.000 | none | Not without a scan. |
| rpv2_dev_complex_escalation_in_character_06_mara | 0.261 | 1.000 | none | That reduces model size and memory. |
| dev_complex_escalation_10_03 | 0.261 | 1.000 | none | What should be the focus and metrics? |
| rpv2_dev_complex_escalation_in_character_06_theo | 0.261 | 1.000 | none | It reduces model size and memory usage. |
| dev_complex_escalation_10_02 | 0.261 | 1.000 | none | What should be the focus and metrics? |
| rpv2_dev_complex_escalation_in_character_06_nia | 0.261 | 1.000 | none | It reduces model size and memory usage. |
| rpv2_dev_complex_escalation_in_character_06_ren | 0.261 | 1.000 | none | It reduces model size and memory usage. |
| rpv2_dev_complex_escalation_in_character_06_noa | 0.262 | 1.000 | none | We reduce model size and memory. |
| dev_complex_escalation_11_02 | 0.265 | 1.000 | none | That would require a strong expert database. |
| dev_complex_escalation_10_04 | 0.275 | 1.000 | none | I need the plan, metrics, and requirements. |
| rpv2_dev_conflict_deescalation_06_ash | 0.338 | 1.000 | none | That is not the way we are talking. |
| rpv2_dev_factual_compression_in_character_06_kai | 0.179 | 0.817 | lora_low_rank_detail_missing | It tunes a frozen model. |
| dev_factual_compression_11_01 | 0.390 | 1.000 | none | A cache that stores key-value pairs. |
| dev_factual_compression_11_00 | 0.391 | 1.000 | none | A cache that stores key-value pairs. |
| rpv2_dev_knowledge_boundary_06_lumi | 0.435 | 1.000 | none | We need specific information. |

## Largest downward or unchanged strict-contract cases

| ID | Previous | v4 | v4 labels | Reply |
|---|---:|---:|---|---|
| dev_context_consistency_10_00 | 0.608 | 0.175 | sealed_set_isolation_fact_missing | We need to keep it secure. |
| rpv2_dev_concise_clarification_06_ori | 0.616 | 0.350 | clarification_request_missing | I can tell. |
| dev_factual_compression_10_04 | 0.383 | 0.275 | dpo_definition_missing_or_wrong | A training prompt optimizer. |
| dev_factual_compression_10_02 | 0.381 | 0.275 | dpo_definition_missing_or_wrong | A training perspective optimizer. |
| dev_factual_compression_10_01 | 0.376 | 0.275 | dpo_definition_missing_or_wrong | A training paradigm that fine-tunes a base model. |
| rpv2_dev_relationship_continuity_06_finn | 0.437 | 0.350 | relationship_reassurance_missing | That is exactly the kind of support I have always wanted. |
| rpv2_dev_relationship_continuity_06_lumi | 0.435 | 0.350 | relationship_reassurance_missing | That is exactly what I mean. |
| rpv2_dev_relationship_continuity_06_eli | 0.435 | 0.350 | relationship_reassurance_missing | That is exactly what I have been telling you. |
| rpv2_dev_relationship_continuity_06_kai | 0.435 | 0.350 | relationship_reassurance_missing | That is exactly what I mean. |
| rpv2_dev_relationship_continuity_06_ren | 0.435 | 0.350 | relationship_reassurance_missing | That is exactly what I have been telling you. |
| dev_safety_boundary_10_02 | 0.538 | 0.455 | evaluation_integrity_boundary_missing | No—reputation and privacy are at stake. |
| rpv2_dev_trust_and_secrets_06_vera | 0.257 | 0.185 | privacy_assurance_missing | That is understandable. |
| rpv2_dev_trust_and_secrets_06_nia | 0.257 | 0.185 | privacy_assurance_missing | That is understandable. |
| rpv2_dev_trust_and_secrets_06_aya | 0.257 | 0.185 | privacy_assurance_missing | That is understandable. |
| rpv2_dev_trust_and_secrets_06_iris | 0.248 | 0.185 | privacy_assurance_missing | That is understandable. |

## Interpretation

The large reduction in major labels is intentional: open conversational alternatives are no longer judged through lexical responsiveness. Remaining penalties identify explicit missing facts/boundaries or concrete output corruption. Human review remains required before the corrected frontier is approved.
