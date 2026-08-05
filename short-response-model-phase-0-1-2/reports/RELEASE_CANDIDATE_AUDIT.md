# RELEASE CANDIDATE AUDIT

## Automated status

`PASS_PENDING_HUMAN_RELEASE_DECISION`

- Selected model: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model`
- Selected candidate: `dense_reference`
- Deployment mode: `DENSE_RUNTIME`
- Model tree SHA-256: `07519cfa613a4fdfcf23e7d0e86489377f9878b208ca3ced05d39da7b82d4d52`

## Gate chain

- Phase 1: `PASS`
- Phase 2: `PASS_BY_DEVIATION`
- Phase 3: `PASS`
- Phase 4: `PASS`
- Phase 4B: `PASS`
- Phase 5: `PASS`
- Phase 6: `PASS`
- Phase 7: `PASS`

## Sealed continuous quality evidence

- Cases: `110`
- Mean quality: `0.8523948484848484`
- Worst-decile quality: `0.7174242424242424`
- Dimension means: `{'brevity_appropriateness': 0.9636363636363636, 'comprehensibility': 1.0, 'context_consistency': 1.0, 'contract_fulfillment': 1.0, 'conversation_naturalness': 0.9962121212121212, 'factual_correctness': 1.0, 'logic_factual_consistency': 1.0, 'mechanical_integrity': 1.0, 'prompt_function_alignment': 0.4742727272727273, 'response_completeness': 1.0, 'response_mode_fulfillment': 0.4742727272727273, 'safety_boundary': 1.0, 'task_adequacy': 0.4742727272727273}`
- Category quality: `{'acknowledgment': 0.8297966666666665, 'celebration': 0.9047166666666666, 'clarification': 1.0, 'complex_escalation': 0.8096, 'context_consistency': 0.7759166666666666, 'empathy': 0.8139966666666666, 'factual_compression': 0.7452, 'follow_up': 0.8076000000000001, 'naturalness': 0.8655999999999999, 'safety_boundary': 0.8539, 'uncertainty': 0.9700166666666666}`
- Mean issue cost: `0.0`
- Severe issue case rate: `0.0`
- Diagnostic issue-free rate: `1.0`
- Response-control mode: `one_sentence`
- One-sentence violation rate (diagnostic): `0.0`

No absolute numerical quality value in this section independently blocks or approves release.

## Relative release-candidate comparison

- Source: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_7_router\runtime_comparison.json`
- Deployment mode: `DENSE_RUNTIME`
- Primary resource: `measured_decode_latency_reduction`
- Resource gain: `0.0`
- Quality opportunity loss: `0.0777570445536157`
- Opportunity margin: `-0.0777570445536157`
- Relative decision: `NOT_PROPORTIONALLY_WORTHWHILE`

Sparse release is technically blocked when this same-case, same-hardware relative evidence is invalid or non-positive. Dense release remains the comparison reference and still requires human approval.

## Ablation evidence coverage

- Available entries: `9`
- Missing entries: `phase_2B_progressive_depth, phase_4B_structural`
- Stale legacy-quality entries: `A_base, B_all_layer_sft, C_best_single, D_best_2_4, E_random_control, F_selective_dpo, G_phase_4_accepted`

## Technical blocking reasons

- None.

## Human release decision

- Decision: `PENDING`
- Reviewer:
- Date:
- Version:
- Rollback checkpoint verified: `PENDING`
- Notes: