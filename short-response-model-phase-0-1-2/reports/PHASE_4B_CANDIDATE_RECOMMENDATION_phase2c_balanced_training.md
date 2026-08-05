# PHASE 4B CANDIDATE RECOMMENDATION

## Policy

Structural candidates are evaluated by proportional parameter-memory reduction versus combined proportional quality opportunity loss. No absolute score or reduction floor applies.

- Candidate registry: `artifacts\phase_4b_structural_reduction_phase2c_balanced_training\candidate_registry.json`
- Quality evaluator: `3.0`
- Quality profile: `balanced`

| Candidate | Resource gain | Quality opportunity loss | Opportunity margin | Mean quality | Pareto | Decision |
|---|---:|---:|---:|---:|---:|---|
| dense_reference | 0.000% | 0.000% | +0.000% | 0.7013 | YES | WORTHWHILE |
| mlp_long_specific_10pct | 4.110% | 4.439% | -0.329% | 0.6643 | YES | NOT WORTHWHILE |
| uniform_drop_2 | 6.923% | 16.709% | -9.786% | 0.5594 | YES | NOT WORTHWHILE |
| mlp_random_10pct | 4.110% | 18.097% | -13.987% | 0.5468 | NO | NOT WORTHWHILE |
| drop_last_2 | 6.923% | 30.209% | -23.286% | 0.3980 | NO | NOT WORTHWHILE |

## Automated recommendation

- Candidate: `dense_reference`
- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model`
- Opportunity margin: `+0.000%`
- Resource gain: `0.000%`
- Quality opportunity loss: `0.000%`

This remains a recommendation. Human review may choose the dense fallback or another Pareto candidate.
