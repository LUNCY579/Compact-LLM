# Report policy after Evaluation v3

Reports already stored in this repository may have been generated under the
superseded Evaluation v2 absolute-threshold policy. They remain historical
evidence and are not silently rewritten.

Before using old generated replies for a new official decision:

1. rescore the preserved `results.jsonl` with
   `scripts/common/rescore_quality_artifacts.py`;
2. for a Phase 2B artifact tree, use
   `scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py`;
3. regenerate comparisons, opportunity margins, catalogs, and reports under
   Quality Evaluator v3;
4. do not mix v2 and v3 summaries in one automated comparison.

The current governing policy is documented in
`RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and
`BLUEPRINT_AMENDMENT_RELATIVE_OPPORTUNITY_v1.0.md`.
