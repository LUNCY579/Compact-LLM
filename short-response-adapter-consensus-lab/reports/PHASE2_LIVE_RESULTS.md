# Phase 2 Live Results

- Updated UTC: `2026-08-14T20:39:16.177657+00:00`
- Cornerstone verification: **PASS (30/30 checks)**
- Existing full-TEST rescue/harm, divergence, coalition, and failure analysis: **complete**
- DEV teacher-forced cache and analysis: **complete (4,050 examples; 18,760 token decisions; 127 committees)**
- Robust soft-voting controls: **DEV_ROBUST_SOFT_CONTROLS_COMPLETE**
- Rank top-k implementation parity: **RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED**
- Paper figures: **14 rendered; Figure 9 complete**
- Paper tables: **ALL_PLANNED_TABLES_COMPLETE**
- Adaptive K3→K7 policy: **FROZEN** at threshold `0.34`; free-running analysis **DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE**
- Selected committee/leave-one-out panel: **DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE**
- Selected calibrated v2.2 TEST confirmation: **V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE**
- v2.1.2 provenance gate: **PASS_WITH_CAVEATS (27/27 mandatory checks)**; cross-corpus analysis **V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE**
- Final synthesis fact packet: **FINAL_SYNTHESIS_FACTS_COMPLETE**

## Selected same-split free-running DEV

- Progress: **16,200/16,200 rows (100.00%)**
- Durable output bytes: **293,598,899**
- Last completed method/example: `reciprocal_rank_top5` / `v22_agents_diagnostic_priority_0089`
- Last file update UTC: `2026-08-13T07:56:48.090538+00:00`
- Frozen canonical config SHA256: `a887a4874f9717584aacb3a0f7019224d4a1571c8b76f28a92c04b965e56552e`
- Duplicate method-example rows: **0**
- Malformed complete lines: **0**
- Rows with OOM retry telemetry: **3,418**; retried examples: **0** (0 retries)

- `hard_plurality`: **4,050/4,050**
- `probability_mean`: **4,050/4,050**
- `calibrated_probability_mean`: **4,050/4,050**
- `reciprocal_rank_top5`: **4,050/4,050**

The reference decoder is intentionally used because the optimized separate-cache parity gate failed. The reciprocal-rank block spans an exact-parity implementation boundary and bounded fresh-process resumes; the source-bound parity artifact proves stable top-k equivalence. Methods run in blocks, so partial accuracies are not reported as cross-method evidence.

## Current verified headline

Hard K=7 scores 17.259% on 4,050 v2.2 TEST examples, versus 14.593% for Track A and 14.099% for probability mean.

## Interpretation discipline

Teacher-forced and free-running DEV are model-selection/mechanism evidence. The complete v2.2 TEST result remains the confirmatory cornerstone. No new TEST or v2.1.2 accuracy is claimed until its procedure is frozen and the run is complete.
