# PHASE 2C — IMPORTANCE-GUIDED TARGETED RECOVERY REPORT

## Purpose

Preserve the approved Phase 2B reduced architecture while finding the smallest surviving-layer LoRA placement whose proportional trainable-state saving is greater than its proportional quality opportunity loss.

## Evidence boundary

- Source Phase 2B profile: `balanced`
- Source reduced model: `artifacts\phase_2b_quality_v4_full_20260802_213049\frontier\stage_09_drop_14_24_6_23_8_10_18_9_7\model`
- Sealed evaluation used: **No**
- Layer-ranking calibration cases: 64
- Development evaluation cases per candidate: 180
- Base seed: 11; repeat seeds: 23,37

## Surviving-layer importance ranking

| Rank | Current layer | Original layer | Combined | Taylor | Ablation NLL delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.9005 | 1.361e-03 | +10.5018 |
| 2 | 2 | 2 | 0.8387 | 1.485e-03 | +9.2170 |
| 3 | 1 | 1 | 0.7350 | 1.429e-03 | +7.0921 |
| 4 | 4 | 4 | 0.3928 | 1.647e-03 | +0.2244 |
| 5 | 6 | 11 | 0.3895 | 1.651e-03 | +0.8827 |
| 6 | 5 | 5 | 0.3759 | 1.713e-03 | +0.3232 |
| 7 | 3 | 3 | 0.3601 | 1.529e-03 | +0.3809 |
| 8 | 7 | 12 | 0.2857 | 1.464e-03 | +0.3248 |
| 9 | 10 | 16 | 0.2697 | 1.217e-03 | +0.6841 |
| 10 | 8 | 13 | 0.2519 | 1.300e-03 | +0.2271 |
| 11 | 9 | 15 | 0.2303 | 1.257e-03 | +0.2292 |
| 12 | 11 | 17 | 0.2118 | 1.072e-03 | +0.1879 |
| 13 | 12 | 19 | 0.1841 | 1.029e-03 | +0.6679 |
| 14 | 18 | 27 | 0.1631 | 5.456e-04 | +0.6698 |
| 15 | 13 | 20 | 0.1492 | 9.034e-04 | +0.5357 |
| 16 | 15 | 22 | 0.1055 | 6.926e-04 | +0.7080 |
| 17 | 14 | 21 | 0.1015 | 7.851e-04 | +0.3954 |
| 18 | 16 | 25 | 0.0785 | 6.028e-04 | +0.4889 |
| 19 | 17 | 26 | 0.0402 | 4.738e-04 | +0.7365 |

The ranking combines first-order target-loss sensitivity, temporary layer-bypass NLL damage, and optional final-stage Phase 2B structural evidence. It ranks where limited updates may be useful; it does not authorize deleting high- or low-ranked layers.

## Candidate frontier

| Candidate | Current layers trained | Modules | Quality index | Trainable reduction | Quality opportunity loss | Opportunity margin | Critical | Repeat seeds | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| all_remaining | all | both | 0.7292 | 0.000% | 0.000% | +0.000% | 0 | 2 | WORTHWHILE |
| top1 | 0 | both | 0.6458 | 94.737% | 11.380% | +86.355% | 2 | 2 | WORTHWHILE |
| top2 | 0,2 | both | 0.6760 | 89.474% | 7.648% | +81.826% | 0 | 0 | WORTHWHILE |
| top4 | 0,1,2,4 | both | 0.6742 | 78.947% | 7.466% | +71.482% | 1 | 0 | WORTHWHILE |
| top8 | 0,1,2,3,4,5,6,7 | both | 0.6936 | 57.895% | 4.836% | +56.196% | 0 | 2 | WORTHWHILE |
| complementary4 | 0,4,10,18 | both | 0.6640 | 78.947% | 8.746% | +70.202% | 1 | 0 | WORTHWHILE |
| random4 | 3,8,14,17 | both | 0.6762 | 78.947% | 7.115% | +71.832% | 0 | 0 | WORTHWHILE |
| low4 | 14,15,16,17 | both | 0.6382 | 78.947% | 12.090% | +66.857% | 4 | 0 | WORTHWHILE |
| top4_attention | 0,1,2,4 | attention | 0.6508 | 90.431% | 10.759% | +79.672% | 3 | 0 | WORTHWHILE |
| top4_mlp | 0,1,2,4 | mlp | 0.6893 | 88.517% | 5.303% | +85.483% | 1 | 2 | WORTHWHILE |

## Recommended retained profiles

| Profile | Candidate | Trained layers | Trainable parameters | Quality index |
|---|---|---|---:|---:|
| maximum_quality | all_remaining | all | 6,848,512 | 0.7292 |
| balanced_training | top1 | 0 | 360,448 | 0.6458 |
| minimum_training_memory | top1 | 0 | 360,448 | 0.6458 |

## Selection rule

A selective candidate is compared with a same-seed all-surviving-layer control. Proportional trainable-state savings must be strictly greater than combined proportional quality opportunity loss. Semantics, content anchors, completion, categories, issues, and target NLL contribute continuously and never act as private absolute vetoes. Peak training VRAM and wall time remain measured secondary evidence.

## Handoff

After human approval, Phase 3 loads the same reduced Phase 2B checkpoint plus the selected Phase 2C SFT adapter. Phase 3 must write profile-specific artifacts so incompatible architectures or adapter placements are never mixed in Phase 4 soups.
