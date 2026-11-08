# Programme status

Snapshot: **2026-09-13 17:42 UTC**. The full research goal remains open. Neither track has confirmatory generation or a demonstrated capability gain.

| Track | Completed evidence | Current execution | Gate |
|---|---|---|---|
| A — Native recurrence | v2.4 CPU/GPU qualification 40/40 PASS; A0 and pretrained AM each 128/128; complete control audit 3,743 checks PASS | A1 running on GPU; A3 and A16 scheduled | Full five-arm architecture audit required; confirmation CLOSED |
| B — Donor transformation | 16 pairs / 96 cells, independent 4,931-check audit; B3 S19/D24 selected; B-only XM calibration and 17-cell CUDA registration committed and verified | X0 CPU qualification queued after the Native controller exits | MECHANISM_GO only; runtime qualification and dose screen remain; confirmation CLOSED |
| Native versus donor | Shared paired-analysis infrastructure | No jointly frozen efficacy results | Comparison CLOSED |
| Routing, 8B, MLA/MHA, MTP, recovery, distillation | Not executed | Not licensed | CLOSED |

Native registration: `081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e`. AM uses three parallel pretrained FFNs at 90% of A3's added dominant matrix FLOPs, with fit-only alpha 0.764362266301519. Its 128 outputs have zero invalidity or safety stops. The baseline identity audit passed 5,383 checks: all 128 v2.4 A0 token sequences, NLL values and saved logit files equal v2.3 exactly. These are mechanics/reproducibility results, not efficacy comparisons.

Three zero-example AM RAM failures remain intact, as do both earlier random-control mechanical rejections. The current Native controller is PID 3512, queue `native_armr/evidence/00_execution_queue/20260913T171349_1363153Z_2732bedc/`. It resumes only resource-admission failures. The B qualification queue is PID 110104, `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T1720_X0_cpu/`; it waits for Native to exit and adequate memory before one X0 CPU reference phase. Heavy model jobs remain sequential.

Selected B3 ordinary-token cosine/MSE is 0.849178 / 0.282078 versus B0 0.710979 / 0.507394. Special-token fidelity is worse than B0. B XM alpha is 1.6186783995925944, independently reproduced in 3,373 checks; the separate commitment audit checked all 132 artifacts. CUDA screen registration: `dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3`. No B generation or runtime qualification phase has run yet.

Tests: 331 + 148 subtests for Native/shared, and 243 + 6 subtests for B Stage 2; overlapping suites are not summed. Current full control audit: `00_audit/native_v2_4_complete_control_review_20260913/`. Both banks, scorer, checkpoints and frozen sources retain their boundaries; sealed data stays closed.

Inspected branch `codex/supa-adaptive-compute-v2`, evidence HEAD `914a267af74ffc164b046471b1189d8744661413`; remote HEAD last verified this day as `a6ee824584a9e28881826ab92a11db48e124d90c`. Live evidence and preserved untracked historical graft work keep the worktree dirty. No push is claimed.

[Full report, evidence, power, tests, plots and exact next experiments](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/CURRENT_REPORT.md). [Runtime details](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/GPU_STATUS.md).
