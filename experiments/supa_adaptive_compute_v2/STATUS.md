# Programme status

Snapshot: **2026-09-13 17:30 UTC**. The full research goal remains open; neither track has generated confirmatory outputs or established a capability gain. Historical records stay in their original directories and Git history.

| Track | Mechanics | Candidate selection | Current execution | Gate |
|---|---|---|---|---|
| A — Native recurrence | v2.4 CPU/GPU qualification PASS, 40/40 comparisons | A0/A1/A3/A16 plus separately calibrated pretrained AM frozen | A0 128 complete; AM 16 complete at last audited cutoff; new queue started after old worker's absence was recorded | Full architecture matrix required; confirmation CLOSED |
| B — Donor transformation | CPU alignment 16/16 pairs, 96/96 objective cells; independent audit 4,931 PASS checks | B3 Supa L19 / Qwen3-4B L24 selected from held-out calibration | B-only XM calibration COMPLETE, committed and verified; independent audit 3,373 PASS checks; alpha 1.6186783995925944; CUDA screen configuration derived | MECHANISM_GO only; dose screen awaits per-cell runtime qualification; confirmation CLOSED |
| Native versus donor comparison | Paired infrastructure implemented | No jointly frozen efficacy results | No capability/cost winner established | CLOSED |
| Routing, Qwen3-8B, MLA/MHA, MTP, recovery, distillation | Not executed | Not licensed | None | CLOSED |

Current Native registration: `081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e`. Native v2.4 AM uses three pretrained FFNs at 90% of A3's added matrix FLOPs and fit-only alpha 0.764362266301519. Earlier random-control failures remain failed; their output is not mixed into v2.4.

Selected B3 held-out ordinary delta cosine/MSE: **0.849178 / 0.282078**, versus same-pair B0 **0.710979 / 0.507394**. This establishes improved transformation fidelity, not efficacy. Stage 2 fixes X0 and XG/XS/XR/XM at four doses, 17 cells; XN is unavailable. No scalar, dose or threshold may be chosen from confirmation.

Latest cited validation: **243 tests + 6 subtests** for Stage 2; **331 tests + 148 subtests** for Native/shared (separate, overlapping suites). Independent B audit validates metadata/scalars/provenance, not an independent model rerun. The Native resume audit passed 978 checks, and a fresh byte audit passed all 198 historical files plus the canonical checkpoint. Shared heavy-model execution remains serial; available GPU does not eliminate host-memory admission requirements. The earlier B calibration queue exhausted its 30-minute budget at 03:51:35 UTC without launching a model; the new calibration is a distinct execution.

Inspected branch `codex/supa-adaptive-compute-v2`, HEAD `ffad33f26920de5f0b1274968a1d87e45298fe63`; live remote HEAD `a6ee824584a9e28881826ab92a11db48e124d90c`. New execution evidence may accumulate; this documentation is pending commitment. No push is claimed.

[Current report: protocols, exact evidence, commands, power and limitations](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/CURRENT_REPORT.md). [Runtime status](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/GPU_STATUS.md).

Live B qualification queue: `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T1720_X0_cpu/`. It waits for the Native controller to exit before one X0 CPU reference phase; no B qualification or generation is claimed yet. Latest Native counts come from terminal-manifest snapshot `00_audit/execution_snapshot_20260913T172116.json`, not a new independent model audit.

Latest operational snapshot (**17:30:42 UTC**): A0 128/128; AM 91/128 in 21 complete shards; A1/A3/A16 pending. AM shard 22 stopped before model loading at 0.43 GiB available RAM versus the unchanged 0.50 GiB floor, with zero cases. There are three preserved pre-load memory failures in total and no completed-matrix efficacy result. Native queue PID 3512 and B qualification queue PID 110104 were both confirmed live; the Native controller waits for capacity, and B waits for Native to exit. Exact terminal-manifest inventory: `00_audit/execution_snapshot_20260913T173042.json`.
