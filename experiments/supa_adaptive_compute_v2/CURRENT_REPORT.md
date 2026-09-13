# Two-track Supa research — current report

Snapshot: **2026-09-13 17:21 UTC**, The research goal remains open. Track B has a complete, audited mechanism screen and a selected bridge; its separately registered control calibration is complete, committed and audited. Track A's v2.4 architecture screen has resumed and remains incomplete. **Neither track has confirmatory generation or a demonstrated capability gain.** Historical receipts and earlier reports remain preserved.

## Repository and scope

Repository: `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale`; origin: [LUNCY579/ChatBubble](https://github.com/LUNCY579/ChatBubble). Branch: `codex/supa-adaptive-compute-v2`. Inspected local HEAD: `ffad33f26920de5f0b1274968a1d87e45298fe63`; live remote HEAD: `a6ee824584a9e28881826ab92a11db48e124d90c` (`git ls-remote origin HEAD`, this snapshot). No push is claimed. The worktree retains untracked historical graft work; new execution evidence may accumulate while queues run. These status-document edits are pending commitment, so it is not clean.

The [existing-work inventory](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/00_audit/existing_work_inventory.md) records exact branches, commits and reuse classifications. Native hooks/block recurrence come from `experiments/anchor_routed_middle_recurrence_v1/src/armr`; actual donor-layer loading, masks and residual graft mechanics reuse preserved `experiments/cross_scale_layer_graft_v1/graft.py`; streamed checkpoint access comes from `Rineya_Cross_Scale_Capability_Transfer_Lab_v1`. The untracked historical graft has hash-bound local provenance, not an invented source commit.

Shared model loading, generation, hidden capture, scoring, paired statistics, hashing, frozen-state checks, runtime qualification and machine leases live in `experiments/supa_adaptive_compute_v2/shared/`. Repairs add strict deadlines, sealed/role guards, immutable attempts and complete paired coverage. Candidate operators, calibration, selection, protocols and evidence remain scientifically separate. Track B Stage 2 has its own lifecycle under `cross_scale/generation_stage/v1/`; it does not reuse Stage 1's generation-closed registration.

## Data and analysis

Each independently frozen track bank has **128 calibration scenarios (96 fit/32 held out), 128 architecture originals, 1,024 confirmatory originals and 128 linked diagnostic variants**. Eight families cover 32 templates; eight templates are excluded from engineering. Linked variants add no independent primary N. Frozen cases/scorers are not edited after model outputs. Sealed/TEST2/private evaluation remains closed.

The eventual paired analyses report accuracy, wins/losses, exact McNemar, scenario bootstrap confidence intervals, absolute/relative error reduction, family/template/variant effects, NLL, output identity, validity and measured cost. There is presently no complete confirmatory comparison for either track and no licensed cross-track capability/cost frontier.

## Track A — native recurrence

Current configuration: [protocol_gpu_v2_4.json](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/native_armr/configs/protocol_gpu_v2_4.json), with the original Native preregistration and [v2.4 amendment](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/native_armr/protocol/gpu_fp32_v2_4_amendment.md). Registration: `081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e`.

Arms are A0 intact, A1 one additional L12, A3 one additional L12→L13 block, A16 one additional L16, and AM a frozen pretrained FFN residual after L13. AM applies intact L12/L13/L14 FFNs and their learned normalizations in parallel. It copies 28,314,624 parameters and adds 56,623,104 dominant matrix FLOPs/token, **90% of A3's added matrix work**. Its fit-only, label-free scalar is **0.764362266301519**; no backbone training, scalar search or holdout selection occurred.

v2.4 CPU/GPU qualification **passed all 40 comparisons**, exact greedy/repeated-GPU and disabled/restored checks. Maximum absolute first-logit error: **7.247925e-5**; hidden relative L2: **7.856709e-5**; absolute NLL error: **4.553795e-5**. Peak CUDA reservation was 2,663,383,040 bytes; wall time 313.01 seconds. This establishes runtime compatibility, not efficacy.

The independent 978-check resume audit at **03:21:25 UTC**, consistent with all 37 registration-bound attempt receipts inspected again at 17:13 UTC, found **A0 has 128 cases in 30 complete nonempty shards; AM has 16 cases in five complete shards; A1/A3/A16 have no v2.4 completed cases yet**. The nominal 32-shard schedule has two naturally empty shards. Two AM attempts failed RAM admission before any examples; the latest at 03:05 UTC saw 0.31 GiB versus the unchanged 0.50 GiB GPU staging floor. These are capacity failures, not mechanical rejection. The old queue's missing session and absent PID 230268 were recorded in an additional interruption receipt without inventing its stop time or post-run integrity. A new controller started at 17:13:49 UTC in `00_execution_queue/20260913T171349_1363153Z_2732bedc/`; later progress requires current run receipts. Partial AM output is not an accuracy denominator. A later terminal-manifest snapshot at **17:21:16 UTC** counts **A0 128/128 and AM 51/128 (12 complete shards)**, with A1/A3/A16 still pending. The two earlier RAM failures remain the only failed attempts; this count snapshot is not a new independent numerical audit. Evidence: `00_audit/execution_snapshot_20260913T172116.json`.

An additional score-blind baseline audit passed **5,383 checks**: all 128 v2.4 A0 token sequences, reference NLL values and saved logit-file bytes equal v2.3 exactly. It rehashed 1,071 files including all 147 current Native source files; it does not claim a model or tensor rerun. Evidence: `00_audit/native_v2_4_baseline_review_20260913/`.

Original random-control v2.2 and calibrated-random v2.3 mechanical failures remain preserved and are not relabeled. The complete new five-arm architecture gate precedes untouched confirmation. The confirmatory criterion remains at least +2 accuracy points or 5% relative error reduction, supportive paired 95% CI and McNemar, then A3 versus matched compute, with preservation/validity and quality-per-cost requirements. **Current Native GO/NO-GO: unresolved; confirmation closed.** Historical 12/16, 12/16 and 13/16 A0/A1/A3 results remain hypothesis-generating.

## Track B — donor transfer

The unchanged CPU [alignment protocol](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/cross_scale/protocol/transformation_alignment_v2.md) completed **16/16 depth pairs and 96/96 objective cells**, using Supa post {8,12,16,19} × Qwen3-4B {12,16,20,24}. Objectives are B0 absolute ridge, B1 token-normalized state, B2 delta-aware, B3 both, BS shuffled pairs and BR seeded random. Only bridges were fitted; both backbones remained frozen. Run: `alignment-cpu-v2-002`; CPU registration: `f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea`.

The complete-matrix selector chose **B3 at Supa L19 / donor L24** using held-out calibration activations only. The independent audit passed **4,931 checks**, reconstructing selection and checking metadata/scalar/source/artifact consistency. It did not independently rerun models or deserialize bridge/activation tensors.

| Same-pair held-out ordinary-token metric | B0 | Selected B3 | BS | BR |
|---|---:|---:|---:|---:|
| Donor delta cosine | 0.710979 | **0.849178** | 0.213387 | 0.051463 |
| Token-relative delta MSE | 0.507394 | **0.282078** | 0.994955 | 8.000689 |

Full 96-cell figures: [ordinary-token cosine](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/00_audit/alignment_matrix_visual_20260913/ordinary_delta_cosine.png) and [relative MSE](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/00_audit/alignment_matrix_visual_20260913/ordinary_delta_relative_mse.png). Vector PDFs, full-precision values, renderer and input hashes accompany the plots.

B3 ordinary delta RMS ratio is 0.922427; all-token delta cosine is 0.845053; final-token cosine is 0.916452. Ordinary up/down-map cosines are 0.911083/0.966440. Special-token fidelity is worse than same-pair B0: B3 cosine/MSE is **0.783316 / 1.351649**, versus B0 **0.922944 / 0.162114**. Strong ordinary-token alignment does not imply uniform improvement. All frozen mechanism gates passed. **MECHANISM_GO licenses a separate small-screen freeze, not donor efficacy or large generation.**

Stage 2 source/protocol/callers passed **243 tests + 6 subtests** and were committed at `115b1d5a8d1c0c61a4acf691d938a50844623c17`. The fixed screen is 128 B architecture originals, X0 plus XG/XS/XR/XM at gates .03/.10/.20/.40: 17 cells, eight shards each. Every candidate requires a complete matching X0 baseline. Failed cells remain failed; other cells continue; any failed same-dose control makes that dose ineligible. XN is unavailable because no independently selected best Native operator is frozen.

XM is an independent frozen 1024→34560→1024 SwiGLU (seed 2702, std .02), 106,168,320 matrix parameters and 212,336,640 added matrix FLOPs/token. This exactly matches the selected donor-plus-bridge dominant matrix work; the selected graft has 100,930,816 donor parameters plus 5,246,464 bridge coefficients. Its scalar must come solely from the registered B calibration fit energies. The selected-map export `selected_20260913_001` passed full upstream verification and was committed at `96c4605`. B-only calibration registration `bf99e713104573b7aafb8f5511278fd473bcd406ff1d6fece3d0ebe5787534a0` was committed at `effb3c15198f8e7f65468ef740897403155d77bc`; its preflight returned VERIFIED.

The earlier B calibration queue ended **QUEUE_BUDGET_EXHAUSTED at 03:51:35 UTC**, with no model launched. The actual fixed CPU calibration in `00_control_calibration/20260913T171252_566611Z_669e5d42/` is now **COMPLETE**: 96 fit + 32 holdout scenarios in **63.72 s**, peak RSS **2,019,373,056 bytes**, alpha **1.6186783995925944**. Fit teacher/scaled-control relative energies both equal 0.098211512; held-out energies are 0.098558545 / 0.098295374. The report records unchanged frozen states and checkpoints, zero gradients/trainable parameters, no answer labels and zero generated tokens. Official verification passed, evidence was committed at `2ae253f`, and an independent audit passed **3,373 checks with 310 input hashes**. The artifact-derived CUDA configuration is `cross_scale/generation_stage/v1/configs/generation_screen_gpu_v1.json`; registration `dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3` is committed at `513d194`. **No B Stage 2 runtime qualification, dose-screen generation or confirmation has run.** Calibration uses B-only cached activations and frozen operators, with no efficacy answers or Native calibration imported.

The fixed dose screen requires all four candidate/control cells at a dose to complete 128 examples, no additional invalid outputs versus X0 and no safety stops. XG must have at least two paired wins and one net correct answer over X0, at most one additional pooled preservation error, and raw accuracy at least that of XS, XR and XM. Eligible doses rank by margin over the strongest control, then gain over X0, then lower dose. These prospective selection rules reject inert or destructive behavior; they are not confirmatory significance criteria.

The earlier B GPU alignment profile remains **FAILED**: 161/656 absolute-error comparisons exceeded .002, despite passing relative/cosine criteria and all 16 gradient comparisons. No threshold was widened. Stage 1 therefore used its CPU reference; Stage 2 CUDA needs its own per-cell CPU/GPU qualification.

## Power, integrity and limitations

[Power planning](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/data/generated/v2/power_planning.json) covers +1/+2/+3 points and 5% relative error reduction at N=256/512/1024/2048. For a +2-point paired gain, approximate power at discordance 5%/10%/20% is respectively 30/17/11%, 53/30/17%, 82/53/30%, and 98/82/53%. At N=1,024, 80%-power MDE is 1.96/2.77/3.92 points. These are planning assumptions, not observed effects; multiplicity and scenario dependence matter. The 128-case screen cannot establish generalizable capability.

Pinned canonical checkpoint SHA256: `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`; Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`. Native/B bank contract hashes: `7ebdac9f8a5ce14277d6c74a138b96303495c82b9a98bcbaf74a498e8fedd4a2` / `6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46`. Scorer SHA256: `62937db0530db2f8b43672ff1ea3e9ffa96dce5e8b7c9cf5bac6996b3d29c11e`.

Latest Stage 2 validation records unchanged 147 Native v2.4 and 139 original B source bindings. The preserved 198-file historical inventory has root `b2b4986d6f3abc151ab0c7ec8748c421c919dbcf648afa526e0005d4bdcbfa8f`. Successful real runs require before/after checkpoint/state hashes, zero trainable backbone parameters and zero gradients. Failed/abandoned attempts retain partial artifacts and cannot become completed denominators. A fresh independent streamed-byte check at **03:22:45 UTC** reverified all 198 historical files and the canonical checkpoint SHA256 without loading tensors; receipt: `00_audit/integrity_20260913T0323/receipt.json`. The documentation worker did not load models.

Execution uses the pinned Python environment (torch 2.12.1+cu126, transformers 5.14.1), FP32/eager/four threads/no cache. The authorized RTX 4070 Laptop GPU uses deterministic CUDA, highest matmul precision and TF32 disabled. Heavy jobs are serial. Contending external workloads can delay admission and confound wall-time comparisons; no controlled accelerator speedup claim is made. The bank consists of bounded synthetic English tasks, not a broad external benchmark; preservation and template generalization claims must respect that scope. Routing, 8B, MLA/MHA, MTP, PEFT/backbone recovery and distillation remain closed. No cross-track efficacy winner can be declared yet.

## Evidence and validation

All paths below are under `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/`.

| Evidence | Exact directory |
|---|---|
| Native v2.4 calibration | `native_armr/evidence/00_learned_control_calibration/20260906T014443_419834Z_b790e5e7/` |
| Native v2.4 qualification | `native_armr/evidence/00_runtime_qualification/20260906T203929_861509Z_7b0b0fc8/` |
| Current Native queue and shards | `native_armr/evidence/00_execution_queue/20260913T171349_1363153Z_2732bedc/`; registration-bound runs in `native_armr/evidence/01_architecture_dev/` |
| Preserved interrupted Native queue | `native_armr/evidence/00_execution_queue/20260913T021009_4930074Z_585eaafd/`, including `interruption_observed_20260913T1713.json` |
| B complete matrix and selection | `cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/aggregate/attempt_000/` |
| B pair measurements | `cross_scale/evidence/02_bridge_objectives/alignment-cpu-v2-002/` |
| B independent matrix audit | `cross_scale/evidence/07_independent_audit/matrix_audit_20260907T004813_5299633Z/` |
| B verified committed export | `cross_scale/generation_stage/v1/evidence/00_upstream_exports/selected_20260913_001/` |
| Stage 2 passing tests | `cross_scale/generation_stage/v1/evidence/00_preparation_validation/20260913_screen_retry1/` |
| Native/shared passing tests | `00_audit/native_v2_4_validation_20260906/` |
| Current Native independent audit | `00_audit/native_v2_4_resume_review_20260913/` |
| B exhausted calibration queue | `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T032108_4919029Z_calibration/` |
| B completed, verified XM calibration | `cross_scale/generation_stage/v1/evidence/00_control_calibration/20260913T171252_566611Z_669e5d42/` |
| B independent control calibration audit | `00_audit/cross_control_calibration_review_20260913/attempt_001/` |

Commands below reproduce the two cited suites (overlapping coverage is not summed). Recorded results are **331 tests + 148 subtests in 106.19 s** for Native/shared and **243 tests + 6 subtests in 42.77 s** for Stage 2. The preceding Stage 2 run's three string-path failures are preserved; the retry passed after repair. Tests use small fixtures, not capability evidence.

```powershell
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$programme = 'C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2'
& $researchPython -m pytest "$programme/native_armr/tests" "$programme/tests" -q
& $researchPython -m pytest "$programme/cross_scale/generation_stage/v1/tests" -q
```

## Exactly one next causal experiment per track

**A:** Finish the fixed v2.4 five-arm architecture screen, retaining its pretrained matched-compute control and unchanged guards. A complete pass is required before the frozen 1,024-case confirmation. A new controller is live; the command below is for later resumption only after verifying its worker has exited. Never launch a duplicate campaign:

```powershell
& "$programme/scripts/run_native_when_available.ps1" -ConfigName protocol_gpu_v2_4.json -ArmOrder control-first -Role architecture_dev -MaxQueueSeconds 3600
```

**B:** Run the separately preregistered 17-cell dose/control screen for the selected B3 S19/D24 transformation. Its fixed label-free XM energy calibration, derived CUDA configuration and registration are committed. The next executable prerequisite is X0 CPU qualification, followed by the required per-cell CPU/GPU qualification before generation. A live one-phase queue already waits for the Native controller to exit, release its lease, and leave sufficient host memory. Its evidence is `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T1720_X0_cpu/`; do not launch a duplicate. The direct command, for later verified resumption only, is:

```powershell
$stage = "$programme/cross_scale/generation_stage/v1"
$screenConfig = "$stage/configs/generation_screen_gpu_v1.json"
& $researchPython "$stage/scripts/qualify_runtime.py" --config $screenConfig phase --cell X0 --phase cpu
```

Official registration verification passed for all 17 CUDA cells. The independent calibration audit and its 265-check verification of all 132 committed artifacts are committed at `ffad33f`. The frozen registration and per-cell qualification must precede `run_campaign.py` using `configs/generation_screen_gpu_v1.json`; no passing runtime receipt is claimed. A screen GO would still require a separate confirmation freeze.
