# Two-track Supa research: current execution report

Latest development: calibrated random AM v2.3 failed the same architecture
malformed-generation guard at shard 20. A0 completed 128 cases; AM has 84 cases
in complete shards plus one preserved case in the failed attempt. Its failure
integrity receipt confirms unchanged canonical state/checkpoints and zero
backbone gradients. No v2.3 A1/A3/A16 or confirmation run followed the rejection.
The full A0 baseline passed a separate 5,373-check audit, with every token
sequence, NLL value and logit-file byte identical to original v2.2 A0.

One separately preregistered v2.4 control is being prepared: three intact
pretrained FFNs (L12/L13/L14), applied in parallel to the same post-L13 state
through their own learned RMSNorms, with one residual injection. It uses
56,623,104 added dominant matrix FLOPs/token, explicitly 90% of A3, and copies
28,314,624 existing parameters without learning new coefficients. It changes
neither original native arms nor data/scorer/statistical/validity gates.
The 331-test/148-subtest suite and independent static review passed before its
calibration registration was committed at `cb5dc2e66fd3971019a633550bbf8ecabd703939`.
Calibration then completed in 54.94 seconds at 1,290,117,120 bytes peak RSS.
Its fixed fit-only formula produced alpha **0.764362266301519**. Fit A3/scaled
control energy is 0.09559580669801247; held-out energies are 0.09606493426847566
and 0.09659610573025923. No labels, clipping or scale search were used.
Artifact: `native_armr/evidence/00_learned_control_calibration/20260906T014443_419834Z_b790e5e7/`.
It was committed at `c72e6df1f06251824869d16c0e324e283eb693e0` before deriving
the new inference configuration. Fresh v2.4 qualification and a complete new
run on the existing architecture bank still precede untouched confirmation.
The two random-control failures stay intact; a failure of this specified
alternative does not license an automatic search for controls until one passes.
The final v2.4 registration is
`081a8f6bf3c7396c97f13e3d4a7d55bf7311b37dcc0b0bc7bfd32478fa44241e`.
Its source/config/registration were committed before execution eligibility.
The 600-second resource queue expired at 02:05 UTC without launching a stage;
its last observation was 0.53 GiB host RAM and 2,435 MiB free VRAM. No programme
model worker remains running. Qualification and donor fitting await capacity.
Queue receipt: `00_audit/execution_queues/20260906T015442_9607258Z_28dcf54d/queue_manifest.json`.

Track B remains at two complete pairs. S8/D20 attempt 000 failed pre-load RAM
admission (1.66 GiB versus 3.00 GiB), with no fitting. Its next attempt uses a
new directory. The original 139 Track B source hashes remain unchanged.

Execution update at 23:52 UTC: two of 16 Track B pairs completed all six objectives,
with unchanged donor state and exact native replay. S8/D12 took 128.65 seconds;
B3/B0 held-out ordinary delta cosine is 0.79367/0.32144 and token-relative MSE
0.37049/1.39826. S8/D16 took 111.50 seconds; B3/B0 cosine is 0.73081/0.27899 and
MSE 0.46297/48.51333. Their evidence is under
`cross_scale/evidence/02_bridge_objectives/alignment-cpu-v2-002/S8_D12/attempt_000/`
and `S8_D16/attempt_000/`. No bridge is selected; the next fixed pair is S8/D20.

Native v2.3 qualification **PASSED** at
`native_armr/evidence/00_runtime_qualification/20260905T235200_982255Z_36998242/`.
The first attempt failed RAM admission at 2.94 GiB versus 3.00 GiB. The next
attempt, `20260905T233921_233855Z_41832612`, preserved all 40 CPU reference logits
but failed CUDA initialization before GPU model loading. Windows logged an NVIDIA
TDR reset during that CPU phase; a fresh CUDA health probe passed afterward.
The reset's initiating cause is unknown. Both failed attempts stay FAILED; the
successful retry changed no scientific setting or threshold. All 40 comparisons,
greedy identities, exact GPU repeats and disabled/restored checks passed. The
admission validator recomputed frozen gates and verified all 80 logit artifact
hashes. Max absolute logit error was 4.19617e-5, max hidden relative L2 7.85671e-5,
and max absolute NLL error 4.55379e-5; peak CUDA reservation was 2.50391 GiB.
The 313.05-second run and report were committed at
`39122c92c9fa930ebab99a90135ae92aa06c4f7b`. A fresh GPU architecture matrix is
running in order A0, AM, A1, A3, A16. Confirmation remains closed until that
complete matrix passes its original gates.

Prepared generic export and graft interfaces under `cross_scale/generation_stage/v1/`
passed 56 tests and 6 subtests in 21.86 seconds. These use tiny CPU decoders and
synthetic/mocked provenance fixtures. No real export, generation protocol, dose
choice or generation caller exists. Both original source maps remain unchanged.
Latest code/evidence commit at this update is `0bd77ad3219a1650df391a75101bb7cce2983107`.
The research goal remains open.

Snapshot: **2026-09-05 22:45:35 UTC / 17:45:35 America/Chicago**. The shared
framework, independent banks, protocols and callers are implemented. Real runtime
qualification, Native architecture execution, Native control calibration and
Track B activation capture have run. Both capability questions remain open;
**zero confirmatory examples have been generated**. This report updates the
earlier delivery snapshot without modifying [FINAL_REPORT.md](FINAL_REPORT.md).

| Track | Question | Completed evidence | Current decision |
|---|---|---|---|
| A: Native ARMR | Does bounded native recurrence generalize beyond the inspected pilot? | Random AM v2.3 mechanically rejected; separately registered pretrained FFN calibration completed | v2.4 qualification and complete architecture matrix required; no efficacy claim |
| B: donor transfer | Can a frozen Qwen3-4B transformation be faithfully transferred into Supa? | GPU qualification failed its frozen absolute-error threshold; CPU capture and two six-objective pairs completed | 14 pairs pending; no selected bridge or generation license |
| Downstream mechanisms | Routing, MLA/MHA, MTP, 8B, backbone recovery and distillation | Not executed or licensed | All gates CLOSED |

The resource queue is active; a queued stage is not a running model. At this
snapshot, queue session `27927`, PID `207960`, had no `launch.json`; its latest
record showed 0.791 GiB available host RAM, 1,812 MiB free GPU memory and no
programme lease. It waits for 4 GiB host RAM and prefers the pending v2.3 Native
qualification when at least 3.5 GiB VRAM is free. Otherwise it may run the first
pending Track B CPU pair when host admission permits. It executes at most one
stage and returns for review. It does not retry failed science. Queue evidence:
`00_audit/execution_queues/20260905T224301_9855922Z_043745b4/`.

Repository: `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale`; origin:
`https://github.com/LUNCY579/ChatBubble.git`; branch:
`codex/supa-adaptive-compute-v2`; local HEAD:
`553f7fde97333339d54399f25b03b1dda22e2265`. Remote HEAD was verified by the root
worker at 22:44 UTC as `a6ee824584a9e28881826ab92a11db48e124d90c`; the local
`origin/main` ref agrees. No local remote-tracking ref exists for this new branch;
this report does not claim it was pushed. The local historical native remote ref
is `d8475ea3b9de3e17c5193a00ed01cc4aa6f1a136`, and the larger lab remote ref is
`fde553c908db39e87a15a52ed239d1ba0a68bfd3`. At inspection there were no tracked
modifications, but the working tree was not clean: preserved untracked graft work
and new audit/queue evidence were present. This new report adds another file.

The [existing-work audit](00_audit/existing_work_inventory.md) classifies exact
paths and commits as reused, repaired, historical, superseded, scientifically
different, unsafe or missing. Native recurrence reuses the scoped wrappers in
`experiments/anchor_routed_middle_recurrence_v1/src/armr/hooks.py` and
`recurrence.py`; whole-block order remains L12,L13,L12,L13. Canonical resolution,
hidden/logit/Unicode diagnostics and hashing reuse that historical implementation.
The preserved `experiments/cross_scale_layer_graft_v1/graft.py` supplies actual HF
layer loading and masks; the larger lab's `StreamingWeights` supplies checkpoint
streaming. Shared paired/scenario statistics reuse audited statistical primitives.
Repairs cover deadline checks, strict scoring, sealed/role guards, exclusive
evidence, complete paired coverage, token-balanced alignment and delta-aware
bridge fitting. Historical source/evidence directories remain separate and intact.

Both scientific callers use `shared/model_loading.py`, `hidden_capture.py`,
`generation.py`, `scoring.py`, `statistics.py`, `integrity.py`, `hashing.py`,
`manifests.py`, `runtime.py` and `timing.py`. These provide canonical identity,
full-prefix no-cache generation, NLL/logit/validity diagnostics, exact scoring,
paired analysis, cost accounting, immutable receipts and sequential heavy-job
ownership. Operators, calibration inputs, selection rules, registrations and
evidence belong to their own tracks. Track B fitting never consumes Track A
answers. Native v2.3 calibration exposes only its own prompt/identity fields.

Each track independently contains **1,024 primary original scenarios**, 128
linked diagnostic variants, 128 architecture cases and 128 calibration cases
(96 fit, 32 held out). Eight families cover 32 templates; eight templates are
excluded from engineering. Linked variants stay with their latent scenario and
add no independent primary N. Separate semantic oracles and tokenizer validation
passed; frozen prompts, answers, generator and scorer were not edited after
model output. Canonical manifest digests are Native
`7ebdac9f8a5ce14277d6c74a138b96303495c82b9a98bcbaf74a498e8fedd4a2`
and Track B `6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46`.
The strict scorer identity is
`62937db0530db2f8b43672ff1ea3e9ffa96dce5e8b7c9cf5bac6996b3d29c11e`.

Track A follows the original [preregistration](native_armr/protocol/preregistration.md)
and separately frozen [v2.3 GPU amendment](native_armr/protocol/gpu_fp32_v2_3_amendment.md).
Arms remain A0 intact, A1 one extra L12, A3 one extra L12-L13 block, A16 one extra
L16, and AM an independent frozen SwiGLU residual after L13. AM retains seed 7301,
normal std .02, width 10240, 31,457,280 added matrix parameters and 62,914,560
dominant matrix FLOPs/token, matching A3's two-layer matrix work. Attention and
normalization costs are excluded from this approximate compute match.

Original GPU v2.2 qualification passed all 40 CPU/GPU case-arm comparisons, with
identical greedy sequences, exact same-GPU repeats and disabled/restored identity.
Maximum absolute first-token logit error was `4.1961669921875e-05`, maximum
per-token hidden relative L2 error `7.856709228698431e-05`, and maximum absolute
answer-NLL error `4.553794860839844e-05`. Canonical and seeded control state hashes
matched across CPU/GPU. This qualified that runtime; it did not measure capability.
The first resource-interrupted qualification remains invalid and preserved.

Under v2.2, A0/A1/A3/A16 each completed **128 cases in valid complete architecture
shards**. Original AM has **84 admissible cases**, plus one saved case in failed
shard 20. The exact failure was the combined repetition/script/Unicode generation
guard. The rejected prefix was not saved, so its particular subtype is unknown.
The original five-arm matrix is incomplete. Its four completed native arms and
partial AM output do not license an efficacy comparison or confirmation. An
interrupted A16 attempt remains invalid even though all five retained token/NLL/
logit artifacts match its separate successful rerun. The architecture audit
checked 3,514 invariants plus seven failure-integrity checks, with no failures.

The prospective [v2.3 calibration](native_armr/protocol/control_calibration_v2_3.md)
was designed after that architecture failure. Its rule and source were frozen
under registration
`cf1440a9bbf3344bcc0cca316a1d5f9cb172a448b06bf1696b27646fc673fe57`
and committed at `4791635ac1a1ad22511c355bc59ab42956407dee` before execution.
CPU FP32 streamed raw post-L13 states and actual extra L12/L13 deltas use only
Track A calibration prompts. Each ordinary token's delta mean-square is divided
by `max(mean-square(h13),1e-8)`; ordinary-token means are then averaged equally
across the 96 fit scenarios. The sole scalar is
`sqrt(fit_native_energy / fit_control_energy)`; it was written before held-out
transformation measurements. No answer, accuracy, NLL, architecture output,
confirmation data, scalar search, clipping or holdout selection entered fitting.

The resulting **alpha is 1.091369554461779**, above the original 1.0 and retained
without alteration. Fit native energy is `0.09559580669801247`, versus unscaled
control `0.08025926419711565`; scaled control is `0.0955958066980125`. Holdout
native energy is `0.09606493426847566`, versus scaled control
`0.09577791697352757`. Execution took 62.75 seconds at 1,236,889,600 bytes peak
RSS. All recorded checkpoint, layer, control and buffer identities were preserved,
with zero trainable parameters and gradients. The control hash remains
`eedbc0ca788ee8a1193b455f7cf2c94cd1b5eeab288f1f2df448fa9ae208584c`.
Calibration establishes residual-energy matching; generation safety and efficacy
remain unmeasured for v2.3.

The calibration report SHA256 is
`20a552138f91ee37c3df261f576fc7ae4f0ff6214ddc9cdb8b383ff4d2e2789c`.
Its complete artifact was committed at
`a6126f2a3c51bba5d312ae6db2f7cb4af1092522` before deriving the final GPU profile.
Native GPU v2.3 registration is
`86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44`.
Fresh all-arm CPU/GPU qualification has passed. A fresh full architecture matrix
remains required; original receipts cannot be mixed into the new paired matrix.

No fresh Native accuracy gain, paired effect, preservation gain or cost frontier
is claimed. Historical A0/A1/A3 scores 12/12/13 of 16 remain hypothesis-generating.
The frozen eventual analysis tests A3>A0, then A3>AM, requiring positive 95% paired
CI lower bounds, exact McNemar p≤.05, and at least 2 absolute points or 5% relative
error reduction versus A0. A1/A16 have a separate Holm family. Scenario bootstrap,
family/template/variant analysis, canonical-answer NLL, validity and measured
latency/RSS/VRAM accompany accuracy. Preservation must exclude a pooled loss over
2 points in the registered families; no additional invalid output or safety stop
is allowed. These are future gates, not passed results.

Track B follows [transformation alignment v2](cross_scale/protocol/transformation_alignment_v2.md):
Supa post {8,12,16,19} × actual Qwen3-4B layers {12,16,20,24}, 16 pairs and six
objectives per pair. B0 is historical absolute ridge; B1 token-normalized state
alignment; B2 delta-aware alignment; B3 both; BS cross-scenario shuffled pairs;
BR seeded random bridges. B2/B3 fit only upward affine bridge parameters through
the actual frozen donor transformation, for the fixed 96-step budget. Backbones
and downward maps stay frozen. Six bridges each have been fitted for S8/D12 and
S8/D16; none is selected while the other 14 depth pairs remain incomplete.

Track B GPU qualification recorded 656 comparisons; 161 exceeded the frozen
absolute-error cap .002 (116 state, 45 projected-delta). Maximum absolute error
was .021484375. Relative L2/cosine and all 16 gradient comparisons passed, and
same-GPU repeats were exact. The absolute criterion still makes this profile
**FAILED**. Its thresholds remain unchanged. The failure concerns runtime
agreement and says nothing about donor usefulness.

The unchanged CPU reference completed `alignment-cpu-v2-002` capture in
**491.5613612 seconds**, with **4,450,623,488 bytes peak RSS** (4.45 GB / 4.15 GiB).
All four streamed Supa sites matched direct canonical hooks exactly. Captures
cover 96 fit and 32 held-out scenarios, without truncation or generation.
The activation bundle SHA256 is
`8d132b3992a8b7599926d692e1ea776772b6fa5186e5a8a57ac3f3382dc5f872`.
Metadata verification checked completed receipts, registration, partition and
bundle identity. The CPU registration ID remains
`f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea`.
Track B's top-level receipt field `registration_sha256` is instead the hash of
the expanded runtime registration, including its path:
`a87248479cb5e98025c080f09c05e4ead533b7769195f6c4a2634f1d76f0d6b6`.
These are distinct digests in the preserved caller, not different registrations.

There are no new held-out donor transformation metrics, selected bridge, gate
study or efficacy outputs. The historical delta cosine .443202525 and 16/16
identical graft/baseline outputs remain historical evidence of mechanical viability.
The forthcoming complete-matrix selector requires ordinary delta cosine≥.60,
improvement≥.10 over same-pair B0 and both null controls, token-relative MSE≤.8 B0,
all-token cosine≥.50 and a residual RMS ratio in [.25,4]. No partial matrix can
select a winner. Only a subsequent separately frozen architecture stage could
study doses [.03,.10,.20,.40]; Track B generation and confirmation remain closed.

[Power planning](data/generated/v2/power_planning.json) uses assumptions, not
observed effects. At N=1,024, a 2-point paired gain has approximately 81.7%, 52.6%
or 29.9% power when discordance is 5%, 10% or 20%. Corresponding 80%-power minimum
detectable effects are 1.96, 2.77 and 3.92 points. Relative-error planning assumes
a hypothetical 75% baseline. Diagnostic variants add no independent N, exact
tests become discrete with few discordances, and multiplicity further reduces
sensitivity. A 1,024-scenario bank cannot guarantee detection of a 1–2-point gain.

The CPU reference remains FP32/eager, four threads, deterministic and no-cache.
The authorized NVIDIA RTX 4070 Laptop GPU profile uses FP32, disabled TF32,
highest matmul precision and deterministic CUDA; CPU/GPU science is kept separate.
Resource interruptions and failed admissions are retained in fresh directories.
The latest independent byte check at 22:44 UTC verified all 198 historical files
unchanged and reread canonical checkpoint bytes matching
`a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`.
Historical inventory root SHA256 remains
`b2b4986d6f3abc151ab0c7ec8748c421c919dbcf648afa526e0005d4bdcbfa8f`.
No canonical or donor checkpoint was saved or trained. Sealed/TEST2/private data
remain closed. Source/config/data commitments are checked before model execution.

Validation includes the earlier combined **321 tests and 148 subtests**, followed
by 54 focused runtime checks, and the amendment's **215 tests and 148 subtests**
in 76.71 seconds. The 49 new calibration/integration tests passed in a separate
20.76-second focused run. Counts describe overlapping suites and are not summed.
The independent runtime review checked 206 invariants; separate NumPy arithmetic
reproduced all 40 Native logit comparison triples from 80 saved hash-verified
tensors. The new standard-library-only calibration audit passed **3,012 checks**:
all 128 case identities, scalar aggregation, partition separation, source/data
hashes, control identity and recorded module-integrity fields reconcile.

Exact amendment suite command recorded in its immutable receipt:

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' -m pytest experiments/supa_adaptive_compute_v2/native_armr/tests experiments/supa_adaptive_compute_v2/tests -q --junitxml=experiments/supa_adaptive_compute_v2/00_audit/native_control_v2_3_validation/junit.xml
```

The focused run covered `test_control_calibration.py` and
`test_calibrated_protocol.py`. The earlier combined suite's exact invocation is
retained in `00_audit/gpu_validation_v1/validation_receipt.json`. These are saved
validation commands: a future invocation must use a new output path.

Principal evidence paths, all relative to this programme directory:

| Evidence | Exact location |
|---|---|
| Audit/reuse and requirements | `00_audit/existing_work_inventory.md`; `00_audit/requirements_audit.md` |
| Latest historical/checkpoint verification | `00_audit/integrity_20260905T2244/receipt.json` |
| Original Native GPU qualification | `native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e/` |
| Original AM failure | `native_armr/evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0/` |
| Complete original architecture inventory | `00_audit/native_architecture_failure_20260905/` |
| Calibrated scalar and all cases | `native_armr/evidence/00_control_calibration/20260905T223621_219545Z_a6c46c00/` |
| Independent calibration review | `00_audit/native_control_v2_3_independent_review/` |
| Amendment tests and hashed logs | `00_audit/native_control_v2_3_validation/` |
| GPU implementation validation | `00_audit/gpu_validation_v1/`; `00_audit/runtime_independent_review_20260905/` |
| Failed Track B GPU qualification | `cross_scale/evidence/00_runtime_qualification/20260905T193023_051203Z_29be300a/` |
| Completed CPU capture | `cross_scale/evidence/01_alignment_matrix/alignment-cpu-v2-002/` |

Limitations remain material. These banks are bounded synthetic English tasks;
grounded QA does not establish broad factual knowledge. AM is untrained and a
slightly larger energy-matched alpha offers no demonstrated validity improvement.
The v2.3 design follows an observed architecture failure and cannot rehabilitate
the failed v2.2 matrix. Calibration saves hashes of raw tensors, so its independent
audit reconstructs scalar arithmetic rather than each underlying token energy.
Qualification timing was affected by other workloads and is not a controlled
speed benchmark. Alignment capture alone establishes no transfer fidelity. The
pre-existing `experiments/cross_scale_layer_graft_v1/` dependency remains untracked,
hash-bound and unchanged; a clean clone needs those preserved local bytes and
cannot attribute them to a source commit. Long CPU fitting and resource contention
remain practical constraints. Partial outcomes establish neither success nor
scientific failure of either capability hypothesis.

Exactly one next causal experiment per track:

- **Track A:** execute the separately registered pretrained-FFN-control v2.4
  campaign, first passing its full CPU/GPU qualification, then all five arms on the 128-scenario
  architecture set with unchanged guards. Only if both gates pass, run its
  untouched 1,024-scenario paired confirmation and separate linked diagnostics,
  followed by complete independent audit and the registered statistical hierarchy.
- **Track B:** execute the frozen CPU 16-pair/six-objective alignment matrix using
  verified `alignment-cpu-v2-002` captures, then apply its complete held-out
  mechanism gate and deterministic selection. The first pending pair is S8/D20;
  no efficacy generation or dose selection is part of this experiment.
