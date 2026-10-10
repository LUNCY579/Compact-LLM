# One exact prospective qualification proposal — native blocked

The machine-readable specification is emitted as `NATIVE_QUALIFICATION_PROPOSAL.json`
beside the validated packet and binds its complete source manifest and registry digest.
It describes a **future**, separately implemented/reviewed native worker. No runnable native
payload or operator grant is included in this offline milestone; current dispatch always
refuses. Review of this specification cannot replace review of future worker bytes.

Source: published Titan-Transition FP32, SHA256
`a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567`.
Standard Qwen3 class, preserved tokenizer/config, no custom code, offset, router or
quantization. Runtime packages/interpreter are those pinned by the packet's source identity.

One cold-load attempt, 540-second internal wall and 600-second kill deadline, separately
bounded <=120-second cleanup. Maximum three load-begun attempts, all costs retained within
the one-hour qualification pool. Create LOAD_BEGUN before from_pretrained; reserve and
charge the attempt before entry. Refusal before loading is recorded; failure never refunds
budget or exposure. Shared OS handle plus temporary/central token leases, CRS owner
coordination and qualified native Job containment remain admission requirements.

Resource floors: host available >=12 GiB, GPU >=6 GiB, disk >=20 GiB; active host reserve
1.5 GiB, GPU reserve .5 GiB; CUDA fraction .78, measured worker-tree RSS <=10 GiB, new
output <=6 GiB. These are prospective conservative caps, not measurements of this lane.
Failure means BLOCKED, without lowering floors or killing another owner.

Exact mechanics body: first two canonical DEVELOPMENT worlds per repair family (eight
generations, 64 output-token cap, greedy FP32/eager, no thinking/TF32); two padding probes,
eight steps each padded and unpadded (32 forward requests); disposable LoRA using the fixed
registry configuration for two optimizer updates, accumulation eight (16 target-bearing
forward/backward microbatches); standard FP32 dense serialization and deterministic reload
readout: one identical teacher-forced prompt/target on the already merged dense model before
serialization and after reload (two additional forwards; no adapter-versus-merge bit-parity
assumption). Total registered generation/parity/training/reload forward requests are 562 before any
explicitly measured recomputation. Backward requests are counted separately. No J chain,
final row, historical protected bank or permanent published checkpoint participates.

Before updates, validate exact HF serialization and assistant labels. Require finite loss
and gradients, bound memory/wall/process-tree lifetime, and preserve source files. The
disposable result earns mechanics evidence only, never model quality evidence.

Measured load/update/serialization/generation costs must support the complete plan: 16
trials, 22,016 evaluation calls including selected-export DEV rechecks, plus the separately
allocated qualification/parity calls. A fifteen-hour evaluation pool gives about 2.45 s
per evaluation before overhead. Training needs 250 updates inside each 30-minute cap.
If the measured upper-time envelope cannot fit, keep the study blocked without changing
sample sizes, dtype, recipes, seed selection or decision thresholds.

Next gate: independent review of this offline packet/specification, a source-bound native
worker implementation and containment qualification, then an explicit human grant naming
an exclusive slot and attempt. No model has been loaded by this package.
