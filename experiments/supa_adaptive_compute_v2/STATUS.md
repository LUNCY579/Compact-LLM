# Programme status

2026-09-05. GPU acceleration is authorized and in use for Native ARMR after
track-specific CPU/GPU qualification. Track B's GPU profile failed its numerical
qualification; its unchanged CPU reference completed calibration capture.
The scientific questions remain open. No confirmatory generation has run.
The consolidated evidence report is [CURRENT_REPORT.md](CURRENT_REPORT.md).

Latest execution checkpoint (2026-09-06 00:00 UTC / September 5 local): Track B
has two complete depth pairs, S8/D12 and S8/D16, each with all six objectives.
Native v2.3 qualification **PASSED** in a fresh process at
`native_armr/evidence/00_runtime_qualification/20260905T235200_982255Z_36998242/`.
Its earlier attempt `20260905T233146_952002Z_9d317bbe` failed RAM admission before
model loading (2.94 versus 3.00 GiB). Attempt `20260905T233921_233855Z_41832612`
completed all 40 CPU reference fixtures, then failed with CUDA unknown error
before GPU model loading. All 40 raw CPU logits remain preserved, but the attempt
is FAILED and no passing qualification report exists for it.
Windows recorded an NVIDIA TDR reset at 23:42 UTC during that attempt;
`00_audit/gpu_driver_events_20260905T2347.json` preserves the events. A fresh
CUDA-process health probe passed afterward (`00_audit/gpu_health_20260905T2344.json`).
This is consistent with a lost CUDA context; the reset's initiating cause is
not established. The successful retry used unchanged source/config/fixtures and
thresholds. All 40 CPU/GPU comparisons, greedy identities, exact GPU repeats,
disabled/restored checks and state identities passed. Max absolute logit error
was 4.1961669921875e-5; max hidden relative L2 7.856709228698431e-5; max absolute
NLL error 4.553794860839844e-5. Peak CUDA reservation was 2,688,548,864 bytes;
qualification wall time was 313.05 seconds. The independent admission validator
recomputed all frozen gates and checked all 80 logit artifact hashes. Receipt
and report are committed at `39122c92c9fa930ebab99a90135ae92aa06c4f7b`.
An additional independent NumPy audit reproduced all 40 logit metric triples
from all 80 tensors and passed 120 numerical gates; 40/40 CPU logit files were
byte-identical to the failed TDR attempt, which remains FAILED. That audit did
not independently rerun hidden states or NLL:
`00_audit/native_v2_3_gpu_independent_review/`.
The fresh GPU architecture matrix is now executing in registered order
A0, AM, A1, A3, A16, with original evidence excluded by registration identity.
Queue: `native_armr/evidence/00_execution_queue/20260906T000050_6403813Z_434bb308/`.

| Track | Question | Mechanics | Candidate selection | Large evaluation | Gate |
|---|---|---|---|---|---|
| Native ARMR | Does bounded recurrence generalize? | v2.3 CPU/GPU qualification PASS; fresh architecture matrix running; original AM failure preserved | Label-free v2.3 AM scale frozen; all five arms still required | 0 confirmation examples | CLOSED pending a valid complete architecture matrix |
| Cross-scale | Can a donor transformation transfer faithfully? | GPU numerical qualification FAILED; CPU reference capture `alignment-cpu-v2-002` COMPLETE | S8/D12 and S8/D16 completed all six objectives; 14 pairs pending, none selected | Not licensed | Complete 96-cell alignment mechanism screen first |
| Routing | Are useful operators complementary? | Out of scope | — | — | CLOSED |
| MLA/MHA/MTP | Other adaptive mechanisms | Out of scope | — | — | CLOSED |
| 8B / distillation | Escalation / compression | Out of scope | — | — | CLOSED |

Native evidence audit: 128 valid cases per A0/A1/A3/A16 arm; AM has 84 valid
cases plus one case preserved in a failed shard. All five interrupted A16
case outputs/NLL/logit bytes match its separate successful rerun. The interrupted
receipt remains invalid. The audit passed 3,521 checks and found no recorded
confirmation artifacts: `00_audit/native_architecture_failure_20260905/`.

Track B capture completed in 491.56 seconds at 4,450,623,488 bytes peak RSS.
All four streamed Supa capture sites matched direct canonical hooks exactly;
the activation bundle covers 96 fit and 32 held-out scenarios. Its SHA256 is
`8d132b3992a8b7599926d692e1ea776772b6fa5186e5a8a57ac3f3382dc5f872`.
The metadata-only verifier rechecked the bundle hash, source registration,
frozen dataset, partition counts and complete receipts. No bridge or generation
result follows from capture alone.

S8/D12 completed in 128.65 seconds at 3,768,635,392 bytes peak RSS; donor state
was unchanged and native replay was exact. Held-out ordinary-token delta cosine
was B0=0.32144, B1=0.64488, B2=0.46854, B3=0.79367, BS=0.04241, BR=0.03288.
B3 token-relative MSE was 0.37049 versus B0=1.39826. These are descriptive
measurements from one pair; no eligible/selected-candidate decision is made until
all 16 pairs complete and the independent selection audit passes. Evidence:
`cross_scale/evidence/02_bridge_objectives/alignment-cpu-v2-002/S8_D12/attempt_000/`.

S8/D16 completed in 111.50 seconds, peak RSS 3,769,110,528 bytes. B3 ordinary-token
delta cosine was 0.73081 versus B0=0.27899; token-relative MSE was 0.46297 versus
48.51333. BS/BR cosine was 0.11282/0.04988. Both pairs preserve donor state and
exact native replay. These are two partial-matrix results, not a selection or
mechanism GO. The next fixed pair is S8/D20.

Generic future-stage export/provenance and frozen graft interfaces are prepared
under `cross_scale/generation_stage/v1/`. The suite passed 56 tests and 6 subtests
in 21.86 seconds using tiny CPU decoders and synthetic/mocked receipts. Actual
export requires complete committed mechanism GO plus its independent PASS audit;
no real bridge export, generation protocol, dose choice or model caller exists.
The old 139 Track B and 143 Native-calibration source hashes remain unchanged.
Validation: `cross_scale/generation_stage/v1/evidence/00_preparation_validation/20260905T2347/`.

The original AM guard failure is an engineering rejection, not evidence for
recurrence efficacy. Its arm, scale, source registration, partial outputs and
failure receipts remain intact. The separate v2.3 label-free calibration rule
was frozen in registration `cf1440a9bbf3344bcc0cca316a1d5f9cb172a448b06bf1696b27646fc673fe57`
and committed at `4791635ac1a1ad22511c355bc59ab42956407dee` before execution.
Calibration completed in 62.75 seconds, peak RSS 1,236,889,600 bytes, with all
state/buffer/checkpoint identities preserved and zero trainable parameters or
gradients. Report:
`native_armr/evidence/00_control_calibration/20260905T223621_219545Z_a6c46c00/report.json`.

The exact fitted scalar is **1.091369554461779**, slightly above the original
1.0. Fit relative residual energy is 0.0955958067 for A3 and the scaled control.
Held-out energy is 0.0960649343 for A3 versus 0.0957779170 for the scaled control.
The formula was applied without search or clipping; held-out measurements did
not select alpha. No accuracy, answers, architecture outputs or confirmation
examples entered fitting. Energy matching is not generation validity or efficacy.
The verified calibration artifact was committed at
`a6126f2a3c51bba5d312ae6db2f7cb4af1092522` before deriving the new GPU config.
Fresh runtime qualification and a complete fresh architecture matrix remain
required; original generation guards are unchanged.

Current execution registrations:

- Native GPU v2.3: `86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44`.
- Original native GPU v2.2: `7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5`.
- Track B CPU reference on current shared source: `f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea`.
- Failed Track B GPU profile: `beec1869aad324d29191cd21e5f300bd6da3bdc0a52d8c5f11839ac4b98f5b7c`.

Validation before the v2.3 amendment: **321 tests and 148 subtests passed** in
88.05 seconds, followed by **54 focused tests** after runtime metadata changes.
The independent runtime review checked 206 invariants. Independent NumPy
recomputation reproduced all 40 native logit comparison triples from 80
hash-verified saved tensors. Evidence is in
`00_audit/gpu_validation_v1/` and
`00_audit/runtime_independent_review_20260905/`.
The amendment additionally passed **215 tests and 148 subtests** in 76.71 seconds;
49 new calibration/integration tests also passed in a focused 20.76-second run.
Exact commands and hashed logs: `00_audit/native_control_v2_3_validation/`.
The calibration artifact verifier reconstructed all 128 scenario aggregates,
the fit-only formula, frozen partitions and state/buffer preservation checks.
An independent standard-library review passed 3,012 checks and reproduced the
scalar exactly from saved per-scenario measurements. It does not claim a model
rerun or reconstruction of raw tensors: `00_audit/native_control_v2_3_independent_review/`.
At 22:44 UTC, a fresh streamed SHA256 check found all 198 historical files and
the canonical checkpoint unchanged: `00_audit/integrity_20260905T2244/receipt.json`.

See [GPU_STATUS.md](GPU_STATUS.md) for qualification measurements and receipts.
All full-model jobs execute sequentially under a machine lease; engineering
and independent audit run concurrently across tracks. No current capability
or cross-track cost frontier follows from runtime qualification or an incomplete
architecture matrix.

## Earlier CPU-only delivery snapshot (preserved)

The records below describe the initial delivery, before GPU implementation and
the executions above. The original CPU registrations do not authorize newer
source revisions. [FINAL_REPORT.md](FINAL_REPORT.md) is also that earlier snapshot.

2026-09-05. Implementation and independent validation completed. Both banks and
independent registrations were committed at
`067fa33b2ed78fc2a0f9361d313629549f214f08`. Execution was attempted sequentially;
both attempts stopped at the registered physical-memory admission guard before
model loading. No v2 model evaluation or new donor transformation metric exists.

| Track | Question | Mechanics | Candidate selection | Large evaluation | Gate |
|---|---|---|---|---|---|
| Native ARMR | Does bounded recurrence generalize? | PASS in tiny-model/guard tests; real qualification pending | A0/A1/A3/A16/AM frozen | 0 examples; memory admission failed | Await resources, then complete architecture qualification |
| Cross-scale | Can a donor transformation transfer faithfully? | PASS in tiny-model/guard tests; real capture pending | Six objectives across 16 pairs frozen; none selected | Not licensed | Capture memory admission failed; alignment first |
| Routing | Are useful operators complementary? | Out of scope | — | — | CLOSED |
| MLA/MHA/MTP | Other adaptive mechanisms | Out of scope | — | — | CLOSED |
| 8B / distillation | Escalation / compression | Out of scope | — | — | CLOSED |

Historical A3 13/16 versus A0 12/16 remains hypothesis-generating. Historical
aligned 4B graft 12/16 with 16 identical outputs remains mechanical viability
only. Their directories have not been merged or rewritten.

Validation: **234 tests and 148 subtests passed** in the combined new/historical
suite; compilation passed. The 198-file historical source/evidence inventory is
unchanged, root SHA256 `b2b4986d6f3abc151ab0c7ec8748c421c919dbcf648afa526e0005d4bdcbfa8f`.
Exact command and source digests: `00_audit/validation_20260905T1727/validation_receipt.json`.

Each track has 1,024 original confirmatory scenarios, 128 linked diagnostics,
128 architecture cases and 128 calibration cases (96 fit/32 holdout). All targets
pass independent semantic validation; tokenizer-only validation found no cap
violations. Track B calibration max length is 129 tokens under its 256-token cap.

Frozen registration identities:

- Native: `a2bb36ff710ba95b371e6e4dbb1c74840477beca050226d8678f5c334d43964f`
- Cross-scale: `99d73b899bdc057ab8d3e61e528841a2254714348b1e2b8f1ed79eacc36df9b8`

Both are stored separately under each track's `evidence/00_preregistration/`.
Execution verifies that the registrations, textual protocols, source, config and
bank manifests have been committed before loading a model. The actual model
execution state will be updated from immutable receipts, not inferred from tests.

Sequential real execution admission attempts:

- Native A0, architecture shard 0/32: **FAILED before model loading**, available
  RAM 1.70 GiB vs registered 3.00 GiB; zero completed examples. Receipt:
  `native_armr/evidence/01_architecture_dev/20260905T173248_727906Z_9ed863b1/run_manifest.json`.
- Track B capture `alignment-v2-admission-001`: **FAILED before model loading**,
  available RAM 1.80 GiB vs 3.00 GiB; no activation/bridge artifact or generation.
  Receipt: `cross_scale/evidence/01_alignment_matrix/alignment-v2-admission-001/run_manifest.json`.

Both processes are terminal; the machine lease was released. Limits were not
changed and failures were not overwritten. These are resource failures, not
scientific NO-GO outcomes. The full research goal remains open. Exactly one next
causal experiment per track is specified in [FINAL_REPORT.md](FINAL_REPORT.md),
with resumable commands in the programme and track READMEs.
