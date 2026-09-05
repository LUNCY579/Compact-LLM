# Programme status

2026-09-05. GPU acceleration is authorized and in use for Native ARMR after
track-specific CPU/GPU qualification. Track B's GPU profile failed its numerical
qualification; its unchanged CPU reference completed calibration capture.
The scientific questions remain open. No confirmatory generation has run.

| Track | Question | Mechanics | Candidate selection | Large evaluation | Gate |
|---|---|---|---|---|---|
| Native ARMR | Does bounded recurrence generalize? | GPU runtime PASS; A0/A1/A3/A16 completed 128 architecture cases each; original AM failed a malformed-generation guard | Original matrix incomplete; prospective label-free AM scale calibration under development | 0 confirmation examples | CLOSED pending a valid complete architecture matrix |
| Cross-scale | Can a donor transformation transfer faithfully? | GPU numerical qualification FAILED; CPU reference capture `alignment-cpu-v2-002` COMPLETE | Six objectives across 16 pairs frozen; fitting pending, none selected | Not licensed | Alignment mechanism screen first |
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

The original AM guard failure is an engineering rejection, not evidence for
recurrence efficacy. Its arm, scale, source registration, partial outputs and
failure receipts remain intact. A separate v2.3 amendment will calibrate only
the frozen random control's scalar using Track A calibration activation energy.
The formula will be preregistered before calibration; no accuracy, answers,
architecture outputs or confirmation examples select the scalar. A replacement
configuration requires committed calibration evidence, a new registration,
fresh runtime qualification and a complete fresh architecture matrix.

Current execution registrations:

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
New amendment tests and provenance checks must pass before its execution.

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
