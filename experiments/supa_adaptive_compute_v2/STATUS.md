# Programme status

**GPU continuation:** the user has authorized GPU acceleration. Separate GPU
FP32 configurations and runtime qualification are implemented and preregistered.
See [GPU_STATUS.md](GPU_STATUS.md) for current execution. The CPU-only admission
record below is preserved as the earlier stage, not the current resource strategy.

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
