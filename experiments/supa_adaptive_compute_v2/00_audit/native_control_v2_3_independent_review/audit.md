# Native control v2.3 independent calibration review

The saved calibration artifact passes this independent audit. The 96 fit scenarios yield the fixed, unclipped scalar **alpha = 1.091369554461779**. All 128 cases, the registered 96/32 partition, source/data hashes, scalar aggregates and recorded module-integrity checks reconcile. This establishes calibration-artifact consistency; it does not establish generation safety, efficacy or GPU runtime qualification.

Audit time: 2026-09-05T22:41:56.875540+00:00. Evidence: `experiments/supa_adaptive_compute_v2/native_armr/evidence/00_control_calibration/20260905T223621_219545Z_a6c46c00`. Recorded execution commit: `4791635ac1a1ad22511c355bc59ab42956407dee`; this review did not run Git or independently check commit status. All audit code uses Python's standard library, imports no torch or programme comparison helpers, and performs no model/GPU execution. It writes only this new audit directory.

## Independent arithmetic

The audited source specifies per-token relative energy as mean squared delta divided by `max(mean squared post-L13 hidden state, 1e-8)`, calculated in float64 from float32 inference. It averages ordinary tokens within a scenario and then gives each scenario equal weight. The independent audit starts with the saved per-scenario ordinary-token measurements and reimplements the scenario means and fixed square-root formula without calling the calibration helper. Ordinary-token counts do not weight the scenario mean.

| Quantity | Fit | Holdout |
|---|---:|---:|
| Scenarios | 96 | 32 |
| Ordinary tokens | 7,069 | 2,338 |
| Native relative energy | 0.09559580669801247 | 0.09606493426847566 |
| Unscaled control relative energy | 0.08025926419711565 | 0.08041215831685194 |
| Scaled control relative energy | 0.0955958066980125 | 0.09577791697352757 |

`alpha = sqrt(0.09559580669801247 / 0.08025926419711565) = 1.091369554461779`.

The recomputed means, scenario/ID lists, token counts, all three RMS aggregates, alpha and scaled-control energies match the report exactly using the protocol's float arithmetic. `fit_scale.json` matches the fit-only aggregate and alpha exactly and matches its registered report hash. Search, clipping and holdout selection are all false in the config/report and absent from the inspected formula implementation. Alpha exceeds one and is retained unchanged; the audit does not infer a safety improvement from its magnitude.

An additional 80-digit Decimal calculation from the same saved binary floats differs from the report's alpha by one binary64 ULP (`2.220446049250313e-16`), consistent with intermediate rounding. It does not replace the registered floating-point result. All/ordinary/special strata also reconcile from their recorded scalar means/RMS values within the audit's predeclared relative tolerance `1e-12` and absolute tolerance `1e-14`. This is scalar consistency checking, not a reconstruction of raw hidden tensors.

## Identity, coverage and integrity

Frozen registration: `experiments/supa_adaptive_compute_v2/native_armr/evidence/00_control_calibration_registration/cf1440a9bbf3344bcc0cca316a1d5f9cb172a448b06bf1696b27646fc673fe57/registration.json`.

Registration ID: `cf1440a9bbf3344bcc0cca316a1d5f9cb172a448b06bf1696b27646fc673fe57`. Registration file SHA-256: `629857b141fe7729b8f95c8d6f09ddd1d7c8b4ee9ac37e1688ab5ff71822950f`.

The registration contract digest, all 143 registered source hashes, and the three config/dataset files match current bytes. Started/final manifests and the report bind that same contract. All 128 expected case filenames and byte hashes match; every prompt/example/scenario/family/template/partition identity matches the frozen original Track A calibration file. The 96 fit and 32 held-out scenarios are disjoint and follow the registered scenario-index modulo-four rule. No scenario was omitted. The full/retained token lengths match, ordinary-token counts agree with the saved 26-ID exclusion set, and all diagnostics are finite and nonnegative. No case includes answer, scorer, generated-response, correctness or NLL fields. The report declares zero generated tokens and no answer-label use, consistent with the inspected prompt-field projection and execution path.

The 16 layer receipts cover ordinary prefix L0-L13 and replayed L12/L13 exactly. Together with rotary and control receipts, all 18 have equal before/after state and buffer digests, zero trainable parameters and zero recorded parameter gradients. Replayed L12/L13 hashes match their corresponding prefix layers. The rotary receipt explicitly includes its two nonpersistent frequency buffers. Separate buffer-gradient fields are not present; buffer preservation is established here only by the saved digests.

The unchanged 31,457,280-parameter control hash is `eedbc0ca788ee8a1193b455f7cf2c94cd1b5eeab288f1f2df448fa9ae208584c`, matching both CPU/GPU identities in the original native runtime qualification. Canonical checkpoint and sidecar hashes, model config and installed HF implementation hash match the preserved CPU reference receipt. These are cross-receipt checks: this audit did not reread checkpoint weights, rebuild modules or inspect live gradients.

## Fit-before-holdout order

The inspected source computes fit transformation measurements, writes `fit_scale.json` exclusively, then evaluates the held-out transformations. Prefix hidden-state capture for all prompts occurs earlier; the held-out cases do not participate in the alpha formula.

Filesystem timestamps corroborate the intended write order:

- Last fit case write: `2026-09-05T22:37:18.187780+00:00`.
- Fit-scale creation/write: `2026-09-05T22:37:18.190781+00:00`.
- First held-out case creation: `2026-09-05T22:37:18.323359+00:00`.
- First held-out case write: `2026-09-05T22:37:18.324358+00:00`.

Every fit case write precedes the fit-scale creation, and its write precedes every held-out case creation/write. **These mutable filesystem times are not cryptographic timestamps.** The fit-scale content hash binds the scalar/fit values but does not prove historical execution order by itself.

## Evidence and limits

Report SHA-256: `20a552138f91ee37c3df261f576fc7ae4f0ff6214ddc9cdb8b383ff4d2e2789c`.

Terminal manifest SHA-256: `7dc2641160138247840be5775b3b1dbde5fa98a89724206c7b5cc0abb571abd6`.

Fit-scale SHA-256: `59d157281ad3f69fbd084149685c108f2efb5eb3eca98fb4a586a12a284ff897`.

The audit records 3,012 checked invariants with zero failures. `receipt.json` contains the recomputed values, source/case hashes, individual filesystem times and limitations. `audit_calibration.py` is the standalone audit implementation; it creates its receipt exclusively.

Raw hidden, native-delta and control-delta tensors are represented by hashes only, as are the captured embeddings/masks/positions/RoPE tensors. Consequently, this review independently recomputes scalar arithmetic from recorded per-scenario measurements; it does not independently verify each underlying per-token energy or rerun model computation. The engineering calibration was designed after the original AM architecture failure, cannot rehabilitate that failed v2.2 matrix and provides no capability evidence. Any derived v2.3 GPU configuration still requires its separate registration and CPU/GPU qualification.
