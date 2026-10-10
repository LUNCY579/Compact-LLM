# FP32 EOS ordering checker correction, 001

This is a post-terminal analysis correction for native-cpu-causal-replay-003.
It requires independent source review before the coordinator reconstructs a new
study report. It authorizes no model execution, retry, controller fitting or
endpoint selection. The original failed report remains primary-null.

## Preserved lineage and correction chronology

The dispatched old analyzer remains exactly
`bfe6a0405d0acb4a9217438da04ece00db8a53604c2a6e8573fbc46c29d43253`,
its tests remain `ae15b60e3550fa2b36e5d45e3f3491ed13aeac70185033ee9faf9d8b7dc38a10`,
and analysis plan001 remains
`6282ac717afc6575587d6c5615518441038d7c8db2f54fca7eb562f980cb6a11`.
All three, the old null REPORT JSON/Markdown, and the complete 27-file raw run
were copied into `versions/cpu_causal_replay_analysis_before_fp32_ordering_003`.
Its 32-file manifest SHA is
`a80b7538b86c52f6cf0d3da2fc42333386f48e818bf78cb29908c136924254a2`.

After terminal exit0, the author initially began this auxiliary checker edit in
the old analyzer. When the coordinator specified an immutable successor, those
offline edits were moved into the new filename and the old analyzer restored
byte-exact from the archive. No model ran under the temporary edit; neither the
dispatched native code nor raw receipts or outcomes were replaced. The old
tests and plan were never edited. Both old source/test hashes were checked again.

The successor is `cpu_causal_replay_analysis_fp32.py` SHA
`5a62452a6991b60fbf1562d9554b1cb780432e194bd95c14c7191872384e74a1`;
its new tests are `tests/test_cpu_causal_replay_analysis_fp32.py` SHA
`6b33b5e9be529fc2ca94653c2cb6150b163f5aae8a5eb76a4797274200f9f023`.
The successor still validates the actual old analyzer and plan in the dispatched
source closure, and its exported report records its own successor source hash.

## Why the auxiliary fixed threshold is unsupported

For EOS, the frozen worker computes FP32
`logsumexp(all_logits)-logsumexp(logits[151643,151645])`.
Canonical EOT151645 CE is computed separately with stable `cross_entropy`.
In real arithmetic the accepted-set mass loss cannot exceed single-EOT CE.
The two rounded implementations need not preserve that ordering: cancellation
depends on logit/LSE operand magnitudes, including a common additive logit shift
that does not change the real probabilities. The frozen endpoint/FD plan does
not register an absolute `1e-5` cross-kernel ordering error bound.

The old analyzer rejected nominal/plus/minus EOS values on
`program_state:8:v1`, both in saved case and target triplets. Their raw signed
mass-minus-EOT gaps are respectively `2.3126602172851562e-5`,
`2.4080276489257812e-5`, and `2.4199485778808594e-5`.
These three observed gaps are not repaired, clipped, offset-corrected or deleted.

A no-model CPU numerical reproduction used Torch2.12.1+cu126, four CPU threads,
FP32 and 151936 synthetic logits with the same expressions/accepted IDs.
The synthetic distribution has competing logit0, canonical EOT logit-.75,
alternate EOS logit-80, and other logits-40, with a common additive offset.
At offset0 both losses are `1.1368709802627563`; at offset2048 mass loss is
`1.136962890625` versus unchanged stable CE `1.1368709802627563`, a positive
gap `9.191036224365234e-5`. Fixed offsets128/256/512 also exhibit scale-dependent
rounding. This demonstrates that the old bound is invalid in general. It does
not establish the cause or a tight rounding bound for the actual native case:
actual logits and LSE operands were not logged.

## General correction and unchanged gates

Every EOS row now carries `UNVERIFIED_FINITE_PRECISION`, the raw mass/EOT values,
all three signed gaps, and the original `gap>1e-5` flags as descriptive metadata.
The report retains all192 EOS rows and summarizes observed positive gaps and
legacy-check exceedances. No new enlarged threshold or ordering PASS is used.
The complete status explicitly includes `EOS_ORDERING_UNVERIFIED`.

All registered native finite, exact EOS-set, source/runtime/state/parameter,
event chronology, full population/count, causal-prefix, INPUT_ONLY feature,
RMS/resource, forecast and qualification guards remain required. Target-level
and per-case mean FD tolerances remain `.001+.03*abs(adjoint)`; neutrality and
numeric sign-uncertainty rules, all192 cases/96 parents, equal weights and
canonical answer mean/sum endpoints are unchanged. EOS is still a separate raw
accepted-stop surrogate; canonical EOT CE is descriptive. There is no composite
primary, semantic correctness, on-policy termination or quantified EOS-accuracy
claim. Unlogged-operand ordering is explicitly unverified even on rows with a
nonpositive observed gap.

## Author fixture verification and review boundary

The successor's 37 standard-library fixture cases pass in5.46s using the existing
interpreter with `-B` and `-p no:cacheprovider`. They retain the prior mechanical
regressions and add a representable-logit shift counterexample, all-row precision
metadata/raw-value preservation, invalid FD/EOS IDs/missing or nonfinite EOT,
missing counts/source proofs, and exact primary/EOS endpoint invariance versus
the archived checker when its auxiliary threshold does not reject. AST checks
preserve the proof, chronology, ranking, numerical helper and vector functions.

These are author validation, not independent acceptance. The author has not
generated a study report with this successor. Independent source/fixture review
and a new exclusive coordinator report are still required. The frozen raw run,
failed report and scientific endpoints remain unchanged.
