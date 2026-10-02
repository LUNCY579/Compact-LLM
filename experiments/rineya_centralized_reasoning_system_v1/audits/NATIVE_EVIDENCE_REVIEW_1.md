# CRS-001 native D2 worker evidence review, operational revision 2

Recorded 2026-10-02, America/Chicago. Reviewer `/root/scientific_audit` independently
read saved raw execution, witness, source and accounting artifacts. This reviewer
did not launch or reload a model. Shared model lineage/host privileges remain
disclosed; byte/source joins are not independent hardware attestation.

Run: `results/d2_17c01965_epoch001_rev2`, source commit
`17c01965b048a130f82095e2177d626449d0de4a`, contract SHA256
`291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707`.

Disposition: **RECORDED NATIVE WORKER NUMERICAL OBSERVATION; WHOLE SUPERVISED
PATH FAILED AND NOT QUALIFIED.** The scientific observations below are retained;
the supervisor's failure cannot be rewritten as a successful complete launch.
Reader576, capture, training, efficacy and protected-bank gates remain closed.

## Immutable reviewed bindings

All paths below are within this run directory in the managed
`crs-native-desktop-v2` worktree's CRS experiment package.

| Artifact | SHA256 |
| --- | --- |
| `DIAGNOSTIC_RESULT.json` | `b9a09927bf34e1cd2bcea0dacb9444b5e0508a4477b75ff15ae408469f1c3337` |
| `MODEL_RECEIPT.json` | `a3407d1534b8b2d2ddb1c3e888d2ce9498655307d807c7d1b051afc6fa0ca8d0` |
| `RUNTIME_RECEIPT.json` | `a5cf29a521f9374356a383b8b9a8807ee7e027f9fd71f86780b3b97608f934ba` |
| `CONTAINMENT_RECEIPT.json` | `4b79f4599dfec9bd7a9e862d68a2fb541a06940f16d2ed4bd0728fa0fdf993fd` |
| `EVENTS.jsonl` | `3c7db7b53ca71b7105516a13c49dbded708a5896c9ae30fb35abf46c74cc5a4b` |
| `SUPERVISOR_RESULT.json` | `c2f90ce741a6dbc9f32dc27837a6b3f5242daf7645f9f8b2d2a1b181fff5bffc` |
| `LAUNCH_INTENT.json` | `723000389227f09090e50cdfbbb9b55aad946e5d664e883ecfa2df3c8a52cf79` |
| `WATCHDOG.jsonl` | `e95bc8b505c8374c93c8a7650248e133fa0064e16c5a9535882daa2e55d3aefd` |

The launch intent binds clean source at exact 17c01965 and the frozen registered
environment. Runtime receipt matches registered Python 3.11.15, Torch
2.10.0+cu126, Transformers 4.57.1, tokenization/asset support packages and CUDA
12.6. GPU is reported RTX 4070 Laptop. Exact source/runtime/asset checks are
performed by the committed worker; this review joins their logged identities
and checks without re-executing the physical model.

## Native worker facts supported by raw records

MODEL_RECEIPT binds Titan weight SHA256
`537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`
and the unchanged config/tokenizer/generation/vocabulary assets. The streamed
2 MiB loader records 310 serialized tensors and 310 loaded tensor digests,
BF16 dtype, no full CPU model and zero trainable parameters. One model construct
and one pretrained checkpoint load are returned.

The native worker records `COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY`. That status in
the pinned producer is reached after the frozen parameter/no-gradient/dtype/
loaded-tensor digest checks and final source/asset checks. A separate post-run
GPU tensor dump is not present; the unchanged-base fact is supported by those
in-process checks and code/control flow, not a second model run by this reviewer.

Independently replayed event accounting found exactly 316 unique contiguous
call IDs, each with one prior attempt and one matching return, and no failed or
pending operation: one construction, one load, 310 tensor copies and four model
forwards. Forward stages are baseline, mutate_suffix_128, mutate_suffix_129 and
mutate_suffix_130; attempted/returned input positions both total 524. Training,
backward, optimizer, new generation and teacher calls are zero.

The four EVENTS measurement payloads exactly equal the four saved trial records.
Baseline hidden/logit hashes and original-target logprobs match the frozen
historical **failed bulk endpoint**. This is reproduction of that endpoint, not
agreement with prefix likelihood:

- Logits: `88cc954c4360789ab7847248e1d454401a5cb4023f3b2e8df6bb6b3dd05c7891`.
- Hidden: `4e3b28f16426ea847f95bc40ba5b862470b09784f621fd86ae93766c38f9c278`.
- Selected original-target logprobs:
  `[-0.46126696467399597,-0.3731182813644409,-0.0006275591440498829]`.

## Causal witnesses and non-vacuous suffix interventions

All three planned future suffixes were recorded at absolute starts 128,129,130;
their protected rows are respectively 127, 127–128 and 127–129. Six protected
comparisons, three each for full-vocabulary logits and normalized hidden, record
matching left/right raw hashes, raw identity true, numeric equality true and
maximum/mean absolute differences zero. Original-target logprobs are identical
on their protected prefix positions.

The full last-four-row hidden/logit fingerprints change for all three mutant
trials, while the protected prefixes remain identical. Recorded nonprotected
target logprobs change for starts 128 and 129; start 130 changes the fourth row
outside the three target-scoring rows. Thus the recorded controls are not
explained by identical whole output arrays for every intervention.

For each forward, passive-hook counts are mask=1, rotary=1, norm=1. Replayed the
complete BF16 little-endian 131×131 mask representation independently from its
words: allowed 0 and forbidden `0xff7f`; all four actual/reference mask hashes
are `492bdc6b45ccd2fe917c2ffbfa189c050c9c150c8f9f601c45d52e29be87a028`.
Replayed the complete int64 little-endian position array 0–130; all four hashes
are `949c3ef67313a22a57f13c1c324bfcd5547cd0ad5b9dacf3e086a30b613a78d1`.
Their shapes/dtypes and explicit positions agree with the registration.

Runtime flags are retained: TF32 fixed false; BF16/FP16 reduced precision and
split-K reduction flags available/true; FP16 accumulation available/false.
The receipt states no reduction flag change. These identify the observed
registered runtime condition; they do not identify a kernel cause for the
original prefix/full discrepancy.

## Supervision outcome, cost and resource limits

Supervisor record is **FAILED** for `NoSuchProcess: process PID not found
(pid=626604)`, despite observed worker-tree launcher exit code 0 and eventual
owned-tree empty proof. The actual runtime worker is PID 630392; exact owned Job
flags 0x2000 and query-handle closure before framework are recorded. These are
distinct worker/scientific and supervisor/operational outcomes. Operations
review must establish the completion-race repair; this evidence review does not
silently convert the old run to success.

Recorded resource measurements for this exact four-forward envelope:

| Measurement | Recorded value / scope |
| --- | --- |
| External peak process-tree RSS | 1.362274169921875 GiB, sampled by the supervisor |
| Worker CUDA peak allocated | 1.400038242340088 GiB |
| Worker CUDA peak reserved | 1.4296875 GiB |
| Worker minimum sampled GPU free | 5.404296875 GiB |
| Worker minimum sampled system available | 5.932464599609375 GiB |
| Worker wall | 31.718000000000757 seconds |
| Charged qualification worker/supervisor elapsed | 34.56199999999808 seconds |

The outcome observation binds the known exited tree and the 34.562-second charge.
Together with the retained earlier 1.109-second pre-framework failure, these two
attempts consume 35.67099999999846 seconds. Neither failure nor restart resets
the cumulative budget. Sampled peaks/floors support the exact observed envelope;
they do not establish sustained training, a 576-token decoder or unobserved
sub-poll transients. No extra model run was used for this audit.

## Bounded interpretation and next gate

The recorded worker execution supports a trace-local observation: under the
registered fixed 131-token shape, selected future-token mutations leave the
specified earlier normalized hidden/logit rows bitwise invariant. It is useful
native numerical evidence even though the whole supervised path failed.

It does not prove universal causal correctness, prefix/full shape invariance,
GEMM/attention mechanism, likelihood-equivalence repair, long-context attention,
reader efficacy, autonomous procedure discovery, training/internalization,
retention or calibration. Historical capture still fails its original
0.03125/0.01 limits; native/capture/reader/training qualification flags in this
saved worker result remain false.

Preserve every artifact, failure and cost under 17c01965. A fresh operational
successor may be reviewed on its own source/contract and tested within remaining
budgets after current admission. It must leave numerical/model/math checks
unchanged and must not erase this supervisor failure. This is the final allowed
repair round for the milestone; any unresolved supervision claim remains
quarantined instead of accumulating unbounded retries. No protected bank or
new dataset was accessed in this evidence audit.
