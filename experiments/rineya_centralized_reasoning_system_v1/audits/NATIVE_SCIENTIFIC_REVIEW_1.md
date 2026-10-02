# CRS-001 fixed-shape native diagnostic scientific review

Client date: 2026-10-01, America/Chicago. Reviewer: `/root/scientific_audit`.
Milestone: separate native diagnostic review; the prior CPU milestone does not
qualify native execution. Initial disposition: **REPAIRS REQUIRED**, followed by
a fresh artifact closure before scientific diagnostic approval.

## Scope, access and evidence

Read CRS `native.py`, `tests/test_native.py` and
`contracts/NATIVE_DIAGNOSTIC_V1.json`; original central Amendment 016, causal D2
configuration/source; raw failed capture and D1 geometry records; and the
original tensor-digest/runtime qualification helpers. No protected evaluation
labels, native model loads or CUDA tensors were used by this reviewer. A bounded
CPU BF16 signed-zero control used the registered Torch installation only.

The reviewer did not build the native worker. Model lineage, workspace and host
privileges are shared with the coordinator/builders. Independent here means a
separate inspection of raw artifacts and tests, not independent ground truth or
an OS-isolated observer. Operations/containment review is separately assigned.
This report does not grant authority beyond the owner's existing conditional
authorization and is not a native execution receipt.

Reviewed source began as mutable working files. Initial native scientific logic
was read at native SHA256
`cdd66c2fbc1085b6b12813c1c110d53fe0c81869fc79803c844f445b1e1b45c7`;
concurrent operations repairs later yielded
`d01bb7a6dc7599c481345d8aedb924314eec864420830093969af900f8d7097a`
while the identified scientific comparison/witness gaps remained. Initial CRS
diagnostic contract SHA256:
`7242ed791f009394bdcb0a26ab1bcac0d8418351e474ce542505453502da13d7`.
Final acceptance must bind repaired source, updated contract and committed tip.

Raw historical bindings freshly checked:

| Artifact under `experiments/post_training_mimo_centralized_v1/` | SHA256 |
| --- | --- |
| `results/headless_capture_qual_v1_20260930_1140/CAPTURE_RESULT.json` | `e669a6945520b7c91bb5d00382ceab2ac4ab5cdc8f09d9273ebd4915ae8017a2` |
| `results/capture_geometry_d1_20260930/DIAGNOSTIC_RESULT.json` | `b827f11c29e9a568f4baa37f618b05b9ee96956bd8a9b2cc87b6468634ee0808` |
| `configs/CAPTURE_CAUSAL_DIAGNOSTIC_V1.json` | `6db4eb3d02c6a0526720e3e1846533ccc48bf7a42214dfc5d05d38f1df29a11d` |
| `AMENDMENT_016.md` | `73708dc892af904ab0f26761a8c535ca2989c43921ccfc53d223a6692676ce3f` |

Original failed capture remains `BLOCKED_CAPTURE_QUALIFICATION`. Its historical
maximum/mean likelihood errors are 0.04143241047859192 / 0.02629935139945398,
above the unchanged 0.03125 / 0.01 limits. D1 exactly reproduced this gap and
localized it upstream of held-hidden output-head projection; it did not identify
an attention/kernel cause. These historical records were read, not re-executed.

## Scientific design checks that pass statically

The 128 prompt IDs and generated IDs `[16,22,151645]` exactly match the failed
mechanical trace. The baseline is the 131-token prompt-plus-generated sequence.
All four forwards use the same length, batch, dtype, attention configuration,
cache policy and output geometry; only the declared future suffix token values
change. This controls input/output shape while probing selected causal boundaries.

The worker retains the last four normalized hidden/logit rows. For length 131
these are absolute positions 127–130; the first three predict the originally
generated tokens at positions 128–130. The implemented protected-prefix counts
are correctly aligned:

| Forward | Mutated absolute positions | Protected rows compared against baseline |
| --- | --- | --- |
| Baseline | None | Frozen fingerprint over last four rows; original-target logprobs from 127–129 |
| `mutate_suffix_128` | 128,129,130 → 17,23,151646 | 127 |
| `mutate_suffix_129` | 129,130 → 23,151646 | 127,128 |
| `mutate_suffix_130` | 130 → 151646 | 127,128,129 |

Independent CPU plan replay confirmed every row length is 131, all preceding
tokens stay unchanged, each intended suffix token changes, and total planned
input positions are 524. Two tensor kinds for three suffix comparisons give six
protected comparisons. There are no decoding or optimizer calls in this design.

The CRS baseline expected fingerprints match raw D1 artifacts:

- Last-four BF16 logits:
  `88cc954c4360789ab7847248e1d454401a5cb4023f3b2e8df6bb6b3dd05c7891`.
- Last-four final-normalized BF16 hidden states:
  `4e3b28f16426ea847f95bc40ba5b862470b09784f621fd86ae93766c38f9c278`.
- Original-target bulk logprobs:
  `[-0.46126696467399597,-0.3731182813644409,-0.0006275591440498829]`.

Baseline agreement is an admission/reproduction check on the original failed
bulk endpoint, not proof that this endpoint agrees with prefix scoring. A
baseline drift should reject the diagnostic; it must not change expected bits,
runtime identity or tolerances after observing a failed run.

Static hooks check the first decoder additive mask has shape 1×1×131×131,
BF16 dtype, permitted entries zero and forbidden entries the finite BF16
minimum; rotary positions are exactly 0–130; normalized hidden output is
1×131×1024 BF16; all expected hook calls occur once. They return no replacement
tensors. Finiteness, unchanged-base digests, exact runtime/source assets and
four-forward/524-position/316-logical-call completion are checked. Shape-576
reader and training interfaces explicitly remain closed.

## NATIVE-SCI-01 — Numeric equality is not raw bit identity

**Priority P1 for the declared diagnostic rule. Evidence:** initial native
comparison calls `torch.equal(left,right)` and passes it as `bitwise` to
`numerical_decision`. That checks tensor element equality, rather than raw byte
identity. The [official torch.equal documentation](https://docs.pytorch.org/docs/2.10/generated/torch.equal.html)
describes equality of size and elements; dtype equality is not its contract.

**Reproducer:** using Torch 2.10.0+cu126 in the registered interpreter, construct
CPU BF16 tensors `[+0.0]` and `[-0.0]`. Observed torch.equal=true and maximum
absolute numeric difference=0, but original `tensor_digest` SHA256s differ.
The original numerical decision with zero max/mean and bitwise=true passes.
No model or CUDA tensor was loaded for this control.

**Impact:** a receipt may label a comparison bitwise invariant while protected
raw tensor bits changed. Fixed dtype/shape reduce other equality confounds but
do not eliminate this signed-zero counterexample.

**Verification:** compare raw bytes or protected raw-byte hashes at fixed
shape/dtype, record both identities, and require raw identity plus finite
maximum/mean differences of zero for every protected hidden/logit comparison.
Retain torch.equal only as a separate numeric diagnostic. A CPU signed-zero
negative and identical-byte positive must exercise the actual comparison helper.
No new native forward or widened tolerance is required.

## NATIVE-SCI-02 — Numerical witnesses and exposed math flags are not retained

**Priority P2; closes reproducibility before launch. Evidence:** initial worker
validates masks/positions in hooks but retains only counts as temporary values;
the published measurement lacks replayable mask/position witnesses. It also
omits the read-only BF16/FP16 reduction/split-K flag record required by original
Amendment 016 and implemented in the central D2 producer.

**Counterexample:** two environments share model/package strings but expose
different matmul reduction flags, or a failed protected comparison occurs. The
initial receipt permits no reconstruction of the observed mask/rotary witness
or the exposed flag state, even though these are relevant alternative explanations.

**Verification:** publish each forward's mask shape/dtype/hash, exact triangular
reference/sentinel and value verification; full rotary position IDs/dtype/hash;
hook counts and protected absolute positions. Write a runtime receipt identifying
the installed runtime/GPU/CUDA plus available BF16/FP16 reduction, split-K and
accumulation flags, marking unsupported names unavailable. Read these flags
without altering them. Keep mask validation and passive hooks; a receipt of
expected values alone cannot substitute for observed-tensor checks.

## NATIVE-SCI-03 — Bind historical endpoints as immutable source evidence

**Priority P2 provenance closure.** Initial contract copies exact reference IDs,
bits, logprobs and failed limits, which this review independently joined to raw
historical records. Its runtime source map initially does not include the raw
failed trace, D1 record and central D2 registration/amendment that establish that
provenance.

**Counterexample:** a later local edit to an original raw receipt leaves CRS's
copied constants intact; no machine source-binding check detects the changed
historical evidence, although the runtime claims to reproduce that original trace.

**Verification:** add the four original source/evidence paths and exact hashes
above to the frozen CRS source/contract receipt and verify them before/after
the run. Do not edit original evidence. Publish mutation-start/protected absolute
positions so indexing can be replayed without private conventions.

## Test observations and limits

Initial native CPU suite under the bundled workspace interpreter: 14 passed,
two dependency errors because that interpreter lacks psutil for newly added
supervisor fault fixtures. This is an environment/test dependency result, not a
model failure. Re-running all 16 tests under the registered
`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` passed, including
lost-launch-response and proven-before-spawn-failure fixtures. These tests create
no native worker, checkpoint or CUDA model. They do not qualify operational
containment; separate operations review governs that evidence.

Inference from any future successful native run must remain narrow: these
three chosen future-suffix interventions leave these protected hidden/logit rows
bitwise invariant at this exact 131-token trace and execution path. That is not
a universal causality theorem, an accuracy gain, attention-window qualification,
bulk-versus-prefix equivalence, or identification of GEMM shape effects. It can
rule out the tested fixed-shape future-content explanation within scope; shape
effects remain a competing explanation requiring a separately registered test.

No repetition/parameter sweep is required for this deterministic trace-local
mechanics question. A failure remains a failed diagnostic with costs and raw
evidence; it cannot tune the same frozen expected bits. Reader 512+64/576,
capture likelihood, language/long-context, sustained training, update/export/
resume, acquisition, retention and CHECK/FINAL remain separately closed.

## Fresh repair closure — SCIENTIFICALLY_APPROVED_FOR_BOUNDED_DIAGNOSTIC

The three findings are closed for the repaired diagnostic artifacts below.
Scientific disposition permits the planned four-forward diagnostic **only after**
separate operations acceptance, a clean committed source binding, exact review
receipt, fresh asset/runtime/host admission and lease/budget conditions. It does
not assert that the diagnostic has run or that any checkpoint is native qualified.

| Accepted artifact | SHA256 |
| --- | --- |
| `native.py` | `183a177417b81ea938e2476bb49e8e739dd0316e42479bd11f6b5818e5061782` |
| `tests/test_native.py` | `9a5ae99fd421906eeff4fe4fe316eca6d1433e1fcd13768545cc83a0f6e264f2` |
| `contracts/NATIVE_DIAGNOSTIC_V1.json` | `a57a7fa091aeda2a63e8b1452fdeed53474f627246a3f4429b62af924a45babe` |

These paths resolve within the CRS package. The committed tip is to be bound by
the coordinator's subsequent exact launch-review receipt; a different source or
contract hash cannot inherit this disposition.

**NATIVE-SCI-01:** protected comparisons now record left/right raw digests and
require raw identity plus finite zero maximum/mean differences. torch.equal is
retained as a separate numeric diagnostic. Independently replayed CPU negative:
opposite BF16 signed-zero hashes reject even when numeric deltas and a caller's
numeric-equality boolean are zero/true. Identical bytes pass. The actual worker
derives its identity flag from tensor digests at already checked shapes/dtypes.

**NATIVE-SCI-02:** each successful forward now publishes passive mask/rotary
witnesses and invocation counts. The mask's complete BF16 little-endian
representation has 8,646 permitted uint16-zero words and 8,515 forbidden words
65407 (`0xff7f`, the negative finite BF16 minimum). Its independently replayed
SHA256 is `492bdc6b45ccd2fe917c2ffbfa189c050c9c150c8f9f601c45d52e29be87a028`.
The hook checks both actual values and actual raw hash against that representation.
Rotary IDs include the entire 0–130 sequence, int64 dtype and observed raw hash.
Mutations/protected absolute positions are recorded per trial and comparison.
Runtime receipt records available/unavailable BF16/FP16 reduction, split-K and
accumulation flags without changing them; inherited fixed TF32=false is explicit.
Allocator allocated/reserved peaks and sampled minimum GPU free memory are
separately retained. These are planned/CPU-tested instruments; actual GPU
witness values remain unmeasured until the admitted run.

**NATIVE-SCI-03:** source pins now include the original failed CAPTURE_RESULT,
D1 geometry, Amendment 016, and the central numerical-identical D2 wait successor
registration `CAPTURE_CAUSAL_DIAGNOSTIC_WAIT_V2.json` at
`b254ec5b880e9911d4ca8ba7edc27665c63ca09cbfe66ffafde76cf025fb4c65`.
That registration preserves the original trace, four forwards, fixed masks and
bitwise scope; its resource-only wait is a separate historical operational
amendment, not a transfer of permission or native qualification to CRS. This
review independently verified all registered source/runtime file hashes. Weight
asset rechecking and actual current host admission remain the launcher's duties.

Fresh registered-interpreter native CPU suite: **23/23 passed**, reported test
interval 0.795 seconds. The suite includes an actual owned Windows stdlib
subprocess tree and exit proof, and fabricated launch-fault controls. It executes
no Rineya checkpoint, model forward, training update or CUDA model operation.
Operations' interpretation of those containment/recovery controls has its own
review. Static plus CPU scientific closure cannot substitute for physical GPU
diagnostic evidence.

Original bulk capture remains `FAILED_AND_RETAINED`; its 0.03125/0.01 limits and
known errors are unchanged. Passing all six future protected comparisons would
support only the recorded fixed-shape trace-local invariance. Shape-576 reader,
exact-prefix likelihood successor, training/resume/export/sustained fit,
capability/retention and all protected-bank gates remain closed. No GEMM/kernel
mechanism, universal causal property or model improvement is approved here.
