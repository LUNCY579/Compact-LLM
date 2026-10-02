# Learning tokenization scientific instrument review 1

2026-10-02. Reviewer `/root/scientific_audit`; builder `/root/data_audit`.
Disposition: **REPAIR_REQUIRED_FOR_MEASUREMENT_RECEIPT_ASSOCIATION; ACTUAL
MEASUREMENT REMAINS CLOSED.** This is round1 of the new instrument milestone.
One consequential finding is demonstrated below; no speculative feature or
model experiment is requested.

I read the complete frozen source, tests and contract, accepted TRAIN32 source/
registration and both tokenization scope reports. I did not build this instrument.
I previously reviewed the bank and root's positional reference; that involvement,
shared model/workspace privileges and supplied grammar limit independence.
Only stdlib and explicitly synthetic token/observer adapters were exercised.
No actual tokenizers/Jinja/backend/framework/model/psutil, subprocess measurement,
host database, protected labels, holdout or training was used. No source was edited.

| Frozen artifact | SHA256 |
| --- | --- |
| learning_tokenization.py | 2d7ad4a0d752b7ca205530a8672986b3cc3c3152857e0f54715ad3d17c11b3b0 |
| tests/test_learning_tokenization.py | bdb9ad308d5447f44a8a8bd7b44d95f5f634c7d2d9cf3d93c90cb9dbacbb37ae |
| contracts/LEARNING_TOKENIZATION_V1.json | d17a60a72d00ff4f58e47d418b3a1fed85be10ce8cdd3dd4c93f793b9a688abb |

## Supported source behavior

The fixed instruction and compact public source/query/rules JSON omit case/parent/
family/split/oracle/gold/feedback from user content. Registered pairing binds
accepted TRAIN32, the four bank hashes, ordered IDs and canonical status/value
targets through the accepted target API. This is intentional TRAIN supervision,
ordinary answer adaptation, not procedure discovery or distillation.

Static inspection of the pinned template's single-user/final-assistant branches
is consistent with the manual renderer: user im_start/header/content/im_end/newline,
then assistant header and empty thinking block; assistant content is followed by
im_end/newline. Unsupported control sentinels reject. Actual Jinja parity has not
been executed here. Producer code requires exact manual/Jinja bytes and IDs,
special-preserving decode, whole training-sequence encoding, and prompt-prefix
ID equality. It explicitly removes exactly the last template newline in text,
retaining full render and training-text hashes instead of silently trimming targets.

Production causal_ownership and validate_ownership agree with the root-authored
reference11a1afda: prompt3/response3 produce labels
[-100,-100,-100,123,456,151645], shifted mask[0,0,1,1,1], target positions3/4/5,
logit positions2/3/4, teacher positions6 and final-EOS served prefix5. The prompt
EOS is ignored and response EOS owned once. All four independent negative vectors
were replayed against production helpers and rejected: prompt EOS leakage,
missing assistant EOS, wrong shifted ownership and a post-EOS newline. This is
synthetic positional validation, not tokenizer or numerical model evidence.

All21 supplied tests passed independently in0.598 seconds, zero failures/skips.
The fixture preserves all32 cases, including failed/oversized cases; no resampling,
truncation or UNKNOWN-answer fallback occurs. Encode attempts, including failures,
are capped at184 (24 controls and160 case calls). Successful teacher/supervised
positions have18432/2048 aggregate ceilings and512/64/576 per-case geometry.
Tokenization eligibility and native/training flags remain false.

The future worker/supervisor path is implemented: direct pinned base CPython,
deferred tokenizers CPU load and Jinja sandbox, fixed dependencies/assets,
source/reviewer/GO binding, observations and independent timeout, final live
kernel-peak witness, immutable raw result references and failure/unknown reporting.
It is not a permanently closed placeholder. Its current registration is explicitly
closed, so the frozen bytes authorize no real measurement. Future activation is a
separate immutable reviewed registration transition; preserve this closed contract
and closure test. Source review alone supplies no measured fit or resource result.
Detailed process/resource enforcement remains subject to independent OPS review.

## LTK-SCI-01: full-render returned ID/count association is incomplete

**Evidence:** Encoder records only hashes/counts for returned IDs. In
validate_measurement, successful manual_full/jinja_full calls are required to have
equal ids_sha256, but their token counts are neither compared nor replayed against
retained full-render vectors. Those vectors are discarded from the result; only
joined-training ownership IDs and prompt prefix can be independently reconstructed.
Producer measure_case compares its actual full ID lists, but the externally
validated receipt can contradict that producer invariant.

**Executed synthetic production-API counterexamples:** create the supplied
registered pairs and a successful SyntheticBackend fixture, then mutate a copy:

```python
pairs = token.registered_pairs(data.build_train32())
good = token.measure_pairs(pairs, SyntheticBackend(), token.manual_render,
                          expected_pair_sha256=token.digest(pairs), fixture=True)
bad = copy.deepcopy(good)
bad['encode_calls'][26]['token_count'] = 1  # first case manual_full
assert token.validate_measurement(bad, pairs, fixture=True)
bad = copy.deepcopy(good)
for index in (26, 27):  # manual_full and jinja_full
    bad['encode_calls'][index]['ids_sha256'] = '0' * 64
assert token.validate_measurement(bad, pairs, fixture=True)
```

Both assertions returned True. The first changes only one count while the paired
encoding keeps its original count and the same ID hash: equal ID sequences cannot
have those conflicting lengths. The second demonstrates missing retained-vector
association rather than proof of a tokenizer-truth attack. No actual backend or
measurement was invoked. The supplied tests mutate a control hash and joined
ownership, but do not cover these full-render records.

**Impact:** complete receipt replay may accept corrupted render-encoding metadata
while claiming manual/Jinja ID association. This does not demonstrate a wrong
actual tokenizer, wrong producer output or numerical model effect; it is a
consequential internal measurement/provenance gate defect before actual results.

**Verification criterion:** preserve/replay the existing calls' returned full ID
vectors (or equivalent immutable ID evidence) and bind their actual lengths/hashes,
manual/Jinja equality, and relevant render/training-sequence association. No new
encode, tokenizer, bank, model or larger limit is needed. Reject one-sided count
changes and both altered full hashes, missing/duplicate/invalid vectors and any
internally contradictory same-hash/different-length record; valid producer output
must still pass. Keep backend truth/authority distinct from byte consistency and
retain the existing output/encode/resource ceilings and failure denominators.

## Claim limits and next disposition

The corrected bank, prompt schema, template subset and causal label arithmetic
are scientifically appropriate CPU prerequisites. LTK-SCI-01 withholds acceptance
of the measurement-receipt instrument on these exact bytes. Obtain fresh frozen
SCI/OPS review after the bounded repair, within the three-round limit.

No real token IDs, Jinja parity,512+64 fit, tokenizer peak/time, model loss/gradients,
bulk-prefix equivalence, acquisition/efficiency/retention/calibration, training
resume/export or native/reader/probe qualification is established. Teacher N and
served prefix N-1 differ in shape; correct index arithmetic cannot repair the old
bulk numerical failure. Existing quarantines, source/bank histories and authority
limits are unchanged.
