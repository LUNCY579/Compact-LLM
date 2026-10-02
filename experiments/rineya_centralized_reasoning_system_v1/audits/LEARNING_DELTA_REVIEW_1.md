# CRS-001 learning delta scientific review

Recorded 2026-10-02. Reviewer `/root/scientific_audit`; builder `/root/data_audit`.
Disposition: **CPU_ASSOCIATION_HELPER_ACCEPTED; PL-SCI-02 CLOSED FOR THIS API'S
ARITHMETIC SCOPE.** Native integration/measurement truth, qualification, causal
learning and promotion remain unestablished.

Fresh inspection covered raw `learning_delta.py`, its contract, all tests and
unchanged `atlas_v2.py`, independently of the builder's interpretation. The reviewer did not build
these files. Reviewer and builders share model lineage, workspace privileges and
evidence; this review is not independent ground truth or secret-label isolation.
Only stdlib test/in-memory fixture processes were used, with no Torch,
Transformers, tokenizers, models, datasets, holdouts or GPU calls.

| Frozen artifact | SHA256 |
| --- | --- |
| learning_delta.py | bfe7411b19725e141f5d2ed5f764525594c9f83f45a96530875dd4c02eff3ef8 |
| tests/test_learning_delta.py | 456ee6bb1aa3a3a655a83f0944e317982292c4c1443ad4d4f1baa0389c4cbf7f |
| contracts/LEARNING_DELTA_V1.json | a410ab373475168fb0b56a72146f6054e5a9c0247c01839478de69596c6254ad |
| unchanged atlas_v2.py | 403cec6b49c37b9b623ccf03c39d19c695cbe288ce50f9062e6e62644f078a57 |

The prior PL-SCI-02 counterexample let a row with candidate proportion .1 bind
declared gain .9 without establishing its baseline/endpoint. This API instead
requires actual bounded receipt bytes plus matching SHA256, complete typed
endpoint/item schema and an independently recomputed item numerator/scheduled
denominator. Baseline/candidate panel ID/hash, scorer/hash/version, endpoint,
scope and canonical parent/case/variant/family membership must match before any
difference. The gain is computed as candidate minus baseline; a declared gain is
only a consistency assertion. Duplicate JSON and parent-variant identities,
missing available items, boolean/nonfinite/out-of-range estimates and inconsistent
strict/semantic/format or operational-error scores are rejected.

The atlas adapter binds source/candidate checkpoint and scorer identities, the
chosen metric's candidate proportion/scope and exact derived gain to the unchanged
measurement-receipt API. Whole failed/unavailable/missing endpoints yield null
gain and null efficiency ratios; item failures count as zero success and not
abstention. Negative changes and paired repair/damage counts are retained.
Semantic accuracy and strict accuracy have explicit separate endpoint names.

Validation: all 12 learning-delta tests passed as part of the registered
interpreter's 24 reader/delta tests (.016 seconds total). Own separately authored
four-case endpoints gave baseline .5, candidate .25, gain -.25, 0 repairs and
1 damage. Altered scope, omitted scheduled item, NaN proportion and invented +.9
gain were independently rejected. No test receipts were published as native
observations. `native_execution_attested`, `qualification_granted` and
`promotion_allowed` remained false.

Remaining integration conditions are substantive. `expected_membership` is
optional for the standalone association utility; every future native runner must
supply a prospectively frozen immutable panel membership and separately verify
that panel bytes/scorer/checkpoint/evidence actually produced those scores.
Self-consistent caller metadata/bytes are not execution attestation. The numeric
estimate here is scheduled-item accuracy; a whole-parent mean or hierarchical
interval is a different estimand when variants per parent differ, so freeze the
weighting and uncertainty procedure before combining instruments. This helper
computes no confidence interval, calibration, retention noninferiority, training
dose causality, general frontier or validated prediction. Retention/transfer
endpoint association does not establish previously acquired-skill preservation.

No native rows, training result, promotion or canonical publication were created
by this review. Existing atlas/scorer versions and prior failures remain retained.
