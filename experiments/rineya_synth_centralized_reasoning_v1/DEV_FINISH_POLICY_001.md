# Whole-response correctness for EOS and generation-cap finishes

The [frozen finish policy](DEV_FINISH_POLICY_001.json) uses the same strict answer
metric for both EOS and the registered 32-token cap. A cap-ended response may
pass if the entire decoded generated-only response is a valid, correct JSON
answer. Missing EOS is recorded separately. Malformed JSON, extra text or an
incorrect answer fails; no extraction, repair or additional generation is allowed.
Use the verifier's exact Boolean `accepted` field, never the truthiness of its
returned dictionary.

This policy was selected after 192 completed M0 rows and failed batch04 were
collected. No actual responses have been graded. It is not claimed prospective
to those outputs; it is frozen before remaining M0 or any M1/M2 collection.
The choice measures completed answer correctness without making finish reason
an additional capability requirement. All rows stay in the fixed denominator,
and finish-stratified reporting cannot select or change the primary policy.

Exact compact002 input/order, verifier source, statistics, pilot and importer
hashes are bound. This resolves the finish-rule choice only. Native scoring is
still closed: the complete audited M0 population, qualified actual M1/M2 study
checkpoints, all three arm identities, scoring qualification and a complete
source/receipt-bound authorization remain absent. Qualification checkpoints
cannot be substituted. The pre-score importer remains unable to grade native
records; its earlier unregistered-policy rejection is retained until a complete
native scoring registration is reviewed.
