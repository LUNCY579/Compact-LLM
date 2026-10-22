# Mechanical source revision after packet 001

The 14-test source used to prepare packet 001 was followed by evaluation integrity
hardening before root coordination had established that preparation had completed.
This was a sequencing error. The source-binding check correctly blocks current
source against packet 001. No bypass or receipt overwrite is authorized.

Late changes to study.py: optional frozen review SHA check in load_review;
EVALUATION_FREEZE.json before review/private-join parsing; review-drift checks;
original-byte label snapshots instead of canonical JSON reserialization;
evaluation_freeze_id and frozen review source hashes in report metadata.
Late PROTOCOL.md change: section 7 documents those same integrity checks.
Late __main__.py changes: failed-command elapsed/resource cost metadata and
confinement of BLOCKED receipt writes to the assigned package runs directory.
Two tiny tests were added for review drift and whole-world bootstrap behavior
(14 -> 16 cases). README was updated to document evaluation integrity hardening.

No prepare logic, IDs, ordering, packet/group generation, RUBRIC, permitted labels,
annotation schema, disagreement bounds, scoring, aggregation or bootstrap endpoint
changed. No answer-aware change or outcome tuning occurred. Preparation should
produce byte-identical annotation.packet.jsonl and REVIEW_INSTRUCTIONS.md, but
this must be verified prospectively; it is not assumed. The source-bound freeze
and private PREPARE metadata necessarily change when code/protocol hashes change.

These create-only source snapshots were reconstructed from exact reverse patches
and verified against all four original INPUT_FREEZE source SHA-256 values before
writing. Current source remains unchanged. Snapshots are preservation evidence,
not permission to execute an old or new worker against a blocked host.

Root decision: keep current hardening; retain packet 001, reviews and source-drift
receipts. Once fresh CPU admission passes, prospectively prepare packet 002 with
current source in a new directory. Reuse completed reviews only after packet,
rubric/instructions and full blinded population are verified byte-identical, with
review packet-hash bindings unchanged. Then evaluate under current source and the
new freeze without adjudication, threshold changes, or annotation rewriting.
Hardware admission remains blocked; no study, Python worker or model execution
was performed during snapshot reconstruction.
