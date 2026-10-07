# Decoded long guard collection: CPU preparation

The staged producer uses the original long inference component for generated-only
greedy decoding, with synchronous per-token and final-row preservation through
the original journal. Prompt lengths 992/2016/4064 retain their 32-token reserve
inside context envelopes 1024/2048/4096. The per-batch forward and position caps
are declared before collection; no prompt is padded, shortened or truncated.

The canonical source order is world, length, then four answerability/adversarial
variants. A frozen index map selects 36 homogeneous batches of 64 rows, twelve
per envelope. The saved-raw consumer validates each capture batch's original
packet hash and then restores all 2,304 rows to canonical order. It retains
source indices, capture ordinals and raw packet hashes, preserving all 192 worlds
and externally declared domain membership. It does not rewrite raw hashes to fit
the older mixed-envelope importer.

The initial source passed four authored controls, including the complete
36-batch producer/saved-raw canonical join, 32-token cap behavior at 4096, failure
and tamper controls, and closed native entry. Source hashes remained unchanged;
three deliberate real-input probes were denied and no forbidden import occurred.
This initial receipt is preserved separately because review found native-audit,
journal-order, path and trace-type gaps that were not covered by those controls.
It is not the accepted corrected-auditor result.

The subsequent corrected source passed three focused controls covering the three
envelopes and rebound journal/type negatives. It closes native audit before reads,
uses bounded regular-file descriptors, verifies original causal journal order and
selected partial/final/batch references, and rechecks consumed bytes before
returning. Canonical partial comparisons reject Boolean/numeric substitution;
strict JSON rejects duplicate keys and nonfinite values. The final source also
requires the original journal's terminating newline; its targeted positive and
rebound missing-newline control passed. These are separate receipts rather than
a claimed full 36-batch rerun on the final source.

Eight independently authored controls pass on the final frozen source. They
replay representative batches at all three envelopes and reject rebound journal
order, source/plan changes, Boolean trace/partial fields, duplicate JSON keys and
missing final newline. Native audit refuses before reads, and synchronous token
sink failure stops the next forward/decode. The independent run monitors 3,763
reads with no forbidden imports or real-data/model/API access. It is a focused
review, not a second complete-population replay.

The final source and all predecessor versions are preserved under
`evidence/long_guard_collection_reviewed_cpu_20261007/`. Journal timestamps are
syntax-checked; awareness, chronological timing and native wall-clock coverage
are not established by this CPU consumer.

Native dispatch, a qualified owned asset loader, the decoded long guard worker,
native decoder binding and independent native capture qualification remain
absent. The existing TRAIN-only qualification worker and its NoDecode boundary
are unchanged. This producer takes already loaded assets from its caller; CPU
mocks do not establish fresh M0 identity, native tensor preservation, context fit,
scorer qualification, enrollment or retention. No native worker, model, training
or scoring is executed in this preparation.

Read-only CIM/vendor-CLI inventory at 2026-10-07 01:21:25 UTC observed 941 MiB GPU
free memory, below the unchanged 4,096 MiB floor, and 12,183,846,912 bytes of host
available memory. This volatile snapshot is not pinned supervisor admission.
Qualified language ratings, complete scored M0 and the matched learning/transfer
comparison remain outstanding.
