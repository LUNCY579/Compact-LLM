# Amendment 015: diagnose the captured likelihood disagreement

30 September 2026. The first frozen short-greedy capture pilot loaded exact Titan
weights and returned all 318 selected calls. Its first fixture emitted tokens
16,22,151645, visible text 17 and natural EOS. Single-prefix selected model
log-probabilities disagreed with full-131-token teacher forcing: errors
0.0414324105, 0.0373841524 and 0.0000814913, mean 0.0262993514. Both original
0.03125/0.01 bounds failed. Fixture384 was unattempted; final base/source checks
were not reached. Preserve the failure, incorrect answer and all partial work.

Memory checks passed: maximum internal sampled RSS 1.404602 GiB, minimum sampled
host headroom 2.554611 GiB and GPU free 5.400391 GiB. The external 100 ms RSS
observation can miss the instantaneous internal sample. Final result, cost ledger
and journal show 318 calls; the earlier failure receipt contains stale initial
zero-call accounting. Preserve both and use final accounting as the authority.
A future failure writer must snapshot current accounting before publication.

Prospectively register CAPTURE_GEOMETRY_DIAGNOSTIC_V1.json on the exact failed
prompt128 and generated IDs. There are no new generations, datasets, adapters,
optimizer updates, teacher calls or grading. Preserve the original capture config,
producer/checker byte pins, numerical thresholds and raw receipts unchanged.

First reproduce the original three single-prefix keep=1 model calls and full
131-token keep=4 teacher-forced call. Then obtain final hidden states independently
for prefixes128/129/130 and the full131 sequence. Compare a two-by-two grid of
prefix-derived versus full-derived hidden values and single-row versus four-row
LM-head projection. Hold each hidden tensor's values fixed when changing its
projection layout; use the common fourth full-last row only as a disclosed layout
row, without treating it as an additional generated target.

Repeat those held-hidden head projections with a GPU-only FP32 upcast of the tied
BF16 weight. This is a diagnostic comparator, not an FP32 base, new capture policy
or recipe promotion. Record hidden hashes/differences, selected logits/logprobs,
argmax IDs and full-vocabulary logit error summaries. Layout effects and upstream
hidden-context effects are distinct; an upstream difference does not by itself
identify attention as the cause. Canonical per-prefix replay agreement cannot
relabel the failed bulk gate or authorize capture.

Retain the registered 2 GiB RSS cap, 1.25 GiB reserve, GPU4/0.5 GiB and disk4 GiB
floors, 600-second ceiling, Windows containment and one GPU worker. Allow at most
ten model/base forwards and sixteen separately journaled head projections. Preserve
partial/failed diagnostic calls and final base/source identities. No qualification,
DEV/source enrollment, language grading, long-context or E1 grant follows from a
diagnostic success. The full research objective and unchanged best remain intact.
