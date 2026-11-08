# Source-joined TRAIN compatibility with CE512

30 September 2026. AMENDMENT_013 requires actual source, token and label binding
before E1. The existing e1_data binder validates the frozen 5120 arrays and full
dose, but accepts sequences up to 1024 tokens and only checks the syntax of each
source-row hash. The new data-only contract closes that specific gap.

The prospectively frozen E1_CE512_TRAIN_SCOPE_V1 manifest pins 28 files: actual
TRAIN source and token files, measured compiler receipts, registrations, language
lineage/profile, unchanged historical data-binding config, model/tokenizer receipts
and the distinct CE512 runtime config/source. Verify exact file identities before
importing the pure deterministic generators. Never import tokenizer preparation
modules into a model worker: they deliberately disable model frameworks.

Join every acquisition and deterministic replay row to its complete canonical
source body, regenerate its semantic parent and TRAIN allocation, and independently
verify prompt and target. Preserve the 640 deterministic replay parents and 128
answerability twin parents. Join language arrays to complete curated conversations,
class/cluster and frozen lineage/profile. Language allows one conversation with
multiple turns; all prior assistant turns remain inside the masked prompt, and
only the final assistant reply plus EOS is supervised. Do not equate multiple
turns within one registered conversation with sequence packing.

Keep all 5120 rows, 680532 input positions and 63546 supervised positions. Enforce
one row per microbatch, at most 512 total tokens, zero padding, no packing/custom
positions/cache, validated all-one masks, full-sequence logits and the unchanged
mean-per-sequence completion CE followed by the mean over 20 sequences. No target
truncation, data selection, model/tokenizer execution or dataset rewrite occurs.

Stream source and array lines with a 64 KiB line ceiling and bounded compact
identity maps. Metadata is capped at 2 MiB. The actual CPU audit must measure
execution time and peak process memory against its prospective 600-second and
128 MiB ceilings. Optional output is write-once, contains compact byte bindings,
and never stores a second copy of all source text or token arrays.

A successful data contract establishes local source-to-array compatibility,
not independently rerun tokenization, authenticated execution, human language
review, complete pretraining exclusion, native qualification or learning gain.
Language original TRAIN origin and prior quality selection rely on the frozen
earlier provenance audit; current curated source and lineage are joined again.
The original native 512 save/resume trial remains pending RAM admission. Every
native, training and E1 authorization flag stays false. CHECK/FINAL remain closed,
and current best stays unchanged Titan. The full research objective is unchanged.
