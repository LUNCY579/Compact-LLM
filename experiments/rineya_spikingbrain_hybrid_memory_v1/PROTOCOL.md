# SBHM-001 scientific and execution contract

## Registered question, not a success declaration

Can replacing one source attention site with local + gated recurrent memory
preserve useful behavior after a bounded conversion/recovery opportunity, while
removing that site's growing full-history KV allocation?

H1: D retains useful distant evidence beyond C under matched exposure.
H2: D's converted-site state remains bounded with sequence length.
H3: recovery preserves enough original capability to justify broader conversion.
H4 (later, separate): an optimized implementation improves actual cost.

H2 is a mechanics property, not evidence for H1/H3/H4. A reduction in distillation
loss or activation sparsity does not establish any behavioral improvement.

## Four controls

A: exact frozen source, no updates.
B: original architecture, same-site LoRA, self-teacher distillation + LM recovery.
C: local replacement, same-site LoRA, attention distillation + LM recovery.
D: local + recurrent replacement, same-site LoRA/new memory parameters,
attention distillation + LM recovery.

B/C/D share rank, initialization of LoRA factors, seed, consumed rows/order,
steps per stage, input/target tokenization and optimizer. Keep teacher source
frozen and separate from all candidates. Report added trainable parameters and
unequal elapsed compute; equal data exposure is not equal compute. The supplied
comparison verifies exposure rather than claiming all resources are matched.

No layer search, automatic threshold tuning, full-model finetuning, teacher KL,
MiMo dataset mixing, recurrence-depth changes, quantization, or model promotion
is bundled into this first pilot. The one-site/config default is an engineering
starting point. Site selection and eventual acceptance margins must be frozen
on new development/calibration evidence before protected confirmation.

## Data and scoring

Use fresh, explicitly unprotected TRAIN/DEV with disjoint family IDs and
reviewed near-duplicate separation. Include ordinary capability/retention and
long-context families: exact lookup, multiple facts, distant evidence joins,
updates/contradictions, identifiers/numbers, distractors and missing evidence.
These families are a corpus specification, NOT a claim that the eight-row
engineering fixture implements a sufficient benchmark.

Record context/evidence positions, exact source data revisions, creation process,
answer contract and missing-evidence semantics in a separately reviewed corpus
record. Current row schema is intentionally small. Add richer reviewed metrics
in a new code revision/qualification rather than pretending normalized string
match is semantic evaluation. Do not touch existing CHECK/FINAL, sealed, spent,
ANY_READ or confirmation instruments. All native outputs here remain DEV.

Collect per-row predictions, output token IDs, stopping, cache bytes, timing,
CUDA allocated/reserved peaks and sampled RSS. Compare repairs and damages
against A with paired row identity. Predeclare task-specific retention margins
and uncertainty analysis before later confirmation. One seed and repeated
inspection do not establish a general quality winner.

## Execution gates

G0: syntax, scoped tests, random fixture pipeline. Never a native capability claim.
G1: exact source/tokenizer inventory + supported standard-Qwen3 tensor geometry.
G2: dated operator permission, environment pins, fresh memory and disk guards.
G3: source/arm/delta-bound native inference qualification at registered shapes.
G4: separately granted distillation/recovery; finite gradients/updates, explicit
    trainable tensors, data exposure receipts; no assumed training-fit guarantee.
G5: separately granted matched DEV evaluation, fresh qualification after updates.
G6: owner-reviewed decision, then independently registered larger/optimized study.

Current delivery establishes G0 only. G1-G6 are not established by this commit.
The 2026-10-03 follow-up resolves the skipped pinned-library integration and
checks Titan-Transition's locked source/tokenizer/config and tensor headers.
These read-only source checks do not replace G3 actual-source execution or G4
gradient/optimizer fit. Fresh resource guards remain required at every launch.
On 2026-10-04 the fresh memory-equivalent matrix completed G3-G5 on the exact
Titan-Transition source and admitted synthetic corpus. All 10 qualifications,
six matched 512-step adaptation stages and four DEV measurements completed.
Training fit is established only for the actual 42-259-token inputs used, with
the original thresholds preserved. H1 was not supported: D scored 171/224
distant rows versus C 172/224, no D-only repairs. This is descriptive DEV,
not G6 promotion, a general recurrence rejection or broad retention evidence.
An operator grant alone is not a native measurement. No gate can be lowered by
a fixture flag, public-source substitution, hidden data, or a skipped test.

Resource guards: 7.4 GiB free host RAM before load; 6 GiB during runtime;
4 GiB worker RSS ceiling; explicit CUDA:0 without CPU fallback; at least 1 GiB
free VRAM reserve. These are stop rules, not predictions of fit. Teacher target
cache is capped at 512 MiB with at least 1 GiB disk reserve. There is no native
host available in the delivery environment. Local CPU fixtures intentionally
call pure functions without granting the native CLI any bypass.

## Interpretation and preservation

An unfavorable result can distinguish conversion error, insufficient state,
poor layer placement, training opportunity, or slow reference implementation.
Do not label it a universal failure of recurrent memory. Conversely, a mechanics
PASS must not be called model improvement or lower energy.

This experiment is complementary to centralized evidence/procedure memory,
TAPS/depth recurrence, skill-composition/retention work and existing attention
studies. It has no authority to replace, merge, delete or deprecate those tracks.
Model-only and external-system improvements remain separate scoreboards.
