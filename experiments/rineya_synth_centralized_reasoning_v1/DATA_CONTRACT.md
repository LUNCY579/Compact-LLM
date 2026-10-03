# Proposed data, leakage, and evidence contracts

These are requirements for future implementation, not an audited corpus or an
implemented schema. `validate_manifest.py` checks registration metadata only.

**2026-10-03 addition:** [CPU_COMPILER.md](CPU_COMPILER.md) records the implemented
bounded R/S schema, source/record/export boundaries, and CPU audits. The broader
data-production requirements below remain the founding contract. No tokenized
native training corpus is admitted and no process-supervision verifier exists.

## Required record separation

| Record | Required contents | Access |
| --- | --- | --- |
| Source/world | source ID/version/hash, provenance, license/attribution, complete dependency-family ID | Compiler and audit; permitted excerpts only to student |
| Task lineage | row ID, parent family, role/split, skill signature, variant intervention, source dependencies, generator revision/seed | Compiler/export policy; not accidental prompt tokens |
| Student example | allowlisted messages and explicit target representation | Student, subject to role and tokenizer checks |
| Verification | independent verifier version, outcome/reason, checked certificate and intermediate states, public evidence references | Train acceptance or isolated evaluation, never gold leakage |
| Attempt/cost | accepted and rejected attempts, teacher calls/tokens, wall time, stop reason | Audit/cost ledger |
| Evaluation | prompt hash, checkpoint/decoding binding, prediction, reference outcome, prior-correct status, parent family | Isolated scorer and reporting only |

An export function must construct model inputs from an allowlist, not delete a
few known sensitive fields from an otherwise unrestricted source dictionary.
Oracle targets, verifier objects, private teacher hints, test labels, and future
state must never enter deployed model input. A target visible during supervised
training is not evidence that the model receives it at evaluation.

## Family-level isolation

Assign splits before amplification. All paraphrases, counterfactuals, renamed
entities, alternate renderings, and explanations of a parent stay together.
When tasks depend on several sources, connect their provenance dependencies and
split the entire connected component; a shared seed source cannot silently cross
the boundary through a different row ID. Record exact duplicates and structural
near-duplicates. Grouping is not proof of freedom from all contamination.

The proposed first study permits only R/S atomic TRAIN data and reserves RS.
Inspect generated task semantics, not just a generator-supplied skill label.
Do not use existing exposed test/challenge rows as fresh confirmation. Do not
open or incorporate the central CHECK/FINAL banks. DEV-informed selection must
be followed by new, unexposed confirmation under a frozen policy.

Procedures, retrieval indexes, teacher prompts, controller settings, verifier
versions, and acceptance thresholds selected on TRAIN/DEV are frozen before
evaluation. Test-source indexing is allowed only when the arm explicitly permits
it, with instance-local state and recorded cost; answer-derived state is not.
Evaluation must not write to shared memory or update the procedure library.

## Grounding and answerability

For R, verify reference chains, ambiguity, missing links, and cycles. For S,
verify exact cardinality and target sum; enumerate valid subsets independently
of a candidate's chosen solution. Accept any genuinely valid answer under the
frozen task contract; do not reject correct alternatives merely because a teacher
selected another. Validate every supervised intermediate state when present.

Distinguish solved, proven impossible, insufficient evidence, contradictory
sources, invalid format, and execution failure. Abstention format is a pinned
task contract, not a casual change to historical Rineya UNKNOWN conventions.
The first atomic acquisition pilot need not mix these categories into training;
answerability supervision is separately controlled when introduced.

## Importing released SYNTH data later

No released data is downloaded by this package. Before an import, pin exact
revision and shard hashes, inspect source-license metadata and attribution,
sample for factual/trace errors, check overlap with evaluation sources, reject
incompatible model-identity/self-description examples, and filter for the intended
skills and length limits. Preserve the released card's terms and source metadata;
this note makes no independent legal determination about all upstream content.

Do not treat all published synthetic examples as verified because our future
pipeline uses a verifier. Record whether data is reused, regenerated, or newly
constructed, and compare its total cost under the appropriate accounting.
