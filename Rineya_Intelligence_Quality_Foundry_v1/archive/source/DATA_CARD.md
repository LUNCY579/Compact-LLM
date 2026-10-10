# Reviewed structured-context research pilot

Current release: `releases/pilot-v3/release_manifest.json`.
Status: **REVIEWED_RESEARCH_PILOT; training_ready=false**.

## Purpose and composition

This original synthetic pilot exercises exhaustive dataset supervision for Rineya.
It contains72 distinct parameterized worlds, twelve in each of six families:
authorized revisions, relational aggregation, lexicographic precedence planning,
finite affine rule learning, trusted instructions versus quoted documents, and
grounded answerability.48 worlds are TRAIN,12 CAL and12 exposed DEV. No sealed
CHECK/FINAL data was constructed or inspected.

Worlds are reserved before construction, rendering and solving. Each row has one
public system/user problem and one compact JSON final-answer target. Public facts
are structured JSON rather than realistic prose documents. Private world and
reasoning metadata are kept in canonical audit records and excluded from the SFT
export. Author-created synthetic world provenance is declared CC0-1.0; no imported
source text or benchmark question is included in this pilot. This does not clear
rights for future outside sources.

## Measured lengths

All lengths use the exact local Titan tokenizer and bound non-thinking chat
template, target and terminal EOS. No padding, packing or target truncation.

| Authoring profile | Rows | Complete sequence tokens | TRAIN rows |
|---|---:|---:|---:|
| short |24|278-372|18|
| medium |24|1486-1981|18|
| long |24|3188-4429|12|

All72 rows together contain138,026 complete input positions and790 supervised
completion positions.24 complete sequences fit CE512 by length alone. The48-row
TRAIN candidate includes longer unqualified contexts. It must not be sent as a
drop-in CE512 training dose. Length-fit checks do not grant native training
eligibility or permission to substitute into the existing frozen E1 study.

Tokenizer0.22.2/Transformers4.57.1 and Python3.11.15 were used offline with model
backends disabled. No model weights were read. Tokenizer, template, source and
policy identities plus per-row input/completion/label hashes are recorded in
`profiles/pilot-v3/`. Token arrays are derived during profiling, rather than
published as a native trainer-ready data pack.

## Quality evidence

Every row passed independent public-prompt reconstruction, solution and strict
target verification.432 applicable mechanical mutation checks passed, including
wrong/duplicate-key/extra-key/type-confused answers, public/private mismatch and
answer-changing evidence controls. The independent adversarial reviewer additionally
executed240 applicable invariance/evidence controls across2,075 public records.

Every row has an assigned semantic and adversarial review receipt with an
independently derived public-input answer, concrete record evidence, exact row and
packet hashes, method and limitations. Reviewers inspected compact computations
for all72 configurations; every archive record was parsed/type/scope checked
programmatically. This is not calibrated human prose grading of every sentence.
The adversarial agent had previously seen source conventions and aggregate audit
results; its executable review received public packets only. Both agents used
the inherited model. Their independence is procedural rather than epistemic.

There are39 passing stdlib tests for authoring, independent verifiers and release
faults. The release controller was challenged separately using deliberately
fabricated fixtures; those synthetic receipts are tagged as non-reviews and never
supply genuine row-review evidence. Original flawed controller outputs and their
source snapshots remain in `evidence/`.

## Important limitations

Longer rows mostly add explicitly out-of-scope archives. The active reasoning
problem is bounded and the scope field makes filtering straightforward. These rows
are correctness and scope-handling controls, not a broad long-context intelligence
corpus. Larger active evidence horizons, naturalistic documents, low-overlap
associations and deeper causal reasoning are priorities for subsequent data work.

Only six shared task templates are present. Unique normalized active worlds do
not establish graph-isomorphism novelty, unseen-structure transfer, broad language
diversity or absence from pretraining. CAL/DEV are exposed instance-development
banks; they cannot supply fresh independent confirmation. Historical semantic
overlap is NOT_AUDITED. No old-skill replay bank or actual baseline source-correct
preservation enrollment was produced here.

No native training, model acquisition, retention or intelligence improvement was
measured. Those claims require separately qualified before/after student behavior
and parent/family-level uncertainty. Local hashes detect ordinary drift but cannot
authenticate execution or reviewer identity against a malicious artifact author.

## Version history and use

pilot-v1 was materialized before the controller red-team review. pilot-v2 repaired
four publication gaps; pilot-v3 repaired two remaining typed-metadata gaps. Their
canonical row/public-packet bytes are identical. They are the same72 worlds, not
216 independent examples. Original reviews are reused with complete byte-equality
provenance, while mechanics and tokenizer profiling were rerun for each producer.

Use `releases/pilot-v3/sft_train.candidate.messages.jsonl` only as a reviewed
research candidate. It has48 entries with the sole top-level key `messages`.
Each entry has intended system/user messages and final assistant target. Separate
lineage, canonical, audit and review files support admission; never append private
oracle/specification columns to actor prompts. Preserve this version and create
new immutable outputs for later changes.
