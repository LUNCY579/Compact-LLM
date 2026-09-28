# Continuation 005: next-action scope audit

Date: 2026-09-28. **VERIFIED SOURCE inspection; the implementation recommendation
below is PROPOSED, not a native authorization.** The current cycle has produced
the language candidate export. Its tokenizer measurement, exclusions and neighbor
review are current-cycle work and must be completed and reported as such. This
audit did not rerun them, load a model/tokenizer, read protected payloads, make a
new memory measurement, or change goal status. It read implementation and existing
receipts and checked the existence/size of named open files; it did not open DEV
payloads or freshly rehash those data files.

The coordinator reports the fifth recurrence of the native host-RAM blocker. The
current turn nevertheless has real data-preparation progress. After its current
deliverables finish, **one concrete finite FAST task remains: bind the existing
open DEV files to the response-import contract and supply their actual deterministic
scorers.** This is a missing data-to-evaluation connection, not another generic
qualification wrapper. It cannot satisfy E0, native capture, calibrated language
grading, long-context runtime qualification or E1 execution requirements.

## Concrete gap found

`scripts/enrollment.py` validates a caller-declared manifest. Its required row
fields are `row_id`, `parent_id`, `family`, `stratum`, `variant`, `role`, `panel` and
`prompt_token_hash`, with optional `scoring_payload`. Its optional API scorer must
return strict `correct` and `malformed` booleans. The CLI calls `enroll` without
supplying any scorer. It does not read the frozen banks, join measured token rows
back to their source specifications, or choose an existing deterministic verifier.
Thus passing a declared row list does not yet establish that it is the actual
registered DEV population.

`scripts/e1_data.py` supplies the corresponding concrete binding for TRAIN only:
`validate_array` requires role TRAIN and training-role eligibility; `bind` fixes
three TRAIN files and training quotas. It must not be reused by relabeling DEV as
TRAIN or relaxing that gate. A scoped search of the package's scripts found no
other implementation connecting the DEV banks to enrollment's expected rows and
scorer callback. Language packet/rating code addresses a separate source-review
workflow and does not fill this gap.

The strongest practical reason for a real adapter is **reference leakage at the
policy-input boundary**. The bridge and deterministic-retention token files store
`input_ids` containing prompt **and verified reference completion**. A collector
must take only `input_ids[:prompt_tokens]` and verify its canonical hash against
`tokenized_prompt_sha256`. Passing the full stored array would expose the reference
answer. Long-context rows instead store prompt-only `input_ids`, and language
profiling uses `prompt_token_ids` with reference/control tokens in separate fields.
Those different layouts need an explicit, tested mapping rather than a generic
same-field loader.

## Existing inputs and usable pure semantics

Paths beginning `artifacts/` below are repository-relative; other paths are
relative to this experiment package. Sizes/counts/hashes in this table are recorded
in the named existing receipts, not new tokenizer or file-hash measurements.

| Panel | Fixed open inputs | Recorded population and concrete mapping |
| --- | --- | --- |
| Acquisition | `datasets/bridge_v1/DEV.jsonl`; `artifacts/mimo_central/bridge_tokenized_v1/DEV.tokenized.jsonl`; `receipts/BRIDGE_TOKENIZATION_RECEIPT.json` | 256 rows/256 parents, 64 per four families. Join token rows to source rows by ID and canonical `source_row_sha256`; source rows contain `spec`, generator metadata and `structure_id`, while token rows contain the measured prompt prefix/hash. Map integer `variant_id` explicitly to a stable string. A prospective mapping can preserve `structure_id` as the enrollment stratum without inventing new strata. |
| Deterministic retention | `artifacts/mimo_central/retention_v1/DEV.jsonl`; `artifacts/mimo_central/retention_tokenized_v1/DEV.tokenized.jsonl`; `receipts/RETENTION_TOKENIZATION_RECEIPT.json` | 256 rows/192 parents: 64 retrieval, 64 instruction-following and 64 two-row answerability parents. Preserve source `stratum` and the exact `standard` or `answerable`/`missing_evidence` variants; join back to source semantics before scoring. |
| Long context | `artifacts/mimo_central/long_context_dev_v1/DEV.jsonl`; `receipts/LONG_CONTEXT_DEV_DATA_RECEIPT.json` | 256 rows/32 parents, eight rows per parent. Prompt-only array hash is `input_token_ids_sha256`; stratum is `evidence_position`; importer variant must include length as well as `variant_id`, e.g. `2024-answerable_neutral`. Preserve both lengths, all four answerability/distractor cells and the disclosed two-mechanism limitation. |
| Language | `artifacts/mimo_central/language_dev_cal_v1/DEV.jsonl` and its export receipt; the current cycle's completed tokenizer profile when available | 64 DEV worlds; 16 CAL worlds and their controls are a separate grading-input population, never additional model-quality DEV rows. `scripts/tokenize_language_eval.py` returns prompt-only IDs/hash separately from illustrative reference tokens. Bind DEV identities and input bytes if useful, but language scoring remains BLOCKED and references must not be treated as exact answers. |

The named acquisition source and three deterministic token/data files exist at
these paths. The language export receipt also exists. This inspection does not
assert successful completion of the still-running current-cycle measurements.

Deterministic scoring can reuse existing **stdlib** modules:

- `bridge_data.allocate_parent` reconstructs and validates identity/role from the
  source specification. `parse_answer` enforces the JSON envelope, and
  `verify_answer` independently evaluates the supplied semantics without trusting
  a stored reference answer.
- `retention_data.allocate_parent`, `parse_answer` and `verify_answer` supply the
  analogous strict behavior, including the exact twin variant.
- `long_context_tasks.solve_visible` can recheck the bound visible context/query
  against the stored target; `parse_answer` and `score_answer` implement the strict
  case-sensitive symbol/UNKNOWN contract. Do not substitute the relaxed
  `diagnostic_answer` parser for the primary score.

Do not import `tokenize_bridge`, `tokenize_retention`, `build_long_context` or
`tokenize_language_eval` into this adapter or a future native worker.
`tokenize_bridge` sets `USE_TORCH=0` at import time, and the others import it.
Read their receipts and stored arrays as data instead. This task does not need to
repeat measured tokenization or claim that byte/hash checks requalify native token
offsets.

## Smallest bounded implementation that would complete this preparation

One concrete DEV-input/scoring module, focused tests and one immutable binding
artifact are sufficient; no new generator or expandable framework is needed.

1. Use fixed allowlisted open DEV paths and bind exact raw source/token file bytes,
   their current construction/profile receipts, registrations, static Titan and
   tokenizer/template identities, and the pure verifier source/dependency hashes.
   Parse once from the bytes hashed and recheck the source snapshot before
   publishing. Never enumerate or read CHECK/FINAL reservations or payloads.
2. Validate the actual row/parent/family quotas, source-to-token joins, roles,
   prompt-prefix hashes, target separation and full twin/eight-row coverage. Export
   policy inputs containing only prompts/IDs, with evaluator semantics in a
   separate bound sidecar. Preserve source rows; no resampling, replacement,
   relabeling or deduplication of derivatives.
3. Provide fixed deterministic scorer adapters for these three panels, validating
   reconstructed semantic identity and using the existing independent verifiers.
   Freeze what counts as malformed before responses: distinguish a wrong value
   from a bad envelope, duplicate keys, forbidden extra text or incompatible answer
   type/shape. Preserve raw responses and failed attempts through the existing
   importer. Bind imported helper code as well as the wrapper: the current
   `scorer_binding` hashes the callback's file, which alone does not bind changed
   imported verifier implementations.
4. Exercise the actual open-bank binding once, without manufactured student
   outputs. Use clearly synthetic fixture responses only in tests for positive,
   wrong, malformed, EOS/failure, missing/duplicate, identity-drift and pair-coverage
   paths. A mutation test must prove that target tokens cannot enter policy inputs
   and that changing a source semantic row or scorer dependency invalidates binding.
   Verify that no framework/environment side effects occur on import.
5. Keep outputs explicitly source/data-only and diagnostic. The completion evidence
   is exact bank binding plus scorer mechanics, not a baseline accuracy table,
   native source-correct enrollment, grading calibration or automatic promotion.

This is finite: bind the three existing deterministic DEV panels, totaling 768
rows and 480 parent groups, plus the existing 64 language DEV input identities if
their current profile is complete. Across all four, that is 832 rows/544 parent
groups, not 832 independent samples or 544 independent reasoning mechanisms. The
language adapter must remain input-only. No CAL controls may be added to these
counts or source-correct membership.

The current enrollment manifest also requires run-specific runtime identity,
decoding, seeds and a freeze time. **Do not invent these to make an artifact pass.**
Produce the static bank/scorer bindings now; a small assembly function may require
a separately supplied, subsequently frozen run registration before making a full
import manifest. Keep separate panel manifests when decoding caps differ: a single
global cap cannot silently reconcile the short-panel 128-token and long-context
24-token provisions. Preparation hashes are not observed native runtime hashes.

## Entry needs and stop boundary

The proposed work satisfies a real subset of eventual entry needs: knowing exactly
which prompts will be sent, preventing reference leakage, checking the complete
expected population, and selecting the correct deterministic scorer. These facts
are available from existing bytes and pure semantics; measured E0 throughput is
not required to establish them. Leaving them manual would defer necessary work
until the collector is written and invite avoidable data/score mismatches.

It does **not** authorize or verify collection. `enrollment.py` must continue to
report structural/diagnostic status and false native/grader qualification flags.
`e1_gates.py` must continue to block unavailable concrete capture, language and
long-context evidence. Do not resurrect callback-based qualification. Current
language candidate annotations cannot replace independent qualified reference
labels, reviewer assignments, calibration or actual unchanged-source responses.

Prioritize this narrow adapter over a full runnable trainer. Writing collector or
trainer source is not logically impossible before E0, but native policy/budget
registrations and qualified execution still require measured runtime evidence.
The new adapter removes a known prerequisite without building on unmeasured load,
gradient, save/resume, throughput or long-context behavior. Do not alter the E0
qualifier simply to share code, because its committed source is part of its gate.

Once this bounded connection and the current language preparation are complete,
reassess actual missing prerequisites. If only E0/resource change, native capture,
the 2048/4096 extension and externally qualified grading remain, report that impasse
honestly. Do not generate more bank versions, repeat unchanged blocked launches,
clone a runtime without new evidence, or add generic wrappers merely to extend
activity. The fifth native blocker recurrence alone does not make the entire goal
blocked while this specific, necessary connection remains unimplemented.
