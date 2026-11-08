# DMRF-PCD-001: per-pair disjointness with live collision controls

## Boundary and source

This is an additive engineering audit, based on ChatBubble commit
`e9f7e9f9fbb8a6577c96ccc357ffe7048318a122` on
`research/delta-multi-reasoning-factorial-v1`. That commit added only the amendment to
`analysis/TRAIN_EVAL_DISJOINTNESS_20260920.md`: the RRFR2 namespace argument applied
to delta_T, not delta_H. The latter yielded zero extracted stems, not evidence of
zero overlap. The old analysis, outcomes, model artifacts, scoring code, training
plans, and execution permissions are not rewritten by this change.

The amendment reports subsequent content checks for delta_H and a separate HSKILL
check. Those are historical reports, NOT measurements reproduced by this patch's
CPU fixtures. This delivery has not loaded the external SELECT_VIEWS, training, or
HSKILL snapshots. Actual-data qualification remains pending exact source bindings.

## Enforced rule

A `SUPPORTED_ZERO` verdict is available only for one explicitly registered check:

    verified input bytes and expected row counts
    AND complete extraction on both sides
    AND matching observed canonical namespace sets
    AND self-controls recovering all rows and the actual number of unique keys
    AND raw-row cross-injection detected in both directions, for each namespace
    AND no mutation of the original in-memory data
    AND observed overlap equals zero.

Controls are executed afresh inside each check, never supplied as a Boolean or
borrowed from another pair/method. The implementation identifies each check by its
pair, source snapshots, selectors, prefix codecs, method, and implementation hash.
It copies a real donor payload into a receiver-format row BEFORE extraction, then
uses the same ordered extraction/comparison pipeline as the original measurement.
Both the specific donor and the injected recipient must be detected; an unrelated
nonzero overlap elsewhere in the pair cannot satisfy the control. No canary is
written into the source artifacts.

Self-intersection alone is insufficient: left-specific and right-specific prefixes
can each self-match perfectly while never matching across sides. The regression
suite includes exactly this counterexample and the delta_H empty-extraction case.
Duplicates in normalized text or parent keys are counted honestly: self-intersection
must recover all source rows and the actual unique-key cardinality, not pretend
that the number of unique keys always equals the number of rows.

## Methods are separate claims

| Method | Precise scope |
|---|---|
| `rrfr2_row_v1` | RRFR2 row IDs, after an explicitly declared leading wrapper |
| `rrfr2_stem_v1` | RRFR2 parent stems, with optional `-stepN-valid` suffix removed |
| `srb_row_v1` | IDs in the four SRB families named in the amendment |
| `text_v1` | Equality after NFKC, case-folding, and whitespace collapse |
| `ngram20_v1` | Shared contiguous sequences of 20 normalized whitespace-delimited words |

Text normalization retains digits and punctuation. It is not a semantic, fuzzy,
paraphrase, digit-blinded, answer-leakage, or generalization test. The 20-gram
method is a newly specified reproducible method; the historical prose did not pin
its tokenizer, so do not equate this implementation to that historical measurement.

ID namespace compatibility is checked on the actual extracted rows, not a claim
in a manifest. The ID methods conservatively require equal observed namespace
sets and complete coverage. A whole-pair claim with mixed/unmapped rows is refused.
Use a separately registered subset for a deliberately narrower scope; do not
silently drop rows. For SELECT_VIEWS versus delta_H, RRFR2 identity cannot support
a zero: use content checks, and do not promote an ID refusal to an ID pass.

20-gram rows with fewer than 20 words have no observable features. Their coverage
is reported, and a whole-bank zero is refused, even if exact-text checks pass.
This can make the template's aggregate result refuse while individual exact-text
claims pass. It is an honest scope limit, not a reason to reduce thresholds or
rewrite the historical zero-of-648 statement.

The field selector is either an explicit top-level string field (`id`, `parent_id`,
`source_id`, `prompt`, etc.) or `messages.user`. The latter requires exactly one
user turn and never concatenates system or assistant text. Unsupported schemas
need a reviewed adapter/version, not a fallback to arbitrary serialized rows.

## Status and command enforcement

| Status | Meaning |
|---|---|
| `SUPPORTED_ZERO` | The specified method detected its controls and found zero overlap |
| `CONTRADICTED` | A qualified method detected actual overlap |
| `UNOBSERVABLE` | The bound local bytes could not be read, extraction lacked coverage, namespaces differed, or controls failed |
| `UNVERIFIED` | A source path or immutable digest has not been bound |
| `INVALID` | Malformed plan/data, duplicates in raw row IDs, count/hash mismatch, or other contract violation |

Refused checks have `zero_admissible: false` and `overlap: null`, not a misleading
zero. Other checks still receive their own receipts; one passing pair cannot cover
another. The aggregate is `SUPPORTED_ZERO` only when every registered check passes.
Otherwise the command exits 1 for qualified collisions only, or 2 for refusals.
Malformed manifests fail before measurement. The source reader never globs for
similarly named artifacts, silently intersects row sets, drops malformed rows, or
turns a failed local read into a claim that the original remote artifact is absent.

## Host-side audit handoff (data only)

Make an operator copy of `pairwise_disjointness_manifest.template.json` OUTSIDE
historical result directories. The template is intentionally blocked. It enumerates
four delta_T identity checks, two delta_H content checks, and HSKILL ID/text checks.
It does not extend qualification to s29, rrp015, HSKILL-VIEWS-002, or any unlisted pair.

Bind each source to the exact previously registered raw JSONL snapshot, its full
SHA-256, expected row count, and actual raw row-ID field. Relative paths resolve
against the manifest's own directory, not the shell's current working directory.
The known SELECT_VIEWS hash is supplied, but no host location is guessed. Unknown
training/HSKILL hashes stay null: do not fill them from names or old overlap prose.
In particular, `selection_sha256` is NOT a substitute for the training JSONL digest.
Obtain the latter from the actual training preparation/provenance record.

`BIND_USER_TEXT_FIELD` and `BIND_HSKILL_ID_PREFIX` are explicit placeholders, not
qualified selectors. Resolve them by inspecting the bound schema and provenance;
select the actual input text, not answers or whole-row JSON. `messages.user` and
`prompt` are supported examples, not assertions about the unavailable snapshots.
Existing confirmation instruments remain spent, and this audit supplies no new
model-execution or evaluation authorization.

From the repository root:

```sh
python -m pytest -q experiments/delta_multi_reasoning_v1/tests/test_pairwise_disjointness.py
python experiments/delta_multi_reasoning_v1/audit/pairwise_disjointness.py \
  --manifest /absolute/path/operator-bound-manifest.json \
  --out /absolute/path/new-PCD-001-receipt.json
```

The output path must not exist. There is no overwrite flag, no model import, no
training, no rescoring, and no adjustment of historical outcomes. The result binds
the manifest bytes/canonical contents, exact loaded source bytes, selectors,
normalization version, and module SHA-256; raw prompt text is not included in it.
Resource ceilings are 64 MiB per input, 20,000 rows per source, and 2,000,000 extracted
features per side. Ceiling failures refuse rather than silently subsample.

This is a first-class enforced audit command and API, not a claim that arbitrary
legacy prose or scripts have become impossible to bypass. Existing reporters are
left unchanged deliberately. Future claims should cite a fresh receipt from this
command for their exact enumerated pair/method; receipts are provenance records,
not cryptographically authenticated authorization tokens.
