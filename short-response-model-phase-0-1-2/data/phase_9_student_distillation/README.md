# Phase 9 data contract

## v1.7.5 canonical research corpus

The default serious-run corpus is now `rag_reasoning_v3_large/`. It contains 5,900 deterministic canonical rows, 6,000 fictional factual documents, 200 trusted procedures, 600 topic families, authored train/dev/sealed-test splits, preference/evaluator assets, and dedicated supported/procedural recovery views. The generator marks deterministic gold as `review_status: auto_verified`; this means schema-verified synthetic gold, **not human-reviewed teacher output**.

The grounded-recovery runner defaults to the `support_procedure_recovery` sampling profile: 55% supported evidence use, 25% procedure application, 8% unsupported/missing, 4% conflict, 5% injection, and 3% social rehearsal. Before model evaluation it audits that every supported DEV row can retrieve its authored support document in the FTS5 top 3.

## Knowledge rows

Knowledge rows require `document_id` and `text`. Recommended fields are `title`, `source`, `topic_id`, `updated_at`, and `trust_level`. Curated conflict tests may add `claim_key` and `claim_value`.

## Procedure rows

Procedure rows require `procedure_id`, `name`, and `text`. Procedures are trusted reusable methods. Ordinary retrieved documents must never redefine system behavior.

## Prompt rows

Prompt rows require `id`, `question`, `mode`, and a target `decision`. Oracle teacher-data generation may include `evidence`, `procedures`, `answer`, `support_document_ids`, and an explicit `response_contract`.

## Review requirement

Human-generated teacher packets still require `review_status: accepted` plus either `selected_candidate_index` or `approved_answer`. The integrated v1.7.5 deterministic corpus instead uses `auto_verified`, with explicit provenance and repository validation. Pending human-review rows remain rejected by default.

## Leakage rule

Official splits are grouped by `topic_id` or supporting document IDs. Related evidence must not be randomly scattered across train/dev/test. The sealed test remains unavailable to Phase 9 tuning decisions until a final candidate is frozen.
