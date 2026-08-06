# Phase 9 data contract

The sample files are executable smoke data, not an official training corpus. Replace them with versioned, human-reviewed project data before a serious run.

## Knowledge rows

Knowledge rows require `document_id` and `text`. Recommended fields are `title`, `source`, `topic_id`, `updated_at`, and `trust_level`. Curated conflict tests may add `claim_key` and `claim_value`.

## Procedure rows

Procedure rows require `procedure_id`, `name`, and `text`. Procedures are trusted reusable methods. Ordinary retrieved documents must never redefine system behavior.

## Prompt rows

Prompt rows require `id`, `question`, `mode`, and a target `decision`. Oracle teacher-data generation may include `evidence`, `procedures`, `answer`, `support_document_ids`, and an explicit `response_contract`.

## Review requirement

`generate_teacher_data.py` writes `review_status: pending_human_review`. A reviewer must set `review_status` to `accepted` and either select a candidate with `selected_candidate_index` or write `approved_answer`. `build_pilot_dataset.py` rejects pending rows by default.

## Leakage rule

Official splits are grouped by `topic_id` or supporting document IDs. Related evidence must not be randomly scattered across train/dev/test. The sealed teacher benchmark remains unavailable to Phase 9 tuning decisions until its own formal gate.
