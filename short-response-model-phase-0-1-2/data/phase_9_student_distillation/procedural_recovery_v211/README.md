# Phase 9 Procedural Recovery Corpus v2.1.1

This corpus is the preservation-first successor to v2.1. It is designed for a student that already performs strongly on the original Phase 9 DEV set when `query_aware_v21` retrieval is used, but still needs residual training on harder procedural evidence-binding cases.

## Canonical corpus

- Rows: **6560**
- Train / DEV / TEST: **4976 / 792 / 792**
- Procedure families: **24**
- Knowledge documents: **5040**
- Trusted procedure records: **2320**

Supported rows intentionally span clean binding, moderate distractors, and hard sibling-record collisions rather than making every example maximally adversarial. Domains include release/version selection, API latency/rate limits, support priority, capacity/quota, incident routing, retry backoff, deployment gates, inventory, shipping SLA, freshness, compatibility, access policy, failover, maintenance windows, and backup retention.

## Training views

- `train_stage_a_short_binding.reviewed.jsonl`: **768** hard answerable rows for a short binding burst.
- `train_stage_b_preservation_first.reviewed.jsonl`: **4820** rows, of which **3120** are original Phase 9 train-only preservation replay and **1700** are new v2.1.1 procedure rows.
- `dev_preservation_probe.reviewed.jsonl`: **512** mixed DEV-only monitoring rows.

The preservation replay includes factual answers, abstention/conflict boundaries, social responses, **verified original procedure answers**, and original procedure boundary cases. The 18 known inconsistent historical latency-tie rows are excluded from replay but left untouched in the historical corpus for reproducibility.

## Sealed-test rule

`reviewed_splits/test.reviewed.jsonl` is not used for training, schedule selection, or retrieval-gate recalibration. Freeze one DEV-selected candidate first, then evaluate TEST once.
