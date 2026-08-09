# Phase 9 v2.1.1 Dataset Audit

## Canonical corpus

- 6,560 reviewed/auto-verified deterministic rows
- 4,976 train / 792 DEV / 792 sealed TEST
- 5,040 factual documents
- 2,320 trusted procedure records
- 24 procedure families
- validator status: PASS
- generated latency-gold audit: 186 checked train answer rows, 0 violations

## Evidence-profile balance

| Profile | Rows | Purpose |
|---|---:|---|
| Clean binding | 1,440 | Ordinary evidence use without same-topic overload |
| Moderate distractor | 1,440 | One sibling/unrelated distractor while preserving a clear target |
| Hard collision | 2,880 | Exact-locator binding under near-identical sibling records |
| Boundary | 800 | Missing-input and equal-authority conflict behavior |

The corpus is intentionally not 100% hard-collision examples. This corrects the v2.1 training-distribution mismatch where almost every new answerable procedural row was collision-heavy.

## Stage A

`curriculum_views/train_stage_a_short_binding.reviewed.jsonl`

- 768 rows
- answerable only
- hard-collision only
- deterministic family-stratified sample
- intended for a short 60-step residual-binding burst

## Stage B

`curriculum_views/train_stage_b_preservation_first.reviewed.jsonl`

| Curriculum role | Rows |
|---|---:|
| Original factual answers | 1,200 |
| Original factual boundaries | 700 |
| Verified original procedure answers | 800 |
| Original procedure boundaries | 300 |
| Social replay | 120 |
| New clean/moderate procedure | 900 |
| New hard residual procedure | 350 |
| New procedural boundaries | 450 |
| **Total** | **4,820** |

Original preservation replay is **3,120 / 4,820 = 64.7%** of the Stage B examples. The 18 known inconsistent historical latency-tie rows are excluded from replay but remain unmodified in the source corpus.

## Real-world-reflected procedure domains

Examples include release/version selection, API probes, support tickets, capacity/quota, procurement, incident routing, retry policy, deployment gates, inventory replenishment, shipping SLA/cost selection, freshness selection, compatibility windows, access control, regional failover, maintenance windows, and backup retention.

The cases remain synthetic and deterministic so that golds can be independently checked by code; they are "real-world-reflected" rather than claimed to be production logs.
