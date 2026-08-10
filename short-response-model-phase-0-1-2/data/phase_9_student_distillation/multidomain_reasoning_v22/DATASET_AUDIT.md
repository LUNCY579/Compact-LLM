# v2.2 Multi-Domain Dataset Audit

- Canonical rows: **40,500**
- Unique canonical user prompts: **40,500 / 40,500**
- Domains: **27**
- Reasoning archetypes: **15**
- TRAIN / DEV / TEST: **32,400 / 4,050 / 4,050**
- TRAIN-only social rehearsal rows: **3,000**

## Design checks

- Every canonical answer is derived from prompt-contained information.
- Canonical user prompts are exact-string unique, including across splits.
- The assistant target contains no chain-of-thought/rationale field.
- Legal, health, and finance cases use hypothetical stated rules rather than external professional advice.
- Specialist and anchored curriculum views are built from TRAIN-only sources.
- Social rehearsal is TRAIN-only and is not used as a direct evaluation set.

## Domains

general, research, writing, communication, translation, coding, devops, systems, math, data_analysis, science, engineering, education, business, operations, finance, legal, health, consumer, travel, home, creative, media, personal, coaching, tools, agents

## Reasoning archetypes

compare_metric, filter_then_rank, aggregate_delta, capacity_fit, state_transition, rule_exception, evidence_binding, insufficient_information, conflicting_evidence, dependency_order, two_hop_relation, constraint_preserving_transform, constraint_satisfaction, evidence_synthesis, diagnostic_priority
