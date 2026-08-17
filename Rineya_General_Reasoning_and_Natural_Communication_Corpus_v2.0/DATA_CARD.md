# Data Card — Rineya General Reasoning and Natural Communication Corpus v2.0

## Purpose

This release is the second Rineya behavior corpus and is intended for the Rineya1.1 research cycle. It expands capability breadth without abandoning the defining product behavior: read enough context, reason internally, and answer in a few natural words or one compact sentence.

The v2.3.1 observatory indicated that raw example count was not the dominant bottleneck. Accordingly, v2.0 increases scale only together with stronger structure: 40 balanced classes, counterfactual pairs, multiple difficulty levels, broad domains, multi-turn state changes, grounded mini-contexts, and failure-specific DPO negatives.

## Contents

- 32,000 new SFT records.
- 8,000-row stratified v1.0 preservation slice.
- 40,000-row recommended combined SFT.
- 12,000 DPO preference pairs.
- 40 response classes spanning reasoning, communication, and integrated dialogue.
- Schemas, byte-reproducible generator, structural and semantic validators, manifests, checksums, audit packet, and leakage reports.

## Curriculum documentation

See `reports/curriculum_map.md` for the objective, representative use cases, and intended failure contrast of every class. The corpus uses equal class counts so broad aggregate improvement cannot be produced merely by overrepresenting a few easy arithmetic or acknowledgment templates.

## Intended capabilities

Reasoning covers arithmetic chains, percentages, ratios, units, time, scheduling, ranking, formal implication, explicit rules, contradiction detection, patterns, set counting, constraints, causal and physical reasoning, spatial relations, estimation, evidence inference, underdetermination, and troubleshooting.

Communication covers updates, follow-ups, empathy, tactful disagreement, clarification, preferences, apology/repair, de-escalation, compromise, boundaries, informal texting, and conversation closure.

Integrated examples combine reasoning with concise decision explanations, multi-turn corrections, plan revision, recommendations, practical next steps, evidence-grounded dialogue, message rewriting, and in-character reasoning.

## What this corpus does not do

- It does not contain hidden chain-of-thought targets.
- It does not try to teach exhaustive current world knowledge.
- It does not replace the Phase 9 RAG corpus.
- It does not include repository DEV, SEALED, or preference DEV rows.
- It does not claim human approval; all synthetic rows remain `human_review_status: pending`.

## Recommended use

For the 454M teacher, begin with one QLoRA epoch on the 40,000-row recommended file plus the locally converted 4,410-row RAG training set, producing a 44,410-row dual-skill mix. Preserve the original teacher checkpoint, evaluate the same official chat and RAG DEVs, and freeze a winner before any sealed pass.

DPO should follow only after selecting the SFT candidate. Start with a conservative DPO dose and keep the SFT checkpoint available for rollback because preference tuning can improve naturalness while damaging exact reasoning or evidence behavior.

## Quality-assurance evidence

- Seven core data artifacts regenerate byte-for-byte from seed `20260816`.
- 124,000 schema-record validations passed across the component, union, preservation, combined, and DPO files.
- 15,200 deterministic or explicit-contract semantic checks passed across 19 classes.
- All 12,000 DPO chosen responses and rendered prompts match their source SFT records exactly.
- Exact latest-user overlap is zero against 5,482 rows from the available evaluation snapshots.
- A 520-item stratified human-review packet is included, but its rows remain pending human sign-off.

## Preservation lineage note

The 8,000-row v1.0 preservation slice retains the original conversations and targets. Four source prompts containing the grammatical defect `1 hours` were normalized to `1 hour`; their arithmetic targets were unchanged, and the affected rows carry `metadata.preservation_surface_correction = "singular_hour_grammar"`.

## Limitations

Programmatic validation proves schema, role order, uniqueness, target-length bounds, and available-snapshot isolation. A separate deterministic semantic audit recomputes or contract-checks 15,200 rows across 19 classes and verifies DPO chosen/source linkage. It cannot prove every social response is the uniquely best reply. Human review is required before a public release claim.

The leakage report uses the v1.0 corpus and evaluation files available from the v2.1.1 repository snapshot. The exact current repository must be checked locally because later DEV/SEALED revisions were not provided as local files in this turn.
