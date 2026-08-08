# Phase 9 RAG Reasoning Corpus v3 Large — v1.7.5

This deterministic fictional corpus is a corrective curriculum for students that improve overall but still fail supported/procedural answering. It contains **5,900** canonical cases, **6,000** evidence records, and **200** trusted procedures.

The authored split is `4410` train / `752` dev / `738` sealed test by topic family. Gold rows are `auto_verified`, not mislabeled as human-reviewed.

Training views:
- `support_procedure_recovery.train.reviewed.jsonl`: answer-heavy corrective SFT with a small abstention rehearsal set.
- `procedure_bootstrap.train.reviewed.jsonl`: only answerable procedural cases.
- `full.train.reviewed.jsonl`: complete authored training split.

The canonical set covers direct support, paraphrase, corroboration, distractors, stale/current precedence, retrieval injection, tentative evidence, ambiguity, missing/partial evidence, equal-authority conflict, and eight procedure families.
