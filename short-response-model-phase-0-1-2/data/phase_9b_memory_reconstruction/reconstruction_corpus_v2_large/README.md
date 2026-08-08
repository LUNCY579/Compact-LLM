# Phase 9B Memory Reconstruction Corpus v2 Large — v1.7.5

This corrective corpus contains **3,600** canonical source-state reconstruction cases across 300 topic families, with **1,500** explicit memories, **600** counterfactual pairs, and all five reconstruction decisions.

The authored split is 2700 train / 456 dev / 444 sealed test. Every case contains current state; every memory contains source state and explicit validity conditions. Gold rows are `auto_verified`, not mislabeled as human-reviewed.

`curriculum_views/accept_adapt_recovery.train.reviewed.jsonl` emphasizes useful memory application/adaptation while rehearsing rejection/conflict/insufficient behavior.
