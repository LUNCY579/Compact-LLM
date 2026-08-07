# Phase 9B Memory Reconstruction Corpus v1.1 — integrated for model v1.7.4

This is the repository-native dedicated memory-reconstruction curriculum.

## Scale

- 300 external memory records
- 600 accepted canonical reconstruction rows
- 60 topic families
- 120 counterfactual pairs
- decisions: `{'accept': 96, 'adapt': 204, 'conflict': 60, 'insufficient': 120, 'reject': 120}`
- authored splits: `{'train': 440, 'dev': 80, 'test': 80}`

## Canonical training boundary

Only `approved_reconstruction` and `approved_answer` are supervised targets. The
`preferences/` and `evaluator/` directories contain deliberate wrong/naive
alternatives and must never be concatenated into ordinary SFT.

## Split policy

The authored topic-family split is preserved with a top-level `intended_split`.
Topic families do not cross train/dev/test. Dev is used for reconstruction-policy
selection; test remains sealed until a single strategy is frozen.

## Counterfactual note

The source package described its counterfactuals as minimal edits. v1.7.4 makes
that claim precise: `decisive_fields` records the causal premise/goal edit, while
`changed_fields` records all concrete fields that changed. Some examples add a
small number of supporting state-coherence fields so the resulting state and
answer remain well-defined.

## Runtime role

Phase 9B is the primary consumer. Phase 10 uses the approved Phase 9B checkpoint
and can run a reconstruction-retention probe. Phase 11 can enable prompted or
trained two-pass reconstruction and benchmark it separately from ordinary RAG.
