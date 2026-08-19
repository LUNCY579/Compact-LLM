# Dataset Card — Rineya Sequence-Independence Reasoning Dataset v1.0

## Summary

This dataset contains two exactly paired representations of 48,000 synthetic reasoning problems:

- a full-context representation with one uninterrupted solution per problem;
- a compact-state transition representation with multiple independently trainable records per problem.

The split-transition records are globally shuffled to remove chronological presentation order. Every transition retains the original problem and a compact state derived from the canonical complete trajectory.

## Intended research use

The primary use is a controlled adapter comparison testing whether independent sufficient-state transitions can preserve, improve, or reduce learned reasoning capability relative to full-trajectory supervised fine-tuning.

Appropriate uses include:

- supervised fine-tuning of small language models;
- next-step reasoning evaluation with oracle compact states;
- iterative stateful rollout evaluation;
- state-sufficiency and history-removal ablations;
- training-budget and sample-efficiency analysis;
- error-propagation studies.

## Not intended for

- factual-knowledge pretraining;
- medical, legal, or financial decision support;
- claims about hidden chain-of-thought faithfulness;
- measuring general intelligence from this dataset alone;
- evaluating historical-sequence removal without controlling for the larger number of split records.

## Composition

| Split | Full problems | Split transitions |
|---|---:|---:|
| TRAIN | 43,200 | 170,941 |
| DEV | 2,400 | 9,460 |
| TEST | 2,400 | 9,521 |
| **Total** | **48,000** | **189,922** |

Every one of the 30 categories contains exactly 1,600 problems: 1,440 TRAIN, 80 DEV, and 80 TEST.

## Domains

- Mathematics: arithmetic, fractions, percentages, algebra, geometry, probability, sequences.
- Data and symbolic reasoning: statistics, sets, Boolean logic, tables, string transformations.
- Planning and state: elapsed time, scheduling, inventory, ordering constraints, recursive state updates.
- Daily applications: ratios, finance arithmetic, cooking scaling, unit conversions.
- Science: introductory physics, chemistry, and biology problems with explicit formulas or rules.
- Coding: short deterministic Python-like program traces.
- Language: deterministic application of explicitly stated pluralization rules.

## Construction process

For each problem:

1. A deterministic generator creates an original problem, exact answer, complete reasoning trajectory, and compact states.
2. The complete full-context response is assembled from that trajectory.
3. The same trajectory is expanded into state-to-next-step records.
4. The source problem is assigned to TRAIN, DEV, or TEST before expansion.
5. Split transitions are globally shuffled using deterministic hashes.
6. A validator regenerates every example from its stored seed and verifies full/split parity.

## Compact-state policy

A compact state should retain information needed for the next correct decision, such as:

- the current derived value;
- unresolved operations or commands;
- current program variables;
- current position/direction;
- current schedule time;
- remaining constraints.

It should not replay all completed reasoning prose. The original problem remains available in every transition prompt.

## Quality controls

The validator confirms:

- 48,000 unique normalized problem texts;
- no cross-split problem leakage;
- exact source hashes;
- complete transition chains;
- deterministic regeneration equality;
- no adjacent transition rows from the same problem;
- final-answer and completion-flag consistency.

## Known limitations

1. **Procedural templates.** The examples are highly verifiable but less linguistically diverse than a large human-authored corpus.
2. **Simple-task emphasis.** This first release intentionally favors tasks with exact state transitions and exact answers.
3. **Representation confound.** The split version has about 3.96 records per source problem. Native one-epoch training therefore changes both representation and exposure quantity.
4. **Oracle state construction.** Compact states are generated from known canonical solutions. Real deployment requires a model or harness that can construct and verify states from its own outputs.
5. **Synthetic rationales.** The full reasoning traces are pedagogical targets, not guaranteed descriptions of how a trained model internally computes.
6. **English only.** Prompts and targets are in English, with mathematical and code notation.

## Provenance

All content was generated specifically for this project. No external dataset row was copied. The external Apache-2.0 dataset described by the user served only as conceptual inspiration for decomposing complete solutions into local transitions.
