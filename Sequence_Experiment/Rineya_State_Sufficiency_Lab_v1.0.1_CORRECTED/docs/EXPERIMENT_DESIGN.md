# Experiment Design

## Core hypothesis

Let the complete reasoning history before decision `t+1` be `H_t`, and let `s_t = g(q, H_t)` be a compact state derived from the complete canonical trajectory for problem `q`. The primary hypothesis is:

```text
P(next decision | q, H_t) ≈ P(next decision | q, s_t)
```

The claim does not say that the underlying problem is non-sequential. It asks whether chronological replay and chronological training-row adjacency are necessary when every transition record contains a sufficient current state.

## Primary treatment arms

### FULL

```text
problem -> complete reasoning trajectory + final answer
```

### SPLIT-UNORDERED

```text
problem + compact state_t -> next step + compact state_(t+1) + done + final answer?
```

All transition rows are globally shuffled. A row for step 5 may be optimized before a row for step 2, and the model receives no hidden activation or conversational memory from a prior dataset row.

## Required controls

1. **Same base checkpoint and tokenizer.** Record immutable hashes or revisions.
2. **Problem-level partitioning before transition expansion.** No state from one source problem may cross TRAIN/DEV/TEST.
3. **Same adapter architecture.** Match LoRA target modules, rank, alpha, dropout, and initialization.
4. **Same optimization family.** Match optimizer, scheduler, effective batch size, and random seeds.
5. **Matched update budget.** The primary configs use the same `max_steps`. Report token exposure as a secondary quantity.
6. **Per-problem weighting.** SPLIT uses inverse transition-count weights so a long solution does not receive automatically greater total loss weight.
7. **Frozen evaluation.** Freeze prompts, parser, scorers, margins, and generation settings on DEV before unlocking TEST.

## Recommended experimental matrix

| Arm | Representation | Row order | Target | Purpose |
|---|---|---|---|---|
| A | FULL | problem shuffle | full trajectory | Main baseline |
| B | SPLIT | global shuffle | stateful transition | Main treatment |
| C | SPLIT | trajectory order | stateful transition | Isolate row-order effect |
| D | SPLIT | global shuffle | next step only | Isolate state-generation burden |
| E | SPLIT | one transition/problem/epoch view | stateful transition | Control expanded-row exposure |
| F | FULL/SPLIT | matched target-token budget | matched target | Exposure sensitivity |

## Primary estimand

The confirmatory estimand is paired end-to-end correctness:

```text
Delta = accuracy(SPLIT generated-state rollout) - accuracy(FULL direct solve)
```

The main decision is non-inferiority under a margin frozen before TEST. Local transition accuracy is mechanistic evidence, not a substitute for end-to-end completion.

## Secondary estimands

- oracle compact-state next-step correctness;
- compact state minus full-history transition correctness;
- teacher-forced versus generated-state rollout gap;
- first divergence depth;
- loop and nontermination rate;
- contract-valid structured-output rate;
- worst-category and macro-category performance;
- direct-solving ability of the SPLIT adapter;
- state length and number of rollout turns.

## Seeds

Run at least three training seeds for a paper-quality claim. Keep data split fixed while changing initialization/data-order seeds. Aggregate by seed and by problem; do not treat expanded transition rows as independent end-to-end test units.

## Interpretation guardrails

- `SPLIT ~= FULL` supports state-sufficient supervision only on the evaluated distribution and harness.
- Strong oracle-transition results with weak rollout results indicate error propagation or state-generation failure.
- Strong rollout results with poor direct generation may indicate a harness-dependent capability rather than internal single-call reasoning.
- Better SPLIT results can arise from denser process supervision or greater token exposure; use controls C–F before attributing the gain to sequence independence.
