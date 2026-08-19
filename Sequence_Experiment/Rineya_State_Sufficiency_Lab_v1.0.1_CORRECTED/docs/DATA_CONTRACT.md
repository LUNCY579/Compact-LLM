# Data Contract

## Full-context record

Required fields include `problem_id`, `problem`, `answer`, `answer_type`, `messages`, `reasoning_steps`, `compact_state_trace`, `split`, and `source_pair_sha256`.

The state trace must have exactly one more element than the reasoning-step list:

```text
state_0, state_1, ..., state_T
  step_1 -> state_1
  step_2 -> state_2
  ...
  step_T -> state_T
```

## Split-transition record

Each record contains:

- source `problem_id` and immutable pair hash;
- `step_index` and `total_steps`;
- `current_state` and `next_state`;
- `target_step`;
- `done` and optional `final_answer`;
- next-step-only `messages`;
- stateful `stateful_rollout_messages` and serialized target.

## Validation invariants

The strict validator checks:

- complete TRAIN/DEV/TEST layout;
- unique problem IDs and normalized problem texts;
- unique transition IDs;
- no orphan or cross-split transition;
- exact source problem, answer type, and pair-hash alignment;
- contiguous transition indices;
- exact current-state, target-step, and next-state alignment to the full trajectory;
- final-answer and `done` consistency;
- valid conversation shapes and stateful JSON targets;
- adjacency statistics confirming that transition rows are not trajectory-consecutive.

## Semantic limitation

Mechanical validation proves alignment with the canonical trajectory. It cannot prove that a state contains every causally necessary fact for every possible solver. A stratified human audit of compact-state sufficiency remains required.
