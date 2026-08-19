# Evaluation Standard

## Principle

Evaluation must distinguish **output**, **parsed claim**, **local transition**, **maintained state**, and **final outcome**. A fluent response is not automatically correct, and a formatting defect is not automatically a reasoning defect.

## Protocols

### Direct solve

The model receives only the original problem. The FULL adapter is expected to emit a complete reasoning response; the final answer is extracted and scored deterministically.

### Oracle transition

The model receives the original problem and the canonical compact state. It predicts the next step and updated state. Report:

- strict JSON contract validity;
- `done` agreement;
- next-state exact match;
- conservative lenient state equivalence;
- next-step exact match and token F1;
- final-answer correctness on final transitions.

### State/history ablation

Use the same expected transition while changing context:

- compact state;
- full prior reasoning history;
- last previous step only;
- no state, retaining only step position;
- state from another problem;
- correct state plus an irrelevant distractor.

The important contrast is `compact_state` versus `full_history`. Perturbed states test whether the model actually conditions on state rather than memorizing problem-to-step mappings.

### Generated-state rollout

The model starts at the first compact state and repeatedly generates its own next state. Gold state is never restored. Report final correctness and all termination/failure reasons.

## Strict and lenient tracks

### Strict

Strict scoring requires unambiguous extraction and, for stateful transitions, the declared JSON contract. Canonical exactness is intentional: it is reproducible and conservative.

### Lenient

Lenient scoring credits safely recoverable answers and state paraphrases only when:

- token overlap exceeds the configured threshold;
- numeric signatures do not conflict;
- required final values still pass the task-aware scorer;
- no conflicting explicit answers exist.

Lenient results are diagnostic and must be shown next to strict results.

## Denominators

Every requested case remains in the denominator, including:

- empty output;
- refusal;
- backend error;
- malformed JSON;
- missing fields;
- conflicting finals;
- nontermination;
- loop detection;
- state overflow.

Never discard a case because it is hard to parse.

## Group reporting

Always report:

- micro accuracy;
- macro category accuracy;
- every category;
- every domain;
- difficulty bands;
- trajectory depth or step count;
- strict-lenient gap;
- worst category;
- termination taxonomy for rollouts.

## Manual audit

The harness exports a deterministic sample emphasizing strict failures and strict/lenient disagreements. Human review should label:

1. parser error;
2. scorer error;
3. equivalent but noncanonical reasoning;
4. genuine reasoning error;
5. state-maintenance error;
6. formatting-only defect;
7. ambiguous problem or target.

Any parser/scorer amendment must be versioned and rerun on all compared systems, not only the model that benefits.
