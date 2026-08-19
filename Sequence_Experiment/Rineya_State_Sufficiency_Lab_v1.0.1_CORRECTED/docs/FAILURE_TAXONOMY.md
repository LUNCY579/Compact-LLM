# Failure Taxonomy

## Input and backend

- `BACKEND_ERROR`: server, model, OOM, timeout, or transport failure.
- `EMPTY`: no generated text.
- `REFUSAL`: explicit inability or refusal.

## Parsing and contract

- `NO_ANSWER_FOUND`: no safe final candidate.
- `CONFLICTING_FINALS`: explicit answers disagree.
- `PARTIAL_STRUCTURED_OUTPUT`: JSON exists but required fields are missing or invalid.
- `LENIENT_STRUCTURED_RECOVERY`: fields recovered from labels rather than the strict JSON contract.

## Local transition

- incorrect next reasoning step;
- incorrect next compact state;
- correct wording but conflicting numeric signature;
- incorrect `done` flag;
- premature final answer;
- final transition without an answer.

## Autonomous rollout

- `COMPLETED`;
- `IMPLICIT_FINAL`;
- `DONE_WITHOUT_FINAL`;
- `MISSING_UPDATED_STATE`;
- `REPEATED_OUTPUT_LOOP`;
- `REPEATED_STATE_LOOP`;
- `STATE_TOO_LARGE`;
- `MAX_STEPS`;
- `BACKEND_ERROR`.

## Interpretation

A local transition failure indicates decision or state-update error. A rollout-only failure after correct oracle transitions indicates compounding state error, termination-policy weakness, or exposure bias.
