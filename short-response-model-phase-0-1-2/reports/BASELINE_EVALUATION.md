# BASELINE EVALUATION

## Scope

Evaluate the untouched base model on `shortchatbench_pilot.jsonl`.

## Inputs

- Repository revision:
- Base model:
- Model revision / SHA:
- Tokenizer revision / SHA:
- Benchmark version:
- Generation configuration:
- Hardware:
- Software versions:

## Work Completed

- [ ] Validate benchmark JSONL.
- [ ] Run deterministic base-model evaluation.
- [ ] Save every prompt, reply, token count, latency, and check result.
- [ ] Review the lowest-scoring cases manually.
- [ ] Review factual and safety cases manually.

## Aggregate Results

| Metric | Result |
|---|---:|
| Cases | TBD |
| Median generated tokens | TBD |
| P90 generated tokens | TBD |
| Complete-utterance rate | TBD |
| Mean heuristic score | TBD |
| Human short-response preference | TBD |
| Peak allocated VRAM | TBD |
| Mean generation latency | TBD |

## Category Results

| Category | Automated score | Human review | Notes |
|---|---:|---|---|
| Acknowledgment | TBD | TBD | |
| Celebration | TBD | TBD | |
| Empathy | TBD | TBD | |
| Simple fact | TBD | TBD | |
| Clarification | TBD | TBD | |
| Uncertainty | TBD | TBD | |
| Context consistency | TBD | TBD | |
| Safety boundary | TBD | TBD | |
| Complex escalation | TBD | TBD | |
| Naturalness | TBD | TBD | |

## Failure Analysis

Document echoing, verbosity, incomplete answers, incorrect compression,
invented context, repetitive templates, and unsafe brevity.

## Gate Status

`NOT RUN`

This describes original base-model behavior. It does not satisfy the dense-SFT
stop gate.
