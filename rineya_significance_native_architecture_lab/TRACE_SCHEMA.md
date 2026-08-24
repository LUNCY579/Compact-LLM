# Trace schema and replay boundary

Traces make expensive model observations reusable only where the resulting
counterfactual is mathematically valid. They do not turn a changed model
trajectory into an open-loop replay problem.

## Directory format

Each trace is an immutable directory:

```text
<case-trace>/
  trace_manifest.json
  events.jsonl
  event_00000.npz
  event_00001.npz
  ...
```

The implementation writes each NPZ shard before appending its checksummed JSONL
index row. It refuses to overwrite a nonempty event stream.

## Trace manifest

Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | currently `1.0` |
| `case_id` | stable benchmark case identifier |
| `support` | `FULL_CANDIDATE` or `ON_POLICY` |
| `execution_mode` | `OPEN_LOOP_TRACE_SIMULATION` or `CLOSED_LOOP_MODEL_EXECUTION` |
| `score_semantics` | exact definition and reduction of each score/count observation |
| `information_access` | explicit answer/future/oracle access declaration |
| `model_id`, `model_hash`, `tokenizer_hash` | required for model-derived traces |

`information_access` always declares `answer`, `future_tokens`,
`future_hidden_state`, and `oracle_evidence`. Answer, future-token, and future
hidden-state access are forbidden. Oracle evidence may be attached after
execution for analysis, but runtime policy code must not receive it.

## Event state

Each event shard stores:

- stable semantic `entry_ids`;
- absolute token `positions` and `token_ids`;
- pre-observation `scores` and `counts`;
- active-entry mask;
- `score_delta` and `count_delta` for the event;
- policy-selected indices;
- optional post-hoc oracle-evidence mask.

The JSONL row records event identity, kind, boundary position, policy ID, NPZ
SHA-256, and fingerprints of both the pre-observation and observed states.
Event indices must be contiguous.

## Support levels

`FULL_CANDIDATE` means the trace records observations for every candidate entry
needed by the declared policy family. It may support counterfactual ranking and
selection under different exponents or count transformations.

`ON_POLICY` means observations exist only along the executed policy trajectory.
It cannot support claims about entries or future boundaries that the policy did
not retain and observe.

Support is a property of a trace, not an aspiration. Missing counterfactual
observations cannot be imputed and presented as measured.

## Valid open-loop replay

Examples that can be valid when full candidate support exists:

- sweeping `e` over a fixed score/count trajectory;
- constant, age-derived, shuffled, delayed, noisy, or head-averaged count
  perturbations;
- ranking and matched-budget selection comparisons;
- tiny controllers whose outputs do not change recorded hidden states.

Every replay result is labeled `OPEN_LOOP_TRACE_SIMULATION`.

## Closed-loop-only variants

These change future hidden states or attention and require real model execution:

- significance-aware logit priors;
- memory slots exposed to later layers/tokens;
- recomputation or context restoration;
- a controller that changes future cache contents or compute;
- any policy whose earlier selection changes later score contributions.

Open-loop results may screen such ideas but cannot establish their model-level
quality.

## Required validations

- array shapes match the significance state;
- score/count deltas are finite;
- counts and count deltas are non-negative;
- inactive entries receive no observation;
- selected indices resolve to stable entry IDs;
- NPZ SHA-256 and state fingerprints match;
- event indices are contiguous;
- source semantics and information-access fields are present.

The machine-readable manifest contract is
`schemas/trace.schema.json`. Array shards are validated by the Python trace
reader because JSON Schema does not inspect NPZ contents.

