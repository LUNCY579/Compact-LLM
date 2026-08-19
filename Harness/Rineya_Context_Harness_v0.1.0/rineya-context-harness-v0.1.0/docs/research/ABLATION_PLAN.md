# Research and Ablation Plan

## Primary research question

Does explicit recursive-state forgetting improve long-horizon task success and context efficiency for a 344M–454M Rineya model compared with full history, sliding windows, and ordinary summarization?

## Core ablations

| ID | Condition | Deletion | Structured state | Archive/restore |
|---|---|---:|---:|---:|
| A | Full history | No | No | No |
| B | Sliding window | Implicit | No | No |
| C | One-shot summary | Implicit | Partial | No |
| D | Structured state | No | Yes | Yes |
| E | Deterministic RSF | Yes | Yes | Yes |
| F | Learned RSF | Yes | Yes | Yes |
| G | Oracle RSF | Yes | Yes | Yes |

## Isolating combo effects

1. Search only.
2. Search + ranking.
3. Search + one-shot compression.
4. Search + structured reconstruction.
5. Search + reconstruction + explicit deletion.
6. Search + reconstruction + deletion + answerability.
7. Full system + restore path.

## Failure taxonomy

- failed to delete obsolete memory;
- premature deletion;
- lost global invariant;
- retained contradiction;
- wrong supersession;
- unsupported claim;
- citation mismatch;
- repeated query without new information;
- context overflow;
- malformed tool call;
- state revision conflict;
- recovery failure.

## Training-data opportunities

Convert failures into SFT/DPO records for:

- query reformulation;
- claim-evidence linking;
- keep/delete distinctions;
- replacement-before-delete;
- invariant protection;
- exception memory;
- restore decisions;
- concise grounded answers;
- abstention.
