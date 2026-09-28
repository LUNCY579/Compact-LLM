# Every CE number in this programme understates the damage by 1.3–1.5×

Cross-entropy here has always been averaged over **all** target positions. Only **7.7%** of a
DEV row's positions are the assistant's actual answer; the rest is the system prompt, the user's
question, chat-template scaffolding and the empty `<think></think>` block. So the headline CE is
overwhelmingly a measure of how well a model predicts **the prompt** — which is not its job at
inference time.

This measures the split. At initialization only, no training, one arm resident at a time.

Exploratory, CPU FP32, 40 DEV rows (4,979 target positions, 381 of them answer). Outside the
hashed root — no receipt, no gate. Open splits only, no efficacy claim.

## Result

| arm | displaced | overall CE | **reply CE** | non-reply CE | penalty overall | **penalty reply** | concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `source_control` | 0 | 5.5782 | 2.6492 | 5.8209 | — | — | — |
| `ced_last4` | 3 | 6.5339 | 4.0426 | 6.7404 | +0.9558 | +1.3934 | **1.46×** |
| `ced_last8` | 7 | 8.4218 | 6.9610 | 8.5428 | +2.8436 | +4.3118 | **1.52×** |
| `C28` | 13 | 12.8880 | 12.2181 | 12.9435 | +7.3098 | +9.5689 | **1.31×** |

`concentration` = (reply penalty) / (overall penalty). Above 1.0 means the damage is worse in
the answer than the average position reports.

## What it shows

**Displacement damage is 1.31–1.52× more severe in the answer than overall CE reports.** The
direction is consistent across all three CED arms. Every CE penalty quoted anywhere in this
programme therefore *understates* the harm to the only part that matters.

This also refines the displaced-layer law: the ~0.5 nats/bank initialization price, measured on
overall CE, corresponds to roughly **0.7 nats/bank in the answer region**.

## A hypothesis of mine that this falsified

I predicted the opposite mechanism. The programme has repeatedly measured CE decoupling from
capability — C28's CE improving 9.8 nats while its task score stayed at 0/180 — and I proposed
that a converted model posts a cheap CE win by relearning high-frequency, low-entropy
boilerplate while learning nothing about answering.

**That is not what is happening.** The non-reply region is not cheap: its CE is 5.8209, more
than **twice** the reply's 2.6492. It is dominated by the user's novel question, which is genuinely
unpredictable, not by repeated scaffolding. The prompt region is the *harder* region.

So the decoupling still has no mechanism. What this scan establishes is narrower and in the
opposite direction from my guess: overall CE is not being *inflated* by easy boilerplate — it is
**diluting** a real and larger answer-region penalty.

## Limits

- **Initialization only.** No trained checkpoint was scanned, so this does not directly explain
  the trained-model decoupling; it only shows the metric is biased at init. `long_budget.py` now
  records `dev_response_ce_nats` at every milestone, which will test it under training.
- **One split, 40 rows, 381 answer positions.** No seeds — the measurement is deterministic.
- **Historical numbers are not recomputed.** Every CE figure in prior documents remains
  all-position and remains, by this measurement, an understatement.
- The mask locates the assistant marker and the `</think>` terminator in the row's own token
  sequence rather than by re-tokenising a text prefix, because a prefix can split differently at
  the boundary and silently misalign the mask.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/response_ce_scan.py 40
```

~2 min CPU. Artifact: `evidence/response-ce/RESPONSE_CE_SCAN.json`.
