# MQ-001 — improve model quality, not architecture novelty

**An isolated, executable GPU-study pipeline plus a staged research plan.**

Branch: `research/model-quality-targeted-gpu-v1`  
Parent: `a36de96b30eb03265db3a18e464a2e2a23ee67bf` (live RAB continuation reviewed on 2026-09-14).

**Delivery boundary:** CPU synthetic mechanics tested; native Qwen tests explicitly skip without Transformers. No official source weights, CUDA model calls, native qualification, GPU optimizer steps, or quality improvement were executed in this implementation environment. See `evidence/LOCAL_RECEIPT.json` for the exact tested scope. A remote commit is publication evidence, not a native execution receipt.

## What is executable now

MQ-001 implements the next **objective × peak-learning-rate screen**, independently of the historical RAB scripts:

| Arm | Loss-bearing positions | Peak LR |
|---|---|---:|
| whole_3e-6 | All next-token positions | 0.000003 |
| whole_1e-5 | All next-token positions | 0.000010 |
| whole_3e-5 | All next-token positions | 0.000030 |
| response_3e-6 | Assistant answer plus EOS only | 0.000003 |
| response_1e-5 | Assistant answer plus EOS only | 0.000010 |
| response_3e-5 | Assistant answer plus EOS only | 0.000030 |

Each cell starts from the exact dense **Supa 1.7.0-devfreeze** checkpoint, trains blocks **8–27**, and freezes embeddings, output-table tying, final normalization and blocks 0–7. The native trainable inventory must be exactly **314,618,880 parameters in 220 tensors**. The published Opallo checkpoint is **not** silently substituted or continued.

All six arms share one immutable selected token stream, row order, full-row update groups, development rows, and tokenizer. Peak LR and objective are the only cell differences. WSD uses 20% warmup / 60% stable / 20% `1-sqrt` decay to 10% of peak; AdamW uses beta1=.9, beta2=.95, eps=1e-8, weight decay=0 and norm clipping=1. The default budget is a **25,000 processed-token ceiling**, with at least 90% coverage and no truncation or silent row skipping. Actual processed and supervised counts are reported separately.

This is a **new training lane**, not a byte-for-byte historical replay: it uses a separate pinned native runtime, token-group accumulation (up to 2,048 full-sequence tokens per update), globally token-normalized loss within each group, and a new dataset. It does not inherit the historical 73/74 panel number, RAB retention policy, or any native PASS.

## Main guarantees and limits

* Native template and generation-prefix equality are checked at both text and token levels. Response masks include EOS and exclude the prompt, assistant header/thinking stub, and post-EOS newline. V1 accepts a single user/assistant exchange with an optional system message; multi-turn, tools, packing and truncation are rejected rather than guessed.
* Chunked, recomputed output-head CE avoids retaining the whole vocabulary-logit matrix. CPU-resident Adam moments reduce GPU optimizer storage. Synthetic tests compare values/gradients with ordinary CE and updates with PyTorch AdamW.
* Create-only checkpoints include trainables, both Adam moments, RNG streams and exact next-update cursor. Only completed update boundaries can be saved. Resume uses a **new attempt directory**, the same source/plan/code/runtime binding, and validates the complete tensor inventory before mutation.
* A source-, dataset-, code-, runtime- and checkout-bound native GPU qualification with **zero optimizer steps** is required before training. It must be generated on the actual source and GPU; fixture records cannot satisfy it.
* Native commands run as one supervised foreground worker with an exclusive cooperative GPU lease, RAM/VRAM guards and a 3,600-second watchdog. No auto-retries, process eviction, network downloads or hidden jobs. Historical runners do not honor this new lock, so operator coordination is still necessary.
* Development evaluation records unmodified outputs, EOS/limit status and a fixed 128-token output budget. Accuracy is assigned only through explicit exact/numeric contracts; prose has no invented semantic score. Literal UNKNOWN and semantic abstention are not conflated. Summaries require a separately qualified evaluator.
* Completed training reports **ENGINEERING_COMPLETE_QUALITY_UNDECIDED**, never automatic success, equivalence, superiority or release permission.

## Read and run

Start with `PLAN.md`, `DESIGN.md`, `AUDIT.md` and `RUNBOOK.md`.

```bash
python -m experiments.model_quality_gpu_v1 environment
python -m pytest experiments/model_quality_gpu_v1/tests -q
```

The intended native path is **prepare → independent data review → qualify → train one cell → compare development outputs → independently register confirmation**. A suitable quality corpus and its review remain pending; synthetic test text is not the training deliverable.

The existing `REPAIR_PREREG.md` experiment and sibling `opallo_behavior_screen_v1` are preserved, not replaced or silently relabeled. Their native completion is recommended first in the roadmap, but MQ-001 does not pretend to execute either one.
