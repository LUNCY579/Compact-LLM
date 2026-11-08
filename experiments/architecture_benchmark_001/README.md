# RAB-001 | CED architecture experiment — HOTFIX-001

**Implementation hardening and recovery preparation, not a model-quality success.**

This package continues the uploaded `experiments/architecture_benchmark_001` snapshot. It preserves the original evidence and source lock, repairs numerical/evidence gates, and adds local restartable qualification plus optional matched **25-step engineering** smokes with cached teacher targets. G3–G5 remain closed. No official model weights or new official-model execution results are included.

Start with **[REVIEW_REPORT.md](REVIEW_REPORT.md)**, then **[RUNBOOK.md](RUNBOOK.md)**. Data recommendations and acquisition instructions are in **[RECOVERY_DATASETS.md](RECOVERY_DATASETS.md)**. Read the explicit protocol amendment **[HOTFIX_001.md](amendments/HOTFIX_001.md)** before running anything.

## What this architecture is

A dense, causal shared-memory ablation inspired by the user's supplied architecture design—not a reproduction of an entire DeepSeek release. The actual source is the exact ungated **Supa Core 1.7.0**, derived from the project's Qwen3-compatible core, not DeepSeek weights. No release-dependent DeepSeek claim is needed to execute this benchmark.

The causal encoder computes token states. One learned normalized K/V projection turns encoder-final states into shared memory. Each decoder block has its own Q/O projections and FFN, reads that shared memory causally, and has **no decoder-local self-attention or private K/V projection**. For tail-only prefill, prompt tokens traverse the encoder but only the last valid token needs the decoder; each generated token still traverses both stacks. This does not introduce variable-depth recurrence, sparse attention, MoE or a new tokenizer.

| Arm | Blocks | Residual width | FFN | Q/KV heads | Parameters |
|---|---:|---:|---:|---:|---:|
| B28 | 28 ordinary | 1,024 | 3,072 | 16/8 | 596,049,920 |
| D40 | 40 ordinary | 896 | 2,496 | 14/7 | 597,263,232 |
| C28 | 14 encoder + 14 decoder | 1,024 | 3,392 | 16/8 | 596,311,424 |
| C40 | 20 encoder + 20 decoder | 896 | 2,752 | 14/7 | 594,279,808 |

Every arm retains head_dim=128, vocabulary 151,936, tied embeddings, Q/K normalization, SwiGLU and RoPE theta 1,000,000. Parameter matching is within 0.5%; it is not FLOP/latency/FFN-shape matching. The analytic C40 persistent-KV ratio to B28 is 0.65625; this is not measured total memory or speed.

## Source and transfer

`source_lock.json` pins every asset, the exact weight file, BF16 tensor inventory and config. Weight SHA-256: `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`. Source directories are read-only. No automatic downloads, stock-Qwen fallback, adapters, custom readout, permissive loading, or source mutation is allowed.

First qualify unmodified native/reference behavior. Then qualify same-width identity expansion, calibrate trained B28 blocks, slice a single global residual basis and whole GQA groups into W28, then expand to D40. C28 converts B28; C40 converts calibrated D40. A declared first-decoder donor initializes shared memory. FFN growth duplicates gate/up rows while zeroing added down columns; that preserves the FFN function initially, **not the changed attention topology**. Lineage records every selection.

## Current executable boundaries

| Command/lane | Boundary |
|---|---|
| `audit`, `status`, `matrix` | Source availability, runtime and analytic shapes; not qualification |
| `qualify-source` | G0: exact source, native/reference parity, open parser/termination pilot |
| `qualify-identity`, `calibrate-d40`, `convert-ced` | G1: conversion and runtime mechanics; not quality recovery |
| `diagnose-numerics` | Bounded FP32/FP64 observations from saved checkpoints, including failed historical ones; no gate authorization |
| `smoke` | Preserved legacy D40 G2: 25 optimizer engineering steps |
| `cache-teacher` | Offline normalized teacher features, exact token/source/data bindings; no training authorization |
| `smoke-cached` | G2_CACHED: 25 steps; C28/C40 require matching B28/D40 controls |
| `qualify-quality` | G1_QUALITY: an arm's output vs the frozen source on open splits; catches token salad that every construction gate accepts |
| `qualify-retention` | G1_RETENTION / G2_RETENTION: typed answer scoring on the held-out probe, against an immutable reference; the only capability-side gate |
| `prepare-data` | Group-first, source-declared, short-context data preparation; no download or training |
| `recovery-readiness` | Reports missing preregistration fields; always keeps G3–G5 blocked |
| `python -m ...architecture_benchmark_001.ladder` | Foreground local process-isolated ladder; validated resume, failed attempts preserved |

Quality and retention are deliberately separate from construction: an arm can be structurally perfect, pass the full parity grid with zero violations, and answer none of the questions its donor answers. Likelihood is not a substitute — at a token-matched 500k dose a CED arm reached held-out CE 5.11 nats better than its own teacher while answering 2/60 against the teacher's 20/60, and D40 has better cross-entropy than the source while scoring 0/150 on the retention probe.

Strict CED elementwise parity is mandatory. Argmax agreement and a global-scale bound do not replace it. Schema-1 historical receipts are retained but cannot authorize the new code. Schema-2 parents are recursively validated. Rerun after code, protocol, data, source or runtime changes. A passing 25-step smoke is never a recovery result.

## Install and inspect

Use a fresh environment and the pinned native requirements. Select the correct official PyTorch wheel for the intended device; the runbook contains CPU and CUDA cautions. From the ChatBubble repository root:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/architecture_benchmark_001/requirements-native.txt
python -m pytest -q experiments/architecture_benchmark_001/tests
python -m experiments.architecture_benchmark_001 matrix
python -m experiments.architecture_benchmark_001 status
python -m experiments.architecture_benchmark_001 audit-evidence
```

Native random-Qwen tests must execute, not skip, before claiming native compatibility. The delivery's verification files explicitly record what this CPU-only environment could and could not run. Missing Transformers is not silently replaced with a random or stock official model.

## Reading historical evidence correctly

The six uploaded receipt JSON hashes and included small artifact hashes are internally consistent. Four referenced checkpoint binaries are absent from the ZIP. Three of four recorded CED comparisons fail the strict elementwise rule even though top-1 tokens matched. The prior claim that this establishes a benign architectural effect, rather than a possible defect, is not supported by that test alone. See `evidence/REVIEW_CORRECTIONS.md`.

Original historical markdown and JSON are retained **byte-for-byte**, including their old PASS labels and obsolete reproduction commands. They are historical records, not the current execution policy. The uploaded code matches the historical implementation digest when its Windows path-key convention is reproduced; a POSIX-key digest differs for an understood reason.

## Research continuation

Structural recovery by teacher distillation is a reasonable next research phase once runtime qualification and data/resource registration are complete. Compare B28 continuation, D40 continuation, C28 and C40 with a common frozen teacher and explicit equal-budget controls. G3 still requires the recovered-ordinary-donor plan, three registered seeds, frozen disjoint manifests, budgets, meaningful effect/regression margins and a separately reviewed campaign runner. Neither a larger dataset nor a relaxed numeric gate substitutes for those controls.

Background design references retained in the original materials: Minitron, https://arxiv.org/abs/2407.14679; MobileLLM, https://arxiv.org/abs/2402.14905. Neither establishes efficacy for this specific inherited CED conversion.
