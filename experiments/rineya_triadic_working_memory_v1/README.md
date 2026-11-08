# TWM-001 — Rineya Triadic Working Memory

**Dedicated experiment:** `research/rineya-triadic-working-memory-v1`  
**Base:** `post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`  
**Status:** reference implementation tested; actual Rineya checkpoint/CUDA qualification is not yet established.

## Question

Can a frozen Rineya 0.6B model keep overlapping associations distinct, update the intended fact, and compose retrieved facts more reliably with an additional triadic gated-delta memory? Compare against the unchanged model, a parameter-matched stateless adapter, ordinary gated-delta memory, and a recurrent-state-matched larger matrix memory.

This is working memory for the current sequence, **not permanent skill acquisition without forgetting**. Neither more state nor passing software tests proves greater intelligence. No production defaults, model weights, existing experiments, or existing sealed data are changed by this package.

## Implemented

- Pure PyTorch tensor and independent flattened-matrix delta-rule implementations, with all-input gradient checks. The correction residual is computed **jointly across all slices**.
- FP32 recurrent memory with normalized first/second addresses, per-slice decay, causal convolution, explicit state/history caches, write masks, document resets, padding handling, and allocation limits. FP64 is available for numerical tests.
- One identity-initialized residual sidecar after a native decoder layer. Its scalar gate starts at zero while its output projection remains nonzero, preserving a trainable gate. The original softmax attention, RoPE, GQA, MLPs, and KV cache remain intact.
- Frozen Qwen3 hook wrapper, combined native/sidecar cache ownership, stale-cache rejection, greedy cached decoding, prefill/decode timing, and explicit unsupported-mode guards. Native integration targets **Transformers 4.57.1**, not an untested latest version.
- Local-only checkpoint/tokenizer SHA-256 locks; no automatic model downloads. Native training/evaluation require an explicit execution flag and a successful source/config/checkpoint/device-bound native qualification receipt.
- A bounded sidecar-only training loop, answer-token supervision, optional same-prompt teacher KL, gradient clipping, frozen-weight checks, adapter-only checkpoints, and new-directory-only run outputs.
- Fresh procedural binding, correction, unrelated-fact retention, scope, two-/three-hop, missing-fact, conflict, role-state, and formatting cases. Optional direct-edge probes distinguish retrieval from composition. Oracle prompts expose supporting facts, not an asserted final answer.
- Exact-match evaluation, paired cluster bootstrap intervals, repair/damage accounting, causal memory ablations, offline within-document second-key shuffling, resource accounting, and effective-rank diagnostics.

## Controls and actual pilot sizes

The checked pilot has hidden width 1024, four memory heads, key/value dimensions 64, one insertion point after zero-based layer 13, and convolution width four.

| Arm | Added trainable parameters | FP32 memory + convolution buffers, batch 1 |
|---|---:|---:|
| `baseline` | 0 | 0 B |
| `stateless` | 1,106,945 | 0 B |
| `gdn_e1` | 1,060,937 | 74,752 B |
| `triadic_e4` | 1,106,133 | 271,744 B |
| `matrix_state_matched` | 2,639,945 | 289,792 B |

The stateless parameter difference is below 1%; all its parameters are active, with no padding parameters. The matrix control matches **recurrent tensor elements**, not parameter count or total convolution buffers. Report these distinctions; do not claim every budget is matched simultaneously.

These buffer sizes exclude native weights, original KV cache, optimizer state, activations, workspaces, and gradients. A sidecar does **not** remove the native KV cache. Training uses a sequential reference implementation, not the authors' H100 kernels, and may be slow on a laptop.

## Files

`core.py` implements the equations and offline address-shuffle utility. `sidecar.py` provides the recurrent and stateless arms. `native.py` owns the frozen wrapper and combined inference cache. `data.py` provides fresh data and split guards. `provenance.py` locks checkpoint files and checks resources. `runner.py` exposes the executable workflow. `analysis.py` provides diagnostics and paired comparisons. Tests distinguish structural test doubles from optional actual tiny Qwen3 checks.

Read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) before interpreting results and [RUNBOOK.md](RUNBOOK.md) before a native run. Source references and pinned commits are in [SOURCE_LOCK.json](SOURCE_LOCK.json).

## Quick CPU check

Run from the repository root with an existing CPU or CUDA PyTorch installation:

```console
python -m pytest -q experiments/rineya_triadic_working_memory_v1/tests
python -m experiments.rineya_triadic_working_memory_v1.runner smoke --out experiments/rineya_triadic_working_memory_v1/artifacts/reference_smoke.json
```

The smoke command runs a tiny fixed-tensor prefix-mean regression through all arms. It tests optimization mechanics only. It does not load Rineya and does not evaluate language or reasoning.

The recorded execution produced **70 passed, 4 skipped**. The four skipped tests require Transformers 4.57.1 and an actual tiny random Qwen3 architecture. The current working environment had CPU PyTorch 2.10.0, no CUDA, no Transformers, and no mounted Rineya checkpoint; dependency installation was blocked by network/DNS availability. See [evidence/QUALIFICATION_RECEIPT.json](evidence/QUALIFICATION_RECEIPT.json) and the raw JUnit evidence.

## Boundary of this delivery

Implemented: reference mechanism, controls, synthetic correctness/optimization tests, native integration code, guarded native training/evaluation, fresh diagnostic data, and reproducibility/reporting infrastructure.

Not yet established: tiny-Qwen3 execution in the pinned dependency environment; native 0.6B cache parity; real-checkpoint learning gains; multi-seed retention; RTX 4070 memory/latency; long-context performance. The pilot is capped at 512 total input/output tokens and cannot substantiate 64k claims.

Future conditional work: feedback-conditioned multiple reads, a learned read controller, safe read-only deliberation, optimized kernels, and gradual attention substitution. They are not silently included in the first acquisition experiment. Preserve the existing TGSA, FMA, CSS, and recurrence branches; this branch does not replace their evidence or ownership.
