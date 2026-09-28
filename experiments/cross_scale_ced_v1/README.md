# XSC-001: native-width E28 / D36 cross-scale causal-memory hybrid

**Implemented mechanics and bounded engineering pipeline. No official pretrained execution or efficacy result.**

Branch: `research/ced-cross-scale-e28-d36-v1`  
Parent: `cd6ef9bd49724da6879bfa4d945af3bfaa6d9758` (ChatBubble dual28)

All additions live in `experiments/cross_scale_ced_v1/`. Existing dual28, full-width transition, RAB-001, source assets, historical evidence, workflows and other refs are not changed. This is not a reproduction of a complete named DeepSeek architecture.

## Construction

The encoder is the exact ungated Supa Core 1.7.0-devfreeze source: 28 layers, residual width 1024, FFN 3072, Q/KV heads 16/8, explicit head dimension 128. The decoder is exact `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`: 36 layers, residual width 2560, FFN 9728, heads 32/8, head dimension 128.

Both complete ordinary backbones keep their own embeddings and final normalization. Only the decoder's tied vocabulary table produces logits. A learned memory writer reads the normalized encoder states. Every complete decoder block is followed by an added memory read. No decoder self-attention or FFN is removed. Query projections and query/input norms initialize from the corresponding decoder block, in independent tensors. Writer projections initialize from seeded N(0, 0.02); writer norms start at one. Read output projections start at **exact zero**. There is no extra scalar gate whose zero could block the output-projection gradient.

| Arm | Meaning | Unique parameters | Added/trainable parameters |
|---|---|---:|---:|
| B4 | Ordinary Qwen3-4B on the same decoder-visible input | 4,022,468,096 | 0 in engineering |
| H | Full E28 / D36 with shared causal-memory reads | 5,375,687,808 | 757,169,792 |
| P | Exact H tensor inventory, same-token sigmoid memory control | 5,375,687,808 | 757,169,792 |

These are analytic counts cross-checked on real module inventories on **meta**, not measured allocations. H/P retain 65 KV banks (28 encoder + 36 decoder + one shared memory). Their BF16 weights alone are 10,751,375,616 bytes; BF16 persistent KV at 8192 tokens and batch one is 2080 MiB, excluding activations and allocator overhead. Neither implies GPU fit, training cost or speed.

P is **contextual** through both encoder and decoder. It is not context-blind. A sigmoid same-token gate, rather than one-key softmax, preserves Q/K gradients. H/P match parameters, initialization, data order and optimizer policy, **not FLOPs**. Decoder-width queries are projected to 32 x 128; encoder-width states are projected to 8 x 128 K/V. Hidden width is never inferred from attention width or head count.

## Executable boundary

- `inspect`: configuration, budget, dependency verification and runtime availability; no weights.
- `tiny-qualify` / `tiny-smoke`: explicit non-authorizing random CPU fixtures. No official receipt.
- `qualify`: exact-source, native-tokenizer/native-Qwen, active-memory, causal, cache, gradient, frozen-weight and checkpoint qualification. Requires full clean checkout, pinned CPU runtime and 64 GiB available physical/cgroup RAM. Outputs a new P1 receipt only if everything passes.
- `smoke`: separately requested, P1-bound **25 steps per H/P arm**, new interface parameters only. Fixed engineering paragraphs, next-token CE plus baseline KL, matched initialization/order, complete gradient coverage and source-bound interface checkpoints. **P2 licenses nothing.**

There is no recovery, general training, DEV/TEST, pure-CED conversion, quantization or deployment command. Source restoration/download is not automatic. No sealed bank is read by this package.

The implementation uses an **aligned physical token grid**: normally both stacks see the same causal stream, including each newly generated token. Optional separate encoder IDs are an explicit mechanical intervention on that same grid. This is not a general unequal-length encoder-decoder API, prompt-once encoding, a compressed cache, or a Hugging Face `generate()` serving implementation.

## Evidence

See `verification/LOCAL_VERIFICATION.json` for the exact tested-file hashes, test scope and missing native runtime. Local verification covers 2+4 and 28+36 random stacks at tiny widths, including different Q/KV counts and head dimensions between backbones. Full official shapes are counted on meta only. Official weights are absent in the implementation environment. Native random-Qwen tests explicitly skip when Transformers is absent; those skips never satisfy P1.

Read `DESIGN.md`, `RESEARCH_PLAN.md`, `RUNBOOK.md`, `protocol.json` and `sources.json` before continuing.
