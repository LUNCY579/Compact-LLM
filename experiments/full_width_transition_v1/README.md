# FWT-001 — Full-width, capability-preserving memory transition

**Implementation and bounded mechanical research, not a model-quality result.**

New isolated branch: `research/ced-full-width-transition-v1`.
Parent: `aa9bdcd1ff805066b9566230006542704f8b9321` of ChatBubble.
All changes are under `experiments/full_width_transition_v1/`. Historical
RAB-001 code, source locks, data, receipts and branches remain unchanged.

## Decision

Establish an architecture at the source's original dimensions before combining it
with width compression. More parameters are acceptable for this phase. But original
width alone is not a remedy: the earlier original-shape 14+14 CED conversion also
failed its initialization diagnostic. This experiment changes the transition,
not merely the name or width of that failed arm.

The order is:

1. Full-width 28-layer capability-preserving augmentation and useful-pathway test.
2. Full-width D40 as a separate, identity-initialized depth control.
3. Full-width C40 only after the transition and depth questions are interpretable.
4. Width/FFN/head compression only after a useful full-width endpoint exists.

The research rationale, controls, hypotheses, interpretations and deferred work are
in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Commands and boundaries are in
[RUNBOOK.md](RUNBOOK.md). Machine-readable rules are in `protocol.json`.

## What is implemented

The exact official ungated Supa 1.7.0-devfreeze source remains the donor. Its
residual width is **1024**, FFN width **3072**, query/KV head counts **16/8**, and
head dimension **128**. Do not infer head dimension as 1024/16: the query projection
is 1024 to 2048. Vocabulary, tied embeddings, RoPE and normalization are unchanged.

| Arm | Purpose | Total parameters | Added trainable parameters in the smoke |
|---|---|---:|---:|
| B28 | Frozen ordinary source / teacher | 596,049,920 | 0 |
| M28 | Original backbone plus causal shared-memory reads | 656,884,608 | 60,834,688 |
| P28 | Exact parameter-matched tokenwise gated control | 656,884,608 | 60,834,688 |

M28 builds one normalized K/V bank after original layer 13 (zero-indexed), then
adds a read after each intact original block 14 through 27. Every original
self-attention, private K/V projection and FFN remains. Added Q and memory K/V
weights are initialized from declared donors; only added output projections are
zero. No extra zero scalar gate suppresses their gradients.

P28 has the exact same added tensors and initialization. Its added computation is
`sigmoid(q_t dot k_t / sqrt(head_dim)) * v_t`, using only the same token's
encoder-derived memory. It has no added cross-token read, but the original backbone
and each encoder state remain contextual. It is **not a context-blind model**.
A one-key softmax is deliberately avoided because it makes Q/K gradients zero.
This control matches parameters, not FLOPs or the softmax/sigmoid inductive bias.

Both variants implement full forward, request-local cached decoding, chunked
prefill, causal/padding/packed-segment masks, checkpointed gradients, create-only
safetensors checkpoints and stale-cache rejection. Tail-only mode saves output
projection work only; it does not skip the original upper attention stack.

**Neither variant is pure CED or cache-saving.** Both retain 28 ordinary cache
banks plus the additional bank. An identity-initialized model is not an efficacy
result; an augmentation that cannot replace original attention is not proof of
pure CED.

## Executable boundaries

| Lane | Implemented behavior | What a pass authorizes |
|---|---|---|
| `inspect` | Shapes, budgets, dependency hashes and environment | Nothing |
| P1 `qualify` | Exact source assets, native/reference parity, zero-path identity, reload, active-path cache/mask/causality grid | Only the separately requested fixed P2 smoke |
| P2 `smoke` | Both M/P arms, same seeded order, 25 steps each, added-only AdamW, CE plus frozen-teacher KL | No subsequent stage |
| P3+ | No campaign runner or command | Closed |

P1 explicitly activates output projections temporarily at a fixed scale and checks
the nonzero path. Zero-initialization cannot conceal a broken attention or cache
implementation. P2 checks all added read-Q gradients plus shared K/V gradients,
nonfinite values, original-backbone immutability and post-step cache/causal parity.

The P2 objective is next-token CE over the short engineering paragraphs plus
`0.5 * T^2 * KL(teacher_T || student_T)`, with T=2. At identity the KL gradient is
zero; CE supplies the initial learning signal. This is not a capability-retention
certificate, a general replay corpus, or registered recovery training.

## Current evidence

The delivery includes scoped CPU tests on small random weights, full-shape
parameter counting on meta tensors, and a paired 25-step tiny-model engineering
exercise. See `verification/LOCAL_VERIFICATION.json` for the measured local result.

**No official weights were available in the implementation environment and
Transformers was absent. No native official P1, official P2, recovery, DEV, sealed
evaluation, GPU-fit, speed, intelligence or deployment result is claimed.**
The official loader and qualification runner must execute on the locked source
before the new branch can advance. Synthetic tests cannot authorize official work.

## Minimal inspection

From the repository root:

```bash
python -m pytest -q experiments/full_width_transition_v1/tests
python -m experiments.full_width_transition_v1 inspect
```

Do not modify `architecture_benchmark_001` to make this package pass. Its two
imported Python files and source lock are pinned independently in
`dependency_lock.json`. Source-code hashes normalize CRLF to LF for Windows
checkouts; model/tokenizer asset hashes always use exact raw bytes.
