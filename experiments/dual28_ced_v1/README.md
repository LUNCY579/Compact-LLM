# D28-001 — Two full-width pretrained stacks

**Implementation and bounded mechanical experiment; no pretrained result or model-quality claim.**

Branch: `research/ced-dual28-full-width-v1`.
Parent: `94fbe17945a9d7d88ac2422dab124a95d35ee653` (ChatBubble).
All additions are confined to `experiments/dual28_ced_v1/`.
The existing FWT-001 and RAB-001 experiments, receipts, source assets and branches are unchanged.

## User's research decision

Build a **28-layer causal encoder plus a 28-layer decoder** from two independently
copied instances of the same official Supa checkpoint, instead of allocating half
of one model to each role. Keep residual width **1024**, FFN width **3072**, query/KV
heads **16/8**, explicit head dimension **128**, vocabulary, RoPE and normalization
epsilon. Architecture viability comes before compression. Parameter growth is
accepted, but is not itself evidence of improved intelligence.

This is a dense shared-memory research design inspired by the user's proposal,
not a reproduction claim for a complete named DeepSeek release. The pure endpoint
has no decoder-local SWA, MoE, MLA, sparse indexing or custom serving kernels.

## Implemented arms

| Arm | Construction | Unique parameters | KV banks |
|---|---|---:|---:|
| B28 | Unmodified official reference/teacher | 596,049,920 | 28 |
| H56 | Full E28 and ordinary D28, plus causal memory reads | 1,156,087,424 | 57 |
| P56 | Exact H56 tensor inventory; same-token sigmoid memory control | 1,156,087,424 | 57 |
| D56 | Ordinary depth control, source 28 followed by 28 identity-initialized blocks | 1,036,516,352 | 56 |
| C28 | Separate 14+14 memory-only split diagnostic at unchanged width/FFN | 568,786,304 | 15 |
| C56 | Serial E28 -> D28 memory-only construction | 979,890,816 | 29 |

Counts are checked against real module inventories on the meta device. They are
not measured GPU-fit, speed or quality results. H/P each have **119,571,072**
trainable interface parameters in the engineering smoke. Their full original
encoder blocks, decoder blocks, embedding and output normalization remain frozen.

## The two architectures must not be conflated

**H56/P56:** the encoder processes the full causal token stream. A shared K/V bank
is projected from its final residual states. The decoder starts from the original
token embeddings and retains all 28 self-attention/FFN blocks. A new read is added
**after each complete decoder block**, including its FFN. Only new read output
projections start at zero, giving the intended B28 function at initialization.
The new bank's normalization copies the source final norm; K/V and K normalization
copy source layer 0. Read Q and norms copy the corresponding decoder donor layer.
The bank normalization is part of the trainable interface, not a second frozen
encoder output head. All block tensors are independent copies; one embedding
Parameter is shared by both input paths and the tied output projection.

**C56:** the decoder instead starts from the final encoder residual stream. Its
28 blocks retain donor Q/O, norms and FFNs, but read shared encoder K/V rather than
private decoder K/V. There is no retained decoder self-attention and no additive
read branch. This is deliberately **not function-preserving** and is not the
trained H56 with a flag flipped. Reaching it from H56 would change both attention
and the decoder input stream; that migration is not implemented or authorized.

P56's added branch reads only the same token's contextual encoder memory using a
sigmoid gate. It is **not a context-blind model**. H/P match unique and trainable
parameters, data order and optimizer policy, not FLOPs or every inductive bias.

## Executable boundary

`inspect` reports locked dependencies, runtime and analytic budgets without weights.
`qualify` performs new source/runtime/checkout-bound **P1** checks, including native
parity, every donor tensor, H/P and D56 identity, active causal/cache/mask grids,
pure constructions and sequential H/P/C checkpoint roundtrips.
A P1 PASS licenses only a **separately requested 25-step P2 engineering smoke**.
`smoke` runs H and P for 25 steps each on new engineering paragraphs, using added-only
AdamW, next-token CE plus teacher KL, complete gradient coverage and frozen-weight
checks. **P2 opens nothing.** There is no recovery, DEV, TEST or deployment command.
`tiny-smoke` runs the same engineering code on a 2+2 random fixture and produces
explicitly non-authorizing evidence, never an official P1/P2 receipt.

Read [RESEARCH_PLAN.md](RESEARCH_PLAN.md), [RUNBOOK.md](RUNBOOK.md), `protocol.json`
and [verification/LOCAL_VERIFICATION.json](verification/LOCAL_VERIFICATION.json)
before continuing. Tests include random 28+28-layer modules at small width as well as 2+2 fixtures.
Neither replaces a full-checkout run with
real official weights. The implementation environment lacks those weights and
Transformers; official execution, recovery, efficacy and GPU fit remain unproved.
