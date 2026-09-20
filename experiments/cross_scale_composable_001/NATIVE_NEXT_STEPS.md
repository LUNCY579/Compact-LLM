# Proposed native work — no launch authority

No pretrained model, native tokenizer, optimizer, or behavioral probe was executed
in this first pass. The following is an execution proposal, not a launcher or grant.

## Smallest useful next qualification

After fresh explicit authority, qualify the frozen 32-prompt calibration in
CALIBRATION_BINDING.v2.json and perform **one sequential three-capture session**:

| Operation | Model forwards | Generation | Optimizer steps |
| --- | ---: | ---: | ---: |
| Supa base, 32 prompts, three post-block hooks | 32 | 0 | 0 |
| Qwen3-4B base in the existing qualified NF4 runtime, same prompts | 32 | 0 | 0 |
| Same Qwen base + existing QWEN-FT-001 adapter | 32 | 0 | 0 |
| CPU least-squares maps, controls, MAP_CHECK and residual receipts | 0 | 0 | 0 |
| **Maximum model-forward total for this proposal** | **96** | **0** | **0** |

There are 32 canonical prompt encodings, reused only after identical tokenizer and
rendering bindings are verified. No extra warm-up, smoke generation, retries, or
qualification forwards are hidden in the count. If loader validation requires such
calls, amend the envelope first. Tokenizer/template eligibility is checked before
model loading. Fail overflow at 256 tokens; no replacement sampling.

This cheaply tests whether a known donor skill displacement is nonzero and can be
transported through a map that predicts heldout student representations. It does
**not** show that a trained student acquired a skill. The low fit rank is deliberate.

## Smallest experiment that can test actual skill transfer

After Stage C and all bindings pass, the smallest informative behavioral experiment
is a paired **S0 ordinary adaptation versus X0 donor-residual adaptation**, with the
same 384 training rows, fixed 64 optimizer steps per arm, seed, rank, modules and
evaluation bank, plus reciprocal norm-matched exports. A single X0 result cannot
answer the question. This is a limited first contrast, not the full six-arm pilot
and not a composability claim. Cross-draw acquisition still needs (net, nd, exact p)
and |net|>=20; replication remains necessary.

Exact additional native work for that contrast, assuming the registered batch-1
recipe and already qualified runtime:

1. Freeze/reconstruct the 384 training prompts and token/row order. Capture donor
   base, donor skilled and student base at the required training layer: 384 each,
   **1,152 forward calls**, sequentially, no optimizer. Persist targets as hashes.
2. Train S0 and X0 separately: **64 forward/backward optimizer steps each**, 128
   total. X0 adds frozen target residual MSE to the same CE. No teacher lives beside
   the student. Unused training rows in the bounded 64-step prefix are identical
   across arms and must be reported; 384-row availability does not mean all were seen.
3. Export raw S0/X0, scalar grid S0, X0 matched to selected S1, and reciprocal S0
   matched to X0. Exporting transforms requires no model calls. Effective ranks,
   parameter budgets, component norms and alpha are bound before probes.
4. Score each distinct export on the **same registered bank and complete exact IDs**.
   Required probe families: SELECT acquisition, retention, applicable abstention,
   and existing long-context endpoints. The exact prompt/token totals and decoding
   bounds must be frozen from their source-bound manifests in a new Stage D
   execution amendment; they are intentionally not invented here. Until then the
   evaluation call count is unresolved and Stage D is blocked.
5. To complete full Stage D, bind a genuine independent protected basis, qualify X2
   and O0 plus R0 controls, and evaluate the remaining distinct exports. No generic
   base vector may stand in for a protected skill. Stage E additionally requires
   independent family A/B training and fresh registrations; the mixed donor adapter
   is not two independently acquired skills. Stage F repeats the registered design
   for at least three seed pairs, sequentially.

## Resource estimates and gates

These are planning estimates, not measured process peaks. GiB means 2^30 bytes.
The next grant must bind and measure the actual runtime; refusal is the correct
outcome if a bound cannot be met. Never lower an inherited memory guard.

| Stage | Tensor/storage lower bound | Proposed planning allowance |
| --- | --- | --- |
| A/B offline fixtures | Tiny NumPy matrices; no model tensors | <1 GiB process headroom; measured fixture usage recorded separately |
| C student capture | 596,049,920 weights: 1.110 GiB BF16 or 2.220 GiB FP32 | At least 7 GiB free host before launcher; estimate 3–5 GiB VRAM depending on qualified precision |
| C donor NF4 capture | ideal 4-bit payload 1.873 GiB, plus quant metadata, unquantized tensors and 0.123 GiB FP32 adapter | Plan 12 GiB free host and 6–8 GiB VRAM until a measured loader bound exists; original native guards still apply |
| C donor BF16 alternative | 7.492 GiB weights alone (FP32 14.985 GiB) | Not the registered first arm; likely 12–18 GiB host and 10–12 GiB VRAM, separate grant/qualification |
| C CPU transport | One FP64 2560×1024 map =20 MiB; three =60 MiB; 24×width design matrices are small | 1 GiB headroom for solver workspace; process one pair at a time |
| C persisted captures | Three layers ×32 prompts ×(1024+2560+2560)×4 bytes =2.25 MiB at last-token pooling | Stream and hash; all-token alternative at256 tokens would be0.5625 GiB and is not first arm |
| D training target capture | One layer ×384×(1024+2560+2560)×4 =9 MiB at last-token pooling | Same sequential loader bounds as C |
| D student rank8 final4 q/v LoRA | 163,840 FP32 trainable scalars; params+grads+two moments =2.5 MiB plus base | Plan 7–9 GiB free host and4–6 GiB VRAM at256 tokens, qualify actual precision/optimizer placement |
| D/E probes and merges | FP32 student base2.220 GiB plus one delta/streamed export; KV depends on length | At least inherited7 GiB launcher and6 GiB child guards; increase for measured requirements |
| E independent skills | Same per-student bound, or same per-donor bound if fresh donors trained | Sequential processes; never infer that two skills require co-loading two models |
| F replication | Same per-run bound as D/E | More total time/disk, not permission for concurrent models |

For long-context planning, unshared BF16 Supa KV storage is
`2 ×28 layers ×8 KV heads ×128 head_dim ×tokens ×2 bytes`:
0.875 GiB at8192 tokens, before activations/logits/runtime. Qwen36-layer equivalent
is1.125 GiB. Registered probe implementation and cache policy may differ; measure
them. A low adapter-state cost does not imply low total host RAM.

## Preconditions and stop conditions

Before any proposed native session: check exact committed source and clean tree;
rehash assets and calibration; qualify prompt tokens, hooks and module inventory;
bind donor quantization/runtime; inspect the terminal, grant, journal and shared
`rineya-gpu-0.lock`; obtain a new bounded execution grant with deadline/call limits;
and pass launcher plus child RAM guards. Acquire the existing shared lease through
its qualified owner protocol. Do not reinterpret an existing lock as stale.

Stop on missing authority, occupied lease, insufficient RAM, stale/dirty source,
wrong bytes, invalid/overflowing prompt, unexpected call count, nonfinite tensor,
degenerate residual, failed child, or incompatible runtime. Preserve failure receipts.
No automatic retries, precision changes to force launch, package changes to pinned
environments, or termination of unrelated applications.

DOSELORA is separate. Its2026-09-20T09:39:05Z envelope expired; wait-002 is terminal.
This study does not start wait-003, inspect model endpoints, alter its registration,
or consume its grant. A genuine external RAM change and fresh bounded authority
are prerequisites for its independent runbook.
