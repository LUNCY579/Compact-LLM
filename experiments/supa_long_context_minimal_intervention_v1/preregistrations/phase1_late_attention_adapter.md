# Preregistration — Phase 1, tiny late-attention adapter

Fixed before either adapter arm was trained.
Protocol version `1.0.0-pilot`.

## Question

Can a tiny adaptation of Supa-A1.1's late attention pathway materially improve
long-context reasoning without touching the catastrophic early stem?

This is causally separate from Phase 0. Training and context refinement are not
combined; Phase 1 evaluates at **full context** only, because the question is
whether the *checkpoint* improved.

## Trainable set

```
theta_train = {q_proj, k_proj, v_proj, o_proj}_{L17, L18, L19}
```

Layers **0, 1, 2** — the catastrophic band in the identity-skip screen — are
never trained, looped, altered or removed. `build_lora_config` raises if any
protected layer is requested; this is verified rather than trusted
(`results/phase1_adapter_verification/receipt.json`,
`protected_stem_refused: true`).

Not trained in Phase 1: MLP/gate/up/down projections, embeddings, the LM head,
norms, any other layer, or any dense backbone parameter.

Layer choice motivation: 17, 18 and 19 are late layers with measurable inference
sensitivity that remain coherent under ablation. **This does not imply the
ablation ranking is a training ranking.** It motivates one cheap candidate
intervention and nothing more.

## Rank

`r = 4`, `alpha = 8`, `dropout = 0.0`. r=4 was chosen because nothing in the
existing infrastructure forces r=8; `peft` 0.19.1 handles r=4 directly. Only one
rank is tested.

Verified module targeting under `peft` 0.19.1: `layers_to_transform=[17,18,19]`
with `layers_pattern="layers"` and suffix-form `target_modules`. Exactly 12 of
196 candidate modules match; matched layer set is `{17,18,19}`; `lm_head` and
`model.embed_tokens` do not match, which also sidesteps the
`tie_word_embeddings=true` aliasing hazard. Fully-qualified target names must
not be used — they take a different code path that bypasses the layer filter.

Trainable budget: **122,880 parameters = 0.0206%** of 596,049,920.

## Arms

* **A** — frozen canonical Supa-A1.1, no training.
* **B** — LoRA as above.
* **C** — NoRA-init: same distribution and same seed for `lora_A`, then each of
  its columns rescaled to unit L2 norm; from there trained exactly as ordinary
  LoRA.

`lora_B` is zero at init in both arms, so both start as an exact functional copy
of frozen Supa.

Matched between B and C: dataset, data order, seed (17), rank, alpha, dropout,
optimizer (AdamW), LR (1e-4), scheduler (cosine, 10% warmup), context length,
accumulation, step count, precision, and gradient clipping (1.0). **Only the
initialization differs.**

Not run: DoRA, LoRA+, continuous NoRA, rank sweep, layer sweep, optimizer sweep,
LR sweep. One seed only; multi-seed confirmation is not licensed unless a
positive result appears first.

## Mechanical checks (executed, receipt on disk)

| check | LoRA arm | NoRA arm |
|---|---|---|
| `lora_A` mean column norm | 0.030929 | 1.000000 |
| max deviation from unit norm | 0.998 (correctly *not* normalised) | 1.2e-07 |
| `lora_B` all zero at init | true | true |
| zero adapter reproduces frozen Supa (max logit delta) | 0.0 | 0.0 |
| trained layers | {17,18,19} | {17,18,19} |
| protected stem untouched | true | true |
| only attention projections trainable | true | true |
| serialisation round-trip delta | 0.0 | 0.0 |
| merge parity delta | 0.0 | 0.0 |

## Training curriculum

`Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl`, train split, whole scenario packs
sampled stratified by task type (seed 20260905). Ten reasoning task types
crossed with base / paraphrase / counterfactual / unanswerable, so the model must
compare rather than look up.

Verified disjoint from the evaluation suite: **0** shared contexts, **0** shared
questions, **0** shared titles.

Two mechanical renderings, both fixed before training:

1. **Protocol alignment.** LCR ships a "brevity is optional" system message and
   16-word sentence answers; the evaluation demands the shortest exact answer
   (2-word median) under strict exact match. Rows are re-rendered under the
   evaluation's own system prompt and user template — already character-identical
   between the datasets — so Phase 1 measures reasoning rather than answer length.
2. **Short target.** For answerable rows the target is the single `must_include`
   span absent from the question. This is exact, not heuristic: it resolves to
   exactly one span on all 6000 answerable rows. Unanswerable rows target
   `UNKNOWN`.

Loss is taken on answer tokens only; every prompt token is masked to -100.

**Sequence length.** `max_length = 2560`. Measured with the actual Supa
tokenizer through this lab's own rendering, LCR examples run ~1958-2169 tokens
(p50 2048). An initial `max_length` of 2048 was found — before any training run
was analysed — to silently drop about half the curriculum; 2560 retains 100%.
This sits at the low end of the 2K-4K guidance, which is accepted deliberately:
matching the evaluation's own length distribution matters more than hitting a
token count, and it is cheaper.

## Budget

Smoke: 6 optimizer steps. Bounded run: 300 optimizer steps, accumulation 8
(2400 examples ≈ 1 epoch over 600 packs), batch size 1.

Tracked: train loss, validation loss on the held-out LCR dev slice, gradient
norms, adapter norm, trainable parameter count, peak memory, wall-clock,
steps/sec.

## Abstention gate handling

`Qwen3AbstainForCausalLM.forward` adds its constant offset *after*
`Qwen3ForCausalLM.forward` has already computed the loss, so the gate never
enters the objective. It also mutates the logit tensor in place, which is an
autograd hazard once a LoRA parameter makes that tensor require grad. The gate
is therefore zeroed for the duration of training — provably a no-op for the loss
— and asserted back on before any evaluation. Both directions are asserted, not
assumed.

## Evaluation

Same frozen suite for all three arms, **full context**.

* **TARGET** — `Rineya_Long_Context_Demo_v1`, 256 rows, the same protocol and
  scorer as Phase 0 condition A.
* **PRESERVATION** — 200 short-form items (100 `arithmetic_chain`, 100
  `counting_sets`) from `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`,
  seed 20260905. Short context, deterministic numeric replies, strictly
  scoreable. A regression tripwire, not a second target.

## Decision gate

An adapter is **meaningfully positive** if

* target gain ≥ **2 absolute points**, **or** relative error reduction ≥ **5%**,

**and**

* average preservation regression ≤ **1 absolute point**.

NoRA-init is separately **interesting** if it beats standard LoRA at matched
budget, or reaches similar quality materially sooner. The full loss trajectory is
reported; no checkpoint is cherry-picked.

## Consequences

* Phase 1 positive → Phase 2 is licensed: the winning method only, comparing
  L17-19 against L17,18,19,25,27, everything else identical.
* Phase 1 null or negative → **STOP**. No layer sweep, no compensating search.
* The 2x2 composition check requires Phase 0 query selection to have won
  independently. It did not, so that check is **not licensed** regardless of
  Phase 1's outcome.

## Claims that will not be made

No claim of statistical significance on this corpus. No claim of intrinsic
capability improvement from a retrieval wrapper. No claim of general capability
improvement from target-only gains.
