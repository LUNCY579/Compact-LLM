# Architecture patterns: what the whole programme actually shows

Synthesis across every structural edit measured in RAB-001. This is the pattern layer above
the individual results; each claim links to the measurement that supports it.

All numbers are from the frozen official Supa 1.7.0-devfreeze source, CPU FP32, open splits.
No efficacy claim, no sealed evaluation.

---

## 1. Four structural edits, measured

| edit | what changes | CE vs source | generation | verdict |
|---|---|---:|---|---|
| **Depth expansion** (identity, 28→40 layers) | nothing — identity maps inserted | **0.0000** (bitwise) | unchanged | **free** |
| **Width reduction** (1024→896, `calibrate_d40`) | every layer's hidden width | 5.36 (**better** than 5.55) | **degenerate `</think>` loop** | **broken** |
| **KV sharing** (CED, local, any topology) | which K/V a layer attends to | +0.5 nats/displaced layer *at init*; ~0.09 after 5.2k tokens | recovers fully at 3 displaced, not at all at 13 | **priced** |
| **Self-attn → shared memory** (full CED) | 13–14 layers displaced at once | 12.59 (**worse than uniform** 11.93) | token salad; 0/180 even after training | **destroyed** |
| **Per-layer low-rank KV** (latent, whitened) | how the same K/V is stored — displaces **nothing** | **−0.006 at 0.250× KV** | unchanged, 19/60 vs source 20/60 | **works** |

Uniform-distribution reference: `ln(151936) = 11.9312`.

---

## 2. The organising principle

**A structural edit is free exactly when it preserves every layer's input distribution.**

Identity depth expansion inserts layers that compute the identity, so no existing layer sees
anything new — and it is *bitwise* exact, `max_abs_logit_error = 0.0`. That is the only
lossless edit found in this codebase, and it proves losslessness is achievable rather than
merely hoped for.

Every destructive edit here violates that principle, and the damage scales with **how many
layers** have their input distribution changed:

- **CED `last N`**: N−1 layers are asked to attend to a bank they did not produce.
- **Local cross-layer sharing, group g**: 28 − ⌈28/g⌉ layers are.
- Both schemes therefore price identically, at **~0.5 nats per displaced layer at
  initialization** (~0.09 after a 5.2k-token matched recipe), and the
  measurement confirms it: CED at 13 displaced gives CE 12.5854, local sharing at 14 gives
  12.9136 — two structurally unrelated schemes on one curve.

**Every removed KV bank displaces exactly one layer, by construction** — a layer that does
not produce its own K/V is simultaneously a saved bank and a displaced layer. So
CE-versus-displaced-count *is* CE-versus-KV-saving for **any** sharing topology. There is no
clever wiring waiting to be found, and the search for one is closed.

**The escape is to stop sharing.** The law prices *displacement*, not KV reduction. A design
that compresses each layer's own K/V displaces nothing and is outside the law entirely — which
is exactly what the latent result measures (§6). The closed search was over sharing topologies,
not over KV reduction.

---

## 3. Two distinct failure modes, and why one metric cannot gate them

This is the most useful structural finding, and it was nearly missed.

| failure mode | example | teacher-forced CE | autoregressive generation |
|---|---|---|---|
| **Attention-target displacement** | C28, C40, local sharing | **collapses** (12.59, worse than uniform) | token salad |
| **Autoregressive collapse** | D40 width reduction | **looks fine** — 5.36, *better* than the source | degenerate `</think>` loop |

D40 is the important case. Width reduction is a *calibrated lossy projection*: it preserves
the teacher-forced conditional well enough that CE actually improves, while destroying the
model's ability to run on its own output. The displaced-layer law does **not** extend to it —
all 40 layers have changed input widths, which would predict catastrophe, and CE says the
opposite. Width reduction is a different kind of edit and fails in a different regime.

The consequence is concrete and was demonstrated, not argued: **a CE-only gate passes D40
outright** (its `ce_increase_within_margin` check PASSES at −0.1308). Only KL-to-source and
the nondegeneracy/termination checks catch it. That is why `G1_QUALITY` has five checks
rather than one, and why an absolute nondegeneracy floor of 2 was too weak — a `</think>`
loop still produced 4 distinct strings across 24 prompts.

---

## 4. Numerical conditioning is a separate axis from quality

CED's strict FP32 cache-order parity failure is **real, CED-specific, and mechanistically
explained** — and almost irrelevant to whether CED works.

- FP64 is clean for both arms (B28 3.8e-13, C28 4.1e-12), so there is no logic, causality,
  masking or cache bug. `causal_suffix_isolation_exact` holds for every arm, row and
  precision.
- Divergence starts at **encoder layer 0** and amplifies ~1000× through depth, so the shared
  bank already carries dV = 4.6e-04 — 4.6× outside tolerance — *before any decoder read*.
- B28's per-layer drift matches C28's encoder drift digit-for-digit, and B28's final layer
  drifts **more** (8.8e-03 vs 3.9e-03), yet B28's logits show 4 violations against C28's
  401,863. **CED creates no extra drift.**
- The difference is the path to the logits: a plain model's drift reaches the final RMSNorm,
  which suppresses the correlated component (8.8e-03 hidden → 1.7e-04 logit). CED freezes the
  *unnormalised mid-network* output into a shared bank and re-injects it into all decoder
  layers **before** that norm.

So the property is: **any architecture that exposes a mid-network tensor to a cache-order
comparison forfeits final-norm suppression.** That is a design consequence, not a defect, and
no read-side or bank-projection precision fix can recover it — both were tried and falsified,
and full-context reconstruction is ruled out analytically (it needs an O(T²) encoder re-run).

Chasing a 1e-4 logit tolerance on a model at perplexity 292,267 was the wrong order of work.
The parity result is retained because it rules out routes, not because it gates anything.

---

## 5. Training rescues a *small* displacement completely, and a large one not at all

> **Corrected.** This section previously read "Training does not rescue a displaced
> initialization", generalising from the C28-vs-control pair alone. A three-seed, four-arm
> frontier since measured the dose-response and the general claim is **wrong**; the C28 numbers
> themselves stand. Full result: `evidence/trainability/RECOVERABILITY_FRONTIER.md`.

Pooled over 3 seeds, n = 180 task prompts per arm, at the same matched 40-step recipe:

| arm | displaced | KV | dev CE | task | % of control | CIs disjoint? |
|---|---:|---:|---:|---:|---:|---|
| `source_control` | 0 | 1.000× | 1.7857 | 73/180 (40.6%) | — | — |
| **`ced_last4`** | 3 | 0.893× | 2.0657 | **69/180 (38.3%)** | **95%** | **No** |
| `ced_last8` | 7 | 0.750× | 2.1978 | 36/180 (20.0%) | 49% | Yes |
| `C28` | 13 | 0.536× | 3.1104 | 0/180 (0.0%) | 0% | Yes |

So the corrected claim is a dose-response: **at 3 displaced layers recovery is complete to
within measurement error; at 7 it is partial; at 13 it does not occur at all within this
budget.** Two consequences:

- **~0.5 nats/bank is an *initialization* price.** ~80% of it is discharged by 5.2k supervised
  target tokens, leaving a flat 0.06–0.10 nats per displaced layer. Everywhere this document
  quotes ~0.5 nats/bank, read it as the untrained-conversion price.
- **A capability cliff sits between 7 and 13 displaced layers that CE cannot see.** C28
  discharges 82% of its CE penalty — the same fraction as `ced_last8` — and still answers 0 of
  180 prompts.

Whether C28's 0/180 is structural or merely under-trained is **not settled**: every number here
rests on ~5.2k supervised target tokens against a registered ceiling of 1,000,000. A 500k-token
matched run is underway to test it.

The original single-seed pair, retained because it is what the earlier text rested on:

| dose | source CE | C28 CE | CED penalty | task |
|---|---:|---:|---:|---|
| lr 1e-4 | 1.6213 | 2.2796 | +0.66 nats | both 0/6 |
| **lr 1e-5** | **1.5165** | 2.7216 | **+1.21 nats** | source **6/6**, C28 **0/6** |

The gap *widens* at the dose that preserves capability. C28 acquires the answer **format**
(`10.`, `19.`) and never the arithmetic — 3 distinct outputs across 6 prompts.

Two endpoint findings fall out, and both matter beyond CED:

- **Held-out CE cannot certify a run.** At lr 1e-4 the source's CE improved 4.05 nats while
  its task accuracy went 6/6 → 0/6.
- **KL-to-teacher cannot either.** It *rises* with training even when capability is fully
  preserved (source at lr 1e-5: 0.0000 → 3.4908), because it tracks drift toward the training
  corpus rather than quality.

Both are the registered G3 endpoint shape. A budget-and-likelihood plan would have certified
a destroyed model.

---

## 6. A viable design was found — and it is not a sharing topology

> **Superseded.** This section previously opened "Nothing measured here is a viable KV-reduction
> path for *these* weights." That was true of every **sharing** scheme and is now false in
> general. Per-layer low-rank KV compression gives **4× reduction with no training and no
> capability loss detectable at n=60**, and **2× is exact by shape arithmetic** on all ten
> published cores. See `evidence/latent/LATENT_KV_RESULT.md` and `FREE_EXACT_KV_REDUCTION.md`.
> The constraints below still bind on *sharing* designs and are retained for that reason.

The reason the two families diverge so completely is the organising principle of §2. Sharing
changes **which** K/V a layer attends to, so every removed bank displaces a layer. Low-rank
compression changes only **how the same K/V is stored** — each layer still attends to its own
bank, nothing is displaced, and the ~0.5 nats/bank toll simply does not apply. Measured at the
same 0.250× budget where `local group 4` gives CE 16.86 (garbage), the whitened latent gives CE
5.5725 against a source of 5.5782 and 19/60 on the task probe against the source's 20/60.

One caveat that nearly buried the result: the **fit objective** decides everything. At the same
rank 512, plain SVD gives CE 8.2881 and 0/60 with zero terminations; activation-whitened SVD
gives 5.5725 and 19/60. The hidden-state covariance has an effective rank of 3.5–5.3 out of
1024, so plain SVD spends its budget on directions the model never activates.

The constraints the evidence imposes on any **sharing** candidate:

1. **Function-preserving at initialization**, or it starts below uniform and spends its
   budget climbing back. `gated_memory.py` implements this — decoder self-attention retained,
   shared-memory read added with a zero-initialised output projection, so step 0 *is* the
   source. **It is implemented but not yet verified on real weights**, and it *adds* KV rather
   than saving it: the saving would have to be earned later by training the gate up and then
   pruning, which is a curriculum, not a conversion.
2. **Priced honestly.** ~0.5 nats per displaced layer, with a usable budget of 2–3 banks
   (~10% KV). Any design claiming more must explain how it escapes the exchange rate, which
   is a property of sharing itself and not of any particular wiring.
3. **Gated on capability, not likelihood** — both CE and KL decouple from task accuracy here.
4. **Trained from the start** if large KV savings are the goal. Repurposing pretrained weights
   into a different attention topology is what fails, in every variant tested.

Orthogonal methods that do not repurpose these weights — quantised or paged KV — are outside
this experiment's registered scope and are untested here. They are named only to be explicit
that "no viable path was found" applies to *weight-repurposing* approaches, not to KV
reduction generally.

---

## 7. Confidence and limits

**Strong** (multiple independent measurements, exact controls): the displaced-layer law *as an
initialization law*; CED's initialization being worse than uniform; D40's degeneracy; the FP32
mechanism; identity expansion being bitwise exact; the latent r=1024 exactness control and the
free 2x shape-arithmetic reduction across all ten published cores.

**Moderate** (single deterministic run or small eval): the 4x latent result at 0.250x KV
(one calibration set, 40 dev rows, 60 task prompts, 19-vs-20 is within noise at n=60); the
three-seed recoverability frontier (real replication, but one budget and one probe of three
numeric classes); the CE/capability decoupling.

**Weak or unverified**: anything past ~5.2k supervised target tokens — a 500k-token matched run
is underway and may change the C28 conclusion; ridge sensitivity of the whitened fit
(condition number 1.08e8 against a 1e-6 ridge, unswept); any recipe with replay or
frozen-teacher regularisation; generalisation of any coefficient beyond this one 596M model.

**A measurement caveat that touches every CE number in this document — now measured.** Only
**7.7%** of a DEV row's target positions are the assistant's actual answer. The other 92.3% is
the system prompt, the user's question, chat-template scaffolding and the empty
`<think></think>` block. Every cross-entropy figure in this document averages over *all*
positions, so it is overwhelmingly a measure of how well a model predicts **the prompt** — which
is not the model's job at inference time.

Position-resolved CE at initialization (`evidence/response-ce/RESPONSE_CE_SCAN.md`):

| arm | displaced | overall | reply | non-reply | penalty overall | penalty reply | concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `source_control` | 0 | 5.5782 | 2.6492 | 5.8209 | — | — | — |
| `ced_last4` | 3 | 6.5339 | 4.0426 | 6.7404 | +0.9558 | +1.3934 | **1.46x** |
| `ced_last8` | 7 | 8.4218 | 6.9610 | 8.5428 | +2.8436 | +4.3118 | **1.52x** |
| `C28` | 13 | 12.8880 | 12.2181 | 12.9435 | +7.3098 | +9.5689 | **1.31x** |

Two corrections fall out, and the first corrects a hypothesis of mine, not a prior document.
The non-reply region is **not** cheap repeated boilerplate — its CE (5.82) is more than twice
the reply's (2.65), because it is dominated by the user's novel question. So "the model posts a
CE win by learning the scaffolding" is **not** what is happening.

What *is* happening is worse for the metric: **displacement damage is 1.3-1.5x more severe in
the answer than the overall CE reports.** Every CE penalty quoted in this document therefore
*understates* the harm to the part anyone cares about. The direction of the bias is consistent
across all three arms. Historical numbers have **not** been recomputed position-resolved;
`long_budget.py` now reports `dev_response_ce_nats` alongside the overall figure.

**Process caveat.** Adversarial review of this work has repeatedly failed to complete. Three
early review workflows died on usage limits returning nothing. A later six-dimension pre-flight
audit of the large-budget scale-up ran 85 agents and **51 of them — including the entire
documentation-accuracy verification pass and the final synthesis — died on a session limit**.
Per the standing rule, that is **not a clean review**. One finding did survive independent
adversarial verification and was acted on: the registered CALIBRATION split had leaked into the
scaled TRAIN set (94 of 156 families, 10 byte-identical rows), which is now fixed and recorded.
**No conclusion in this document has survived a complete external adversarial pass.** Seven
measurement defects have been found internally — by exact controls, smoke tests and reading raw
samples — and the honest inference is that the internal controls are load-bearing, not that the
work is clean.
