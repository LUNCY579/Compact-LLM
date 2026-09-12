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
| **KV sharing** (CED, local, any topology) | which K/V a layer attends to | +0.5 nats per displaced layer | collapses with depth | **priced** |
| **Self-attn → shared memory** (full CED) | 13–14 layers displaced at once | 12.59 (**worse than uniform** 11.93) | token salad | **destroyed** |

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
- Both schemes therefore price identically, at **~0.5 nats per displaced layer**, and the
  measurement confirms it: CED at 13 displaced gives CE 12.5854, local sharing at 14 gives
  12.9136 — two structurally unrelated schemes on one curve.

**Every removed KV bank displaces exactly one layer, by construction** — a layer that does
not produce its own K/V is simultaneously a saved bank and a displaced layer. So
CE-versus-displaced-count *is* CE-versus-KV-saving for **any** sharing topology. There is no
clever wiring waiting to be found, and the search for one is closed.

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

## 5. Training does not rescue a displaced initialization

CED is **trainable** — held-out CE falls ~10 nats in under 5k supervised tokens, and
generation recovers from token salad to fluent English. But at a matched recipe it **loses**:

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

## 6. What a viable design would have to look like

Nothing measured here is a viable KV-reduction path for *these* weights. The constraints the
evidence imposes on any candidate:

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

**Strong** (multiple independent measurements, exact controls): the displaced-layer law;
CED's initialization being worse than uniform; D40's degeneracy; the FP32 mechanism; identity
expansion being bitwise exact.

**Moderate** (single deterministic runs, small evals): the matched training comparison at
both doses; the CE/capability decoupling.

**Weak or unverified**: anything past 40 training steps; the gated variant on real weights;
any recipe with replay or frozen-teacher regularisation.

**Process caveat that applies to all of it:** three adversarial-review workflows were run
against this work and all three died on usage limits, returning nothing. **No conclusion here
has survived an external adversarial pass.** Six measurement defects were found internally —
by exact controls, smoke tests and reading raw samples — and the honest inference is that the
internal checks are load-bearing, not that the work is clean.
