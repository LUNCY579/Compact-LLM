# RAB-001 comprehensive analysis: every option researched, and what the evidence says

Branch `research/ced-rab001-hotfix001-qualification-v1`, pushed to
`LUNCY579/ChatBubble`. Pinned runtime python 3.11.15 / torch 2.10.0+cpu /
transformers 4.51.3, official Supa 1.7.0-devfreeze weights (`audit_digest 97cd76af…`).

**Digest state — read this before trusting any receipt reference below.** The parity and
engineering receipts cited in §5 were produced at implementation `fc44701e` (186 tests).
Adding the `G1_QUALITY` gate (HOTFIX-003) changed `implementation_manifest()` and the digest
is now **`7d48de21`** with **189 tests, 0 skipped**. So:

- every §1–§4 quality/architecture number is unaffected — those diagnostics run outside the
  hashed root and bind no digest;
- the §5 ladder receipts under `evidence/ladder-fc44701e-v1/` bind `fc44701e` and are
  therefore **stale relative to the working tree**; requalification at `7d48de21` is pending
  and is recorded as pending rather than assumed.

HOTFIX-003 only *adds* a gate, so no previously measured number should change on
requalification — but that is a prediction, not yet a verified result.

Every number below was produced in this worktree and is reproducible from committed
artifacts. Nothing here is an efficacy claim, and no sealed evaluation was touched.

---

## 0. The short version

**The CED hypothesis, as implemented, is falsified — for a reason that has nothing to do
with the numerical parity work that dominated the earlier sessions.**

The registered CED arms are not "structurally transplanted models awaiting recovery". They
are models whose next-token distribution is **worse than uniform over the vocabulary**, and
whose free generation is multilingual token salad. The ordinary control arm is also broken,
in a different way. Only the frozen source is a working model.

No gate in the ladder tested for this, because every gate tests plumbing or
self-consistency: tensor accounting, save/reload exactness, parameter counts, cache-order
parity. A model emitting token salad satisfies all of them.

The measured quality/efficiency curve shows the CED design cannot be rescued by tuning how
much of it you apply: the task is lost entirely at a **10.7% KV saving**, long before the
46% saving the registered arm targets.

A cheaper sharing topology was hypothesised, implemented and **falsified**: local
cross-layer sharing is worse than CED at matched KV budget. The two schemes lie on a single
frontier, because every KV bank removed displaces exactly one layer by construction, and
each displaced layer costs roughly **0.5 nats**. No rearrangement of sharing topology
escapes that exchange rate — only training can.

Training was then measured, and it produced the two results that matter most.

**CED loses to the matched ordinary control.** At identical recipe, identical rows and
identical token budget, the ordinary architecture reaches held-out CE **1.6213** against
CED's **2.2796**. There is no measured axis on which CED wins.

**The available training data destroys the capability being measured.** The frozen source
went from **6/6 to 0/6** on the pilot arithmetic in 40 steps and under 5k supervised tokens,
while its held-out CE *improved* 4.05 nats. Held-out CE and task accuracy are
**anti-correlated** here, so a recovery plan with a CE or teacher-KL endpoint — which is the
registered G3 shape — would have reported success while destroying the model.

---

## 1. What each arm actually is, and what it scores

Teacher-forced CE, perplexity, KL-to-source and top-1 agreement on the open calibration
split; greedy generation on the open pilot split. `ln(151936) = 11.9312` is the
uniform-distribution reference.

| arm | CE | perplexity | KL→source | top-1 | pilot hits | generation behaviour |
|---|---:|---:|---:|---:|---:|---|
| **B28** frozen source | 5.546 | 256 | 0 | 1.000 | **6/24** | works; answers `3`, `5`, `7`, `9` |
| **D40** ordinary width | 5.360 | 213 | 1.468 | 0.500 | 0/24 | **degenerate** `</think>` loop |
| **C28** CED | **12.585** | 292,267 | 9.154 | 0.014 | 0/24 | token salad, never terminates |
| **C40** CED | **12.960** | 425,015 | 9.878 | 0.004 | 0/24 | token salad, never terminates |

Both CED arms are **above the uniform baseline** — as initializations they carry negative
information about the next token.

D40 deserves separate attention because it is the *control*. Its CE is **better** than the
source (5.360 vs 5.546) while its generation collapses into a repeated `</think>`. That
combination — good teacher-forced loss, degenerate free running — is why CE alone is not a
capability measure, and why the earlier sessions' reliance on structural gates hid the
problem.

Evidence: `evidence/arm-quality/ARM_QUALITY_24rows.json`,
`evidence/arm-quality/ARM_GENERATION.json`,
`evidence/arm-quality/FINDING_CONVERSIONS_ARE_NOT_USABLE.md`.

---

## 2. Why the conversion destroys the model

`to_ced` keeps donor layers 0..13 as the encoder, builds **one** shared K/V bank from
layer 13's output, and converts donor layers 14..27 into `ReadBlock`s. A `ReadBlock` has
**no self-attention**: its `q_proj` queries only that bank, and decoder positions can no
longer see one another at all.

So weights trained to attend over their own layer's keys are asked to attend over a
completely different key space, 14 layers removed. `to_ced` already records
`function_preserving: false`; the tables above are the magnitude of that.

The codebase proves lossless structural change is achievable: `G1_IDENTITY`, pure depth
expansion, is **bitwise exact** (`max_abs_logit_error = 0.0`). The damage comes specifically
from (a) the 1024→896 width reduction in `calibrate_d40` and (b) the
self-attention→shared-memory replacement in `to_ced`.

---

## 3. The quality/efficiency curve — how much CED can be applied at all

Converting only the last *N* of 28 layers, zero training, 24 calibration rows / 8 pilot
prompts:

| N converted | encoder | CE | perplexity | KL→source | top-1 | **KV ratio** | pilot hits | terminated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (source) | 28 | 5.5458 | 256.2 | 0.0000 | 1.000 | 1.000x | 6/8 | 8/8 |
| 1 | 27 | 5.5458 | 256.2 | 0.0000 | 1.000 | 1.000x | 6/8 | 8/8 |
| 2 | 26 | 5.5916 | 268.2 | 0.4626 | 0.700 | 0.964x | 6/8 | 8/8 |
| 3 | 25 | 5.8439 | 345.1 | 0.8178 | 0.644 | 0.929x | 4/8 | 5/8 |
| 4 | 24 | 6.4065 | 605.8 | 1.7691 | 0.436 | 0.893x | **0/8** | 1/8 |
| 6 | 22 | 7.2025 | 1,342.7 | 2.7976 | 0.315 | 0.821x | 0/8 | 0/8 |
| 8 | 20 | 8.3245 | 4,123.7 | 4.1418 | 0.179 | 0.750x | 0/8 | 0/8 |
| 10 | 18 | 9.6920 | 16,187.8 | 5.9453 | 0.072 | 0.679x | 0/8 | 0/8 |
| 14 (= C28) | 14 | 12.5854 | 292,267.1 | 9.1543 | 0.014 | 0.536x | 0/8 | 0/8 |

Two things make this table trustworthy. N=1 is an **exact control**: converting one layer is
bitwise function-preserving at `kv_ratio 1.000`, because a single decoder layer's shared
bank *is* its own self-attention KV — so both damage and saving must begin at N=2. And N=14
reproduces the independently-measured C28 arm to four decimals (12.5854).

**The trade-off is not favourable anywhere.** The task is gone by N=4 for a 10.7% KV saving.
Reaching the registered 0.536x budget costs a model worse than uniform.

Evidence: `evidence/arm-quality/CED_DEPTH_SWEEP.json`.

---

## 4. The governing law: quality depends on how many layers are displaced, not on topology

My first reading of the curve was that damage tracks *distance* to the bank, since N=2 costs
only +0.046 nats. That predicted local cross-layer sharing — each layer reading its immediate
neighbour's K/V instead of one distant bank — would sit on a much better frontier.

**I implemented it, measured it, and the prediction was wrong.** Recorded here because the
falsification is more useful than the hypothesis was.

| variant | displaced layers | CE | perplexity | KL→source | top-1 | KV ratio | pilot hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | 5.5458 | 256.2 | 0.0000 | 1.000 | 1.000x | 6/8 |
| local group 1 *(control)* | 0 | 5.5458 | 256.2 | 0.0000 | 1.000 | 1.000x | 6/8 |
| ced last 4 | 3 | 6.4065 | 605.8 | 1.7691 | 0.436 | 0.893x | 0/8 |
| ced last 8 | 7 | 8.3245 | 4,123.7 | 4.1418 | 0.179 | 0.750x | 0/8 |
| **ced last 14** | **13** | **12.5854** | 292,267 | 9.1543 | 0.014 | 0.536x | 0/8 |
| **local group 2** | **14** | **12.9136** | 405,791 | 9.9648 | 0.002 | 0.500x | 0/8 |
| local group 3 | 18 | 15.2306 | 4,116,880 | 12.4193 | 0.001 | 0.357x | 0/8 |
| local group 4 | 21 | 16.8612 | 21,024,648 | 14.1379 | 0.001 | 0.250x | 0/8 |
| local group 7 | 24 | 16.2356 | 11,247,353 | 13.3071 | 0.000 | 0.143x | 0/8 |
| local group 14 | 26 | 15.1577 | 3,827,274 | 12.2741 | 0.001 | 0.071x | 0/8 |

Local sharing is **worse**, not better: 14 displaced layers at 0.500x KV gives CE 12.91,
against CED's 13 displaced layers at 0.536x KV giving CE 12.59. CED is the *better* of the
two schemes at comparable budget.

`local group 1` is an exact control — CE identical, KL 0.0000, top-1 1.000, kv 1.000x — so
the harness is validated and the degradation is real.

**Why the two schemes coincide.** Define a *displaced* layer as one that attends to a K/V
bank it did not produce. Then:

| scheme | banks resident | banks saved | layers displaced | ratio |
|---|---:|---:|---:|---:|
| ced last N | 29−N | N−1 | N−1 | **1.00** |
| local group g | ⌈28/g⌉ | 28−⌈28/g⌉ | 28−⌈28/g⌉ | **1.00** |

Every removed bank displaces exactly one layer — **by construction**, since a layer that
does not produce its own K/V is simultaneously a saved bank and a displaced layer. So
*CE-versus-displaced-count is CE-versus-KV-saving for any sharing topology whatsoever.*

The measurement confirms it: CED at 13 displaced (12.5854) and local at 14 displaced
(12.9136) land on the same curve despite being structurally unrelated. Distance to the bank
is second-order; the count is what matters, and the cost is roughly **0.5 nats per bank
removed** over the usable range.

Above CE 11.93 (uniform) all variants are garbage, and differences between garbage levels
are not meaningful — which is why `local group 14` (15.16) appears "better" than
`local group 4` (16.86). That non-monotonicity should not be over-read.

**Consequence.** There is no clever sharing topology waiting to be found. Any zero-training
KV-sharing conversion of these weights buys its savings at ~0.5 nats per bank, and the
budget of a usable model is about 2–3 banks. To move the frontier at all, the weights have
to be trained for the new information flow — which makes trainability, not topology, the
binding question.

Evidence: `evidence/arm-quality/ARCHITECTURE_COMPARISON.json`, `local_shared_kv.py`.

---

## 5. The numerical-parity investigation, and why it mattered less than it appeared

Chronologically this came first, and it is correct work on a question that turned out to be
secondary. Recorded in full because it rules out routes.

**Finding.** Both CED arms fail the strict elementwise 1e-4 cache-order parity gate
(`strict_tail_only_parity`, `strict_chunked_cache_parity`, `causal_prefix_parity`), while
the matched ordinary control D40 passes the identical grid with **zero** violations.

| arm | precision | max abs err | total violations |
|---|---|---:|---:|
| D40 ordinary | FP32 | 8.11e-05 | **0** |
| B28 official | FP32 | 1.73e-04 | 4 |
| C28 CED | FP32 | 8.23e-04 | 401,863 |
| C40 CED | FP32 | 5.89e-03 | 726,228 |
| B28 official | FP64 | 3.77e-13 | 0 |
| C28 CED | FP64 | 4.10e-12 | 0 |

`causal_suffix_isolation_exact` and `decode_choices_exact` are True for every arm, row and
precision — there is **no causality, masking or cache bug**, and FP64 is clean.

**Mechanism, measured.** Divergence starts at encoder layer 0 and amplifies ~1000x through
depth, so the shared bank already carries dV = 4.6e-04 — 4.6x outside tolerance — before any
decoder read. B28's per-layer drift matches C28's encoder drift digit-for-digit
(3.815e-06 at layer 00, 1.678e-04 at layer 13) and B28's final layer drifts *more*
(8.8e-03 vs 3.9e-03), yet B28's logits show 4 violations against C28's 401,863. So CED
creates no extra drift; a plain model's drift reaches the final RMSNorm which suppresses it
(8.8e-03 hidden → 1.7e-04 logit), while CED re-injects the *unnormalized mid-network* bank
into all 28 decoder layers before that norm.

**The fix that was tried and failed.** HOTFIX-002 added a registered, default-inert
`--memory-precision float64` raising accumulation for the shared-memory read:
C28 401,863 → 399,698 (0.5%, max error *worse*); C40 726,228 → **803,104** (worse).
Falsified. Precision at the read cannot recover information already wrong in the bank, and
raising the bank's projection precision cannot help either because its input is already
perturbed.

**Why this matters less than it looked.** All of the above is a hygiene property of a model
at perplexity 292,267 whose `decode_choices_exact` passes anyway. Chasing a 1e-4 logit
tolerance was the wrong order of work. The negative result stands and rules out a route.

Evidence: `evidence/ladder-fc44701e-v1/`, `evidence/HOTFIX_002_DIAGNOSTIC_RESULT.md`,
`evidence/ladder-eab80e9f-v1/ARM_NUMERICS_COMPARISON.json`.

---

## 6. The gate ladder's blind spot

| gate | what it checks | would a token-salad model pass? |
|---|---|---|
| G0 | source hashes, native reference agreement, parse/termination on pilot | no — it scores the *source* |
| G1_IDENTITY | bitwise identity of depth expansion | no — it is exact by construction |
| G1_WIDTH | registered dims, tensor accounting, finite logits, save/reload, param count | **yes** |
| G1_CED_* | the above plus strict cache-order parity | **yes** (D40 proves parity ≠ quality) |
| G2_CACHED_* | 25 steps completed, params reached, teacher frozen, save/reload | **yes** |

`G1_WIDTH` PASSed a model that emits a `</think>` loop. No stage compares any arm's
*output quality* to the source. That is the structural reason a broken conversion travelled
this far, and it is a protocol defect independent of CED.

**Recommended gate addition** (prospective, not applied here): a G1_QUALITY stage requiring,
on the already-open calibration/pilot splits, a bounded CE increase and bounded KL to the
frozen source, plus a nondegeneracy check (termination rate and distinct-output count).
Cheap — forward-only — and it would have failed D40, C28 and C40 immediately.

---

## 7. Every option considered, and its status

| # | option | status | evidence |
|---|---|---|---|
| 1 | CED as delivered (14 encoder + 14 shared-memory decoder) | **falsified** — worse than uniform, 0/24 pilot | §1 |
| 2 | Tune how many layers are converted | **falsified** — task lost at 10.7% KV saving | §3 |
| 3 | Higher-precision shared-memory read (HOTFIX-002) | **falsified** — no improvement, C40 worse | §5 |
| 4 | Higher-precision bank *projection* | **ruled out analytically** — its input is already perturbed | §5 |
| 5 | Full-context bank reconstruction | **ruled out analytically** — see §8 | §8 |
| 6 | Higher-precision encoder | not attempted — most of the model; a throughput decision, and §1 shows precision is not the binding problem | — |
| 7 | Prospective equivalence-policy amendment | not written — would be legitimate, but §1 shows parity is not the blocker | — |
| 8 | Function-preserving zero-gated augmentation | **implemented, not yet verified on real weights** — the verification run OOMed against concurrent jobs and was not retried | `gated_memory.py` |
| 9 | Local cross-layer KV sharing | **implemented and falsified** — worse than CED at matched KV; both lie on one frontier | §4 |
| 10 | Recovery/distillation from this initialization | **measured**: recovery is fast (10 nats in <5k tokens) but CED loses the matched control by 0.66 nats and the endpoint is invalid | §9 |
| 11 | Ordinary D40 as the matched control | **invalid** — control is itself degenerate | §1 |

---

## 8. Why a full-context bank cannot work

The request was to build the shared bank from a full-context forward. It cannot fix
cache-order parity, for a structural reason:

During chunked prefill the bank for early positions is projected from an encoder output `h`
that was itself computed under chunking, and §5 shows `h` is chunk-dependent from layer 0
onward. Recomputing the bank *projection* from the cached `h` therefore re-derives a wrong
bank from wrong inputs. Making it genuinely full-context would require re-running the whole
encoder over all positions at every step — O(T²) per token, which is precisely what KV
caching exists to avoid, and which would erase the efficiency claim the architecture is for.

The only chunk-invariant option is **single-pass prefill** (no chunked prefill at all,
incremental decode only). That is a legitimate declarable constraint and would remove the
chunk-schedule failures — but not the `tail_only` failures, which occur with a full-context
bank in both compared paths, and not the §1 quality problem, which is the actual blocker.

---

## 9. Recovery and distillation — measured, not projected

An earlier revision of this section argued a 1M-token recovery was both infeasible and
"mislabelled", and that nothing suggested 1M tokens would be close. **The first half was
right for the wrong reason and the second half was wrong.** Measurement:

### 9a. The arms are highly trainable

C28, from a worse-than-uniform start, on 25 rows with AdamW lr 1e-4:

| step | tokens | held-out CE | KL→source | pilot hits | generation |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 12.2003 | 8.9176 | 0/6 | token salad |
| 20 | 2,585 | 2.5368 | 3.5077 | 0/6 | all 6 outputs identical |
| 40 | 4,968 | 2.2796 | 5.1657 | 0/6 | fluent English, wrong |

So the destroyed initialization is a starting point, not a dead end: ~10 nats of held-out CE
recovered in under 5k supervised tokens, and fluency returns from token salad to grammatical
English. A 1M-token budget is not remotely required to move CE. *(This run was interrupted at
step 40 of a planned 120 and is labelled as such.)*

### 9b. But CED loses the matched comparison

Same recipe, same 25 rows in the same order, same 4,968 supervised target tokens, 40 steps:

| | source untrained | **source trained** | C28 untrained | **C28 trained** |
|---|---:|---:|---:|---:|
| held-out CE | 5.6679 | **1.6213** | 12.2003 | **2.2796** |
| held-out KL to original source | 0.0000 | 5.4721 | 8.9176 | 5.1657 |
| **pilot task hits** | **6/6** | **0/6** | 0/6 | 0/6 |

CED is **0.66 nats worse** than the ordinary architecture at equal training. C28's 2.28 had
looked like it beat the frozen source's 5.67; that is now fully explained as corpus
adaptation, which the ordinary architecture simply does better. Both arms also drift *away*
from the teacher as they train (KL rises), so neither is distilling — both are converging on
the training corpus.

### 9c. The serious finding: the endpoint is invalid

**The frozen source went 6/6 → 0/6 on pilot arithmetic after 40 steps.**

    expected 3 → "1."      expected 5 → "3."
    expected 7 → "4."      expected 9 → "6."

It learned the corpus's terse numeric answer *format* and lost the *arithmetic*.

| arm | Δ held-out CE | Δ task |
|---|---|---|
| source | −4.05 nats (much better) | 6/6 → 0/6 (**destroyed**) |
| C28 | −9.92 nats (much better) | 0/6 → 0/6 (unchanged) |

Every arm improved sharply on the metric and either lost or failed to gain the capability.
On this corpus, **held-out CE is not a valid proxy for capability and optimising it actively
harms capability.** That invalidates the endpoint, not just the dose — and the registered
G3 shape (`recovery_tokens_per_arm`, a CE/KL objective, `minimum_effect` on a likelihood)
is built around exactly that endpoint.

This is the failure the brief warned about, arriving in its strongest form: training-loss
reduction is not reasoning improvement, and here it is reasoning *destruction*.

### 9d. Dose versus data — open

lr 1e-4 on 25 rows for 40 steps is a deliberately small probe. A lower learning rate, a
replay mixture, or a frozen-teacher distillation term might preserve arithmetic while still
adapting. A dose test at lr 1e-5 (the registered engineering learning rate) is the direct
experiment and its result belongs here. Until it lands, "this data is unusable" and "this
dose is wrong" are **both live**, and only the latter is cheap to fix.

### 9e. Measured cost, which stands

| quantity | measured |
|---|---|
| student throughput | ~9 supervised targets/s (25-step engineering smoke), ~22 s/step for the 40-step pilot |
| teacher cache | 4,110 bytes/token → 1M targets ≈ 4.1 GB |
| peak process RAM | ~7 GB engineering smoke, ~10 GB with AdamW state for 596M params |
| 1M targets, one arm | ~31 h CPU |

The infeasibility conclusion survives: a 1M-token matched pair is ~62 h on the only qualified
runtime. But it is now beside the point, because 5k tokens already move CE by 10 nats and the
thing CE measures is not the thing that matters.

Evidence: `evidence/trainability/MATCHED_COMPARISON.md`,
`evidence/trainability/RECOVERY_PILOT_C28.md`,
`evidence/trainability/RECOVERY_PILOT_source_control.json`,
`evidence/trainability/PARTIAL_recovery_pilot_C28_step40.log`.

---

## 10. What is claimed, and what is not

**Supported by evidence here.** The CED arms as delivered are worse than uniform and produce
no usable output. The ordinary D40 control is degenerate. The CED quality/efficiency
trade-off is unfavourable at every measured point. CED's FP32 cache-order failure is real,
CED-specific, mechanistically explained, and not fixable at the read. No gate in the ladder
tests output quality. Local cross-layer sharing is the direction the curve's shape implies.

**Not supported, and not claimed.** No efficacy, latency, memory, throughput, 8-GB-fit or
deployment claim. No arm has been compared to another *after training*. No recovery or
distillation result. No sealed or registered DEV evaluation. No multi-seed anything — every
number is a single deterministic CPU run. The generation probe is 8–24 open pilot prompts of
one narrow arithmetic family, which is a smoke test, not a capability benchmark.

**Process caveats.** Two independent-review attempts both failed on usage limits and
**neither is a clean review**: the adversarial audit of the delivered package lost 60 of 123
agents, and a later adversarial audit of *this* analysis lost all 6 of 6 and returned
nothing. So none of the conclusions here have survived an independent adversarial pass. The
self-checks that do exist are internal: exact controls (`ced_last1`, `local group 1`,
inert-default requalification at three digests) and cross-script reproduction of C28's CE to
four decimals. All §1–§4 diagnostics run **outside** the hashed experiment root
(`experiments/rab001_exploration/`), deliberately: adding a `.py` under
`experiments/architecture_benchmark_001/` changes `implementation_digest()` and invalidates
every receipt. They therefore produce **no receipts** and open no gates — they are
measurements, not qualifications.

---

## 11. Recommendation

0. **Fix the endpoint before any further training work.** §9c shows held-out CE
   anti-correlated with task accuracy on this corpus: the source's CE improved 4.05 nats
   while its arithmetic went 6/6 → 0/6. Any budget, margin or stop rule defined on a
   likelihood will certify a destroyed model. The endpoint must be a task measure with a
   registered capability-retention check the optimiser may not trade away. `G1_QUALITY`
   (HOTFIX-003) is the initialization-side half of this; the training side does not exist yet.
   This is now the highest-priority item, ahead of anything architectural.

1. **Stop the CED-by-conversion arm.** It is falsified on capability grounds, independently
   of the parity question; the trade-off curve leaves no viable operating point; and it loses
   the matched comparison at equal training.
2. **Add a quality gate before any further architecture work** (§6). It is cheap and it is
   the reason this went undetected.
3. **Retire D40 as the "ordinary control".** A degenerate control cannot support any
   comparison. The frozen source is the only valid reference today.
4. **Do not pursue another sharing topology.** §4 shows the exchange rate — one displaced
   layer per bank saved, ~0.5 nats each — is a property of sharing itself, not of CED's
   particular wiring, and it was confirmed by two unrelated schemes landing on one curve.
   The usable budget is 2–3 banks (~10% KV), which is not worth an architecture programme.
   If KV cost is the real target, the honest options are a trained-from-the-start design,
   or orthogonal methods that do not repurpose these weights (quantised or paged KV), which
   are outside this experiment's registered scope.
5. **Do not run a 1M-token recovery** on this hardware. If training is wanted, first
   establish a CUDA runtime and requalify under it, then run a bounded pilot against a
   registered quality endpoint — not a token budget.
