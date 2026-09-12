# RAB-001 comprehensive analysis: every option researched, and what the evidence says

Branch `research/ced-rab001-hotfix001-qualification-v1`, pushed to
`LUNCY579/ChatBubble`. Pinned runtime python 3.11.15 / torch 2.10.0+cpu /
transformers 4.51.3, official Supa 1.7.0-devfreeze weights (`audit_digest 97cd76af…`).

**Digest state.** Current implementation is **`8b47b395`** (HOTFIX-004), 31 manifest files,
**191 tests, 0 skipped**. Current receipts are `evidence/ladder-8b47b395-v1/` and
`evidence/quality-8b47b395/`; **requalification is complete, not pending.**

**Five digests have now been qualified end to end** — `eab80e9f`, `8d54f540`, `fc44701e`,
`7d48de21`, `8b47b395` — and every one reproduces the same verdicts, the same `checks` and
the same per-path violation counts **to the element**. That repeated inertness is the
evidence that the audit fix, the HOTFIX-002 diagnostic and the HOTFIX-003/004 quality gate
each changed tooling without perturbing any measurement.

Sections 1–4 and 9 bind no digest at all: those diagnostics run from
`experiments/rab001_exploration/`, outside the hashed root, and produce no receipts.

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

**CED loses to the matched ordinary control at both doses, and loses harder at the good
one.** At identical recipe, rows and token budget: at lr 1e-4 the ordinary architecture
reaches held-out CE 1.6213 against CED's 2.2796 (+0.66 nats); at lr 1e-5 — the dose that
preserves the source's arithmetic at 6/6 — it reaches **1.5165** against CED's **2.7216**
(**+1.21 nats**), with CED still at **0/6** on the task. There is no measured axis on which
CED wins.

**An excessive learning rate destroys capability while improving the metric — but the dose,
not the data, was the problem.** At lr 1e-4 the frozen source went from 6/6 to 0/6 on the
pilot arithmetic while its held-out CE *improved* 4.05 nats. At lr 1e-5 the same 25 rows and
the same 4,968 tokens give **lower** held-out CE (1.5165) with arithmetic **intact**. So CE
can decouple from capability and cannot certify a run on its own; the corpus does not need
replacing. An earlier revision claimed the data itself was destructive — **withdrawn**, along
with a scorer bug that scored a correct `"3."` as a miss against `"3"`. See §9c.

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

Everything researched across the whole programme, with the verdict and where the evidence
lives. "Falsified" means measured and rejected; "ruled out analytically" means no measurement
was needed because the mechanism forbids it; "not attempted" means exactly that.

### Architecture / conversion

| # | option | status | evidence |
|---|---|---|---|
| 1 | CED as delivered (C28 = 14 encoder + 14 shared-memory decoder) | **falsified on capability** — CE 12.59, worse than uniform, 0/24 pilot | §1 |
| 2 | Dose-tune the conversion depth (convert only last *N* layers) | **falsified** — task lost by N=4 at a 10.7% KV saving | §3 |
| 3 | Local cross-layer KV sharing (share within groups of neighbours) | **falsified** — worse than CED at matched budget; both on one frontier | §4 |
| 4 | Function-preserving zero-gated shared-memory augmentation | **verified function-preserving on real weights**: bitwise identical logits (max delta 0.000e+00), CE identical, branch live under perturbation. Adds 60.8M params (10.21%) and *adds* KV rather than saving it; value is as a training scaffold, untested as one | `gated_memory.py`, `GATED_VERIFY_split14.json` |
| 5 | Identity depth expansion (G1_IDENTITY) | **works, bitwise exact** (`max_abs_logit_error = 0.0`) — the one lossless structural change available | §2 |
| 6 | Width reduction 1024→896 (`calibrate_d40`, the "ordinary control") | **invalid** — passes every structural gate while generating a degenerate `</think>` loop | §1 |
| 7 | Any other sharing topology | **ruled out analytically** — every removed bank displaces exactly one layer by construction, so all topologies share one frontier at ~0.5 nats/bank | §4 |

### Numerics / parity

| # | option | status | evidence |
|---|---|---|---|
| 8 | Strict elementwise FP32 cache-order parity as the CED gate | CED **fails** (401,863 / 726,228 violations) while the matched ordinary control passes with **zero** | §5 |
| 9 | FP64 diagnostic lane | both arms **pass** (3.8e-13 / 4.1e-12) — proves no logic, causality or cache bug | §5 |
| 10 | Higher-precision shared-memory **read** (HOTFIX-002) | **falsified** — C28 0.5% better, C40 10.6% *worse* | §5 |
| 11 | Higher-precision bank **projection** | **ruled out analytically** — its input `h` is already chunk-dependent | §5 |
| 12 | Full-context bank reconstruction | **ruled out analytically** — would need an O(T²) encoder re-run, erasing the efficiency the design exists for | §8 |
| 13 | Single-pass prefill only (declare chunked prefill unsupported) | **viable but insufficient** — removes chunk-schedule failures, not the `tail_only` failures nor the quality problem | §8 |
| 14 | Higher-precision **encoder** | **not attempted** — that is most of the model, a throughput/memory decision, and §5 shows precision is not the binding problem |  |
| 15 | Prospective equivalence-policy amendment for mid-network-tensor architectures | **not written** — would be legitimate, but §1 shows parity is not the blocker |  |

### Training / recovery

| # | option | status | evidence |
|---|---|---|---|
| 16 | Recovery from the CED initialization | **works and is fast** — ~10 nats of held-out CE in under 5k supervised tokens | §9a |
| 17 | CED vs matched ordinary control at equal training | **CED loses at both doses** — +0.66 nats at lr 1e-4, **+1.21 nats** at lr 1e-5, and 0/6 task against the source's 6/6 | §9b |
| 18 | Learning-rate dose | **measured** — lr 1e-5 preserves 6/6 arithmetic *and* reaches better CE (1.5165) than lr 1e-4 (1.6213), which destroyed it | §9c |
| 19 | Held-out CE as the recovery endpoint | **invalid on its own** — CE improved 4.05 nats while capability went 6/6 → 0/6 at lr 1e-4 | §9c |
| 20 | KL-to-teacher as the recovery endpoint | **invalid on its own** — rises with training even when capability is fully preserved; tracks corpus drift, not quality | §9c |
| 21 | 1M supervised-token budget per arm | **wrong by ~200x for this corpus** — 5k tokens already saturates 25 rows; and ~62 h for a matched pair on the only qualified runtime | §9e |
| 22 | Distillation from the frozen source into the gated variant | **not attempted, and gradient-free at step 0** — a function-preserving student *is* the teacher, so the distillation term starts at zero; it needs a curriculum | §9 |
| 23 | Replay mixture / frozen-teacher regularisation to protect capability | **not attempted** — the obvious next lever now that lr is known to matter |  |

### Protocol / instrumentation

| # | option | status | evidence |
|---|---|---|---|
| 24 | `G1_QUALITY` forward-only quality gate (HOTFIX-003) | **implemented and unit-tested; NOT yet demonstrated on real arms** | §6, `amendments/HOTFIX_003.md` |
| 25 | `audit-evidence` schema-2 strict-parity reporting | **fixed** — previously reported `{}` for every schema-2 FAIL receipt | git `fb0e2b8` |
| 26 | Training-side capability-retention gate | **does not exist** — this is recommendation 0 and the highest-value remaining item | §11 |
| 27 | Strict-string task scoring | **fixed** — scored a correct `"3."` as a miss and inverted a published conclusion | §9c |
| 28 | Leading-integer task scoring | **fixed** — too lenient the other way: `10:55 AM` matched `10:25 AM` for the 31/50 `time_duration` clock-time answers, and `62.9` matched `62.5`. Now compares the full number sequence | §12 |
| 29 | Seeded replication | **was vacuous, fixed** — nothing consumed the seed, so three "seeds" were one run; the seed now shuffles TRAIN | §12 |
| 30 | `G1_QUALITY` nondegeneracy floor | **fixed (HOTFIX-004)** — an absolute floor of 2 passed a `</think>` loop at 24 prompts; now scales as `max(2, ceil(0.25·prompts))`, and D40 fails on 3 checks | §6 |
| 31 | Task-probe class selection | **corrected** — `arithmetic_chain` is unusable (frozen source 0/20); probe is now `percentage_reasoning`/`ratio_proportion`/`time_duration`, measured at 65/50/30% | §12 |
| 32 | Running training on the GPU | **ruled out for this recipe** — AdamW fp32 on 596M needs 8.89 GB against 6.86 GB free; only SGD/Adafactor fit, and changing the optimizer breaks the matched comparison. Ada also defaults fp32 matmul to TF32 | §12 |

### Every measurement bug found in this programme, and how each surfaced

Six defects in my own harness and design. **None was found by reading code.** Each surfaced
from an exact control, a cheap smoke test, or looking at raw samples.

| # | defect | how it surfaced | consequence if missed |
|---|---|---|---|
| 1 | local-KV cache bytes re-derived from hard-coded head count and scalar width, disagreeing with `budget()` | the `group=1` identity control reported `kv=2.000x` for a config that *is* the source | every KV ratio in the comparison wrong |
| 2 | task scored by full-string equality | raw samples showed `3.` scored as a miss against `3` | **inverted a published conclusion** ("the data destroys capability") |
| 3 | task scored by leading integer | a trajectory log showed correct answers while hits read zero | ~21% of the probe set inflated (clock times, decimals) |
| 4 | the seed consumed nothing | two seeds produced **bit-identical** output | would have reported **one run as three replicated seeds** |
| 5 | a `print` could kill a run | three C28 runs died with their compute already spent | lost runs, no corrupted numbers |
| 6 | probe class chosen without checking the teacher | smoke test: frozen source scores **0/20** on `arithmetic_chain` | a confidently empty result — 0/N for every arm, before and after |

Defect 4 is the most serious: it does not produce a wrong number, it manufactures false
confidence in a right one. Defects 2 and 3 are the same bug in opposite directions, and both
came from the scorer being **duplicated across scripts and drifting**. That is now
structurally impossible: one implementation in `rab001_exploration/scoring.py`, imported
everywhere, with an 11-case self-test that runs **at import** so a drifted scorer fails
loudly rather than quietly producing numbers.

**The operational lesson, stated plainly.** Three separate adversarial-review workflows were
launched against this work and *all three* died on usage limits, returning nothing. Every
defect above was instead caught by:

- an **exact control** that must be bitwise identity (`ced_last1`, `local group 1`,
  inert-default requalification across five digests, step-0 identity across seeds);
- a **two-step smoke test** before committing hours of compute;
- **reading the raw samples**, not just the aggregate.

Those three habits found six bugs; external review found none. Keep them in every run.

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

### 9b. But CED loses the matched comparison — at both doses

Identical recipe, rows, order and 4,968 supervised target tokens; 40 steps; held-out on 8
calibration rows never used for a gradient step.

| arm | dose | held-out CE @40 | KL→orig @40 | **task @40** | distinct @40 |
|---|---|---:|---:|---:|---:|
| source | lr 1e-4 | 1.6213 | 5.4721 | 0/6 | 5 |
| source | **lr 1e-5** | **1.5165** | 3.4908 | **6/6** | 6 |
| C28 | lr 1e-4 | 2.2796 | 5.1657 | 0/6 | 4 |
| C28 | **lr 1e-5** | 2.7216 | 3.0280 | **0/6** | 3 |

CED penalty: **+0.66 nats** at lr 1e-4, **+1.21 nats** at lr 1e-5. The lr 1e-4 comparison was
confounded because that dose destroys the source's task ability; at lr 1e-5, where the source
keeps 6/6, CED is further behind and still scores zero.

**CED learns the format and never the arithmetic.** C28 at lr 1e-5, step 40, all six prompts:

    expected  3 → "10."      expected  9 → "10."
    expected  5 → "19."      expected 11 → "10."
    expected  7 → "10."      expected  8 → "The current one."

3 distinct outputs across 6 prompts — it emits `10.` almost regardless of the question.
The source at the same dose gives `3. 5. 7. 9. 11. 8.`, all correct, 6 distinct. Note that
**termination reads 6/6 for both arms**: only the distinct-output count separates a model
answering questions from one reciting a single answer, which is why `G1_QUALITY` checks both.

### 9c. CE can decouple from capability — a dose effect, not a data defect

**Corrected from an earlier revision of this section.** Two errors, both material.

*The scorer was wrong.* Pilot hits used full-string equality, so a model answering in the
corpus's own style — `"3."` rather than `"3"` — scored 0/6 while being entirely correct.
`recovery_pilot.py` now reports leading-integer `exact_hits` (primary, since these answers
are integers) alongside `strict_hits`, and saves every sample.

*And it was the dose.* Same 25 rows, same 40 steps, same 4,968 supervised target tokens:

| | lr 1e-4 | lr 1e-5 |
|---|---|---|
| held-out CE | 1.6213 | **1.5165** |
| answers for 3 / 5 / 7 / 9 | `1.` `3.` `4.` `6.` — wrong | `3.` `5.` `7.` `9.` — **correct** |
| capability | destroyed | **preserved** |

The gentler dose wins on **both** axes. The withdrawn claim was "this training data destroys
the capability"; what survives is narrower and still sufficient:

> An excessive learning rate can drive held-out CE down while destroying task accuracy, so
> **CE alone cannot certify a training run** and a capability-retention check is required.

Recommendation 0 therefore stands, on the weaker ground that CE *can* decouple from
capability rather than that it always does. The registered G3 shape — a CE/KL objective with
`minimum_effect` on a likelihood — would still have certified the destroyed lr 1e-4 model.

### 9d. Resolved: the comparison was redone at the good dose

An earlier revision of this section flagged that §9b's conclusion held only at lr 1e-4, the
destructive dose, and that the matched lr 1e-5 pair was outstanding. **It has been run.** CED
loses at lr 1e-5 by a wider margin (1.21 vs 0.66 nats) while scoring 0/6 against the
source's 6/6, so the conclusion is now established at both doses and is strongest at the one
a practitioner would choose. Two earlier attempts at this cell were killed by session
teardown and produced no data; the completed run is
`RECOVERY_PILOT_C28_lr1e-05_s40.json`.

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
   the matched comparison at equal training **at both doses tested, by more at the better
   one** — 0/6 on the task against the ordinary control's 6/6.
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
