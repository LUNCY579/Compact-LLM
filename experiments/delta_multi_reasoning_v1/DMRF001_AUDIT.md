# DMRF-001 Stage 0 — audit of the claims this experiment would have been built on

**Every number below was recomputed from the primary artifact on this host. Where a claim
reproduces, it is marked VERIFIED and the reproduction is stated. Where it does not, the
discrepancy is diagnosed rather than noted. Three claims are contradicted, one is
mischaracterised, one is resolved in the report's favour after I initially suspected it, and
two new measurements change what the programme can say next.**

Receipts: `receipts/DELTA_MEASUREMENT.json`, `receipts/SPECTRAL_GAPS.json`.
Read-only with respect to every existing artifact. 0 model calls, 0 optimizer steps.

---

## 0. Identity chain — VERIFIED by recomputation

| role | artifact | sha256 | verdict |
|---|---|---|---|
| **W0** source | `Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/model.safetensors` | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | **recomputed, matches** |
| **WT** trained | `rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors` | `07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d` | **recomputed, matches** |
| **D** k=0.25 core | `…/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/model.safetensors` | `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284` | **recomputed, matches** |

Provenance: RRP-015 clean control, selection seed 17, `…/rrp015-20260917/control-journal/runs/train-001/microfit/checkpoint-00028`, manifest `3d5cef31…`.

**A correction to how WT is described.** `trained_fp32.safetensors` is 1,258,500,416 bytes =
314,618,880 float32 parameters. That is **not a full checkpoint** — it is only the 220 trained
tensors, consolidated. A full FP32 596M model would be 2.38 GB. Every reference to WT as "the
trained checkpoint" should be read as "the trained tensors"; the untrained 281,431,040
parameters are W0's by construction, never by measurement.

My own count of the structure, from the files: **310 tensors total, 220 trained
(140 two-dimensional + 80 one-dimensional), 90 untouched**, 0 trained tensors with a zero
delta. `tie_word_embeddings: true`, one `model.embed_tokens.weight` in the file, no separate
`lm_head`. This matters in §1.

---

## 1. "The reported norm ratio is approximately 0.000558" — **CONTRADICTED, and diagnosed**

Source of the claim: `int001-20260917/INT001_PREREG.md:28`, `||dW||/||W|| = 0.000558`.

Measured in float64 from the two verified inputs:

| scope | value |
|---|---|
| **whole model** (denominator = all 310 tensors as stored) | **0.000575927** |
| **trained tensors only** (the 220 that moved) | **0.000622030** |
| 2D trained matrices only | 0.000621985 |

None of these is 0.000558. The gap is 3.2%, which is too large for a rounding story, so I
looked for the cause rather than reporting a discrepancy.

**Cause: the tied embedding is counted twice in the denominator.**

```
||dW||                                    = 0.829831
||W||  (file as stored, one embedding)    = 1440.861608  ->  ratio 0.000575927
||embed_tokens||                          =  364.255100
||W||  with the embedding counted twice   = 1486.191100  ->  ratio 0.000558361   <-- 0.000558
```

`tie_word_embeddings` is true, so iterating a *loaded model's* named parameters exposes the
same 151936x1024 tensor as both `model.embed_tokens.weight` and `lm_head.weight`. Summing
squares over that iteration adds 364.2551² to the denominator and yields exactly the reported
figure. The safetensors file contains one embedding tensor; the model object reports two names
for it.

**Disposition.** The order of magnitude is right and the qualitative claim — a ~0.06% relative
update — stands. The specific figure is a tensor-alias artifact and should be restated as
**0.000576 whole-model / 0.000622 over the trained scope**, with the scope named, because the
two differ by 8%.

**Corroboration that the rest of that measurement is sound:** the same prereg line reports
"largest single-tensor relative change: 0.00209". I measure **0.002073**
(`model.layers.15.self_attn.k_proj.weight`). That matches. The defect is specific to the
aggregate denominator, not to the delta computation.

---

## 2. "Tolerant acquisition scoring rises from 35/384 to 234/384" — **CONTRADICTED on both endpoints**

**There is no 35.** Searching every record for an acquisition figure of 35 returns nothing. The
nearest 35 in the corpus is `rrp015-20260917/summary-001/SUMMARY.md:10`, where 35 is the
*rehearsal arm's retained count out of 47* — a different instrument, a different endpoint, and
a different arm.

The source model's actual acquisition, all three parsers, from
`qwen001-20260918/scored/` and `QWEN001_RESULTS.md`:

| model | strict | canonical | contract-tolerant | parsed |
|---|---:|---:|---:|---:|
| **Supa 1.7 (the source)** | **6** | **14** | **17** | 353/384 |
| Titan-Experimental (full delta, BF16) | 234 | **234** | 234 | 384/384 |
| Titan-Sparse **k=0.25** (the promoted core) | 232 | **232** | 232 | 384/384 |

So the claim is wrong three ways:

1. The baseline is **14** canonical or **17** tolerant, never 35.
2. **234 is the full-delta condition**, not k=0.25. The promoted k=0.25 core scores **232**.
3. 234 is a **canonical** figure. Calling the pair "tolerant" mislabels the parser, and for
   these two models tolerant and canonical are identical anyway (format-only gain 0).

**The verified pair is 14 → 234**, recorded at
`rrp015-20260917/control-preserved-hf/README.md:13`, source → RRP-015 control, canonical.
For the promoted core the pair is **14 → 232**.

This is exactly the condition-splice the protocol warned against: a baseline that exists
nowhere, joined to a trained score from a different weight condition, under a parser label
that belongs to neither.

---

## 3. "deep-any" — **the named mechanism does not exist; the real one is label-blind**

No `deep_any`, `deep-any`, or any-value-match scorer exists anywhere in the records or the
harness. I grepped for it and for equivalent constructions.

The real tolerant mode is `contract_tolerant`, `qwen001-20260918/stage3_score.py:105-116`:

```python
obj = json.loads(text); parsed_ok = True
canon = canonical(obj) == ref
tol = canon
if not canon and isinstance(obj, dict) and len(obj) == 1:
    k = next(iter(obj))
    if k in UNWRAP:
        tol = canonical(obj[k]) == ref
```

with `UNWRAP = {after, result, state, answer, output, value, final, new_state, updated}`,
frozen by QWEN-001 Amendment 1 before the scorer existed.

**Verdict: label-blind extraction of one answer, not a search for any gold-matching value.**

- It unwraps **at most one** level.
- Only when the parse yields a dict with **exactly one** key (`len(obj) == 1`), so there is
  never a choice among candidate values.
- The key must be in a **pre-frozen allowlist**.
- `ref` enters only the comparison. The extraction decision depends on the output's structure
  and the frozen allowlist alone.

And it cannot drive a decision: `stage3_score.py:144` records
`"floor_rule": "applies to canonical_correct ONLY; contract_tolerant never clears a floor"`.

One residual worth keeping visible: Qwen3-4B zero-shot goes 0 canonical → 66 tolerant, a large
gap. That looks oracle-ish from the outside, and it is not: it is one model emitting
`{"result": ...}` where the contract wants the bare value. QWEN-001 then applied the *same*
binomial floor rule to the tolerant counts as a reported-for-information diagnostic and found
it clears **1 of 3** floors rather than 3 of 3. Tolerance was tested and did not rescue it.

---

## 4. The "384 rows" — **resolved, and the report's phrasing is itself the ambiguity**

I expected to find contamination here and did not. Stating that plainly because the initial
reading was mine.

There are **two different 384-row sets**:

| set | path | declared role |
|---|---|---|
| **TRAIN** | `rrp015-20260917/control-prepared/microfit-train.jsonl` | 384 rows, `source_release: forge_v2`, `source_split: train` |
| **SELECTION** | `select-views-001/SELECT_VIEWS.jsonl` | 384 views, `view_role: ADMITTED_FOR_SELECTION` |

`select-views-001/BUILD.json` declares, before any model call:

```
"admissible_uses":   ["SELECTION","DEV"]
"inadmissible_uses": ["TRAIN","CONFIRMATION"]
"training_admitted": false
```

So the selection bank bars training. **Measured overlap between the two sets:**

| level | overlap |
|---|---|
| item (`train.id` ∩ `views.source_id`) | **0** |
| parent (`train.parent_id` ∩ `views.parent_id`) | **0** |
| identical prompt string | **0** |
| identical user-message content | **0** |

384 distinct parents each side, disjoint. Corroborated independently by
`control-preserved-hf/README.md`: the control was "trained on 384 verified Forge RL v2
transition examples, **excluding both scored-parent instruments**."

**Conclusions.**

1. **Nobody trained on the selection bank.** Both the 596M control and the Qwen3-4B LoRA
   trained on `microfit-train.jsonl`. QWEN-FT-001's own header says "the RRP-015 control's
   EXACT rows … same 384 rows, same label masks, same row order" — that is a matched-recipe
   statement about the TRAIN file, not a confession of training on the eval bank.
2. **270/384 is therefore not a training-fit number.** It is **same-generator
   generalisation**: trained on 384 forge_v2 train-split parents, evaluated on 384 disjoint
   forge_v2 train-split parents.
3. **The training gate is not blocked by contamination.** Training authority runs on
   `microfit-train.jsonl`, which is TRAIN material. The protocol's instruction to block the
   branch on an ambiguous "384 rows" is satisfied by resolving the ambiguity, not by stopping.
4. **What is still not measured** is the third category: a *genuinely different held-out
   structure*. Both banks come from one generator's train split. Nothing here speaks to
   transfer beyond forge_v2.

One residual, reported because it is the only thing I found: **40 of the 315 distinct
(family, reference) value pairs in the views also occur as (family, reference) pairs in
training.** That is answer-value coincidence — reference values are small dicts like
`{"A":-15,"B":16,"C":9,"D":-3}` and collisions are expected — not item leakage, since parents
and prompts are disjoint. Its exploitability is already bounded by the published
`constant_majority` floors (1/128, 10/128, 1/128), which a frequent-answer predictor cannot
beat.

---

## 5. Titan/full-delta versus k=0.25 — **VERIFIED as different conditions; no lossless preservation**

The two must not be pooled, and the k=0.25 table does not establish preservation. Both parts
of the protocol's caution are correct.

**First, a correction that changes the experiment's own formalism.** The protocol writes the
static-delta reference as `D = W0 + k*Delta`, a scalar multiple. The authoritative k=0.25
artifact is **not** a scalar multiple. `CORE.json` gives the formula as

```
theta_supa + M_k (x) delta,  mask = per-tensor top-k by |delta|, k = 0.25
```

a per-tensor **magnitude mask**. The programme measured the two operations separately and they
diverge sharply at the same nominal value (`SPARSE001_RESULTS.md`):

| operation at nominal 0.50 | acquisition /384 | retained /47 | long ctx |
|---|---:|---:|---:|
| α = 0.50 — shrink every element by half | 219 | **42** | **164** |
| k = 0.50 — keep the top half at full strength | **235** | 39 | 159 |

16 acquisition rows apart. **Pinning, as the protocol requires when the authoritative artifact
differs: `D = W0 + M_0.25 ⊙ Delta`, the promoted core, hash `a28629c9…`.** The α line is a
separate, already-measured family (INT-001) and is used here as the uniform magnitude control,
not as D.

**Second, the damage.** k=0.25 against the frozen source:

| endpoint | Supa | k=0.25 | source successes lost |
|---|---:|---:|---:|
| acquisition /384 | 14 | 232 | — |
| arithmetic permissive /150 | 47 | 72 | **6 of 47** |
| arithmetic strict /150 | 41 | 63 | — |
| source answers retained /47 | 47 | **41** | **6** |
| long context /256 | 161 | 164 | **7 rows** |
| literal `UNKNOWN` /64 | 13 | **12** | **1** |

`CORE.json` states the decomposition itself: 72 is "41 preserved, 6 lost, 31 newly gained".
Long context nets +3 by losing 7 and gaining 10 at p = 0.63. **Gate v2 PASS coexists with the
legacy per-class retention monitor FAIL on all three classes under both policies.** Abstention
regresses.

Nothing here is lossless. The correct summary is a trade with a favourable net on two
endpoints and observed damage on three.

**Titan's `AMENDMENT_001.json`** records that its own long-context figure was substantially a
BF16 quantisation artifact — so the 162 in Titan's row is not comparable to an FP32 arm's
figure without that caveat, and the report's reuse of Titan numbers inherits it.

---

## 6. The seven-arm oracle bound — scope is narrower than the framing

`BRAINSTORM_CLAIMS_20260918.md` C1: across seven arms (α ∈ {0.25,0.5,0.75,1.0}, k ∈
{0.10,0.25,0.50}) a perfect per-input oracle router adds **+0 long context, +3 acquisition,
+2 retained, +1 abstention** over the best single static arm.

Taken as written this is a statement about **selection among those seven arms' recorded
outputs on those instruments**. It bounds nothing about new prompts, other subspaces, other
layerwise interventions, or future training, and the claim's own "consequence if true" —
*"improve the small model" cannot mean "apply the delta more cleverly"* — is a much broader
statement than the measurement supports. It is well supported for *this delta under scalar and
mask modulation*; it is not a ceiling on weight-space intervention in general.

It is also an **oracle**: label-aware by construction, diagnostic only, and it gets no
deployment credit. The protocol's warning that "a table combining one model's acquisition score
with another's long-context score is not an executed routing result" applies directly and is
why routing here must be scored on actual per-item dispatches.

---

## 7. Four sampled tensor spectra — **energies VERIFIED; quoted ranges describe the sample, not the population**

`RANK001_PREREG.md` characterises the delta's spectrum from **five named tensors** and states
"effective rank 16–72 of 1024; 16 directions carry 59–88% of the energy". I measured **all 140
2D trained matrices**.

**The prereg's energy numbers reproduce exactly**, to three decimals, on every named tensor:

| tensor | prereg top-1 | mine | prereg top-16 | mine |
|---|---:|---:|---:|---:|
| L8 `q_proj` | 0.117 | **0.117** | 0.587 | **0.587** |
| L17 `q_proj` | 0.253 | **0.253** | 0.774 | **0.774** |
| L17 `mlp.down_proj` | 0.223 | **0.223** | 0.675 | **0.675** |
| L27 `o_proj` | 0.369 | **0.369** | 0.884 | **0.884** |
| L27 `mlp.up_proj` | 0.268 | **0.268** | 0.794 | **0.794** |

The effective-rank column differs by a factor of ~2.5 (mine: 28.2, 10.6, 14.3, 6.1, 9.8).
That is a **definitional difference in estimator**, not an error — I use the participation
ratio `(Σσ²)²/Σσ⁴` and state it in the receipt. Their energy measurements, which is what the
compressibility argument actually rests on, are sound.

**Population versus sample:**

| statistic over all 140 2D trained matrices | min | median | max |
|---|---:|---:|---:|
| effective rank (participation ratio, ambient ≤ 1024) | 6.1 | **16.4** | 35.8 |
| energy in the top 16 directions | **0.508** | **0.674** | 0.921 |

The quoted 59–88% band is a five-tensor sample. On the population the band is **50.8–92.1%**
with a median of 67.4%, and **23 of 140 matrices fall below the quoted 59% lower bound**.

**What the spectra do and do not establish, each answered separately:**

- **Whole-model compressibility of this delta — yes, and it is now measured on the population
  rather than inferred from four tensors.** Median 67% of the energy in 16 of up to 1024
  directions. RANK-001's behavioural result confirms the consequence.
- **Independent skill composition — no.** That is C7 and it is untested by construction: one
  delta cannot show that two deltas coexist. The prereg says so itself.
- **A whole adapter's storage size — only partly, and the report conflates two numbers.** The
  factored delta at r=16 is **7.26M parameters / 27.7 MB** (I reproduce 7.21M in the 2D
  factors + 0.0461M 1D kept in full = **7.26M**, matching exactly). But the **deployable
  artifact on the current native path is a full dense model**: every arm this programme built
  is a ~2.38 GB FP32 `model.safetensors`, and the k=0.25 core's own `CORE.json` records
  `weight_bytes: 2384234992`. "A skill is a 28 MB object" is true of the *stored delta* and
  false of the *thing that gets served*. No factored serving path exists or is qualified, and
  no inference-overhead measurement exists — **UNAVAILABLE**.

---

## 8. NEW — per-block norms: C2's evidence is the wrong normalisation

C2 says per-block `||dW||` is 0.169–0.199, "near-uniform", so per-layer magnitude gating has
nothing to exploit.

Measured over the 20 trained blocks:

| normalisation | range | spread |
|---|---|---|
| **absolute** `||dW_block||` | 0.1795 – 0.2027 | **1.13×** |
| **relative** `||dW_block|| / ||W_block||` | 0.000278 – 0.001641 | **5.89×** |

Absolute values: `[0.2027, 0.1990, 0.1927, 0.1865, 0.1826, 0.1799, 0.1809, 0.1811, 0.1803,
0.1805, 0.1822, 0.1849, 0.1875, 0.1880, 0.1859, 0.1859, 0.1857, 0.1824, 0.1811, 0.1795]` —
monotone front-loading at block 8, exactly as the prereg's "slightly front-loaded" says, but
my range sits ~2–6% above the quoted one (unreproduced; my value stated).

**The relative profile is not near-uniform: it falls 5.9× from block 8 to block 26.** A gate
scales an update relative to the weights it perturbs, so the relative norm is the quantity a
per-layer gate acts on. C2's *conclusion* may still be right — C1's separate oracle-headroom
measurement is the stronger leg — but "near-uniform, nothing to exploit" is not supported by
the normalisation that matters. There is a systematic depth gradient.

---

## 9. NEW — the magnitude-versus-subspace question is already answered by INT-001

This is the finding that most changes what the planned experiment should claim.

RANK-001 reports r=16 scoring **235/384, exactly the full delta's 235**, at 2.31% of the
parameters, and reads it as the skill being carried by 16 singular directions. But `P16` does
two things at once: it selects 16 directions *and* it shrinks the update norm to **0.812** of
the full delta (measured; RANK-001 reports 0.812 and I reproduce 0.81225).

INT-001's α line is a **uniform magnitude control that already spans that norm**, because
scaling by α scales the norm by exactly α:

| α (= norm ratio) | acquisition /384 | retained /47 | long ctx |
|---|---:|---:|---:|
| 0.50 | 219 | 42 | 164 |
| **0.75** | **234** | 39 | 163 |
| **1.00** | **235** | 40 | 157 |
| *r = 16 (norm 0.812)* | *235* | *39* | *159* |

Acquisition on the α line is monotone (145 → 219 → 234 → 235), so α ≈ 0.81 is bracketed at
**234–235**. The rank-16 arm scores **235**.

**A plain uniform shrink to the same norm achieves the same acquisition score as discarding
1008 of 1024 directions.** On this endpoint, direction selection contributes ≤ 1 row of 384 —
well inside REP-001's 7-row single-draw spread. The same holds on retention: r=16 retains 39
and scores 159 long context; α=0.75 retains 39 and scores 163.

**Consequence.** RANK-001's Finding 1 is correct as a *storage* result and unsupported as a
*mechanism* result. "A skill is a rank-16 object" should be "this delta can be stored at rank
16 without behavioural loss" — which is valuable, 43× compression — and not "the skill lives
in 16 directions", because a full-rank update of the same magnitude behaves identically. The
protocol's demand for the N16 control is therefore well founded, and the control is registered
in `DMRF001_PREREG.md` with the prediction that it lands at 234–236.

N16 is still worth building: my per-tensor scale factors span **0.712–0.960**, so N16 is a
*heterogeneous* norm match to r=16, not the uniform α ≈ 0.81 that INT-001 already measured. It
tests whether the per-tensor norm *profile* carries anything the global norm does not.

---

## 10. NEW — RANK-001 Erratum 2: the registered cause 3 is falsified

A concurrent session completed RANK-001 at 23:19 and filed
`RANK001_ERRATUM_002.md`: the r=4 arm does not rebuild to its recorded hash, while r=16 and
r=64 both rebuild byte-identical. Its **registered prediction** is that cause 3 —
an ill-conditioned rank-4 subspace — "is the substantive one", expecting
"the σ₄/σ₅ gap ratios to be systematically tighter than σ₁₆/σ₁₇", which would mean r=4's
"170× compression clears all three floors" is a claim about "one draw from a family of rank-4
truncations, not about a determinate object".

I measured the relative gap `(σ_r − σ_{r+1})/σ_r` at all three ranks across all 140 matrices
(`receipts/SPECTRAL_GAPS.json`). Singular **values** are unique even where singular
*subspaces* are not, so this statistic is well posed regardless of the defect.

| rank | median relative gap | tensors with gap < 1% |
|---|---:|---:|
| **4** | **0.0860** | **1 of 140** |
| 16 | 0.0334 | 7 of 140 |
| 64 | 0.0128 | 46 of 140 |

**The prediction is falsified, and the ordering is the reverse of what it expected.** The
σ₄/σ₅ gap is **2.6× wider** than σ₁₆/σ₁₇ and **6.7× wider** than σ₆₄/σ₆₅. No matrix has a
rank-4 gap below 0.1%. The rank-4 leading subspace is the **best separated** of the three, and
r=64 — with the tightest gaps, 46 matrices under 1% — rebuilt byte-identical.

**Two consequences, both favourable to the existing record:**

1. **Cause 3 is eliminated.** The defect is not a property of the rank-4 object, so the
   diagnosis should go to causes 1 and 2 (LAPACK reduction order / thread-count dependence),
   which are build-environment defects.
2. **RANK-001's Finding 2 does not need the weakening the erratum prepared for.** "r=4 is an
   intrinsically unstable object" is not supported: r=4 is the most determinate of the three
   truncations spectrally. The r=4 row should stay suspended until the bytes question closes,
   but it should not be restated as a draw from a family.

This is recorded here rather than by editing their erratum; the receipt is create-only and
their document is theirs to amend.

---

## 11. Runtime and comparability

| lane | interpreter | transformers | torch | used for |
|---|---|---|---|---|
| **qualified** | `.venv-rab001-cu126` | **4.51.3** | 2.10.0+cu126 | every 596M probe; digest `a329eb7bb0` |
| arm builds | `.venv-mq001` | 4.57.1 | 2.10.0+cu126 | `build_rank.py` |
| Qwen lane | `Chat-Bubble/.venv` | **5.14.1** | 2.12.1+cu126 | QWEN-001/QWEN-FT-001 (+peft, bitsandbytes) |

The Qwen lane's own scorer stamps `comparable_to_qualified_a329eb7bb0_runs: false` on every
output. **But QWEN-001 ran a path control and it is strong:** Supa 14/161,
Titan 234/162, Titan-Sparse 232/164 — **all six qualified values reproduce exactly** on the
new runner. The runner is faithful on these instruments, so the 4B figures may be read against
the existing table.

**Residual confounds that are not eliminated**, and they are dtype rather than runner:
Qwen3-4B ran **BF16 with 11 of 40 modules on CPU** (FP32 needs ~16.1 GiB against 15.22 GiB
host RAM) while the Rineya cores ran FP32; QWEN-FT-001's arm ran **4-bit NF4**, which its own
control shows costs 11 long-context rows. And the 150-row arithmetic panel is **NOT RUN** for
any Qwen arm: `verify_hf_candidate` requires all six sidecars byte-identical to Supa plus a
receipt hashing to Supa's weights, so it structurally cannot accept a non-Supa model. **The 4B
has no retention measurement on the gate's own instrument.**

---

## 12. The noise floor that governs every comparison

REP-001, three independent draws of the same recipe, is the scale on which one-draw
differences must be read:

| endpoint | observed spread across draws |
|---|---:|
| acquisition | **7** of 384 (228–235) |
| long context | **13** of 256 (144–157) |
| arithmetic strict | **13** of 150 (51–64) |

Applied to the differences the report leans on:

| comparison | difference | verdict |
|---|---:|---|
| k=0.25 vs Titan, long context | +2 | **within noise** |
| k=0.25 vs Titan, retained | +2 | within noise (no registered spread; smaller than arithmetic's 13) |
| k=0.25 vs Titan, acquisition | −2 | **within noise** |
| k=0.25 vs r=16, long context | +5 | **within noise** |
| 4B vs best 596M core, acquisition | +38 | **exceeds noise** |
| 4B vs best 596M core, long context | +16 | **exceeds noise** |

So "k=0.25 essentially dominates the published core" rests entirely on differences smaller than
the measurement's own draw-to-draw spread. The 4B's two advantages are the only differences in
the whole comparison that clear it. This is encoded in `analysis/paired.py::noise_verdict` and
exercised by `tests/test_paired.py` so no table in this package can quote a within-noise
difference as a direction without the verdict beside it.

---

## 13. Resource state, and one thing the audit prevented

Measured at Stage 0: host 15.22 GiB total, **4.2–4.6 GiB free**, **25.25 GiB committed** —
already ~10 GiB over physical and paging. GPU RTX 4070 Laptop, 8188 MiB, ~7.5 GiB free.
**Disk: 31 GB free of 927 GB (97% used)** — each FP32 arm is 2.38 GB, which bounds how many
arms can coexist.

Guards, both fail-closed, neither weakened: `retention_probe.py:97` `require(5.0)` with
`guard(1.5)` mid-flight at line 240; `long_context_probe.py:114` `require(3.0)`.

**Process inspection found a concurrent battery.** Two live processes (PIDs 183184/183276,
started 23:17:38) were running `supervised_probe.py --kind retention … rank001_r64-retention`
out of the shared pilot clone under `.venv-rab001-cu126`. That is another session completing
RANK-001. Had I started my own battery I would have (a) violated the programme's
one-model-job-at-a-time rule against a host already 10 GiB over-committed, for which
`hostbudget.py` records a measured 10× degradation, and (b) written into
`rank001-20260918/eval/` paths that session owns. **No native run was started.** My own
`build_n16.py` carries a matching `require_free` check rather than relying on noticing again.

---

## Summary of dispositions

| # | claim | verdict |
|---|---|---|
| 0 | W0 / WT / D identity and hashes | **VERIFIED** by recomputation |
| 1 | norm ratio 0.000558 | **CONTRADICTED** — tied-embedding double count; true 0.000576 / 0.000622 |
| 2 | tolerant acquisition 35 → 234 | **CONTRADICTED** — no 35 exists; pair is 14 → 234 canonical, and 234 is the full delta, not k=0.25 |
| 3 | "deep-any" is an oracle | **REFUTED** — one-level unwrap, frozen allowlist, label-blind, cannot clear a floor |
| 4 | the 384 rows are contaminated training material | **REFUTED** — two disjoint 384-row sets; overlap 0 at item, parent and prompt level; 270/384 is same-generator generalisation |
| 5 | k=0.25 shows lossless preservation | **CONTRADICTED** — 6 of 47 answers and 7 long-context rows lost; two gates disagree |
| 5b | `D = W0 + k·Delta` (scalar) | **RE-PINNED** — the artifact is a magnitude mask; α and k diverge by 16 rows at nominal 0.50 |
| 6 | seven-arm oracle bounds future intervention | **OVERSTATED** — bounds selection among those recorded outputs only |
| 7 | four spectra establish compressibility / composition / adapter size | **SPLIT** — energies verified and population-confirmed; composition untested; deployable artifact is 2.38 GB, not 27.7 MB |
| 8 | per-block norms near-uniform | **OVERSTATED** — 1.13× absolute but **5.89× relative** |
| 9 | rank-16 shows the skill is low-rank | **UNSUPPORTED AS MECHANISM** — a uniform shrink to the same norm scores the same |
| 10 | Erratum 2 cause 3 (ill-conditioned r=4 subspace) | **FALSIFIED** — r=4 gaps are 2.6× **wider**; cause 3 eliminated |


---

# ERRATUM 1 — two claims in this document were wrong, and they share a cause

**Appended 2026-09-18. The body above is left unedited; this is the correction.**

**§3 "deep-any — the named mechanism does not exist" is WRONG.** It exists. The peer's report
carries a `deep-any (format-free)` column with Supa at 35, Titan at 234 and Qwen3-4B at 81, so
"35 → 234" is a like-for-like comparison inside one parser and **not** the condition splice §2
called it. What I had actually established was that it is not implemented in
`rineya-dataset-retention-records/` or in `experiments/rab001_exploration/`. That narrower
statement still holds and is still worth acting on — the headline premise cannot be reproduced
from committed code — but it is not what §2 and §3 say.

**§2's "there is no 35" is wrong for the same reason.**

**The cause, which recurred a third time.** Later in this session I wrote that no Supa per-item
PROCESS artifact "exists on this host". The pilot clone holds **36** PROCESS artifacts I had not
searched, including the one that answered my own question — the peer found it and used it to
test my hypothesis (result: not supported, P(X≥3) = 0.1023).

Twice the claim outran the search: I looked in one tree, found nothing, and wrote "does not
exist" instead of "is not in X". **The fix is phrasing discipline — state the scope inside the
sentence** — and it costs nothing.

A third error in this session was a different class and is kept distinct rather than lumped in:
I reported a +7/+10 long-context oracle headroom as available without applying
`analysis/paired.py::noise_verdict`, which I wrote for exactly that purpose and had applied to
the peer's margins but not my own. Three errors, two classes.

Current corrected position on everything above: see `COMBINED_RECORD_20260918.md`.
