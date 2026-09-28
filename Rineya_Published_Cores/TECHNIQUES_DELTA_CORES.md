# How the Titan cores were obtained — techniques, notation, and what each number means

**Written 2026-09-18. Covers `Rineya-Titan-Experimental-1.0.0-devfreeze` and
`Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, and the notation (`k`, `α`, `r`, `N`) used
across the delta-weight screens behind them.**

**Status, stated first because it governs everything below:** every artifact described here
carries `quality_improvement_established: false` and `release_permitted: false`. These are
**experimental devfreeze cores kept for research provenance**, not releases. Their presence in
this private repository is a backup, not a promotion.

---

## 1. The common structure: one frozen source, one update, many recipes

Every core here has the same shape:

```
theta_arm  =  theta_supa  +  f( delta )
delta      =  theta_trained  -  theta_supa          (per tensor, FP32)
```

- **`theta_supa`** — `Rineya-Supa-1.7.0-devfreeze`, frozen, never modified.
  `model.safetensors` sha256 `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
  Stored BF16, upcast to FP32 for all arithmetic.
- **`theta_trained`** — the output of one training run (§2).
- **`f`** — the recipe. `f = identity` gives the trained model; every other core here is a
  different `f`, and `f` is what `k`, `α` and `r` parameterise.

**Consequence worth stating plainly:** given `theta_supa` (in this repo) and the consolidated
trained tensors (in `research-records/delta-weight-screens-20260918/_trained_tensors/`), **every
arm ever built in this programme can be regenerated exactly.** That is why those two things are
backed up and the dozens of intermediate arms are not.

`delta` touches **220 of 310** stored tensors — 20 of the 28 transformer blocks, plus their 1-D
norm vectors. The other 90 tensors, including the embedding, are copied from the source
untouched.

---

## 2. Where `delta` came from

The delta behind both Titan cores is the **RRP-015 control arm**:

| | |
|---|---|
| corpus | 384 one-step state-transition rows, drawn at selection seed 17 |
| processed tokens | 54,172 |
| supervised tokens | 5,254 (9.7% — only the assistant response plus EOS is supervised) |
| optimizer updates | **28** |
| optimizer | MQ CpuAdamW, β = (0.9, 0.95), ε = 1e-8, grad clip 1.0, weight decay 0 |
| schedule | WSD 20/60/20 decaying to 0.1 × peak, peak LR 1e-5 |
| precision | **FP32 throughout**, CPU optimizer moments |
| consolidated output | `delta_T_seed17.safetensors`, sha256 `07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d` |

So the entire update is **28 optimizer steps on 5,254 supervised tokens**. Its Frobenius norm is
**0.8298**, against the model's **1440.86** — a relative change of **0.000576** whole-model
(0.000622 over the trained scope). The largest single tensor moves by 0.00207 relative.

> **Correction of record.** An earlier pre-registration quoted this relative norm as 0.000558.
> That figure was a **tied-weight double count**: `tie_word_embeddings: true`, so walking named
> parameters exposes `embed_tokens` twice and adds 364.2551² to the denominator
> (1440.8616 → 1486.1911). The correct values are the ones above.

Three further independent training draws exist, at selection seeds 23 and 29 (REP-001) and on a
different corpus (COMPOSE-001's `delta_H`, 648 epistemic/safety-boundary rows, 54,217 tokens,
28 updates). All four are in the backup.

---

## 3. `k` — magnitude pruning *(the recipe behind Titan-Sparse)*

**`k` is the fraction of each tensor's update elements that are KEPT, chosen by absolute
value. Everything else is set to exactly zero.**

```
for each trained tensor t:
    n      = number of elements in delta_t
    keep   = floor(k * n)
    order  = argsort(|delta_t|, descending, STABLE)     # ties -> lower flat index first
    M_t    = 1 on order[:keep], 0 elsewhere
theta_arm = theta_supa + M * delta
```

So **`k = 0.25` means: within every trained tensor independently, keep the 25% of update
elements with the largest magnitude and discard the other 75%.** Titan-Sparse stores
78.7M of the delta's 314.6M element values as non-zero; the rest are exactly the source weights.

Four design points, each of which was a decision and not a default:

- **Per tensor, not global.** A single global threshold would let one large-norm tensor consume
  the whole budget and leave others untouched. Per-tensor keeps the *shape* of the update.
- **Ties broken deterministically.** The first implementation used `torch.topk(sorted=False)`,
  which is unordered: two builds from identical inputs produced two different arms
  (`cde1502a` vs `2f57fe72`) with identical norms. That is SPARSE-001 Erratum 1. The fix is a
  **stable** descending argsort, making the mask a function of the input alone. A full
  deterministic re-measurement afterwards reproduced every endpoint exactly.
- **Why it works at all.** The update is heavy-tailed in magnitude: **25% of its elements carry
  84% of its Frobenius norm.** Pruning the small-magnitude tail removes most of the *elements*
  and little of the *norm*.
- **Why pruning beats scaling.** At a matched nominal 0.50, pruning retained **+16 acquisition
  rows** over uniform scaling. SPARSE-001's reading: the small-magnitude tail carries
  disproportionate *forgetting*. REP-001 then found this replicates in direction across three
  independent draws (3/3) but not in magnitude (+0 to +15) — pruning is **a repair that pays
  most where the damage is worst**, so its size is not a constant of the method.

### FP32 is mandatory, and this is a trap

**A BF16 round zeroes 28.49% of a `k = 0.25` arm's *kept* delta elements, and 60.61% of a full
update's elements** — because the update is ~1e-4 the scale of the weights it is added to, so
`w + δ` rounds back to `w` in BF16 for most elements.

Titan-Sparse is therefore stored **FP32 only** (2,384,234,992 bytes), and its `config.json`
sets `torch_dtype: float32` — the one sidecar that differs from Supa — specifically to stop
`from_pretrained(torch_dtype="auto")` from silently destroying it on load. **Loading
Titan-Sparse in BF16 does not give you a degraded Titan-Sparse; it gives you something much
closer to plain Supa.**

---

## 4. `α` — WiSE-FT interpolation *(and what Titan-Experimental actually is)*

```
theta_arm = theta_supa + alpha * delta          # uniform scalar, every trained tensor
```

`α = 0` is the frozen source, `α = 1` is the trained model. INT-001 measured
α ∈ {0.25, 0.50, 0.75, 1.00}; **α = 0.50 won the registered selection rule** and produced the
highest long context measured on this delta (164/256).

### Titan-Experimental is a BF16 export of the α = 1 arm — and that matters

`Rineya-Titan-Experimental-1.0.0-devfreeze` is `theta_supa + delta`, exported **BF16**
(1,192,135,064 bytes), weight sha256
`537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.

**Per §3, that export zeroes 60.61% of the update's elements and erases 24 of 220 tensors
entirely.** So the artifact that was *evaluated* is not the α = 1 arm — it is an accidental,
uncontrolled interpolation somewhere below it. Its measured long-context advantage was
substantially a **quantization artifact**, and that is recorded in the core's
`AMENDMENT_001.json`, not buried.

Kept as-is deliberately: it is the artifact whose numbers are on record, and re-exporting it
would silently change what those numbers refer to. Anyone wanting the true α = 1 arm should
rebuild it in FP32 from the backed-up tensors.

---

## 5. `r` — SVD rank truncation, and why it is a *storage* result

```
for each 2D trained tensor:  U, S, Vh = svd(delta_t)
                             delta_r  = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]
1D norm deltas are kept in FULL (rank truncation is undefined for a vector; the 80 of them
total 0.015% of the delta).
```

| r | stored params | share of delta | size | ‖δ‖ retained | acquisition |
|---|---:|---:|---:|---:|---:|
| 4 | 1.85M | 0.59% | 7.1 MB | 0.626 | 211 / 384 |
| **16** | **7.26M** | **2.31%** | **27.7 MB** | **0.812** | **235** |
| 64 | 28.88M | 9.18% | 110.2 MB | 0.934 | 236 |
| *full* | *314.62M* | *100%* | *1.26 GB* | *1.000* | *235* |

**`r = 16` reproduces the full delta's score at 43× compression.** That is a real and useful
compression result.

> **It is NOT evidence that the skill occupies 16 directions**, and an earlier write-up said so
> before being corrected. A peer experiment (DMRF-001 `N16`) took the **full-rank** delta and
> merely rescaled it per tensor to r=16's norm — no directions removed at all — and it scored
> **236 / 3-3 / 39 retained / 162 long context**, matching r=16 on every endpoint inside the
> measured draw spread. And `α = 0.75` (full rank, 0.75 norm) scored 234 / 39 / 163.
>
> **r = 16 ≈ N16 ≈ α = 0.75.** What rank truncation buys is **magnitude reduction**, not
> direction selection. `r` is a good compressor and a bad explanation.

`N` in `N16` denotes exactly that control: *norm-matched to rank 16, full rank retained.*

---

## 6. What the endpoint numbers mean

| endpoint | instrument | what it measures | permission |
|---|---|---|---|
| **acquisition** *(x / 384)* | SELECT-VIEWS-001 | the newly trained one-step transition contract, exact canonical JSON match | **selection-admissible**; per-family floors 0.2344 / 0.0781 / 0.0703 published pre-hoc |
| **floors** *(x / 3)* | same | per-family exact binomial 95% **lower bound** must exceed the stronger of `copy_before` and `constant_majority` | — |
| **arithmetic** *(x / 150)* | frozen retention panel | pre-existing arithmetic, `permissive / strict` reported separately, never pooled | the gate's own instrument |
| **retained** *(x / 47)* | 47 source answers | pre-existing answers still produced | the gate's own instrument |
| **long context** *(x / 256)* | Long-Context Demonstration Suite | 1,000-word-context recall | **SPENT** — report-only, may never select an arm |
| **literal `UNKNOWN`** *(x / 64)* | the same bank | emits the token `UNKNOWN` on unanswerable rows | same bar — report-only |
| **`J_decision`** | HSKILL-VIEWS-002 | abstention *discrimination*: sensitivity + specificity − 1, floor exactly 0 | selection-admissible |

**Gate v2** = `arithmetic panel ≥ 62 AND retained ≥ 37 of 47 AND all three family floors
cleared`. Both Titan cores pass it. The older per-class monitor reports FAIL for every arm ever
measured here and is reported, not used.

**One caveat that applies to every retention figure in this family.** REP-001 measured, across
three independent training draws, that control long context spans **13 rows** and strict
arithmetic **13 rows** while acquisition spans only 7. So **acquisition figures are
trustworthy; single-draw retention figures are not**, and differences under ~13 long-context
rows should be read as within-noise.

---

## 7. Reproducing either core

```
theta_supa      : Rineya-Supa-1.7.0-devfreeze/model.safetensors   sha 962abc5b…
trained tensors : research-records/delta-weight-screens-20260918/_trained_tensors/
                  delta_T_seed17.safetensors                      sha 07156fb5…

Titan-Sparse  = build_sparse.py --keep 0.25   (FP32, stable-argsort tie-break)
Titan-Experimental = theta_supa + delta, exported BF16  (lossy — see §4)
```

Builders, pre-registrations, results and errata are in
`research-records/delta-weight-screens-20260918/`.

**Two reproducibility facts that cost real time to learn:**

1. **Pin the thread count.** `torch.linalg.svd` on this build returns different results at
   different thread counts (verified at r = 4, 16 and 64; an in-process repeat differs on 0 of
   140 tensors). Rank-truncated arms are reproducible only at the thread count recorded in
   their receipt.
2. **Do not verify an arm by its `model.safetensors` sha256.** `safetensors` serialises
   `__metadata__` from a hash map whose iteration order is randomised **per process**, so any
   writer passing two or more metadata keys produces a **coin-flip file hash over a bitwise
   identical model**. This generated two false "non-reproducible arm" alarms before it was
   diagnosed; in both cases 0 of 310 tensors actually differed. **Compare the data section or
   per-tensor hashes.** The recorded `out_weight_sha256` values in each core's `CORE.json` are
   valid identifiers of *a* valid encoding, not the only one.

---

## 8. What these cores do and do not establish

**Established.** A 28-update, 5,254-supervised-token update installs a new one-step transition
contract worth **+220 acquisition rows** over the frozen source, at no *detectable* retention
cost, and both compression recipes (`k = 0.25`, `r = 16`) preserve that while shrinking what has
to be stored — r = 16 by 43×.

**Not established, and measured to be false in three specific cases.**

- *Delta modulation cannot raise general capability above the base.* Across seven arms the best
  long context is 164/256 against the base's own 161, and a perfect per-input oracle over all
  seven would add **+0** long-context rows.
- *Directions do not explain the compression* (§5).
- *Two skills do not compose.* Adding a second, separately trained, budget-matched delta cost
  **60 acquisition rows** even though the two rank-16 subspaces are near-orthogonal (mean cosine
  0.151 against ≈0.125 for chance). Composition also came out **50 rows worse than simply
  mixing the two corpora into one training run**.

The working account consistent with all of it is that **magnitude is the operative variable**
for this delta. That is a hypothesis under test, not a result.
