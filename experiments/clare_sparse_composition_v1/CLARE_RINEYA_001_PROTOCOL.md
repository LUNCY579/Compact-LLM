# CLARE-RINEYA-001 — Capacity-Aware Sparse Skill Composition

**Status: REGISTERED before any native arm exists.** Registered configuration:
`configs/clare_rineya_001.json` (its canonical-JSON sha256 is recorded in every run artifact as
`config_sha256`). Changing any registered value is a dated AMENDMENT in a new file, never an edit.

- Branch: `research/titan-clare-sparse-composable-skills-v1`
- Parent: `research/titan-model-soup-mini-v1` @ `816a1e8f4d1ab1c9d1ab847697d962a141521bac` (not modified)
- Method source: CLARE, Fu, Lou, Liao & Yu, *Scalable Class-Incremental Continual Learning via a
  Sparsity-Based Framework*, arXiv:2609.17026 (BMVC 2026). Read in full for this registration.

---

## 1. Question

> Can learned, capacity-aware sparse parameter allocation make additive skill deltas in a compact
> (596M) language model interfere less, retain better and compose better on unseen skill
> combinations than dense sequential adaptation, independent-delta addition, magnitude pruning, or
> random sparse allocation at identical capacity?

The implication chain under test — **not assumed**:

```
PARAMETER SEPARATION -> LOWER FUNCTIONAL INTERFERENCE -> BETTER RETENTION -> BETTER COMPOSITION
```

### 1.1 What is mechanical and what is empirical

| quantity | status in the disjoint arms (D, E) |
|---|---|
| support intersection `|supp_i ∩ supp_j|` | **0 by construction** (strict allocation) |
| parameter dot product `<Δ_i, Δ_j>` | **exactly 0.0 by construction** |
| cosine similarity | **0 by construction** |
| hidden-state / logit independence | EMPIRICAL — not implied |
| retention of earlier skills | EMPIRICAL — not implied |
| accuracy on unseen combinations | EMPIRICAL — not implied |

`f(θ + Δ_A + Δ_B)` is not a linear combination of `f(θ)`, `f(θ + Δ_A)`, `f(θ + Δ_B)` in a
transformer. Disjoint supports remove *parameter-space* overlap only. The zero rows of the table
above are reported as mechanical checks and are **never** presented as findings.

The distinctions kept apart throughout: geometric orthogonality · support disjointness ·
gradient alignment · functional independence · retention · composability.

## 2. Lineage: why this extends Titan / model soup

```
naive model averaging        SOUP-MINI-001 (Phase A: five 0.5 soups)
  -> additive delta combination   FA-001, CONFIRM-001/002 (ADD-100, never-average)
  -> delta geometry / cosine      SOUP-MINI-001: cos(dSparse, dOpallo) = 0.0022
  -> sparse delta analysis        Titan-Sparse (top-25 % |dT|; 28.5 % of its support outside stored dT)
  -> orthogonality observations   delta_subspace_geometry_v1; ARMPERP (exact orthogonality bought nothing)
  -> learned support allocation   CLARE-RINEYA-001 (this study)
  -> functional composability     the 2x2 interaction + held-out combinations below
```

**Model soup asks** how already-learned models or deltas should be combined.
**CLARE-RINEYA asks** whether controlling *where* each skill learns, from the start, makes later
combination easier.

What the parent line already measured, and which this study must not re-discover:

- every post-hoc separation tested so far equals plain addition or a matched-norm scalar: a random
  mask acts like its expectation (CONFIRM-001 C6, 279 = 279), DARE/DISJOINT-DARE = ADD-100
  (409/412/415 vs 413), module routes = scalar (OSIDE-001 NULL), exact orthogonality = nothing
  (ARMPERP 66 vs 68);
- interference sat on low-margin rows and was nested in the coefficient, not in coordinates;
- low cosine (0.0022) coexisted with measurable functional interaction (FA-001 κ −0.064 clipped,
  −0.104 unclipped).

**The new ingredient is training-time allocation.** In every earlier test the deltas were trained
dense and separated afterwards; here each skill *learns inside* its support. CLARE's own ablation
(IN-R 10 tasks) put learned vs random masks at 79.73 vs 79.22 (−0.51 points) and at −5.65 points only
after 100 tasks. **For four skills the prior expectation is therefore a small or null E−D effect.**

Norms from the parent line are NOT reused: ‖dT_stored‖ = 0.8035 (BF16-stored) vs the 1.8464 dense
delta the retention law was fitted on are different conventions. Every norm in this study is
measured fresh from this study's own FP32 artifacts.

## 3. Model, footprint, runtime

- Base: `Rineya-Supa-1.7.0-devfreeze` (sha256 `962abc5b…`), Qwen3ForCausalLM, 28 layers, hidden
  1024, intermediate 3072, vocab 151936, tied embeddings, 596,049,920 parameters.
- Stored BF16 (as published); **computed in FP32**. Every weight is BF16-representable, so the
  per-module FP32 upcast is exact. Gate: the patched model's logits must match a plain FP32 load
  (tiny-model tests: bit-identical logits and gradients; native: ≤ 1e-4 and equal argmax).
- **Footprint: the MLP weights (gate/up/down) of blocks 8–27 = 60 tensors, 188,743,680
  coordinates.** Same block range as Titan. Attention, norms, embeddings and blocks 0–7 are frozen
  in every arm. Chosen so that a DENSE AdamW state (arms A/B and the E discovery probe) fits on the
  8 GB card without host offload; it is a registered restriction, not a claim that skills live in
  MLPs. The full 314.6M-coordinate tail is supported by the code but not registered.
- Runtime: hermes venv, Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1; TF32 off,
  matmul precision "highest", deterministic algorithms, `set_per_process_memory_fraction(0.78)`,
  eager attention, gradient checkpointing, `enable_thinking=False`.

## 4. Allocation (CLARE two-stage), schedule, arms

`ρ` = fraction of the *currently free* capacity left free after each skill.
`k_t = min(|A_t|, max(1, round((1 − ρ)|A_t|)))` (CLARE Eq. 6–7, round-half-up explicit).
**Registered ρ = 0.9** → budgets 18,874,368 / 16,986,931 / 15,288,238 / 13,759,414 (34.39 % claimed).
Skill 1 therefore gets ~1.9× the parameter count of a rank-16 LoRA on every linear of the model
(10.09M). The ρ ∈ {0.80, 0.90, 0.95, 0.975} sweep is implemented (`cli.py plan` prints it) and
**not registered**.

Deviation from CLARE: the paper includes every threshold tie, so its masks can exceed `k_t`. Here the
budget is **exact** (ties broken by a seeded hash, then global index), because the learned-vs-random
comparison requires identical capacity.

| arm | procedure | role |
|---|---|---|
| **A dense_sequential** | each skill trains a zero-initialised dense delta over the whole footprint, on top of the frozen sum of the earlier skills | forgetting/interference baseline |
| B independent_sum | each skill trains a dense delta from the base alone; composition = Σ Δ | standard additive baseline |
| C magnitude_sparse | B's deltas magnitude-pruned to the same budgets (supports may overlap) | is sparsity alone enough? |
| **D random_disjoint** | same budgets, random disjoint masks, masked retraining on top of frozen earlier skills | separation without learned discovery |
| **E learned_disjoint** | Stage 1: L1 probe on free coordinates → top-\|Δ\| mask → **reset** → Stage 2: masked retraining → freeze | CLARE (primary) |
| F shared_private | a jointly-learned shared support trainable by every skill + CLARE private masks | config support only; not run in 001 |

Primary arms: **A, D, E**. Secondary (run if budget allows, seed 0 first): B, C.

- **Stage 1 (E):** dense trainable delta over the free coordinates only (gradient mask), the same
  recipe compressed to the first 25 % of the skill's update groups, plus a decoupled proximal L1 step
  after each AdamW step with threshold `lr_t × λ`, **λ = 0.1** (Adam-scale; CLARE's λ = 1e-4 is
  SGD-scale and does not transfer). Importance `q_j = |Δ_j|` (CLARE Eq. 5), free coordinates only.
- **Reset:** every probe value and the probe optimiser are discarded. The reset is verified: the
  checksum of the effective FP32 weights before the probe must equal the checksum after it.
- **Stage 2 (D, E):** only the mask's coordinates exist as parameters (sparse path); no L1.
- **Freeze:** the skill's delta joins the frozen accumulated delta; its coordinates are claimed and
  can never be re-claimed (strict mode). Frozen deltas are checksummed before and after every later
  stage.
- λ build gate (TRAIN rows only, before any arm): if the probe leaves fewer non-zero free coordinates
  than 2× the budget, λ is lowered to 0.03, then 0.0, and the change is recorded as an amendment.

## 5. Data (MAXRL-COMP-002, read in place)

`C:/Users/clclc/BusinessProject/Rineya-Dataset`, manifest sha256 `86ac8f0d…`, code commit
`847cd27e`. Every file is verified against the manifest before use. The dataset's verifier is
imported with bytecode writing disabled, so the dataset tree is never written.

- **Skills:** R reference binding · U state updates · S exact subset selection · P precedence.
- **TRAIN signatures:** R, U, S, P and the seen pairs **RU, US, SP**. **Held out** (absent from
  TRAIN/CALIBRATION/WARMSTART): **RS, RP, UP, RUS, RUP, RSP, USP, RUSP**. RU/US/SP are therefore
  *seen-pair* signatures, not held-out composition.
- **Training rows:** the atomic banks `manifests/atomic_{R,U,S,P}.json` only (no bridges, no
  answerability bank). 1,000 rows per skill, taken in ascending
  `sha256("CLARE-RINEYA-001|order|<seed>|<skill>|<id>")`, identical across arms. Caveat: all four
  atomic banks are views of the **same 3,000 parent worlds**, so the donor data differ by stage, not
  by world.
- **Target:** oracle `references[0]`. **Declared deviation:** the corpus was built for online RL;
  this study uses SFT on one valid certificate. Loss: answer tokens plus `<|im_end|>` (Titan's
  contract, vendored `compile_row`).
- **Evaluation:** DEV only (role `dev`, tag `core`), 64 rows per signature × 15 signatures, taken in
  ascending `sha256("CLARE-RINEYA-001/dev-v1|<id>")`. **TEST, CHALLENGE and the guard banks are never
  read** by this study: COMP-RS-001, a concurrent live study, registered TEST/guard_test as
  one-read-per-arm, and CHALLENGE is a distribution shift. CALIBRATION (train signatures only) is
  used only for the LR build gate and smoke readings.
- **Leakage controls (code-enforced, tested):** purpose-gated access (train/discovery → TRAIN only;
  evaluation → DEV only; every other role refused); parent-world disjointness between TRAIN and the
  DEV sample is checked; no DEV/TEST/CHALLENGE row or answer ever reaches discovery, selection or
  training. Sibling rows of one DEV world can share answers across signatures (e.g. S→RS 39/250);
  that affects all arms equally but is why **floors are reported per signature**
  (`data.shortcut_floors`) and composition claims use floor-adjusted accuracy as well.

## 6. Training recipe (Titan-matched, vendored)

FP32 AdamW (`foreach=False, fused=False`), β = (0.9, 0.95), ε = 1e-8, weight decay 0 (required: it
keeps masked coordinates exactly at zero), global grad-norm clip 1.0, per-row forward/backward
without padding, loss = summed CE / the group's supervised tokens, update groups ≤ 2048 tokens
(vendored `group_updates`), WSD 20/60/20 schedule with sqrt decay to 0.1× (vendored `rate_at`) by
processed-token progress. A fresh optimiser per skill stage in every arm.

**Peak LR build gate (outcome-blind, before any arm):** train R dense at 1e-5 (Titan's LR); score
the 150 CALIBRATION R rows; if accuracy ≤ base + 0.10, repeat at 3e-5, then 1e-4. Freeze the first
LR that clears the bar **for every arm**. If none clears it, the study is BLOCKED on acquisition
(not a negative result). This gate never reads DEV/TEST and never looks at D or E.

## 7. Endpoints and decision rules

All accuracies are exact verification of greedy generation (strict JSON, any valid certificate),
on fixed row sets, paired by row across arms. Seeds {0, 1} vary the row order and the mask RNG;
the DEV rows are the same for both seeds, so seeds replicate *training* variance, not rows.

**Evaluation schedule** (per arm, seed, order): the base model once on all 15 signatures; after each
stage k the 4 atomic signatures plus every signature whose skills are all learned by stage k; after
the final stage all 15 signatures.

**Registration-time observation (native qualification, before any arm, CALIBRATION only):** Supa
scores 0/42 strict on the train signatures because it answers with a bare list instead of the
required `{"answer": [...]}` object. Acquisition therefore includes learning the answer format, and
in the sequential arms the first skill learns the shared format for all the others. No rule changes;
this is why seen pairs, floors and retention are reported beside every composition number.

### 7.1 Four separately reported outcomes
1. **Acquisition** — accuracy on skill s's atomic rows right after stage s, per arm.
   *Interpretability gate:* a skill whose acquisition does not exceed the base by ≥ 0.10 in an arm is
   flagged NOT ACQUIRED, and contrasts involving it in that arm are reported as uninterpretable.
2. **Retention** — the full retention matrix (skills × stages), `forgetting_final`,
   `forgetting_max` and backward transfer (`metrics.retention_matrix`). Damage uses the
   damage-denominator rule: losses over the rows the arm got right at acquisition.
3. **Composition** — accuracy on the 8 held-out signatures at the final stage (and at every prefix
   stage that covers them), raw and floor-adjusted; seen pairs reported separately.
4. **General retention** — the programme's n = 47 panel (`retained = 47 − lost_total`,
   Supa-correct denominator). **SECONDARY and pending an operator decision**, because the
   authoritative probe writes into another session's worktree. Format and value channels are split
   (`lenient_value_present`), since R/U/S/P training imports a JSON answer convention that the
   typed scorer reads as damage.

### 7.2 Registered comparisons
- **P1 (primary) — composition, E vs D:** pooled accuracy over the 8 held-out signatures at the final
  stage. **E > D** is claimed only if the difference is positive in BOTH seeds AND the
  parent-cluster bootstrap 95 % interval of the seed-averaged paired difference excludes 0. **E < D**
  is symmetric. Otherwise: **no detectable difference**, reported together with the MDE (below).
  Never "equivalent".
- **P2 (primary) — retention, E vs D:** mean `forgetting_final` over the first three skills of the
  order, same rule.
- **P3 (primary, functional) — interaction, E vs D:** per skill pair, the teacher-forced second
  difference `I = lp(θ+Δa+Δb) − lp(θ+Δa) − lp(θ+Δb) + lp(θ)` (unclipped), κ normalised by Δa's own
  gain (FA-001's estimator) and by the sum of both gains, and the log-softmax vector interaction
  (full vocabulary and top-32 union). Descriptive, with row-bootstrap intervals.
- **Secondary:** E vs A and D vs A (does separation beat dense sequential?), E vs B, C vs B, and the
  same three outcomes for each.
- **The opposite result is a registered outcome.** If D and E are perfectly separated (mechanical),
  P1 finds no E−D difference, and neither D nor E beats A on retention or composition, the report
  states: *parameter separation — random or learned — did not translate into functional
  composability here.*
- **Norm confound:** ‖Δ_s‖ is reported for every skill and arm. If E and D differ in total norm by
  more than 10 %, an exploratory norm-matched readout is added and labelled EXPLORATORY.
- **Order:** the secondary order (P, S, U, R), seed 0, arms D and E. If the E−D sign flips between
  orders, the result is flagged ORDER-DEPENDENT and escalation is recommended; no permutation sweep
  is run in 001.

### 7.3 Power, honestly
Per seed, P1 uses 8 × 64 = 512 row pairs. With a paired discordance rate around 0.15, the SE of the
difference is ≈ √(0.15/512) ≈ 1.7 points, so the **MDE ≈ 5 points** (α 0.05, power 0.8). CLARE's
own four-to-ten-task gap was ~0.5 points. **This study can detect only a large learned-allocation
advantage; a null here is not evidence of equivalence.** The same arithmetic gives an MDE of about
7 points per skill for retention (64 rows).

## 8. Geometry, layer and rank diagnostics (all offline, all reported as geometry)

For every pair of skill deltas in every arm: cosine, raw dot, norms, support overlap, Jaccard,
sign agreement on overlapping non-zeros. For every skill: allocated coordinates per slot/layer,
active non-zeros (separately from allocated capacity), delta energy by layer, free capacity after
each skill. Layer specialisation is **observed, never enforced**. Rank structure (top-k singular
energy per 2-D slot) is an offline diagnostic only; **rank truncation is never combined with
masks in 001.** Gradient cosines are optional and not registered.

## 9. Controls (all enforced in code and recorded in manifests)

Same base, tokenizer, train rows and order, optimiser family, recipe and update groups for every arm;
identical capacity for D and E per skill (tested); held-out banks never used for allocation
(gated and tested); no tuning against TEST/CHALLENGE (never read); seeds, artifact hashes, the parent
commit, model/tokenizer identity, runtime, trainable counts and active non-zeros are all recorded in
each skill manifest. D and E both retrain through the same sparse code path. The dense and sparse
paths are the same algorithm but agree only to float32 rounding, not bit-for-bit (measured).

## 10. What would and would not follow

- E > D (P1 or P2) → learned discovery carries structure beyond capacity partitioning, *for these
  skills and this footprint*.
- E ≈ D, both > A → separation helps, discovery does not add (CLARE's rule does not transfer).
- E ≈ D ≈ A (or < A) → separation is insufficient for composability here.
- Nothing here licenses "orthogonal = independent": interaction is measured, not inferred.
- No claim extends to attention, to the full tail, to other ρ, to TEST/CHALLENGE, or to RL training.

## 11. Explicitly out of scope for 001

Main/parent-branch changes; ρ sweeps; CLARE × rank truncation × gradient routing; MoE, task IDs or
per-skill models; training on held-out banks; changing dataset definitions; changing base weights;
native training before qualification. Gradient-aware reuse is a separate design note
(`GRADIENT_AWARE_REUSE_DESIGN_NOTE.md`).
