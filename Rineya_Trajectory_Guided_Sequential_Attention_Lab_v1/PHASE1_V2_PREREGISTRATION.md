# TGSA Phase 1 preregistration — study v2

**Registered:** 2026-08-29
**Study ID:** `tgsa-phase1-identifiability-v2-2026-08-29`
**Supersedes:** `tgsa-phase1-model-identifiability-2026-08-28` (no results were ever generated)
**Branch:** `research/tgsa-phase1-identifiability-v2-launch-ready`
**Pinned parent commit:** `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e`
**Configuration:** `configs/phase1_identifiability_v2.json`

This amends `PHASE1_PREREGISTRATION.md` in response to `PHASE1_LAUNCH_AUDIT.md`,
which returned **NO-GO** on the v1 head. The mechanism, the alignment falsifier,
the statistical estimator, and the confirmatory-exposure policy all changed, so
this is a new study ID with a new output root rather than a patch. The v1 study
and its no-result state remain unchanged and immutable.

`PHASE1_V2_REMEDIATION.md` maps every audit finding to the change that closes it.

## Research question

Does the learned finite-difference signal add held-out capability beyond the
sequential cascade itself and beyond an information-matched learned history
pathway?

## Hypotheses

- **H1a:** `learned_delta` exceeds `history_control` on the macro-average of
  held-out pointer-chasing and state-update strata.
- **H1b:** `learned_delta` exceeds `cascade` on the same estimand.
- **H1c:** aligned `learned_delta` exceeds the same trained model when each
  example's trajectory control is replaced by another **same-stratum,
  same-length** example's *aligned* control.
- **Guardrail:** learned delta is non-inferior to history control on direct lookup.

Failure of H1a prevents a claim that finite differences themselves add
capability. Passing only H1b permits, at most, the interpretation that an extra
learned recurrent pathway helps. Failure of H1c prevents a claim that
example-aligned trajectory content is causal.

## What changed from v1, and why

### 1. The control path is now genuinely information-matched

v1 routed the control through `RMSNorm -> Linear -> SiLU -> Linear`. `RMSNorm`
discards magnitude and `SiLU` is not injective, so the downstream mechanism never
saw the raw history mean or the raw finite difference. The registered claim of
information matching was therefore not realized, and a `learned_delta` advantage
would only have compared two matched *normalized nonlinear encodings*.

v2 makes the encoder a **residual around the identity**:

```
encode(x) = x + up(SiLU(down(RMSNorm(x))))
```

`up` is zero-initialized, so at step 0 the encoder is exactly the identity, and
at every step the raw same-shaped control tensor remains additively present on
the path the mechanism consumes.

The information-matching argument is now exact. Given the current carrier `c_j`:

- `learned_delta` receives `d_j = c_j - c_{j-1}`, and `c_{j-1} = c_j - d_j`;
- `history_control` receives `m_j = (c_j + c_{j-1}) / 2`, and `c_{j-1} = 2 m_j - c_j`.

Both are affine bijections of `c_{j-1}` given `c_j`, under identical
parameterization and identical initialization. The comparison therefore tests the
**trajectory inductive bias**, not access to extra history.

A related v1 defect is also fixed: the model-wide `apply(...)` initializer
silently overwrote the trajectory-encoder initialization with a generic
`N(0, 0.02)` draw, so the documented encoder initialization never took effect.

### 2. Shared initialization is constructed, not hoped for

v1 reset the global seed and then built structurally different module graphs. The
learned trajectory factors consumed random numbers that the cascade arm did not,
so "same seed" did not imply "same shared tensors" — and the preflight audit
that claimed to prove equality would have raised.

v2 builds **one canonical `learned_delta` supernet per seed** and copies every
exactly-compatible tensor into each arm by key, dtype, and shape. Arm-only
tensors are initialized under an explicitly derived, recorded seed. Equality is
proved for all five registered seeds before any optimizer is constructed.

### 3. The alignment falsifier is length-matched and replayed

v1's `roll_batch` shifted control tensors by one row inside heterogeneous padded
batches, so a valid position could receive a donor's **padded-position** control;
and later heads received controls that were already products of the intervention
at earlier heads.

The registered v2 falsifier is `replay_derangement`:

- donors come from a **fixed derangement within (registered stratum, exact valid
  sequence length) groups**, so donor and receiver always share a valid length;
- each batch runs an **unperturbed capture pass first**, and the intervened pass
  substitutes the donor's *captured aligned* control, so head `j` never receives
  a signal already perturbed at heads `< j`;
- three deterministic derangements (`derangement_indices: [0, 1, 2]`) are
  registered so the contrast can be shown robust to the permutation choice.

An example in a singleton group has no valid donor. It keeps its own control and
is recorded as `intervention_applied: false`. Such rows contribute a zero paired
difference, which biases the contrast **toward** zero and therefore cannot
manufacture a positive result. Coverage is a registered gate (minimum 0.95).

`roll_batch`, `zero`, and `negate` are retained as **descriptive diagnostics
only** and do not enter the advancement decision.

### 4. The bootstrap treats the design as crossed

Every training seed is evaluated on the same materialized test split, so seed and
example are **crossed**, not nested. v1 resampled seeds and then independently
resampled examples inside each selected seed, which destroys the shared-example
structure and can understate uncertainty when example-level effects are
correlated across seeds.

The registered v2 estimator is `crossed_paired`: each draw resamples seed IDs
and, independently, example IDs within each stratum, then evaluates **every
selected seed on the same resampled examples**.

`seed_cluster` — resample training seeds only, treating the generated test set as
fixed — is registered as the named alternative. Selecting it narrows the claim to
"across training seeds, for this materialized test split" and must be declared
before outcomes are visible.

### 5. Confirmatory test outcomes are not exposed progressively

v1 evaluated and wrote test predictions immediately after each cell trained, so
test outcomes from early cells were visible while later cells were still
training.

v2 splits execution into two phases:

1. **train** — every registered cell trains to exactly 2,500 updates and writes a
   hashed final checkpoint, a train log, telemetry, and **development**
   predictions. No test prediction is produced. Run status is `trained`.
2. **evaluate** — only after every registered cell is `trained`, hash-verified,
   step-count-verified, and confirmed to share one source head does a single
   frozen sweep evaluate the test split across all checkpoints.

Development evaluation stays available for operational debugging throughout.

### 6. Task difficulty is controlled and frozen

Pointer mappings are random permutations, so fixed points and short cycles let a
nominal multi-hop item resolve to the start node or be reachable in fewer
transitions than its hop label claims. Modular step sequences can cancel. On the
v1 registered dataset this affected **30.2% of pointer-chase** and **27.8% of
state-update** test items, and a naive "copy the queried start value" baseline
scored **0.150** on pointer chase.

v2 registers `reject_trivial: true` for the two **primary** families. Rejection
happens at generation time, before any model exists, and is recorded in the
dataset manifest. On the v2 registered dataset both primary families report a
trivial rate of **0.000** and a copy-start baseline of **0.000**.

The guardrail family `direct_lookup` is deliberately **not** filtered: it is a
one-hop lookup by construction, and filtering its natural fixed-point rate
(0.037) would make the non-inferiority guardrail unrepresentative.

### 7. Held-out ranges are described accurately

Registered test ranges partially overlap training ranges:

| Dimension | Train | Test | Overlap |
|---|---|---|---|
| `pointer_chase` hops | 2–4 | 4–8 | {4} |
| `state_update` steps | 3–7 | 6–12 | {6, 7} |
| `direct_lookup` nodes | 8–18 | 14–24 | {14…18} |

Every claim therefore says **held-out generated-task performance on
instance-held-out examples**, never uniform out-of-range extrapolation. The
frozen difficulty report records this and sets
`uniform_out_of_range_extrapolation: false`.

## Fixed design

- Five paired seeds: 101, 202, 303, 404, 505.
- Three required arms: cascade, history control, learned delta.
- The exact same materialized examples, minibatch order, update count, optimizer,
  model width/depth, and canonical initial shared tensors within each seed.
- History and learned-delta state dictionaries and parameter counts are exactly
  identical at initialization; every arm is a byte-exact restriction of the
  seed's canonical supernet on its shared keys.
- Dropout is zero.
- Training stops after exactly 2,500 updates. No dev-based checkpoint selection
  or early stopping.
- The fixed final checkpoint is hashed at the end of training and re-verified
  before the evaluation phase reads it.
- Test predictions are retained at the instance level.

## Data

Generated deterministically from seed 731928 and materialized before training.
Cross-split latent signatures are disallowed. Registered counts are 5,000 train,
900 dev, and 1,800 test examples, balanced within dev/test across direct lookup,
pointer chasing, and state update. Primary families reject trivial items as
described above.

The split is controlled and generated, not independently authored or sealed. That
limitation is part of every claim.

## Primary estimand

For each paired seed and each registered non-guardrail stratum:

1. compute the paired per-example correctness difference;
2. average within stratum;
3. macro-average strata within seed;
4. average the five paired seed effects.

The confidence interval is the **crossed paired bootstrap** described above:
10,000 draws, 95% confidence, seed 91273. Answer-NLL improvement is reported as a
supportive endpoint, not a replacement for accuracy.

## Power

An outcome-blind power simulation of the registered decision rule is frozen
before launch (`scripts/phase1_power_simulation.py`). It reports the empirical
false-positive rate under a null scenario and the detection probability at the
registered threshold under stated seed- and example-variance assumptions.

The simulation is run and published **before** the matrix is dispatched. Its
result does not license moving a threshold afterwards: a low detection
probability is a design finding to be reported with the result, not a reason to
re-register a gate.

## Fixed gates

- delta minus history: observed accuracy delta at least +0.02 and lower 95% bound above 0;
- delta minus cascade: observed delta at least +0.02 and lower bound above 0;
- aligned minus replay-deranged delta: observed delta at least +0.015 and lower bound above 0;
- direct lookup delta minus history: lower bound at least −0.01;
- alignment falsifier coverage: minimum per-seed coverage at least 0.95;
- closed evidence chain: every check in `scripts/verify_phase1_evidence.py` passes.

The integrity gate is **derived** from the verifier's result. It is not a
constant.

No threshold may be changed after inspecting confirmatory test outcomes under the
same study ID.

## Decision rule

`decision` is forced to `smoke_only` unless **all** of the following hold:

- the configuration declares `confirmatory: true`;
- the analyzed matrix is the complete registered arm/seed matrix;
- the matrix's study ID equals the registered study ID;
- the matrix's config hash equals the hash of the configuration being analyzed.

A non-confirmatory configuration can never emit an evidentiary decision, however
complete its own matrix is.

## Exclusions and failures

No post-hoc example exclusions. Difficulty rejection happens at generation time,
before any model exists, and is recorded in the dataset manifest.

A corrupted, missing, duplicated, or hash-mismatched artifact invalidates the
affected run. Failed attempts are written to immutable, never-overwritten records
under `runs/<arm>/seed_<n>/failures/`. A replacement run requires a new registered
seed or a documented rerun with identical code and configuration and preserved
failed artifacts.

`--force-data` is rejected outright for a confirmatory study. `--overwrite` and
`--force-data` cannot be combined. Confirmatory execution from a dirty checkout
is rejected.

A partial matrix may be analyzed only with `--allow-partial`; its decision remains
`smoke_only` and cannot support advancement.

## Claim ceiling

Even if every gate passes, the strongest supported statement is:

> In a controlled tiny decoder trained on the registered synthetic iterative
> tasks, an aligned finite-difference representation improved held-out
> generated-task capability beyond an information-matched history representation
> and beyond the sequential cascade alone.

The study does not establish natural-language generalization, long-context
performance, Qwen/GQA compatibility, wall-clock efficiency, memory efficiency,
scaling behavior, novelty, or an arXiv-primary claim.

## Change control

Any change to task generation, counts, arms, seeds, architecture, optimizer,
training steps, interventions, estimand, bootstrap, or thresholds after
confirmatory outcomes are visible requires a new study ID, a new output
directory, and an amended preregistration. The original artifacts remain
immutable.
