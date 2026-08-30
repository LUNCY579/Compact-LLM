# Phase-1 v4 development pilot — competence report (Stages 1–5)

Development-only. Nothing here reads a confirmatory bank, and **no between-arm
difference was used to select anything**. Configurations are scored by the
minimum absolute held-out accuracy across all three arms.

Grid recorded before execution. Primary axis was training-set size, because the
v3 postmortem showed pure memorization; capacity was deliberately *not* the
first lever, since more capacity on 5,000 examples memorizes better rather than
generalizing.

**Current state: `V4_COMPETENCE_NOT_YET_ESTABLISHED`.** A regime that clears
every accuracy-based check exists, but it fails a reliability check added as a
result of this pilot. See §2 and §4.

---

## 1. OBSERVED

Every regime trains `cascade`, `history_control`, and `learned_delta` under
identical settings on 2 development seeds, and evaluates on a held-out set drawn
from the **same band** with a different dataset seed (zero latent-signature
overlap), so difficulty is fixed and only instance novelty varies.

| regime | train ex. | steps | band | trivial | min-arm holdout | margin | mem gap | max NLL |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `v3_replica` | 5k | 2.5k | hard | 0.0413 | 0.0463 | +0.0050 | **0.1054** | **3.662** |
| `data_50k` | 50k | 2.5k | hard | 0.0413 | 0.0327 | −0.0087 | 0.0084 | 3.463 |
| `data_200k` | 200k | 2.5k | hard | 0.0407 | 0.0323 | −0.0083 | 0.0056 | 3.461 |
| `data_200k_long` | 200k | 10k | hard | 0.0407 | 0.0477 | +0.0070 | 0.0013 | 3.416 |
| `easy_5k` | 5k | 2.5k | easy | 0.0400 | 0.0940 | +0.0540 | **0.1747** | 3.323 |
| `easy_50k` | 50k | 2.5k | easy | 0.0400 | 0.0937 | +0.0537 | 0.0124 | 3.182 |
| `easy_50k_long` | 50k | 10k | easy | 0.0400 | **0.2633** | **+0.2233** | 0.0683 | **2.063** |

Uniform-answer NLL reference: ln(32) = **3.466**.

`OBSERVED` **`v3_replica` reproduces the v3 pathology** — margin +0.005 with a
memorization gap of 0.105 and NLL *worse* than uniform. The harness reproduces
the failure it is meant to diagnose, which is what makes the other rows
trustworthy.

`OBSERVED` **More data alone removes memorization but does not create
competence.** `data_50k` and `data_200k` collapse the memorization gap from
0.105 to 0.008/0.006 — and land *below* the trivial baseline, at exactly uniform
NLL. The model stops memorizing and learns nothing instead.

`OBSERVED` **More data plus more updates barely clears trivial on the hard
band.** `data_200k_long` reaches +0.0070 over trivial with NLL 3.416. Real but
negligible: 4.8% absolute on a 1-of-32 task.

`OBSERVED` **The easy band is genuinely learnable.** `easy_5k` reaches 9.4%
against a 4.0% baseline (+5.4 points, 2.3×) — but with a 0.175 memorization gap.

`OBSERVED` **`easy_50k` is the first regime with real generalization.** Same
accuracy as `easy_5k` (9.4%) but the memorization gap collapses to 0.012 and NLL
improves to 3.182, comfortably below uniform.

`OBSERVED` **`easy_50k_long` reaches 6.6× the trivial baseline on the mean** —
min-arm 0.2633 against 0.0400, NLL 2.063, memorization gap 0.068. On every check
the gate had at the time, this was a `GO`.

---

## 2. THE FINDING THAT CHANGED THE PLAN

`OBSERVED` **`easy_50k_long`'s per-seed results are bimodal.**

| arm | seed 1 | seed 2 | mean |
|---|---:|---:|---:|
| `cascade` | 0.1167 | 0.4100 | 0.2633 |
| `history_control` | 0.3993 | 0.1280 | 0.2637 |
| `learned_delta` | 0.3167 | 0.4307 | 0.3737 |

`OBSERVED` The loss curves say why. A phase transition begins around step
5000–7000. Four of six runs complete it, ending at train loss 1.33–1.63. Two do
not — `cascade` seed 1 ends at 2.278 and `history_control` seed 2 at 2.155, and
**both are still descending at the final step**. They did not fail; the
10,000-step cosine schedule annealed their learning rate to zero partway through
the transition.

`SUPPORTED` **The mean describes no model that exists.** It averages a converged
model and a truncated one. Realised between-seed SD is ≈ 0.20 — an order of
magnitude above the σ_a = 0.02 central / 0.04 robustness values the confirmatory
design is powered for. At that dispersion no realistic seed count detects the
+0.020 target effect, so a confirmatory launch on this regime would be
uninterpretable for a *new* reason rather than v3's old one.

`SUPPORTED` **This fully explains the apparent `learned_delta` margin.**
`learned_delta` drew two converged seeds; each control drew one converged and one
truncated. 0.3737 vs 0.2633/0.2637 is a coin flip over which seeds finished
training, not a mechanism effect. It was never used for selection — the rule is
min-arm — and it must not be read as evidence in either direction.

---

## 3. UNSUPPORTED CLAIMS

* That `learned_delta`'s higher `easy_50k_long` number means anything. §2 gives
  the arithmetic that accounts for it without reference to the mechanism.
* That `easy_50k_long` competence is reproducible. Two of six runs did not reach
  the state the mean reports.
* That the 20,000-step regime now under test will converge for all seeds. It is
  the indicated next step because the truncated runs were still descending, but
  it is untested.
* Anything about the TGSA mechanism.

---

## 4. CONSEQUENCE: AN EIGHTH GATE CHECK

`competence_is_reliable_across_seeds` was added to `tgsa/phase1_competence.py`:

* the **worst individual run** — not the arm mean — must clear the absolute floor;
* the within-arm **coefficient of variation** must be ≤ 0.35;
* fewer than `minimum_seed_runs` (3) recorded runs, or none, fails closed.

The check draws a line the first version of it blurred. **Dispersion among
runs that are all competent is a power problem** whose remedy is more seeds;
**some runs failing to become competent at all is a competence problem** that
no seed count fixes, because each arm becomes a different mixture of learners
and non-learners. The gate owns the second and exports the realised SD under
`detail.realised_seed_sd_for_power_audit` for the first. Bounding the raw SD at
the power analysis's own σ_a conflated them and would have rejected a usable
regime — including `easy_50k_long`'s `learned_delta`, whose two runs were both
post-transition and 0.11 apart.

The schema version is at `v3` and pinned on load, so a report produced under
an earlier version of the gate cannot satisfy `require_go`.
`scripts/regate_competence_report.py` recomputes a verdict from a stored
report's **raw runs** so a report need not be re-measured when the gate moves.
It cannot manufacture a `GO`: a stored `GO` whose raw runs fail the current
thresholds re-gates to `NO_GO`, and the input is never modified.

`tests/test_phase1_competence.py::test_a_bimodal_regime_is_refused_despite_a_healthy_mean`
pins the exact `easy_50k_long` numbers and asserts that the other seven checks
pass while the verdict is `NO_GO` — the gate must fail for the right reason.

Note that the 0.1167 run still clears the 0.07 absolute floor, so the mixture
bound is the load-bearing half. A floor alone would not have caught this.

---

## 5. ANOMALIES

`OBSERVED` `data_50k` and `data_200k` land **below** the trivial baseline
(−0.009, −0.008) with NLL at exactly uniform. The models converged to
approximately uniform prediction over the answer space — a "learned to guess"
state, which is a coherent optimum when the task is out of reach and
memorization is unavailable.

`OBSERVED` **The pilot's training-set seed was not reproducible.** It was
`1_000_000 + hash(band) % 1000`, and Python salts `hash()` of str per process.
Since each regime ran in its own process, regimes were compared across
*different* training draws, and `easy_50k_long`'s exact dataset cannot be
regenerated from its own record. Within-regime measurements are unaffected —
train and holdout stayed consistent and signature-disjoint within a process — but
cross-regime comparison carried an uncontrolled nuisance factor. Fixed in
`d3028d5`; `tests/test_seed_determinism.py` walks the AST of every module under
`tgsa/` and `scripts/` and fails on any builtin `hash()` call.

`OBSERVED` The same defect existed in `tgsa/phase1_power.py`. The **v3 launch
decision is not affected**: it was made by `scripts/phase1_power_simulation.py`,
which seeds deterministically.

`OBSERVED` Two CUDA OOM failures in a single-process pilot loop as allocator
fragmentation accumulated across regimes. Infrastructure, not science; the fix
is a fresh process per regime.

---

## 6. LIMITATIONS

* 2 development seeds per arm per regime — which is precisely how a bimodal
  regime came to look like a competent one. The competence run uses 3 and the
  gate now requires it.
* 7 of 10 planned regimes completed. `easy_200k_long` is in flight.
  `easy_50k_wide` and `medium_50k_long` were **dropped**: both were scheduled at
  10,000 steps, which §2 shows is inside the truncation window, so neither would
  have answered the question that now matters.
* Only three difficulty bands were defined, and the gap between `easy_nontrivial`
  (2 hops, 5–8 nodes) and `hard` (2–4 hops, 10–20 nodes) is large.

---

## 7. COMPETENCE GATE THRESHOLDS (frozen)

Derived from this pilot, outcome-blind, before any v4 confirmatory run:

| check | threshold | rationale |
|---|---|---|
| margin over strongest trivial baseline | ≥ **+0.030** | `easy_50k` achieves +0.054; `v3_replica` scored +0.005. Sits above the failure and below the demonstrated pass. |
| absolute held-out accuracy, weakest arm | ≥ **0.070** | `easy_50k` weakest arm is 0.0937; v3 was 0.045. |
| answer NLL | < **0.97 × ln(32)** = 3.362 | `easy_50k` reaches 3.182; v3 was 4.03 and `data_200k` 3.461. Excludes the "learned to guess" state. |
| memorization gap (seen − holdout) | ≤ **0.10** | v3 was 0.105, `easy_5k` 0.175, `easy_50k` 0.012. |
| distinct predictions | ≥ 4 | rejects a constant predictor. |
| per-family accuracy | above the same floor, and ≤ 0.95 | v3's `state_update` was below chance while `pointer_chase` was near baseline. |
| **between-seed coefficient of variation, per arm** | ≤ **0.35** | a balanced two-point mixture of a competent and an at-floor population has CV ≈ 0.6 by construction; ordinary dispersion among converged runs sits at 0.07–0.28. `easy_50k_long`'s two bimodal arms realised 0.73 and 0.79. |
| **worst individual run** | ≥ the absolute floor | the arm mean is not the unit of competence. |
| **recorded development runs per arm** | ≥ 3 | two seeds cannot establish that a third behaves the same way. |

Every threshold is placed strictly between a demonstrated failure and a
demonstrated pass, so the gate is calibrated against observed behaviour rather
than chosen to be permissive.

---

## 8. CLAIM CEILING

> A development pilot establishes that the v3 regime's failure is attributable to
> band difficulty combined with training-set size, not to model capacity or
> optimization, and that all three registered arms reach several times the
> strongest trivial baseline on an easier band given a sufficient update budget.
> It does **not** yet establish that this competence is reproducible across
> seeds, because the one regime that reached it did so, for two of three arms, by
> averaging over a training phase transition that some seeds did not complete.

Nothing here bears on the TGSA mechanism.

---

## 9. NEXT PERMITTED EXPERIMENT

1. Re-run `easy_nontrivial` / 50k examples at **20,000 steps with 3 seeds**,
   recording per-seed held-out accuracy, per-family accuracy, and distinct
   prediction counts — the full gate input set. The truncated runs were still
   descending, so the budget is the indicated lever.
2. Run the frozen competence gate. Dispatch v4 **only** on `GO`.
3. On `GO`, re-run the outcome-blind power simulation with the *realised* σ_a
   from step 1 before confirming the seed count. The v3 count of 20 was powered
   for σ_a = 0.02; it is retained only if the realised value supports it.
4. On `NO_GO`, follow the registered escalation ladder in
   `PHASE1_V4_PREREGISTRATION.md` §Escalation — at most four competence
   measurements, each rung at most once, each tied to the specific check that
   failed, and rejection of the regime family on exhaustion. The step budget is
   not to be tuned until the gate happens to pass.
