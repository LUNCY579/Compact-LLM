# INTERFERENCE-MECH-001, step 1: registered predictions (exploratory, exposed bank)

Written 2026-09-24. **No step-1 arm had been built or scored when this was committed.**
- Bank: T_BANK_V3X (864 rows). CONFIRM-001 spent its confirmatory read, so it is now exposed and
  every result here is exploratory.
- Reference arms are CONFIRM-001's recorded predictions: SPARSE 447, ADD-100 413 primary.
- Builder: `build_arms.py`. Its positive controls reproduce ADD-100, SPARSE and AVG within 1e-6.
  Smoke-tested on a 4-tensor subset: exact controls, mask keep rates 49.96% and 10.01%, exact
  rescaling, disjoint supports.
- Scorer: `score_arms.py` in the development environment (torch 2.6.0+cu124, transformers
  5.14.1), `max_new_tokens` 48. Analysis: `analyse_step1.py`.
- Bands (CONFIRM-001's n = 864 bands): SAME is |diff| ≤ 18. HIGHER/LOWER is |diff| ≥ 30 with exact
  McNemar p < 0.05. Anything else is inconclusive.

## Why these arms

CONFIRM-001 established two things:
- C6: a random Bernoulli mask of a task vector acts like the vector scaled by its keep rate.
- C3: a real interference cost, which step 0 shows falls on SPARSE's low-margin rows.

Two families of merging tricks claim to reduce interference. **DARE** (drop and rescale) and
**disjoint-support merging** (the overlap argument behind TIES/DARE) both assume that
coordinate-level overlap or redundancy is what hurts. C6 predicts they do nothing here, because a
mask followed by a rescale has the same expectation as the full delta, and the unstructured
remainder did nothing at norm 0.34. Step 0 instead suggests a margin remedy.

## Predictions

| id | arm | construction | prediction | why |
|---|---|---|---|---|
| P1 | DARE-O50 | dS + M₀.₅(dO)/0.5 | **SAME as ADD-100** (\|diff\| ≤ 18) | C6: mask ≈ expectation. Noise norm = ‖dO‖ = 1.36 |
| P2 | DARE-O10 | dS + M₀.₁(dO)/0.1 | **SAME as ADD-100** | C6 again, but the noise norm is 3‖dO‖ = 4.07. **This is the risky one:** C6 was measured with noise 0.34. A LOWER result would bound C6's range |
| P3 | DISJOINT-DARE | M(dS)/0.5 + (1−M)(dO)/0.5 | **SAME as ADD-100** | Removing coordinate overlap does not remove interference, which is functional, not coordinate-level |
| P4 | SPARSE-125 | 1.25·dS | **no directional prediction** | Dose beyond the trained strength is unmeasured for this skill. Reported two-sided |
| P5 | ADD-A125 vs SPARSE-125 | 1.25·dS + dO | **interference shrinks**: I = (ADD-A125 − SPARSE-125) − (ADD-100 − SPARSE) ≥ +10, with the bootstrap 95% CI excluding 0 | Step 0: the cost falls on low-margin rows, and raising the skill coefficient raises margins |
| P6 | ADD-A125 vs SPARSE | — | **SAME or HIGHER** | The practical claim: a boosted composite loses nothing against the single donor |

**Reading.**
- If P1–P3 all come out SAME, DARE and disjoint merging buy nothing on this pair, and the mechanism
  is C6's.
- If P2 alone comes out LOWER, C6 holds for small noise and breaks for large noise.
- If P5 is met, the knife-edge account yields a practical remedy. If it fails, the step-0
  correlation is not causal. P4 must be read first: if SPARSE-125 is already much lower than
  SPARSE, P5 compares two regimes, and that has to be stated.

**Limits.** An exposed bank, one donor pair, one skill. Confirming any of this needs a fresh bank.
Cost: 5 arms × 864 rows at about 9 min each, plus about 5 × 3 min of builds, roughly 1 h under
the GPU lease. It runs only after DOSELORA/SMALLVD sends "ended".
