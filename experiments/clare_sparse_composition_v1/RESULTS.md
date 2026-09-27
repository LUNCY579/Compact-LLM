# CLARE-RINEYA-001 — results so far

**No skill has been trained. No arm exists. No research result is claimed.** Everything below
is qualification: evidence that the machinery does what the protocol says, on synthetic models
(Phase 2) and on the real 596M model (Phase 3). Labels follow the result schema: **MEASURED** is
read from an artifact, **DERIVED** is computed from measured values, and **HYPOTHESIZED** is
interpretation.

## Phase 1 — implementation: complete

The `clare/` package, CLI, zero-argument chain, registered configuration and documentation are
all in this directory.

## Canonical branch and the ported CPU tests

`research/titan-clare-sparse-composable-skills-v1-claude` is canonical. The parallel CPU-only
implementation (`e320ba33`) is merged with strategy "ours", and its 87 CPU test cases are ported
(`tests/test_ported_e320ba33.py`, 99 cases). **MEASURED:** 154 offline tests pass. Run against the
pre-port commit, the ported cases fail 30 times; 28 of those are real gaps, all fixed, and 2 are
control-environment artifacts (see `CLARE_RINEYA_AUDIT.md` §6b).

## Phase 2 — synthetic qualification: PASS

`qualification/SYNTHETIC_QUALIFICATION.json` (tiny CPU models only):

| check (MEASURED) | value |
|---|---|
| offline test suite | all passing (count in the report) |
| D and E mask budgets == schedule (ρ 0.9, 4 skills) | [51, 46, 42, 37] == [51, 46, 42, 37] |
| max pairwise support intersection, D and E | 0 |
| max \|parameter dot product\|, D and E | 0.0 |
| frozen deltas unchanged by every later stage | True (both arms) |
| discovery reset restores the pre-probe effective weights | True for all 4 skills |
| selected coordinates with zero importance | 0 per skill |
| D and E select different supports at identical capacity | True |
| base weights unchanged | True |

Zero intersection and zero dot product are **mechanical** consequences of strict allocation, not
findings. They are listed only because they are the checks that would catch a broken allocator.

**Finding from testing (MEASURED): dense-masked and sparse training are not bit-identical.**
They are the same algorithm, with identical losses and gradients after the first update. But the
grad-norm reduction and the vectorised elementwise kernels round differently for the two tensor
layouts, so the deltas agree only to float32 rounding. The code's first docstring claimed bit
identity; that claim was corrected. Consequence for the design: no arm mixes the two paths, since D
and E both retrain through the sparse path.

## Phase 3 — native qualification on Rineya-Supa-1.7.0: QUALIFIED

`qualification/native-20260927-attempt2/NATIVE_QUALIFICATION.json`. Attempt 1
(`qualification/native-20260927/`) was **BLOCKED at preflight on host RAM**: 3.55 GiB free
against the 3.6 GiB floor. The floor was not changed; attempt 2 ran when RAM allowed.

**Re-qualified on the committed code (current reference):**
`qualification/native-20260927-committed-6759d95b/NATIVE_QUALIFICATION.json`: QUALIFIED, 11/11
gates, git `6759d95b` with a clean experiment tree, code digest `a0306194` (= the committed
`clare/` package), the same `config_sha256`. **MEASURED: its numerics are bit-identical to
attempt 2**: the same dense losses (1.35407…, 0.64093…), the same non-zero count after the probe
(169,812,686) and the same selection report, the same sparse-delta norm (0.0447604578…), the same
parity (1.9e-5), the same memory peaks, and the same 14 base-competence responses. Only wall-clock
timings differ (dense 1.80 s/update, sparse 1.84 s, generation 1.18 s/row, selection 37 s). So the
port-motivated fixes changed no training arithmetic, and the GPU path is run-to-run deterministic
on these checks. Attempt 2 had run on the uncommitted tree before those fixes and did not record a
code digest; it is kept as history.

| gate (MEASURED) | result |
|---|---|
| unique parameters = 596,049,920; footprint = 60 modules / 188,743,680 coordinates | pass |
| no base parameter trainable; base checksum unchanged after all steps | pass |
| Titan training contract (answer + EOS labels, no truncation) on TRAIN R rows | pass |
| **parity: patched BF16-storage/FP32-compute vs plain FP32 load** | max \|Δlogit\| **1.9e-5** (gate 1e-4), argmax equal, chunked and cached head |
| dense AdamW updates finite | pass |
| discovery probe respects the free mask | 0 coordinates moved outside it |
| discovery reset exact | effective-weight checksum restored |
| λ gate reading (2-update probe, λ 0.1) | 169,812,686 non-zero free coordinates = 10.0× the skill-2 budget |
| sparse step: active non-zeros ≤ allocated | 16,986,931 = 16,986,931 |

| measurement (MEASURED) | value |
|---|---|
| load to GPU (device_map) | 13.1 s |
| dense update (≤ 2048 tokens) | 1.69 s, 1,050 tokens/s |
| sparse update | 1.70 s |
| exact-budget selection over 169.9M free coordinates | 50 s |
| greedy generation, uncached weights | 1.62 s/row |
| VRAM allocated: after load / dense peak / discovery peak / sparse peak | 1.11 / 4.46 / 4.64 / 4.20 GiB (cap 6.24) |
| dense AdamW state | 1.41 GiB |
| TRAIN R row length (16 rows) | 538–1,265 tokens, mean 893 |

**Base competence, CALIBRATION smoke (6 rows × 7 train signatures): 0/42 strict.** Supa answers
with a bare list (`[0, 1, 3, 4, 5, 6, 7, 8, 9, 11]`) instead of the required `{"answer": [...]}`
object, so the strict verifier scores 0. *DERIVED consequence:* acquisition in every arm will
include learning the answer FORMAT, and in the sequential arms the first skill learns the shared
format for the others. This is why the protocol reports seen pairs, floors and retention separately,
and does not read a composition number in isolation.

*PROJECTED (DERIVED from the measured rates, not yet observed):* one dense skill stage on 1,000 R
rows ≈ 14 min; one probe ≈ 3.5 min plus 1 min selection; scoring 960 rows ≈ 22 min with the
evaluation cache (below), ≈ 26 min without.

**Evaluation-only FP32 weight cache — native check PASS** (`qualification/EVAL_CACHE_CHECK_2.json`,
reference = the committed-code qualification; CALIBRATION rows only). MEASURED: with the caches ON
the 14 recorded base-competence responses are reproduced byte-for-byte (14/14 canary), and cached and
uncached generation give identical outputs. Speed 1.786 → 1.355 s/row (**1.32×**). The uncached pass
ran first and so also carried any warm-up, so the speed-up may be slightly overstated; the
correctness result does not depend on order. The first attempt (`qualification/EVAL_CACHE_CHECK.json`)
was BLOCKED at preflight on disk and is kept as history.

## Phase 4 — not started

Status: **ready to launch, resource-gated.** The chain (`run_phase4.py`) is qualified to start; no
training step, arm or evaluation has run. Open items:

1. **disk:** C: has 5.42 GiB free (it fell to 0.84 GiB during the session from writes outside this
   study, then partly recovered). Arm A's training preflight needs ≈ 6.8 GiB (2.81 GiB of dense deltas
   plus the 4 GiB floor), so tier 1 refuses at its first training step; all of tier 1 needs ≈ 7.5 GiB
   and tiers 0–4 ≈ 14 GiB. The LR gate (tier 0) writes almost nothing and fits now. Freeing space is
   an operator decision;
2. the n = 47 general-retention panel needs an operator decision (it writes into another session's
   worktree).

## Recommended next single experiment

Launch the chain (`NATIVE_SMOKE_AND_HANDOFF.md` §4) for **tier 0 + tier 1 only**: the registered
LR build gate, then seed 0 / order RUSP for arms **A, D, E**. That is the smallest run that answers
the crucial question: learned vs random disjoint allocation at identical capacity, against the
dense sequential baseline, with retention, held-out composition and the 2×2 functional interaction
measured together.
