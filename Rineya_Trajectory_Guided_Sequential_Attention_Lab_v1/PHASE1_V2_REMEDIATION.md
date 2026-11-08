# TGSA Phase 1 v2 remediation ledger

**Remediation branch:** `research/tgsa-phase1-identifiability-v2-launch-ready`
**Audit answered:** `PHASE1_LAUNCH_AUDIT.md` (verdict: NO-GO on `8ffa52f`)
**New study ID:** `tgsa-phase1-identifiability-v2-2026-08-29`
**Amended registration:** `PHASE1_V2_PREREGISTRATION.md`

The v1 study `tgsa-phase1-model-identifiability-2026-08-28` and its no-result
state are preserved unchanged, as the audit's remediation sequence requires.

## Status

Every P0 and P1 finding is closed in code with a regression test. **The launch
recommendation is still NO-GO**, for a new reason: the outcome-blind power
simulation the audit itself demanded shows the registered design is
underpowered at its own threshold. See "Findings the remediation surfaced".

Test suite: **93 passed** (was 46 passed / 4 failed).

## Audit findings

| ID | Finding | Status | Where |
|---|---|---|---|
| P0-1 | CI installs a nonexistent `[dev]` extra; pytest never installed | Closed | `pyproject.toml`, `.github/workflows/tgsa-phase1.yml` |
| P0-2 | Cascade/delta shared initialization not implemented | Closed | `tgsa/phase1_model.py` |
| P0-3 | Stale completed runs reused without validation | Closed | `tgsa/phase1_runner.py`, `scripts/run_phase1_model_study.py` |
| P0-4 | Smoke results not forced non-evidentiary | Closed | `tgsa/phase1_runner.py`, `tgsa/phase1_analysis.py` |
| P0-5 | Analyzer establishes no closed evidence chain | Closed | `tgsa/phase1_verify.py`, `scripts/verify_phase1_evidence.py` |
| P0-6 | "Information-matched" control not realized | Closed | `tgsa/module.py` |
| P0-7 | Bootstrap treats a crossed design as nested | Closed | `tgsa/phase1_analysis.py`, `tgsa/phase1_power.py` |
| P0-8 | Confirmatory test outcomes exposed progressively | Closed | `tgsa/phase1_runner.py` |
| P1-1 | Resume can duplicate training-log records | Closed | `tgsa/phase1_runner.py` |
| P1-2 | Failure evidence deleted or overwritten on retry | Closed | `tgsa/phase1_runner.py` |
| P1-3 | Full workflow destroys resumable evidence at startup | Closed | `.github/workflows/tgsa-phase1.yml` |
| P1-4 | Batch-roll falsifier has avoidable confounds | Closed | `tgsa/phase1_model.py`, `tgsa/phase1_runner.py` |
| P1-5 | Iterative difficulty not guaranteed by the hop/step label | Closed | `tgsa/phase1_data.py`, `tgsa/phase1_difficulty.py` |
| P1-6 | Held-out ranges overlap training ranges | Closed (documented) | `tgsa/phase1_difficulty.py`, registration |

### P0-1 — packaging

`pyproject.toml` declared only a `test` extra; the workflow installed `[dev]` in
both jobs. pip treats an unknown extra as a *warning*, so the install step
exited 0 and the job died one step later with `pytest: command not found`.

Fixed: `dev` and `test` are both declared and both provide `pytest` and
`jsonschema`. The workflow now fails closed with an explicit import probe after
install, calls `python -m pytest` rather than a bare `pytest`, and installs
CPU-only torch wheels so a CPU smoke job stops pulling a multi-gigabyte CUDA
stack. `test_p0_1_every_extra_the_workflow_installs_is_declared` parses the
workflow and asserts every requested extra exists, so the two files cannot drift
again.

### P0-2 — canonical shared initialization

`build_model` reset the global seed and then constructed structurally different
graphs. The learned trajectory factors consumed random numbers the cascade arm
did not, so shared tensors diverged. `paired_initialization_audit` asserted
equality and would have raised before training.

Fixed: `build_canonical_supernet` builds one canonical `learned_delta` model per
seed; `copy_shared_tensors` copies every exactly-compatible tensor by key, dtype,
and shape; arm-only tensors are initialized under a derived, recorded seed. The
preflight audit proves equality for all five registered seeds and records a
per-arm fingerprint.

A second, unreported defect was found and fixed here: `TinyTGSAForCausalLM`
ran `apply(self._initialize_weights)` over every `nn.Linear`, silently
overwriting the trajectory-encoder initialization with `N(0, 0.02)`. The
documented encoder initialization had never taken effect.
`reset_registered_parameters` restores it after the global pass, and
`test_p0_2_global_apply_does_not_clobber_the_trajectory_encoder_init` locks it.

### P0-3 — stale run reuse

`train_one_run` returned any existing complete manifest without comparing
anything. The test suite already expected `_validate_completed_run_reuse`; the
symbol did not exist, so two tests failed.

Fixed: `validate_completed_run_reuse` compares study config SHA-256, dataset
manifest SHA-256, and source head for every finished cell, and is called by
`run_study_matrix` before any cell is dispatched. Incomplete runs are left alone
because they are resumable, not stale. `--force-data` is rejected for a
confirmatory study, `--overwrite` cannot be combined with `--force-data`, and
confirmatory execution from a dirty checkout is refused.

### P0-4 — smoke results cannot be evidentiary

The analyzer never read `confirmatory`, never emitted `confirmatory_config`, and
computed `is_full` relative to the smoke configuration itself. A complete
one-seed smoke matrix could therefore be labelled
`supports_phase1_mechanism_hypothesis`. The emitted analysis also failed the
checked-in JSON schema, which requires `confirmatory_config`.

Fixed: `load_study_config` requires an explicit `confirmatory` boolean. The
analyzer emits `confirmatory_config` and `registered_confirmatory_matrix`, and
forces `decision = "smoke_only"` unless the config is confirmatory **and** the
matrix is the full registered matrix **and** the study ID matches **and** the
config hash matches. CI validates every artifact against its schema.

Verified end to end: the v2 smoke run reports `is_full_registered_matrix: true`
with `confirmatory_config: false` and `decision: smoke_only`.

### P0-5 — closed evidence chain

The analyzer checked that runs agreed *with each other*, never that they agreed
with the configuration, dataset, and checkout in front of the analyst. Recorded
checkpoint and train-log hashes were never re-verified, and the
`paired_initialization_and_artifact_integrity` gate was hard-coded `True`.

Fixed: `tgsa/phase1_verify.py` binds config → dataset manifest → split files →
preflight → matrix manifest → every run manifest → checkpoints, train logs, and
prediction files, with exact SHA-256 at each link. It also enforces one clean
source head, a compatible runtime, exactly the registered cell count, exactly the
registered update count, the registered evaluation-condition set, and per-row
prediction identity against the materialized test split. The integrity gate is
now derived from that report. `scripts/verify_phase1_evidence.py` runs the same
chain as a standalone post-run verifier, and CI runs it.

Negative tests in `tests/test_phase1_evidence_chain.py` mutate a prediction file,
a final checkpoint, a train log, the dataset manifest, the study config, and a
whole run directory, plus a hash-consistent but semantically mislabelled
prediction row — each is rejected.

### P0-6 — information-matched control

See `PHASE1_V2_PREREGISTRATION.md` §1. The encoder is now a residual around the
identity with a zero-initialized `up` factor, so the raw control tensor is always
additively present and the encoder is exactly the identity at step 0.
`test_p0_6_history_mean_and_velocity_are_affine_bijections_given_the_carrier`
proves the exact property the registration claims.

This changes the mechanism, which is why the study is re-registered.

### P0-7 — crossed, not nested

`hierarchical_paired_bootstrap` resampled seeds, then independently resampled
examples inside each selected seed, breaking the shared-example structure.

Fixed: `crossed_paired_bootstrap` applies one example resample per draw to every
selected seed. `paired_seed_stratum_values` now carries example IDs explicitly and
refuses input where the per-stratum example sets differ across seeds, so a
non-crossed design fails loudly instead of being silently mis-analysed.
`seed_cluster_bootstrap` is the registered alternative. Only registered schemes
resolve; the old nested estimator is gone.
`test_p0_7_shared_example_effects_widen_the_crossed_interval` demonstrates the
concrete failure mode.

### P0-8 — deferred confirmatory evaluation

Fixed: `train_one_run` produces development predictions only and leaves the run
in status `trained`. `run_study_matrix` trains every cell, then applies a barrier
that verifies completeness, step counts, and a single source head, and only then
runs `evaluate_one_run` across all fixed checkpoints. The evaluation phase
re-verifies each checkpoint hash and the restored model state hash before reading
the test split.

### P1-1 / P1-2 — resume and failure evidence

`_truncate_train_log` rewinds the log to the restored checkpoint boundary and
rejects duplicate or non-increasing step sequences.
`_record_failed_attempt` writes immutable, content-addressed, never-overwritten
attempt records under `runs/<arm>/seed_<n>/failures/`; the unconditional
`failure_bundle.json` unlink is gone.
`test_interrupted_and_uninterrupted_training_are_equivalent` crashes a run,
resumes it, and asserts identical final model state hash, examples seen, tokens
seen, and byte-identical predictions, with a unique increasing log.

### P1-3 — non-destructive evidence root

The manual full job began with `rm -rf full_evidence`. Fixed: the evidence root
is now `full_evidence/<study_id>/<head_sha>`, an existing root is **archived**
rather than deleted, and archiving is an explicit dispatch input.

### P1-4 — alignment falsifier

See `PHASE1_V2_PREREGISTRATION.md` §3. `length_matched_derangement` plus
`ControlCapture`/`ReplayIntervention` fix both the padding confound and the
compounding confound. `roll_batch` is retained as a diagnostic with an explicitly
narrowed interpretation and no role in the decision.

### P1-5 / P1-6 — task difficulty

Measured, then controlled. The generator records `trivial`, `effective_hops`,
`start_cycle_length`, `answer_equals_start`, `cancels_to_start`, and
`effective_steps`. `tgsa/phase1_difficulty.py` produces a frozen, outcome-blind
report with shortcut baselines and the held-out range overlap table.
`reject_trivial` is registered for the primary families only.

Measured on the registered dataset:

| Family | trivial (before) | trivial (after) | copy-start baseline (before → after) |
|---|---:|---:|---|
| `pointer_chase` | 0.302 | **0.000** | 0.150 → **0.000** |
| `state_update` | 0.278 | **0.000** | 0.037 → **0.000** |
| `direct_lookup` (guardrail, unfiltered) | 0.037 | 0.037 | 0.037 → 0.037 |

Mean effective hops on pointer chase rose from 4.88 to 5.95.

## Findings the remediation surfaced

These are new. They are not in the original audit.

### N-1 — the registered design is underpowered at its own threshold

The audit required an outcome-blind power simulation before launch. Running it
(`scripts/phase1_power_simulation.py`, 150 trials, crossed bootstrap, 5 seeds,
20 strata × 60 examples) gives:

| Scenario | True effect | Seed SD | Gate fire rate |
|---|---:|---:|---:|
| null | +0.0000 | 0.020 | 0.073 |
| at_threshold | +0.0200 | 0.020 | **0.380** |
| double_threshold | +0.0400 | 0.020 | 0.740 |
| at_threshold, high seed variance | +0.0200 | 0.060 | 0.227 |

The simulator is calibrated: mean observed effect tracks the true effect
(+0.0206 vs +0.0200; +0.0379 vs +0.0400).

Two consequences:

1. **Detection probability at the registered +0.02 threshold is about 38%**, and
   about 23% if seed variance is at the higher end of the registered range. A
   real mechanism effect exactly at the registered minimum would more often than
   not produce `inconclusive` after spending the entire compute budget.
2. **The empirical false-positive rate is about 7%**, not the 2.5% a naive
   reading of the 95% interval suggests. The conjunction gate
   (`observed >= threshold` **and** `lower bound > 0`) is effectively a
   one-sided test at roughly `observed > 0.026`.

The variance components are registered assumptions, not measurements — true seed
variance is unknown before launch. The honest reading is that the design's power
is *uncertain and plausibly low*, not that it is exactly 38%.

This is a design decision for the study owner, not something to patch silently.
Options, in the order I would consider them:

- **Add seeds.** Seed-level resolution is the binding constraint. Going from 5 to
  10–15 seeds is the only change that raises power without touching the estimand.
  It is also the most expensive.
- **Re-register a larger minimum effect.** Legitimate now, before any outcome
  exists; illegitimate afterwards.
- **Launch as registered and report the power alongside the result.** Defensible
  if the result is framed as "underpowered at the registered threshold" and
  `inconclusive` is treated as an expected outcome rather than a failure.
- **Drop to `seed_cluster`** and narrow the claim to this materialized test
  split. Cheaper, but a weaker claim.

I did not choose among these: it changes the registered design and the compute
budget, so it is the study owner's call. The registration currently stands at
five seeds and +0.02.

### N-2 — model-wide initializer clobbered the encoder initialization

Reported under P0-2 above. Pre-existing, unreported by the audit, fixed here.

## Pre-launch acceptance gates

| Gate | State |
|---|---|
| Clean checkout, verified ancestry from `aefb70e` | Enforced at runtime; requires the pinned launch commit |
| Complete test suite passes, including the reuse guard | **Pass** — 93 tests |
| End-to-end smoke matrix completes and schema-validates | **Pass** |
| Smoke analysis always `confirmatory_config=false`, `decision=smoke_only` | **Pass** |
| Canonical shared initialization for all five registered seeds | **Pass** |
| Stale config/data/source/checkpoint/prediction mutations rejected | **Pass** — 7 negative tests |
| Interrupted and uninterrupted training equivalent | **Pass** |
| Failed-attempt artifacts immutable across retry/resume | **Pass** |
| Statistical implementation matches the amended registration | **Pass** |
| Task-difficulty and shortcut-baseline report frozen | **Pass** |
| All checkpoints complete before confirmatory test evaluation | **Pass** — enforced by the phase barrier |
| One post-run verifier validates the whole chain | **Pass** — `scripts/verify_phase1_evidence.py` |
| Outcome-blind power audit | **Run — result is adverse; see N-1** |
| CI green on the exact pinned launch commit | **Not yet** — needs a commit and a CI run |

## Launch decision

**NO-GO**, pending two things:

1. a study-owner decision on N-1 (seeds, threshold, or accept-and-report);
2. a green `unit-and-smoke` CI run on the pinned, clean launch commit.

Every mechanical and evidentiary blocker from the original audit is closed.
