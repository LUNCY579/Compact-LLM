# TGSA Phase 1 full-matrix launch audit

**Audit status:** NO-GO  
**Audited source branch:** `research/tgsa-phase1-model-identifiability`  
**Audited source head:** `8ffa52fb6d91dfd91c0aee94859a5a606ad0f040`  
**Pinned Phase 0 parent:** `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e`  
**Audit child branch:** `research/tgsa-phase1-model-identifiability-launch-audit`

## Executive verdict

Do **not** launch the registered 3-arm × 5-seed confirmatory matrix from the audited head.

The core study intent is strong: the preregistration, full configuration, primary contrasts, fixed-step policy, task families, and claim ceiling are substantially aligned. The current implementation, however, cannot yet produce defensible claim-bearing evidence. Some defects prevent execution; others can silently admit stale or misclassified evidence; and two scientific contracts are stronger than the implemented controls justify.

Launching now risks spending the full compute budget on a matrix that either fails before training, cannot pass its own evidence schema, or supports only a weaker interpretation than the registered claim.

## What is already sound

- The pinned parent commit is an ancestor of the audited source head.
- The full configuration matches the preregistered arms, seeds, task counts, fixed 2,500 updates, dropout-free training, interventions, bootstrap draw count, and advancement thresholds.
- `history_control` and `learned_delta` have the same module structure, parameter count, and initialization when constructed from the same seed.
- Data are deterministically materialized, split-file hashed, instance-paired across arms, and checked for exact latent-signature overlap.
- The runner uses a fixed final checkpoint rather than development-set checkpoint selection.
- Per-instance predictions, checkpoint hashes, telemetry, training signatures, and failure metadata are designed into the artifact layout.
- The claim ceiling correctly excludes Qwen/GQA transfer, natural language, long-context generalization, efficiency, scaling, novelty, and paper-primary claims.

These strengths should be retained in the remediated study.

## Launch-blocking findings

### P0-1 — The exact handoff commit has a failed CI run

The workflow run for `8ffa52f...` failed before executing the test suite or smoke matrix. The workflow installs `"[dev]"`, while `pyproject.toml` defines only the `test` extra. Pip warns that the `dev` extra does not exist, and the next step fails because `pytest` is unavailable.

Required remediation:

- either define `dev = ["pytest>=8", ...]` and use it consistently;
- or replace every `.[dev]` / `${LAB}[dev]` invocation with the existing `.[test]` extra;
- pin or constrain the PyTorch installation used by CI so a CPU smoke job does not download an unnecessary multi-gigabyte CUDA stack;
- require a green unit-and-smoke workflow on the exact launch commit.

### P0-2 — The claimed cascade/delta shared initialization is not implemented

`paired_initialization_audit()` states that cascade and learned-delta shared tensors must be identical. The actual `build_model()` path merely resets the random seed and constructs two different module graphs. The learned-delta graph contains additional learned trajectory-encoder layers, which consume random numbers during construction and global reinitialization. Consequently, common tensors are not initialized identically merely because the same seed was used.

This contradicts the implementation summary's description of a canonical supernet copy and should cause the preflight audit to raise before any training run begins.

Required remediation:

1. Build one canonical `learned_delta` supernet per seed.
2. Construct every arm without relying on coincident RNG order.
3. Copy each compatible tensor from the canonical state dictionary by exact key, dtype, and shape.
4. Initialize arm-only tensors under explicitly derived, recorded seeds.
5. Prove exact shared-tensor equality for all five seeds before the first optimizer is created.
6. Add an end-to-end test that would fail if module registration order changes.

### P0-3 — Stale completed runs are reused without validation

`train_one_run()` returns an existing complete `run_manifest.json` immediately when `--overwrite` is absent. It does not compare that run's configuration hash, dataset-manifest hash, source head, dirty-tree state, or expected parent lineage with the current invocation.

The test suite expects `_validate_completed_run_reuse()`, but that function is absent from the checked-in CLI. After the dependency problem is repaired, this should be an immediate test failure.

This permits a dangerous sequence:

1. complete one or more runs;
2. change code, configuration, or materialized data;
3. rerun the matrix in the same output directory;
4. silently combine old complete runs with new runs.

Required remediation:

- implement the tested reuse guard before dispatching any run;
- validate study ID, exact config SHA-256, exact dataset-manifest SHA-256, exact source head, parent ancestry, arm, seed, completed-step count, final checkpoint hash, train-log hash, and required prediction hashes;
- reject dirty-source confirmatory execution;
- never permit `--force-data` with existing confirmatory runs;
- make the matrix manifest enumerate exactly one valid run for every registered arm/seed cell.

### P0-4 — Smoke results are not forced to remain non-evidentiary

The smoke configuration contains `"confirmatory": false`, and the workflow requires the analyzer to emit `confirmatory_config: false` with decision `smoke_only`. The analyzer currently ignores the `confirmatory` field, omits `confirmatory_config`, and defines a full matrix relative to the smoke configuration itself. A complete one-seed smoke matrix can therefore be labeled `supports_phase1_mechanism_hypothesis`, `inconclusive`, or `contradicted` instead of being unconditionally classified as `smoke_only`.

The emitted analysis also fails the checked-in analysis JSON schema, which requires `confirmatory_config`.

Required remediation:

- require a boolean `confirmatory` field when loading every study configuration;
- emit `confirmatory_config` in the matrix and analysis artifacts;
- force `decision = "smoke_only"` whenever `confirmatory` is false, regardless of scores or matrix completeness;
- prohibit a confirmatory decision unless the study ID, exact registered config hash, and complete registered matrix all match;
- validate generated artifacts against their JSON schemas in CI and before upload.

### P0-5 — The analyzer does not establish a closed evidence chain

The analyzer checks some prediction hashes and verifies that run manifests share one config hash and one dataset hash. It does not establish that those shared hashes equal the current registered configuration, current matrix manifest, current materialized dataset, or current source checkout. It also does not enforce a common clean Git head across all runs.

Additional recorded artifacts are not fully revalidated: final checkpoint hashes, train-log hashes, completed steps versus the registered 2,500, environment consistency, run membership, expected prediction condition set, prediction schema/identity fields, and the relationship between preflight initialization hashes and each run's initial model hash.

The `paired_initialization_and_artifact_integrity` advancement gate is consequently marked true after a narrower audit than its name and preregistration imply.

Required remediation:

- bind current config → dataset manifest → preflight → matrix manifest → every run manifest → checkpoints/logs/predictions → analysis with exact hashes;
- require one clean source head and compatible runtime environment across the matrix;
- verify every final checkpoint and train log, not only prediction files;
- verify exactly 15 distinct registered run cells, exactly 2,500 updates each, and the registered evaluation conditions;
- compare prediction IDs and answer tokens against the materialized test split;
- reject duplicate, missing, non-finite, mislabeled, or internally inconsistent prediction rows;
- make the integrity gate derive from these checks rather than hard-coding success.

### P0-6 — The strong “information-matched” interpretation is not realized exactly

The preregistration argues that current carrier `c_j` plus raw history mean `m_j` or raw finite difference `d_j` gives algebraically equivalent access to the previous carrier. In the implementation, the downstream mechanism does not receive raw `m_j` or `d_j`. The control first passes through `RMSNorm`, a learned projection, SiLU, and another learned projection. This path can discard magnitude and is not generally invertible, even when the trajectory rank equals the head dimension.

Therefore a learned-delta advantage would currently compare two matched-parameter **normalized nonlinear coordinate encodings**, not prove that finite differences beat an information-equivalent history representation.

Required remediation before observing confirmatory outcomes:

- preferred: preserve an explicit, same-shaped raw identity/residual control path in both arms and place any learned residual transform on top of it;
- alternatively: weaken the preregistered claim to the actually implemented normalized-encoding comparison;
- add a test demonstrating the exact information-preservation property being claimed;
- because this changes either the mechanism or the estimand wording, issue a new study ID, new immutable configuration, new output root, and amended preregistration before the full run.

### P0-7 — The registered bootstrap treats a crossed design as nested

All training seeds are evaluated on the same materialized test examples. Seed and example are therefore crossed factors. The current bootstrap samples training seeds and then independently resamples examples inside each selected seed. This breaks the shared-example structure across seeds and can understate uncertainty when example-level effects are correlated across model seeds.

Ten thousand draws do not compensate for having only five independent training seeds.

Required remediation before outcomes are visible:

- choose and preregister either:
  - a crossed/two-way paired bootstrap that resamples seed IDs and example IDs while preserving all selected seed × example cells; or
  - a seed-cluster analysis that treats the generated test set as fixed and limits the claim accordingly;
- run an outcome-blind power simulation for the +0.02 and +0.015 gates under plausible seed variance;
- record the statistical implementation and synthetic test cases before launching the full matrix.

### P0-8 — Confirmatory test outcomes are exposed progressively

Each arm/seed run evaluates and writes test predictions immediately after its own training. The 15-run matrix is sequential, so test outcomes from early cells are visible before later cells train or before failure/replacement decisions are made. The repository has no blinding or deferred-evaluation contract proving that operators did not inspect partial confirmatory outcomes.

Required remediation:

- split the pipeline into training and evaluation phases;
- train and hash all 15 fixed final checkpoints first;
- verify matrix completeness and source identity;
- only then run one frozen test-evaluation phase over all checkpoints;
- keep development evaluation available for operational debugging, but prevent partial test summaries from being printed during confirmatory execution.

## High-priority operational and power risks

### P1-1 — Resume can duplicate training-log records

A crash may occur after log records have been appended but before the next checkpoint. Resume rewinds the deterministic batch stream to the older checkpoint but leaves later log rows in place, then appends repeated steps. The model trajectory can still resume correctly, but the evidentiary log is no longer a unique step sequence.

Before launch, truncate the log to the restored checkpoint boundary or write checkpoint-generation-specific logs and merge them with an audited rule. Add an interrupted-versus-uninterrupted byte/equivalence test for final model state, optimizer state, counters, batch signatures, and canonicalized log records.

### P1-2 — Failure evidence is deleted or overwritten on retry

At the start of a run, `failure_bundle.json` is unlinked. A later failure writes to the same filename. This contradicts the preregistration's requirement to preserve failed attempts and permits loss of the original failure evidence.

Use immutable attempt directories or timestamped/content-addressed failure bundles. Never delete a failed attempt during confirmatory reuse or resume.

### P1-3 — The full workflow destroys resumable evidence at startup

The manual full job begins with `rm -rf full_evidence`. A rerun after interruption discards checkpoints and failure evidence instead of resuming. Use a persistent study-ID/source-SHA output directory and refuse destructive reuse. Archive an old directory before any deliberate restart.

### P1-4 — The primary batch-roll falsifier has avoidable confounds

The intervention rolls control tensors among examples inside heterogeneous padded batches. A valid position can receive a donor's padded-position control. In addition, later-head controls are generated from already intervened hybrid carrier states rather than replayed from the donor's aligned forward pass. A large aligned-versus-rolled gap can therefore include padding mismatch and compounded counterfactual instability.

Before treating the contrast as a precise alignment falsifier:

- use a fixed derangement matched by valid sequence length and preferably registered stratum;
- consider capture-and-replay of aligned control tensors so every donor signal comes from an unperturbed pass;
- preregister more than one deterministic derangement or show robustness to the chosen permutation;
- retain the current intervention as a useful diagnostic if its interpretation is narrowed.

### P1-5 — Iterative-task difficulty is not guaranteed by the requested hop/step label

Pointer mappings are random permutations. Fixed points and short cycles can make a nominal multi-hop answer equal the start node or require fewer distinct transitions than the hop label suggests. State-update sequences can likewise cancel modulo the domain. These cases are valid generated examples, but they dilute the iterative mechanism signal and reduce power.

Before launch, report and freeze difficulty diagnostics such as minimal effective pointer-cycle distance, answer-equals-start rate, cancellation rate, and accuracy of registered shortcut baselines. Either balance/exclude trivial cases before outcomes under a new dataset version or retain them with a claim and power analysis that acknowledges the mixture.

### P1-6 — Held-out ranges overlap training ranges

The test examples are instance-held-out and shifted upward on average, but some registered hop/step values overlap training. Describe the result as held-out generated-task performance, not uniformly out-of-range extrapolation, unless the generator is amended before launch.

## Required remediation sequence

Because several corrections change the mechanism, causal interpretation, statistical analysis, or confirmatory exposure policy, do not patch the existing study in place and reuse its study ID.

1. Preserve `tgsa-phase1-model-identifiability-2026-08-28` and its current no-result state unchanged.
2. Create a remediated implementation branch with an intuitive versioned name, for example `research/tgsa-phase1-identifiability-v2-launch-ready`.
3. Assign a new study ID and output root.
4. Amend the preregistration before any new confirmatory test result is generated.
5. Repair initialization, reuse, confirmatory classification, integrity validation, resume/failure preservation, deferred test evaluation, and the control/statistical contracts.
6. Run all unit tests plus an end-to-end smoke matrix.
7. Require the exact launch commit's CI workflow to be green.
8. Run an outcome-blind power and task-difficulty audit.
9. Freeze the launch commit and record all source/config/schema hashes.
10. Only then dispatch the full 3 × 5 matrix.

## Pre-launch acceptance gates

A launch recommendation may change to GO only when all of the following are demonstrated on the exact launch commit:

- clean checkout and verified ancestry from `aefb70e...`;
- no uncommitted or untracked files at doctor/preflight time, excluding a predeclared external evidence directory;
- complete test suite passes, including the currently missing reuse guard;
- end-to-end smoke matrix completes and schema-validates;
- smoke analysis always reports `confirmatory_config=false` and `decision=smoke_only`;
- canonical shared initialization passes for all five registered seeds;
- stale config/data/source/checkpoint/prediction mutations are rejected by negative tests;
- interrupted and uninterrupted training are equivalent under the registered determinism contract;
- failed-attempt artifacts remain immutable across retry/resume;
- statistical implementation matches the amended preregistration and passes simulation tests;
- task-difficulty and shortcut-baseline report is frozen;
- all 15 training checkpoints are complete before confirmatory test evaluation begins;
- one post-run verifier validates the complete hash chain and all evidence schemas.

## Conditional PowerShell launch package

The commands below are a template for the **remediated, newly registered study only**. Do not substitute the audited no-go head and do not run the full matrix until every acceptance gate above passes.

```powershell
$ErrorActionPreference = "Stop"

$Branch = "research/tgsa-phase1-identifiability-v2-launch-ready"
$LaunchCommit = "<PINNED_REMEDIATED_COMMIT_SHA>"
$ParentCommit = "aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e"
$StudyId = "<NEW_REGISTERED_STUDY_ID>"

git fetch origin --prune
git checkout $Branch
git pull --ff-only origin $Branch

$Head = (git rev-parse HEAD).Trim()
if ($Head -ne $LaunchCommit) { throw "Wrong launch commit: $Head" }
git merge-base --is-ancestor $ParentCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Pinned parent is not an ancestor" }
if ((git status --porcelain).Length -ne 0) { throw "Launch checkout is dirty" }

cd .\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install -e ".[dev]"
python -m pytest .\tests -q --tb=short --durations=20

$Config = (Resolve-Path ".\configs\phase1_model_identifiability_v2.json").Path
$EvidenceBase = "D:\tgsa_evidence"
$Root = Join-Path $EvidenceBase (Join-Path $StudyId $LaunchCommit)
$Data = Join-Path $Root "dataset"
$Output = Join-Path $Root "study"
$Repo = (Resolve-Path "..").Path

if (Test-Path $Root) { throw "Evidence root already exists: $Root" }
New-Item -ItemType Directory -Path $Root | Out-Null

python .\scripts\prepare_phase1_data.py `
  --config $Config `
  --output-dir $Data

python .\scripts\phase1_doctor.py `
  --config $Config `
  --dataset-dir $Data `
  --repo-root $Repo `
  --output (Join-Path $Root "doctor.json")

# The remediated runner must train and hash all registered checkpoints before
# exposing confirmatory test predictions.
python .\scripts\run_phase1_model_study.py `
  --config $Config `
  --dataset-dir $Data `
  --output-dir $Output `
  --repo-root $Repo

python .\scripts\analyze_phase1_model_study.py `
  --config $Config `
  --output-dir $Output

python .\scripts\verify_phase1_evidence.py `
  --config $Config `
  --dataset-dir $Data `
  --output-dir $Output `
  --repo-root $Repo
```

Confirmatory execution must not use `--force-data`, `--overwrite`, `--allow-partial`, or a reused output root.

## Evidence package to preserve

- exact launch commit and complete Git status;
- amended preregistration and config hashes;
- dependency lock/freeze and runtime/GPU metadata;
- dataset manifest plus all split and instance-ID hashes;
- doctor and canonical initialization proofs for every seed;
- matrix manifest and exactly 15 run manifests;
- immutable attempt/failure records;
- all fixed final checkpoints, train logs, telemetry, and prediction files with hashes;
- statistical source hash and simulation/power report;
- schema-validation report and final closed-chain verifier output;
- analysis JSON/Markdown and explicit pass/fail state for every frozen gate.

## Launch decision rule

- **Current audited head `8ffa52f...`: NO-GO.**
- **Superficial dependency-only repair: still NO-GO.**
- **GO is available only after the remediated study has a new registration, all acceptance gates pass, and CI is green on the exact pinned launch commit.**
