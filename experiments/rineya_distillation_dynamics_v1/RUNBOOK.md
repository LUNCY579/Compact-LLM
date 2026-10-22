# Operator runbook

## 0. Preserve the laptop pause and existing work

Native execution is NOT part of this delivery. While the laptop is being serviced,
use only CPU tests, `plan`, and the explicitly labeled tiny `demo`. Do not enable
schedules, run model qualification, start training, or invoke a paid teacher.

Use a clean dedicated worktree rather than discarding unrelated local changes:

```bash
git fetch origin
git worktree add ../ChatBubble-rdd001 origin/research/rineya-distillation-dynamics-v1
cd ../ChatBubble-rdd001
python -m pytest experiments/rineya_distillation_dynamics_v1/tests -q
```

The worktree command creates a detached checkout suitable for running reviewed
code. Use a separately named local branch for subsequent edits. Keep all private
models, datasets, receipts and outputs outside the source worktree. There are no
automatic dependency installers or native CI jobs in this experiment.

## 1. Prepare reviewed data and a pilot plan

The 2026-10-05 preparation uses `configs/selected-pilot.json`, not `pilot.json`.
Inspect it offline with `plan --plan experiments/rineya_distillation_dynamics_v1/configs/selected-pilot.json --stage selected`.
It freezes A/B/D/SFT at one seed/rate and two steps with per-step checkpoints.
Exactly 16 teacher-dev rows with distinct parents are required. Only approved
fresh catalogs/exclusion metadata can be enrolled; no confirmation bank is read
to prepare them. Long-context retention is outside this 1024-token pilot's claims.
The current preparation's inventory, proposal and validation are retained under
`C:/Users/clclc/BusinessProject/rdd001-records/preparation-20261005/`.

Before any future treatment decision, the one prospective native action is the
supervised `qualify` subcommand. It requires the same genuine registration/E0,
export parity, host admission/shared lock, `--allow-model-load`, `--allow-training`
and `--ack-host-available` gates as treatment. It makes exactly two disposable
updates and stops before `run_arm`. It is not authorized by this preparation.
Do not chain it into treatment, retry a failed output, or enumerate full/screen
cells from the selected plan. The proposed source is exact Titan-Experimental;
the fixed Supa-1.7 candidate teacher is unqualified until review and screening.
Halo's tokenizer backend differs and is rejected by the existing parity rule.

The local inventory found no RDD-admissible E0 or reviewed native export receipt.
The historical CE512 pass has a different status, stopping/masking semantics and
resource scope. Retain it as historical evidence; do not relabel it or import
its lower host reserve. Current memory readings and an absent lock are observations,
not host authorization. Other owner processes have been observed on this host.

The sample `configs/pilot.json` is not an execution grant. Choose pilot learning
rates, the effect/retention margins, context limits, step/checkpoint/sample budgets,
and seeds before examining treatment outcomes. Use a distinct plan/registration
for a one-seed engineering pilot versus the replicated campaign.

Catalogs are UTF-8 JSON objects:

```json
{
  "schema": 1,
  "provenance": "reviewed_external",
  "tasks": [
    {
      "key": "owned-state:train:0001",
      "parent": "owned-state:world:0001",
      "source": "owned-state",
      "revision": "immutable-source-revision",
      "role": "train",
      "prompt": "Start with 5. Add 3, then double it. End with <final>integer</final>.",
      "answer": "16",
      "solution": "(5+3)*2=16. <final>16</final>",
      "verifier": "final-int-v1"
    }
  ]
}
```

The one-row illustration is NOT a sufficient catalog. Enroll all required roles
from `PROTOCOL.md`, at least 16 distinct TEACHER_DEV parents, enough TRAIN/replay
rows for each batch, and meaningfully sized fresh acquisition/retention/hard DEV
suites. Retention-training and retention-evaluation source sets must agree, but
parents/prompts must be disjoint. Held-out solutions are empty strings. The other
supported grader, `exact-v1`, compares the complete stripped response. Neither
grader proves intermediate reasoning correctness.

An exclusions file has `parents` and `prompt_hashes` arrays. A prompt hash is
`contracts.digest(data.normalized(prompt))`, i.e. canonical-JSON SHA-256, not raw
UTF-8 text SHA-256. Import approved exclusion metadata; do not open historical
spent/sealed banks to populate this experiment. Audit semantic overlap and rights
separately; an empty registry is not a decontamination certificate.

Keep a distinct, independently prepared confirmation catalog unopened. Its roles
are `confirm_acquisition`, `confirm_retention`, `confirm_hard`, all with empty
solutions. Commit its file SHA-256 in the binding before training.

```bash
python -m experiments.rineya_distillation_dynamics_v1 audit \
  --catalog /absolute/private/catalog.json --exclusions /absolute/private/exclusions.json
python -m experiments.rineya_distillation_dynamics_v1 plan \
  --plan /absolute/private/pilot.json --stage screen
```

## 2. Qualify and bind exact local assets

Use the existing Rineya native/E0 workflow unchanged on its original reviewed
branch. Do not generate a fake E0 PASS or repurpose CPU fixture evidence. RDD
requires the matching E0 config/result and these checks: exact native load,
finite scoped backward, frozen-base preservation, EOS/pad masking, and bitwise
save/reload/resume. RDD itself does not resume treatment runs.

This branch does not contain the central native exporter. Before using an HF
export, obtain independent exact-native versus exported-model output parity and
lineage review. Only a standard local `Qwen3ForCausalLM` export with safetensors,
config, tokenizer and template files is supported. There is no automatic converter
for custom Titan/Supa artifacts and no stock-model fallback. A model name and
matching parameter count are insufficient proof of provenance.

```bash
python -m experiments.rineya_distillation_dynamics_v1 inventory --model-dir /absolute/student-export
python -m experiments.rineya_distillation_dynamics_v1 inventory --model-dir /absolute/teacher-export
```

On Windows, absolute paths may use `C:/...`. Inventory only hashes files; it does
not load or qualify weights. The whole `files` mapping determines the snapshot
SHA-256. Copy those actual digests and the actual parameter counts into a private
copy of `configs/native-binding.example.json`. Its zero digests, missing teacher
count and false review flag intentionally prevent execution.

Supply an externally produced export-review JSON containing:

```json
{
  "status": "REVIEWED_NATIVE_EXPORT",
  "student_snapshot": "ACTUAL_STUDENT_INVENTORY_SHA256",
  "teacher_snapshot": "ACTUAL_TEACHER_INVENTORY_SHA256",
  "e0_result_sha256": "ACTUAL_E0_RESULT_FILE_SHA256",
  "native_greedy_parity": true,
  "teacher_prefix_reviewed": true,
  "reviewed_by": "ACTUAL_REVIEWER"
}
```

Set these booleans only after the actual review. This is an operator attestation,
not a generated qualification certificate. Review teacher feedback at representative
student error prefixes, not only teacher answers to clean prompts. RDD additionally
screens teacher accuracy and teacher-minus-student advantage on TEACHER_DEV.

Every file binding includes its absolute path and actual byte SHA-256. The native
plan, asset snapshots, catalog/exclusions, confirmation commitment and implementation
hashes are frozen by:

```bash
python -m experiments.rineya_distillation_dynamics_v1 register \
  --plan /absolute/private/pilot.json --binding /absolute/private/binding.json \
  --output /absolute/private/registration.json
python -m experiments.rineya_distillation_dynamics_v1 preflight \
  --registration /absolute/private/registration.json
```

`preflight` is read-only identity/runtime checking, not successful model loading,
resource admission, teacher qualification, or native training. Unsupported assets,
missing receipts, fixture catalogs, changed files or runtime drift fail closed.

## 3. Future host execution — only after the host is available and approved

Use the existing Windows Python 3.11 / torch 2.10.0+cu126 / Transformers 4.57.1 /
tokenizers 0.22.2 / safetensors 0.7.0 environment. Do not change the CUDA environment
merely to run this package. The supervisor preserves 7.4 GiB host admission twice,
6.0 GiB running host headroom, 4.0 GiB initial GPU free memory, 0.5 GiB running GPU
headroom, 4.0 GiB disk free space, 4.0 GiB process-tree RSS and 1,800 seconds per
invocation. It uses the existing `rineya-central-gpu0.lock` location. It does not
stop other GPU applications or steal their locks.

A teacher too large for these floors remains blocked. CPU offload is not a way to
ignore total RAM or the process-tree ceiling. There is no paid/API fallback.

Run one registered arm at a time, with a fresh output path:

```powershell
$py = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
$m = "experiments.rineya_distillation_dynamics_v1"
& $py -B -m $m run --registration "C:/rdd/registration.json" `
  --arm A-lr0-s17 --output "C:/rdd/runs/A-lr0-s17" `
  --allow-model-load --allow-training --ack-host-available
```

That command performs real disposable qualification updates before treatment.
It verifies zero-adapter output equivalence, native FKL/RKL value/gradient parity,
finite scoped updates, frozen teacher/base preservation, bitwise adapter save/reload,
and teacher competence. It restores initialization and starts a fresh optimizer
for the arm. Qualification costs are separate from treatment results.

Enumerate the remaining names with `plan`. Do not parallelize arms on this laptop.
An interrupted treatment is failed/incomplete, not resumable or automatically
restarted. Diagnose it first; a newly approved attempt uses a new directory from
the common source, never a partial checkpoint. The pilot must fit its full
registered horizon in the resource budget before scaling up.

## 4. Lock selection and perform one independent confirmation

After **all** runs in the registered stage and all seeds finish, pass their
`arm/RESULT.json` files explicitly to `select`:

```bash
python -m experiments.rineya_distillation_dynamics_v1 select \
  --registration /absolute/private/registration.json --stage screen \
  --results /absolute/run1/arm/RESULT.json /absolute/run2/arm/RESULT.json \
  --output /absolute/private/selection
```

The two paths illustrate syntax, not sufficient stage coverage. Missing arms,
seeds, baselines/checkpoints or provenance cause refusal. A no-learning or harmful
campaign yields `NO_ELIGIBLE_CONFIGURATION`; do not lower thresholds to force a
winner. Selection chooses one recipe and step across seeds, not the best seed.

Only after eligible selection and explicit confirmation approval:

```powershell
& $py -B -m $m confirm --registration "C:/rdd/registration.json" `
  --selection "C:/rdd/selection/SELECTION.json" `
  --confirmation "C:/rdd/confirmation.json" --output "C:/rdd/confirm-once" `
  --allow-model-load --allow-confirmation --ack-host-available
```

The marker is written before the confirmation catalog is opened. Wrong catalog,
interruption, resource refusal after opening, or a model error stays spent. The
same registered acquisition/retention gates are reported for every seed. Even a
scoped pass does not authorize model promotion or production replacement.

## 5. GD-001 handoff and later work

`export-peft` converts a complete native adapter checkpoint into standard PEFT
filenames/key names and writes a lineage sidecar. It verifies the original
registration, run and checkpoint but **does not establish PEFT runtime parity**.

```bash
python -m experiments.rineya_distillation_dynamics_v1 export-peft \
  --registration /absolute/private/registration.json \
  --checkpoint /absolute/run/arm/checkpoints/000008 --output /absolute/export
```

Qualify actual HF/PEFT outputs against the custom q/v wrapper before calling
GD-001's `record_snapshot(..., kind='peft')` on a separately integrated branch.
GD-001 is not present in this parent and is not imported automatically. Its own
snapshot, selection and confirmation gates still apply. Do not reuse RDD
confirmation for adaptive GD selection. Composition and centralized-reasoning
extensions likewise require fresh protocols and independent confirmation.

For future selected runs, every `backend.save` also preserves an original
standard-PEFT child snapshot. Every checkpoint and complete/failed run retains a
consumed-prompt manifest; qualification provenance survives the source reset.
Complete forward records require full causal input IDs and their hash, including
generated conditioning tokens. The verifier rejects prompt-only forward records;
preserve such historical records rather than silently upgrading their coverage.
Parity acceptance rehashes the full base/tokenizer inventory; any changed or added
snapshot file requires a fresh qualified receipt rather than reuse of old parity.
The GD owner now requires `gd001_attachment_v1` with an independently approved
prospective producer plan, two original snapshots including the final, genuine
post-save receipts, complete raw source/filter/consume/unused JSONL and exact
rendered text/token IDs. A complete RDD run alone does not invent that approval.
Use the GD owner's attachment verifier only after native parity and those inputs
are available. Its successful metadata audit is not native export qualification.
