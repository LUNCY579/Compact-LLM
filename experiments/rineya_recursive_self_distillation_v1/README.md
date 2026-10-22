# RSD-001 — Verified recursive self-distillation for Titan 0.6B

**Implemented experiment software; not a Titan improvement result.** The primary
student is `Rineya-Titan-Experimental-1.0.0-devfreeze` (596,049,920 base parameters).
No 344M run, pretrained load, native optimizer update, or paid teacher call was
performed when implementing this package. See `VALIDATION.json` for exact local
CPU evidence, and `PROTOCOL.md` for the hypotheses and limitations.

Parent repository branch: `research/post-training-mimo-centralized-v1` at
`5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80`. This package is additive. Central E0/E1,
production inference, model weights, historical results and other experiments
remain unchanged. A new experiment does not inherit a prior native execution grant.

## Offline readiness continuation (2026-10-05)

Current RSD source has no qualified parent, reviewed real catalog or native RSD
receipt. Preserved E0 v2 is `BLOCKED_E0/HOST_RAM_GUARD`, with zero loads/updates;
that September receipt does not measure today's host. Centralized E1 preparation
is not an SFT parent. RSD remains secondary to CE8/pruning/matched_mlp.

The new model-free readiness entry point writes only a new BusinessProject
directory outside tracked checkouts:

```powershell
$py = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
& $py -B -m experiments.rineya_recursive_self_distillation_v1.readiness --output "C:/Users/clclc/BusinessProject/rsd001-readiness-NEW"
```

`READINESS.json` and `SOURCE_MANIFEST.json` bind the inspected bytes and missing
prerequisites. `MINIMAL_COMPARISON_PROPOSAL.json` selects only `sft`, `frozen` and
`evolving`, seed101, four acquisition plus one independent replay task per update,
and **two** updates at a prespecified evaluated step2. Teacher version0 is common
at update1; refreshing first differs at update2. The configuration is
`configs/tiny_guidance_v1.json`. No SRCL arm is selected. Input identities remain
unresolved, so this is a prospective design draft requiring exact qualification
and preregistration before native work, not completed enrollment or permission.

Native binding **schema2** now requires `catalog_review`. E0 consumption checks
the producer's result, source/runtime and supervisor/GO receipts, including
zero-adapter parity, sampled likelihood, padding/causality and all original
checks. Process-only PASS is insufficient. Relevant E0 producer bytes, including
the mask and watchdog-contract dependencies at its recorded commit, must match;
its historical Git HEAD need not equal RSD HEAD. Parent review/enrollment now
parses a completed schema2 RSD ordinary-SFT training receipt and checks the
exported adapter, base, input/source hashes, runtime, steps and terminal evaluation.
The recorded parent commit must exist in this module's fixed local repository,
and every source hash must match a regular `.py` Git blob at that exact commit;
older committed source may differ from current HEAD. Missing or mismatching
history fails closed, without fetching or executing historical source. Git reads
have a 10-second process timeout and limits of 128 source files, 1,024 UTF-8 path
bytes, 2 MiB per blob and 16 MiB total source bytes.
Fixtures and unfinished/non-SFT receipts stay ineligible even with review flags.
Unknown external SFT receipt formats require a separately reviewed provenance
adapter contract; template flags or a newly written attestation cannot substitute
for genuine completed training evidence.

Catalog review uses the strict contract in `catalog_qualification.py`: schema1,
experiment `RSD-001`, status `REVIEWED_CATALOG_NOT_CAPABILITY_EVIDENCE`,
`evidence_kind: real_review`, exact catalog/exclusion file bindings, a named
reviewer, source/revision coverage with separate hash-bound ownership/license and
independent reference-review evidence, historical exclusion authority evidence,
and grouping/reference/parent-training-overlap attestations. Hashes check drift;
they cannot prove review truth, intermediate reasoning or semantic decontamination.
Existing schema1 native bindings must be rebuilt rather than relabeled.

## What runs

| Component | Implementation |
|---|---|
| Ordinary SFT parent | Optional bounded native `parent` stage, starting from exact Titan with zero q/v LoRA; exports a candidate requiring explicit operator review. This is **not centralized E1**. |
| Treatments | Continued SFT, frozen guidance, evolving guidance, frozen+SRCL, evolving+SRCL, and SRCL-only. |
| Guidance | Full-vocabulary forward KL `KL(teacher || student)` at identical student response-token prefixes; detached teacher; refresh after each update only in evolving arms. |
| Output projection | Chunked exact-vocabulary first-order autograd; recomputes the frozen LM projection during backward, instead of retaining a response-length by vocabulary probability history. |
| Concise learning | Gold-free rewrite input, strictly shorter completed output, structural/endpoint verification, token-mean CE on accepted rewrites; no accepted rewrite means no concise loss. |
| Preservation | Independent retention-training CE with fixed weight/exposure; source-specific retention evaluation and stop rules. No automatic promotion or rollback. |
| Accounting | Attempted generation, full-prefix scoring and conservative processed-token units; optimizer and checkpoint versions; diagnostic/qualification costs reported separately. |
| Persistence | Immutable checkpoint directories, file hashes, atomic latest pointer, optimizer/adapter/RNG state, exclusive writer locks, clean-boundary resume. Interrupted transactions are quarantined. |
| Native host | Local-only strict Titan loader, existing q/v rank-8 FP32 LoRA wrapper, BF16 frozen base, Windows watchdog, shared GPU lock, memory/time checks. |

The native program is deliberately gated. The inspected centralized branch still
had a host-memory block before E0. The `parent` stage avoids depending on an
unfinished centralized E1 executor, but **does not bypass E0**. Existing genuine,
reviewed SFT parents may instead be supplied through the parent manifest contract.

## 1. Check out and run the CPU tests

Use a clean checkout. Do not discard unrelated local changes to switch branches.
From an existing clone, first-time branch checkout is:

```powershell
git fetch origin
git switch --track origin/research/rineya-recursive-self-distillation-v1
```

When the local branch already exists, use `git switch research/rineya-recursive-self-distillation-v1`.
The following variables assume the repository root is the working directory:

```powershell
$py = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
$m = "experiments.rineya_recursive_self_distillation_v1"
$pkg = "experiments/rineya_recursive_self_distillation_v1"
$plan = "$pkg/configs/pilot.json"
$art = Join-Path (Split-Path -Parent $PWD.Path) "rsd001-artifacts"
New-Item -ItemType Directory -Force $art | Out-Null
& $py -B -m unittest discover -s "$pkg/tests" -v
```

CPU tests require PyTorch and safetensors; they do not import Transformers, load
pretrained models, or allocate a GPU. Local implementation validation used Python
3.13.5 / torch 2.10.0+cpu; that does not qualify the native interpreter. Native
execution requires the existing Python 3.11.15 / torch 2.10.0+cu126 /
Transformers 4.57.1 / safetensors 0.7.0 / tokenizers 0.22.2 environment. Do not
replace the host CUDA installation just to run CPU tests.

An offline mechanics fixture can exercise planning, but is rejected by native enrollment:

```powershell
& $py -B -m $m fixture --output "$art/fixture.json" --per-role 4
& $py -B -m $m plan --plan $plan --catalog "$art/fixture.json" --output "$art/fixture-plan.json"
```

Both commands are model-free. The fixture is not a research dataset or evidence
that Titan improved. Never change its provenance label merely to bypass enrollment.

## 2. Enroll a real, reviewed catalog

Use a UTF-8 JSON **array**, maximum 10,000 rows / 32 MiB, with exactly these task
fields. Each row needs a reliable solution, not just an unverified teacher claim:

```json
{
  "key": "owned-state:train-0001",
  "source": "owned-state",
  "revision": "immutable-generator-or-source-revision",
  "parent": "owned-state:world-0001",
  "split": "train",
  "prompt": "Start with 5. Add 3, then multiply by 2. Show your calculation and end with <final>integer</final>.",
  "answer": "16",
  "solution": "5+3=8; 8*2=16. <final>16</final>",
  "verifier": "final-int-v1",
  "provenance": "Record the actual source, ownership/license and solution review here."
}
```

The five required roles are `train`, `train_selection`, `calibration`,
`retention_train`, and `retention_eval`. All variants of a globally namespaced
parent must stay in one role. Retention training and evaluation must enroll the
same set of source names, but disjoint parents/prompts. TRAIN_SELECTION is only
for diagnostics, not gradients; calibration is for adaptive monitoring, never
training. `sealed`, `test`, `challenge`, FINAL and unknown labels are rejected.

For an initial feasibility bank, use at least 32 training parents, 8 selection
parents, 16 calibration parents, and 16 independent parents in each retention
role per source. This is a feasibility suggestion, **not a powered efficacy
study**. Freeze a larger mechanism/confirmation design after measuring costs.
Acquisition parent count must cover `batch_tasks`; replay count must cover
`replay_tasks`. Native probing uses at most eight selection parents.

Supported graders are `exact-v1` (entire stripped response matches expected text)
and `final-int-v1` (exactly one terminal canonical integer answer tag). Final
answers are checked; **intermediate reasoning and semantic self-containment are
not certified**. Review reference solutions and audit accepted rewrites. Arbitrary
code, fuzzy matching, human language graders and live MiMo environments are not
implemented. MiMo environments are not automatically a bank of verified solutions.

Also supply an exclusion JSON object:

```json
{"parents": [], "normalized_prompt_sha256": []}
```

Populate it with the actual historical/confirmation exclusion registry rather
than treating an empty object as a decontamination certificate. Hash entries use
`contracts.digest(data.normalized(prompt))`: SHA256 of canonical JSON containing
the normalized string, **not** SHA256 of raw prompt bytes. No protected evaluation
content needs to be opened to import already-approved parent/hash exclusions.
These exact/normalized checks do not detect every semantic near-duplicate.

The `data.from_prlc(task, solution, provenance)` adapter preserves PRLC identities
and roles for deterministic text/exact tasks. The solution and provenance must be
provided explicitly; no split relabeling or automatic MiMo capability is inferred.

```powershell
& $py -B -m $m plan --plan $plan --catalog "$art/catalog.json" --exclusions "$art/exclusions.json" --output "$art/registered-plan.json"
```

Register source revisions, all splits, comparator/checkpoints, seeds, evaluation
budget and practical effect threshold **before reading treatment outcomes**.

## 3. Pass the existing exact Titan E0 gate

Use the existing supervisor, unchanged:

```powershell
$central = "experiments/post_training_mimo_centralized_v1"
$e0cfg = "$central/configs/E0_TITAN_BF16.json"
& $py -B -m experiments.post_training_mimo_centralized_v1.scripts.supervise_native --config $e0cfg --output "$art/e0"
```

This is a real native qualification command, including E0's disposable updates.
Read `e0/E0_RESULT.json` and its supervisor receipt. Continue only on
`PASS_E0_SCOPED_NO_EFFICACY_CLAIM`. The required checks include exact load,
finite scoped backward, unchanged frozen base, EOS handling and bitwise resume.
An earlier CPU receipt or a process-only watchdog PASS cannot substitute for E0.

Keep the existing floors: 7.4 GiB host admission (two observations in the new
supervisor), 6.0 GiB worker free RAM, 4.0 GiB GPU admission, 0.5 GiB GPU running
headroom, 4.0 GiB disk. The independent watchdog also caps worker-tree RSS at
4.0 GiB and each invocation at 1,800 seconds. Do not relax these to force a run.
Resolve competing processes or host headroom first. Do not delete a shared GPU
lock until verifying that its owner is no longer running.

## 4. Produce or supply the common SFT parent

The native parent stage is ordinary solution CE plus independent retention CE.
It is a separate RSD warm-start instrument, **not a reproduction or replacement
of centralized E1**. It first performs disposable SFT mechanics checks with exact
native KL-reference and disk-resume checks, restores initialization, then starts
bounded ordinary SFT. Every treatment later receives the exact same reviewed
parent adapter and starts a fresh optimizer.

Create a hash-bound parent-stage configuration (no model work):

```powershell
& $py -B -m $m bind --e0-config $e0cfg --e0-result "$art/e0/E0_RESULT.json" --catalog "$art/catalog.json" --exclusions "$art/exclusions.json" --catalog-review "$art/catalog-review.json" --parent-stage --output "$art/binding-parent.json"
& $py -B -m $m native --mode parent --arm sft --plan $plan --binding "$art/binding-parent.json" --output "$art/parent-run" --max-updates 1 --allow-model-load --allow-training
```

Inspect the printed `result_path`, resource samples, baseline evaluation and
checkpoint. Only a clean `PAUSED` run can continue:

```powershell
& $py -B -m $m native --mode parent --arm sft --plan $plan --binding "$art/binding-parent.json" --output "$art/parent-run" --max-updates 31 --resume --allow-model-load --allow-training
```

The bundled plan has a **total** 32-update ceiling, not 32 additional updates per
invocation. Stop rules or resource limits may end the run earlier. A completed
parent writes `SFT_PARENT_ADAPTER.safetensors`, `PARENT_TRAINING_RECEIPT.json`, and
`PARENT_CANDIDATE.json`. Completion alone is not quality qualification.

Review source purity, reference validity, unassisted outputs, task accuracy,
formatting/termination, and every retention source. Require a useful parent for
this study, not an all-EOS or all-wrong checkpoint. Only after that actual review:

```powershell
& $py -B -m $m review-parent --candidate "$art/parent-run/PARENT_CANDIDATE.json" --reviewed-by "YOUR_REVIEWER_NAME" --confirm-reviewed-parent --output "$art/REVIEWED_PARENT.json"
& $py -B -m $m bind --e0-config $e0cfg --e0-result "$art/e0/E0_RESULT.json" --catalog "$art/catalog.json" --exclusions "$art/exclusions.json" --catalog-review "$art/catalog-review.json" --parent "$art/REVIEWED_PARENT.json" --output "$art/binding-rsd.json"
```

The review command records an operator attestation; it cannot prove that someone
actually reviewed the evidence or that the model is competent. The original
candidate and training receipt remain unchanged. Existing genuine SFT parents can
use `configs/parent.template.json` with real adapter/evidence hashes and an actual
review plus a separately reviewed provenance adapter contract instead; unsupported
external receipt formats currently fail closed. Do not manufacture a native-success
receipt to satisfy this schema.

## 5. Run the no-update privilege probe

```powershell
& $py -B -m $m native --mode probe --plan $plan --binding "$art/binding-rsd.json" --output "$art/privilege-probe" --allow-model-load
```

For eight selection parents, the probe compares unaided versus privileged
continuations from the same partial student attempt, under paired generation
seeds and total response-token caps. Only the privileged role sees the solution.
The result is a **diagnostic**, not proof of generalization or a statistically
powered treatment effect. A weak result calls for reviewing difficulty, reference
quality and the explanation format before spending on a larger recursive run.
Do not route training examples using calibration/retention answers.

## 6. Qualify each arm, then run one bounded update

Start with `sft`, `frozen`, and `evolving`. Add `frozen_srcl`, `evolving_srcl`, and
`srcl_only` only as their registered follow-up comparisons. Use a separate output
and qualification receipt for every arm. The frozen and evolving arms have
**identical** privileged prompt placement; teacher refreshing is the intervention.

```powershell
$arm = "evolving"
& $py -B -m $m native --mode qualify --arm $arm --plan $plan --binding "$art/binding-rsd.json" --output "$art/qual-$arm" --allow-model-load --allow-training
```

Qualification performs three **disposable** updates: one initial step, one next
step, and a replay of that next step after adapter/optimizer/RNG reload. It checks
bitwise equality, native full-vocabulary KL value/gradient parity, unchanged frozen
weights, and parent restoration. `qualify` needs training permission despite not
exporting a learned candidate. A no-signal or malformed arm does not pass.

The command prints a supervisor `result_path`. Its `RESULT.json` must have
`PASS_RSD_MECHANICS_ONLY`, the exact arm, input/plan/source identity and runtime.
Set `$qualification` to that **actual returned path**, not a guessed receipt name:

```powershell
$qualification = "PASTE_THE_RETURNED_SUPERVISOR_RESULT_PATH"
& $py -B -m $m native --mode train --arm $arm --plan $plan --binding "$art/binding-rsd.json" --qualification $qualification --output "$art/run-$arm" --max-updates 1 --allow-model-load --allow-training
```

Read the run's `LATEST.json`, its referenced `STATE.json`, and the new supervisor
result. The first update may not be an evaluation checkpoint: default evaluations
are at steps 0, 4, 8, ..., 32. Do not call the step-zero score a step-one outcome.
After reviewing host behavior, continue the **same** run:

```powershell
& $py -B -m $m native --mode train --arm $arm --plan $plan --binding "$art/binding-rsd.json" --qualification $qualification --output "$art/run-$arm" --max-updates 31 --resume --allow-model-load --allow-training
```

Run arms serially; the shared GPU lock excludes overlapping native workers. A
changed plan, catalog, exclusion registry, parent, relevant source or runtime
requires fresh qualification and a new run, not an in-place identity override.

## 7. Inspect and compare the results

`RUN.json` binds the experiment. `LATEST.json` points to immutable checkpoint
state/adapter/optimizer/RNG files. `STATE.json` includes the update history, fixed
panel evaluations, source-specific retention results and cumulative training
costs. `NATIVE_RECEIPT-*.json` and sibling supervisor folders retain invocation,
load/qualification and resource evidence. Treat them as private research artifacts.

A paired comparison at a **prespecified evaluated step** is:

```powershell
& $py -B -m $m compare --plan $plan --left "$art/run-frozen" --right "$art/run-evolving" --step 32 --output "$art/frozen-vs-evolving-step32.json"
& $py -B -m $m compare --plan $plan --left "$art/run-sft" --right "$art/run-evolving" --step 32 --output "$art/sft-vs-evolving-step32.json"
```

Use an earlier prespecified common checkpoint for an early screen; do not silently
substitute each arm's best checkpoint. The comparison refuses incomplete pairs,
mismatched metadata/budgets, different starting identities, different seeds/plans,
active or quarantined runs, and checkpoint hash drift.

Primary output is parent-macro calibration accuracy difference with paired
parent-bootstrap intervals; repairs, damages, lengths and retention sources are
reported separately. These are adaptive-panel, exploratory intervals. The default
0.02 practical effect threshold is a **proposed two-percentage-point target**, not
a power guarantee, significance certificate, or universal falsification boundary.
Use independently frozen confirmation tasks and multiple seeds for final claims.
The CLI never opens sealed/FINAL banks and does not perform multi-seed inference.

## Failures, costs and limitations

`NO_SIGNAL` means this configuration produced no usable update, not that all
self-distillation is impossible. `STOPPED_RETENTION` and `STOPPED_TRUNCATION` stop
the run without automatic promotion. They preserve the terminal candidate and
older immutable checkpoints; **automatic rollback is not implemented**.

An exception leaves `INFLIGHT.json` plus a failure receipt. Do not remove that
file or a live lock to force `--resume`. Interrupted spending is uncertain, so the
run is quarantined. Preserve it and register a fresh investigation/run. Only clean
`PAUSED` checkpoints are resumable; terminal runs are intentionally not resumable.

The training ledger counts attempted generation and conservative forward/backward
processed-token units, not measured FLOPs. Rewrite failures and discarded
trajectories remain in costs. Qualification/probe/parent-mechanics costs are
reported separately. Sum invocation wall times/load costs separately from the
latest cumulative training counters; do not sum cumulative counters repeatedly.
Each watchdog invocation is capped at 1,800 seconds and training's active ledger
also has a cumulative wall cap. Repeated model loading between clean resumes is
reported but is **not** charged to the training-ledger wall counter. No throughput,
8GB fit, optimal learning-rate, or budget-efficiency claim follows from CPU tests.

This first implementation excludes architecture changes, WSD sweeps, MaxRL/GRPO
mixing, delta merging, automatic curriculum changes, arbitrary sandbox tools and
paid APIs. It isolates the mechanism before those extensions. Native success is
not implied by the code compiling or by 80 CPU tests passing.
