# Progressive RL Curriculum — PRLC-001

A reusable, capability-gated post-training controller for compact Rineya models.
It separates dataset eligibility, sampling proportions, model execution, reward
verification, evaluation, and durable run state. This is an additive experiment:
it does not replace production inference, existing loaders, or other research arms.

**Release boundary:** controller and real PyTorch loss/update code are implemented
and CPU-tested. Real Rineya weights, the local Hugging Face loader, RTX 4070 memory
fit, and MiMo agent environments have **not** been qualified in this release.
The synthetic demo is a workflow fixture, not evidence of Rineya improvement.

## What actually runs

The complete deterministic-text path is:

```
validate / plan (no model import or network)
  -> native frozen inference qualification (explicit model-load permission)
  -> READY or BOOTSTRAP_REQUIRED / NO_SIGNAL
  -> on-policy grouped rollouts
  -> deterministic verifier + incomplete-group filtering
  -> clipped GRPO + independent retention-training CE
  -> fixed calibration / per-source retention evaluation
  -> HOLD, PROMOTE, ROLLBACK, EXHAUSTED, or COMPLETE
```

When the model has almost no on-policy success, an explicitly supplied,
verifier-approved teacher cache can enter bounded SFT, followed by requalification.
There is no automatic paid teacher call, training launch, model download, Docker
pull, shell execution, remote-code loading, or access to sealed evaluation banks.

A pure-Python Bernoulli arithmetic policy exercises all controller stages. The
separate PyTorch tests perform actual autograd, AdamW updates, SFT, frozen-reference
checks, and optimizer/RNG checkpoint restoration on a tiny random CPU model.

## Quick start from the repository root

The controller/demo requires only Python 3.11 or later:

```bash
python -m unittest discover -s experiments/progressive_rl_curriculum_v1/tests -v
python -m experiments.progressive_rl_curriculum_v1 demo --run-dir ./runs/prlc-demo
python -m experiments.progressive_rl_curriculum_v1 report --run-dir ./runs/prlc-demo
```

Torch tests skip explicitly when PyTorch is absent. To exercise them, install a
compatible CPU or CUDA PyTorch build. No pretrained weights are downloaded by tests.
The release's complete demo finished all three synthetic stages in 12 optimizer
updates; this is not a suggested native training budget or efficacy measurement.

To test deterministic resume rather than start another run:

```bash
python -m experiments.progressive_rl_curriculum_v1 demo --run-dir ./runs/prlc-resume --max-actions 2
python -m experiments.progressive_rl_curriculum_v1 demo --run-dir ./runs/prlc-resume --resume
```

Use new output directories for new runs. Do not edit committed checkpoint files.

## Native model workflow

Install native dependencies only on the intended host, preserving its existing
CUDA installation. The sibling `requirements-native.txt` documents optional
requirements; the root environment is not changed by this experiment.

Prepare a bounded canonical JSONL catalog and edit a profile's source names to
match that catalog. `rineya_exact` in the profiles is a logical enrollment name,
not a claimed published Hugging Face dataset or a qualified checkpoint.

```bash
python -m experiments.progressive_rl_curriculum_v1 plan \
  --config experiments/progressive_rl_curriculum_v1/configs/rineya_344m.json \
  --catalog /path/to/enrolled_tasks.jsonl

python -m experiments.progressive_rl_curriculum_v1 qualify \
  --config experiments/progressive_rl_curriculum_v1/configs/rineya_344m.json \
  --catalog /path/to/enrolled_tasks.jsonl \
  --model-dir /path/to/local_hf_compatible_model \
  --run-dir ./runs/prlc-native-001 --device cuda --allow-model-load
```

`qualify` performs no optimizer updates. It establishes fixed evaluation baselines,
checks generated-token/EOS/behavior-logprob alignment, and measures initial reward
signal. A READY result is necessary, not a claim that a subsequent optimizer step
will fit in memory. Native backward/optimizer behavior must still pass on the host.

Only after reviewing that receipt, explicitly authorize a bounded first update:

```bash
python -m experiments.progressive_rl_curriculum_v1 train \
  --config experiments/progressive_rl_curriculum_v1/configs/rineya_344m.json \
  --catalog /path/to/enrolled_tasks.jsonl \
  --model-dir /path/to/local_hf_compatible_model \
  --run-dir ./runs/prlc-native-001 --device cuda \
  --allow-model-load --allow-training --max-actions 1
```

`resume` accepts the same arguments. It requires training permission for optimization,
but can finish interrupted NEW/REQUALIFY inference without `--allow-training`.
Identity drift in source code, plan, catalog, model files, parameter selection,
runtime versions, or device blocks resume. A different model uses a different run.
A synthetic receipt cannot authorize a native run.

The bundled loader supports local, safetensors-backed, standard Hugging Face
causal LMs with `q_proj` and `v_proj` LoRA targets. It uses `trust_remote_code=False`
and `local_files_only=True`. It does **not** claim to load every packed/custom
Rineya family artifact. An existing qualified custom loader may construct a
`TorchBackend` in Python, but needs its own exact model/tokenizer/runtime binding.

## Dataset and split contract

Each JSONL row follows `Task` in `schema.py`, for example:

```json
{"id":"arithmetic-0001","source":"rineya_exact","revision":"immutable-source-revision","group":"family-a:parent-0001","split":"train","prompt":"What is 2 + 3?","answer":"5","domain":"reasoning","verifier":"exact-v1","requirements":["text"],"metadata":{"provenance":"owned-generated-example"}}
```

IDs are unique within a source. Parent groups are global identifiers and must be
namespaced across independent sources. Variants of a parent must remain in one
split. The four roles are strictly distinct:

| Role | Used for |
| --- | --- |
| `train` | Curriculum selection, RL, verified teacher bootstrap |
| `retention_train` | Rehearsal CE, never retention scoring |
| `calibration` | Adaptive promotion decisions, never gradients or difficulty routing |
| `retention_eval` | Fixed preservation evaluation, never replay |

`test`, `challenge`, `sealed`, and unknown labels are rejected rather than silently
mapped to training. Supplied DEV/TEST banks that were already exposed must not be
relabeled "sealed." Split construction and parent identification remain an explicit
data-owner responsibility. Cross-split normalized-prompt and parent collisions are
checked; this is **not** a semantic decontamination guarantee.

No gold answer or grading metadata appears in the `Prompt` DTO passed to the model.
The exact verifier accepts only stripped exact text and completed EOS-terminated
responses. It does not execute generated Python, perform fuzzy matching, or
silently strip arbitrary reasoning text. Other answer envelopes need a separately
registered and tested verifier/runtime revision.

Local JSONL exports are capped at 256 MiB to avoid accidentally materializing a
multi-gigabyte environment export in constrained host RAM. Prepare a bounded,
parent-disjoint subset before enrollment; this release is not a disk-backed index
for an arbitrarily large raw corpus.

## Unlocking versus proportions

`fraction` is the maximum fraction of **compatible training parent groups per
eligible source**, not a percentage of all raw rows. Groups are hash-ranked using
the fixed seed and admitted by a rounded-down prefix. Growing fractions and
nested eligible source sets therefore preserve earlier access. A zero source
weight changes sampling, not data access.

`source_weights` chooses among eligible sources. `band_weights` selects difficulty
within the chosen source. `replay_fraction` is a separate quota. Small integer
batches retain at least one acquisition and one replay task, and report the actual
realized mixture. Empty bands fall back within that same source and are counted;
held-out examples are never used to fill a shortage. Missing positively weighted
sources block enrollment before model allocation.

Per-task success estimates use a Beta(1,1) posterior mean. Too-few or stale
observations remain `unknown` and are eligible for exploration. In particular,
0/4 is not a reliable estimate of zero success. Hard tasks are not permanently
locked by an empirical success estimate. `p(1-p)` is only a sampling heuristic;
variance, including grader noise, is not proof that a task teaches a useful skill.
The current exact-verifier path has binary rewards.

## Gates and preservation

Promotion uses a frozen calibration panel, paired parent-cluster bootstrap
intervals, positive observed capability gain, minimum independent training-group
coverage, mixed reward groups, valid actions, and truncation limits. Two passing
windows at distinct policy versions are required by default. Every retention
source is evaluated separately against the original baseline; a gain in one bank
cannot conceal a loss in another.

The bootstrap intervals are operational controller thresholds. Repeated adaptive
use of calibration means they are **not** confirmatory confidence statements or
family-wise error guarantees. Final scientific claims require a separately
registered untouched evaluation; this controller has no sealed-data command.

Replay includes supervised preservation loss, not just replay prompts in GRPO.
This matters because all-correct replay groups have zero centered task advantage.
KL regularization is also retained. Neither CE nor KL guarantees no forgetting.
Clear retention regression restores the last accepted model/optimizer checkpoint
and stops as ROLLED_BACK. Uncertain preservation gates can increase rehearsal
within a bounded range instead of promoting. Spent budgets never roll back.

## Native objective and resource policy

The implementation is clipped **sequence-mean** GRPO with population-standardized
within-prompt advantages and a frozen-reference sampled KL estimator. It is not
advertised as DAPO, GSPO, RoLA, NGU, or a reproduction of MiMo's training recipe.
It preserves the existing RoLA/NGU branch's sequence-mean loss convention without
importing that branch's interventions or authorization.

Sampling is categorical at temperature 1 with no top-k/top-p truncation. Actual
sampler scores are retained and compared with teacher-forced log probabilities;
a mismatch above 1e-3 blocks qualification/update. This conservative tolerance is
not yet validated for every BF16/cache implementation. Dropout is disabled during
sampling and loss computation. EOS participates in the loss, prompt tokens do not,
and incomplete trajectories are excluded from policy updates but counted in costs
and truncation metrics. Optimizer gradients and updated trainable parameters must
remain finite.

The native loader uses fixed rank-8 LoRA on q/v projections. Parameter patterns
are bound for the entire run, and the generic backend rejects more than 20% trainable
parameters. Automatic layer/rank unfreezing is intentionally **not enabled** in this
study: changing data and model capacity together would confound the experiment.
A different capacity policy requires a new plan and qualification.

There is one GPU model and a frozen CPU reference. Microbatching is one sequence
at a time. CPU reference scoring trades speed for VRAM; it may become a major
bottleneck. No throughput, memory-fit, or 4070 training-time claims follow from
these CPU tests. The two native profiles keep rollout group size and generation
cap fixed across stages rather than pretending larger contexts were qualified.

Admission checks two consecutive host-RAM observations, free disk, and GPU memory
before loading. Runtime checks preserve a separate worker RAM threshold. These
are project defaults, not portable measurements. Token, call, optimizer, stage,
and wall-time budgets cover qualification, training, evaluation, and bootstrap.
The `processed_tokens` ledger is a conservative forward/backward accounting unit,
not measured FLOPs. Offline teacher-generation cost is outside the local run and
must be budgeted separately. Wall time is checked at operation boundaries; it is
not a hard real-time preemption of a running GPU kernel or `generate` call.

## Bootstrap from a verified cache

A teacher JSONL row has exactly `task_key`, `response`, `teacher`, and `provenance`.
Only currently unlocked training tasks with verifier-passing responses are accepted.
Cache bytes/provenance are recorded in the run events. No calibration or retention
answers may be converted into teacher examples. Run `bootstrap` with the native
arguments above plus `--demonstrations /path/to/teacher.jsonl` and both permissions.
The run must already be BOOTSTRAP_REQUIRED. SFT includes preservation replay and
has a total step ceiling; after it, original retention baselines remain in force.

## MiMo release boundary

MiMo supplies agent environment specifications, not a ready-made successful-answer
corpus. The import adapter preserves those specifications and exact source revision:

```bash
python -m experiments.progressive_rl_curriculum_v1 import-mimo \
  --input /path/to/local_mimo_subset.jsonl --output /path/to/mimo_registry.jsonl \
  --revision EXACT_40_HEX_HF_COMMIT --domain code --split train \
  --group-field YOUR_AUDITED_PARENT_FIELD
```

The revision and parent field are intentionally not guessed. Non-text prompt
content is rejected. Every imported task requires a `mimo:<domain>` capability,
which the bundled runtime does not advertise. **Live MiMo agents, Docker sandboxes,
visual graders, music graders, and cyber environments are not implemented or
qualified by this commit.** Supporting them requires an audited multi-turn token
contract, environment/verifier integration, sandbox resource limits, and fresh
qualification. The generic `Backend` protocol is a boundary, not a claim that the
single-turn `Episode` structure already implements all of Uni-Agent.

## Durable state and failures

Each operation reserves worst-case work and saves the pre-operation model,
optimizer, RNG, and controller before expensive work begins. Immutable checkpoint
directories have content hashes and hash-linked receipts; LATEST is atomically
replaced only after checkpoint completion. One admission lock excludes competing
writers before native allocation. Orphan checkpoints are preserved, not overwritten.

On interrupted work, resume restores pre-operation state but retains the full
reservation. Update counters and pending evaluation state are committed together
with updated weights. Promotion cannot be repeated by evaluating one checkpoint
twice. Rollback restores parameters/optimizer, retains spending, and advances the
policy version to invalidate stale trajectories. A live or stale lock is never
automatically deleted: verify no writer remains before removing a stale lock.
Checksums detect accidental corruption, not malicious rewriting of the entire run.

## Files and extension points

- `schema.py`, `data.py`: contracts, offline registry, source/teacher import.
- `curriculum.py`, `engine.py`: answer-unaware routing and state transitions.
- `objectives.py`, `backends.py`: GRPO/CE and explicitly invoked runtimes.
- `evidence.py`, `cli.py`: accounting, durable state, resource checks, entrypoints.
- `tests/`: control-plane, failure injection, and tiny real-PyTorch tests.

The repository base, test evidence, and external references are recorded in
`RECONNAISSANCE.md`, `VALIDATION.json`, and `SOURCES.md`. No existing research branch,
production checkpoint, dataset, or sealed bank was changed for this implementation.
