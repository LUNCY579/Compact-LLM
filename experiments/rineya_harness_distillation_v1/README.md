# Rineya Harness Distillation — RHD-001

**Qualification-first implementation, not a native training result.** This isolated
experiment turns student-first, privately reviewed interaction trajectories into
replay-verified next-action SFT data. The training-only reviewer either passes a
proposal byte-for-byte or supplies a complete replacement in the same minimal
harness action space. Deployment evaluation has no reviewer or reference guidance.

Parent: `research/progressive-rl-curriculum-v1` at
`5517361ec09e02308eae0352b2e53a70853c5245`. All changes are additive. Production
inference, upstream PRLC code, weights, prior experiments, and existing branches
are not modified. No previous model-load, training, or paid-API grant is inherited.

## What is implemented

- Immutable public-view, reference, review, and action contracts. Student/reviewer
  callbacks receive public views, not task golds, record stores, or graders.
- Student-first collection with PASS/REPLACE, request/session binding, bounded
  steps/context/response/calls, failure accounting, and independent attempt seeds.
- A real deterministic local environment: bounded integer arithmetic checks,
  key-based retrieval, answer, and abstain. There is no Bash or arbitrary code tool.
- Replay auditing against the enrolled task/environment, source hashes, reference
  identity, observed tool outputs, and terminal outcome. Imported `success` flags
  alone are insufficient. Successful, complete TRAIN trajectories produce SFT rows.
- Per-action SFT: prior actions and tool observations are prompt context; only the
  current accepted action plus EOS is supervised. Private reviewer notes and the
  rejected proposal are excluded from the student trajectory and SFT targets.
- Separate ablation datasets, parent/prompt split checks, provenance, pair-candidate
  export, and intervention statistics with an explicitly defined denominator.
- Reviewer-free calibration/retention evaluation, fixed panel identities, complete
  paired coverage checks, per-source retention results, and parent-cluster intervals.
- An explicit bridge to the inherited PRLC `Prompt`, `Episode`, and `Task` contracts.
  This can wrap an already-created backend for action proposals and transport
  audited action rows to its SFT interface. It does not load weights or run training.
- Offline CLI, synthetic end-to-end demo, adversarial tests, and additive CPU CI.

## What is not qualified or implemented

No native Rineya weights, real tokenizer, GPU allocation, teacher model, native
optimizer update, or live MiMo environment has been run. The full inherited PRLC
suite has not been rerun here; the bridge is tested against its exact schema blob.
There is no live teacher/API adapter, standalone native RHD trainer, automatic
source-harness evolution, or native execution of all six comparison arms.

The source-harness SFT arm needs a source-space adaptation implementation. The
answer/oracle review control is deliberately disabled and must use a separately
isolated privileged-data pipeline. It cannot feed the ordinary distillation data.
MaxRL, DPO optimization, WSD comparisons, capacity unlocking, and delta merging are
registered follow-ups, not mixed into the first mechanism experiment.

## Run from the repository root

Python 3.11+; the offline path uses only the standard library.

```bash
python -m unittest discover -s experiments/rineya_harness_distillation_v1/tests -v
python -m experiments.rineya_harness_distillation_v1 plan
python -m experiments.rineya_harness_distillation_v1 demo --out ./runs/rhd-demo --tasks 24
python -m experiments.rineya_harness_distillation_v1 verify --out ./runs/rhd-demo
```

The demo uses `FixtureStudent`, `RuleReviewer`, and a **hand-coded** `ScriptedExpert`.
The expert is not a trained model. Its comparison only checks the plumbing of the
reviewer-free evaluator; its apparent improvement is not distillation evidence.
The optional full-size fixture run is `--tasks 256 --attempts 2`: 512 attempts per
arm for three scripted modes, not 1,536 real-model or paid-teacher trajectories.

New runs require new output directories. A run writes its manifest last; interrupted
folders remain incomplete and are not automatically deleted or resumed. Hashes
provide corruption/drift detection, not cryptographic proof against malicious
rewriting. Keep `trajectories.private.jsonl` and the grader-bearing catalog private.

## Compile audited trajectories

```bash
python -m experiments.rineya_harness_distillation_v1 compile \
  --catalog ./runs/rhd-demo/catalog.jsonl \
  --trajectories ./runs/rhd-demo/trajectories.private.jsonl \
  --mode review_k --out ./runs/rhd-sft
```

A catalog row is `Task` in `contracts.py`: key, source, immutable source revision,
parent group, split, skill, public prompt, private expected answer, and immutable
record pairs. `expected: null` means the task grader expects abstention. This is a
controlled local verifier, not a general uncertainty-calibration evaluator.

Collection and SFT accept `train` only. Calibration and `retention_eval` are
reviewer-free evaluation roles. `retention_train` is a separate replay-data role
and is not collected by this pipeline. `sealed` may be reserved in a catalog but
cannot be collected, exported, or evaluated by this CLI. Register confirmation
panels outside the adaptive pipeline before any native efficacy claim.

Split checks cover exact parent identity and normalized prompt collisions, not
semantic near-duplicate detection. Data owners must audit licenses, provenance,
parentage, and semantic contamination before enrollment. No MiMo dataset was
imported, and no earlier project-corpus row count is represented as MiMo's size.

The compiler replays every selected record, including failed attempts; errors
are counted rather than silently discarded from success-rate denominators. Only
successful complete trajectories enter SFT. Multiple ablation modes in one export
are rejected. CLI selection is explicit and reports unselected records.

Each row has `prompt`, `completion`, task/parent/source provenance, attempt, step,
and trajectory identity. `tokenize_sft` checks EOS, label masks, template wrapping,
and context overflow without truncation. Real-tokenizer equivalence still needs
native qualification; the byte tokenizer in tests is only a fixture.

## PRLC integration boundary

```python
from experiments.rineya_harness_distillation_v1.prlc_bridge import (
    PRLCStudent, to_prlc_demonstrations,
)

# backend must already be constructed, host-qualified, and explicitly authorized.
student = PRLCStudent(backend, max_new_tokens=128, allow_model_calls=True)
# audited_rows must come from compile_dataset on the original trajectories.
demonstrations = to_prlc_demonstrations(audited_rows)
# No optimizer is invoked by either helper.
```

The exact JSON transcript envelope returned by `render(view)` is used both for
collection and SFT. PRLC's ordinary user-message wrapper remains unchanged.
An SFT `Task.answer` here is the next action, **not** the original task's final
answer. Passing an action-imitation verifier is not evidence of multi-turn agent
success. Use RHD's reviewer-free environment evaluation for that question.

The first host exercise must compare serialized prompt token IDs between the
collector and loss path, confirm a single EOS, inspect action/observation masks,
and reject truncation. Custom/packed Rineya artifacts need a qualified loader;
inheritance from PRLC does not make every family checkpoint HF-compatible.

## Experiment design

`configs/rhd001.json` freezes a 256-parent, two-attempt proposal, three procedural
skills, and the initial mechanism subset: untreated student, empty-reference
review, and procedural-reference review. Teacher generation is supported as a
protocol mode but has no bound external teacher. Six conceptual comparisons are
registered with explicit implementation/native-status labels.

Before comparing trained arms, match initial checkpoint, parent tasks, attempts,
decoding, supervised-token and optimizer budgets, adapter scope, and independent
retention replay. Report both attempted-collection cost and retained token counts;
success filtering can otherwise confound the supervision-source comparison. The
manifest registers this matching requirement; it is not an implemented six-arm
native training scheduler. Use multiple seeds only after single-host feasibility.

Intervention rate is REPLACE / (PASS + REPLACE) among **executed reviewed actions**;
review-call failures remain visible separately. A lower rate alone is not proof
of internalization: quality, trajectory length, and reviewer behavior also matter.
Use frozen reviewer/reference versions and reviewer-free task/retention outcomes.

Behavior flags record actual tool execution rather than words such as “I checked.”
They are deliberately coarse and do not prove that a check was relevant or that
abstention was calibrated. A future native study needs task-aware behavior
validators, coverage/selective-risk metrics, and an independent abstention panel.

Replacement pairs are diagnostic candidates, not automatic preferences. The
replacement was executed but the rejected branch usually was not. Every exported
pair is `preference_eligible: false` pending independent same-state counterfactual
verification. No automatic DPO/GRPO conversion is provided; reviewed corrections
are off-policy with respect to the student's original action distribution.

## Resource and trust boundaries

The offline ledger limits calls and characters, not provider tokens, currency,
FLOPs, or VRAM. Wall time is checked between operations, not by preempting a callback.
Live adapters require their own enforced API/token/cost budgets, timeout/cancellation,
request persistence, and a fresh grant. There is no persistent native resume here.

Public DTO isolation prevents accidental gold leakage through this API. It is not
an OS security boundary against a hostile Python callback that reads local files.
Run untrusted teachers/environments out of process under separately audited access
controls. Schema validation cannot prove that arbitrary generated answer text is
free of semantic leakage; training targets still need content review.

See `HOST_HANDOFF.md` for the next gate, `SOURCES.md` for the paper relationship,
and `VALIDATION.json` for exact local evidence and the native NOT_RUN boundary.
