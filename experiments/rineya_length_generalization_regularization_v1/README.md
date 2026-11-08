# LGR-001 — Length-Generalization Regularization

**Status: implemented and CPU-fixture tested; native Rineya efficacy is UNMEASURED.**

Dedicated branch: `research/rineya-length-generalization-regularization-v1`.
Pinned parent: `post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Nothing outside this package changes. No default recipe, pretrained weights,
production loader, workflow, game project, or preexisting evidence is modified.

## Question

At matched training exposure and comparable new-skill acquisition, does ordinary
pre-linear dropout or VPAD preserve a 0.6B Rineya checkpoint's ability to use longer
contexts better than the no-dropout reference? Preserving inherited context after
short SFT is the first target, NOT a claim of extrapolation beyond pretraining.

## Implemented

* Reversible, explicit Qwen-style block-norm intervention: none, pre-linear,
  VPAD, and a separately named pre-norm control. The final norm and Q/K norms are
  excluded. Parameters/state-dict keys are untouched by the norm patch.
* Q/V-only rank-8 LoRA reference with frozen base, FP32 adapters, zero adapter
  dropout, CPU adapter checkpoints, strict reload and explicit merge utilities.
* Completion-only bounded training, exact sequence accounting, accumulation,
  clipping, an explicit warmup/linear-decay schedule and 0/25/50/100% snapshots.
* Exact-token synthetic recall/multiple-recall/state-update/absent diagnostics,
  parent/prompt split audits, no-truncation greedy collection and paired analysis.
* Parent-and-seed-resampled intervals, endpoint macro-averages, acquisition,
  retention and abstention checks; missing/fixture evidence cannot be promoted.
* An optional local-safetensors-only native CLI with explicit load/training
  permissions, shape and resource guards, content-bound qualification receipts,
  expiry, data-eligibility attestations and create-only run directories.

The native CLI is a real implementation, but it was NOT exercised on a pretrained
checkpoint here. It supports only a compatible local Hugging Face Qwen3-shaped
0.6B export (28 layers, width 1024, FFN 3072), not arbitrary native Rineya loaders,
quantized/streamed graphs, custom fused kernels or distributed training.

## CPU validation and smoke run

From the repository root (Bash or PowerShell):

```text
python -m pytest -q experiments/rineya_length_generalization_regularization_v1/tests
python -m experiments.rineya_length_generalization_regularization_v1 plan
python -m experiments.rineya_length_generalization_regularization_v1 smoke --output /path/to/NEW/lgr-smoke
```

Use a new output directory each time. `smoke` runs all three arms on a tiny,
randomly initialized CPU fixture. Its optimizer steps validate mechanics only;
it does not load Rineya, download models, use the owner's laptop, start Codex,
launch a GPU run or measure a useful learned capability.

Install dependencies in a separate environment; do not change the repository's
root requirements. `requirements-test.txt` is enough for the mechanics tests;
`requirements-native.txt` describes the optional, still-unqualified native lane.

## Files

| File | Responsibility |
|---|---|
| `config.py`, `configs/` | Immutable single-arm plans and controlled factorial generation |
| `regularization.py` | VPAD, placement audit, reversible norm patch, bounded probes |
| `adapters.py` | Explicit Q/V LoRA, state validation, merge reference |
| `data.py` | Token-exact generated banks and split/identity checks |
| `training.py` | One-pass completion-only bounded optimizer loop |
| `evaluation.py` | Greedy collection and paired statistical reporting |
| `provenance.py`, `native.py` | Hash bindings, native qualification, resources, explicit execution |
| `fixtures.py`, `tests/` | CPU fixtures, including actual training/qualification code paths |
| `RESEARCH_PROTOCOL.md` | Hypotheses, ablations, interpretation and decision rules |
| `RUNBOOK.md` | Ordered operator steps, input/output contracts and limits |
| `SOURCES.md`, `NOTICE.md`, `COPYING` | Source pins, attribution and package-specific GPL terms |
| `VALIDATION.json`, `SOURCE_MANIFEST.json` | Delivered validation status and byte-level provenance |

## Research boundary

This is complementary to MiMo post-training, retention-dose, replay-composition,
and composable-skill research. It neither replaces those branches nor proves
that regularization caused historical forgetting. Do not combine recurrence,
RoPE changes, random attention, KV compression, RL, new curricula or new replay
composition into the initial three-arm test.

The old `Rineya_Long_Context_Demo_v1` suite is **report-only/quarantined**, not
eligible training, selection, or confirmation material. Do not import its rows.
Existing positive and negative records stay unchanged.

## What is not claimed

No native checkpoint qualification, completed pretrained training, RULER/HELMET
replication, context extension, retention improvement, parameter-composition
improvement, serving-speed result, or general intelligence gain is established.
Passing fixture tests does not open CHECK/FINAL or override any existing grant.
The code has no automatic promotion, branch merge, scheduler or background job.

Start with `RUNBOOK.md`; the next native step is a separately authorized exact-
checkpoint qualification, not an automatic sweep.
