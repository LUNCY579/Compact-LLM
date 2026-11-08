# TRVC-001: Tropical-inspired verified composition for Rineya

**Implemented and CPU-tested. Native Rineya training is not qualified or executed.**

Dedicated branch: `research/rineya-tropical-verified-composition-v1`.
Repository: `LUNCY579/ChatBubble`.
Parent: `post-training-Mimo@03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Intake date: 2026-10-07. All changes are confined to this experiment directory.

## Research question

Can an exact-state archive turn useful fragments into verified training solutions,
and does hard selection, coverage rehearsal or soft weighting best transfer those
solutions into a 0.6B model without unacceptable loss of existing behavior?

This is an independent, bounded adaptation of [Tropical Reinforcement Learning](https://arxiv.org/html/2610.02478v1),
not an author-code reproduction or a claim that changing an algebra prevents
neural forgetting. `RESEARCH_PLAN.md` is the analysis and prospective study contract.
`RESULTS.md` distinguishes executed software/toy evidence from unexecuted native work.

## Implemented

- A deterministic two-register environment with closed action grammar, task/rule/
  goal/budget-bound state identity, and parent/semantic-world split checks.
- Observed-transition DAGs, root and frontier rollouts, explicit restart-prefix
  replay, frozen-revision rescoring, top-k forward/backward composition, full-path
  verification, bounded append-only archives and edge-coverage rehearsal.
- Separate root-sample, restart-completion and graph-composition provenance. Failed
  attempts retain valid steps, never their invalid final action.
- Sequence-uniform, hard-max, temperature-controlled log-sum-exp, coverage-selected,
  and separately labeled per-token imitation losses. Root-only data is a control.
- A causal-LM object bridge with one causal shift, explicit prompt/padding/EOS masks,
  differentiable path scoring, detached reference KL and separate acquired/legacy
  replay loss components. No weights or tokenizer are downloaded or loaded.
- Model-only evaluation, paired parent-clustered summaries, fixed-first-draw
  retention accounting, immutable-output receipts and explicit compute counters.

The native bridge requires an **already loaded, separately qualified** model plus
state-prompt/tokenization and generation callbacks. The shared training engine has
been tested with a mock causal LM and with a 230-parameter symbolic actor. Those
checks do not qualify a real Rineya checkpoint, tokenizer, adapter, GPU or resume.

## Run from the repository root

Python, PyTorch and pytest must already be available in the chosen environment.
The recorded environment is described in `evidence/validation.json`.
No package installation, GPU job, paid service or remote worker is launched here.

```bash
python -m pytest -q experiments/rineya_tropical_verified_composition_v1/tests
python -m experiments.rineya_tropical_verified_composition_v1.study demo --out /tmp/trvc-demo.json
python -m experiments.rineya_tropical_verified_composition_v1.study toy --out /tmp/trvc-toy.json
python -m experiments.rineya_tropical_verified_composition_v1.study preflight --out /tmp/trvc-native-intake.json
```

On Windows, replace `/tmp/...` with a new path in a local scratch directory. These
module commands also work in PowerShell. Existing outputs are refused. `preflight`
intentionally exits **2** with a written `BLOCKED_NATIVE_UNQUALIFIED` receipt;
providing plausible hashes cannot manufacture native qualification.

The toy command runs three initialization/collection seeds, seven arms, eight
updates, eight training worlds and sixteen exposed evaluation worlds. It is a
small end-to-end engineering pilot, not a statistically decisive model study.
Symbolic actions are not reported as LLM tokens. The primary shared-bank arms have
the same update cap and candidate-forward inventory; root-only data and the
per-token treatment are explicitly different controls, not a matched-compute claim.

## Files and boundaries

| File | Purpose |
|---|---|
| `domain.py` | Exact state interfaces, transitions and split audit |
| `runtime.py` | Rollouts, bounded generation, replay and actual-cost ledger |
| `archive.py` | Observed DAG, provenance, certificates, DP and coverage |
| `objectives.py` | Path objectives, causal masks, reference KL, replay composition |
| `model_bridge.py` | Object-bound native callback interface; no loader |
| `metrics.py` | Paired outcomes and parent-clustered descriptive uncertainty |
| `study.py` | CPU fixtures, tiny actor, shared training engine and static intake |
| `native_preregistration.json` | Proposed native envelope; identities unresolved |
| `tests/` | Scoped executable regression suite |
| `evidence/` | Executed receipts and content-hash manifest |

`train_fixed_bank` is the **unprotected acquisition control**. It does not quietly
turn coverage into legacy protection. Replay/KL primitives are implemented and
tested, but a full sequential R/S/legacy scheduler, its reservoirs and its native
promotion contract must be qualified as a separate treatment before use.

`evidence/toy_pilot.json.gz` preserves the complete pilot JSON, including candidate
banks, per-update losses, raw evaluation actions/outcomes and costs. Read it with
Python's standard-library `gzip` module. No native weights or proprietary external
dataset are included. See the source manifest for the exact tested bytes.
