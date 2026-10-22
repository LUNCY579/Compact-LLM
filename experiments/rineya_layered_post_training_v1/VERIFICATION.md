# RLP-001 verification record

This record describes work actually executed in the implementation environment.
It does not report a trained Rineya checkpoint or a positive scientific result.

## Executed validation

Scoped test command, from the repository root:

```sh
python -m pytest -q experiments/rineya_layered_post_training_v1/tests
```

Result: **89 passed in 5.30 seconds** on Python 3.13.5 and PyTorch 2.10.0+cpu.
The exact captured output and environment are in `validation/environment.json`.
The existing PRLC objective used for testing was copied from the pinned parent and
its Git blob SHA verified as `b43663f69c21318ff4281e6627cfd1da1c21f550`.
No upstream file is changed by this experiment.

The suite covers configuration and budget rejection, grouped split isolation,
balanced protected-suite sampling, exact/JSON/constraint verifiers, convex soups,
independent deltas, the LoRA-factor cross-term trap, immutable checkpoint integrity,
response-only DPO gradients, preceding-action feature alignment, frozen references,
GRPO and actor-critic gradient updates, optimizer reset, stale/truncated rollout
rejection, per-suite selection floors, paired confirmation, lock/data tampering,
and all six complete training graphs. Native-only gate rejection is tested with
an injected fixture identity; that is a unit test, not native qualification.

Separately, **18 end-to-end smoke runs and 18 separate confirmations completed**:
all six arms at seeds 20261007, 20261008 and 20261009. Each arm had 28 actual actor
optimizer updates. Critic-bearing arms had 6 offline plus 4 online critic updates.
The complete summary is `validation/smoke_matrix.json`. The maximum observed
sampler-versus-teacher-forced selected-token log-probability difference was 0.0
in these CPU fixtures. This does not establish BF16/KV-cache parity on Qwen3.

All 18 reports are labeled `SYNTHETIC_TORCH_MECHANICS_PASS`; their confirmation
reports are labeled `SYNTHETIC_CONFIRMATION_ONLY`. A tiny, grammar-constrained
causal GRU is used, not a pretrained 0.6B model. Raw terminal and selected checkpoint
scores are both recorded; some runs fall back to an earlier checkpoint. These
scores are not evidence of Rineya improvement or monotonic skill accumulation.

Only **1 of the 6 critic-bearing smoke runs** passed the scientific critic audit.
The other 5 used the explicitly labeled synthetic-only bypass so that actor-critic
mechanics could still be exercised. Native training has no such bypass and must
stop before critic RL when the held-out critic audit fails. Do not cite the smoke
matrix as evidence that a useful critic has been learned.

## Native status: BLOCKED / UNQUALIFIED

The native preflight was actually invoked without an available local checkpoint.
`validation/native_preflight.json` records the observed blockers: insufficient
free host RAM, unavailable CUDA/free VRAM, missing Transformers, and a missing
local checkpoint directory. These are properties of this execution environment,
not measurements of the user's laptop.

No native Qwen3/Rineya model was loaded, no CUDA training ran, and no hardware
memory-fit or quality claim is made. The native loader and training path are
implemented but require the local pilot in `RUNBOOK.md`. Full-repository regression
and remote CI were not run. No external model/data downloads, Codex jobs, or
background training tasks were started.

## Evidence boundary and next gate

The branch is an executable experimental implementation with passing scoped CPU
checks, not a finished efficacy study. Next, qualify a local 0.6B checkpoint and
frozen seven-suite dataset at the 2K profile; run control/soup-only/layered, compare
actual compute, freeze development selections, then perform independent confirmation.
Only consider full critic RL after the separate critic audit succeeds. Replicate
selected comparisons across seeds and account for multiplicity before a promotion.

Length-changing curricula, 65K/131K contexts, trained critic backbones, vector
reward/world models, cryptographically blind evaluation handoff, native LoRA
recompression, full benchmark/judge adapters and exact optimizer resume are not
implemented in this version. See `PROTOCOL.md` for the intended follow-on tests.
