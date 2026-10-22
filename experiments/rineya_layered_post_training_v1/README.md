# RLP-001 — Rineya Layered Post-training

A runnable, isolated experiment inspired by the Limite/Violetto post-training diagram:
**acquire → preference learning → RL → verified consolidation → constrained weight soup
→ repeat**, with an optional prefix-value critic. This is an adaptation, not an exact
reproduction of an unpublished/full training recipe or evidence of native improvement.

Parent: `post-training-Mimo@03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Branch: `research/rineya-layered-post-training-v1`.
Status and measured validation are recorded in `VERIFICATION.md` and `validation/`.

## Implemented

The PyTorch runtime performs response-only SFT/off-policy trace distillation,
sequence-sum DPO, PRLC GRPO, verifier-approved RL-trace consolidation with protected
replay, and token-prefix actor-critic updates. These are gradient updates, not a
Bernoulli controller simulation. The complete graph runs on a trainable tiny causal
GRU in the CPU smoke test; the guarded native loader targets local Qwen3-format 0.6B
checkpoints. Custom-architecture Rineya checkpoints require a separately audited loader.

Weight operations include the last 2–5 available checkpoints (default tail length 5),
endpoint-inclusive interpolation grids, a three-parent second soup, explicit
common-base independent delta coefficients, and an exact rank-concatenating LoRA-delta
utility. Native arms use a fixed **dense projection subset**, not LoRA: q/v/o weights
in the last four decoder layers, capped at 5% of total parameters. This keeps full
weight interpolation mathematically meaningful and trainable capacity fixed. The
LoRA helper is tested but deliberately not wired into training because concatenation
would increase rank and confound comparisons.

Every soup records coefficients and ancestry, invalidates stale rollouts and resets
Adam moments. No soup changes the original checkpoint on disk. Development selection
has a separate floor for every protected suite; an aggregate score cannot buy a loss
in instruction following or retention. A base-checkpoint fallback is explicit.

| Arm | Sequence |
| --- | --- |
| `control` | distill → DPO → RL1 → RL2 |
| `soup_only` | control → RL2 tail-checkpoint soup |
| `layered` | distill → tail soup → DPO → RL1 → SFT1 → weighted soup1 |
| `full` | distill → tail soup → DPO → RL1 → RL2 → SFT1 → soup1 → SFT2 → soup2 → critic fit/audit → critic RL3 |
| `full_no_critic` | full graph without critic fitting; RL3 uses GRPO |
| `full_no_soup` | full graph without the three soup operations |

All arms receive the same configured number of actor optimizer steps. That does NOT
make their compute equal: rollout generation, reference forwards, critic updates and
coefficient-search evaluation are metered separately. `compare` flags unequal token
proxies rather than manufacturing a compute-matched claim. Set `replay_weight=0` in a
separately frozen configuration for the replay ablation.

## Quick start from repository root

Use an isolated Python research environment. The repository's existing PRLC package is
required; do not copy or alter it. Install the scoped requirements only in that environment.

```sh
python -m pytest -q experiments/rineya_layered_post_training_v1/tests
python -m experiments.rineya_layered_post_training_v1 plan --arm full
python -m experiments.rineya_layered_post_training_v1 smoke --arm full --output outputs/rlp_full_smoke
python -m experiments.rineya_layered_post_training_v1 confirm --synthetic --output outputs/rlp_full_smoke
```

The smoke command creates its own deterministic synthetic data. It never downloads or
loads a 0.6B checkpoint. `confirm` emits an explicit synthetic label for this run and
refuses to overwrite an existing confirmation artifact.

## Native pilot

See `RUNBOOK.md` for commands, data preparation and stop rules. Start at 2K, with the
critic-free `layered` arm and controls. The 4K configuration is an **unqualified candidate**,
not a claim that the user's 8GB GPU can train it. No 65K/131K run is enabled. Within a
run, all stages share the same qualified context ceiling; a length-changing curriculum
is deliberately a subsequent factorial experiment rather than a hidden confound.

Seven suite names are mandatory by default: acquisition, retention, instruction,
general QA, roleplay contract, abstention and long context. Exact, JSON-equality and
finite text-constraint verifiers are implemented. These are objective contract probes,
NOT a comprehensive free-form roleplay or general-intelligence evaluator. External
judges and full benchmark adapters must be audited separately before broader claims.

The scalar critic uses hidden states preceding each response action and half-MSE to a
terminal binary return (gamma=1/lambda=1 equivalent Monte Carlo target). It is a linear
head on detached actor features, not a second trained billion-parameter backbone.
Outputs are unconstrained raw values, not calibrated probabilities. A native critic
must pass held-out ranking AND MSE-vs-constant checks before RL3. Synthetic smoke tests
explicitly bypass that scientific gate to exercise code paths; their reports expose it.

## Outputs and safety

Runs create exclusive directories with config/data hashes, qualification diagnostics,
immutable safetensors subset checkpoints and parent hashes, per-stage development
scores, an immutable selection lock, phase-by-phase compute accounting and failure
records. Optimizer resume is not supported: checkpoint exports are weight snapshots,
not resumable Adam states. Restart an interrupted experimental arm from its frozen base
in a new output directory. Never describe that as an exact resumed training run.

`run` validates and fingerprints confirmation rows but never generates on or scores
them. `confirm` is a separate command that verifies the frozen choice and data hash.
This is operational split isolation, not cryptographic concealment of labels from a
user who owns the input JSONL. For a genuinely blind campaign, keep confirmation data
with an independent evaluator and use a future manifest-only handoff adapter.

See `PROTOCOL.md` for statistical interpretation, `SOURCES.md` for evidence boundaries,
and `INTEGRATION.md` for the precise relationship to earlier Rineya work.
