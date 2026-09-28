# CLARE-RINEYA-001 — capacity-aware sparse skill composition

New branch: `research/titan-clare-sparse-composable-skills-v1`.
Parent: `research/titan-model-soup-mini-v1` at
`816a1e8f4d1ab1c9d1ab847697d962a141521bac`.

**Implemented and CPU-mechanics tested. Native model qualification BLOCKED;
no Rineya training or efficacy result.** A native loss/rollout/verifier integration
and the final eligible tensor pool remain unqualified. This is not a turnkey
native causal-study launcher. The CLI deliberately has no full-training switch.

The new mechanism learns **where** a skill can update a shared model, rather than
only choosing how already-trained checkpoints should be combined. Strict support
separation makes a parameter dot product zero; it does not make a transformer
functionally additive or guarantee retention/composition.

## Entry points

Run from the repository root, in the existing qualified Python environment:

```sh
python -m pytest -q experiments/clare_sparse_composition_v1/tests
python -m experiments.clare_sparse_composition_v1.cli validate-config --config experiments/clare_sparse_composition_v1/configs/learned_disjoint.json
python -m experiments.clare_sparse_composition_v1.cli synthetic --out /absolute/NEW/output --source-commit ACTUAL_SOURCE_COMMIT_SHA
python -m experiments.clare_sparse_composition_v1.cli preflight --binding experiments/clare_sparse_composition_v1/configs/host.example.json --out /absolute/NEW/preflight.json
```

The example binding is intentionally incomplete: preflight must return BLOCKED.
Every output directory/checkpoint must be new. See `HOST_HANDOFF.md` for the
Windows, local-checkpoint-only rollback smoke and its limitations.

## Components

- `allocation.py`, `training.py`, `config.py`: exact global budgets, stable ties,
  deterministic random control, L1 movement discovery, reset and fresh optimizer,
  AdamW-safe projection, transactional failure rollback and A/B/C/D/E arms.
- `data.py`, `artifacts.py`: read-only existing atomic manifests, policy/verifier
  separation, non-pickle explicit masks/deltas, provenance and boundary resume.
- `metrics.py`, `evaluation.py`: measured norms/support geometry, bounded offline
  SVD, shared-once composition helper, logit interaction, paired source-correct
  retention and parent-cluster uncertainty; no invented retention bank.
- `native.py`, `cli.py`: inherited Titan checkpoint pins and GPU lease, additional
  RAM/VRAM checks, offline/synthetic/native-smoke separation.

The public `SkillRunner.train_skill` accepts a **trusted** differentiable loss
callback. It does not sandbox arbitrary Python. The callback must feed only
`row.messages` to the policy and keep `row.verifier_json` in the verifier. Tests
exercise mechanics with synthetic losses, not native RL/SFT response masking.

## Reading order

`CLARE_RINEYA_AUDIT.md` records inspected sources and reuse.
`CLARE_RINEYA_001_PROTOCOL.md` specifies the falsifiable comparison and controls.
`HOST_HANDOFF.md` gives exact commands and remaining native gates.
`result.schema.json` defines the summary result envelope.
`evidence/` contains the measured CPU qualification and honest blocked preflight.
