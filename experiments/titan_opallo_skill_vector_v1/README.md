# Titan–Opallo skill-vector composition — TITAN-OPALLO-SKILL-VECTOR-001

This is a new, isolated experiment. It does not modify `titan_model_soup_mini_v1` or
any source checkpoint.

The construction is:

```text
W0 = FP32(Supa stored BF16 bytes)
dS = Titan-Sparse FP32 - W0
dO = Opallo FP32 trained tensors - W0
W(alpha,beta) = W0 + alpha*dS + beta*dO
```

The Opallo FP32 trained-tensor checkpoint is the exact `B28last20` arm used to
materialize the published Opallo checkpoint. The published Opallo BF16 weights are
used only to verify that `W0 + dO`, cast once to BF16, reproduces the published bytes.
Titan-Sparse is treated as an atomic sparse vector; it is not re-masked.

Arms:

| Arm | `(alpha, beta)` |
|---|---:|
| BASE | `(0, 0)` |
| SPARSE | `(1, 0)` |
| OPALLO | `(0, 1)` |
| AVG | `(0.5, 0.5)` |
| ADD-100 | `(1, 1)` |
| ADD-S75 | `(0.75, 1)` |
| ADD-O75 | `(1, 0.75)` |
| ADD-75 | `(0.75, 0.75)` |
| CONFLICT-LOCAL | `(1, 1)`, except beta `0.5` on tensors with cosine `<= -0.20` |

Run the offline phase from the repository root:

```powershell
python experiments/titan_opallo_skill_vector_v1/compose.py audit `
  --manifest experiments/titan_opallo_skill_vector_v1/manifest.json `
  --output experiments/titan_opallo_skill_vector_v1/GEOMETRY.json
```

Then materialize one arm at a time into the new `arms` directory. The writer is
create-only and emits FP32 weights plus copied sidecars:

```powershell
python experiments/titan_opallo_skill_vector_v1/compose.py build `
  --manifest experiments/titan_opallo_skill_vector_v1/manifest.json `
  --geometry experiments/titan_opallo_skill_vector_v1/GEOMETRY.json `
  --arm ADD-100 `
  --output experiments/titan_opallo_skill_vector_v1/arms/ADD-100
```

This phase records geometry and materializes candidates only. It performs zero
model forwards and makes no quality, retention, release, or composability claim.

Reload every arm and verify its tensor inventory, FP32 dtype, finiteness, and
exact BASE/SPARSE/OPALLO endpoint tensors:

```powershell
python experiments/titan_opallo_skill_vector_v1/compose.py verify `
  --manifest experiments/titan_opallo_skill_vector_v1/manifest.json `
  --arms-root experiments/titan_opallo_skill_vector_v1/arms `
  --output experiments/titan_opallo_skill_vector_v1/VERIFICATION.json
```

An optional exploratory functional probe is also included. It uses only the
newly authored `data/fresh_probe.jsonl` and writes only to this experiment:

```powershell
python experiments/titan_opallo_skill_vector_v1/run_native_probe.py `
  --arms-root experiments/titan_opallo_skill_vector_v1/arms `
  --suite experiments/titan_opallo_skill_vector_v1/data/fresh_probe.jsonl `
  --output experiments/titan_opallo_skill_vector_v1/FUNCTIONAL_PROBE.json
```

This probe is exploratory and must not be read as a formal confirmation or
release gate.
