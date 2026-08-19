# Rineya Published Cores

Immutable vault for **dense** Hugging Face checkpoints (`model.safetensors`).
This folder is the only place new merged/copied release cores are written.

It does **not** replace official phase artifacts. Sources under
`short-response-model-phase-0-1-2/artifacts/` are never modified.

## Policy

- Never overwrite a core directory that already exists.
- Never write back into phase `artifacts/`.
- Each core has a stable `core_id` and a `CORE.json` provenance file.
- Weights stay local (gitignored). Provenance files may be committed.

## Cores

| Directory | core_id | Kind |
|---|---|---|
| `Rineya-Pulse-1.0.0-rc1` | `rineya.core.pulse.v1.0.0-rc1.19L.454M.dense` | Copy of the Phase 8 dense teacher (already merged) |
| `Rineya-Aegis-1.1.0-devfreeze` | `rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4` | LoRA merge of teacher dual-SFT v1 `5e-4` into Pulse's base |

Add new cores with `python _materialize_core.py`. The script refuses if the destination already exists.
