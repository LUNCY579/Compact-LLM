# Rineya Published Cores

Developer ledger of recorded names, timestamps, bases, and datasets: [`../roadmap.md`](../roadmap.md).

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
| `Rineya-Halo-1.2.0-devfreeze` | `rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v2@1e-4 |
| `Rineya-Nimbus-1.3.0-devfreeze` | `rineya.core.nimbus.v1.3.0-devfreeze.28L.qwen3-crossdepth-aegis` | 28-layer Qwen3; 19 kept blocks 50/50 with vault Aegis |
| `Rineya-Lyra-1.4.0-devfreeze` | `rineya.core.lyra.v1.4.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-1e5` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v1@1e-5 |
| `Rineya-Halo-Uncensored-1.5.0-devfreeze` | `rineya.core.halo-uncensored.v1.5.0-devfreeze.28L.qwen3-replace-v1-5e4-v2-1e4` | Same Halo donor; kept blocks fully replaced (α=1.0). Uncensored pair for Halo |

Add new cores with `python _materialize_core.py`. The script refuses if the destination already exists.
