# Rineya Published Cores

Developer ledger: [`../roadmap.md`](../roadmap.md) (index), [`../lineage.md`](../lineage.md) (recipes and dates), [`../comparison.md`](../comparison.md) (evals), [`../announcement.md`](../announcement.md) (public copy).

Immutable vault for **dense** Hugging Face checkpoints (`model.safetensors`).
This folder is the only place new merged/copied release cores are written.

It does **not** replace official phase artifacts. Sources under
`short-response-model-phase-0-1-2/artifacts/` are never modified.

## Policy

- Never overwrite a core directory that already exists.
- To replace a named roster slot, archive the old directory under `_superseded/` first, then write a new versioned directory.
- Never write back into phase `artifacts/`.
- Each core has a stable `core_id` and a `CORE.json` provenance file.
- Weights stay local (gitignored). Provenance files may be committed.

## Active cores

| Directory | core_id | Kind |
|---|---|---|
| `Rineya-Pulse-1.6.0-devfreeze` | `rineya.core.pulse.v1.6.0-devfreeze.28L.qwen3-late-rineya-alpha-shape` | 28L late_rineya α-shape (Halo donor). Roster name **Pulse** |
| `Rineya-Aegis-1.1.0-devfreeze` | `rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4` | LoRA merge of teacher dual-SFT v1 `5e-4` into archived Pulse 1.0 base |
| `Rineya-Halo-1.2.0-devfreeze` | `rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v2@1e-4 |
| `Rineya-Nimbus-1.6.0-devfreeze` | `rineya.core.nimbus.v1.6.0-devfreeze.28L.qwen3-x-halo-uncensored-alpha-0.40` | 28L soup-lab global_best α=0.40. Roster name **Nimbus** |
| `Rineya-Lyra-1.4.0-devfreeze` | `rineya.core.lyra.v1.4.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-1e5` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v1@1e-5 |
| `Rineya-Halo-Uncensored-1.5.0-devfreeze` | `rineya.core.halo-uncensored.v1.5.0-devfreeze.28L.qwen3-replace-v1-5e4-v2-1e4` | Same Halo donor; kept blocks fully replaced (α=1.0). Uncensored pair for Halo |

## Superseded (archived, not deleted)

| Directory | Was | Replaced by |
|---|---|---|
| `_superseded/Rineya-Pulse-1.0.0-rc1` | Phase 8 commercial 19L teacher | `Rineya-Pulse-1.6.0-devfreeze` |
| `_superseded/Rineya-Nimbus-1.3.0-devfreeze` | 28L Aegis-donor 50/50 | `Rineya-Nimbus-1.6.0-devfreeze` |

Add new cores with `python _materialize_core.py` (refuses if the destination already exists) or by the archive-then-write pattern above.
