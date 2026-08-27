# Rineya Published Cores

Developer ledger: [`../roadmap.md`](../roadmap.md) (index), [`../lineage.md`](../lineage.md) (recipes and dates), [`../comparison.md`](../comparison.md) (evals), [`../announcement.md`](../announcement.md) (public copy).

Immutable vault for dense Hugging Face checkpoints (`model.safetensors`) and
versioned routed systems under [`systems/`](systems/README.md).

It does **not** replace official phase artifacts. Sources under
`short-response-model-phase-0-1-2/artifacts/` are never modified.

## Policy

- Never overwrite a core directory that already exists.
- To replace a named roster slot, archive the old directory under `_superseded/` first, then write a new versioned directory.
- Never write back into phase `artifacts/`.
- Each core has a stable `core_id` and a `CORE.json` provenance file.
- Weights stay local (gitignored). Provenance files may be committed.

## Published frontier systems

| Name | Directory | Form | Showcase | Stability | All 256 |
|---|---|---|---:|---:|---:|
| **Supa 2.0 System48** | `systems/Rineya-Supa-2.0.0-system48` | 9-arm ordered consensus cascade | **48/64** | **128/192** | **176/256** |
| **Baya 2.0 System46** | `systems/Rineya-Baya-2.0.0-system46` | 3-arm UNKNOWN cascade | **46/64** | **125/192** | **171/256** |
| Nimbus DualScale PEFT Gate45 | `systems/Rineya-Nimbus-DualScale-1.0.0-system45` | 2 PEFT executions | **45/64** | **124/192** | **169/256** |
| Supa-Baya Dense Gate42 | `systems/Rineya-Supa-Baya-DenseGate-1.0.0-system42` | 2 dense-core executions | **42/64** | **124/192** | **166/256** |

Supa and Baya refer to the 48/46 **systems** at the family frontier. Their
immutable 1.7 dense cores remain below as 40/64 dependency anchors. A nonlinear
prediction cascade cannot be honestly collapsed into one `model.safetensors`;
each system therefore ships `SYSTEM.json`, exact adapters, routing code, hashes,
and frozen replay evidence.

## Active dense cores

| Directory | core_id | Kind |
|---|---|---|
| `Rineya-Pulse-1.6.0-devfreeze` | `rineya.core.pulse.v1.6.0-devfreeze.28L.qwen3-late-rineya-alpha-shape` | 28L late_rineya α-shape (Halo donor). Roster name **Pulse** |
| `Rineya-Aegis-1.1.0-devfreeze` | `rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4` | LoRA merge of teacher dual-SFT v1 `5e-4` into archived Pulse 1.0 base |
| `Rineya-Halo-1.2.0-devfreeze` | `rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v2@1e-4 |
| `Rineya-Nimbus-1.6.0-devfreeze` | `rineya.core.nimbus.v1.6.0-devfreeze.28L.qwen3-x-halo-uncensored-alpha-0.40` | 28L soup-lab global_best α=0.40. Roster name **Nimbus** |
| `Rineya-Lyra-1.4.0-devfreeze` | `rineya.core.lyra.v1.4.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-1e5` | 28-layer Qwen3; 19 kept blocks 50/50 with v1@5e-4 + v1@1e-5 |
| `Rineya-Halo-Uncensored-1.5.0-devfreeze` | `rineya.core.halo-uncensored.v1.5.0-devfreeze.28L.qwen3-replace-v1-5e4-v2-1e4` | Same Halo donor; kept blocks fully replaced (α=1.0). Uncensored pair for Halo |
| `Rineya-Supa-1.7.0-devfreeze` | `rineya.core.supa.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125` | **Supa Core 1.7**, dense 40/64 low-scale dependency arm |
| `Rineya-Baya-1.7.0-devfreeze` | `rineya.core.baya.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-1p25` | **Baya Core 1.7**, dense 40/64 high-scale dependency arm |

**Rename aliases (2026-08-26).** Supa was formerly Taria (`rineya.core.taria.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125`); Baya was formerly Lumina (`rineya.core.lumina.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-1p25`). Weights and frozen evidence are unchanged.

## Superseded (archived, not deleted)

| Directory | Was | Replaced by |
|---|---|---|
| `_superseded/Rineya-Pulse-1.0.0-rc1` | Phase 8 commercial 19L teacher | `Rineya-Pulse-1.6.0-devfreeze` |
| `_superseded/Rineya-Nimbus-1.3.0-devfreeze` | 28L Aegis-donor 50/50 | `Rineya-Nimbus-1.6.0-devfreeze` |

Add new cores with `python _materialize_core.py` (refuses if the destination already exists) or by the archive-then-write pattern above.

Supa and Baya were originally published under their former names with the historical `_publish_taria_lumina.py`, which requires pinned source hashes, BF16 tensor inventory checks, a complete 256-row parity packet, pair promotion, and catalog preservation. The legacy `_materialize_core.py` does not know the current 1.6/1.7 roster and must not be used to rebuild the catalog.

Publish or verify the four routed frontier bundles with
`python _publish_family_frontier.py preflight|apply|verify`. Dense cores are not
overwritten by that transaction.

Private Hugging Face backup: `Luncy1978/rineya-published-cores` main commit
`548fdc6b99205aca09326652d5ac0009fe860707` (systems, catalogs, adapters,
evidence, and runtime verified byte-for-byte after upload).
