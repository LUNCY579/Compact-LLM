# Rineya local core roadmap

Short index of **officially recorded local models**. Weights live under `Rineya_Published_Cores/` (gitignored). Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A model is “officially recorded” only after it has a vault directory, a `core_id`, and a `CORE.json`. Adapters, unfinished runs, and eval-only outputs stay in [`lineage.md`](lineage.md) under “not recorded cores”.

Last updated: 2026-08-21.

| File | What it holds |
|---|---|
| [`comparison.md`](comparison.md) | Every eval and performance number for all six named cores. No ranking. |
| [`lineage.md`](lineage.md) | Timeline, spectrum (what each core is), ancestry, donors, and train mixes. |
| [`announcement.md`](announcement.md) | What may be named in public copy, and what must not. |

---

## Recorded cores (vault)

| Recorded (UTC) | Name | Version | Status | Local directory | Weight SHA prefix |
|---|---|---|---|---|---|
| 2026-08-21 01:14:07 | **Pulse** | 1.6.0-devfreeze | roster replace (`late_rineya`); sealed TEST not run | `Rineya_Published_Cores/Rineya-Pulse-1.6.0-devfreeze` | `2357574d…` |
| 2026-08-19 00:25:16 | **Aegis** | 1.1.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | 1.2.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-21 01:14:12 | **Nimbus** | 1.6.0-devfreeze | roster replace (soup-lab `global_best` α=0.40); sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze` | `7b66b8d0…` |
| 2026-08-19 19:54:59 | **Lyra** | 1.4.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | 1.5.0-devfreeze | recorded uncensored pair for Halo | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |

### Superseded (archived under `_superseded/`)

| Archived (UTC) | Name | Version | Local directory | Notes |
|---|---|---|---|---|
| 2026-08-21 | Pulse (old) | 1.0.0-rc1 | `_superseded/Rineya-Pulse-1.0.0-rc1` | Phase 8 commercial 19L teacher; Aegis base |
| 2026-08-21 | Nimbus (old) | 1.3.0-devfreeze | `_superseded/Rineya-Nimbus-1.3.0-devfreeze` | Aegis-donor 50/50 (24/64) |

---

## Next

| When (UTC) | Event | Track |
|---|---|---|
| in progress | Halo-pair layerwise soup (`qwen3_halo_uncensored`) | layer_influence → layer_search → freeze; may beat Nimbus 1.6 global α=0.40 |
| next | Hugging Face upload of Pulse 1.6 + Nimbus 1.6 | see commands in chat / vault README |
| next | Public / website copy for the named family | see [`announcement.md`](announcement.md); **do not** call Pulse 1.6 “Rineya 1.0” |
| next | Optional sealed chat TEST (110) + RAG TEST | `not_run`; not blocking announcement |
| next | DPO on Aegis lineage | `not_run` |
| later | Student cores | not a publish candidate yet |

---

## How to add the next recorded core

1. Materialize with `Rineya_Published_Cores/_materialize_core.py` (refuses overwrite), **or** archive the old dir under `_superseded/` then write a new versioned directory.
2. Confirm `CORE.json` has `core_id`, timestamps, sources, and weight SHA.
3. Add a row to the vault table above, a full section in [`lineage.md`](lineage.md), and rows in [`comparison.md`](comparison.md) for every eval that was run.
4. Update [`announcement.md`](announcement.md) only if the core may be named in public copy.
5. Leave unfinished adapters in [`lineage.md`](lineage.md) “not recorded cores” until merge + `CORE.json`.
