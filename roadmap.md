# Rineya local core roadmap

Short index of **officially recorded local models**. Weights live under `Rineya_Published_Cores/` (gitignored). Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A model is “officially recorded” only after it has a vault directory, a `core_id`, and a `CORE.json`. Adapters, unfinished runs, and eval-only outputs stay in [`lineage.md`](lineage.md) under “not recorded cores”.

Last updated: 2026-08-19.

| File | What it holds |
|---|---|
| [`comparison.md`](comparison.md) | Every eval and performance number for all six named cores. No ranking. |
| [`lineage.md`](lineage.md) | Timeline, spectrum (what each core is), ancestry, donors, and train mixes. |
| [`announcement.md`](announcement.md) | What may be named in public copy, and what must not. |

---

## Recorded cores (vault)

| Recorded (UTC) | Name | Version | Status | Local directory | Weight SHA prefix |
|---|---|---|---|---|---|
| 2026-08-19 00:23:36 | **Pulse** | 1.0.0-rc1 | Phase 8 human commercial PASS | `Rineya_Published_Cores/Rineya-Pulse-1.0.0-rc1` | `58ce704c…` |
| 2026-08-19 00:25:16 | **Aegis** | 1.1.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | 1.2.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-19 19:54:38 | **Nimbus** | 1.3.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.3.0-devfreeze` | `7ae45098…` |
| 2026-08-19 19:54:59 | **Lyra** | 1.4.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | 1.5.0-devfreeze | recorded uncensored pair for Halo | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |

---

## Next

| When (UTC) | Event | Track |
|---|---|---|
| next | Public / website copy for the named family (Pulse 1.0 + four others) | ready to write; see [`announcement.md`](announcement.md) |
| next | Optional sealed chat TEST (110) + RAG TEST | `not_run`; not blocking announcement |
| next | DPO on Pulse/Aegis lineage | `not_run` |
| later | Student cores | not a publish candidate yet |

---

## How to add the next recorded core

1. Materialize with `Rineya_Published_Cores/_materialize_core.py` (refuses overwrite).
2. Confirm `CORE.json` has `core_id`, timestamps, base path, adapter path, and weight SHA.
3. Add a row to the vault table above, a full section in [`lineage.md`](lineage.md) (base + every train file), and rows in [`comparison.md`](comparison.md) for every eval that was run.
4. Update [`announcement.md`](announcement.md) only if the core may be named in public copy.
5. Leave unfinished adapters in [`lineage.md`](lineage.md) “not recorded cores” until merge + `CORE.json`.
