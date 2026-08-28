# Rineya local core roadmap

Short index of **officially recorded local models**. Weights live under `Rineya_Published_Cores/` (gitignored). Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A dense model is recorded by `CORE.json`; a routed family system is
recorded by `SYSTEM.json`, bundled runtime assets, and `SYSTEM_CATALOG.json`.

Last updated: 2026-08-27.

| File | What it holds |
|---|---|
| [`Rineya_Published_Cores/MODEL_REGISTRY.md`](Rineya_Published_Cores/MODEL_REGISTRY.md) | Every weight artifact under its **original working label**, load-bearing adapters, arm recipes, and per-item shippability. |
| [`comparison.md`](comparison.md) | Every eval and performance number for eight dense cores and four routed systems. |
| [`lineage.md`](lineage.md) | Timeline, spectrum (what each core is), ancestry, donors, and train mixes. |
| [`announcement.md`](announcement.md) | What may be named in public copy, and what must not. |

---

## Recorded frontier systems (vault)

| Recorded (UTC) | Name | Form | Local directory | Demo 64/192/256 |
|---|---|---|---|---:|
| 2026-08-27 19:59:37 | **Supa 2.0 System48** | 9-arm cascade | `Rineya_Published_Cores/systems/Rineya-Supa-2.0.0-system48` | **48 / 128 / 176** |
| 2026-08-27 19:59:37 | **Baya 2.0 System46** | 3-arm cascade | `Rineya_Published_Cores/systems/Rineya-Baya-2.0.0-system46` | **46 / 125 / 171** |
| 2026-08-27 19:59:37 | Nimbus DualScale PEFT Gate45 | 2 PEFT arms | `Rineya_Published_Cores/systems/Rineya-Nimbus-DualScale-1.0.0-system45` | **45 / 124 / 169** |
| 2026-08-27 19:59:37 | Supa-Baya Dense Gate42 | 2 dense cores | `Rineya_Published_Cores/systems/Rineya-Supa-Baya-DenseGate-1.0.0-system42` | **42 / 124 / 166** |

These are inference systems, not one-weight merges. Supa Core 1.7 and Baya Core
1.7 below remain the immutable dense dependencies that individually score 40.

## Abstention-gated single-pass results

Source evidence remains under `artifacts/rineya_abstention_gate_v1/` (gitignored)
and the baked checkpoint evidence is mirrored into the Supa-A1 vault core. Base arm is
`nimbus_low` = Nimbus 1.6 + step200 @ 0.8125, whose dense form is Supa Core 1.7.
**Showcase64 was the threshold-tuning split. S4 was added after the preregistered
S1 Stability run, so Stability192 is previously inspected validation for Supa-A1.**

| Working label | Form | Passes | Cores | showcase64 | stability192 | p vs `nimbus_low` |
|---|---|---:|---:|---:|---:|---:|
| `nimbus_low` (reference) | base + adapter | 1 | 1 | 42 | 121 | — |
| **S1 gate** | processor, normalised | **1** | **1** | **47** | **130** | **0.0490** |
| **Supa-A1** (formerly Rineya-Nimbus-Abstain-1.0) | dense + baked logit bias | **1** | **1** | 43 | **125** | 0.2188 |
| Rineya-Nimbus-Dense-1.0-nobias | dense merge, Supa-A1 ablation control | 1 | 1 | 40 | 121 | 1.0000 |

Two things follow, and they change the roster:

- **S1 is the strongest preregistered primary gate** (130 vs System48's 128).
  Exploratory S2 reaches 131 but was not primary. S1's preregistered row-level
  p=0.0490 is marginal and does not
  establish superiority over System48; the 48/64 System48 headline is a
  tuning-split number.
- **Supa-A1 ties the three-pass System46** (125 = 125) at one
  third the inference cost, in one directory, with weights **byte-identical to
  Supa Core 1.7** (`962abc5b…`).

Consequence: System46, Gate45 and DenseGate42 are performance/compute dominated
under this runtime, but remain useful compatibility and routing references.
System48 survives as a research upper bound and distillation target.
See [`MODEL_REGISTRY.md` §6](Rineya_Published_Cores/MODEL_REGISTRY.md) for why the
cascades can never be collapsed into one checkpoint, and why S1 can ship as a model
class but not as weights.

## Recorded dense cores (vault)

| Recorded (UTC) | Name | Version | Status | Local directory | Weight SHA prefix |
|---|---|---|---|---|---|
| 2026-08-21 01:14:07 | **Pulse** | 1.6.0-devfreeze | roster replace (`late_rineya`); sealed TEST not run | `Rineya_Published_Cores/Rineya-Pulse-1.6.0-devfreeze` | `2357574d…` |
| 2026-08-19 00:25:16 | **Aegis** | 1.1.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | 1.2.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-21 01:14:12 | **Nimbus / Nimbus-A1** | 1.6.0-devfreeze | roster core; A1 tracking alias (soup-lab `global_best` α=0.40); sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze` | `7b66b8d0…` |
| 2026-08-19 19:54:59 | **Lyra** | 1.4.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | 1.5.0-devfreeze | recorded uncensored pair for Halo | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |
| 2026-08-26 23:47:23 | **Supa** | 1.7.0-devfreeze | Showcase-selected, Stability-validated low-scale long-context arm | `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze` | `962abc5b…` |
| 2026-08-26 23:47:26 | **Baya** | 1.7.0-devfreeze | Showcase-selected, Stability-validated high-scale abstention arm | `Rineya_Published_Cores/Rineya-Baya-1.7.0-devfreeze` | `1737cd1e…` |
| 2026-08-27 | **Supa-A1** | 1.0.0-devfreeze / tracking A1 | promoted single-pass baked-abstention core; custom code; sealed TEST not run | `Rineya_Published_Cores/Rineya-Supa-A1-1.0.0-devfreeze` | `962abc5b…` |

### Superseded (archived under `_superseded/`)

| Archived (UTC) | Name | Version | Local directory | Notes |
|---|---|---|---|---|
| 2026-08-21 | Pulse (old) | 1.0.0-rc1 | `_superseded/Rineya-Pulse-1.0.0-rc1` | Phase 8 commercial 19L teacher; Aegis base |
| 2026-08-21 | Nimbus (old) | 1.3.0-devfreeze | `_superseded/Rineya-Nimbus-1.3.0-devfreeze` | Aegis-donor 50/50 (24/64) |

---

## Next

| When (UTC) | Event | Track |
|---|---|---|
| complete | Supa/Baya dense paired publication + exhaustive family search | See `Rineya_Long_Context_Weight_Geometry_Lab_v1/PUBLISHED_FAMILY_SEARCH_REPORT.md` |
| complete | Hugging Face backup of System48/46/45/42 | private main commit `548fdc6b99205aca09326652d5ac0009fe860707` |
| complete | Private Hugging Face backup of Supa-A1 and refreshed family metadata | commit `b14103c71c240998171694a0d0397bd4e5d64d71`; model, evidence, manifests, and root custom-code module included |
| complete | Abstention gate lab; single-pass 130/192 and a baked 125/192 checkpoint | `artifacts/rineya_abstention_gate_v1/` |
| **next** | **Create new pack-disjoint calibration and untouched final splits** | Highest-value open item. Four biases tie on Showcase but differ by 4 points on inspected Stability; select on the new calibration split and touch the new final test once |
| complete | Promote `Rineya-Nimbus-Abstain-1.0` as **Supa-A1** | Self-contained core, sanitized provenance, frozen evidence, transactional verifier, and smoke-tested custom-class load |
| complete | Replicate the Tier 2 adapters (`step200` above all) | Local artifacts remain gitignored; adapter weights are backed up in the private Hugging Face system bundles |
| complete | Reconcile Halo/Nimbus current-protocol showcase metadata | Live roster now uses standardized vault reruns: Halo **29** / Nimbus **30**. Historical source runs were valid at 30/29 and remain labeled as such; this was protocol drift, not a transposition |
| next | Reassess System46 / Gate45 / DenseGate42 after a deployable S1-equivalent core exists | Supa-A1 is cheaper but does not dominate every Showcase/compatibility dimension; retire routed references only after a verified replacement |
| next | Public / website copy for the named family | see [`announcement.md`](announcement.md); **do not** call Pulse 1.6 “Rineya 1.0” |
| next | Optional sealed chat TEST (110) + RAG TEST | `not_run`; still the only untouched split — required before any public score claim |
| next | DPO on Aegis lineage | `not_run` |
| later | Distil System48 into one dense model | Only honest route to collapse a cascade; low expected payoff (its +6 is p=0.109 and a 1-pass gate already exceeds it) |
| later | Student cores | not a publish candidate yet |

---

## How to add the next recorded core

1. Materialize with `Rineya_Published_Cores/_materialize_core.py` (refuses overwrite), **or** archive the old dir under `_superseded/` then write a new versioned directory.
2. Confirm `CORE.json` has `core_id`, timestamps, sources, and weight SHA.
3. Add a row to the vault table above, a row in [`MODEL_REGISTRY.md`](Rineya_Published_Cores/MODEL_REGISTRY.md) under its working label, a full section in [`lineage.md`](lineage.md), and rows in [`comparison.md`](comparison.md) for every eval that was run.
4. Update [`announcement.md`](announcement.md) only if the core may be named in public copy.
5. Leave unfinished adapters in [`lineage.md`](lineage.md) “not recorded cores” until merge + `CORE.json`.
