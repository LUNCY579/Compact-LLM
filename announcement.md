# Public announcement board

What may be named in public copy, and what must not. Numbers for review are in [`comparison.md`](comparison.md). How each core was built is in [`lineage.md`](lineage.md). Index: [`roadmap.md`](roadmap.md).

Last updated: 2026-08-27.

---

## Status

Roster refresh 2026-08-21: **Pulse** and **Nimbus** names kept; weights replaced.

| Name | Role to announce | Status | Next dated item |
|---|---|---|---|
| **Pulse** | 28L family; Halo-donor `late_rineya` α-shape (31/64 showcase) | `dev_freeze` (1.6.0); sealed TEST not run | Optional sealed TEST; do **not** call this commercial 1.0 |
| **Aegis** | 19L dual-skill (chat + grounded RAG) | `public_announcement_ready` | Optional sealed TEST; DPO not started |
| **Halo** | 28L family; strong frozen long-context among 50/50 (30/64) | `public_announcement_ready` | Optional sealed TEST |
| **Halo Uncensored** | Halo donor α=1.0 replace; uncensored pair (31/64; oracle abstain 0.00) | `recorded_uncensored_pair` | Censored vs uncensored experiments vs Halo |
| **Nimbus** | 28L family; soup-lab global α=0.40 vs Halo Uncensored (29/64) | `dev_freeze` (1.6.0); sealed TEST not run | Optional sealed TEST; may revise after layer_search freeze |
| **Lyra** | 28L family; chat auto-Q highest of the older 50/50 set (0.867) | `public_announcement_ready` | Optional sealed TEST |
| **Supa 2.0** | Nine-arm answer-blind ordered consensus system; **48/64**, 128/192, 176/256 | `published_system48`; sealed TEST not run | Call it a routed family system, never one merged checkpoint |
| **Baya 2.0** | Three-arm Nimbus/Pulse UNKNOWN system; **46/64**, 125/192, 171/256 | `published_system46`; sealed TEST not run | Recommended efficiency/performance system; three model executions |
| Nimbus DualScale PEFT Gate45 | Two PEFT scale executions; 45/64, 124/192, 169/256 | `published_system45` | Do not inherit this score to dense Supa Core 1.7 |
| Supa-Baya Dense Gate42 | Two dense core executions; 42/64, 124/192, 166/256 | `published_system42` | Distinct from the coincidental 42-point single PEFT arm |

**Commercial 1.0.** The Phase 8 human PASS teacher is **archived** at `_superseded/Rineya-Pulse-1.0.0-rc1`. Do not advertise active Pulse 1.6.0 as Rineya 1.0. Prefer Aegis for 19L dual-skill public copy until a new 1.0 claim is decided.

Do **not** announce Halo Uncensored as dual-skill or as Rineya 1.0. Pair it with **Halo** for censored vs uncensored work. Do not announce unfinished student runs or raw v2 adapters.

**How to talk about the 28L family.** Chat quality stays ~0.86 auto. Long-context showcase: Pulse 31 / Halo Uncensored 31 / Halo 30 / Nimbus 29 / Lyra 28 (vs official Qwen3-0.6B 14 and 19L Aegis 4). Grounded RAG is still weaker than 19L Aegis. Keep Aegis as the 19L RAG specialist in public copy.

**2026-08-27 frontier publication.** Supa 2.0 names the published 48/64
nine-arm system; Baya 2.0 names the published 46/64 three-arm system. The 45
PEFT and 42 dense gates are published as explicit lower-cost systems. Supa Core
1.7 and Baya Core 1.7 remain immutable dense 40/64 dependency anchors. Always
state that Showcase was selection data and that these system scores require
multiple inference arms.

---

## How to read auto scores (DEV chat)

Hard failures **0** and intelligibility **1.0** on the dual-skill chat DEV protocol for the current 28L set. Low auto `task_adequacy` (~0.50) is largely scorer prompt-mode mismatch, not incoherence.

Do not retune Aegis’s frozen `5e-4` winner after seeing sealed TEST.
