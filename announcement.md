# Public announcement board

What may be named in public copy, and what must not. Numbers for review are in [`comparison.md`](comparison.md). How each core was built is in [`lineage.md`](lineage.md). Index: [`roadmap.md`](roadmap.md).

Last updated: 2026-08-19.

---

## Status

Human DEV chat review on 2026-08-19: auto Phase 1 scores are **reference only**. Replies from Nimbus / Halo / Lyra are coherent, short, and usable. Gate **passed**. These four cores (plus Pulse, already 1.0) may be named in public copy. **Rineya 1.0 commercial claim stays on Pulse.** Sealed chat/RAG TEST is still `not_run` and does **not** block this announcement.

| Name | Role to announce | Status | Next dated item |
|---|---|---|---|
| **Pulse** | Commercial 1.0 short-chat teacher (19L) | Phase 8 human PASS (2026-08-05) | Website 1.0 copy as today |
| **Aegis** | 19L dual-skill (chat + grounded RAG) | `public_announcement_ready` | Optional sealed TEST; DPO not started |
| **Halo** | 28L family; strongest frozen long-context among 50/50 (30/64) | `public_announcement_ready` | Optional sealed TEST |
| **Halo Uncensored** | Halo donor α=1.0 replace; uncensored pair (31/64; oracle abstain 0.00) | `recorded_uncensored_pair` | Censored vs uncensored experiments vs Halo |
| **Nimbus** | 28L family; Aegis donor; strongest 28L RAG of the three | `public_announcement_ready` | Optional sealed TEST |
| **Lyra** | 28L family; chat auto-Q highest of the three (0.867) | `public_announcement_ready` | Optional sealed TEST |

Do **not** announce Halo Uncensored as dual-skill or as Rineya 1.0. Pair it with **Halo** for censored vs uncensored work. Do not announce unfinished student runs or raw v2 adapters.

**How to talk about the 28L family.** Chat quality matches 19L Aegis (~0.86 auto; human pass). Long-context showcase beats official Qwen3-0.6B (14/64) and 19L Aegis (4/64). Grounded RAG is still weaker than 19L Aegis (Nimbus best of the three: oracle 0.891 / real 0.674 vs Aegis 1.0 / 0.736). Keep Aegis as the 19L RAG specialist in public copy.

---

## How to read auto scores (DEV chat)

Read-only audit of the 50/50 chat DEV runs (510 rows each): hard failures **0**, intelligibility **1.0**, safety_boundary **1.0**. Low auto `task_adequacy` (~0.50) is scorer prompt-mode mismatch (generic acknowledgments vs a canned reference), not incoherence. A few fluent replies are world-knowledge-wrong (wrong DPO expansions) while still scoring ~0.72 — the scorer does not check that.

Do not retune Aegis’s frozen `5e-4` winner after seeing sealed TEST.
