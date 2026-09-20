# Composition, decomposed — the three contrasts nobody had paired

**2026-09-20, zero model calls.** All on `select-views-001`, bank `views_sha256 7dd53f2a…`,
digest `a329eb7bb06f`. Computed from the stored per-view `canonical` flags.

COMPOSE-001 and MAG-001 report composition as aggregate differences (−60, +44, −16). The
programme's standing rule is **never quote a pooled net**. Decomposed:

| contrast | scores | gains | losses | net | exact McNemar |
|---|---:|---:|---:|---:|---:|
| T16 → **TH16 raw sum** | 235 → 175 | 9 | **69** | **−60** | 1.38e-12 |
| **TH16 → TH16s** (rescale only) | 175 → 219 | **63** | 19 | **+44** | 1.15e-06 |
| T16 → **TH16s** (composition at matched norm) | 235 → 219 | 11 | 27 | **−16** | **0.01385** |

## Three things the nets were hiding

**1. The residual cost of matched-norm composition is REAL, not noise.** −16 at p = 0.0139 on
38 discordant pairs. This is a **within-draw** contrast — both arms are deterministic transforms
of the same trained deltas — so it is valid at its own discordance and the ~20-row cross-draw
floor does not apply. **Composition at matched magnitude is cheap, not free.**

**2. No contrast here is purely subtractive.** The raw sum still *gains* 9 rows T16 lacked;
matched-norm composition gains 11. Rescaling loses 19 rows that the raw sum had. **Composition
reorders items, it does not merely remove them** — the same shape as `nesting fails 21/21 pairs`
in `per-item-headroom-unaddressable`, where scaling a delta reorders rather than nests.

**3. The rescale's +44 is 63 gains against 19 losses.** Reporting it as "+44" implies a one-way
repair. It is a two-way trade that nets strongly positive.

## What this does NOT establish, and the identity caveat

**The T16 arm used here is `rank001-20260918/arm-r16`, not COMPOSE-001's own `arm-T16`.**
No artifact carries per-view flags for `compose001/arm-T16`, so it cannot be paired directly.

The stand-in is the same construction — P₁₆(δ_T) — and scores **235, matching exactly**. But the
receipts differ in the seventh decimal of the retained-norm fraction:

    compose001 T16      delta_norm_fraction_retained  0.812249925436939
    rank001/arm-r16     delta_norm_fraction_retained  0.8122500916811055

**~1.7e-7 apart — a rebuild, not a byte-identical arm**, the same trap as `arm-rebuild-s10`
against `arm-s10`. The aggregate agreement at 235 is strong circumstantial evidence they behave
identically, and it is **not** a receipt. Treat the −16 as measured against *a* P₁₆(δ_T) at 235,
not against COMPOSE-001's specific bytes.

## Every composition arm's weights are RECLAIMED

| arm | on disk |
|---|---|
| `mag001/arm-TH16s`, `compose001/arm-{TH16,T16,H16}`, `tt001/build-*` | **16 MB sidecars, 0 safetensors** |

So **completing TH16s's blocked battery** (long context, literal `UNKNOWN`,
`references_retained`, gate v2 — all `exit=2` on the host-RAM qualify gate) requires
**rebuilding the arm first**, not merely running probes. That is a materially larger cost than
"the RAM has freed, just run it."

Rebuild inputs that do survive: `compose001-20260918/H` (3 MB — metadata, not weights),
`rrp015-20260917/control-hf` (16 MB sidecar), and `trained_MIX_fp32.safetensors` (1,201 MB).
**Whether δ_T and δ_H themselves are reconstructible on this host is not established here** and
must be checked before anyone costs the battery.
