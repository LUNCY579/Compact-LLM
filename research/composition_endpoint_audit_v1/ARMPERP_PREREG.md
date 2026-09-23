# ARMPERP-001 — does the boundary gain need a DIFFERENT SKILL, or just an ORTHOGONAL DIRECTION?

> **STATUS: REGISTERED, NOTHING BUILT, NOTHING SCORED.** Verified at this line: no arm named
> `perp` exists anywhere in the records tree, and no `HSKILL2_*perp*` artifact exists.

**Class: DISCRIMINATOR. Zero training, zero budget grant, pure tensor arithmetic for the build.**
`quality_claim: false`, `release_permitted: false`. **Promotes nothing** — every arm in this
family sits 38–40 on retention, inside the UNCERTIFIABLE band, and this one is predicted to as
well.

Offered first to the peer who owns INDEP-005, since it is a discriminator for their finding.
**They declined it explicitly** ("taking it because INDEP-005 is my lane would be claiming a
result on the strength of adjacency") and offered to run the build to this prereg without
touching the design. Registered here on that basis.

---

## 1. The confound, which is in every arm that exists

INDEP-005's Finding 1 — *the boundary gain is δ_H-specific* — is solid and I replicated it
independently at **+17, p = 9.105e-04** (`RESULT.md` §3). But:

| second component | encodes | cosine with δ_T | boundary |
|---|---|---:|---:|
| `0.25·δ_H` | a **different** skill | **0.0027** | **85** |
| `0.0554·δ_T(s23)` | the **same** skill | **0.5547** | 68 |
| — (control) | — | — | 69 |

> **"Encodes a different skill" and "points in an orthogonal direction" are PERFECTLY
> CONFOUNDED.** There is one different-skill delta and it happens to be orthogonal. There is one
> same-skill delta and it happens not to be. **No arm separates content from geometry**, so
> INDEP-005's finding is currently compatible with a purely geometric account it never tested.

Why it is not merely academic: if the answer is **geometry**, then a boundary gain can be
manufactured from a *second draw of the corpus you already have*, by orthogonalisation alone,
with no new data and no new training. That is a lever. If the answer is **content**, no amount
of geometry substitutes for a second skill, and the programme stops looking for one.

## 2. The construction — zero training, and the match falls out rather than being tuned

Gram–Schmidt in P16 space, one global scalar, using the flattened-vector inner product that
`delta_cosine` already reports:

```
c        = <P16(dT,s23), P16(dT,s17)> / ||P16(dT,s17)||^2
         = 0.5546986 * 0.684736 / 0.674030            = 0.563615
d_perp   = P16(dT,s23) - c * P16(dT,s17)
||d_perp||^2 = 0.684736^2 - 0.563615^2 * 0.674030^2   = 0.324543
||d_perp||                                            = 0.569687

alpha chosen so the TOTAL lands on indep003_s17_h025's norm exactly:
  0.674030^2 + alpha^2 * 0.324543 = 0.697471^2   ->   alpha = 0.314741

ARM     = theta_supa + P16(dT,s17) + 0.314741 * d_perp
```

| property | `arm-perp` | `indep003_s17_h025` | `indep005_same_a0554` |
|---|---:|---:|---:|
| total ‖ΔΘ‖ | **0.697471** | 0.697471 | 0.695788 |
| cos(2nd component, δ_T) | **0.000000** exact | 0.0027 | 0.5547 |
| ‖2nd component‖ | **0.179** | 0.177474 | 0.037934 |
| skill encoded by 2nd component | **SAME** | different | SAME |

**The second-component magnitude 0.179 is not tuned** — it falls out of matching the total norm
under exact orthogonality, and lands within 1 % of δ_H's 0.177474. That removes the dose
confound that would otherwise apply, and it removes it *by construction* rather than by a check.

## 3. Buildability, verified before registration — not after

This programme has registered an arm that turned out not to be buildable (NORMLINE Amendment 2).
All four inputs confirmed present at this line:

```
int001-20260917/trained_fp32.safetensors            1 258 500 416 B   delta_T seed 17
rep001-20260917/s23/trained_fp32.safetensors        1 258 500 512 B   delta_T seed 23 (SAME skill)
compose001-20260918/trained_H_fp32.safetensors      1 258 500 512 B   delta_H (for the gate only)
Rineya-Supa-1.7.0-devfreeze/model.safetensors       1 192 135 096 B   source
```

`p16()` is reused verbatim from `build_indep5.py` (rank-16 SVD per 2D tensor; **1D tensors pass
through whole**, so the 80 norm tensors are kept, exactly as every other arm in this family).
Both trained checkpoints are tail-20 — **layers 8–27, 220 tensors** — verified by reading their
headers, so `arm-perp` inherits the family's layer coverage and is depth-matched to both
comparators.

## 4. Build gates — every one able to fail, and one that must fail first

1. **`cos(d_perp, P16(δ_T,s17))` measured on the BUILT tensors**, not assumed from the algebra.
   Tolerance `|cos| < 1e-9`. Gram–Schmidt is exact in exact arithmetic; this gate exists because
   it is *not* exact in float, and the residual size is the thing being asserted.
2. **Total norm** lands on `0.697471` to 6 dp against `indep003_s17_h025`'s recorded value.
3. **`‖P16(δ_T,s17)‖` reproduces MAG-001's recorded `0.674029920773733`** — the builder's
   existing joint check on source + checkpoint + tensor mapping + SVD. Refuse on mismatch.
4. **Layer coverage** is exactly 8–27 and the tensor count is 220.
5. **POSITIVE CONTROL, RUN BEFORE ANY CORRECT ARM.** Build two deliberately mutated variants —
   `c` perturbed by 1 % and `alpha` perturbed by 1 % — and require gates 1 and 2 to **REJECT
   both**. A gate that has never rejected anything is indistinguishable from one that passes.
   This is COMPOSE4B-003's gate 6 and it is the reason its arms are trustworthy.
6. **Collision check**: the built weight sha256 must differ from `arm-h025` and `arm-same-a0554`.
   An accidental copy would otherwise reproduce a comparator's score exactly and read as a
   dramatic confirmation.
7. **Effective rank of `d_perp` is MEASURED AND REPORTED at build time**, before any score — see
   §7, it is this design's chief limitation and it must be sized before it can be spun.

## 5. Endpoints — and why retention is NOT one

**PRIMARY: `boundary` (HSKILL2, n = 112, `sensitivity_abstains_on_boundary`).** Exact paired
two-sided McNemar against each comparator on the same 112 rows, Holm–Bonferroni across the two.

**SECONDARY, descriptive, no prediction:** `answerable` (n = 112), acquisition
(`select-views-001`, 384 views, bank `7dd53f2a`, **`max_new_tokens 96`** — the family's budget,
not 128), long context.

> **RETENTION IS DELIBERATELY EXCLUDED, and this is registered so its absence cannot later be
> read as a drop.** `RESULT.md` §1 establishes that across all 28 pairs of this family the
> maximum discordance on the n=47 retention endpoint is **3 rows**, while an exact two-sided
> McNemar needs **≥ 6** to reach p < 0.05. This arm sits at norm 0.697471 — inside that same
> range — so its retention contrast **could not reach significance whatever it scores.**
> Running it would produce a number that licenses nothing. It is not dropped after the fact;
> it is excluded here, before the build, with the reason.

## 6. Registered predictions

### P1 — PRIMARY. The boundary count, read against both comparators.

| outcome | requires | reading |
|---|---|---|
| **GEOMETRY** | significantly **>** `a0554` (68) **and NOT** significantly different from `h025` (85) | Orthogonality buys the boundary; content is irrelevant. **INDEP-005's δ_H-specificity is really a geometry finding wearing a semantic label**, and a boundary gain is manufacturable from an existing corpus. |
| **CONTENT** | **NOT** significantly different from `a0554` (68) **and** significantly **<** `h025` (85) | Orthogonality is necessary bookkeeping only; a genuinely different skill is required. **INDEP-005's Finding 1 strengthens and generalises.** |
| **INTERMEDIATE** | significantly different from **both** | Both factors contribute. Report the point estimate; claim neither account. |
| **INDETERMINATE** | significantly different from **neither** | Underpowered. Report `min_attainable_p` and claim nothing. Registered as a real outcome so it cannot be quietly read as CONTENT. |

**My expectation, stated so it can be wrong: CONTENT.** INDEP-005 shows the boundary *declines*
monotonically with same-skill magnitude (69 → 68 → 64), which suggests same-skill mass is inert
or mildly harmful rather than redundant-but-neutral. **GEOMETRY is the more consequential
outcome and I am predicting against it**, which is the direction that makes this worth running.

### P2 — the resolution check, registered because §5's argument must apply to my own primary.

The `h025`-vs-`a0554` boundary contrast produced **25 discordant rows**. If `arm-perp`'s
discordance against either comparator is **< 6**, that comparison **cannot** reach p < 0.05 and
is reported as INDETERMINATE against it, not as agreement. **Checked and reported before any
verdict is written.**

## 7. What this does NOT license, whatever the numbers

- **`d_perp` IS NOT RANK-16.** A difference of two rank-16 matrices has rank up to 32, so
  `arm-perp`'s second component is higher-rank than every comparator's. **This is the chief
  limitation of the design.** Gate 7 measures the effective rank at build time and reports it
  before any score. If it lands near 32, a GEOMETRY result is confounded with rank and must be
  reported as *"orthogonality or rank"*; if it lands near 16–20 the confound is small and sized.
  **Re-truncating `d_perp` to rank 16 is refused** — it would reintroduce a non-zero cosine and
  destroy the one thing this arm exists to hold exactly.
- **One same-skill draw, one different-skill delta, one base, one seed pair, rank 16.** A single
  discriminator on a single pair of deltas.
- **Not a promotion, and not a retention result.**
- **Not a claim that `d_perp` is meaningful information.** It is the part of a second same-skill
  draw the first draw does not explain. If the arm scores at the control (69), the correct
  reading is that **the residual is inert** and *"new information"* is the wrong frame for both
  accounts — registered here as a real third possibility, not a fallback.
- **Not transferable to the 128-token family.** Every number here is at `max_new_tokens 96`.

## 8. Envelope

One arm, built by pure tensor arithmetic on CPU (no GPU for the build), plus two throwaway
mutated variants for gate 5. Three probe legs (boundary, acquisition, long context), one lease
held for the whole battery, `hostbudget` used unchanged and **not tightened**. No training, no
budget grant, no write to any pinned checkout, source asset or peer tree.
