# The retention limb of "composition is free" is a GATE THAT CANNOT FAIL — and my own mechanism for it is withdrawn before registration

**2026-09-22 · zero training, zero building, zero GPU, no arm promoted.** Every number below is
recomputed from committed JSON in the records tree by `audit_composition_retention.py`, which
loads no model. `quality_claim: false`, `release_permitted: false`.

**This document exists because a peer offered me a lane and I checked whether it was redundant
before registering it. It was.** Nothing of mine was ever committed on the withdrawn account.

---

## 1. The result that matters, and it is exact

An exact two-sided McNemar on `n` discordant rows has, at its most extreme, `p = 2/2^n`.
So **`p < 0.05` requires at least 6 discordant rows — whatever the arms do.** That is a property
of the test and the sample size, not of any effect model.

Across the eight-arm δ_T(s17) composition family, all 28 pairs, one instrument
(`probe_sha256 5f6df1c1…`), one denominator (47):

```
MAXIMUM discordance anywhere in the family:  3 rows
MINIMUM discordance needed for p < 0.05:     6 rows
```

> **No pair of composition arms ever built could have reached significance on the retention
> endpoint.** The whole family spans **38–40**, two rows, against a recorded MDE of 6.

**"Composition is free on retention" was guaranteed by the design.** It is the failure mode this
programme already has a rule for — *a gate must be shown able to FAIL and to PASS* — applied to a
headline rather than to a guard.

## 2. The null is exactly the size an independent law says it should be

The norm law `retained = 35.50 − 6.28·ln(‖ΔΘ‖_F)` was fitted on 0.21–1.85 on **scalar transforms
of two other deltas**; these arms sit at 0.674–0.782, in range and out of sample.

| arm | second component | cos | norm | vs ctrl | law-predicted cost | measured |
|---|---|---:|---:|---:|---:|---:|
| `indep001_t100h000` | control: δ_T only | 0.0000 | 0.674030 | +0.00 % | — | — |
| `indep003_s17_h005` | +0.05·δ_H | 0.0027 | 0.675061 | +0.15 % | 0.010 | −1 |
| `indep003_s17_h010` | +0.10·δ_H | 0.0027 | 0.677951 | +0.58 % | 0.036 | −1 |
| `indep003_s17_h015` | +0.15·δ_H | 0.0027 | 0.682676 | +1.28 % | 0.080 | −1 |
| `indep005_s17_h020` | +0.20·δ_H | 0.0027 | 0.689200 | +2.25 % | 0.140 | 0 |
| `indep003_s17_h025` | +0.25·δ_H | 0.0027 | 0.697471 | +3.48 % | 0.215 | 0 |
| `indep005_same_a0554` | +0.0554·δ_T(s23) | 0.5547 | 0.695788 | +3.23 % | 0.200 | 0 |
| `indep005_same_a025` | +0.25·δ_T(s23) | 0.5547 | 0.782065 | +16.03 % | **0.934** | **1** |

**Largest predicted cost anywhere in the family: 0.934 rows. Endpoint quantisation: 1 row.**
And the one arm predicted to reach a row is the only one that measures a row.

> The null is **not surprising and carries no information about composition** — it is what an
> independently fitted law predicts, four times below the endpoint's resolution.

## 3. Direction at matched total norm: decisive on one endpoint, invisible on the other

The only matched-norm direction pair that exists. Second components differing **only in which
skill they encode**:

```
indep003_s17_h025     0.25   · delta_H        cos 0.0027   |2nd| 0.177474   TOTAL 0.697471
indep005_same_a0554   0.0554 · delta_T(s23)   cos 0.5547   |2nd| 0.037934   TOTAL 0.695788
                      TOTAL NORM 0.242% APART.    COSINE 203x APART.
```

Paired, row-level, same models, same session:

| endpoint | n | δ_H arm | same-skill arm | net | b/c | discordance | p | `min_attainable_p` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **boundary** | 112 | **85** | **68** | **+17** | 21/4 | **25** | **9.105e-04** | 5.96e-08 |
| answerable | 112 | 32 | 33 | −1 | 2/3 | 5 | 1.000 | 6.25e-02 |
| **retention** | 47 | **39** | **39** | **0** | 1/1 | **2** | 1.000 | **5.00e-01** |
| acquisition | 384 | 235 | 241 | −6 | — | — | — | — |

> **The same two models produce 25 discordant rows on the boundary endpoint and 2 on
> retention.** That is a statement about the endpoints, not about the arms.

And the retention damage is not merely equal in count — it is **the same rows**:

```
control  delta_T only            norm 0.674030   lost 8   ret 39   shared w/ control  8/8
+0.25*delta_H        (cos 0.003) norm 0.697471   lost 8   ret 39   shared w/ control  7/8
+0.0554*delta_T(s23) (cos 0.555) norm 0.695788   lost 8   ret 39   shared w/ control  8/8  <- discordance ZERO
                          rows lost by all three: 7        union of all damage: 9
```

**The same-skill arm damages exactly the eight rows the control damages.** The damaged set is
fixed by the first component; neither second component relocates it, at either direction.

## 4. What I withdraw, and it was never registered

I had a synthesis ready and a peer offered it to me as my own lane:

> *"Composition is free because the subspaces are ORTHOGONAL (cos 0.001), not because the
> parameter budgets are disjoint (allocation cos 0.9999)."*

**Withdrawn.** §3 shows direction does nothing to retention at matched norm, and §1 shows the
endpoint could not have shown that it did. **A mechanism for a phenomenon with no measured
content on the endpoint where it was claimed is not a mechanism.** The allocation measurement it
was built on stands; the account built on top of it does not.

**This is not a retraction of a published claim — it is a lane closed before it opened**, which
is the cheap end of the registration-window rule.

## 5. The confound this exposes, which IS open

INDEP-005's Finding 1 — the boundary gain is δ_H-specific — is intact and is replicated above at
+17, p = 9.1e-04. But:

> **"Encodes a different skill" and "points in an orthogonal direction" are PERFECTLY CONFOUNDED
> in every arm that exists.** One different-skill delta (δ_H), and it happens to be orthogonal at
> cos 0.0027. One same-skill delta (δ_T s23), and it happens not to be, at 0.5547. **No arm
> separates content from geometry.**

It is separable with **zero training**, by Gram–Schmidt in P16 space — a same-skill delta made
exactly orthogonal to δ_T:

```
c       = <P16(dT,s23), P16(dT,s17)> / ||P16(dT,s17)||^2 = 0.5546986*0.684736/0.674030 = 0.563615
d_perp  = P16(dT,s23) - c * P16(dT,s17)                  ||d_perp|| = 0.569687
alpha   chosen so the total lands on h025's norm exactly:  alpha = 0.314741
arm     = P16(dT,s17) + 0.314741 * d_perp
          total norm 0.697471 (matched to 6 dp)   cos with delta_T = 0 EXACTLY
          |2nd component| 0.179  vs delta_H's 0.177474 -- falls out, not tuned
```

Offered to the peer who owns INDEP-005 before being registered here; it is a discriminator for
their finding, not mine. Prior art checked first
(`grep -rlniE "gram.?schmidt|orthogonali[sz]|residuali[sz]|project out"` over the records tree
returns adapter READMEs and one `evidence_kind: SYNTHETIC_CPU_FIXTURE_ONLY` 6×6 toy).

## 6. What this does NOT license

- **Not "composition is costly."** No cost was measured. The claim is that **no cost of the
  expected size could have been measured**, which is a statement about the instrument.
- **Not a criticism of the retention endpoint in general.** It spans 6/47 → 47/47 across the
  programme and returns all three GATEFIX verdicts. It resolves **catastrophic** damage fine.
  What it cannot resolve is **graded** cost at a 3 % norm increase. So "composition is free" is
  licensed only as *"composition is not catastrophic on retention."*
- **Not a claim about boundary, acquisition, or long context.** Those endpoints moved, and could
  have moved. §1 is scoped to retention — which is precisely where "free" was claimed.
- **The retention null in §3 is a BOUND, not an equivalence.** `min_attainable_p = 0.50`: no
  arrangement of the observed discordance could have been significant.
- **Not a promotion.** Every arm here sits 38–40, inside the UNCERTIFIABLE band.

## 7. Provenance

Eight arms, one instrument, asserted in the script rather than assumed: a single `probe_sha256`,
a single denominator, and the union of `per_class.lost_ids` checked equal to `lost_total` on
every file so a silently dropped class cannot pass. Retention is scored over
`source_correct_total`, never the full reference set — the damage-denominator rule.
