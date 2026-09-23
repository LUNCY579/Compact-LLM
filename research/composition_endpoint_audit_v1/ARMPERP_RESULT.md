# ARMPERP-001 RESULT — orthogonality does not manufacture the boundary gain, and the registered outcome table had no cell for what happened

**2026-09-23 · one arm, zero training, zero budget grant · 3 registered legs, all scored.**
`quality_claim: false`, `release_permitted: false`. **No arm promoted** — arm-perp sits in the
UNCERTIFIABLE band like every arm in this family.

Registered in `ARMPERP_PREREG.md` (`b7a179ed`) with **zero artifacts on disk**, verified at that
commit. Built `ffbea630`. Independently scored **cold** by a peer session (`feca0e7`, committed
before they saw any reading of mine) whose extraction reproduces every comparator count.

---

## 1. The registered outcome table FAILED, in two independent ways

**Neither defect is about the data. Both are about the table.**

**(a) No cell fits.** CONTENT requires *"NOT significantly different from a0554"*. That limb had
**discordance 2**, whose minimum attainable two-sided p is **0.50** — it could not have been
significant whatever the rows did. P2 was registered precisely so a sub-6 discordance could not
be read as agreement. My scorer implemented the clause as `not sig(a)`, which is **TRUE for an
unresolved comparison**, and printed CONTENT. The guard existed, fired correctly in the per-row
table, and the verdict logic negated straight past it. Caught by the peer's cold scoring; fixed
in `c513625c`.

**(b) Two of the four outcomes were never separable.** CONTENT's reference is `a0554` (68);
INERT's is the control (69). They differ by **one row at discordance 1**. A textbook-clean
CONTENT could not have excluded INERT.

> **The four-outcome table assumed both limbs would be readable and that its reference points
> were distinguishable. Neither held.** Registered outcome: **a fifth state** — one limb
> resolved, one unresolved.

## 2. What IS certified — one cell in twelve

```
h025 vs control   boundary   85 vs 69   +16   nd=24   p=1.544e-03   FLOORLESS
```

Floorless because **δ_H is a different GENERATOR, not a redraw** — both arms carry `δ_T(s17)`
only, so no draw is spanned. The `|net| ≥ 20` floor is a **between-draw** convention, not an
acquisition-endpoint one. *(That correction is the peer's; my original justification — "the
floor is an acquisition convention" — reached the right answer by the wrong route.)*

**δ_H buys boundary. That is the only certified positive in the whole twelve-cell set.**

## 3. What the BOUND licenses, where a null could not

Four separate times in this lane an **unresolved** cell was read as a null — my scorer's control
flow, a peer's "mildly negative", my "behaves like same-skill", and my attempted rescue of the
conclusion onto `perp vs control` at nd=3. **Same direction every time: an absence of detection
read as a detection of absence.**

A bound is a different object. `|net| ≤ nd`, so an exact Clopper–Pearson limit on the
discordance rate bounds the effect without asserting absence:

| contrast | A | B | b | c | nd | 1-sided 95% | \|net\| ≤ | 2-sided 95% | \|net\| ≤ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perp vs control | 66 | 69 | 0 | 3 | 3 | 0.0678 | **7.59** | 0.0763 | **8.54** |
| perp vs a0554 | 66 | 68 | 0 | 2 | 2 | 0.0551 | 6.18 | 0.0630 | 7.06 |
| a0554 vs control | 68 | 69 | 0 | 1 | 1 | 0.0417 | 4.67 | 0.0487 | 5.46 |
| h025 vs control | 85 | 69 | 20 | 4 | 24 | 0.2879 | 32.25 | 0.3019 | 33.81 |

> **arm-perp's boundary effect vs the control is within ±7.59 rows (one-sided) or ±8.54
> (two-sided). δ_H's certified effect is +16. THE BOUND EXCLUDES IT under either convention.**

**LICENSED:** *this* arm-perp does not have δ_H's boundary effect.
**NOT LICENSED:** that arm-perp has **no** boundary effect — the bound is ~8 rows wide, not
zero. Nor anything about orthogonalised same-skill content **in general**, which quantifies over
draws of `s23`; a different draw gives a different `d_perp` and a different arm. **That** is
where the between-draw floor bites, and no bound on one realisation settles it.

**POST-HOC**, labelled: no equivalence test was registered. Additional descriptive statistics on
the registered data — not a failed primary converted into a passed one.

*(A justification I offered for this bound — "cross-draw noise inflates an observed difference
and cannot shrink one, so a small difference is conservative" — is **WITHDRAWN**. False for a
single realisation: `D = δ + ε` with `E[ε]=0` permits ε to mask δ. Jensen gives `E|D| ≥ |δ|`,
which is about the average over draws, not the one draw in hand. The bound does not need it —
it is a within-pair statement about two objects on disk.)*

## 4. Three endpoints, one pattern

```
endpoint          arm-perp   a0554(same)   control   h025(delta_H)
boundary /112         66          68          69        85
acquisition /384     246         241         235       235
longctx all256       158         158         159       164
longctx showcase64    41          41          41        43
longctx stability    117         117         118       121
```

**arm-perp tracks the same-skill arm on all three endpoints and δ_H on none.** On long context
it is *identical* to `a0554` on three of four measures. Orthogonalising same-skill content did
not turn it into different-skill content.

The one floorless resolved comparison in the set, and the cleanest thing this experiment
produced:

```
perp vs a0554, acquisition:  +5, nd=17, p=0.332   RESOLVED NULL, FLOORLESS
                             (both arms carry s17 AND s23 -- no draw spanned)
```

**Orthogonalising changed nothing detectable about what the `s23` component contributes.**

**Not certifiable:** `perp vs control` on acquisition, **+11 at p=0.0266 but BELOW the 20-view
floor**. So the boundary/acquisition dissociation is **certified on one limb and suggestive on
the other**, and it is deliberately NOT drawn here as a 2×2 — three of four cells cannot carry
weight and the shape would imply they can.

**Long context resolves nothing**: the spread is 158–164 against a recorded 13-row floor, and
`LONGCTX` artifacts carry no per-row list so no paired test is possible. Descriptive, as
registered.

## 5. Gate 7 is inert here, as registered in advance

`d_perp` effective rank is **median 23** against comparators' 16 (bound 32). The prereg fixed
in advance that the rank confound bears on a **GEOMETRY** verdict only — rank explains a **gain**,
not its absence. There is no gain. The asymmetry was written before the score, which is what
makes this clean rather than convenient.

## 6. Surface-convention audit — prompted by another lane, and it passes WITH CONTROLS

A peer lane found that **91 of 96 "lost" rows in one of their arms were the gold answer plus one
trailing period** — a donor's answer format imported at high coefficient and judged by the other
donor's strict scorer. Invisible to every weight-space statistic either of us computes.

Checked here across all four arms. **Zero rows recovered** by any of: strip whitespace, strip one
period, strip all trailing periods, casefold, drop all whitespace. **Zero trailing periods
emitted** in 384×4 predictions.

**And because a zero from an absence check is what a broken check also returns:**

```
POSITIVE CONTROL  25 injected gold+'.' rows  ->  recovered 25/25   the check CAN fire
NEGATIVE CONTROL  25 genuinely-wrong rows    ->  recovered  0/25   no false positives
NAMESPACE         384/384 rows have both prediction and reference non-empty
```

The misses are real arithmetic errors (`gold 106 / gen 110`), not formatting. **This arm set is
clean of that defect.**

## 7. What this does NOT license

- **Not a promotion.** One arm, one base, one seed pair, rank 16, one same-skill draw.
- **Not a general claim about orthogonalisation** — see §3, it quantifies over draws.
- **Not "d_perp is inert".** Refuted on acquisition, which the boundary cluster cannot speak to:
  +11 over control at nd=21. Sized honestly — below the floor, so valid about these arms only.
- **Not a retention result.** Retention was excluded in advance, with the reason: across all 28
  pairs of this family the maximum discordance is 3 against the 6 an exact McNemar needs.
- **"Boundary has no floor" is NOT established** — nobody has measured boundary's between-draw
  spread. An absence, not a demonstrated zero, and two of six contrasts sit in that cell.

## 8. Provenance

`p16()` reused verbatim from `build_indep5.py`. All gates passed, and **gate 5 was a positive
control run before the arm existed**: `c` and `alpha` each perturbed 1% were REJECTED at
2.118e-02 and 4.631e-04 against tolerances 1e-09 and 5e-07. Realised `cos = +4.700e-14`; total
norm `0.697471338175574` vs h025's `0.697471338175503`. All four arms share bank `7dd53f2a`,
`max_new_tokens 96`, digest `a329eb7bb06f`, asserted rather than assumed. Long-context ran from
the qualification checkout because the suite is absent from this worktree — disclosed before any
result existed.
