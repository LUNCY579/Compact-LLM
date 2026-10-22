# INDEP-005 result — the boundary gain IS specific to δ_H, and one of my own generalisations is wrong

**2026-09-21 · 2 arms built, 2 controls reused, 8 legs, `BATTERY_COMPLETE`, zero training ·
registered in `INDEP005_PREREG.md` before any arm was built. All four predictions held.**

`quality_improvement_established` false. `release_permitted` false. **No arm promoted.**

## Canary and the build-time rule

Row counts exact (384/150/256/224); every HSKILL artifact reproduces its own published counts.

**The registered build-time rule resolved in favour of the label.** Arm A's planned α = 0.0554
was solved from COMPOSE-001's *subspace* cosine of 0.60; the realised **flattened-vector cosine
is 0.55470** and the realised premium **+3.23 %** against a planned +3.48 %, inside the ±1 pp
band. **Matched-magnitude label earned.**

Incidental and worth keeping: for **same-skill** deltas the flattened-vector cosine (0.555) is
close to the rank-16 subspace overlap (0.60). For **different-skill** deltas those two quantities
are **0.0027 and 0.151** — off by 56×. Same-skill deltas really are aligned as vectors;
different-skill ones are essentially orthogonal as vectors while sharing some subspace.

## The four arms

| arm | magnitude | acq | panel | ret | long ctx | UNK | **boundary** | answerable | `J` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (δ_T only) | — | 235 | 68 | 39 | 159 | 10 | **69** | 112 | 0.6161 |
| **+ 0.25·δ_H** | +3.5 % | 235 | 73 | 39 | 164 | 11 | **85** | 112 | 0.7589 |
| + 0.0554·δ_T(s23) | **+3.2 %** | **241** | 71 | 39 | 158 | 9 | **68** | 112 | 0.6071 |
| + 0.25·δ_T(s23) | +16.0 % | **245** | 67 | 38 | 153 | 8 | **64** | 112 | 0.5714 |

## Finding 1 — P1: the boundary gain is δ_H-SPECIFIC. INDEP-001's missing control passes.

| second delta added | boundary | paired vs control | nd | exact p |
|---|---:|---:|---:|---:|
| **0.25·δ_H** (+3.5 %) | **85** | **20 / 4** | 24 | **0.0015** |
| 0.0554·δ_T(s23) (+3.2 %) | 68 | 0 / 1 | 1 | 1.0 |
| 0.25·δ_T(s23) (+16.0 %) | 64 | 0 / 5 | 5 | 0.0625 |

> **At matched magnitude, a same-skill delta moves the boundary count by ONE row — downward.
> δ_H moves it by twenty, upward, at p = 0.0015.** Neither same-skill arm comes near the ≥80
> falsifier, and both are below the control.

**This is the control INDEP-001's headline never had.** Had a same-skill delta also driven
boundary to ~85, the "second skill arrives" reading would have collapsed into an artifact of
adding mass, taking INDEP-002's replication and INDEP-004's separability with it. It did not.
**P1 held, and it is the prediction that had the most riding on it.**

Note the direction: boundary *declines* monotonically with same-skill magnitude (69 → 68 → 64),
consistent with more magnitude eroding abstention.

## Finding 2 — P2: same-skill addition IMPROVES acquisition

**241 at matched magnitude (+6) and 245 at +16 % (+10)**, against the control's 235 — the two
highest acquisition figures with a comparability-verified `views_sha256` on this bank. Weight-space
ensembling works at an independent coefficient.

**P3 held too, and the cost is measurable.** The +16 % arm scores long context **153 ≤ 159**, and
paired it is **0 gains / 6 losses, p = 0.031** — significant. Literal `UNKNOWN` falls 10 → 9 → 8.
So same-skill addition buys acquisition and pays for it in magnitude, exactly as INDEP-002 found
for δ_H at +13 %.

**P4 held.** Answerable **112/112 with 0 / 0** on both arms — now **6-for-6** across every composed
arm measured, including two that carry no gate skill at all. That result looks structural.

## Finding 3 — MY OWN GENERALISATION IS WRONG: per-block allocation DOES matter, on acquisition

Found while checking whether 245 was a record. **`mag001_sm05` is `T16prof` at profile exponent
−0.5 — δ_T rank-16 with its per-block magnitude profile reshaped at `ratio_arm_over_T16 = 1.0`,
i.e. total norm 0.674029927, identical to T16 — and it scores 243.** `sm10` (exponent −1.0)
scores 239. `T16flat` scores **184**.

| arm | total norm | acquisition |
|---|---:|---:|
| `T16flat` | 0.674 | **184** |
| `T16` (control) | 0.674 | 235 |
| `sm10` (exp −1.0) | 0.674 | 239 |
| `sm05` (exp −0.5) | 0.674 | **243** |

> **At exactly identical total magnitude, redistributing it across blocks moves acquisition by
> 59 rows.** Allocation is not a second-order effect on this endpoint; it is comparable to the
> whole composition question.

**INDEP-004 Finding 3 states: *"magnitude is the variable; its allocation across tensors is
not."*** That was measured on the **boundary** endpoint, where a global scalar and a per-tensor
`s_t` gave 0 gains / 0 losses. **It does not hold on acquisition and I should not have stated it
generally.** This is precisely the error I flagged in H-ALPHA-001 four experiments ago — *a band
certified on one endpoint does not transfer* — committed by me, in the sentence summarising the
experiment that measured it.

**INDEP-004's P4 result stands as measured** (per-tensor allocation is invisible on the *gate*
endpoint). Its generalisation to "allocation does not matter" is **withdrawn**, and an amendment
is being appended there.

## Comparability: TT16s's 247 is NOT compared

`TT001_RESULT.json` records the construction (`supa + s_t*(P16(δ_T,s17) + P16(δ_T,s23))`) and a
data-section hash, but **carries no `views_sha256`, and no `PROCESS_*` artifact for TT16s exists
in `tt001-20260918/`.** TT-001 also documents a bank-fallback incident in which an arm's 224 was
withdrawn as non-comparable and rescored to 238.

**So I cannot verify that 247 was measured on bank `7dd53f2a`, and I do not compare my 245 to
it.** `BLOCKED_ON_COMPARABILITY` — the validity of that measurement is not in question here,
only the admissibility of the comparison.

## What does NOT follow

- **Not a promotion.** `references_retained` is 38–39 on both new arms, inside the UNCERTIFIABLE
  band against both thresholds.
- **Not "same-skill composition is better."** 241/245 beat the control on acquisition and *lose*
  on long context, panel, boundary and `UNKNOWN`. It is a trade, not a dominance.
- **Not a statement about the optimum same-skill allocation.** Finding 3 shows allocation matters
  on this endpoint, and neither arm here explores it.
- **Not confirmed.** Selection-grade scout; no confirmation budget spent.
- One base, one skill pair, two draws, one truncation rank, single seed per arm.

---

# Amendment 1 — "59 rows" is the distance to a BROKEN arm. The honest figure is +8, and its cost is UNCERTIFIABLE.

**Peer catch, recomputed here from the artifacts rather than accepted. Their boundary numbers
reproduce exactly.**

Finding 3 above reports a "59-row unexplored axis" from allocation at identical norm. That span
is **184 → 243**, and **184 is `T16flat`, a degenerate allocation, not a candidate**. Among the
three non-degenerate profiles the range is **235 → 243, eight rows** — and they are not free:

| arm (all at total norm 0.674029927) | acq | Δacq | boundary | paired vs T16 | verdict |
|---|---:|---:|---:|---|---|
| `T16flat` | 184 | **−51** | 68 | 2/3, nd 5 | UNCERTIFIABLE |
| `T16` | 235 | — | 69 | — | — |
| `sm10` (exp −1.0) | 239 | +4 | 65 | 0/4, nd 4 | UNCERTIFIABLE |
| `sm05` (exp −0.5) | 243 | **+8** | 66 | 0/3, nd 3 | UNCERTIFIABLE |

Answerable is **112/112 with 0/0** for all four.

**So the lead is: "+8 acquisition at identical total norm, against a one-directional but
UNCERTIFIABLE boundary deficit of 3 rows" — not "59 rows free."** The deficits are
0-gains-in-both-cases, which is suggestive of a real direction, and at nd = 3 and nd = 4
**p ≤ 0.05 is unattainable**, so the instrument cannot adjudicate it either way.

> **I applied the band-transfer error to my own correction, one paragraph after naming it.** The
> withdrawal of INDEP-004's generalisation is right; the replacement figure I offered was the
> distance to a degenerate arm, quoted as though it were a range of viable ones.

**What survives, correctly sized:** allocation at fixed norm is a real and unexplored axis on
acquisition, worth **+8 rows** at the best profile measured, and `T16flat`'s −51 shows the
downside is far larger than the upside. **The profile training produces (`T16`, 235) is not the
best available at that norm (`sm05`, 243)** — that claim is unaffected and is the part worth
following up.

**One more thing `T16flat` says on its own:** −51 acquisition with boundary at 2/3, nd 5. A
degenerate allocation destroys the generator's endpoint and leaves the gate endpoint untouched —
a third independent confirmation that allocation effects are acquisition-specific.

---

# Note — this directory suffered a NAME COLLISION and I caused the overwrite

A peer session registered its own **INDEP-005** (α_H = 0.20) into this directory at the same
time. **My three `INDEP005_*.md` writes overwrote theirs.** Evidence was never at risk — the arm
labels differ (`indep005_same_a0554`/`_a025` vs `indep005_s17_h020`) so all twelve artifacts
coexist — but their prereg, result and seal were replaced in place. They recovered from commit
`b6e50cd` as `ALPHA020_*` and re-verified the seal byte-identical.

**Their registration survived because it was sealed by CONTENT HASH. One identified only by path
would have been lost silently.** We adopted hashing to prevent post-hoc editing; it turned out to
defend against a collision nobody had anticipated.

**I had a signal and did not read it.** The editor reported *"has been updated successfully"*
rather than *"File created successfully"* on my prereg write — the tell that the path already
existed. I did not register it.

**Root cause is structural:** three sessions registering into one tree under a sequential
`INDEP-00N` scheme with no allocation mechanism. Both of us took the highest existing number and
added one, and both were right to. Same shape as the mislabelled lease earlier — **a shared
namespace that every layer trusts and no layer validates.** The cheap fix is to claim the
directory **create-only** before registering, so the second writer fails at `mkdir` rather than
silently at write.

Neither experiment is renamed: renaming a sealed registration breaks the path its seal records,
and the collision is better documented than tidied away.

---

# Amendment 2 — the depth-profile "lead" is WITHDRAWN ENTIRELY. The sweep was already run, registered in advance, and closed.

**2026-09-21. I proposed running a depth-profile experiment. It has already been run — in
`MAG001_RESULTS_PARTIAL.md` Amendment 2, the same document I appended my own amendment to hours
earlier.** Its title is *"the depth sweep: an ASYMMETRIC optimum, and no demonstrated gain."*

## What it already established

Four points on the exponent, all at fixed total dose (norm ratios 1.0000000087 / 1.0000000085 /
1.0 / 0.9999999799), with bands registered before either new arm was built:

| `p` | allocation | rel. spread | acq | net | **paired McNemar vs T16** | verdict |
|---:|---|---:|---:|---:|---|---|
| −1.0 | steepen hard | 26.35× | 239 | +4 | 11 lost / 15 gained, **p = 0.557** | not significant |
| **−0.5** | steepen | 11.63× | **243** | **+8** | 5 / 13, **p = 0.0963** | **not significant** |
| 0.0 | **trained (T16)** | 5.13× | 235 | — | — | — |
| +1.0 | flatten | 1.00× | 184 | −51 | 60 / 9, **p = 2.24e-10** | **SIGNIFICANT** |

**And the parameterisation was verified before scoring:** building at `p = 1.0` reproduces
`arm-T16flat`'s data-section hash **exactly**, so the exponent genuinely passes through the two
already-measured points.

## So my "+8 lead" is dead on the correct test

Amendment 1 above rescued the claim down to *"+8 acquisition at identical norm, unexplored."*
**Both halves are wrong.**

- **Not unexplored.** It was swept, with pre-registered bands, and closed.
- **Not +8.** On the **paired exact McNemar** — the correct instrument for algebraic variants of
  one delta scored on identical items, and *more* powerful than REP-001's between-draw floor —
  `p = −0.5` is **p = 0.0963, not significant.** I was comparing aggregate counts against a
  draw-spread floor that does not govern within-pair discordance.

That document also flags its own instrument switch explicitly and shows it is **conservative**
for the result it affects: it turns the nominal +8 into a non-result while making the flatten
finding stronger. Its author's registered band *failed by one row* and they declined to use the
better test to rescue it.

## The finding that stands, and it is better than mine would have been

> **The trained allocation is an ASYMMETRIC OPTIMUM. Flattening costs 51 rows at p = 2.24e-10;
> steepening buys nothing demonstrable and turns over — 5.1× steeper is worse than 2.3×.
> Twenty-eight optimizer steps of AdamW landed on an allocation that cannot be significantly
> improved by rescaling depth, and that degrades sharply if depth mass is moved late.**

Stronger than either *"allocation doesn't matter"* (false — 51 rows) or *"allocation is a free
win"* (unsupported — p = 0.096). My Finding 3 asserted the first was wrong and then implied the
second; the truth is neither.

**One live thread it leaves, which is NOT what I proposed:** a per-item oracle over the four
allocation points reaches **258/384** against the best single allocation's 243 — **+15 items no
static allocation captures.** That is a *selection* question, not a profile-sweep question, and
`per-item-headroom-unaddressable` is the standing caution against it.

## The error, plainly

**Fourth instance tonight of a finding recovered from a document that already contained it — and
this one is mine, in a document I had personally amended.** I read its Arm A result, quoted its
`s_t` bookkeeping (0.5175 / 0.6750 / 0.7496) to diagnose the matched-norm defect, appended
Amendment 1 to it, and never read past roughly line 50 to the section at line 153 that had
already answered the question I then proposed as new.

> **Appending to a document is not reading it.** Citing one section is the moment you are most
> likely to believe you know what the rest says.
