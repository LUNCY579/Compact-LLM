# PRES-002 — registered before the first prediction file is read

**Written and committed while the Stage 1 generations are running.** Generation is
answer-neutral; scoring is not, and none has happened. MECHANISM SCOUT — no seal, no read budget.

Build: `probe/build_pres2.py` → `pres002-20260919/`. 379 rows, three strata, floors 0.21–0.33
against a 0.45 bound chosen before any floor was computed. Prompts rendered by
`probe/render_prompts.py`, which reproduces ALIAS-001's 360 prompts **byte-for-byte**.

## Why this instrument exists

Amendment 7 narrowed "direction beats dose" to **one skill**, because neither available second
endpoint could test it: arithmetic is the wrong class (it *rises* with dose) and
`references_retained` is underpowered roughly threefold. **"We tested it and it did not
transfer" is false; "we have not been able to test it" is true.** This makes the test possible.

## Stage 1 — is anything here damaged at all?

Scored: the frozen source **Supa** against **λ=0**, the LWF lane's unanchored arm, on the
identical 379 rendered prompts.

**This stage cannot be dressed up as a result.** Its only job is to decide which strata are
eligible to serve as preservation endpoints. A stratum qualifies only if **both** hold:

1. **Supa clears the stratum's published floor** by `gate()` — otherwise the source cannot do
   the task and there is nothing to preserve;
2. **λ=0 damages it**, exact two-sided McNemar over the paired rows at p ≤ 0.05, with damage
   scored over **the rows Supa got right**.

**A stratum failing either test is reported and DROPPED, not carried.** That is the discipline
the arithmetic panel should have had — an endpoint the delta improves cannot show sparing, and
including it would repeat the mistake in a new coordinate.

### Registered outcomes for Stage 1

| outcome | what it means | what happens next |
|---|---|---|
| **≥ 1 stratum qualifies** | a usable second preservation endpoint exists | Stage 2 runs on the qualifying strata only |
| **no stratum qualifies** | these three skills are not damaged by this delta | **Stage 2 is cancelled**, and the one-skill narrowing in Amendment 7 stands unchallenged and untested |
| **Supa is at/below floor everywhere** | the bank is too hard for the source | the instrument is withdrawn and rebuilt easier, not reinterpreted |

I am registering in advance that **the third outcome is a build failure, not a finding**, because
a bank the source cannot do will produce a clean-looking null for an uninteresting reason.

## Stage 2 — does direction sparing generalise? (conditional)

On qualifying strata only, with the same construction that produced the alias result: each
off-ray arm against a **norm-matched, exactly collinear twin**, so magnitude is held fixed.

Arms: **λ=0.25** (anchor, cos 0.5243) and **ord23** (row order, cos 0.5939), each against a twin
built by `analysis/build_scaled_arm.py` at its own measured norm and verified before scoring.

| account | residual on qualifying strata |
|---|---|
| **GENERALISES** — direction spares unexercised skills as a class | positive, in the same direction as alias, on at least one stratum |
| **ALIAS-SPECIFIC** — the effect belongs to role→person binding | ≈ 0 everywhere, inside a ±2.5 pp band |
| **REVERSED** | negative; no account predicts it |

**Primary: exact two-sided McNemar per (arm × stratum), Holm-corrected across the family.** The
family size is not known until Stage 1 qualifies strata, and Holm is applied over whatever that
family turns out to be — fixed as a rule now so it cannot be chosen after the counts are seen.

**Effect size to beat, registered:** on alias, λ=0.25 recovered **39%** of the damage λ=0 did
(28 of 71 rows). The same fraction is the benchmark here; a stratum with too little damage to
resolve 39% of it at p ≤ 0.05 is **underpowered rather than negative**, and will be reported
that way — which is exactly the error I corrected the peer on for `references_retained`, and I
am not going to commit it myself one document later.

## What no outcome here can do

Settle the mechanism. Three mechanisms already produce the alias effect and none of them is
explained. This asks only whether the effect is one skill's or a class's.
`quality_improvement_established` false, `release_permitted` false.

---

# ADDENDUM 1 — corpus audit: what RRP-015 actually trains, decoded

**Run before Stage 1 was scored, to answer my own contamination question rather than assume it.**
`rrp015-20260917/control-prepared/microfit-train.jsonl`, 384 rows, **decoded from `ids` with the
source tokenizer** — a first pass over the raw JSON was measuring field names and was discarded.

## All three families are arithmetic

| family | n | system prompt, in brief |
|---|---:|---|
| `ledger_replay` | 128 | apply one event to account balances — **add, scale, transfer** |
| `register_machine` | 128 | apply one instruction to registers, **integers modulo 97** — swap, add, mix, conditional |
| `prerequisite_plan` | 128 | an **8-bit flag mask**; delete bits first, then add bits |

**Arithmetic is not a preserved skill for this delta. It is the trained skill.** That settles the
endpoint-class question in Amendment 7 with direct corpus evidence rather than an inference from
the dose curve: the arithmetic panel rising Supa 47 → 71 across α is **acquisition**, and using
it as a preservation endpoint could only ever have produced an uninformative null. It also
corroborates the peer session's finding that `prerequisite_plan`'s collapse "destroyed bitwise
arithmetic, not precondition reasoning" — the family *is* bit manipulation.

## Contamination check for this bank's three strata

| probe | count in the decoded corpus |
|---|---:|
| weekday names (Monday … Sunday) | **0** |
| temporal-order phrases (*immediately before*, *the day before*, *preceding*, *prior to*) | **0** |
| containment words (*inside*, *kept in*, *contained*, *within*) | **0** |
| person-name-shaped capitalised pairs | **0** |

`before` occurs 384 times and `then` 128 times, and **neither is temporal language**: `"before"`
is the JSON key naming the prior state (`{"before":{"A":-10,…}}`), and `then` is the single
phrase *"Apply delete bits first, then add bits"* in `prerequisite_plan`'s instruction.

**Verdict: `object_coreference` and `temporal_predecessor` are clean.** The lexical `before` is a
state label in a JSON object, not ordering over events, and the corpus never asks which of
several things came first.

**One caveat recorded rather than dismissed: `container_nesting` has a weak *structural*
analogue.** The corpus is nested JSON throughout — a state is an object containing keyed values —
and "locate a container, then report its contents" is a nesting traversal. The surface form is
entirely different (natural-language *"the X is kept inside the Y"* against
`{"before":{"A":-10}}`), and no containment vocabulary appears at all, so I judge the overlap
structural-but-remote. **If `container_nesting` behaves differently from the other two strata,
this is the first thing to suspect**, and it is written down now so that reading is available
rather than invented afterwards.

---

# ADDENDUM 2 — reallocation, a known-positive reference, and one suspect stratum

**Written after Stage 1's generations finished and BEFORE any of their predictions were read.**
All three items come from the peer session; two are their corrections to my design and the third
changes what Stage 2 is testing.

## 1. The finding that reframes this: the rotation may MOVE damage, not remove it

The peer ran a third preservation endpoint — **long context, n = 256** — which had never been run
on a rotated arm:

| arm | long context /256 | vs λ=0 | alias (this programme) |
|---|---:|---:|---:|
| frozen Supa | 161 | — | 0.750 |
| **λ = 0** | **162** | **+1** | 0.553 |
| **λ = 0.25** | **145** | **−17** | **0.631** |

**λ = 0 does not damage long context at all** (162 against 161), so it is a fourth endpoint of
the wrong class *for the unrotated delta*. **But λ = 0.25 damages it by 17 rows.** The rotated
arm recovers ~28 alias rows and loses ~17 long-context rows.

**So "direction spares an unexercised skill" may be the wrong description, and "direction
changes WHICH skills are damaged" may be the right one.** Totals are *not* claimed to be
conserved — two instruments on different scales cannot be summed and no sum is computed — but
the sign is wrong for a pure-sparing account, and "sparing" is the word my records currently use.

**One caveat that makes this conservative rather than fragile:** λ=0.25's comparison here is
against λ=0, which has a *larger* norm (0.8323 vs 0.7427), and INT-001's α ladder shows long
context is roughly flat in magnitude (161/163/164/163/157). So the norm difference cannot explain
a 17-row drop, and if anything runs the wrong way. The clean version still needs λ=0.25 against
its **norm-matched collinear twin** on long context — the same construction used for alias — and
that arm does not exist.

### Registered prediction, and PRES-002 can test it

> **If direction REALLOCATES damage, then on at least one qualifying PRES-002 stratum the
> rotated arm should be WORSE than its collinear twin — a negative residual — not merely equal.**

| account | PRES-002 residuals on qualifying strata |
|---|---|
| **SPARING** (records as currently written) | positive or ~0; never reliably negative |
| **REALLOCATION** | **at least one stratum negative** and significant |
| **ALIAS-SPECIFIC** | ~0 everywhere, neither positive nor negative |

This is a sharper test than the one Addendum-free Stage 2 registered, it costs nothing extra, and
**a negative residual is now a predicted outcome rather than an anomaly to explain away.**

## 2. A known-positive reference scale, adopted from the peer

Three unknown strata cannot tell me whether a null means "no effect" or "no sensitivity". So
Stage 2 will **report the ALIAS-001 λ=0.25-vs-collinear-twin contrast beside whatever PRES-002
yields** — `+7.2 pp, 32 gains / 6 losses, p = 2.43e-05`, numbers that already exist.

Then "alias shows it and these strata do not" is a within-session contrast **against a
known-positive measured on the same arms by the same pipeline**, rather than three nulls of
unknown sensitivity. Alias cannot be folded *into* PRES-002 without breaking the no-person-names
constraint that makes this bank interpretable, so it is reported beside rather than inside.

## 3. `container_nesting` is registered as the SUSPECT stratum, not an equal third

Addendum 1 called its structural overlap with the corpus "remote". The peer strengthened that and
they are right: **the corpus JSON is depth-2** (`before` → keyed values) and the stratum is a
**two-hop** traversal, so the analogy is closer than I wrote. It is registered here as the
stratum to distrust **in advance**: if `container_nesting` behaves unlike the other two, the
corpus-overlap reading is available rather than invented afterwards, and if it behaves like them
that is mild evidence the overlap does not bite.

---

# RESULT — PRES-002 FAILED CALIBRATION. Registered outcome 3: a build failure, not a finding.

**Stage 1, scored against `probe/PRES002_PREREG.md`'s registered rule.** Receipt
`pres002-20260919/PRES2_STAGE1.json`.

| stratum | n | floor | **Supa** | λ=0 | repair | damage | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `object_coreference` | 132 | 0.333 | **38** (0.288) | 63 | 25 | 0 | **BELOW_FLOOR** |
| `temporal_predecessor` | 115 | 0.209 | **0** | 0 | 0 | 0 | **BELOW_FLOOR** |
| `container_nesting` | 132 | 0.273 | **0** | 4 | 4 | 0 | **BELOW_FLOOR** |
| pooled | 379 | — | 38 | 67 | **29** | **0** | — |

**The frozen source is below floor on all three strata and scores literally zero on two.** The
prereg registered this exact outcome in advance:

> *"Supa is at/below floor everywhere → the bank is too hard for the source → the instrument is
> withdrawn and rebuilt easier, not reinterpreted."*

**It is withdrawn.** Note what the pooled row would have permitted: **29 repairs, 0 damages,
p = 3.7e-09** — a clean, significant, entirely meaningless "the delta improves this bank"
headline, on a bank the source cannot do. That is precisely the shape the registered rule
existed to refuse, and it refused it.

## Why it failed, diagnosed from the predictions rather than guessed

Not an answer-form artifact — the failures return the **wrong entity**, not the right one in the
wrong shape, so this is not RCI-002's 0/24 problem:

| stratum | what Supa actually answers |
|---|---|
| `container_nesting` | **the container** — *"The banded chest"*, *"The iron strongbox"*. It performs hop 1 and stops. |
| `object_coreference` | **the description** — *"wax-sealed item"*, echoed unresolved. |
| `temporal_predecessor` | **a weekday** — *"Monday"*, *"Tuesday"*. Four hops is out of reach at this scale. |

**The comparison that locates the error is ALIAS-001, where Supa scores 0.750 on the same
two-hop shape.** The structures are identical; the bridge is not:

  * alias bridges **verbatim** — *"It was the provisions marshal who carried X"* → *"The
    provisions marshal that week was <name>"*. The same four words appear in both sentences.
  * mine bridges by **paraphrase** — *"The wax-sealed item was moved"* → *"only the X bears a
    wax seal"*. The model must equate *wax-sealed* with *bears a wax seal* before it can begin.

**I optimised for the floors and forgot the gate.** The floors ask whether the *bank* is
gameable; the gate asks whether the *source* can do the task. Three strata cleared 0.21–0.33 on
the first and failed the second, and the two questions are independent in exactly the way
RCI-002's unreadable-from-either-direction endpoint was.

**One thing the failure does establish**, and it is the only claim taken from it: these three
tasks are **wrong-class endpoints** — λ=0 *improves* them 29/0. Consistent with everything else
here, this delta damages some skills and improves or ignores others, and eligibility has to be
measured before anything else is.

---

# PRES-004 CALIBRATION — article fix worked; one stratum dropped; target missed

**One arm (Supa), scored before spending any others** — the lesson from two dead banks.

| stratum | n | floor | **Supa** | gate | disposition |
|---|---:|---:|---:|---|---|
| `person_binding` | 120 | 0.167 | **0.833** | CLEARS | **usable** |
| `object_binding` | 120 | 0.167 | **0.858** | CLEARS | **usable** |
| `place_binding` | 120 | 0.175 | **0.967** | **ABOVE_CEILING** | **DROPPED** |

**The convention fix is complete: 0 of 360 rows turn on an article**, against PRES-003's 241/360.
Article-free proper nouns plus article-tolerant scoring plus a build-time refusal did the job.

**`place_binding` is dropped by the registered two-sided rule** — a stratum whose source sits at
ceiling cannot show damage. Dropped, not reinterpreted.

## The calibration stratum fired, and it says the bank is easier than intended

`person_binding` rebuilds ALIAS-001's task under ALIAS-001's `low` salience profile, where the
frozen source scores **0.633**. It scored **0.833** — a **20-point miss**. The construction is
not reproducing its reference, and I do not have a diagnosis: line counts, salience profile,
prompt lengths and activity templates all match within tolerance.

**That is exactly what the stratum was carried for, and it reports rather than being hidden.**

**Proceeding anyway, with the reason stated.** `gate()` returns CLEARS, not ABOVE_CEILING: 20 of
120 rows are already wrong, and a relative loss the size of ALIAS-001's (−19.7 pp from 0.750,
~26% of its correct rows) would cost ~26 rows here — comfortably detectable. **The bank has
less headroom than designed but enough to run**, and the two usable strata still differ only in
the answer's type, which is the contrast the whole build exists for.

**The caveat that must travel with any result from it:** PRES-004 is **easier than ALIAS-001**,
so a *smaller* damage effect here than on ALIAS-001 is ambiguous between "persons are special"
and "this bank is easier", while a *comparable or larger* effect is not. Only the second
direction is clean, and that asymmetry is registered now rather than discovered later.

---

# PRES-004 RESULT — the question is answered, and the bank carried a cue I printed myself

## The result (eligibility screen: Supa vs λ=0, 360 rows)

| stratum | Supa | λ=0 | repairs | damages | p | disposition |
|---|---:|---:|---:|---:|---:|---|
| **person_binding** | 0.833 | 0.592 | **0** | **29** | **3.7e-09** | QUALIFIES |
| **object_binding** | 0.858 | 0.717 | **1** | **18** | **7.6e-05** | QUALIFIES |
| place_binding | 0.967 | 0.867 | 0 | 12 | 4.9e-04 | dropped (ceiling) |

> **The delta damages object-valued binding too. The regression is not about PERSONS — it is
> about role→filler binding in general, and `alias_resolution` is one instance of it.**

`person_binding` is **purely subtractive, 0 repairs / 29 damages** — the same signature
`alias_resolution` carries across three banks. Person damage (−24.1 pp) **exceeds** ALIAS-001's
own (−19.7 pp) despite an easier bank, which is the clean direction under the registered
asymmetry. Object damage (−14.1 pp) is smaller and therefore **ambiguous** between "objects are
less affected" and "this bank is easier" — as registered, and not resolved here.

## The defect, found by reading one rendered context

```
mine:       "The the provisioning steward that week was Annika Oyelowo,
             which is not named again in this account"
ALIAS-001:  "The provisions marshal that week was Kwabena Lindholm,
             who is not named again in this account"
```

`BRIDGE` already contained *"the"* and the template prepended *"The {bridge}"*. **Result:
`The the` on 360 of 360 rows, landing on the binding sentence — the one sentence containing the
answer.** A unique token bigram marking the target sentence in every item. Plus *"which"* for a
person on all 120 person rows and a missing article before item names.

**That is a cue marker, the same failure class as RCI-001's lacuna note**, and it explains the
calibration miss: 0.833 against the 0.633 target.

**What survives:** the **paired contrast**. Supa and λ=0 read byte-identical text, so a cue
cannot manufacture a discordant pair — the immunity argument this programme has used throughout.
The eligibility result above stands.

**What does not:** the **absolute level**. 0.833 is inflated and no absolute reading from
PRES-004 is usable.

## The meta-lesson, which is the one that generalises

**Floors ran. The gate ran. `convention_sensitivity()` ran. Mention counts were asserted. Bridge
occurrences were asserted.** Five automated checks, none of which can see `The the` — and the
defect was visible in the first rendered context I printed, which I printed only after the
result was in hand.

> **Read one rendered item before spending any GPU. No check that operates on structure can see
> a defect in prose.**

## PRES-005

Rebuilt with the bridge capitalised rather than prefixed, `who`/`which` selected by answer type,
articles restored before item names, and a **build-time assertion that refuses any context
containing a doubled article**. Floors 0.167 / 0.167 / 0.175, zero article-bearing references,
**zero doubled articles**. Awaiting the lease.

---

# PRES-005 — registered before it is scored: what the cue could have done, and what survives

**Written while PRES-005's screen is generating and before any of its predictions are read.**

The peer session raised a consequence of the `The the` cue that I had not drawn, and it is
correct:

> **If the cue marked the answer's own sentence in every row, it may have made the binding step
> easier to LOCATE — and therefore made the damage easier to SEE.**

The **sign is safe**: `person_binding` returned **0 repairs / 29 damages**, and a cue that
inflates both arms equally cannot manufacture a 29–0 split. Byte-identical text yields
byte-identical shortcuts. **But the magnitude may be cue-assisted**, and that lands on exactly
the comparison I called clean.

## What is registered as surviving, and what is not

| quantity | status |
|---|---|
| **person vs object contrast** within PRES-004 | **SURVIVES** — both strata carried the identical cue, so it cancels between them |
| direction of damage (0 repairs on person) | **SURVIVES** — a shared cue cannot produce a one-sided split |
| **person damage −24.1 pp vs ALIAS-001's −19.7 pp** | **WITHDRAWN pending PRES-005** — I called this "the clean direction under the registered asymmetry"; it is not clean if the cue assisted the magnitude |
| any absolute rate from PRES-004 | void, as already recorded |

**So the claim "person damage exceeds ALIAS-001's despite an easier bank" is withdrawn now,
before PRES-005 can be used to rescue it.** If PRES-005 reproduces a comparable magnitude the
claim returns on clean evidence; if it does not, the cue explains the difference and the
conclusion was mine to lose.

**The headline is unaffected either way**, because it rests on the surviving contrast:

> **The delta damages object-valued binding as well as person-valued. The regression is about
> role→filler binding in general, and `alias_resolution` was one instance of it.**

## Registered prediction for PRES-005

Both strata damaged, purely or near-purely subtractive, with `person_binding` at a **lower**
absolute source accuracy than PRES-004's inflated 0.833 — nearer ALIAS-001's 0.633, since the
cue is gone and every other parameter matched ALIAS-001 `low` exactly (3 distractors, 9.0:1
competing mentions, 153 context words).

**If `person_binding`'s source accuracy stays near 0.833 with the cue removed, the cue was not
the cause and the 20-point calibration miss is still unexplained** — which would be the most
informative outcome and is the one I am not expecting.
