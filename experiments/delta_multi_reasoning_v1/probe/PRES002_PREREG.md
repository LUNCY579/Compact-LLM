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
