# `alias_resolution` — replicated, purely subtractive, and still mechanism-free

**2026-09-18, after RCI-002's read. Zero model calls** — everything here runs on predictions
already recorded by two spent confirmation reads. Receipt: `analysis/alias_salience.py` →
`rci002-20260918/ALIAS_SALIENCE.json`. **Exploratory throughout.**

This was the one cell nothing explained after five dead mechanism accounts and one confirmed
one. It is now much better characterised and no better explained, which is worth writing down
precisely because the temptation is to stop at a sixth story.

---

## 1. It replicates, and it is purely subtractive

| bank | n | B | D | repairs | damages | p |
|---|---:|---:|---:|---:|---:|---:|
| RCI-001 sealed | 78 | 65 | 53 | **0** | 12 | 4.9e-04 |
| RCI-002 sealed | 60 | 41 | 36 | **0** | 5 | 6.3e-02 |
| **pooled** | **138** | | | **0** | **17** | **1.53e-05** |

**Zero reversals in seventeen discordant pairs, across two independently generated banks with
disjoint entity pools and a rebuilt generator.** Neither bank alone is decisive — RCI-002's cell
is p = 0.063 — and pooled they are.

### The context that stops this being over-read

Across both sealed banks and every answerable stratum, the delta **repairs 30 rows and damages
56**. It is *not* uniformly worse, so a stratum with zero repairs is a signature rather than the
background. Strictly nested strata: `alias_resolution` on **both** banks, `rule_exception` on
RCI-001 (now understood as [wh-compliance](confirm/RCI002_CONFIRMATION_001.md)), and
`exclusion_count_number` on RCI-002.

## 2. The hypothesis I tested, and why it is refuted

Both generators name the aliased person exactly once — *"The \<role\> that week was \<name\>,
who is not named again in this account"* — while movers are named up to three times. So the
correct answer is systematically the **least salient** person present.

> **Salience-over-binding:** asked "Who", the delta reaches for the most available person rather
> than the one the text binds to the role.

This would have unified the alias cell with the confirmed result — one mechanism, two surfaces:
where no person is bound it invents one (confirmed, 99/120), where one is bound but others are
louder it takes a louder one.

**It is refuted as stated.** Salience-over-binding *is* the failure mode — on RCI-001, **all 12**
rows the delta lost it named a mover, and on RCI-002 four of five. But:

| | errors | named a mover | error profile |
|---|---:|---:|---|
| **B** frozen source | 13 | **13/13** | 3 none-of-these + 10 most-frequent-and-last |
| **D** on the same shared rows | 13 | **13/13** | 3 none-of-these + 10 most-frequent-and-last |
| D on the 12 rows it uniquely lost | 12 | 12/12 | 4 none-of-these + 8 most-frequent |

Fisher on the mover share of errors, D vs B: **p = 1.000 on both banks.** The profiles are
identical. **The delta did not change the kind of error; it increased the rate of an error the
frozen source already makes.** Mean mention-count of the emitted person barely moves either —
1.603 for D against 1.500 for B, with the correct answer at 1.308.

So the answer to "does the delta reach for salience *more*" is no. The answer to "is salience
what both models reach for when they fail" is yes, and that is a fact about the task, not about
the delta.

## 3. Nothing measurable distinguishes the rows it loses

| | people present in context | times the alias is named |
|---|---:|---:|
| rows D lost (n = 12) | 2.75 | 1.25 |
| rows D kept (n = 53) | 2.92 | 1.40 |

Both differences are small, in the direction you would predict, and nowhere near separable at
this n. This echoes the programme's earlier per-item result — real headroom, no input-visible
feature that reaches it — and it is the third time a "which rows?" question here has come back
featureless.

## 4. What is now established, and what is not

**Established.** A purely subtractive, replicated regression on role→person binding, 0/17,
p = 1.5e-05, whose error character is indistinguishable from the frozen source's own errors and
whose affected rows carry no distinguishing feature.

**Not established.** Any mechanism. Specifically **not** salience-over-binding as a *differential*
account, and not [wh-compliance](confirm/RCI002_CONFIRMATION_001.md) — the interrogative is
congruent throughout this stratum and the answer form is preserved (a person name on 78 of 78
RCI-001 rows).

**This is a sixth account's worth of temptation and I am not taking it.** The honest position is
that the delta weakens a binding step uniformly, and "uniformly weakens" is a description, not a
mechanism. It is compatible with the delta simply having less capacity left for a skill its
training never exercised, which is not testable from these artifacts.

## 5. What would settle it, and what it costs

1. **A dose ladder on this one stratum.** If the effect is a uniform weakening, alias accuracy
   should fall **monotonically** with delta magnitude, and the error profile should stay
   mover-dominated at every dose. If instead it is a threshold, that is a different claim. The
   α line and the k line already exist; this needs no training, only a fresh alias-only bank and
   about six generation passes. **Cheapest decisive next step.**
2. **Vary salience directly.** Name the aliased person three times and the movers once, inverting
   the salience ordering. Uniform weakening predicts the *gap* between B and D persists; a
   salience account predicts it collapses. Also no training, one build.
3. **Check whether the training data contains any role→person binding at all.** If RRP-015's
   recipe is purely state-transition JSON, "no capacity left for an unexercised skill" becomes
   the leading account, and it is checkable against the corpus rather than the model.

(2) and (1) together separate the three live readings with no training and no new instrument
class — and (2) is the one that would have caught me if I had published the salience account.

## 6. One process note

I wrote the salience hypothesis down, predicted the direction, and it failed on the control —
the frozen source does exactly the same thing. That control existed because the last five
accounts died without one. **The measurement that killed this hypothesis took four lines**: count
what *both* systems reach for, not just the candidate. Every previous dead account in this
programme would have been killed the same way and none of them was.
