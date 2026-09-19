# RCI-002 — built, calibrated USABLE, sealed. **The read is not spent.**

**2026-09-18.** `rci002-20260918/`, seal `cf5dac10…`, `reads_used` 0 of 1, verified
`INTACT AND UNSPENT`. Prereg: `RCI002_CONFIRM_PREREG.md`. Generator: `build_rci2.py`.

| | |
|---|---|
| sealed bank | **640 rows, 160 packs, 8 strata, 160 unanswerable** |
| calibration twin | 256 rows, disjoint entity partition, **now spent** (scored; still fine for DEV) |
| entity pools | asserted disjoint CALIB/SEALED **and** disjoint from both RCI-001 pools |
| endpoints | **two**, registered with directions that differ |
| calibration verdict | **USABLE** — both endpoints discriminate on the twin, control clean |
| model calls against the sealed bank | **zero** |

## Calibration twin, both endpoints

| endpoint | n | B | D | discordant | p | direction |
|---|---:|---:|---:|---:|---:|---|
| **E1** correct literal `UNKNOWN` | 64 | **35** | **21** | 0 / 14 | 1.2e-04 | D < B |
| **E2** person emission, incongruent | 48 | **20** | **27** | 7 / 0 | 1.6e-02 | D > B |
| **E2 control**, congruent | 48 | **0** | **0** | 0 / 0 | 1.00 | equal |
| false abstention on 192 answerable rows | | 1 | 0 | | | |

Registered bands for the sealed read: E1 B 42–133 / D 22–83; E2 B 20–80 / D 30–105.

## What this cost, and what that taught

**The gate refused its own bank five times.** Every refusal was a defect introduced while fixing
the previous one, and none was foreseeable from the RCI-001 post-mortem:

1. **Endpoint immunity was specified impossibly.** The first gate demanded a zero shortcut floor
   on *both* endpoints. Only an abstention endpoint can have one — an extractor can never emit
   the sentinel — while any answerable endpoint has some rate by chance. Corrected to extraction
   immunity for E1 and a bounded floor for E2.
2. **The answer was the only vocabulary member present.** With 20 distinct badge numbers, the
   exempt badge was frequently the sole member of its stratum's answer vocabulary in its own
   context, and `first_candidate_in_context` scored **1.000** by reading nothing. Fixed with a
   six-element badge pool rotated so every badge is the reference in some pack.
3. **A decoy register opened a worse leak than it closed.** Added to defeat positional rules, it
   listed three of four roles and **always omitted the exempt one**. Set difference scored
   **1.000** on both carve-out strata and **120/120 on the sealed bank**, while every published
   floor read 0.20–0.29. Found by the red team, before the seal. → `method/floors.py::structural`.
4. **A random subset broke frequency ties.** Listing *some* of each class gave the sampled
   members +1 and pushed `most_frequent_candidate_in_context` past 0.60. Fixed by listing
   **every** member of every class at both ends of the context.
5. **One carve-out meant a ceiling in the control cell.** The congruent cell scored 24/24 for
   the frozen source — it could not have shown an effect either way, and the whole 2×2 leans on
   it. Fixed with **one carve-out per role**, which also makes frequency uniform by construction.

Two further fixes came from the twin rather than the gate:

6. **Silent omission left prose that reads complete.** The frozen source abstained on **0 of 8**
   rows in seven of nine strata and E1 had no discordant pairs at all — 11/72 against RCI-001's
   40/128. A **lacuna note in every row**, naming the determining fact only in the unanswerable
   one, took B to 35/64. "Is there an illegibility note" is constant and worth nothing; "is the
   illegible thing the thing I was asked" is the abstention judgement itself.
7. **E2 could not be measured by accuracy.** See the prereg: a substantively correct sentence
   scores zero against a bare reference, and one stratum came back 0/24 for both systems. E2 is
   an **emission rate**, chosen on the twin, with the failed accuracy version reported alongside.

## What is genuinely new here as method

- **A third predictor class.** Distributional predictors read the answer distribution;
  extraction predictors read positions and frequencies; neither can express *"the member of
  list A missing from list B"*. `structural()` covers set complement, list arity,
  candidate-set collapse and length tells, with fixtures built from the leaks above.
- **The build refuses rather than reports.** RCI-001 published its floors and shipped anyway.
- **An adversarial pass before sealing.** Six lenses, each proposal executed rather than argued.
  That is how defect 3 was found while it was still fixable. RCI-001 had no such pass and its
  floor erratum arrived after the seal was spent.
- **Paired endpoints are immune to answer-agnostic predictors by construction.** A deterministic
  rule emits the same string for both systems, so it yields zero discordant pairs. Floors bound
  what an *absolute* rate means; they do not bear on a paired contrast. This is why E1's high
  mechanical floor does not weaken its verdict, and it was not stated anywhere before now.
- **Claim the read budget before rendering.** Both guards tested: rendering the sealed bank is
  refused without the flag, and refused *with* the flag while `reads_used` is 0.

## Known limits, stated now rather than after the read

- **E2's operationalisation was chosen using calibration data.** That is what a spendable twin
  is for, and the sealed bank was never consulted — but it is selection on the twin and is
  labelled as such.
- **E2 floor is 0.3417 against a 0.35 bound.** It passes, tightly, and the bound was my choice.
  Not load-bearing for the emission endpoint, which is paired.
- **Four strata are uninformative for accuracy** (`carve_out_badge_*`, `exclusion_count_number`,
  `object_state_tracking`, `temporal_correction`). They are not endpoints, and the endpoints are
  not accuracy — but a fifth of this bank is again carrying less than it looks like it does.
- **The big RCI-001 accuracy effect did not replicate as accuracy** on the twin
  (`carve_out_role_incongruent` B 14 → D 12, 2 losses, p = 0.50) while the emission effect did.
  Whether that is because four carve-outs make a harder task or because accuracy was always the
  wrong measure is **not settled**, and the sealed read will not settle it either.
- **One training draw of one delta.** No instrument substitutes for a second draw.

## The decision that is not mine

**The sealed read is deliberately not spent.** Spending it is irreversible, it is authorised
once, and the condition is that it happens on a frozen candidate and never to choose among
arms. B and D are both frozen, so the read is available whenever the operator wants it —
but building an instrument and spending it are two decisions, and only the first is a build step.

`quality_improvement_established` false. `release_permitted` false. Nothing promoted.
