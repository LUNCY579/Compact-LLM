# CONFIRM-001 — preregistration

Written 2026-09-24, **before the bank for this experiment existed.** No row of `T_BANK_V3.json` had
been generated, and no arm had been scored on it, when this file was written.

## Why this experiment exists

The adversarial review of 2026-09-24 established that FRESH-DUAL-SKILL-001's headline is an
**explanation, not a confirmation**:

- its registered prediction failed on its registered strict endpoint;
- the period-tolerant rule that rescued it was chosen after seeing the generations;
- its bank, T_BANK_V2, is **exposed** (14 arms scored, and it informed four amendments) and
  **content-overlapping** on 11/192 rows (Erratum 2).

The review also found that the only test in this programme which could speak to **module location**
— DPLOC-001's random-mask control — was never run on this skill here. MODULE-DISJOINT-001's
scalar control answers a different question (Erratum 1).

CONFIRM-001 fixes both problems. The development findings are turned into registered predictions
that can fail, on a fresh bank that is **read once**.

## The instrument

`T_BANK_V3.json`: **288 rows, 96 per family** (`ledger_replay`, `prerequisite_plan`,
`register_machine`), using the same verbatim system prompts and oracles (`oracles.py`, validated
384/384).

- **Generator:** v2's recorded-marginal sampler (difficulty-matched; it passed the gates on v2).
- **Seed:** 20260924. This differs from every earlier bank.
- **Content-disjoint by construction, not only ID-disjoint** — Erratum 2's lesson. A generated row
  is rejected if its **operative key** (answer-determining fields only) or its
  **(before-state, answer) pair** matches any payload in:
  - the complete historical harvest: 6,462 payloads, in dict, embedded-string, `*.jsonl` and
    `*.json` forms;
  - T_BANK and T_BANK_V2, my own exposed banks.
- **A collision count is reported.** It is required to be 0 for both keys.

**Read budget: ONE.** The arms are scored once. The bank is not regenerated and no arm is added after
any score has been read, except as provided in the gate rule below.

## Scoring — registered now, in advance

- **Primary: period-tolerant exact match.** The generation, stripped of whitespace and with **at most
  one** trailing `.` removed, must equal the reference. This is the rule
  `architecture_benchmark_001/retention.py:_strip_one_period` uses. The conservative choice is
  deliberate: FRESH-DUAL-SKILL-001's secondary stripped all trailing periods, which is looser.
- **Secondary: strict exact match**, reported for every arm.
- `max_new_tokens` 48, greedy, batch 1, FP32, thinking disabled, byte-exact template (self-tested
  against 384 recorded prompts). All GPU work runs under the shared lease, and so does construction.

## Build gates, on BASE and SPARSE only, read first

| gate | requirement (n = 288) |
|---|---|
| G1 base floor | BASE ≤ 57 (20%) |
| G2 donor ceiling | SPARSE ≥ 115 (40%) |
| G3 separation | SPARSE − BASE ≥ 86 (30pp) |

Gates use only the two calibration arms, so reading them reveals no arm contrast. **If any gate
fails, the bank is rejected: no other arm is scored, no confirmatory claim is made, and there is no
attempt 2.**

## Arms

| arm | construction | status |
|---|---|---|
| `BASE` | Supa | exists |
| `SPARSE` | Supa + dSparse | exists |
| `AVG` | Supa + 0.5·dSparse + 0.5·dOpallo | exists |
| `ADD-100` | Supa + dSparse + dOpallo | exists |
| `SPARSE_ATTN` | Supa + dSparse\|ATTN (‖Δ‖ 0.4401) | exists |
| `SCALAR_MATCHED` | Supa + 0.634044·dSparse (‖Δ‖ 0.4401) | exists |
| `RAND_MASK` | Supa + dSparse ⊙ M, where M is a Bernoulli per-element mask over dSparse's non-zeros, seed 20260924, chosen so ‖Δ‖ = 0.4401 to within 0.1% | **to build** |

`RAND_MASK` is DPLOC-001's control: it removes the same squared norm as `SPARSE_ATTN`, but from
**random** elements rather than from a module. dOpallo is the unrounded FP32 reconstruction, verified
220/220.

## Registered predictions

All contrasts are paired over the same 288 rows. p-values come from the exact two-sided McNemar
test. Intervals come from a paired bootstrap, 10,000 resamples, seed 20260924.

**Confirmatory family — the composition headline.** Holm-adjusted at α = 0.05 across C1–C3.

- **C1 — attenuation is real.** SPARSE − AVG ≥ 10% of SPARSE, with p < 0.05 (Holm).
  *Development: 29/93 = 31% on V2 clean rows.*
- **C2 — independent coefficients beat averaging.** ADD-100 > AVG, with p < 0.05 (Holm).
  *Development: +25 on V2 clean rows. This is the practical claim: never average, compose at full
  independent strength.*
- **C3 — interference is small.** The lower 95% bound of (ADD-100 − SPARSE) must exceed −10% of
  SPARSE. This is a **pre-registered margin, not a gain test**: passing it means ADD-100 retains at
  least 90% of SPARSE's skill with 95% confidence. A non-significant difference would **not** count
  as passing. *Development: 89/93 = 95.7%.*

**Descriptive — the convention transfer.** No p-value, no claim of mechanism.

- **C4.** ADD-100 emits a trailing `.` on ≥ 50% of rows, while SPARSE and AVG each emit one on
  ≤ 2%. *Development: 178/192 against 0 and 0.*

**Separate question — module location, DPLOC's design.** Its own α = 0.05; it is not in the Holm
family.

- **C5.** SPARSE_ATTN versus RAND_MASK, both at ‖Δ‖ 0.4401. Three outcomes are registered:
  - **location effect replicates:** SPARSE_ATTN − RAND_MASK ≥ +10 rows and p < 0.05 — attention
    carries more of the skill than random elements of equal norm do;
  - **reversed:** RAND_MASK − SPARSE_ATTN ≥ +10 and p < 0.05;
  - **no location effect:** |difference| ≤ 6.
  - Anything between these is **inconclusive** and is reported as such.

  `SCALAR_MATCHED` is reported alongside as the uniform-shrinkage reference, so that all three ways
  of removing the same norm — module, random elements, uniform scaling — sit on one bank.

## What each outcome would mean

- **C1–C3 all pass:** the headline "compose at independent full coefficients; averaging pays the
  dominant cost" is confirmed on a fresh, unexposed, content-disjoint bank under a rule registered in
  advance. It is no longer only an explanation.
- **Any of C1–C3 fails:** the development result does not replicate, and the write-up says so first.
- **C5**, whichever way it lands, is the first test on this skill of the question MODULE-DISJOINT-001
  was wrongly taken to answer.

## Limits, stated now

- One skill (the transition contract), one base, one pair of donors. This is not a quality,
  retention or release claim.
- It shows whether Titan-Sparse's skill **survives** composition. It says nothing about any Opallo
  skill arriving — Opallo has no identified skill (see memory `opallo-has-no-own-skill`).
- 288 rows resolve effects of roughly 5% of SPARSE or more. Smaller effects are not claimed either
  way.
- No authorization is created or amended. SOUP-MINI-001's `protocol.json` is untouched. Its screening
  limb "arithmetic ≥ 95% of Opallo" is known to rest on a false premise (see
  `titan_opallo_composition_audit_v2/REVIEW_2026-09-24.md`), and this experiment does not use it.
