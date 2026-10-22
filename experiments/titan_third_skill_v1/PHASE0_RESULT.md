# TRISKILL-001 Phase 0 — result

Run 2026-09-26, attempt 2 (15:12–15:22 local, rc 0). Attempt 1 died on a GPU driver reset; see
`ATTEMPT_LOG.md`. Registered in `PHASE0_PREREG.md` (a96fd42f), with execution Amendment 1
(7cdd79a1) committed before attempt 2. Outputs: `PHASE0_SCORES.json` → `PHASE0_RESULT.json`
(`analyse_phase0.py`).

## Canaries

- The template self-test passed, the scorer canary passed on 640 rows, and all three weight hashes
  matched.
- **Determinism:** 7/7 units finished in attempt 1 are **byte-identical** in attempt 2.

## Registered verdict

**G0-HEADROOM(C) PASS. BASE scores 11/96 strict, against the ≤ 48 rule and the 18-row floor. The
candidate third skill is C (rules with exceptions and priority).**

The "partially present before donation" flag is **false**. TT − BASE on C is −11 (0 gained,
11 lost).

## Tables (96 JSON-contract rows per bank; strict / semantic / format-valid)

| bank | floor | BASE (Supa) | SPARSE (Titan-Sparse) | TT (Titan-Transition) |
|---|---:|---|---|---|
| A arithmetic | 23 | 3 / 3 / 52 | 2 / 3 / 21 | 1 / 2 / 14 |
| **C rules** | **18** | **11 / 15 / 47** | **1 / 17 / 6** | **0 / 15 / 0** |
| P routing | 22 | 0 / 0 / 23 | 5 / 5 / 27 | 1 / 6 / 4 |
| R evidence | 20 | 5 / 6 / 32 | 0 / 11 / 0 | 0 / 13 / 0 |
| S state | 17 | 1 / 6 / 2 | 0 / 16 / 0 | 0 / 17 / 0 |

Paired strict contrasts on C: SPARSE − BASE is −10 (1/11, p = 0.0063); TT − BASE is −11 (0/11,
p = 0.00098).

## What it means

1. **No arm has the C skill.** C semantic accuracy is 15, 17 and 15 of 96, all at or below the
   answer-agnostic floor of 18. Titan's transition delta does not transfer C content. That makes C
   a genuinely new skill for every current core.
2. **Titan's delta imports its own answer convention onto SIF prompts.** Format validity falls from
   47 (Supa) to 6 (Titan-Sparse) to **0** (Titan-Transition): these cores answer bare (`6`, or the
   stage name `garnet`) instead of `{"answer": N}`. This is the fourth instance in the programme of
   a surface convention travelling with a delta. The 11 rows TT "loses" on C strict are format
   losses, not content losses.
3. **Consequence for the main design, registered there, not here.** The skill endpoint for C must
   be **semantic** correctness (SIF's value-under-any-clean-wrapper), with strict correctness as a
   separate format endpoint. Otherwise TT's format pull would read as a missing skill, and δ_C's
   format repair as a content gain.
4. The S and R semantic rises (6 → 16–17, 6 → 11–13) sit at or below their floors (17 and 20).
   They are descriptive and support no transfer claim.

No acquisition, retention or quality claim.
