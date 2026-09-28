# There is no instrument that can confirm the results this programme has produced

**Established by enumerating `research/reasoning_posttraining_v1/QUARANTINE.json`
(18 entries, 21,025 rows barred) and checking every candidate bank against it.**

Every number in RRP-007 through RRP-012 is **measured, not confirmed**, and that is
not a matter of discipline or wording — it is a matter of there being nothing left to
confirm on. This document records why, so the gap is a known blocker rather than an
unexplained absence.

## What the results need, and why they cannot have it

The two findings worth promoting are:

- **M4** restored abstention to **exactly Supa's 13/64** on the 64 unanswerable rows
  of the Long-Context Demonstration Suite, while posting the best arithmetic panel of
  any arm (77).
- **D1** clears every transition floor at a quarter of D3's data while retaining
  **40/47** references — the only arm to pass the gate with margin.

Both were measured on instruments barred from confirming anything.

## The barred inventory

| bank | rows | bars | reason |
|---|---:|---|---|
| `Rineya_Long_Context_Demo_v1/data/all_answer_key.jsonl` | 256 | TRAIN, SELECTION, **CONFIRMATION** | **SPENT** |
| `Rineya_Invisible_Limit_Lab_v1/data/test_answer_key.jsonl` | 1,024 | TRAIN, SELECTION, **CONFIRMATION** | **SPENT** |
| `Rineya_Invisible_Limit_Lab_v1/data/test2_answer_key.jsonl` | 1,024 | **ANY_READ** | RESERVED_ELSEWHERE |
| forge v1 `test_iid` / `test_transfer` probes | 200 + 200 | TRAIN, SELECTION, CONFIRMATION | SELECTION_EXPOSED |
| `splits_onpanel/{train,dev}.jsonl` | 845 + 120 | TRAIN, SELECTION, DEV, CONFIRMATION | PANEL_LEAKAGE |
| phase-9 `test.reviewed.jsonl` × 4 | 6,900 | TRAIN, SELECTION, CONFIRMATION | CONSUMED_SEALED |
| phase-9 `fresh_sealed.reviewed.jsonl` × 2 | 3,960 | **ANY_READ** | POLICY_LOCKED_OTHER_PROGRAMME |

**Every long-context instrument and every literal-`UNKNOWN` abstention instrument in
this repository is spent, reserved, or locked to another programme.** The 256-row
Demo suite — the only thing that measures the capability M4 repaired — is `SPENT`,
which is why every use of it in RRP-010/011/012 was registered **report-only** and
why no arm was ever selected on it.

## What IS available, and what it is not good for

**Forge RL v2's held-out splits are clean.** Verified: their ids have **zero**
overlap with the spent `RRP2_DEV_V2_BANK` (336 rows, all of which sit in
`rl/validation`), and none appears in the quarantine registry.

| split | rows | status |
|---|---:|---|
| `rl/test_iid` | 1,600 | unread, unquarantined |
| `rl/test_transfer` | 640 | unread, unquarantined |
| `rl/test_length` | 160 | unread, but **unmeasurable on this host** (p50 ~42,600 tokens, above the model's 40,960 `max_position_embeddings`) |
| `rl/validation` | 1,600 | **SELECTION-EXPOSED** — 336/336 of the DEV bank sits here |

Per-family majority floors were published before any read:
`test_iid` 381/1600 = 0.2381, `test_transfer` 226/640 = 0.3531.

**But this does not solve the problem.** Forge v2 cannot confirm the abstention
result, for a specific and disqualifying reason: its abstention contract is
`{"status":"UNKNOWN"}` **printed in the prompt**, which is why its shortcut floor is
**0.4391** (`copy_prompt_sentinel`). Abstaining correctly on v2 is nearly free. The
Demo suite requires emitting the literal `UNKNOWN` with no such cue, which is what
made 13/64 → 0/64 a real collapse and 0/64 → 13/64 a real repair.

Confirming a literal-`UNKNOWN` repair on an instrument that prints the answer would
not be a confirmation. It would be the CL-040 error in a new costume.

## What a valid confirmation instrument would require

1. **Fresh, never-read rows** — not a resample of a spent bank; a new scenario/entity/
   template partition assigned before generation.
2. **The literal `UNKNOWN` contract**, with the sentinel **absent from the prompt**,
   so the floor is not free.
3. **A published shortcut floor** computed before any model call, using the strongest
   honest predictor, not the first one that comes to mind.
4. **Long-context rows within the model's reach** — ≤ 2048 rendered tokens to train
   against, and comfortably under 40,960 to score.
5. **Registered read budget**, since an instrument is spent the moment it is used for
   selection.

Points 1–4 are exactly what `docs/LCR_GENERATION_PROMPT_V2_ADDENDUM.md` commissions
for the *rehearsal* corpus. **The confirmation instrument should be commissioned in
the same batch, from a disjoint partition**, because generating it separately later
risks it being built to match whatever the rehearsal corpus happens to contain.

## The honest consequence

Until such an instrument exists:

- **No configuration can be promoted or released.** Every run in this programme
  already records `MICROFIT_EXECUTED_NOT_QUALITY_CONFIRMATION` and
  `quality_improvement_established: false`; that is accurate, not conservative
  boilerplate.
- The strongest available statement is: *"the new skill is real and replicates
  3/3 seeds; panel retention is manageable and rehearsal manages it; abstention can
  be fully restored by abstention-shaped rehearsal; long context is damaged by every
  arm and unrepaired."* All measured. None confirmed.
- Any future arm may still be **measured** against the barred suites report-only, and
  must not be **selected** on them.
