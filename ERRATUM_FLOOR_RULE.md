# Erratum: three different floor-clearing rules are in circulation, and they disagree on one measured arm

**Raised by the adversarial review of GATE v2. Every number below was recomputed from the
registration files and the evidence artifacts.**

## The defect

The rule for "this family cleared its shortcut floor" exists in three inconsistent forms:

| source | `ledger_replay` | `register_machine` | `prerequisite_plan` |
|---|---:|---:|---:|
| `PROCESS_PROBE_FLOORS.md:55-58`, "Power, stated in advance" — one-sided exact binomial critical value | **≥40** | **≥13** | **≥4** |
| `RRP008_DOSE_PREREG.md:55`, `RRP009_MIXTURE_PREREG.md:68`, `RRP013_RELIABILITY_PREREG.md:41` — restated shorthand | **≥40** | **12** | **15** |
| **The implemented rule** — two-sided exact binomial (Clopper–Pearson) 95% **lower bound** strictly exceeding the binding floor | **≥42** | **≥13** | **≥16** |

Three triples, no two alike. Two further errors compound it:

- The `PROCESS_PROBE_FLOORS.md` power paragraph computes `prerequisite_plan` against **0.0000**
  (`copy_before`) rather than its **binding** floor of **0.0703** (`constant_majority`) — the very
  substitution the rest of that document exists to prevent. That is where `≥4` comes from.
- The shorthand's second and third entries are each one below the implemented values, and the
  prerequisite/register figures appear to have been transcribed in the wrong order.

## What the published results actually used

**The implemented rule, in every case.** `clears_floor` is computed as the Clopper–Pearson lower
bound against the binding floor, and every one of the 26 `PROCESS_*.json` artifacts carries a
`decision_rule` field stating exactly that. No result was ever computed from a transcribed `k`.

The implemented rule is also the **strictest** of the three. That is the conservative direction:
it can only make a clearance harder to claim, never easier. No published clearance is at risk.

## Where the rules disagree, and what it changes

They disagree at exactly one measured value in the whole programme: **D0 seed 17,
`ledger_replay` = 41/128.**

```
CP 95% lower bound = 0.2406   vs   binding floor 0.2422   ->  BELOW   (short by 0.0016)
shorthand k >= 40:  41 >= 40                              ->  CLEARS
```

Recomputing all three D0 seeds both ways:

| arm | `ledger_replay` | `prerequisite_plan` | `register_machine` | CP rule | shorthand |
|---|---:|---:|---:|:--:|:--:|
| D0 seed 17 | 41 | 91 | 65 | 2/3 | **3/3** |
| D0 seed 23 | 36 | 80 | 67 | 2/3 | 2/3 |
| D0 seed 29 | 39 | 90 | 49 | 2/3 | 2/3 |

**`PROCESS_DOSE_RELIABILITY.md` reports "D0: seeds clearing all three floors 0/3". Under the
shorthand it would be 1/3.** That figure is hereby qualified: it is 0/3 under the implemented
and self-documented CP rule, and 1/3 under the looser shorthand restated in three preregs.

**The finding it supports is unaffected either way.** D0 passes the registered gate **3/3** while
clearing all three floors in **at most 1 of 3 seeds**, against D1's **6/6**. The gate is blind to
acquisition under either reading, which is the whole of RRP-013's argument. Nothing else in
RRP-007 through RRP-014 turns on the difference — no other arm sits between the two thresholds.

## The binding rule from here

**GATE v2's acquisition conjunct binds the Clopper–Pearson predicate evaluated against the
instrument's own `FLOORS.json`, with no transcribed minimum count.** `process_probe.py`'s
`decision_rule` now says so explicitly, and it names the instrument's own floors document rather
than always citing `PROCESS_PROBE_FLOORS.md`.

The three triples above must not be quoted again. Where a prereg restates a shorthand, the
shorthand is superseded by this erratum, not the other way round — the preregs' *rule* was always
"clears its published binding floor"; only the parenthetical count was wrong.

## What is not changed

- **No published result is recomputed or re-ranked.** Every arm was scored under the strictest
  rule and stays as scored.
- **No threshold is adjusted.** Fixing a transcription is not the same as moving a floor, and the
  floors themselves are untouched.
- `PROCESS_PROBE_FLOORS.md` and the three preregs are **not edited**. They are pre-registrations;
  the discipline is that they are corrected by a dated erratum that cites them, never by a silent
  in-place edit.
