# Rehearsal protects the panel and does not protect the model

**RRP-010. Registered prediction 2 is falsified, and it is the prediction the
rehearsal recommendation rested on.**

Every retention number in RRP-007/008/009 came from one instrument: the 150-row
arithmetic panel, covering 3 of v3's 39 reply classes. On it, rehearsal is clearly
better — M1 retains 38–40 across three seeds against D3's 35–37, distributions
disjoint. RRP-010 asked whether that survives a second instrument.

It does not.

## Long-Context Demonstration Suite, 256 rows

| | all 256 | showcase 64 | stability 192 | `UNKNOWN` on 64 unanswerable |
|---|---:|---:|---:|---:|
| **Supa** (source) | **161** | 41 | 120 | **13** |
| **D3** transitions only | 150 | 39 | 111 | 3 |
| **M1** + rehearsal | **139** | 34 | 105 | **1** |

Zero rows skipped for context overflow in any run. Supa's 161 is the frozen
reference measured on this host at this implementation digest (`a329eb7b…`), so
this is a paired comparison, not a comparison against a published figure from a
different stack.

## What the predictions said, and what happened

1. **"Both arms lose long-context ground against Supa's 161."** Correct. D3 −11,
   M1 −22.
2. **"M1 loses less than D3."** **Falsified, and in the wrong direction.** M1 is
   **11 rows worse** than the arm it was supposed to protect. Rehearsal did not
   merely fail to help here — it cost more than doing nothing.
3. Prediction 3 anticipated the case where the arms lose *equally* and said that
   would mean "rehearsal buys the panel and not the model". The actual result is
   stronger than that: rehearsal buys the panel **at the expense of** long context.

## The abstention collapse, which neither arm's gate would have caught

Supa says `UNKNOWN` on **13** of the 64 unanswerable rows. D3 says it **3** times.
M1 says it **once**.

`LONG_CONTEXT_RETENTION.md` already identified abstention as the capability Opallo
lost, and named it as one of the two measured, localized targets worth repairing.
Both arms here make it **worse than Opallo did** (Opallo: 2). Nothing in the D3
gate — panel ≥ 62, references retained ≥ 37 — is sensitive to this at all.

## A plausible mechanism, offered as a hypothesis and not tested

The rehearsal pool is the v3 preservation slice with the panel's three classes
removed: 53 remaining classes, weighted toward `communication` (3,000 rows) and
short conversational replies — `informal_texting`, `conversation_closure`,
`natural_follow_up`, `apology_repair`. Training on short, chatty, confident replies
is a plausible way to make a model *less* willing to answer `UNKNOWN` and *less*
careful over a 1,000-word context.

If that is right, the fix is not "less rehearsal" but "rehearsal that resembles
what you are trying to protect". **This is a hypothesis. It has not been tested,
and no arm here varies the rehearsal composition.**

## What this changes

**The RRP-009 recommendation is narrowed, not withdrawn.** Rehearsal does what its
own pre-registration claimed: it makes *panel* retention reliable (3/3 vs 2/3) and
it beats dose reduction at equal panel retention. Those numbers stand and are
replicated.

What must not be said is the broader thing the result was drifting toward —
"rehearsal lets the model learn this skill without catastrophically forgetting".
**On the one capability Supa is actually known for, both arms forget, and the
rehearsal arm forgets more.**

The honest statement of where the programme now stands:

- The new skill is real, large, and replicates exactly (289/384 three times).
- Panel retention is manageable, and rehearsal manages it reliably.
- **Long-context and abstention are damaged by every arm measured, and nothing in
  the registered gate detects it.**

## What should happen next, and what should not

**Should:** the gate needs a long-context and an abstention term before any of this
is promoted. A configuration selected on a gate blind to two-thirds of the damage
is not a configuration anyone should ship.

**Should not:** picking the arm that happens to look best on long context, now that
long context has been measured. D3 beats M1 here, but D3 was chosen for nothing and
this was a two-arm damage check, not a selection experiment. `QUARANTINE.json` bars
this suite for `[TRAIN, SELECTION, CONFIRMATION]` and this use was **report-only**
by registration. Selecting on it now would convert a legitimate check into the
post-hoc gate the bar exists to prevent.

## Limits

Single seed per arm; a damage check, not a claim. No acceptable-loss threshold was
pre-registered and none is invented here — inventing one after seeing the numbers
is the failure this programme repeatedly guards against, so the results are
reported against Supa's 161 and left as numbers. The mechanism in §4 is untested.
Both arms are reproductions from unchanged data contracts and plans, so they are
the models RRP-007 and RRP-009 scored.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-broad-retention.sh
```

Pre-registration with all four predictions in
`forge-v2-prereg-001/RRP010_BROAD_RETENTION_PREREG.md`, written before either arm
was scored. Artifacts `LONGCTX_broad_D3.json`, `LONGCTX_broad_M1.json`, each
carrying all 256 predictions; the Supa reference is `LONGCTX_supa.json` at the same
digest.
