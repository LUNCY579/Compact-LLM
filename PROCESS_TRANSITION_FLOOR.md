# Neither core can do one state transition. Both score below copying the input.

**Result: no family cleared, on either core.** Supa 1.7 and Opallo 1.0.0 were
scored on 384 frozen one-step transition views — 128 each from `ledger_replay`,
`register_machine` and `prerequisite_plan`, one view per parent — and neither
reaches the shortcut floor on any of the three.

| | Supa | Opallo | binding floor | clears at |
|---|---:|---:|---:|---:|
| `ledger_replay` | **14/128 = 0.1094** | 10/128 = 0.0781 | 0.2422 `copy_before` | k ≥ 40 |
| `register_machine` | **0/128 = 0.0000** | 1/128 = 0.0078 | 0.0547 `copy_before` | k ≥ 12 |
| `prerequisite_plan` | **0/128 = 0.0000** | 0/128 = 0.0000 | 0.0703 `constant_majority` | k ≥ 15 |
| aggregate | 14/384 = 0.0365 | 11/384 = 0.0286 | 47/384 = 0.1224 | — |

On `ledger_replay` Supa's exact 95% interval is **[0.0611, 0.1767]**, entirely
**below** the 0.2422 floor. A predictor that ignores the question and echoes the
supplied state unchanged beats the model, significantly, on its best family.

Paired on identical rows: 4 Supa-only correct, 1 Opallo-only, exact two-sided
McNemar **p = 0.375**. Training neither gained nor cost this capability.

## Why this instrument answers something the others could not

Every reasoning null in this programme carried a confound. Forge v1 prompts run
~1,400 tokens; the puzzle probe ruled out prompt length at ~90 tokens but tested
whole multi-step puzzles; the v2 contract work confounded reasoning with a JSON
schema the model had never seen.

These prompts are **~150 tokens**, carry exactly two keys (`before` and `event`),
and ask for one rule applied once. All 36,752 source labels were independently
re-derived by `oracle.replay_transition` and agreed. Floors were published before
any model was loaded, and two of them were corrected — openly — before the first
call.

## The decomposition, which is the actionable part

The aggregate hides three different failures, and only one of them is what it
looks like.

**`ledger_replay` — the structure is learned; the arithmetic is not.**
114 of 128 Supa predictions parse to a dict with the **exact key set** of the
reference. Among those: 14 fully correct, **36 with exactly one value wrong**, 63
with two wrong, 1 with three.

```
ref  : {"A":-69,"B":18,"C":3,"D":-34}
pred : {"A":-69,"B":60,"C":15,"D":-34}      A and D right, the transfer wrong
ref  : {"A":30,"B":-4,"C":7,"D":15}
pred : {"A": 30, "B": -4, "C": 3, "D": 15}  one account wrong
```

The model copies every untouched account correctly and misses the field the event
touches. The dominant 63-row "two wrong" bucket is the signature of `transfer`,
which moves value between two accounts and requires getting both sides right.
This is **not** a comprehension failure and **not** a format failure. It is
arithmetic precision on the one field that changes.

**`register_machine` — the contract is never reached.** 0 of 128 Supa predictions
even produce the right key set. It echoes its own input instead:

```
ref  : {"X":79,"Y":73,"Z":82}
pred : {"before": {"X": 79, "Y": 73, "Z": 39}, "after": {"X": 79, "Y": 73, "Z": 39}}
pred : {"before":{...},"event":{"kind":"add","other":"Y","register":"Z",...}}
```

Opallo manages 16/128 on key set. Registers are integers **modulo 97**; nothing
in either core attempts the reduction.

**`prerequisite_plan` — genuinely absent, and not a contract artifact.** The
reference is a bare integer; both cores wrap it (`{"after": 27}` against a
reference of `31`). A **post-hoc** tolerant rescore that unwraps
`after`/`state`/`result`/`value` lifts it only to **1/128** — still far below the
9/128 constant floor. Stripping the contract does not rescue it; the bitmask
update itself is wrong.

## A new, measured Opallo regression

**Supa produces parseable JSON on 351 of 384 views; Opallo on 182.** Opallo is
roughly half as likely to emit anything a JSON parser accepts, on an instrument
neither core was trained for. Strict-exact hits are 4 for Supa against 1 for
Opallo, with an identical presentation-only gain of 10 each.

This is consistent with, and independent evidence for, the output-discipline loss
already recorded in `LONG_CONTEXT_RETENTION.md`, measured here on a fresh
instrument with no overlap with the Demo suite.

## What this licenses, and what it does not

**Does not license:** any claim that the model "cannot reason". This bounds
**single-step state-update precision**. It is the atomic unit of the multi-step
search and state tracking the puzzle probe measured at literal zero, but it is
not that capability, and clearing it would not by itself be a reasoning result.

**Does license** a sharper description of the ceiling than this programme has had.
The failure is not uniform: on `ledger_replay` the representation is right and one
number is wrong. That is a narrow, named deficit with 36,752 solver-verified
examples sitting in the repository at ~80–150 tokens per row, well inside the
qualified 2048 training window, against a published per-family floor — and with
both shipped cores currently **below** the trivial shortcut.

It is the best-posed training target in this programme's history, because for the
first time the instrument, the floor, the failure mode and the training data all
name the same thing. It is also the first target where a null would be
informative rather than uninterpretable: the cores are not at chance here, they
are **below** it, so there is room to move in a direction that a shortcut cannot
fake.

## Reproduce

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$PYG experiments/rab001_exploration/process_probe.py --model Supa --label supa
$PYG experiments/rab001_exploration/process_probe.py \
     --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo
```

Views `process-transition-probe-001/CANDIDATE_VIEWS.jsonl`, sha256
`45674251e38e…`, checked on every run and against its own build audit. Floors and
both pre-run amendments are in
`rineya-dataset-retention-records/forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md`.
Results `PROCESS_supa.json` and `PROCESS_opallo.json`, each carrying all 384
predictions. Zero optimizer steps; frozen sources unchanged. The tolerant
`prerequisite_plan` rescore and the key-set decomposition were **not**
pre-registered and are reported as mechanism, not as significance claims.
