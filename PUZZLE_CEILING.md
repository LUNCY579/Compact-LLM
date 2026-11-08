# The reasoning ceiling is not a long-context artifact

**P1 verdict: STOP.** The pre-registered rule required 119 of 450 hits to proceed. Supa
scored **102**. No puzzle training split will be built.

---

## What was being tested

The previous session concluded this 596M model has a **capacity** ceiling on reasoning,
because nothing moved the Reasoning Forge — not schedule, depth, dose, architecture, weight
averaging, or an 11.5× dose of purpose-built reasoning data.

I did not believe that conclusion was earned, and said so. It had an uncontrolled confound:
forge prompts run **~1,400 tokens** against training rows of **~128**, so "cannot reason"
and "cannot reason *over a long record*" had never been separated. Every reasoning null in
this programme came from one instrument with one prompt length.

`Rineya-Puzzle-Reasoning-51k` separates them. Prompts are **~90 tokens** — fifteen times
shorter than forge — every answer is a short string, and every row is solver-verified to
have exactly one solution. Split disjointness was verified here on `id`, `prompt_sha256`,
`template_id`, `structural_signature` and raw prompt text (all zero), with zero panel stems
in its TRAIN.

## Result

450 rows, 30 from each of 15 families, frozen with a sha256 before either model ran.

| | hits / 450 | rate | guessing floor |
|---|---:|---:|---:|
| Supa 1.7 (base) | **102** | 0.227 | **0.229** |
| Opallo 1.0 | 96 | 0.213 | 0.229 |

**Supa is at the guessing floor.** The floor is the majority-answer baseline computed from
the frozen sample before scoring — what a model that always answers each family's most
common answer would score. It is 0.229 because `boolean_switches` and `knights_knaves` are
two-way and `rule_deduction` is three-way; chance on this instrument is emphatically not
zero, and an aggregate quoted without it would look like 23% accuracy instead of nothing.

Opallo versus Supa on the same rows: 24 Supa-only, 18 Opallo-only, **p = 0.44**. Training
neither gained nor cost puzzle reasoning.

## So the confound is ruled out

Prompt length was the alternative explanation and it does not survive. At one-fifteenth the
prompt length, with verified unique answers and a short-string output format the model is
already good at, performance is still indistinguishable from guessing.

**Two independent instruments now agree**, and they fail in different ways for different
reasons, which is what makes the agreement worth something. The capacity reading stands —
not because the earlier session asserted it, but because the best available attempt to
break it failed.

## The per-family structure, which the aggregate hides

Reported because it was pre-registered, and because "the model scores 22.7%" is a different
claim from "the model does nothing."

| family | Supa | floor | p (one-sided exact) |
|---|---:|---:|---:|
| **logic_grid** | 12/30 | 0.133 | **0.0003** |
| epistemic_hats | 16/30 | 0.333 | 0.0188 |
| rule_deduction | 17/30 | 0.433 | 0.0991 |
| … nine families between | | | 0.14 – 0.97 |
| family_relations | 1/30 | 0.200 | 0.999 |
| **code_breaking** | **0/30** | 0.067 | 1.000 |
| **hanoi_planning** | **0/30** | 0.167 | 1.000 |

Bonferroni over 15 families requires p < 0.0033. **Only `logic_grid` survives.**
`epistemic_hats` at p=0.0188 does not, and would be the eighth post-hoc finding this
programme has had to withdraw if quoted as one.

Two families score **literal zero** — `code_breaking` and `hanoi_planning`, which need
multi-step search and state tracking. That is the shape of the ceiling: not uniform
mediocrity, but one narrow constraint-satisfaction family the model can partly do and a
planning/search capability that is simply absent.

## What this changes about the dataset plan

It removes an option rather than opening one, which is the useful thing a negative result
does.

* **Do not build a puzzle training split.** The pre-registered rule fires and the follow-up
  evidence agrees with it.
* **`logic_grid` is not a loophole.** Selecting the one family of fifteen where the model
  already beats its floor, *after* seeing which family that is, is exactly the post-hoc
  selection this programme has been burned by. If it is worth pursuing it needs its own
  pre-registration and its own held-out families.
* **"Teach it to reason" is now falsified on two instruments.** Effort should go where
  there is measured signal: the arithmetic panel, where 47 → 73 already happened, and
  off-distribution output discipline, which is a real measured weakness with a corpus
  built for it (`Rineya_Short_Response_Behavior_Corpus_v1.0`).

## Reproduce

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
$PY experiments/rab001_exploration/puzzle_probe.py --build-sample
$PY experiments/rab001_exploration/puzzle_probe.py --model Supa --label supa_cpu --device cpu
$PY experiments/rab001_exploration/puzzle_probe.py \
   --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo_cpu --device cpu
```

Sample `PUZZLE_SAMPLE_test450_seed17.jsonl`, sha256 `f76a1351…`, checked on every scoring
run. Results in `PUZZLE_supa_cpu.json` and `PUZZLE_opallo_cpu.json`. Scored on CPU because
another session held the GPU; puzzle prompts are short enough that this costs ~5 minutes,
and CPU is this programme's frozen-reference device anyway.
