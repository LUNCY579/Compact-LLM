# TRISKILL-001, Phase 0 — source-competence screen (registered before any model reads a row)

Written 2026-09-26, before any arm has generated on any SIF-001 row. It selects the **candidate
third skill** for TRISKILL-001. It is a build gate, not an acquisition, retention or quality
measurement. No training, no composition arm, no confirmation bank.

## Why this screen comes first

TRISKILL-001 asks whether a **third, independently trained generator skill** can be added onto
Titan-Transition (`Titan-Sparse + 0.25·P16(δ_H)`, sha `a4d73353…`) at its own coefficient without
measurable loss of the transition skill (T) or the boundary skill (H). The composition synthesis
(`COMPOSITION_SYNTHESIS_20260921.md`) names "a second GENERATOR" as the programme's highest-value
missing experiment; δ_H is a gate skill.

A new donor is only measurable if the source **lacks** the skill. Titan qualified because Supa
scored below the answer-agnostic floor on every transition family. This screen checks the same
precondition for the SIF-001 atomic operations before a training grant is requested.

## Instrument

SIF-001 (`C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1`),
**DEV split only**, the five atomic banks A, C, P, R, S: 128 rows each, 640 rows. DEV's registered
role in the dataset is "parser/difficulty development, exploratory inspection". CAL (selection)
and CHECK (reserved) are **not read** by this screen.

- Headline population per bank: the **96 JSON-contract rows** (variants full, counterfactual,
  distractor). The 32 `alternate_format` rows (16 tag, 16 plain) are reported separately.
- Scoring: SIF-001's own `src/sif/scoring.py` (`score_records`), imported, not reimplemented.
  The headline is **strict** correctness (the requested format **and** the right value). Semantic
  correctness under another clean wrapper is a diagnostic column.
- Prompt hashes: all 640 input prompts were checked against their keys before registration
  (sha256 of the user message); 0 mismatches.

### Answer-agnostic floors, computed from the keys before any model call (96 JSON rows)

| bank | operation | majority answer | floor rows | floor rate |
|---|---|---|---:|---:|
| A | arithmetic | 3 | 23 | 0.2396 |
| C | rules with exceptions and priority | 6 | 18 | 0.1875 |
| P | min-cost directed path, mod 7 | 0 | 22 | 0.2292 |
| R | evidence resolution (with UNKNOWN) | 1 | 20 | 0.2083 |
| S | state tracking (with UNKNOWN) | 1 | 17 | 0.1771 |

Floor = max(majority constant, echo the seed). Echo-seed is at most 6 rows on every bank.

## Arms (existing artifacts only; each weight hash verified before load)

| arm | directory | weight sha256 |
|---|---|---|
| BASE | `Rineya-Supa-1.7.0-devfreeze` (BF16, loaded as FP32; exact upcast) | `962abc5b…dd24b` |
| SPARSE | `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` (FP32) | `a28629c9…1284` |
| TT | `Rineya-Titan-Transition-1.0.0-devfreeze` (FP32) | `a4d73353…9567` |

Runtime: hermes venv (python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1), FP32, TF32 off,
greedy, batch 1, `max_new_tokens` 48, Qwen3 chat template with thinking disabled, rendered byte-
exactly by the soup lane's self-tested `render`. One GPU lease (`rineya-gpu-0.lock`) for the window.

## Registered decision rules

**G0-HEADROOM(X).** Skill X has headroom iff BASE's strict score on X's 96 JSON rows is
**≤ 48 (0.50)**. It can fail: a rule lookup or a short path may already be within Supa's reach.

**Candidate order, fixed now: C, then P.**
- C is first because it is the only atomic operation with no UNKNOWN answers, so it does not share
  a surface with δ_H's abstention skill. It also shares no operation with Titan's three transition
  families (`ledger_replay`, `prerequisite_plan`, `register_machine`).
- S (state tracking) and A (arithmetic) overlap Titan's families and the arithmetic panel.
- R has an UNKNOWN answer class and so overlaps δ_H.
- A, R and S are scored as transfer diagnostics only and can never be selected.

**Selection.**
- If G0-HEADROOM(C) passes, C is the candidate.
- Otherwise, if G0-HEADROOM(P) passes, P is the candidate.
- Otherwise TRISKILL-001 stops at Phase 0 and reports that SIF-001 has no measurable-headroom
  generator skill for Supa.

No other bank, variant population or threshold is admitted after the read.

**Descriptive, registered now so they are not chosen later:**
1. TT − BASE and SPARSE − BASE on every bank, strict, with paired exact McNemar p. This tells
   whether Titan's transition delta already transfers to an operation, which would mean part of
   that "new" skill is already present in Titan-Transition. It is expected mainly on S and A.
2. Format validity (the fraction of parseable JSON with key `answer`) per arm and bank. This
   separates contract failures from content failures.
3. The alternate-format rows per arm.

A TT − BASE gain of ≥ 20 rows (p < 0.05) on the selected skill is flagged in the result as
"partially present before donation". The main experiment already measures acquisition over TT,
so it does not stop the screen.

## What this screen cannot show

It cannot show that Supa can **learn** the skill from 28 updates, which only training can show.
It cannot rank skills; 96 rows is go/no-go resolution. It says nothing about retention.

## Canaries (all must pass before the first arm loads, else abort with no reads)

1. The template self-test reproduces the recorded prompt bytes, and a negative control rejects a
   missing think block.
2. The scorer canary: every gold completion scores strict-correct; `{"answer":99}` scores 0 strict
   on every row; a trailing period `{"answer":N}.` is a format error on the JSON rows.
3. Weight sha256 equals the table above for all three arms.
