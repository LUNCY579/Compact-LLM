---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Rineya-Supa-1.7.0-devfreeze
tags:
  - rineya
  - experimental
  - research-artifact
  - not-release-qualified
---

# Rineya Titan-Experimental 1.0.0-devfreeze

**Experimental research core. Not a release, not release-qualified, and not an
established replacement for Supa 1.7 or Opallo 1.0.**

Titan-Experimental is the RRP-015 clean control: exact Supa 1.7 fine-tuned on 384
solver-verified one-step state-transition examples, with **no rehearsal data**. It is
published here because it is the first arm in this programme that acquires a genuinely new
capability without measurably losing long-context performance. It still forgets things,
and it rests on a single training draw.

Stored in this private vault as a binary backup. Private vault storage is not a release
and carries no quality claim.

## What it does that the frozen cores cannot

Both frozen cores score **below the strongest answer-agnostic shortcut floor** on all
three transition families — Supa's `prerequisite_plan` and `register_machine` rates are
literally `0.0000`. This is not a marginal gain over a weak baseline; it is a capability
the source does not have.

| Endpoint | Supa 1.7 | Opallo 1.0 | **Titan-Experimental** |
|---|---:|---:|---:|
| New skill, SELECT-VIEWS-001 /384 | 14 | 8 | **234** |
| Families clearing published floor | 0/3 | 0/3 | **3/3** |
| Arithmetic /150, permissive | 47 | — | **69** |
| Arithmetic /150, strict | 41 | — | **62** |
| Source arithmetic answers retained /47 | 47 | 41 | 39 |
| Long context /256 | 161 | 135 | **162** |
| Paired exact McNemar vs Supa | — | — | **p = 1** |
| Literal `UNKNOWN` /64 | 13 | 0 | 10 |
| Gate v2 | — | — | **PASS** |
| Legacy per-class retention monitor | — | — | **FAIL** |

Per-family acquisition rates: `ledger_replay` 0.4219, `prerequisite_plan` 0.7422,
`register_machine` 0.6641. A family counts as cleared only if the exact two-sided binomial
95% **lower bound** exceeds its binding floor, where the floor is the stronger of
`copy_before` and `constant_majority`, published before any model call.

## Read this before believing the table

**Arithmetic 69 is not 47 preserved plus 22 new.** It is **39 preserved, 8 lost, 30 newly
gained**. The gain survives strict scoring (41 → 62), so it is not a formatting artifact —
but the legacy per-class monitor **fails on all three panel classes** under both scoring
policies, because source answers were lost in each.

**Long context net +1 hides real churn.** 11 source-correct rows were lost and 12 gained.
`p = 1` means the difference was not detected; it is **not** equivalence and **not** proof
of no forgetting. The comparison is properly matched — identical implementation digest
`a329eb7bb0`, transformers 4.51.3, `max_context` 4096, TF32 off, and 0 rows skipped in both
runs — but the suite is **SPENT** and barred for TRAIN/SELECTION/CONFIRMATION, so this was
used report-only and cannot confirm anything.

**Abstention regresses**, 13 → 10 literal `UNKNOWN`. That count is genuinely literal: the
strict and legacy substring counters agree at 10, with zero false abstentions on answerable
rows. Semantic abstention review was **not performed**.

**Rehearsal was tested and made things worse.** The matched treatment arm added 648
rehearsal rows at double the token budget and lost **27 long-context rows**
(paired p = 2.53×10⁻⁵) plus 4 more source answers. The absence of rehearsal here is a
deliberate, measured choice, not an oversight. This does not imply all replay methods are
harmful — only that the one tested was.

## Recipe

- Source: exact Supa 1.7, weight SHA256 `962abc5b…d24b`, 596,049,920 params
- Trainable: blocks 8–27 (tail 20 of 28); objective `assistant_response_plus_eos`
- 384 rows, 1 exposure, 54,172 processed tokens, **28 updates**, max rendered length 162 tokens
- CpuAdamW, peak LR 1e-5, WSD 20/60/20 to 0.1×peak, betas (0.9, 0.95), eps 1e-8, wd 0, clip 1.0
- FP32 training, BF16 export, selection seed 17
- Excludes all 768 scored parents — both the historical 384-row probe and the SELECT-VIEWS-001
  bank. Parent disjointness was rechecked before the worker loaded a model.
- In-run native qualification on the actual source: PASS

## Limitations

1. **One training draw.** No replication. No reliable success rate is established.
2. Two gates disagree on this model: Gate v2 PASS, legacy per-class monitor FAIL. Both are reported.
3. Gate v2 has **no long-context term and no abstention term**.
4. Long-context and abstention numbers are report-only on a spent instrument.
5. Stored BF16 but **evaluated in FP32** DenseReference. **Native BF16 serving is not qualified.**
6. No independent confirmation instrument exists for long context or abstention.
7. Human data review: **NOT PERFORMED**. Semantic abstention review: **NOT PERFORMED**.
8. SELECT-VIEWS-001 and the historical 384-row probe are **different instruments**. Supa scores 14
   on both by coincidence. Do not rank this core against D1/M4 by subtracting headline numbers.
9. No broad model-quality superiority over Supa or Opallo is established.

## Intended use

Research inference and as the replication target for further independent training draws.
Not for production serving, not a quality-validated core, and not a substitute for Supa 1.7
in any published system. The full resumable training checkpoint is retained separately, as
is the unsuccessful rehearsal arm's evidence.

## Provenance

`CORE.json` carries the complete record. `evidence/` retains the original custody,
materialization and preservation receipts that bind these bytes to the training checkpoint
and the exact source. Export weight SHA256:

```
537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7
```

Tokenizer and config assets are byte-identical to Supa 1.7.0-devfreeze.
Measured evidence: `docs/model-quality/audit-2026-09-17/rrp015-evidence/`.
Pre-registration frozen before outcomes: `RRP015_PREREG.md`.
