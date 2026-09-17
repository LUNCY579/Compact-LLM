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
  - fp32-only
---

# Rineya Titan-Sparse-Experimental 1.0.0-devfreeze

**Experimental research core. Not a release, not release-qualified, and not an established
replacement for Supa 1.7 or Opallo 1.0.**

Supa 1.7 plus **the largest 25% by magnitude of the RRP-015 control's fine-tuning update**,
selected per tensor. **No new training** — this core reuses an existing update and changes
only which part of it is kept. It was chosen by a rule registered in
`evidence/SPARSE001_PREREG.md` *before any arm was built*.

## ⚠ This core is FP32 and must not be converted to BF16

Every other core in this vault is BF16. This one cannot be, and the reason is measured, not
stylistic. Its identity *is* a magnitude-selected subset of an update whose total size is
only **0.056% of the weight norm**. Rounding it to BF16 would:

- round **28.49%** of the kept delta elements to **zero**, and
- distort the retained delta norm to **1.0347** through quantization noise.

A BF16 copy would be a different and measurably worse model, and none of the numbers below
would apply to it. `config.json` therefore declares `torch_dtype: float32` so that loading
with `torch_dtype="auto"` cannot silently discard the update. This is the only sidecar that
differs from Supa 1.7.0-devfreeze.

## Measured

| Endpoint | Supa 1.7 | Titan-Experimental | **This core** |
|---|---:|---:|---:|
| New skill, SELECT-VIEWS-001 /384 | 14 | 234 | **232** |
| Families clearing published floor | 0/3 | 3/3 | **3/3** |
| Arithmetic /150, permissive | 47 | 69 | **72** |
| Arithmetic /150, strict | 41 | 62 | **63** |
| Source answers retained /47 | 47 | 39 | **41** |
| Long context /256 | 161 | 162 | **164** |
| Literal `UNKNOWN` /64 | 13 | 10 | **12** |
| Gate v2 | — | PASS | **PASS** |
| Legacy per-class monitor | — | FAIL | **FAIL** |

Per-family acquisition: `ledger_replay` 0.4141, `prerequisite_plan` 0.7500,
`register_machine` 0.6484, against floors 0.2344 / 0.0781 / 0.0703 published before any model
call. A family counts as cleared only if the exact two-sided binomial 95% **lower** bound
exceeds its floor.

Against its sibling it is better on four endpoints and equal within noise on acquisition. Its
**164/256 is the highest long-context score of any model measured in this programme, above
the frozen source's own 161** — but see the limits below before reading anything into that.

## Why this works, and where it came from

The fine-tuning update is strongly heavy-tailed: **25% of its elements carry 84% of its
norm**. Two screens established that the new capability rides the update's large-magnitude
**head** while the long-context damage rides its **tail**:

- At the same nominal 0.50, keeping the top half at full strength scores **235** on
  acquisition where shrinking every element by half scores **219**. Pruning preserves the
  skill that scaling dilutes.
- The sibling core's own BF16 export was an accidental, crude version of this operation, and
  it is why that core scored *better* on long context (162) than its own full-precision
  parent (157). See `Rineya-Titan-Experimental-1.0.0-devfreeze/AMENDMENT_001.json`.

## Read this before believing the table

**Arithmetic 72 is not 47 preserved plus 25 new.** It is **41 preserved, 6 lost, 31 newly
gained**. The gain survives strict scoring (41 → 63), so it is not a formatting artifact —
but the legacy per-class monitor **fails**, as it does for every arm of every screen here,
including arms that lose only 4 of 47 answers. That monitor is near-unpassable and does not
discriminate among these candidates.

**Long context net +3 hides churn**: 7 source-correct rows lost, 10 gained, p = 0.63.
Nonsignificance is not equivalence. The suite is **SPENT** and barred for
TRAIN/SELECTION/CONFIRMATION, so it was used report-only and **did not select this core** —
selection used SELECT-VIEWS-001 and the arithmetic/retention panel alone. 164/256 is
*observed*, never confirmed.

**Abstention still regresses**, 13 → 12.

**One training draw.** The update direction comes from a single seed. Pruning cannot
manufacture replication.

## Reproducing it

Deterministic, no model runs, no training:

```
python evidence/build_sparse.py --keep 0.25 --out <dir>
```

from source `962abc5b…` and the consolidated trained tensors `07156fb5…`. Two independent
builds produced byte-identical output. Weight SHA-256:

```
a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284
```

The mask is per-tensor top-k by `|delta|` with ties broken by ascending flat index via a
stable descending argsort. An earlier implementation used `torch.topk(sorted=False)`, which
does not define which of several equal magnitudes it keeps; a full deterministic
re-measurement then agreed with the first pass on **every endpoint of every arm**, so the
tie-break has zero measured effect. Both the defect and that control are recorded in
`evidence/SPARSE001_PREREG.md` Erratum 1.

## Intended use

Research inference, and the arm to carry forward into independent training draws. Not for
production serving, not quality-validated, and not a substitute for Supa 1.7 in any published
system. `release_permitted` and `quality_improvement_established` are both false.

## Limitations

1. One training draw; no replication.
2. Gate v2 PASS alongside legacy per-class monitor FAIL. Gate v2 has no long-context and no
   abstention term.
3. Long context and abstention are report-only on a spent instrument.
4. FP32 only. Native BF16 serving is **not** qualified and a BF16 copy is a different model.
5. No independent confirmation instrument exists for long context or abstention.
6. Human data review and blind semantic abstention review: **NOT PERFORMED**.
7. No broad model-quality superiority over Supa or Opallo is established.

Codename assigned by the promoting session for clarity, not by the operator; `RENAME.json` is
the vault's supported mechanism if a different roster name is preferred.
