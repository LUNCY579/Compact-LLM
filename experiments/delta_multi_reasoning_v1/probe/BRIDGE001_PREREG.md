# BRIDGE-001 — calibrating the dose bank against select-views, so D2/D4 can be tested

> **STATUS: REGISTERED, NOT RUN.** No arm has been scored on the dose bank by this session as of
> this line. Written before the first generation, because a bank-offset measured after seeing the
> arms it is meant to license is not a calibration.

**Class: MECHANISM SCOUT / instrument calibration. REPORT-ONLY and it cannot become anything
else** — `process-transition-probe-001` declares `view_role: CANDIDATE_NOT_ADMITTED_OR_TRAINED`
and *"never for selection"* in all 26 of its evidence files. **Nothing measured here may select,
promote, or rank an artifact.** It exists only to make two existing tables readable against each
other.

## Why this is needed

The programme's two acquisition tables live on **different banks** and nobody noticed until
today:

| table | bank `views_sha256` | contents |
|---|---|---|
| parameterization / transform | `7dd53f2a751b54c3…` (select-views-001) | every α, k, r, λ, rotation and LoRA arm — 47 labels |
| **dose** | `45674251e38e5742…` (process-transition-probe-001) | RRP-008 D1 239, D2 276, D3 289, **D4 304** — 28 labels |

Both are 384 views, 3 families × 128, and **both report the same implementation digest
`a329eb7bb06f`** — so there is no runner confound and nothing but the bank hash distinguishes
them. A LoRA-at-D2 arm scored on `7dd53f2a` would have **no matched comparator**, and comparing
it to D2's 276 would repeat the error this session already caught once.

**The good route is unavailable.** Re-scoring an RRP-008 checkpoint on `7dd53f2a` would put the
dose ladder on the parameterization bank directly, but `DISK_RECLAIM_001.md` records that each
rung deletes its own checkpoints after scoring, and **none survives** (verified: no
`checkpoint-*` or `.safetensors` under any `dose-D*-journal*`). So the bridge must be built from
the other side: **score models that already have a `7dd53f2a` score on the `45674251` bank.**

## What already exists, and why it is not enough

Three models have been scored on **both** banks with an identical `model_dir`:

| model | dose `45674251` | select-views `7dd53f2a` | dose − sv | 95% CI (rows) |
|---|---:|---:|---:|---:|
| Supa (frozen source) | 14 | 14 | 0 | [−10.2, +10.2] |
| Opallo | 11 | 8 | +3 | [−5.4, +11.4] |
| **TT16s_1729** | **224** | **238** | **−14** | **[−40.6, +12.6]** |

Different item sets, so this is an **unpaired two-proportion** comparison, not McNemar.

**One informative point, and it is underpowered** — z = −1.03, p ≈ 0.30. Two of the three sit at
the floor where both banks are uninformative. The direction (*dose bank not easier*) is the
conservative one for the dose ladder, but **nothing is established**.

## Design

Score models with a known `7dd53f2a` score on the `45674251` bank, via the existing registry:
`process_probe.py --instrument process-transition-probe-001`. The bank file verifies to
`45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e` and `load_views` refuses on
mismatch, so the instrument gates itself.

**The byte-identity requirement, which costs two runs for two of the arms.** A bridge point
requires *the same model* on both banks. `arm-rebuild-s10` and `arm-rebuild-s50` are **NOT**
byte-identical to the `arm-s10` / `arm-s50` that produced the 197 and 235 on select-views:

    arm-s10          out_weight_sha256 2d27f118…   retained-norm 0.6383084667498163
    arm-rebuild-s10  out_weight_sha256 d7b15d2e…   retained-norm 0.6383084679413847

The norm differs at ~1.2e-9 — accumulation order, consistent with RANK-001 Erratum 2 — but
**different bytes are a different model** and pairing a rebuild's dose score with an original's
select-views score is the substitution this programme keeps getting burnt by. So the rebuilds are
scored on **both** banks; only then is their pair internally valid.

| arm | live weights | sv score | runs needed |
|---|---|---:|---|
| `arm-lora-r16-merged` (LoRA arm 1) | yes | **70** (exact dir) | dose only |
| `arm-lora-r16-lr1e4-merged` (LoRA arm 2) | yes | **223** (exact dir) | dose only |
| `arm-rebuild-s10` | yes | — (original scored 197, different bytes) | **dose + sv** |
| `arm-rebuild-s50` | yes | — (original scored 235, different bytes) | **dose + sv** |

Six runs, ~5 min each at the measured 281.8 s/384 rows, ≈ 30 min GPU. Combined with the three
existing points this gives **seven**, spanning roughly 8 → 238 on the select-views scale.

## Registered predictions

1. **|offset| ≤ 15 rows at every measured point.** Basis: the two banks' published floors are
   near-identical (ledger `copy_before` 31 vs 30, register 7 vs 9, prereq `constant_majority`
   9 vs 10), which is what near-equal difficulty looks like from the build side.
2. **No systematic trend of offset with arm strength.** If the offset grows with score, a single
   constant is unusable and the bridge is a **curve**, not a number.
3. **The rebuilds reproduce their originals within the 7-row acquisition floor** on select-views —
   s10 within 7 of 197, s50 within 7 of 235. This is a free apparatus check on the rebuild
   construction and on the bank itself.

## What would falsify, and what each failure means

- **|offset| > 15 rows anywhere** → the banks are not interchangeable at that level, and the
  dose ladder cannot be read against the parameterization table without a per-level correction.
- **Offset varies with arm strength** (heterogeneity across the seven points) → **the bridge is
  not a constant.** Report it as a curve and refuse any single-number correction.
- **A rebuild misses its original by more than 7 rows on select-views** → the rebuild
  construction is not faithful, and both its bridge points are withdrawn rather than explained.

## Analysis rules, fixed now

- **Per-arm differences are primary. Never pool without a heterogeneity check** — pooling a
  strength-dependent offset into one number is exactly the pooled-floor failure.
- **Unpaired two-proportion CIs**, because the banks hold different items. McNemar is
  inapplicable here and using it would overstate precision.
- A null on the trend reports as **"not powered to distinguish"** unless the MDE says otherwise.
  At n = 384 per bank the per-point 95% CI is roughly ±27 rows in the middle of the range; that
  is wide, and seven points is a small sample of arms. **This bridge will bound the offset, not
  pin it.**
- **The offset is measured on the SELECT-VIEWS scale** and is only licensed over the range it was
  measured on. D4's 304 sits **above every point in this design** — the strongest arm available
  is ~238. **Extrapolating the offset to 304 is forbidden**, and this is registered now so it
  cannot be waved through later: it is structurally the cue-differential error, a quantity
  measured over one range applied to a range it was never measured on.

## What this bridge does and does not enable

**Enables:** a future low-invasiveness arm trained at D2/D4 dose can be scored on `45674251` and
compared **directly** to D2's 276 and D4's 304, with no cross-bank step at all. That is the real
deliverable — the calibration is the secondary product.

**Does not enable:** reading D4's 304 as a select-views number. Nothing here licenses that, and
prediction 3's range limit is the reason.
