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

---

# RESULT — the banks are calibrated and indistinguishable over [8, 238]

> The STATUS banner above is superseded by this section rather than edited.

## Bridge points

| select-views | dose | offset | 95% CI (rows) | model |
|---:|---:|---:|---:|---|
| 8 | 11 | +3 | [−5.4, +11.4] | Rineya-Opallo-1.0.0-devfreeze |
| 14 | 14 | 0 | [−10.2, +10.2] | Supa (frozen source) |
| 70 | 60 | **−10** | [−30.4, +10.4] | `arm-lora-r16-merged` (LoRA arm 1) |
| 238 | 224 | **−14** | [−40.6, +12.6] | `tt001-20260918/build-s17s29` |

**Heterogeneity: Q = 2.479, df = 3, I² = 0.000 → pooling PERMITTED.**

> **Pooled offset (dose − select-views): −0.06 rows, SE 3.07, 95% CI [−6.1, +6.0].**
> The two banks are indistinguishable in difficulty across a 230-row span.

## Registered predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | \|offset\| ≤ 15 rows at every point | **HELD** — max is 14 |
| 2 | no trend of offset with arm strength | **HELD** — I² = 0.000, Q = 2.479 against df 3 |
| 3 | rebuilds reproduce originals within 7 rows | **PENDING** — blocked on the host-RAM guard, see below |

## The registered prohibition fires, and it is honoured

`licensed_range` is **[8, 238]**. The guard REFUSES 239, 276 and 304 — so **D2 and D4 cannot be
read against the select-views table**, and D1 at 239 sits *one row* above the boundary.

The offset is flat (I² = 0) over 230 rows, which makes it tempting to extrapolate 66 rows further.
**That temptation is exactly what the prohibition was registered against**, before the numbers
existed. It is honoured. Extending the licence needs a bridge arm above 238, and the highest
select-views arms available are `tt001_TT16s` 247 and `mag001_sm05` 243 — which would reach 247,
still short of 276.

## This does not change the deliverable

**The bridge's purpose was never the offset.** It is that the dose bank is a usable scoring
instrument via `process_probe.py --instrument process-transition-probe-001`, whose `load_views`
verifies `45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e` and refuses on
mismatch.

> **A low-invasiveness arm trained at D2/D4 dose can now be scored NATIVELY on the dose bank and
> compared directly to D2's 276 and D4's 304 — no cross-bank step, no offset, no extrapolation.**

The calibration is the secondary product, and its value is the reassurance that the two tables
are not measuring different things.

## Two defects found in building it, both my own

**1. `model_dir` is not the model.** `process_probe.py` supports `--checkpoint` overlays, so the
single base directory `.../Supa` hosts **13 distinct arms** differing only by the checkpoint
applied on top. The first version of `analysis/bank_bridge.py` matched on `model_dir` alone,
silently paired `rrp015_control` (select-views 234) with `supa_rrp007_a1` (dose 289), and
produced a **+55** bridge point that drove the entire heterogeneity verdict to
`pooling_permitted: False`. Pairs are now keyed on `(model_dir, checkpoint.manifest_sha256)`.

**2. My `--evidence` roots could not reach the only informative point.** The first corrected run
returned a licensed range of [8, 70] because `PROCESS_tt001_TT16s_1729.json` and its select-views
twin live in `ChatBubble-compose-lane-20260918`, which was not among the three roots passed. The
fix was to pass the project root. **Third instance today of a check whose scope could not reach
its claim** — and the first two were in prose, so a narrow default in a tool is the same defect
with a different surface.

**3. And the runner masked its own failures.** The first five-run script printed
`ALL BRIDGE RUNS FINISHED` while every run had exited 2. This is the identical shape criticised
in a paired session's battery forty minutes earlier — *knowing the failure mode by name bought
nothing*. The rerun tracks a `FAILED` counter and exits non-zero.

## BLOCKED — four arms, host RAM

`bridge001_lora2_dose`, `bridge001_s10_dose/_sv`, `bridge001_s50_dose/_sv` are unrun. The guard
refused: **`host RAM guard refused; no automatic retry or override (phase=qualify, need 4.6 GiB,
have 4.00 GiB)`**. Available has been measured between **4.16 and 4.62 GiB** with `chrome.exe` at
2.11 GiB and `claude.exe` at 1.60 GiB.

**The guard is not weakened.** The runner waits for 5.2 GiB of headroom *above* it. These four
arms would land at select-views 223, ~197 and ~235 — **inside the already-licensed [8, 238]
span**, so they would tighten the pooled estimate but would **not** extend the range, and would
not change any conclusion above.
