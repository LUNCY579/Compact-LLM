# Amendment v3 — Diagnose the Transport Operator Before Building Another One

Protocol version: `3.0.0-diagnostic`

Parents: v1 `preregistration.md` `1.0.0` (immutable) and
`amendment_v2_exploratory.md` `2.0.0-exploratory` (immutable)

Research branch: `research/rineya-cross-scale-qwen4b-8b`

Base commit of this amendment: `538687fb8ddab999f825e300e7c91d43947c5c4e`

## Visible evidence at freeze

v1 and v2 are both closed negatives and are not rewritten here.

- Only `P4-D2-W4-rank8-strength0.10` and `P5-D2-WDIR-rank8-strength0.10`
  produced coherent 0.6B-shaped donors.
- The v1 headline cross-donor statistics were cosine 0.932, sign agreement
  0.926, magnitude correlation 0.915 on the P4-s010 deltas.
- v1 mean consensus at alpha 0.15 won MERGEDEV by one case and tied DEV-FULL.
- v2 showed sign/cosine consensus, TIES, DARE, and DARE-TIES all collapse onto
  mean consensus, and that a P5 re-entry ties ungated Supa.
- v2 localized the failure to `T` but did not characterise `T`.
- The fresh sealed holdout has never been opened.

**Pre-freeze reconnaissance disclosure.** Before this file was written, the
`P4` and `P5` source code was read and a throwaway probe was run in a scratch
directory outside the repository. That probe reconstructed the `P4` delta
geometry in closed form from the already-committed
`tensor_projection_map.json` records and ran donor-substitution controls. It
therefore made HT1 and HT2 partly visible before freeze. E1 below is a
re-implementation of that probe as a versioned lab script; its findings are
labelled **post-hoc-visible** and are not presented as blind predictions. No
evaluation bank, no answer key, and no model weights were read by the probe.
HT3, HT4, and HT5 were not observed before this freeze.

## Host-state change recorded at freeze

Two facts changed since the v2 freeze and are recorded before any v3 command:

1. **Donor checkpoints are no longer resident.** The Hugging Face cache retains
   `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
   `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B` retain only lock directories; their
   snapshots are absent. Any v3 measurement that requires donor weights or
   donor activations is therefore blocked until a pinned re-download is
   authorised, and is recorded `blocked_no_donor_weights` rather than skipped.
2. **The installed Torch build is CPU-only** (`2.13.0+cpu`, `cuda.is_available()`
   is false). GPU is unavailable to this study regardless of card occupancy.
   All v3 work is CPU-first by necessity as well as by policy.

## Scientific objective

v2 concluded "the problem is `T`" without measuring what `T` transmits. v3 does
not build a new merge. It characterises the two transport operators that were
actually used, and establishes the target-space reference frame that any future
operator must hit.

v3 is a **diagnostic stage**. It is authorised to produce zero candidate
checkpoints. Producing none is a valid outcome.

## Hypotheses under test

- **HT1 (channel width).** `T` transmits a quantifiably small number of
  donor-derived scalars, and transmits no donor directional information. If
  true, the v1/v2 negative constrains only near-empty operators.
- **HT2 (operator-induced agreement).** Some of the observed cross-donor
  agreement is a property of `T` rather than of the donors. Measured against
  matched controls that hold `T` fixed and vary the donor.
- **HT3 (target-basis relevance).** The shared donor direction is not aligned
  with the target model's own useful adaptation direction
  `Delta_Supa = Supa - Qwen3-0.6B`, and cannot be distinguished from
  norm-matched, layer-shuffled, and sign-shuffled controls in that frame.
- **HT4 (localization).** Cross-donor agreement and target-basis relevance are
  not uniform over depth or module family.
- **HT5 (functional detectability).** The transported donors and the v1/v2
  candidates are functionally distinguishable from their base at matched
  relative depths on calibration prompts. If a candidate is functionally
  indistinguishable from Supa, its evaluation ties are uninformative about
  transfer rather than evidence against it.

## Experiments

| ID | Name | Inputs | Blocks on donors |
|---|---|---|---|
| E1 | Transport channel capacity | recorded projection maps | no |
| E2 | Target-basis alignment of the shared direction | on-disk deltas | no |
| E3 | Depth and module localization | on-disk deltas | no |
| E4 | Functional alignment atlas, 0.6B basis | 0.6B-shaped checkpoints | no |
| E5 | Cross-scale functional atlas, donor side | 4B/8B activations | **yes** |
| E6 | Low-rank functional translator | 4B/8B activations | **yes** |

E1-E4 run under this amendment. E5-E6 are pre-registered here but are recorded
`blocked_no_donor_weights` until a pinned donor re-download is authorised by the
operator. They are not silently dropped.

## Frozen measurement rules

- All parameter arithmetic is FP32 on CPU over memory-mapped safetensors,
  streamed one tensor at a time, with the tensor released before the next read.
- No two models are held in memory simultaneously in E1-E3. E4 holds exactly one
  model at a time and caches its summaries to disk before loading the next.
- E4 uses the frozen `calibration.jsonl` split only
  (SHA `d2f1f415e3cf34a8c7bb7a8cdef4240a8ce993c41d9b25a5d1471f19b8f453a1`),
  prompts only, greedy, thinking disabled, batch size 1, seed 17, float32.
- E4 captures hidden states at the frozen relative depths
  `{0.00, 0.25, 0.50, 0.75, 1.00}` of the 28-layer stack, that is layers
  `{0, 7, 14, 21, 27}` plus the embedding output, at the final prompt token
  only, together with full final-token logits.
- Every E1-E3 similarity statistic is reported beside at least three controls:
  norm-matched random, layer-shuffled, and sign-shuffled.
- Every experiment records wall time, peak RSS, model-load count, and output
  artifact hashes.

## What v3 may and may not claim

v3 may claim:

- a measured description of what `P4` and `P5` transmit;
- whether observed cross-donor agreement survives donor-varying controls;
- whether the shared direction is distinguishable from controls in target space;
- whether existing candidates are functionally distinguishable from Supa.

v3 may **not** claim:

- sealed confirmation of anything;
- that the cross-scale hypothesis is settled in either direction;
- that a new operator works, since v3 builds none.

## Evaluation gates

v3 stops at Gate 1 (parameter and representation diagnostics) by construction.
Gate 2 tiny development inference is used in E4 only as a representation probe
on calibration prompts, never as a selection signal. Gates 3-5 are not entered.
DEV-FULL is not run. The sealed holdout remains CLOSED and is not read.

## Leakage

Invisible-Limit TEST2, Stage 6 outputs, MERGEDEV, MERGEDEV-fast, DEV-FULL, and
the sealed holdout are denylisted for every v3 command. E4 reads calibration
prompts and never reads calibration answers, scorers, or any MERGEDEV/DEV-FULL
row. v3 introduces no new selection over any evaluation bank.

## Amendment policy

Inherited from v1. This file is immutable after the first v3 measurement
command. Corrections are new timestamped files.
