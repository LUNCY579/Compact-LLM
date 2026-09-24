# FUNCTIONAL-ADDITIVITY-001 — Amendment 1 plan (POST-HOC, written before the re-score)

Written 2026-09-23, **after** the registered run (`TF_RAW.json`), its analysis, and a five-audit
review. **Nothing here is registered.** Every number this amendment produces is post-hoc and must
be labelled that way. The registered primary and its label (ADDITIVE) stand as scored.

Authorised in-session by the operator on 2026-09-23 ("Exact re-score, ~2 GPU-min"). The run uses
the five existing arms, teacher-forced only, under the shared GPU lease.

## Why

1. **The clip premise failed.** PREREG said the ±10 clip only touches irrelevant tails. It caps
   21.2% of SPARSE's content positions but only 7.0% of ADD-100's, which pulls I toward 0.
   `TF_RAW` stores only *clipped* log-odds, so the unclipped value has so far been rebuilt from
   float32 log-probability. That rebuild is imprecise above z ≈ 12.
2. **Registered secondary 7 (period propensity) cannot be computed from `TF_RAW`.** It stores no
   non-gold token probabilities.

## What is computed

`score_tf_exact.py` imports the registered scorer's functions unchanged: `score_arms.render`,
`score_tf.logodds_and_logprob`, same tokenization, same positions. Per row and arm it stores:

- **Unclipped** gold-token log-odds at every content and closing position.
- At the **final gold position**: log p of the merged-period token for that row, i.e. the id of
  `tok(final_piece + ".")` when it is a single token (`}.` = 7810 on all 128 merged rows), and log p
  of `.` (13).
- At the **EOS position**: log p of `<|im_end|>` (151645), of `.` (13), and of `<|endoftext|>`
  (151643).

## Canary (must pass before any amendment number is read)

Per arm, the re-computed **clipped** token log-odds must equal `TF_RAW.json`'s stored
`token_logodds`, max |Δ| ≤ 1e-5. The teacher-forced strict set must also equal `TF_RAW`'s. If the
canary fails, the amendment is not read.

## What is reported (post-hoc)

1. κ on content log-odds at clip ∈ {5, 8, 10, 12, 15, 20, none}, with the registered bootstrap
   (10,000 resamples, a fresh `default_rng(20260923)` per readout, ratio of means), and the outcome
   label the registered rule *would* give. At clip = 10 this must reproduce the registered primary
   exactly; that is a second canary.
2. Secondary (b) computed **per position**, as PREREG line 71 defines it: the positions where at
   least one of the five arms has p(gold) < 0.9. It is reported at clip 10 and unclipped.
3. Secondary 7, period propensity at the convention site, per arm. For merged rows this is log p of
   the merged token at the final gold position; for integer rows it is log p of `.` at EOS. Report
   the mean, and the count of rows where the period variant beats the gold continuation.
4. Per-family κ, unclipped.

No outcome label from this amendment replaces a registered one.
