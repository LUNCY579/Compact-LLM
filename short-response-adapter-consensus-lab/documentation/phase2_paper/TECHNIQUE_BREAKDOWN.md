# Technique breakdown

## System

- One frozen 344M-parameter short-response backbone.
- Seven compatible rank-16 LoRA adapters: Track A, Track B, General, Grounding,
  Procedure, Abstention, and Social.
- At every decoding step, each active expert evaluates the same accepted prefix.
- The selected token is appended once and becomes the shared next prefix.

## Hard consensus rule

Each expert contributes its top-1 token identity. The decoder selects the token
with the largest vote count. When vote leaders tie, the implementation uses the
sum of supporting experts' confidence; the remaining deterministic tie order is
documented and tested. Positive temperature scaling cannot change expert top-1
proposals, vote counts, or a unique plurality, but unequal temperatures can
change the confidence-based tie-break.

## Comparators

- standalone backbone and individual adapters;
- raw and per-expert calibrated probability means;
- logit and normalized-logit pooling;
- top-2 approval, Borda top-5, and reciprocal-rank top-5;
- entropy weighting, trimming, median pooling, confidence clipping, and
  validation-weighted hard voting;
- every nonempty expert subset under cached gold-prefix evaluation;
- selected free-running committees and seven leave-one-out removals;
- adaptive K3-to-K7 consultation.

## Evidence stages

1. Independently verify the sealed 4,050-example v2.2 TEST cornerstone.
2. Preserve the n=32 pilot as feasibility evidence only.
3. Build a 4,050-example, 18,760-decision teacher-forced DEV cache.
4. Evaluate calibration, rank rules, all 127 committees, Shapley values,
   diversity, ties, EOS policies, and temperature interventions on DEV.
5. Freeze selected methods before same-split free-running DEV inference.
6. Run the selected committee, leave-one-out, and adaptive systems panels.
7. Freeze and run one calibrated v2.2 TEST confirmation.
8. Gate and run fixed-method v2.1.2 cross-corpus transfer.
9. Regenerate paper artifacts and pass the synthesis and completion audits.

## Statistical design

Exact-match accuracy is primary. Comparisons use paired rescue/harm counts,
exact McNemar tests, paired bootstrap confidence intervals, and Holm correction
within declared families. Answerable and abstention strata, dependence
sensitivity, latency, expert forwards, VRAM, and RSS are reported separately.
DEV-selected results remain model-selection evidence; v2.2 TEST is the
cornerstone; v2.1.2 is a caveated transfer test and is never pooled with v2.2.
