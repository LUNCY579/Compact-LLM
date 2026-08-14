# Central claim

## One-sentence version

Hard token-level plurality can compose independently specialized LoRA adapters
without weight merging or a learned router, but its observed advantage is a
benchmark-specific consequence of calibration-resistant directional consensus
and answer/abstention policy composition—not universal local superiority,
rank-only information, or autoregressive trajectory correction.

## Paper-ready version

Across seven compatible rank-16 LoRA specialists sharing a 344M-parameter
backbone and accepted autoregressive prefix, equal top-1 token voting reaches
17.259% exact accuracy on the complete 4,050-example v2.2 TEST split. This is
2.667 percentage points above the strongest predeclared individual adapter and
3.160 points above raw probability averaging under paired, Holm-adjusted tests.
The gain is not explained by a generally better next-token classifier:
calibrated and raw probability pooling both beat hard voting on same-split DEV,
and the teacher-forced ordering is almost unchanged under free-running
generation. Instead, hard voting limits domination by incomparable confidence
scales while composing a distinct answer/abstention policy. The effect is
composition- and benchmark-dependent: K=4 is the best selected free-running DEV
committee, and hard K=7 fails the fixed-method v2.1.2 transfer test.

## What is supported

- Equal top-1 decisions form a useful composition interface for compatible
  parameter-efficient specialists.
- Unique plurality decisions are invariant to positive confidence rescaling;
  unequal scaling can still affect the declared confidence tie-break.
- Complementary expert decisions and abstention behavior materially change the
  aggregate policy.
- Committee identity matters more than committee size alone.
- Adaptive consultation can retain similar overall DEV accuracy with much less
  measured inference work.

## What is not claimed

- Hard voting is not universally superior to probability fusion.
- The v2.2 result does not generalize to the tested v2.1.2 corpus.
- Early sequence divergence does not establish trajectory correction.
- Specialist semantics are not isolated causally from seed, competence, or
  training-mixture differences.
- Full K=7 is not a deployment-efficiency improvement.
