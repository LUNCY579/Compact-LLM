# Opallo publication pre-registration

Written **before** any materialized-artifact score was observed. Recorded so the bf16
publication decision cannot be made after seeing whether bf16 happened to look good.

## What is being published

`B28last20 + wsd:0.2:0.2 + beta2 0.95`, lr 1e-5, 5,233 supervised tokens, seed 17 --
checkpoint tag `B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95`, 220 trained tensors,
314,618,880 parameters, measured **74/150** on the frozen panel with 41 of 47 reference
answers kept.

Publishing means materializing a **dense Hugging Face `model.safetensors`**: base Supa
1.7.0-devfreeze with those 220 tensors substituted, cast to BF16 to match the base dtype
and the rest of the family vault.

## The thing that could go wrong

The checkpoint is **float32**; Supa and every other published core are **BF16**. Casting
314M trained parameters from fp32 to bf16 discards 16 mantissa bits per element. Every
panel score this programme has quoted was measured in fp32. The published artifact is
therefore *not* the measured artifact until its own score is measured.

## Pre-registered acceptance thresholds

Measured on the 150-row `monitor_probe.jsonl` panel under the frozen protocol
(`retention-a329eb7b/B28/retention.json` reference, greedy, CPU, the same path that
produced 74/150):

| condition | threshold |
|---|---|
| panel `correct_total` | **>= 70** |
| reference answers retained | **>= 38** of 47 |
| dense reload | zero missing / unexpected / mismatched keys, all tensors finite |
| frozen tensors | byte-identical to Supa for all tensors outside the trained set |

70 is the four-seed mean (69.25) rounded up: any value at or above it stays inside the
measured seed band for this recipe and far above the 47-row source. 38 allows the cast to
cost three reference answers out of the 41 the fp32 checkpoint kept.

## What happens on failure

If `correct_total < 70` or `retained < 38`, **BF16 is not published.** The fallback is to
publish the float32 artifact instead (2.4 GB rather than 1.2 GB, off the family dtype
convention) and to record the measured bf16 loss as the reason. Either way the measured
number goes in `CORE.json` -- there is no path here that publishes an unmeasured score.

## What is NOT claimed

* No gate is opened. This is an exploratory research artifact promoted to a named core,
  not a G0/G1 receipt, and `licenses_efficacy_claim` stays false.
* The panel is a 150-row arithmetic-word-problem instrument drawn from three classes
  (`percentage_reasoning`, `ratio_proportion`, `time_duration`). "74/150" is a claim about
  that instrument and nothing else.
* General reasoning is **unchanged**: forge test_iid 30/200 against the source's 32
  (p=0.804). Opallo is an arithmetic-panel improvement that did not cost measurable
  in-distribution reasoning, and it is published as exactly that.
* Out-of-distribution output discipline is **slightly worse** (forge test_transfer, pooled
  p=0.020 across four seeds). This is disclosed in the core README, not buried.
