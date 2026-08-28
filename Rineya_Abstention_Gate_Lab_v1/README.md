# Rineya Abstention Gate Lab v1

Can a decode-time abstention threshold on **one** model recover the routed
cascade's gain on the Rineya long-context suite, removing the multi-pass
requirement?

**Yes on accuracy, no on significance.** A threshold on the UNKNOWN token's
probability mass at the first generated token takes `nimbus_low` from 121/192 to
**128-130/192** on the held-out stability split — level with or above the
nine-arm System48 (128) — from one pass, one base core, no extra weights.

The 130 and its p = 0.0490 hold only on the BF16 base+adapter arithmetic path.
FP32 and both dense merges give 128 at p = 0.12-0.14, while the ungated model is
121 in every realisation. **The durable claim is a stable +7; the significance
is not robust.** Audit: `artifacts/rineya_abstention_gate_v1/NUMERICAL_ROBUSTNESS.md`.

- Findings, caveats and file map:
  `artifacts/rineya_abstention_gate_v1/RESULTS.md`
- Pre-registration written before any threshold was inspected:
  [CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md)

## What changed outside this lab

Two **opt-in** flags on the frozen runner
`Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py`. Neither alters the
default path: with both omitted the script emits the same fields, in the same
order, from the same `generate` call.

| flag | effect |
| --- | --- |
| `--emit-logprobs` | adds a `confidence` block per row: per-step logprob, top1−top2 margin, full-vocabulary entropy, first-step top-k, and the first-step UNKNOWN mass |
| `--adapter-scale` | multiplies every active LoRA scaling (PEFT `set_scale` semantics), so the frozen `checkpoint-200 @ 0.8125` operating point is reachable from the canonical runner |

`--emit-logprobs` only reads logits `generate` already produced. **Verified:** all
256 predictions, input-token counts and generated-token counts from the capture
are identical to the frozen baseline artifacts.

## Scripts

| script | role |
| --- | --- |
| `scripts/gate_core.py` | signals, gate semantics, scoring, exact McNemar. Imports the published `evaluate_predictions.normalize`, so a gate can never be scored by a private rule |
| `scripts/sweep_thresholds.py` | tuning sweep; **refuses to run on a split other than the named tuning split** |
| `scripts/apply_gate.py` | applies a frozen spec; fails loudly if lab bookkeeping disagrees with the published scorer |
| `scripts/compare_systems.py` | head-to-head exact McNemar matrix over prediction files |
| `scripts/signal_diagnostics.py` | threshold-free ROC AUC per signal |
| `scripts/plot_tradeoff.py` | precision/recall frontier and accuracy-vs-threshold figure |
| `scripts/abstention_processor.py` | the two gates as `LogitsProcessor`s, for use with `model.generate` |
| `scripts/build_abstention_checkpoint.py` | merges the LoRA and bakes the abstention offset into one checkpoint directory |

## Reproduce

Paths below are relative to the repository root; artifacts land in the untracked
`artifacts/` tree. Set `PYTHONIOENCODING=utf-8` — the published scorer crashes on
a cp1252 console.

1. Capture confidence for both splits (about 2.5 minutes on an RTX 4070 Laptop):

```bash
python Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py --model Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze --adapter artifacts/halo_long_context_sft_v1/stage_a/checkpoint-200 --adapter-scale 0.8125 --input Rineya_Long_Context_Demo_v1/data/showcase_prompts.jsonl --output artifacts/rineya_abstention_gate_v1/capture/nimbus_low_showcase_confidence.jsonl --max-new-tokens 24 --max-context 4096 --seed 17 --disable-thinking --torch-dtype bfloat16 --emit-logprobs --logprob-top-k 20
```

2. Sweep thresholds on the tuning split only:

```bash
python Rineya_Abstention_Gate_Lab_v1/scripts/sweep_thresholds.py --confidence artifacts/rineya_abstention_gate_v1/capture/nimbus_low_showcase_confidence.jsonl --key Rineya_Long_Context_Demo_v1/data/showcase_answer_key.jsonl --output-dir artifacts/rineya_abstention_gate_v1/tuning_showcase64 --tag showcase64 --tuning-split-name showcase
```

3. Apply the frozen gate to the held-out split:

```bash
python Rineya_Abstention_Gate_Lab_v1/scripts/apply_gate.py --gate-spec artifacts/rineya_abstention_gate_v1/tuning_showcase64/showcase64_gate_S1_primary.json --confidence artifacts/rineya_abstention_gate_v1/capture/nimbus_low_stability_confidence.jsonl --key Rineya_Long_Context_Demo_v1/data/stability_answer_key.jsonl --output-dir artifacts/rineya_abstention_gate_v1/results --tag S1_primary_stability
```

## Baking it into a checkpoint

`build_abstention_checkpoint.py` merges the adapter and folds the abstention
decision into an `auto_map` model class, producing a directory that loads with stock
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and abstains
with no extra arguments. Only the **logit-offset** gate family can be baked:
`logP(UNKNOWN) - logP(top1) = z_UNK - z_top1`, so the softmax normaliser cancels
and the rule is literally a constant added to one logit. Gates that threshold a
*normalised* probability (the confirmatory primary, S1) need
`NormalisedAbstentionGate` from `abstention_processor.py` instead — still one
forward pass, but six lines at the call site.

Two things that do not survive baking, both measured rather than assumed, are
recorded in `artifacts/rineya_abstention_gate_v1/CHECKPOINT_RESULTS.md`.

The builder originally attempted to store the offset as
`GenerationConfig.sequence_bias`. Transformers 5.14 serialises its tuple keys as
strings and then rejects them while loading, so that route cannot round-trip.
The current builder emits the same custom-model mechanism as the evaluated
checkpoint. The original builder hash remains recorded in the checkpoint as
historical provenance.

## Tests

```bash
python -m pytest Rineya_Abstention_Gate_Lab_v1/tests -q
```

50 tests. They pin the McNemar implementation to the audit's published values
(13 vs 6 -> p = 0.167; 9 vs 5 -> p = 0.424), check that the threshold enumeration
realises every achievable partition without ever sitting on an observed value,
verify the vectorised sweep against the reference `evaluate_gate` on every
threshold, and check the confidence arithmetic against hand-computed softmax
values including the EOS and empty-generation edge cases.
