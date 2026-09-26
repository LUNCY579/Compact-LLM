# Answer-only LM-head computation: proposed mechanical memory fix

Status: tested on a tiny random CPU model; **not applied to live training source**. `answer_only_logits_proposed.patch` is reviewable proposed content. `git apply --ignore-space-change --check` passes against the current source; ignoring whitespace accommodates its Windows line endings. The isolated proposed source and tests in `proposed_answer_only/` passed **15/15 CPU safety tests** in 9.21 seconds, including all previous failure-path tests plus suffix-shape validation. Neither check writes the live source.

Run the evidence check from repository root:

```powershell
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/verify_answer_only_logits.py
```

```powershell
../Chat-Bubble/.venv/Scripts/python.exe -m pytest experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/proposed_answer_only/tests/test_training_safety.py -q
```

Evidence: `answer_only_logits_equivalence.json`; required interpreter; both canonical checkpoint file sets hash identically before/after. One tiny random CPU Qwen model, no canonical weights loaded, no GPU use. The tiny model uses the canonical wrapper's actual class and a real PEFT LoRA only on layer 3, with layers 0–2 frozen. Nonreentrant gradient checkpointing is active.

## Algebra and implementation

Let prompt length be P and answer-plus-EOS length be N, total T=P+N. Current labels are `[-100]*P + answer_ids`, including EOS. Transformers `ForCausalLMLoss` shifts these left one position and appends an ignored final target. Consequently only hidden-state positions P−1 through T−2 contribute loss. All earlier logits and the final position have ignored targets.

Pass full `input_ids`, `logits_to_keep=N+1`, and `labels=full_labels[-(N+1):]`. Qwen's local forward computes the decoder over all T tokens, then slices its final hidden states immediately before `lm_head`; the loss receives aligned cropped labels. The shifted suffix targets are exactly the original supervised targets followed by the same final ignored target. EOS remains supervised. The extra `+1` is necessary: keeping N positions loses the first answer token. No context, attention activation, layer, RoPE position, adapter target, training example, causal-loss denominator, or seed changes.

The proposed `_batch` validates equal full-input/label lengths, positive answer count, fully masked prompt, and a contiguous answer suffix equal to the input's answer tokens. For synthetic no-prompt tests it caps K at T, preserving their original objective. Real curriculum examples have masked chat prompts. The same helper is used for validation loss, whose gate remains on under no-grad; generation evaluation is unchanged.

Also delete `out` and `loss` immediately after backward and detached logging, and `out` after each validation-loss accumulation. Python otherwise retains the preceding output while evaluating the next forward's right-hand side. This can overlap large logit tensors even after the previous autograd graph has been freed.

Local authoritative sources: `../Chat-Bubble/.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py:504` slices hidden states before projection; `.../loss/loss_utils.py:49` implements float32 logits and causal label shift. Canonical `modeling_supa_a11.py::forward` forwards arbitrary kwargs to that parent and applies the abstention bias only after the loss is computed. Source hashes are in the receipt.

## CPU checks and observed tolerance

Fifteen cases passed: sequence lengths 2, 3, 81, **3698**, **3840**, each under float32, CPU autocast float16, and CPU autocast bfloat16. Cases include an EOS-only target, minimal prompt, and mult-token answers. Every decoder input length remained full; only head input length shrank. All LoRA A/B gradient tensors were **exactly identical** across the full and reduced computation in all cases (maximum absolute gradient difference 0). Loss differences were at most 9.5367431640625e-7, consistent with float32 reduction rounding; the mathematical objective is identical, while runtime bit identity is not promised. Gate-on no-grad validation also passed. An explicit wrong-K negative control detects the missing first target.

These are local CPU equivalence checks, not a measured GPU-memory guarantee. With canonical vocabulary size 151936, at T=3698 and illustrative K=9 the raw fp16 logits require 1,123,718,656 bytes versus 2,734,848 bytes; the loss's fp32 copy requires 2,247,437,312 versus 5,469,696 bytes. These are individual tensor-size calculations, not additive measured peak savings. Decoder attention activations remain full-length and may still constrain memory.

Apply only after the active smoke has ended or been cleanly recorded/stopped. Document the implementation change, run the focused safety/equivalence checks, and measure the next smoke's actual memory before attempting bounded training. Keep the same complete curriculum, optimizer-attempt budget, LoRA hyperparameters, and decoding protocol.
