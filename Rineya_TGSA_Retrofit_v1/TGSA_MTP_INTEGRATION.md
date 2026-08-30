# Pretrained Rineya TGSA + MTP Integration

This subtree is selectively carried from the pretrained-retrofit lineage because
it already solves the fragile parts of baking a new attention algorithm into a
Qwen3-family checkpoint: GQA geometry, model-supplied RoPE, Q/K normalization,
cache layout, deterministic tensor conversion, and conversion-damage audits.
The old branch's partial experimental outcome is not imported as Phase-2
evidence.

## Combined architecture

1. Convert only the selected attention layers with the existing
   `convert_qwen3_model` path.
2. Keep native Q/K/V/O and Q/K normalization tensors.
3. Start TGSA conditioning at zero.
4. Wrap the final causal hidden state with independent residual MTP heads for
   offsets 2..K.
5. Tie auxiliary vocabulary projection to the existing output embedding.
6. Never route auxiliary output back into the ordinary LM logits.
7. Use exact target-model verification before accepting proposed tokens.

## Warm-start stages

- `adapters`: TGSA trajectory modules and MTP heads only.
- `attention`: additionally release native attention tensors in converted layers.
- `converted_layers`: release all tensors in converted layers plus MTP.
- `all`: full fine-tuning.

Every transition records trainable and frozen parameter counts. Start at
`adapters`; immediate full fine-tuning is a comparison arm, not the default.

## Executables

- `scripts/run_tgsa_mtp_smoke.py`: real-checkpoint conversion drift and exact
  decoding, without training.
- `scripts/train_tgsa_mtp_adapter.py`: fixed-step prompt-masked training on Rineya
  JSONL and adapter-only checkpoint output.

See the main TGSA lab's `PHASE2_RUNBOOK.md` for PowerShell commands and evidence
interpretation.
