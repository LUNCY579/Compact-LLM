# Response-Control Contract

The trained model should learn to emit EOS early. The deployed strict short mode adds a deterministic runtime guarantee so a second sentence is not returned even when learned behavior fails.

## Stop order

1. Model-native EOS.
2. First validated sentence boundary in `one_sentence` mode.
3. `max_new_tokens` as an emergency cap.

The boundary detector protects decimal/version dots, common abbreviations and initialisms, URLs/domains, Markdown code, ellipses used internally, and unmatched quotes or brackets. The visible output is finalized again after generation because one tokenizer token can contain punctuation plus following text.

## Evidence boundary

This is output control, not structural pruning. It can avoid later decode passes and output-side KV growth. It does not reduce weight memory, and it does not make selective-layer training equivalent to inference-time layer skipping.

Phase 1 behavior-only evaluations use `eos_only` so the training effect remains measurable. Phases 5–8 default to `one_sentence` because they evaluate the deployable system. Phase 8 blocks packaging unless sealed evaluation used this mode with a zero one-sentence violation rate.
