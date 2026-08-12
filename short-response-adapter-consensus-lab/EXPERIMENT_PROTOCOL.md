# Experiment Protocol

## Scope and isolation

The official repository is a read-only source. Every new result, cache, manifest, plot, or report belongs to this sibling lab. Output guards reject the official tree and both excluded v22 mini-labs. Protected source files are hashed before and after execution.

## Exact primary decoder

At token step `t`, each independently activated adapter evaluates the complete identical accepted prefix without a KV cache. Each expert votes its stable `argmax` token. The selected token is the plurality winner. Ties are resolved by summed expert log-probability over tied tokens, then smallest token ID. EOS is an ordinary token in the primary condition.

The no-cache decoder is the scientific reference. A separate-cache implementation exists but failed a real-model decision-parity gate (6/8 exact output matches and 5/8 exact decision-trace matches in bf16), so it is prohibited in research configs. Per-example adapter batching and hotswapping also showed nonzero bf16 logit differences and remain disabled.

## Expert pool

The primary pool attaches seven official rank-16 LoRAs to one frozen 344,354,816-parameter Qwen3 backbone: broad Track A, anchored Track B, general, grounding, procedure, abstention, and social. Each adapter has 4,325,376 parameters and the same tokenizer, vocabulary, target modules, and declared base.

## Comparators and controls

- Backbone and individual K=1 experts
- K=2, K=3, K=5, and K=7 hard voting
- Validation-weighted hard vote
- Raw probability mean and raw logit mean
- DEV-temperature-calibrated logit mean
- Confidence-weighted probability and logit fusion
- Independent whole-sequence edit-distance medoid
- Duplicate-expert invariant
- Weak-expert injection, artificially sharpened expert, and fixed random-pool proxy
- Ordinary, strict-majority, confidence-threshold, and minimum-token EOS policies
- Separate disagreement-triggered K=3-to-K=7 consultation

## Data discipline

Official TRAIN data was not needed because existing compatible specialists were sufficient. DEV selected temperatures and finalists. Before any lab sealed access, `artifacts/frozen_configs/frozen_final_config.json` fixed the config, dataset hash, sample size, and post-sealed no-switching rule. A deterministic balanced 32-case sample of the official v2.2 TEST split was then evaluated once. It is a resource-aware sealed sample, not a claim about the full 4,050-row TEST split.

## Measurements

Each raw row records token IDs, per-step expert top tokens/probabilities/margins/entropy, vote counts/entropy/margin, ties, coalitions, dissent, pairwise JS divergence, EOS votes, output, exact correctness, abstention/repetition behavior, latency, TTFT, VRAM, RSS, adapter storage, active/resident parameters, and expert-token forward counts.

Summaries report capability by decision/category, output-length quantiles, EOS/completion/repetition rates, latency quantiles, tokens/sec, memory/storage, uncertainty discrimination/calibration, and quality–cost plots.

Paired binary comparisons use 20,000 bootstrap resamples, exact McNemar tests, exact or Monte Carlo sign-flip tests, and Holm correction over each declared comparison family.

## Interpretation rule

Pilot results guide freezing but are not final evidence. Sealed outcomes cannot trigger a different committee, threshold, or method. Claims distinguish numerical differences, confidence intervals, adjusted hypothesis tests, and unresolved conditions.
