# Release verification receipt

**Rineya Reasoning Forge RL v2.0.0 — machine-validated candidate**

| Executed check | Result |
|---|---:|
| Unit and integration tests | 222 passed; zero failed, errored, or skipped |
| Canonical worlds checked by separate audit solvers | 20,000 |
| Complete structured reference traces checked | 20,000 |
| Stateful/action transitions independently replayed | 198,770 |
| Public RL/SFT view rows checked with JSON Schema | 75,200 |
| Reasoning-mode gold reference responses scored | 20,000 |
| Process labels checked | 36,752; 18,376 valid and 18,376 invalid |
| Synthetic preference directions checked | 32,000 |
| Causal teacher-forced decision views checked | 10,000 |
| Matched held-out diagnostic views checked | 384 from 64 existing parents |
| Exact normalized prompt overlaps against the supplied v1 | 0 of 20,000 new prompts compared with 10,000 original rows |

See the accompanying JSON reports and test log for scope and details. Exact prompt normalization uses Unicode NFKC and whitespace collapse; it is not a semantic benchmark-contamination audit. Gold self-scoring is a software test, not a trained-model result. The separate solvers share a parser and authorship.

TRAIN has 16,000 worlds, validation 1,600, IID test 1,600, transfer test 640, and length stress 160. SFT modes, RL modes, transitions, preferences, episodes, and probes are related views. Their counts must not be added to claim more independent examples.

Same-seed byte reproducibility was executed for two complete 64-world smoke builds. A second full 20,000-world release generation was not executed. The original uploaded archive was inspected but not changed; its rows were not copied.

No native tokenizer, pretrained model, optimizer, or GPU training was run. Human review, native template/loss-mask/terminator qualification, actual resource fit, and measured model benefit remain separate gates. The response parser accepts all released reference reasoning responses, but that is not a native token-window guarantee.

**Evidence status:** suitable for corpus review and separately qualified post-training experiments; not established as a broad-language recovery corpus, model improvement, architecture win, or shipping release.
