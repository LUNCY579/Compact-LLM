# Supa / Qwen3-4B single-layer residual graft

Prospective bounded experiment, 2026-09-05. The previous ARMR replication is
complete (A0/A1/A3: 12/16, 12/16, 13/16). This experiment asks whether ONE frozen
larger sibling layer can supply a mechanically tolerable residual correction.
It does not train a new Rineya model. Settings are in `protocol.json`; each run
records their hash and source hashes before any model forward.

## Frozen design

Use canonical Supa-A1.1, CPU FP32/eager, four threads, seed 17, greedy decoding,
full-prefix recomputation, and no cache. L0/L1/L2 run once. Insert after ordinary
Supa L12, before L13. Use Qwen3-4B L16 (zero-based), chosen prospectively as a
middle layer close to the target's relative depth, not by efficacy screening.
The official pinned donor has width 2560 and head_dim 128 (do not infer 80 from
width/head count). Verify the existing architecture manifest's shard hashes.

For Supa state h and frozen donor D:

    z = RMSNorm(h) @ W_up + b_up
    delta = (D(z) - z) @ W_down
    output = h + gate * delta

The donor owns its RoPE; it receives the same causal mask and token positions.
The gate is fixed, not learned. Exact gate zero bypasses the expert and returns
the original state object. The down-map intercept is used when scoring absolute
state reconstruction but cancels in a difference and is NEVER added to delta.

Bridge fitting uses only `data/evaluation/calibration.jsonl` prompts, from the
existing cross-scale lab. No scorer/answer enters the loss. Render each prompt
with Supa's chat template and verify the donor tokenizer produces identical IDs
for that exact rendered string. Truncate calibration only at 256 tokens, record
all lengths, and use all retained token positions. Seeded permutation of sorted
scenario IDs holds out 10 scenarios; all paraphrases of a scenario stay together.
Center inputs and outputs and fit ridge regression with lambda equal to 0.01
times the mean diagonal of X'X, in FP64, then cast maps to FP32. Up fits normalized
Supa post-L12 to native donor pre-L16. Down fits native donor pre-L16 to Supa
post-L12. This fits absolute-state alignment; transport of a donor delta remains
an unproven extrapolation. Report train and held-out errors/cosines and native
versus bridged donor transformation differences. No hyperparameter selection.

Calibration computes native states using the installed Hugging Face Qwen3
decoder, loading one layer at a time. Before real capture, a tiny Qwen3 fixture
must reproduce the monolithic model, including embedding and captured layers.
Only one donor layer is resident during the actual graft.

## Prespecified arms and gates

Eight existing mechanical prompts, original order/caps, 900 seconds per case:

1. A0: intact Supa.
2. Z: aligned expert with exact zero gate; logits and greedy IDs must be exact.
3. I: aligned bridges, identity donor, gate 0.1; delta and outputs must be exact.
4. N: aligned expert, sigmoid(-8) near-zero gate; descriptive plumbing check.
5. R: seeded random bridges at gate 0.1, calibration-train RMS-scaled outputs.
6. S: fitted bridges using seeded shuffled training-token pairs, gate 0.1.
7. G: aligned donor expert, gate 0.1, the primary graft arm.
8. A1: native Supa L12 repeated once, the measured architecture comparator.

R is a negative control, not evidence against aligned transfer. S fits exactly
the same tensors with correspondence destroyed. These controls do not establish
compute-matched efficacy. I exercises bridge plumbing but omits donor compute.
N passing alone is expected for a tiny perturbation and is not useful transfer.

For each arm retain output IDs/text, termination, Unicode/loop flags, answer
preservation, first-token logit cosine/KL/top-k, canonical-answer NLL, hidden RMS,
delta norms, donor and bridge parameter counts, and actual latency. Correct the
inherited evaluator's dense-matmul estimate for donor geometry and both bridges.
No gradient or backbone/checkpoint mutation is allowed during evaluation.
Exact identity failures, NaN/Inf, cache access, RSS >6 GiB, emergency RMS outside
[0.05,10], and unexpected exceptions stop the campaign and produce a failure
receipt. A loop/script/timeout ends its case; independent prespecified cases
continue. A 24-token cap is reported as truncation, not EOS completion.

G mechanically passes only with all eight cases, no safety stops, all observed
RMS ratios in [0.25,4], mean logit cosine >=0.90, mean baseline-to-arm KL <=0.5,
no invalid-rate increase, and no smoke score loss. Require baseline completion
for development eligibility; never silently change gates after outcomes.

Only if G passes, run the entire existing 16-case MERGEDEV-fast set for A0, G,
R, S and A1 with its existing per-case caps. Recompute paired wins/losses and
family preservation (no overall score loss, no family drop >0.10, no invalid
increase). Do not tune on those outcomes. One inspected development gain is
exploratory, not general capability or independent efficacy. If G fails, stop
at the mechanical result. No router, repeat graft, MLA, MTP, distillation,
sealed evaluation, checkpoint export or publication is included.

Each run gets a new evidence directory and ignored local tensor directory.
Never overwrite historical evidence or canonical model files. Record source,
protocol, datasets, checkpoint files, runtime implementation and loaded state
hashes, and before/after restoration. Large activation/bridge tensors stay local.

## Reproduce

From the repository root, use
`C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`:

```text
python -m pytest -q experiments/cross_scale_layer_graft_v1/tests
python experiments/cross_scale_layer_graft_v1/run.py
```

Sources: [Qwen3-4B pinned configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json).
MLA/MTP composition in [DeepSeek-V3](https://arxiv.org/abs/2412.19437) motivates
later separate experiments; it does not validate this graft.
