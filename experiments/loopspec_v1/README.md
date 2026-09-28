# LoopSpec qualification for Rineya

This branch tests the mechanism from **LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers** (arXiv:2609.17184) without changing any existing Rineya branch.

## Research conclusion

LoopSpec is highly relevant **only to a genuinely recurrent/looped Rineya host**. It is an inference scheduler, not a way to make an ordinary 28-layer Qwen3 checkpoint recurrent. The current published Supa 1.7.0 config is a standard 28-layer `Qwen3ForCausalLM`; therefore applying LoopSpec directly to Supa would be scientifically invalid. The correct target is a future or existing Rineya recurrence host in which the *same block/stack* is reapplied at recurrent depths and an exact final-depth readout defines the target distribution.

The paper's key opportunity for Rineya is unusually strong: recurrence normally trades parameter efficiency for extra serial compute. LoopSpec tries to reclaim that latency by using an early recurrent state as an in-model drafter. The parent token continues toward full-depth verification while a child token begins recurrent work. Because branches execute the same shared weights, mixed-depth states can be batched through one block call. If the draft is accepted, its child keeps already-computed recurrent work. A selective deeper proposal can recover some first-proposal failures.

This is complementary to Rineya's recurrence research: **architecture decides whether repeated computation helps quality; LoopSpec decides whether that repeated computation can be scheduled efficiently at decode time.** It should therefore remain a separate systems variable from recurrence training, gating, carry state, TGSA/MTP, or retention interventions.

## What this implementation contains

`core.py` is a framework-independent scheduler with:
- one early proposal and optional selective deeper residual proposal;
- exact residual verification for greedy and sampling;
- explicit branch pruning/restart semantics;
- branch and context caps;
- separate draft and verification RNG streams;
- EOS only after authoritative commitment;
- work, branch and logical-KV accounting.

`tiny.py` is a frozen randomly initialized CPU looped Transformer used only as mechanism apparatus. It has a unique prelude, shared recurrent GQA/RoPE/SwiGLU block, unique coda, per-depth loop/coda caches, and an independent full-prefix oracle.

The tests exercise probability-law preservation, residual cascades, mixed-depth batching, cache-depth isolation, independent cached-vs-full-prefix parity, greedy equality, sampling replay, branch cleanup, EOS/length behavior, and fail-closed validation.

## Native Rineya boundary

The command `native-preflight` is deliberately **BLOCKED**. It reads config metadata only and never loads weights. A standard Qwen3 config cannot self-certify as a looped backend.

Before a real Rineya run:
1. bind an exact recurrent checkpoint, tokenizer, runtime, dtype, processors and fixed recurrent depth R;
2. implement `prefill/start/advance/readout/cache_bytes` against that native host;
3. independently prove full-prefix vs incremental parity at every proposal/final depth, including RoPE positions, GQA KV layout, coda/final norm/logit processing and mixed-depth batching;
4. freeze proposal depths and branch cap using calibration prompts only;
5. run untouched reasoning/coding/general-language evaluation with baseline AR, one-proposal, two-proposal residual, and gated two-proposal arms;
6. measure output equality/distribution, acceptance, tokens per recurrent execution, recurrent batches, wall time, TTFT, inter-token latency, peak VRAM/RAM, KV bytes, and failure/restart rates.

Do not claim speedup from CPU apparatus timings. The paper's reported maximum is hardware/model/task-specific; Rineya needs its own exact-host measurement.

## Suggested Rineya experiment

Use a fixed recurrent Rineya checkpoint and no training.

- A0: ordinary full-depth autoregressive decoding.
- A1: LoopSpec first proposal only.
- A2: first + residual second proposal, ungated.
- A3: first + residual second proposal, gated.
- Optional A4: the paper's analytically predicted depths versus a small calibration-only depth grid.

Freeze decoding settings separately for greedy and sampling. For sampling, test distributional correctness statistically rather than expecting identical token sequences from equal seeds.

Primary systems endpoint: wall-clock decode speedup at batch 1 on the same GPU/runtime. Secondary endpoints: acceptance rate, equivalent recurrent work, peak memory, and latency distribution. Quality must be equal under greedy exact decoding and statistically indistinguishable under the same processed sampling distribution.

## Local commands

```bash
pytest -q experiments/loopspec_v1/tests
python -m experiments.loopspec_v1 smoke --out /tmp/loopspec-smoke.json
python -m experiments.loopspec_v1 native-preflight \
  --config Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/config.json \
  --out /tmp/loopspec-native-preflight.json
```

The preflight command is expected to exit nonzero with a BLOCKED receipt for Supa 1.7.0.

## Claim ceiling

Implementation/mechanism evidence only. No Rineya checkpoint was executed, no Rineya quality result exists, and no Rineya speedup has been measured.
