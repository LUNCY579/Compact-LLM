# Rineya Positional-Significance / Sequential-Attention Lab

A mechanistic-interpretability lab for one question, asked four ways:

> **Where in a long context does a Qwen3-0.6B attention head actually look — and can
> that be measured well enough to steer it at inference time?**

Concretely:

1. **Measure.** Build a controlled long-context benchmark in which evidence position is
   the only thing that moves, and measure the "lost in the middle" curve for four
   checkpoints that share one architecture and one tokenizer.
2. **Localise.** Estimate a *positional significance matrix* `S[layer, kv_head, bin]` —
   how much the model's answer depends on each KV head attending to each region of the
   context — by attribution (cheap, approximate) and by real ablation (expensive, exact),
   and report how well the two agree.
3. **Intervene.** Feed that matrix back into attention through three reversible hooks
   (logit bias, value scale, head-output scale) and ask whether the middle can be
   recovered without retraining.
4. **Sequence.** Ask whether running attention heads in *stages* — each stage seeing the
   previous stage's output — beats running them in parallel, and whether significance and
   sequentiality interact.

Every claim is paired on `scenario_id`, bootstrapped, and shipped with a provenance
sidecar. Every intervention has a negative control. `SPEC.md` is the binding interface
contract; this README is the map.

---

## The model panel

Four checkpoints, all Qwen3-0.6B shaped (28 layers, 16 query heads, 8 KV heads,
head_dim 128), all sharing one tokenizer, so cross-model comparison runs on
byte-identical token sequences.

| key | checkpoint | store dtype | note |
| --- | --- | --- | --- |
| `qwen3` | `Qwen3-0.6B` | bfloat16 | the base model, and the reference for every delta |
| `halo` | `Rineya-Halo-1.2.0-devfreeze` | float32 | run in bfloat16; the downcast is recorded, not hidden |
| `nimbus` | `Rineya-Nimbus-1.6.0-devfreeze` | bfloat16 | |
| `supa_a11` | `Rineya-Supa-A1.1-1.1.0-devfreeze` | float16 | `trust_remote_code=True`; adds a constant `+5.15625` logit offset on token `51378` |

Supa-A1.1 is reported in **two modes, both recorded**: `gated` (as published, used for
accuracy) and `ungated` (`load_model(..., ungate=True)`, used for teacher-forced NLL and
all causal significance, so `Delta` reflects attention rather than the abstention gate).
Ungating changes the loaded instance only; the checkpoint on disk is never touched.

---

## Two rules that are enforced by tests, not by convention

### 1. The source checkpoints are immutable

Nothing in this lab opens a file under `Qwen3-0.6B/` or `Rineya_Published_Cores/` for
writing. Every writer calls `paths.assert_readonly_target(path)` first, which raises
`PermissionError` if the target resolves inside a checkpoint tree.

`tests/test_immutability.py` enforces this the hard way: SHA-256 of every non-weight file
in all four trees (size + mtime for the multi-GB safetensors), then a representative slice
of the lab — the audit's read paths, a `psa_session` forward, a real two-checkpoint merge
— then the same digest again, asserted equal. It also asserts `assert_readonly_target`
refuses a path inside each checkpoint, and greps the source for a checkpoint path handed
to a writing `open()`.

Merged and modified weights go to `experiments/merges/<merge_id>/`, never next to a source.

### 2. The sealed split is opened exactly once, at Phase 14

Splits are deterministic in `sha256(scenario_id) % 100`: `< 40` calibration, `< 70`
validation, else sealed. The key is the **scenario**, not the item, so every positional
variant of one source example lands in the same split — which is what makes the paired
statistics legitimate.

`tests/test_split_discipline.py` enforces two things. First, that no module or script
outside `scripts/phase14_*.py` names the sealed split *in code* — the scan is over the
AST, so a docstring or a comment is not a violation but a string literal or an identifier
is; four files are allowlisted with a stated reason, and a stale allowlist entry is itself
a failure. Second, that `split_for_scenario` returns identical assignments in a subprocess
run under a different `PYTHONHASHSEED`. If it ever did not, every "held out" claim in the
lab would be false.

`data/sealed/position_examples.jsonl` exists on disk. It has not been read.

---

## Module map

```
src/rineya_psa/
  paths.py                 repo/lab roots, the MODELS registry, assert_readonly_target
  types.py                 frozen shared types: ModelSpec, Arch, BenchmarkItem,
                           TokenizedItem, PositionGrid, split_for_scenario
  loading.py               the only place that calls from_pretrained; dtype + gating policy
  backend.py               the SDPA backend pin (see "Why the backend is forced" below)

  model_audit/     Phase 1   structural audit: hashing, config normalisation, per-tensor
                             and per-head weight deltas. CPU + memmap only; never loads a
                             model. CLI: python -m rineya_psa.model_audit.run_audit
  benchmark/       Phase 2   filler prose -> question families -> evidence placement and
                             four surface formats -> token-space measurement of where the
                             evidence actually landed -> the bank plus its split manifest
  hooks/           SPEC 5    the attention interface. Registers a "psa" kernel in
                             ALL_ATTENTION_FUNCTIONS; psa_session() swaps it in and
                             restores sdpa on exit. Binning, capture, the three modes.
  evaluation/      Phase 3+  metrics (three accuracy columns, gold NLL, answer margin,
                             U-shape fit), the batched runner every phase goes through,
                             and the capability-preservation suites
  significance/    Phase 4-5,7  the I / B / D matrices. Tier 1 attribution and Tier 2
                             ablation, their agreement test, positional bias with no task
                             at all, absolute-vs-distance decomposition, low-rank SVD
  activation_analysis/ Phase 6  what was computed (collect), what is decodable (probes),
                             what is used (patching), and per-head statistics
  merging/         Phase 8   blockwise checkpoint arithmetic and position-aware soups
  interventions/   Phase 8,11  PSA (matrix into parallel attention), SGSA (matrix into
                             staged attention), four cheap corrections that need no
                             matrix, the mandatory negative controls, and FLOPs/VRAM
                             accounting
  sequential_attention/ Phase 10  Arms A-E: the staged-attention wrapper, head orderings,
                             and the activation-stability guard that aborts a diverging arm
  statistics/      SPEC 7,9  paired bootstrap clustered on scenario_id, BCa CIs, Cohen's
                             dz, exact McNemar, the three-way interaction contrast, and
                             stamp()/write_result() for provenance sidecars
  figures/         SPEC 47   one function per figure, a shared style, and a guard that
                             refuses to save a figure with a truncated value axis
```

Import style is absolute (`from rineya_psa.hooks.registry import ...`). Only `hooks/`,
`loading.py`, `backend.py`, `sequential_attention/` and `activation_analysis/` import
torch at module scope; everything else imports it inside functions so CPU analysis and
most of the test suite start instantly.

### Why the backend is forced

Left to itself PyTorch's SDPA dispatcher picks the MATH backend on this machine (no
FlashAttention kernel in the Windows wheel), which materialises the full `[B, H, T, T]`
attention matrix. Measured, Qwen3-0.6B bf16, batch 1: at T=8192 that is 54 s and 11.6 GB
versus 0.83 s and 1.4 GB with CUDNN/mem-efficient pinned. Every forward therefore runs
inside `backend.efficient_attention()`. Batch size is 1 by design: batching is a net loss
on this card and drags in left-padding, which is a positional confound in a study about
position.

---

## Running it

Everything runs under the repo venv with `src` on the path:

```bash
cd rineya_positional_sequential_attention_lab
export PYTHONPATH=src
PY=../.venv/Scripts/python.exe
```

| phase | command | cost |
| --- | --- | --- |
| 1 — checkpoint audit | `$PY -m rineya_psa.model_audit.run_audit` | ~10 min, CPU only, reads 6.5 GB |
| 2 — build the benchmark | `$PY -c "from rineya_psa.benchmark import build; build()"` | ~5 min, CPU only, writes ~256 MB |
| 3 — baseline fingerprint | `$PY scripts/phase3_baseline_fingerprint.py --split calibration --dry-run` | print the estimate first |
| | `$PY scripts/phase3_baseline_fingerprint.py --split calibration` | ~40 min on the GPU |
| tests | `$PY -m pytest -q` | ~60 s, CPU only |
| lint | `$PY -m ruff check src tests` | seconds |

Any script that would exceed ~2 GPU-hours prints its `cost_estimate()` and refuses to run
without `--yes-i-costed-this`.

Phases 4 and later have library code and unit tests but no driver scripts yet; see STATUS.

### Conventions that will bite you if you skip them

* **`enable_thinking=False`.** Qwen3 is a hybrid reasoning model. Rendering the chat
  template without it makes the model emit `<think>` and burn the whole generation budget:
  a pilot scored 0/27 where the correct protocol scores non-zero. Go through
  `evaluation.runner`, which sets it and strips a leaked `</think>` defensively.
* **Gold NLL is the primary metric, not accuracy.** Strict exact match sits near the floor
  on this benchmark; the NLL curve carries the same signal at far higher resolution.
  Accuracy is reported in three columns (`strict`, `contains`, `first_line`) and never
  collapsed into one headline — `contains` is a ceiling, inflatable by a verbose model.
* **`logits_to_keep`.** The memory wall on an 8.6 GB card is the logit tensor, not
  attention: `logits[1, 8192, 151936]` is 2.5 GB before the float32 upcast. Every
  teacher-forced call passes `logits_to_keep`; the runner does this for you.
* **Pair on `scenario_id`.** Not on `item_id`. The bootstrap resamples scenarios.

---

## Tests

```
tests/conftest.py                shared fixtures: the canonical toy Qwen3 (2 layers,
                                 hidden 64, 4 q heads / 2 kv heads, head_dim 16,
                                 vocab 512, float32, CPU, seeded), its Arch, a
                                 tokenizer stub with real offset mappings, temp lab dirs
tests/test_immutability.py       the checkpoints are read-only (see above)
tests/test_split_discipline.py   the sealed split stays sealed (see above)
tests/test_integration_smoke.py  end to end on the toy model: build 12 items, tokenize,
                                 score, reduce to positional metrics, estimate a 2x2x4
                                 significance matrix, build a PSA context from it, run
                                 again inside psa_session, run a two-stage sequential
                                 arm, bootstrap the paired difference, render a figure
tests/test_baseline_equivalence.py  SPEC 5.2: a disabled or zero-coefficient PSA context
                                 must match plain sdpa to atol 1e-5 on 512 tokens
tests/test_gqa_mapping.py        q_heads_for_kv(3) == (6, 7), pinned against the kernel
tests/test_*.py                  one file per package
```

No test loads a real checkpoint onto the GPU. The model-facing tests build a randomly
initialised 2-layer Qwen3 on the CPU; the checkpoint-facing tests read headers and
memmaps and skip cleanly when the checkpoints are absent. One heavy test
(`test_model_audit.py`, reads 6.5 GB) is opt-in behind `PSA_MODEL_TESTS=1`.

---

## STATUS

Last verified: 2026-08-29. `600 passed, 1 skipped`; `ruff check src tests scripts` clean.
Every claim in this section was checked against the files on disk, not from memory.

### Run on real checkpoints, with artifacts

| phase | artifacts | report |
| --- | --- | --- |
| 1 — checkpoint audit | `manifests/model_manifest.json`, `tensor_hashes.csv`, `results/model_pairwise_distance.csv`, `weight_delta_by_{layer,head}.csv` | `phase1_checkpoint_audit.md` |
| 2 — benchmark | 4 752 items; `data/generated/` + three split copies + `position_split_manifest.json` | `phase2_instrument_calibration.md` |
| 3 — positional fingerprint | `baseline_position_results_{calibration,validation}_tf.parquet`, `phase3_positional_analysis.json` | `phase3_positional_fingerprint.md` |
| 4 — significance matrices | `results/phase4_significance/<arm>/kv_head_position_significance.{npz,csv}` + `tier2_ablation_significance.*`, four arms | `phase4_6_7_mechanism.md` |
| 4b — bias `B`, distance `D` | `kv_head_bias_matrix.npz`, `distance_significance.npz` | `phase4b_bias_and_distance.md` |
| 6 — probes, patching | `activation_probe_results.json`, `activation_patch_results.parquet`, `head_ablation_results.parquet` | `phase4_6_7_mechanism.md` |
| 7 — low rank, weight deltas | `significance_svd.json`, `significance_low_rank_models.npz`, `phase7_weight_delta_correlation.csv` | `phase4_6_7_mechanism.md` |

Every artifact has a `*.provenance.json` sidecar except the two Phase-3 parquets — see
"Known gaps".

### Written and dry-run verified, not yet run to completion

* **Phase 8 — the intervention screen** (`scripts/phase8_cheap_interventions.py`).
  20 treatments and 48 matched negative controls, ~4 GPU-hours. Started four times and
  stopped four times: once by GPU contention with a concurrent test run, once by an
  orphaned process from a previous attempt, once by the padding defect below, once
  externally. Its flat-RoPE bit-exactness check passes and the baseline arm completes;
  no treatment arm has produced a result.

### Not started

* Phase 5 (recursive layer refinement beyond Phase 4's ladder), Phase 9 (model
  selection), Phase 10 (the five sequential arms), Phases 11-12 (the
  significance x sequentiality factorial and ordering controls), Phase 13
  (preservation), Phase 14 (sealed test), Phase 15 (final report).
* **`scripts/phase14_*.py` does not exist, which is why the sealed split is still
  sealed.** `tests/test_split_discipline.py` enforces this.
* **`figures/` is empty.** The thirteen figure builders in `src/rineya_psa/figures/` are
  implemented and tested against synthetic inputs; none has been run on real results.
* Phase 3's greedy/accuracy pass and the 16K/32K buckets.

### Known gaps and things a reader should not assume

* **The positional deficit is family-shaped, not model-shaped.** Phase 3 found that
  `multi_hop`/`state_constraint` peak in the true middle while `exact_retrieval` is a
  recency gradient and `semantic_single_hop` has a large positional range with no
  significant *middle* deficit. Do not quote a single "edge-middle gap" per model.
* **`tokenize_item` exists twice**, in `benchmark.tokenize` and `evaluation.runner`, with
  the arguments in opposite orders. Both are legitimate — one measures spans from the
  benchmark's recorded character offsets, the other re-finds the evidence sentences in the
  rendered prompt — so both now raise a pointed `TypeError` on a swapped call rather than
  failing somewhere deep.
* **Halo is run in bfloat16**, not its stored float32. This is a recorded approximation
  (`loading.dtype_is_approximation("halo")` is True), not an oversight, and it must appear
  in any Halo result.
* **The Tier-1 attribution estimator is an approximation.** It is labelled as such
  everywhere it appears, and `validate_attribution_against_ablation` is the gate that
  decides whether it may be used at all. Do not report Tier-1 numbers without the
  agreement statistic beside them.
* **Multi-hop may be unmeasurable on this panel.** The hard variant scored 0/36 on both
  accuracy scorers in the pilot. A `multi_hop_guided` variant exists; if it is also at
  floor, that is a capability limit of a 0.6B model and must be reported as such rather
  than quietly dropped.
* **SDPA kernels are not bit-deterministic on this card.** `torch.use_deterministic_algorithms`
  is deliberately left off; we record the fact rather than pay the speed cost. Generation
  is greedy (`do_sample=False`) everywhere.
* **Two Phase-3 parquets have no provenance sidecar**, and the sidecars that exist record
  the git commit at write time — which for the Phase-3 analysis predates later fixes to
  `phase3_analyze.py`. The published tables were regenerated after those fixes; the
  sidecar is what is stale, not the numbers. A remediation pass is adding a dirty-tree
  and changed-file record so a reader can tell a stale sidecar from a current one.
* **The claim "Supa-A1.1 is better on all five families" is WITHDRAWN.** It was an
  artefact of every `semantic_single_hop` answer beginning with the article *the*; on
  content tokens Supa is indistinguishable from Qwen3 there
  (−0.020 ± 0.137). `multi_hop` shares the same answer form and has **not** been
  decomposed, so its between-model numbers are upper bounds. See
  `reports/phase3_positional_fingerprint.md` §3.
* **Open audit findings.** An adversarial audit raised roughly 40 findings; 15 are fixed
  with regression tests, the rest are being worked through. The two that would most change
  how this lab should be read if they hold up: the baseline-equivalence invariant may be
  weaker than §"Two rules" claims (its identity cases may short-circuit to SDPA without
  exercising PSA arithmetic, and the bf16 arm SPEC §5.2 mandates is untested), and the
  `multi_hop` position axis may be confounded — which matters because those are the
  largest effects in the study. Neither is resolved yet; do not treat either as settled.
  `reports/validation_pass_1.md` records what has been fixed so far.
