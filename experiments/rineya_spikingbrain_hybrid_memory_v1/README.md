# SBHM-001: Rineya selective hybrid-memory conversion

An isolated, staged experiment inspired by SpikingBrain and LoLCATs.

**2026-10-09 publication:** the [audited closeout and quality research archive](evidence/research_closeout_20261009/README.md)
preserves the negative recurrence result, the stopped update-order pilot and
qualified offline discoveries. No new intrinsic quality gain is verified.
The archive includes a read-only portable integrity check and final receipt index;
native execution remains on hold. See [STATUS.md](STATUS.md) for the current state.

**Question:** Can one pretrained attention site be replaced by bounded local
attention plus compressed recurrent memory, then recovered with bounded
training, without unacceptable capability loss?

**Original 2026-10-02 delivery status:** runnable pipeline and offline mechanics tests; no native
Rineya weights loaded, no GPU training, no native capability or efficiency result.
The optional pinned-Transformers integration test was skipped in the delivery
runtime. See [STATUS.md](STATUS.md) and committed [evidence](evidence/).

The 2026-10-03 follow-up resolved that skip in isolated pinned CPU and CUDA
environments: the official-library tiny random integration now executes on CPU.
Titan-Transition 1.0.0-devfreeze was selected from the owner's main Chat-Bubble
published-core catalog. Its complete weight inventory, explicit tensor geometry
and pinned tokenizer/config loading were checked without loading a pretrained
model. Native inference and training still require the unchanged resource guards.
See `evidence/qualification_20261003/`; earlier delivery receipts are preserved.

On 2026-10-04 the FP32 native matrix completed on the exact Titan-Transition
source after all variant qualifications, 512-step distillation/recovery stages,
and four 512-row DEV measurements. **No recurrent addition benefit was observed:**
D scored 171/224 distant rows against C's 172/224, with zero D-only repairs.
C and D each retained 18/22 source-correct cases in the separate 64-row synthetic
retention group. See STATUS and `evidence/native_20261004/` for the raw paired
evidence, source identity, memory checks and limitations. One source/site/seed
and remaining full-attention layers prevent a general recurrent-memory claim.
The first matrix stopped at hybrid recovery's VRAM reserve guard.
A fresh memory-equivalent rerun uses supervised-position vocabulary projection
and pressure-only release of inactive PyTorch cache before the unchanged guard.
Pinned CPU tests verify loss, all trainable gradients and an AdamW update against
the original full-logit recovery. All controls, teacher capture and qualifications
were rerun under new code/config bindings. Use
`configs/titan_transition.v2.memory_efficient.json`; the successful bounded run
exercised 42-259-token TRAIN inputs, not the entire 512-token training cap.
Teacher targets were exactly 479.607 MiB within the 512 MiB cap. The unchanged
RAM/RSS/VRAM guards passed at the checked boundaries, and no cache release was
needed in its completed receipts. Numeric continuous free-RAM/free-VRAM traces
were not saved. Native mechanics and this descriptive DEV result do not
qualify broad quality, energy, optimized speed or model promotion.

## Scope and provenance

- Repository: `LUNCY579/ChatBubble`.
- New branch: `research/rineya-spikingbrain-hybrid-memory-v1`.
- Parent: fetched `main@b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`.
- Parent tree: `571ed17d70bb7a5e9ea0f664ad999ac8be704212`.
- All additions live in `experiments/rineya_spikingbrain_hybrid_memory_v1/`.
- No inherited main files, existing research branches, model assets, or banks
  are edited. This is independent of the centralized reasoning branch, not its
  replacement. Historical findings remain on their original branches.

## What is implemented

| Component | Implementation |
|---|---|
| Architecture | One selected attention replacement; local-only C and local + gated recurrent D; original full attention A/B |
| Qwen3 compatibility | Explicit head_dim and all four projection-shape checks; source QK norms, output projection and GQA retained |
| Request state | Local K/V ring-like bounded history plus FP32 KV-group state; append-only positions; independent per-request caches |
| Distillation | Frozen teacher activation capture, one example at a time; selected attention-output loss; cached train rows only |
| Recovery | Fixed-rank attention LoRA at the same site in B/C/D; answer + EOS language-model loss; frozen backbone |
| Controls | A frozen, B unchanged architecture + recovery, C local-only + recovery, D hybrid + recovery |
| Evaluation | Greedy generation, strict and normalized exact match, token IDs/stop reasons, cache storage, instrumented TTFT/decode, CUDA peaks, sampled RSS |
| Diagnostics | Recurrent-state-only reset; zero-mix local equivalence; independent dense fake-quantization utility |
| Integrity | Exact source/tokenizer inventory hashes, code/config/corpus/variant receipts, dated grants, fresh guards, split checks, safe delta save/reload |
| Orchestration | Plan-only matrix builder; explicit execution runs all stages and stops at any failure; exposure-checked paired comparison |

## Architecture contract

For each KV group, the reference update is

`S_t = g_t[:, None] * S_(t-1) + relu(k_t)[:, None] * v_t[None, :]`

where `g_t = exp(logsigmoid(W_g h_t + b_g) / 16)`. State uses FP32. Queries
read the corresponding group state without duplicating persistent state for
all query heads. The output is

`o_proj(local_RoPE_attention + tanh(mix) * RMS(recurrent_read))`.

Local attention uses the host QK norms and host-provided RoPE. The recurrent
path uses normalized, unrotated Q/K. It includes recent as well as older tokens;
local and recurrent information therefore overlap. This is a deliberately small
intra-layer adaptation, **not a reproduction of SpikingBrain-7B, SpikingBrain2.0,
Gated DeltaNet, or a spiking hardware implementation**.

One site beyond layers 0-2 is converted. The illustrative layer 14 is a proposal,
not a measured optimum. The example's 128-token window/rank 8/8 updates are
engineering defaults, not an efficacy registration. Re-register before changing
sites or expanding to more layers. At a 28-layer equal-KV baseline, converting
one layer can remove at most about 1/28 of total KV storage before accounting
for replacement state: this is a learnability pilot, not a large total-memory
savings claim. Other layers continue retaining full history.

The backend intentionally scans tokens in Python/PyTorch. It is slow and stores
training graphs. Constant persistent recurrent state does not imply constant
training memory, lower total peak VRAM, or faster inference. No optimized CUDA,
Triton, event-driven, beam, padding, static-cache, cache-crop, or batching path
is claimed. Native batch size is one; requests are unpadded and append-only.

## Files

- `sbhm/core.py`: pure-Torch attention, LoRA, request memory, actual-storage accounting.
- `sbhm/native.py`: pinned standard-Qwen3 loader, installation, qualification, deltas.
- `sbhm/protocol.py`: provenance, resource guards, grants and receipt validation.
- `sbhm/data.py`: manifest validation, deterministic exposure, answer-only labels.
- `sbhm/pipeline.py`: teacher capture, attention distillation, recovery, evaluation and comparison.
- `sbhm/matrix.py`: plan-only or explicitly authorized paired execution.
- `sbhm/quantization.py`: numerical diagnostic only; not connected to the main factorial.
- `tests/test_experiment.py`: self-contained CPU fixtures and optional real-library tiny-model integration.

## Offline validation

From the repository root:

```bash
cd experiments/rineya_spikingbrain_hybrid_memory_v1
PYTHONPATH=. python -m pytest -q tests
python -m sbhm preflight --config configs/native.example.json
python -m sbhm.matrix --config configs/native.example.json --output /tmp/sbhm-proposed-run
```

The example preflight is expected to exit **2 / BLOCKED**. The matrix command
without `--execute` only prints 22 commands; it neither creates the output
folder nor loads a model. On Windows, use `python -m pytest -q tests` from this
directory; the package is in the working directory so PYTHONPATH is not needed.

Use a separate virtual environment for native prerequisites:

```bash
python -m venv .venv
# Activate the environment for your OS, then install a host-compatible CUDA torch.
python -m pip install -r requirements.txt
```

Do not reinstall or alter the shared repository environment for this experiment.
The CUDA stack still needs actual-source qualification after installation.

## Prepare a native run (not performed for this delivery)

1. Select an exact **local**, standard ungated HF Qwen3-compatible Rineya
   checkpoint. Nothing downloads automatically. Custom/gated/pickle/quantized
   checkpoints and incompatible projection shapes fail closed. Exporting a
   different source or substituting public Qwen is a different experiment.
2. Copy the example config to a new local file. Set `source.path`, a complete
   relative-filename-to-SHA256 map in `source.files`, the reviewed corpus
   manifest path, and the path where an operator grant will be stored. Lock
   every file, including config, weights/index and tokenizer assets. The source
   root must be a clean, read-only snapshot with no unlisted files or escaping
   symlinks. Do not modify the original checkpoint to meet this requirement.
3. Prepare explicit TRAIN/DEV JSONL files. Each row has exactly `id`,
   `family_id`, `prompt`, `target`. Supply this manifest:

```json
{
  "reviewed_for_sbhm": true,
  "splits": {
    "train": {"path": "train.jsonl", "sha256": "REPLACE", "role": "train", "protected": false},
    "dev": {"path": "dev.jsonl", "sha256": "REPLACE", "role": "dev", "protected": false}
  }
}
```

The flag records a real review; it does not perform one. IDs, family IDs and
normalized prompts must be disjoint across splits. Semantic near-duplicates
still require human review. No final/check/test banks are supported. No dataset
is discovered from the rest of the repository. A generated fixture corpus is
explicitly marked **unreviewed** and must not be promoted to an efficacy corpus
just to bypass a gate.

4. Run preflight and record its `code_sha256`, `source_sha256`,
   `config_sha256`, and `corpus_sha256`. A separately authorized operator can
   create the grant at the already-configured path with those exact fields,
   nonempty `approved_by`, timezone-aware `issued_utc`/`expires_utc`, and explicit
   actions selected from `qualify`, `teacher`, `distill`, `recover`, `evaluate`.
   Maximum grant lifetime is seven days. **No such grant is included here.**
   This is a local execution acknowledgment/audit record, not cryptographic
   authentication or independent proof of approval. Editing an example is not
   a substitute for obtaining the owner's native-run permission.
5. Review the printed matrix before execution. For an authorized run only:

```bash
python -m sbhm.matrix --config /absolute/local-config.json --output /absolute/NEW-run-root \
  --execute --allow-model-load --allow-training
```

Every underlying stage rechecks its own grant and resources. The matrix cannot
skip a failed stage or silently omit a control. A stale qualification (over 24
hours), changed source/config/code/corpus/delta/environment, reused output
folder, or altered cached target blocks execution. Existing artifacts are never
overwritten. Optimizer/full-run resume is intentionally unsupported in v1;
continue only through a fresh authorized run with explicit provenance.

## Individual stage interface

Each stage is available via `python -m sbhm STAGE --help`. Common arguments are
`--config`, `--arm`, `--output`, `--qualification`, `--delta`,
`--allow-model-load`, and, for training, `--allow-training`.

The sequence is: qualify A; capture TRAIN teacher outputs; measure frozen A;
for each B/C/D, qualify its initial variant, distill against the same teacher
rows, qualify the saved distillation delta, perform LM recovery, qualify the
recovered delta, then evaluate. Finally compare all four completed DEV receipts.
Teacher output capture is capped at 512 MiB and covers only unique consumed
training rows. Targets are safetensors, streamed one row at a time. No full
teacher/student pair or full-vocabulary teacher logits is resident together.

Distillation trains new parameters and same-site LoRA. All B/C/D arms receive
the same distillation and recovery row order, tokens, steps and optimizer
recipe. B's initial attention distillation can have exactly zero loss; it is a
self-teacher exposure control, not evidence of learning. Parameter counts and
actual runtime costs differ and are reported rather than called matched.

The stronger proposed corpus is `corpora/distant_evidence_v2/`: 256 TRAIN and
512 DEV rows, seven evidence task categories, eight synthetic retention skills,
independent answer/distractor RNG streams, factual counterfactuals, and matched
near/distant evidence-position controls. The owner requested generation and
review with agents. Admission must transparently record the independent agent
review rather than claiming the owner manually inspected the generated rows.
The initial v1 proposal stays unreviewed and rejected because recent filler
values encoded its answers. Token-distance checks alone did not detect that flaw.

The v2 run proposal uses seed 17, layer 14, window 128, rank 8, FP32, and 512
updates per stage, covering every TRAIN row twice in the same order for B/C/D.
Full unique teacher capture is estimated below 480 MiB including safetensors
headers, below the unchanged 512 MiB cap. These are prospective engineering
choices, not a demonstrated sufficient learning opportunity or promotion rule.

Native qualification checks exact identity-wrapper equality, A/B zero-init
identity where applicable, full/chunk/tokenwise logit agreement and greedy-token
agreement, request independence, finite states, and absence of full-history KV
at the converted site. It is **inference mechanics only**, not training-fit,
capability, abstention, optimized-kernel, or release qualification.

## Measurement and next decisions

Evaluation times the instrumented inference loop, including safety checks but
excluding model loading and tokenization. Decode throughput excludes the first
token; one-token completions return null rather than misleading throughput.
CUDA peak allocated/reserved bytes include the loaded model. Cache bytes count
actual distinct tensor storages, not merely views. RSS is sampled, not a true
allocator-level transient peak. Energy is not measured.

Only strict and whitespace/case-normalized exact match are automated; neither
is semantic correctness. Retain raw predictions, IDs and stop reasons for blind
review. `compare` refuses unmatched data, row order, training exposure, missing
recovery/distillation, or reset diagnostics. Its repair/damage counts are
**descriptive DEV results**, not a confidence interval or promotion decision.

Optional manifest `evaluation_groups` maps group names to disjoint DEV family
lists covering all DEV families. Reports preserve that registration. The
comparison now reports the paired D-minus-C contrast, per-family results and
A-correct retention separately within each registered group. V2 uses
`distant_heldout`, `near_control`, and `retention`; labels alone do not certify
distance. Referenced `design` and `review` metadata are hash-checked before model
allocation. Raw predictions and completed, variant-bound, nonempty training
receipts are required for comparison.

V2 has canonical compact-JSON answer contracts. Strict match is the primary
contract measure; normalized match only strips outer whitespace and ignores case.
Neither parses JSON meaning. Short synthetic retention is a narrow original-model
probe. Other 27 layers still have full attention, so a D/C difference measures
the specific trained one-site intervention, not an exclusive recurrent route.
The default zero-initialized forget projection gives a 16-token half-life;
insufficient distant retention or recovery must remain a possible negative result.

For a fixed recurrent-state-reset diagnostic, set `reset_recurrent_at` in a new
config to an interior prompt boundary, obtain a matching grant/qualification,
and evaluate D separately. Only recurrent state is zeroed; local KV and other
layers remain unchanged. This diagnostic cannot be pooled with the main matrix.

After a successful one-site native pilot, prospectively decide whether to
expand conversion and implement a validated optimized kernel. Quantization,
more layer sites, recurrent-depth scheduling, and external retrieval remain
separate later interventions. Read [PROTOCOL.md](PROTOCOL.md) for the scientific
contract and [SOURCES.md](SOURCES.md) for source/version boundaries.
