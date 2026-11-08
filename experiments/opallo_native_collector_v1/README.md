# OPALLO-NATIVE-001 — narrow OPALLO-BEH-001 native collector

**Implemented candidate; native Opallo qualification remains pending.** This package is
additive to `d59729433fae8632d0e2bbe6bf99e185fb4c2694`. It does not modify the frozen
`experiments/opallo_behavior_screen_v1` scorer, fixture builder, thresholds or protocol.
Mechanical tests are not a native receipt. No optimizer, training authority, retrieval
claim, retention result, baking, model substitution or release authority is introduced.

## Actual forward path

One stock `transformers.Qwen3ForCausalLM`, Transformers **4.51.3**, eager attention,
manual greedy argmax and the official local published Opallo safetensors. This is **not**
`DenseReference` and does not call the RAB `_load(native=True)` dual-model loader.
There is no `accelerate`/`device_map` requirement, sampling, `.generate()` processor,
quantization, adapter, checkpoint update, remote code, package installer or download path.
The selected existing isolated environment is inventoried; the collector never upgrades
an existing pinned environment. The full runtime recipe must be committed before loading.

Supported candidates are Linux or Windows, CPU FP32 or one CUDA device (FP32/BF16),
with 1–8 explicitly recorded intra-op threads. Other platforms and silent device fallbacks
are rejected. Storage is the published BF16 artifact; compute dtype is a separate field.
The native backend and Windows containment have **not** been executed on the user's host.
Historical RAB scores do not qualify this different forward path.

The loader checks all seven official file sizes and byte hashes, rejects unregistered
model-directory entries that could override loading, checks missing/unexpected tensors,
and compares every loaded tensor to the pinned safetensors after the explicit dtype cast.
An absent tied `lm_head.weight` is accepted only with the verified shared embedding alias.
Weights and all named buffers, including nonpersistent RoPE buffers, are checked for
materialization/finiteness and fingerprinted again after each capture. Only one model is loaded at a time.

## Exact signals and EOS

The tokenizer must encode `UNKNOWN` as one non-special token and decode it exactly to
`UNKNOWN`. Its ID is discovered, never transferred from another core. The actual chat
template is checked against the pinned tokenizer configuration. Rendered text plus
`encode(add_special_tokens=False, truncation=False)` must match direct template tokenization.
Thinking is disabled. Input length must be 1–4072 tokens; 24 further generation steps are
reserved within 4096. Context overflow blocks; it is never truncated or scored as wrong.

The fixed loop computes raw full-vocabulary first-step logits before any gate. It records
`exp(log_softmax(raw_logits converted to CPU float64)[unknown_id])`, its log probability,
raw UNKNOWN logit, log-normalizer, first argmax and the full vector's byte hash. The full
vector is retained as little-endian float64. An independent standard-library log-sum-exp
replay checks these values. There is no top-k renormalization, response-dependent extraction,
rounding before calibration, label conditioning, threshold-based generation or early-stop
cost claim. Numerical tolerances apply only to independent FP64 reduction replay, not to
the frozen selector or its thresholds.

Both official EOS IDs, **151645 and 151643**, terminate generation. The complete generated
ID ledger includes the actual EOS. `generation_steps` includes EOS;
`generated_tokens`/`content_ids` exclude it, preserving the historical content-token count
alongside the actual forward-step cost. EOS as the first token is a valid zero-content
termination; no EOS within 24 steps is BLOCKED. An EOS earlier in the ledger followed by
more tokens is rejected. Missing EOS is never reconstructed from a boolean or text.

The independent stdlib signal replayer permits at most `1e-10` absolute plus
`1e-12` relative tolerance for alternate FP64 reduction order. This is only an
audit comparison: the stored scalar remains exactly `exp(stored log_probability)`;
no tolerance, rounding or epsilon is applied to the registered `p >= tau` decision.
A full-vocabulary stress test and adjacent-representable threshold tests cover this.

Prediction text is the exact content-ID decode with `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False`. No stripping, article removal, answer clipping or
hidden removal of non-EOS special tokens happens in the collector. `decoded_with_eos` is
also retained. The unchanged scorer alone applies its registered normalizations. Per-row
latency measures synchronized forward/signal/decode work after prompt tokenization; the
supervisor separately measures full process time, including load and integrity checks.

## Fixed phase plan

| Phase | New generations | Outputs / authority |
|---|---:|---|
| `qualify` | 24 | First 8 B calibration rows: on, off, on-repeat; exact ID/text/EOS parity, repeat signal equality |
| `calibrate` | 120 | Remaining 56 B + all 64 F; reuse the initial on-8 verbatim; unchanged B-only threshold selection |
| External Git freeze | 0 | Full selected spec + canonical and byte hashes of both calibration captures |
| `check` | 256 | Exactly 128 B + 128 F, only after verifying the committed freeze |
| `score` | 0 | Unchanged B/F/G/FG replay; G and FG never make model calls |
| **Total** | **400** | Retention and release remain separate |

The first eight rows cover only two scenarios; qualification is not a reduced efficacy
sample. F calibration remains diagnostic. The registered primary comparison remains FG
versus B, with all original +4 gains and zero-loss requirements unchanged. There is no
alternate-arm winner selection. No native threshold is supplied by this implementation.

Each job receives prompt-only `{id, messages}` records. Their exact subset hashes are
source-pinned for all seven jobs. Before loading weights, the worker independently checks
the committed authorization, durable run claim, phase reservation, prior job completion,
and (for check) the already-recorded Git freeze. A create-only worker-start marker prevents
reusing the private worker entry point as a silent retry. Review validation uses ID/scenario
pins, not a fixture answer key. The worker does not call a fixture
answer-key function, calibrator or scorer. The parent process performs review and offline
selection. Shared fixture templates remain shared: neither isolation nor a Git receipt
makes the data secret, template-held-out or cryptographically sealed.

## Resource and chronology rules

Every generation capture runs in a separately supervised process. The parent enforces
900 seconds per capture, a **3600-second end-to-end screen deadline including the Git-freeze
pause**, at least 3 GiB available host RAM, and at most 8 GiB observed worker-tree RSS.
CUDA additionally requires 3 GiB free before loading. Resource failures block interpretation.
The memory monitor samples every 0.1 s; this is a sampled guard, not an instantaneous
kernel-enforced RSS cap. A process can briefly overshoot between samples.

Linux uses its own new process group plus parent-death SIGKILL; Windows uses an assigned
kill-on-close Job Object before the GO file permits loading. Setup failure blocks before
loading. Cleanup targets only the created worker/job tree. Windows behavior still needs
host-native qualification. macOS is deliberately unsupported in this first collector.

A cooperative per-user host lock prevents simultaneous collectors across clones. A durable
subject claim in Git's common directory prevents restarting the same subject with a fresh
output path or sibling worktree. A phase is reserved before launch; failed/partial phases
are not retried or skipped. A killed supervisor leaves a stale lock for reviewed disposition,
not automatic deletion. No reset/retry command exists. A reboot, clock rollback, source,
asset or runtime drift blocks continuation. Cloning another repository or editing files by
hand can defeat a local workflow; this is not an adversarial sandbox or external attestation
system. Other training jobs must be stopped separately: the required human quiescent-host
attestation and memory guard do not establish that unrelated software honors this lock.

Seven serial model loads are used across the seven captures. Full-vector signal storage is
`392 × vocabulary_size × 8` bytes before JSON/log overhead (about 455 MiB for a 151936-token
vocabulary). This is inference evidence, not a model export. No model weights go into Git.

## Required approvals (none are synthesized)

`templates` emits pending human review for all 192 targets, a pending four-category union
audit and a pending authorization. A completed audit must identify the actual reviewed
training, historical Demo, exposed panel and prior repair-selection inputs; source commits,
paths, byte hashes, row counts, full-prompt/latest-user match counts and near-template
review are required. All audited files must be available under the repository root at
initialization and subsequent phase checks. The implemented first gate requires zero exact
matches and no exclusions/unresolved findings. Missing material or required exclusions
requires a separately reviewed registration; the collector does not waive the audit.

Review receipts are source-bound human attestations. The collector checks their completeness,
input hashes and Git bytes; it does **not** independently establish that a human read them
or rerun a corpus-overlap algorithm. Do not turn pending values into approval without doing
that review. The earlier source inventory and 44 preflight tests did not complete this gate.

## Commands

Use Python 3.11 or newer. Run from the repository root after committing this additive
code to a **new branch**. Preserve LF source bytes; the delivery's new-worktree helper
uses `git -c core.autocrlf=false worktree add`, and this package pins LF in its own
`.gitattributes`. No inherited attributes or source files are changed.
All output files are create-only. Paths in Git receipts are repository-relative POSIX paths;
CLI paths to model/session directories use native `pathlib` semantics on Windows or Linux.
No CI workflow is added or dispatched.

```sh
python -m unittest discover -s experiments/opallo_behavior_screen_v1/tests -v
python -m unittest discover -s experiments/opallo_native_collector_v1/tests -v
python -m experiments.opallo_native_collector_v1.cli inventory --out inventory.json
python -m experiments.opallo_native_collector_v1.cli templates --out opallo_review
```

`inventory` intentionally returns **2 / BLOCKED_BEFORE_NATIVE**. It is not a loading or
qualification command. `--model-dir PATH` additionally checks provisioned files without
loading a checkpoint. A current inventory cannot prove pending human/native gates.

In the intended, already prepared isolated host environment, create a candidate recipe
(no tokenizer/checkpoint load):

```sh
python -m experiments.opallo_native_collector_v1.cli recipe \
  --device cuda:0 --dtype bfloat16 --threads 2 --out opallo_review/recipe.json
```

For CPU choose `--device cpu --dtype float32`. Do not install/upgrade packages in the
historically pinned environment to get this command past a failure. Record a separately
qualified environment. The recipe hashes actual installed dependency bytes and records
package versions, verified unshadowed import locations, Python executable, OS/host, CUDA device identity, dtype, backend,
TF32 flags, thread count, effective decoding policy and the complete source manifest.

Complete the review honestly. Human status becomes `APPROVED`; each row needs `APPROVED`
and three true checks. Audit status becomes `PASS_REVIEWED_UNION`, with named reviewer,
timezone-aware timestamp, all four source categories, zero exact matches, and
`REVIEWED_NO_UNRESOLVED_FINDINGS` for every source's near-template review. Authorization
status becomes `APPROVED_FOR_QUALIFICATION`; supply a unique run ID, explicit quiescent-host
attestation and the final **byte SHA-256** and relative path for each recipe/review file.
Leave every `licenses_*` field false. Commit the code and those receipts before initialization.
`--authorization-commit` must contain exactly the checked working bytes and descend from
the frozen subject. No self-referencing commit SHA belongs inside its own file.

```sh
python -m experiments.opallo_native_collector_v1.cli init \
  --repo . --session /ABSOLUTE/NEW/SESSION --model-dir /EXACT/LOCAL/OPALLO \
  --authorization-commit FULL_40_HEX_COMMIT \
  --authorization-path opallo_review/authorization.json
python -m experiments.opallo_native_collector_v1.cli qualify --session /ABSOLUTE/NEW/SESSION
python -m experiments.opallo_native_collector_v1.cli calibrate --session /ABSOLUTE/NEW/SESSION
```

After calibration, copy `SESSION/calibrate/FREEZE_PACKET.json` **unchanged** into a reviewed
receipt path in this new branch and commit it. That packet includes the entire selected
specification and both capture hashes, binding runtime/source/session provenance. Do not
open or generate check outputs before this commit. The collector performs read-only local
Git verification; it neither creates that scientific approval commit nor claims a remote
or signed attestation. Reserve enough of the fixed screen deadline to complete this pause.

```sh
python -m experiments.opallo_native_collector_v1.cli freeze \
  --session /ABSOLUTE/NEW/SESSION --commit FULL_40_HEX_FREEZE_COMMIT \
  --packet-path opallo_review/calibration_freeze.json
python -m experiments.opallo_native_collector_v1.cli check --session /ABSOLUTE/NEW/SESSION
python -m experiments.opallo_native_collector_v1.cli score --session /ABSOLUTE/NEW/SESSION
```

A failed phase consumes its reservation. Do not delete receipts, copy a session, remove a
run claim or change runtime/source to work around a block. A new attempt requires a separate
reviewed failure/restart disposition and new experiment authority outside this CLI.

Even a complete native screen is only the original engineering screen. The offline scorer
retains `OFFLINE_REPLAY_ONLY` and all closed claims. A passing registered FG-vs-B contrast
can advance only to the protocol's exposed retention review and blind semantic-abstention
annotation; it does not establish retrieval recovery or lower hallucination rates.

## Source references and scope

The binding project sources are the exact subject's `PROTOCOL.md`, `REGISTRATION.json`,
`screen.py`, published `FILES.json` and `generation_config.json`. The prior
`DECISION_AND_NATIVE_EXECUTION_CONTRACT.md` supplied the 24/120/256 handoff and explains
why the historical `DenseReference` probe is not a drop-in stock-native collector.

API verification: Hugging Face Transformers v4.51.3 Qwen3 documentation,
https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen3 — raw logits,
`past_key_values`, `use_cache`, and `logits_to_keep=1` on the stock native forward.
No claim of historical-runtime equivalence follows from using this API.
