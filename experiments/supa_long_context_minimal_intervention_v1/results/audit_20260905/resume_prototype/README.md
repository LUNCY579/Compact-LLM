# UNADOPTED: CPU checkpoint/recovery prototype

This is an isolated review artifact. Nothing here has been applied to live
`src`, scripts, configuration, or canonical results. It does not recover progress
from a prior run that never wrote these checkpoints. Root decides whether to
adopt it; preserving the existing process lifetime may remove the need.

## Evidence and files

- `src/slcmi/training.py`: isolated copy of the current loop with checkpoint hooks.
- `src/slcmi/checkpoint_resume.py`: safe state serialization, atomic replacement,
  fingerprint checks, accounting checks, and RNG restoration.
- `training.patch`: reviewable unified patch for the training file and new helper.
- `baseline_training.py`: source snapshot used for unchanged-loop equivalence.
- `tests/test_training_safety.py`: existing CPU safety tests, copied into this tree.
- `tests/test_resume_equivalence.py`: exact state and interruption tests.
- `cpu_process_worker.py`: separate-process `os._exit(37)` failure fixture.
- `test_results.xml`, `scaler_review_results.xml`, `test_summary.json`: CPU results.
- `REVIEW.md`: independent findings, fixes, and validation sequence.
- `source_manifest.json`: source paths and hashes.

The tests use the existing tiny CPU adapter, with a subclass adding stochastic
Torch, Python and NumPy factors. They never load the canonical model or evaluate
the experiment. A CPU PyTorch build runs the proof. The real GradScaler's CPU
implementation exercises its overflow path; this is not GPU execution evidence.

## Transaction contract

`latest.pt` contains one complete state: only trainable LoRA A/B model tensors,
AdamW moments and parameter groups, scheduler and scaler states, CPU/CUDA Torch
RNG, Python/NumPy RNG, fixed-data cursor, committed attempts, applied/skipped dose
counts, complete trajectory, completed validation records and pending validation.
Frozen model tensors are not written. No state is selected using validation
quality; recovery always reads the newest complete transaction.

The loop writes an attempt-zero checkpoint, then commits after every whole
optimizer attempt and again after each scheduled validation. The same-directory
temporary file is flushed and fsynced before `os.replace` atomically replaces
`latest.pt`. On process death the readable state is the previous or new complete
transaction. A death within accumulation discards that incomplete attempt and
replays its exact examples and RNG from the previous commit. A death after an
attempt commit cannot apply that committed update again. If validation was
pending, recovery performs it before consuming any further training examples.
Final validation is pending at the fixed final attempt too.

Every segment uses the original total `max_steps`; it must never pass the number
of remaining steps. Skipped updates consume their original examples and attempt
budget and do not advance the scheduler. The prototype preserves the original
update mathematics and matched-dose meanings. In the main recipe these remain
attention LoRA layers 17/18/19, protected layers 0/1/2, rank 4, alpha 8, LR 1e-4,
300 attempts, accumulation 8, and full context 3840. Those model/data settings
are the runner's inputs; this prototype does not construct or change them.

Loading uses `torch.load(..., map_location="cpu", weights_only=True)`. Before
copying checkpoint tensors it checks the exact caller identity, actual ordered
data hashes, effective loop options and runtime/source hashes, trainable names,
order/shapes/dtypes, trajectory accounting, validation schedule, AdamW
configuration/LR/moments, fixed scheduler state, scaler progression, and RNG
validity. The loop retains its preexisting adapter-to-float32 initialization;
callers should supply the canonical float32 trainable adapters. Checkpoint
fingerprints identify provenance; they are not a cryptographic signature or a
general guarantee against an adversary rewriting an entire consistent checkpoint.

## Runner interface required before adoption

Call `train` with a separate checkpoint directory per immutable arm/run identity,
`checkpoint_fingerprints`, and `resume_from_checkpoint=True` only for an explicit
recovery. A new run refuses an existing `latest.pt`. Exactly one process may own
and write a directory. This prototype supplies no concurrency lock or supervisor.

`checkpoint_fingerprints` must contain exactly five lowercase SHA-256 digests:

| Field | Runner responsibility |
| --- | --- |
| `recipe_sha256` | Hash the complete frozen recipe, including LoRA arm/initialization, targets, protected layers, context, tokenizer/encoding policy and all other scientific inputs. |
| `train_data_sha256` | `checkpoint_resume.fingerprint(list(train_set))`; includes exact encoded values and array order. |
| `eval_data_sha256` | `fingerprint(list(eval_set))`, or `fingerprint(None)` when absent. |
| `base_checkpoint_sha256` | Verify and identify the immutable frozen base checkpoint and its relevant model configuration. |
| `adapter_initialization_sha256` | Verify and identify the immutable initial adapter for this arm. |

The loop recomputes both data hashes and its effective training options. The
runner must verify the external base/initialization/recipe digests against their
real immutable artifacts before passing them; the loop does not hash a frozen
large model or infer provenance from its in-memory weights.

The runner must construct the same model, adapter topology and immutable data,
then restore through this interface. After a failed save or training exception,
discard the in-memory model/optimizer segment and construct fresh state before
resuming. Do not continue that failed in-memory segment.

`progress_callback` and `log` must be observability-only: RNG-neutral and unable
to mutate model parameters, optimizer, training data or recipe. A committed event
can be missing from an external log if the process dies before its callback.
The runner must reconcile external records from persisted trajectory/validation
and deduplicate by `(run identity, event type, attempt)`. This file transaction
does not promise exactly-once delivery of external events.

The runner must also preserve/aggregate segment IDs, source/checkpoint hashes,
start/end/failure timestamps, model load counts/durations and device resource
measurements. Cumulative loop wall time here excludes uncommitted work and
model-load/restart gaps; it cannot replace resource or end-to-end accounting.
Those integration and aggregation changes are outside this prototype.

## Tested scope and limits

Exact comparisons cover final LoRA tensors, AdamW state, scheduler/scaler state,
all persisted RNG streams, trajectory and validation records, all dose counters,
and unchanged-loop results. They omit only elapsed time and throughput fields.
Tests include safety skips, real CPU scaler overflows, fresh model restoration,
pending periodic/final validation, an exception during validation after RNG
consumption, completed-run idempotence, atomic-replace failure, strict identity
and corrupt-state rejection, and actual child-process death during accumulation
or immediately after a committed attempt followed by a fresh child-process
resume. The original CPU safety suite runs against the isolated training copy.

No GPU bitwise-equivalence, GPU memory/time overhead, multi-GPU topology change,
network-filesystem atomicity, full disk failure, power-loss durability, runner
hosting, or canonical-model integration has been tested. Windows atomic replace
protects the process-death case tested here; portable directory fsync is not
available on Windows. GPU nondeterministic kernels can prevent bitwise identity
even after correct state restoration; the prototype does not change the fixed
recipe's determinism settings.

To rerun from this directory, choose a fresh temporary path contained here:

```powershell
python -m pytest tests -q -o cache_dir=pytest_cache --basetemp=./test_tmp_new --junitxml=test_results.xml
```
