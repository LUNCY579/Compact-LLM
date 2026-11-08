# TWM-001 runbook

Run commands from the ChatBubble repository root. Use a separate environment; do not downgrade the environment of an unrelated experiment. No Codex task, cloud job, or scheduled training is created by these commands.

## 1. Environment and CPU reference checks

Install an appropriate PyTorch CPU/CUDA wheel for the host first, then:

```console
python -m pip install -r experiments/rineya_triadic_working_memory_v1/requirements.txt
python -m pytest -q experiments/rineya_triadic_working_memory_v1/tests
python -m experiments.rineya_triadic_working_memory_v1.runner smoke --out experiments/rineya_triadic_working_memory_v1/artifacts/reference_smoke.json
```

A run with four skipped native tests has not qualified Qwen3 integration. Resolve the pinned dependency environment and run those tests before proceeding. The current checked-in evidence intentionally reports the skips.

The output file must be new. Change the name for a repeat; never erase earlier evidence just to make a command succeed. Reference tests use CPU and do not download any checkpoint.

## 2. Prepare the exact local model and new data (PowerShell)

Set `$Model` to the real local directory containing Rineya 0.6B safetensors and its matching tokenizer. The lock command rejects missing weights and Git LFS pointers. A base Qwen3 checkpoint is not interchangeable with a named Rineya checkpoint.

```powershell
$M = "experiments.rineya_triadic_working_memory_v1.runner"
$E = "experiments/rineya_triadic_working_memory_v1"
$Model = "C:/path/to/your/Rineya-0.6B-checkpoint"

python -m $M lock --checkpoint $Model --checkpoint-id "Rineya-0.6B-exact-local-checkpoint" --out "$E/artifacts/checkpoint.lock.json"
python -m $M generate --split train --count 640 --retrieval-probes --out "$E/data_generated/train.jsonl"
python -m $M generate --split dev --count 320 --retrieval-probes --out "$E/data_generated/dev.jsonl"
```

Edit the checkpoint identity to its actual release name. Do not write weights, tokens, credentials, or generated datasets into Git. The experiment's `.gitignore` excludes run artifacts and local generated data. Data generation has a fixed default data seed independent of the model training seed.

## 3. Native mechanical qualification — before any efficacy training

```powershell
python -m $M qualify --checkpoint $Model --lock "$E/artifacts/checkpoint.lock.json" --out "$E/artifacts/native_qualification" --device cuda:0 --seed 17 --execute-native
```

This loads only local files. It checks exact neutral/disabled identity, active prefill-versus-tokenwise cache parity, and greedy cached/uncached token equality. A failure writes `failure.json` and exits with an error; it is not an efficacy result. Use `--device cpu` only deliberately: loading and repeatedly evaluating a real 0.6B checkpoint can be slow and RAM-intensive.

The native qualification receipt must match the training source, config, checkpoint lock, device, Torch version, and Transformers version. Changing code/config/runtime invalidates it; run qualification again into a new directory.

## 4. First trained arm and paired baseline

```powershell
$Q = "$E/artifacts/native_qualification/qualification.json"

python -m $M train --arm triadic_e4 --checkpoint $Model --lock "$E/artifacts/checkpoint.lock.json" --qualification $Q --data "$E/data_generated/train.jsonl" --out "$E/artifacts/triadic_e4_s17_train" --device cuda:0 --seed 17 --execute-native

python -m $M eval --arm baseline --checkpoint $Model --lock "$E/artifacts/checkpoint.lock.json" --qualification $Q --data "$E/data_generated/dev.jsonl" --out "$E/artifacts/baseline_s17_dev" --device cuda:0 --seed 17 --execute-native

python -m $M eval --arm triadic_e4 --adapter "$E/artifacts/triadic_e4_s17_train/adapter.pt" --checkpoint $Model --lock "$E/artifacts/checkpoint.lock.json" --qualification $Q --data "$E/data_generated/dev.jsonl" --out "$E/artifacts/triadic_e4_s17_dev" --device cuda:0 --seed 17 --execute-native

python -m $M compare --baseline "$E/artifacts/baseline_s17_dev/predictions.jsonl" --candidate "$E/artifacts/triadic_e4_s17_dev/predictions.jsonl" --families binding update unrelated_after_update scope two_hop three_hop --out "$E/artifacts/triadic_vs_baseline_s17.json"
```

Do not stop at a baseline win. Repeat training/evaluation for `stateless`, `gdn_e1`, and `matrix_state_matched`, with the same data and config. Then repeat the campaign for model seeds 43 and 91. A baseline must be evaluated with the same declared seed and comparison metadata for each paired report, even though its weights are unchanged.

All runs start with fresh sidecar initialization and optimizer state. No adapter is merged into the backbone. The saved `adapter.pt` is bound to the arm, seed, config, source hash, and checkpoint lock; incompatible loads fail rather than silently continue.

## 5. Retrieval-versus-reasoning and ablations

Use `--oracle` on development evaluation to present supporting facts without irrelevant context. Score `hop_edge_retrieval` separately using `compare --families hop_edge_retrieval`. Oracle and ordinary runs intentionally have different input/comparison identities; compare their family summaries, not a falsely identical-input paired report.

For causal interventions, repeat a candidate evaluation into a new directory with `--ablation erase_reads`, `--ablation constant_second_keys`, or `--ablation read_only`. Always use a fresh prefill. The combined cache rejects midstream mode changes, stale ownership, and consumed prefixes. The baseline and stateless control have no meaningful memory ablations.

The offline `core.shuffle_second_keys` utility is for carefully delimited evidence replay only, not a native decoding flag. There is no automatic recurrent-depth/read-controller sweep in this first experiment.

## 6. Retention and prospective holdout

An external established retention file must be explicitly supplied, with JSONL rows containing:

```json
{"id":"retention-001","group_id":"source-001","split":"retention","family":"instruction_following","prompt":"An authorized evaluation prompt","answer":"Its expected answer"}
```

Use the same file and decode settings for baseline and candidate. Do not use generated `retention_demo` rows as evidence that existing Rineya skills were preserved.

After selecting and freezing the development configuration/candidate, generate a fresh prospective holdout using `generate --split holdout`; evaluating it requires `--allow-holdout`. This does not authorize any existing sealed/TEST2 resource, which remains excluded. Do not tune based on its scores. Keep the source, job, model lock, dataset hashes, adapter hashes, and complete per-row predictions together.

## 7. Troubleshooting boundaries

- Missing Transformers, a different Transformers version, non-Qwen3/quantized geometry, missing local weights, or a stale qualification receipt: stop and fix/requalify the environment; do not relabel a reference run as native.
- Too little GPU memory or an excessive peak: record the failure. A state-byte estimate is not peak training memory. Reduce resource demands only as a newly recorded development configuration and requalify.
- Overlength rows: the runner refuses silent truncation. Inspect development prompt lengths, then choose a bounded context/data variant before any holdout is opened. The committed pilot has 512 total tokens, not 64k.
- Slow execution: the reference token loop is expected to be slower than a specialized chunkwise kernel. It establishes mechanics and a bounded learning test, not published H100 throughput.
- Concurrent requests, packed native documents, cached padding, gradient checkpointing, device maps, beam search, and speculative decoding are outside the qualified wrapper contract. Use independent model instances or a later explicitly tested integration.

Record conclusions as reference correctness, native mechanical correctness, native learning, acquisition, retention, and performance separately. There is no automatic model promotion or production deployment.
