# Troubleshooting

## “Checkpoint tensor keys differ”

The endpoints are not directly merge-compatible. Confirm that both are full checkpoints of the same 12-layer architecture. If one is a LoRA adapter, configure `base_model_path` so the lab first reconstructs the full endpoint.

## “Quantization markers”

Do not average GPTQ/AWQ/GGUF integer artifacts. Materialize/dequantize the full floating checkpoint first. Merge in FP32 and quantize only the accepted final model.

## “Tokenizer file hashes differ”

Use the exact same tokenizer and chat template for every endpoint and candidate. A vocabulary-row mismatch makes weight averaging invalid even when the Transformer shapes happen to match.

## CUDA out of memory

- use the official 12-layer 344M checkpoints rather than loading extra models concurrently;
- keep probability profiling disabled initially;
- lower `max_new_tokens` and benchmark batch size (the bundled evaluator uses batch size one);
- close other GPU applications;
- use `official_no_think` before the longer thinking protocol.

The merge stage itself is streamed on CPU and does not load both full models on GPU.

## Disk usage is too high

Keep `delete_ephemeral_after_evaluation: true`. Only retained aliases remain materialized. Reduce `max_shard_size_gb` only when your filesystem or upload process requires smaller shards; it does not reduce total bytes.

## Run interrupted

Rerun with the same `--run-dir`. Per-case evaluation rows and stage state are durable. Do not change the configuration in place; create a new run for changed settings.

## Sealed test refuses to run

Check that:

- the `freeze` stage completed;
- `selection/selection_freeze.json` exists;
- no completed test access exists for another candidate;
- the test dataset hash did not change.

## Official backend rejects thinking

The official phase evaluator hard-codes `enable_thinking=False`. Use `bundled_v45` for a separately labeled thinking experiment.

## “Candidate cache output digest mismatch”

A run-owned materialized checkpoint was incomplete or changed after construction. The lab deletes and deterministically rebuilds it from the verified endpoints and recipe. If the mismatch repeats, check disk health, antivirus/file synchronization, and available space.

## Preflight requires lineage acknowledgement

The lab cannot infer common initialization from matching tensor shapes. Confirm from your training records that Anchor and Greedy share the same 12-layer architecture, tokenizer, and ancestry, then set `models.lineage_acknowledged: true`. Do not use this switch to merge unrelated checkpoints.
