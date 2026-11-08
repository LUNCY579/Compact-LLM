# Evidence directory contract

Generated evidence is **not** committed by default. Every run must produce a JSON manifest conforming to `run_manifest.schema.json`, plus immutable raw logs and hashes for all model, data, and configuration inputs.

A paper-result bundle must contain:

- `manifest.json` — identity, commit, config hash, seed, environment, checks, metrics;
- `stdout.log` and `stderr.log` — unedited process logs;
- `config.json` — fully resolved configuration, not only overrides;
- `environment.txt` — package lock, driver/runtime, hardware, and OS;
- `data_manifest.json` — source, license, split, row count, content hashes;
- `model_manifest.json` — base checkpoint, revision, tensor hashes, conversion record;
- `predictions.jsonl` — per-example outputs and scores for paired analysis;
- `timing.json` — warm-up policy, repetitions, prefill/decode separation, memory;
- `telemetry.jsonl` — per-example trajectory summaries without full maps unless explicitly required.

Never overwrite a completed run. Use a content-derived run directory and mark superseded runs without deleting them.
