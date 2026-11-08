# Phase 1 evidence contract

The repository contains schemas and runners, not confirmatory outcome artifacts. Generated evidence belongs in a new immutable artifact directory and should be archived with the exact branch commit.

Required archive contents:

- registered study configuration and its SHA-256 hash;
- materialized train/dev/test JSONL and dataset manifest;
- doctor and paired-initialization preflight reports;
- every arm/seed run manifest, fixed final checkpoint, train log, telemetry, and per-example prediction file;
- any failure bundles and replacement-run rationale;
- matrix manifest;
- frozen `analysis.json` and `analysis.md`;
- environment/package capture and Git commit.

The smoke workflow output is suitable for CI debugging only. It must not be added to the scientific evidence ledger as a capability result.

Do not add independently sealed Phase 3 assets to this directory or expose them to Phase 1 tuning.
