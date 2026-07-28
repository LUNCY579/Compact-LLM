# Post-Phase-3 runnability changelog

This revision supersedes the earlier Phase 4B-only package.

## Phase 4

- Replaced the instruction-only runner with actual compatible-seed DPO training, common re-evaluation, adapter tensor/config validation, FP32 reconstructed-delta greedy merging, trial evaluation, standalone checkpoint creation, reports, and reference manifest.
- Phase 4 now resolves the exact SFT adapter recorded by Phase 3 and re-evaluates the primary adapter with the same evaluator used for every ingredient.

## Phase 4B

- Added automatic matched short/long calibration generation, real component profiling, guided/random pruning candidates, depth controls, common evaluation, recommendation, gate creation, and canonical selected-reference finalization.

## Phase 5

- Added real retrieval traces, contradiction/staleness filtering, auditable retrieval keywords, bounded context assembly, compact-versus-full model comparisons, stable-prefix KV reuse timing, and gate checks.

## Phase 6

- Added non-sealed calibration construction, real activation/oracle-label collection, dense-compute masked generations, quality-frontier selection, and a practical `DENSE_ONLY` fallback when sparsity does not pass.

## Phase 7

- Added per-layer router training, provenance validation, gathered sparse decode, measured latency/VRAM/quality, truthful decode-only routing statistics, and automatic dense-only runtime measurement when Phase 6 rejects sparsity.

## Phase 8

- Added sealed evaluation, A-I evidence collection, complete gate-chain audit including approved Phase 2 deviations, hashing, manifest/copy packaging, and the final human release gate.

## Documentation and environment

- Added a post-Phase-3 installation verifier, end-to-end execution guide, and per-file responsibility documentation for every Phase 4–8 folder.
- Added inline purpose comments and operational/safety comments throughout post-Phase-3 executable files.
