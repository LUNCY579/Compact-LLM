# Repository Architecture

## Data layer

`data.prepare` performs CRC checking and path-safe extraction. `data.validate` reconstructs the full-to-transition alignment and emits a machine-readable report. `data.views` provides deterministic byte-offset indexing without loading 190K records into memory.

## Training layer

`JsonlChatDataset` lazily reads selected JSONL rows, applies the model's chat template, and masks all prompt tokens from loss. `AssistantOnlyCollator` performs dynamic padding. `WeightedCausalTrainer` computes per-example causal loss and supports inverse-problem weighting.

## Inference layer

All evaluation protocols use a common `GenerationRequest` and backend interface. Implementations include:

- local Transformers + optional PEFT adapter;
- OpenAI-compatible HTTP server;
- replay JSONL;
- canonical reference backend for harness validation.

## Evaluation layer

Protocol builders construct direct, transition, ablation, and rollout requests. The parser and deterministic scorers are separate modules so extraction changes can be audited independently of task scoring.

## Reporting layer

Every scored row remains self-contained. Aggregation produces JSON, Markdown, HTML, and a deterministic manual-audit packet. Comparison pairs original problems and emits inferential statistics plus discordant examples.

## Orchestration

`experiment run` executes named stages, persists stage state, and resumes completed work. Training and evaluation can also be run independently for easier debugging.
