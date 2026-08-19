# Small-Model-First Engineering

## Why this harness is not a generic agent framework

A 344M–454M model should not be expected to reliably:

- choose among hundreds of tools;
- parse raw webpages;
- manage transactions;
- infer hidden permission rules;
- decide whether unsupported claims are acceptable;
- remember every previous state;
- recover from malformed state writes without guardrails.

The harness therefore moves complexity into typed software.

## Model-facing contracts

The policy receives compact JSON views and returns strict JSON. It is not asked to emit SQL, shell commands, arbitrary patches, or hidden reasoning.

## Recommended decoding

- `temperature: 0.0` for control stages;
- `enable_thinking: false` for Qwen chat templates;
- constrained JSON decoding where available;
- short output limits by stage;
- one controlled repair attempt for malformed output in future versions;
- deterministic abstention after repair failure.

## Context economics

For Rineya, prefill may dominate because outputs are short. The harness tracks active tokens and aims to reduce prompt work before attempting custom sparse kernels.
