# Rineya Cross-Scale Capability Transfer Lab v1

This lab tests whether parameter-space information from official Qwen3-4B and
Qwen3-8B checkpoints can improve a Rineya Supa / Qwen3-0.6B-shaped checkpoint
without increasing its inference-time parameter count.

The study is governed by the frozen preregistration in
`cross_scale_evidence/00_protocol/preregistration.md`. Development code must
not read any Invisible-Limit TEST2 artifact. Dense model weights, activation
caches, Fisher estimates, and generated candidates remain local; manifests,
hashes, metrics, failure records, and reports are versioned.

## Status

- Base commit: `27f807f039fae5c5cd20934100690606b5f9c9e7`
- Branch: `research/rineya-cross-scale-capability-transfer`
- Protocol state: frozen before architecture inspection or model evaluation
- GPU policy: an unrelated GPU job was active at protocol freeze; do not
  interrupt it. Merges and 0.6B evaluation are CPU-first.

## Evidence layout

The `cross_scale_evidence` tree follows the study directive. Large artifacts
are referenced by immutable content hash rather than committed.
