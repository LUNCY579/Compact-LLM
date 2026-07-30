# Phase 2B Qwen3 layer-types hotfix

## Failure fixed

A physically reduced Qwen3 checkpoint changed `num_hidden_layers` from 28 to
27 but left `config.layer_types` at 28 entries. Recent Transformers validates
that these values have equal lengths, so `save_pretrained()` failed before the
first deletion candidate could be recorded.

## Changes

- `scripts/common/depth_reduction.py`
  - retains the `layer_types` entries matching the physically retained layers;
  - applies the same synchronization to common nested text configs;
  - keeps cache-facing layer indices contiguous;
  - validates unexpected metadata lengths instead of silently saving a corrupt
    checkpoint.
- `scripts/phase_2_layer_selection/create_depth_candidate.py`
  - removes an incomplete candidate directory left by a failed save before
    retrying;
  - refuses to overwrite the source model directory.
- `scripts/phase_4b_structural_reduction/reduce_depth.py`
  - receives the same incomplete-output cleanup because it uses the shared
    depth-reduction implementation.
- `tests/test_phase_2_progressive_reduction.py`
  - adds a Qwen3-style regression test proving that layer count and
    `layer_types` remain aligned after deletion.

## Resume command

The completed probe and full-reference evidence can be reused. Run the original
command again with `-Resume`; the missing candidate manifest makes the runner
retry the failed candidate automatically.
