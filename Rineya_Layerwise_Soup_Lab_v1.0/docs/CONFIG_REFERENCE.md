# Configuration Reference

The authoritative annotated example is `configs/experiment.example.yaml`.

## `project`

- `name`: human-readable experiment family.
- `seed`: general reproducibility seed.
- `run_root`: parent directory for isolated runs.
- `run_name`: fixed name for easy multi-command resume; `null` creates a timestamp.
- `research_mode`: descriptive flag recorded in manifests.

## `models`

- `anchor_path`, `greedy_path`: full Hugging Face checkpoints or PEFT adapters.
- `tokenizer_path`: tokenizer/chat-template source used for every candidate.
- `base_model_path`: required only when an endpoint is a PEFT adapter.
- `expected_layers`: exact layer-count guard; use `12` for the current Rineya student.
- `layer_key_regex`: extracts layer indices from parameter names.
- `merge_accumulation_dtype`: v1.0 requires `float32`.
- `output_dtype`: preserve Anchor dtype or force a supported floating dtype.
- `max_shard_size_gb`: safetensors shard target.
- `require_non_floating_equal`: buffers must match unless explicitly disabled.
- `allow_quantized_sources`: unsafe by default; merging dequantized weights is preferred.
- `lineage_acknowledged`: mandatory manual statement that endpoints share a compatible origin. Shape compatibility cannot prove common initialization.

## `datasets`

- `dev`: coefficient-selection split.
- `test`: sealed final split.
- `allow_id_overlap`, `allow_prompt_overlap`: normally remain `false`.

## `evaluation`

- `backend`: `bundled_v45`, `official_phase_v45`, `external`, or `mock`.
- `selection_protocol`: the one protocol used for every coefficient decision.
- `protocols.*`: generation and response-control settings.
- `capture_token_probabilities`: stores generation-time entropy/EOS diagnostics.
- `run_secondary_protocols_on_finalists`: after selection is frozen, evaluate finalists under other enabled protocols as descriptive evidence only.

## `objective`

Every metric has:

- a dotted path into `summary.json`;
- a nonnegative weight;
- a transform such as `maximize_01`, `minimize_01`, or `target_range`.

`case_utility` controls per-response influence calculations. It is separate from aggregate candidate ranking so both can be inspected.

## `statistics`

- `bootstrap_samples`: 2,000 is a practical default; increase for final analysis.
- `confidence_level`: interval level.
- `minimum_accept_probability`: coordinate-search move threshold.
- `minimum_objective_delta`: minimum observed gain before a move is accepted.

## `experiments.uniform`

Runs the explicit 50/50 control.

## `experiments.global_sweep`

- `alphas`: coarse predeclared grid.
- `refine_around_best`: adds a local grid.
- `refine_step`, `refine_radius`: local grid geometry.

## `experiments.layer_influence`

- `layers`: `all` or an explicit list.
- `bidirectional_swaps`: recommended.
- `centered_alpha_grid`: single-layer values around the global model.
- `smoothing_penalty`: weak adjacent-layer regularization.
- `global_shrinkage_penalty`: pulls noisy coefficients toward global α.
- `minimum_reliability`: zeroes weak deviations before smoothing.

## `experiments.layer_search`

- `max_passes`: coordinate-descent passes.
- `alpha_grid`: values tested for each layer.
- `top_layers_only`: restricts search to the most influential layers.
- `max_candidate_evaluations`: hard evaluation budget.
- `pairwise_top_k`: optional pair search among the top-k layers; default `0`.

## `storage`

- `retain_candidate_weights`: stable aliases retained after evaluation.
- `delete_ephemeral_after_evaluation`: saves disk.
- `max_disk_gb`: stops materialization when the run exceeds the declared budget.

## `sealed_test`

- `enabled`: controls final evaluation.
- `allow_repeat`: should normally remain `false`.
- `require_selection_freeze`: should remain `true`.

## `experiments.probability_profile`

- `enabled`: teacher-force both endpoints over the same target trajectory.
- `max_cases`: development cases included in the diagnostic.
- `top_k`: endpoint top-k size used for overlap and coarse divergence.
- `include_reference_trajectory`: prefer benchmark references; otherwise use the Anchor reply.

The reported divergence is a conservative coarse-grained JS lower bound on a common partition, not full-vocabulary JS, and it is not a coefficient-selection input.

## Cost controls

The largest cost is repeated generation/evaluation, not weight interpolation. Reduce exploratory cost by selecting a subset in `layer_influence.layers`, setting `top_layers_only`, lowering `max_candidate_evaluations`, disabling pairwise refinement, and disabling secondary protocols. Record every such choice before looking at results.
