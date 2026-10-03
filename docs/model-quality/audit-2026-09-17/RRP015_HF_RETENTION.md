# RRP-015 storage-layout amendment before retention measurement

Both training/checkpoint sets and the three surviving historical D1 models are
being retained. Free disk headroom changed while the task ran. A separate FP32
research-format export would cross the 14GiB free-space floor. That floor is not
lowered and no training checkpoint is deleted.

Use one source-bound BF16 Hugging Face export per arm for both remaining probes.
Long-context already uses that layout. Retention now accepts explicit `hf:<path>`
and verifies the materialization's source hash, output weight hash, and all six
configuration/tokenizer sidecars before calling the existing `run._load`.
The alternative research-export path calls the same `import_qwen_state` and
FP32 conversion, then losslessly saves/restores the resulting DenseReference.
No prompt, tokenizer, EOS, tensor values, generation policy, scorer or threshold
changes. The result records `hf_candidate_identity` and the exact selected path.

Eleven CPU tests pass: rejection of changed source/weights/sidecars/layout, plus
bit-identical state tensors, logits and cached continuation between direct loading
and the research save/restore route on a tiny random model. This is a storage
equivalence fixture, not a new actual-checkpoint native qualification. Actual
candidate loading uses the unchanged official-shape loader already used by the
matched process evaluation.

Export reservation is based on the measured source weight/sidecar byte sizes and
a 2MiB metadata/header allowance; both projected and actual remaining disk must
stay above14GiB. The former64MiB estimate was an arbitrary surplus, not a measured
minimum. Cumulative/retained storage bounds decrease with this layout change.
Keep export receipts after reclaiming only the redundant exported weight file;
complete source checkpoints and all measured predictions remain retained.

This amendment precedes either candidate's arithmetic or long-context outcome.
It changes storage scheduling only and does not authorize another training run,
evaluation row, retry, or model-selection decision.
