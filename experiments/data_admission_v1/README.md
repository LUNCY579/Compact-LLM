# DATA-ADMISSION-001

Bounded TRAIN-only readiness audit of IFC-001 and GAA-001. This does not admit
training, load model weights, or measure model intelligence. The authoritative
overnight plan remains `OVERNIGHT_PLAN_V3.json` in the quality campaign outputs.

Select at most 128 complete source/counterfactual groups per bundle by the
smallest SHA256 of `DATA-ADMISSION-001|bundle|group`. Two streaming TRAIN passes
(96,000 corpus rows plus two prior schema observations) close shuffled groups.
Retain only selected records; no other split or default validator is opened.
Freeze source, code, runtime, tokenizer assets and selection before tokenization.

Reproduce each shipped public exporter exactly. Keep private metadata outside
messages and preserve target whitespace and empty answers. Measure the pinned
nonthinking native template at 512/1024/2048 tokens with 128 completion reserve.
Prompt and trailing native newline are masked; answer plus exactly one EOS is
supervised when exact generation/full-prefix and answer boundaries permit it.
Boundary failures are retained blockers, never truncations or target conversions.

One CPU worker runs under `supervise_cpu.py`: 2 GiB available RAM at admission,
1.5 GiB ongoing, 512 MiB tree RSS, 900 seconds, 50 MiB worker output, 4 GiB disk.
CUDA/model backends are disabled. Four tokenizer-only asset copies are local and
offline with `trust_remote_code=False`. No packages or assets are downloaded.

Source licences, human review and historical training/evaluation exclusion remain
unqualified. Fixture mask checks are engineering controls, separate from dataset
results. All failures, partial outputs and supervisor costs remain in ignored runs.
