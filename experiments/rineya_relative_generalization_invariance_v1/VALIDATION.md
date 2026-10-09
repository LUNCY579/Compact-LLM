# Validation receipt — October 9, 2026

## Executed

- **141 tests passed; 2 optional integration tests skipped; 0 failures.**
- Python bytecode compilation of the package and tests completed.
- The CLI synthetic demonstration completed, including uniform-shift, non-unit-slope,
  selective-change and composition-interference canaries.
- Package statement/branch combined coverage: **82.1%** under coverage.py.
  Missing coverage is concentrated in optional backend execution and CLI orchestration;
  this is not native model qualification.

The Torch tests execute answer-position chunking, the causal next-token shift, equivalence to
full-logit cross-entropy, matching gradients, accumulation/token accounting, frozen-host
protection and LoRA-only updates on an independently constructed tiny causal mechanism fixture.
The fixture is not a Qwen3 architecture and does not represent Rineya behavior.

## Not executed or established

Transformers and PEFT are absent. An installation attempt failed because external package
resolution was unavailable in the environment. The real tiny Qwen3/PEFT tests are explicitly
skipped, not mocked into a pass. The available Torch build is CPU-only. Native Qwen3/Rineya
loading, registered-adapter parity, weighted PEFT stacking, CUDA execution, training efficacy,
world-model improvements, composable-skill gains and broad retention remain **unqualified**.

## Reproduce

```sh
python -m pytest -q
python -m rgi demo --out runs/new-demo-directory
# After installing requirements-native.txt:
python -m pytest -m optional_hf -q
```

Require actual passes for both optional tests. A successful command with only skipped tests
is not backend qualification. Actual Rineya runs still need local hashes, resource checks,
registered evaluation data and the explicit native execution commands.

Machine-readable evidence: `validation/RESULTS.json` and `validation/SYNTHETIC_RESULTS.json`.
All synthetic results are labelled and cannot establish native scientific confirmation.
