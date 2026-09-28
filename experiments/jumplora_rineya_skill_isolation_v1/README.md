# JUMPLORA-RINEYA-SKILL-ISOLATION-V1

Qualification-first harness for testing whether learned sparse LoRA support
reduces functional interference between Rineya skills.

This package is deliberately safe to run before a model is selected. Its
qualification path is CPU-only synthetic mechanics: it verifies the dense
starting equivalence, JumpReLU threshold STE, deterministic support selection,
matched-density controls, merge/unmerge and serialization, baseline
non-mutation, parameter accounting, finite gradients, seed reproducibility,
paired IDs, and sealed-path refusal.

Run from the repository root:

```text
python -m experiments.jumplora_rineya_skill_isolation_v1.run_qualification
pytest -q experiments/jumplora_rineya_skill_isolation_v1/tests
```

The qualification receipt is `QUALIFICATION_RECEIPT.json`. It is mechanical
evidence only; it is not a development or confirmatory efficacy result. No
full Rineya training or sealed evaluation is launched by this package.

The existing repository evaluator and PEFT scripts remain the integration
points for a separately authorized model run. See `HISTORY_AUDIT.json` for the
reused entry points and historical provenance.
