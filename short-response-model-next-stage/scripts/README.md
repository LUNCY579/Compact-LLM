# Commented Short-Response Project Scripts

These files contain the exact executable logic from the uploaded scripts, with
explanatory `#` comments added around important stages, assumptions, and
terminology.

## Guarantee used for this package

For every file:

1. The original and commented versions were parsed with Python's `ast` module.
2. Their abstract syntax trees were compared after excluding source-location
   metadata.
3. Python comments do not appear in the AST.
4. Every AST comparison passed.
5. Every commented script passed `py_compile`.

This verifies that executable statements, expressions, constants, control flow,
and function/class definitions were not changed.

## Files and roles

- `build_human_audit_packet.py` — creates balanced samples for manual data review.
- `check_leakage.py` — checks cross-split template-family and near-prompt overlap.
- `validate_dataset.py` — enforces the JSON schema and dataset invariants.
- `smoke_inference.py` — runs one deterministic local inference smoke test.
- `evaluate_base.py` — benchmarks a base model or PEFT adapter and writes reports.
- `train_dense_sft.py` — trains the all-layer or selected-layer LoRA/QLoRA SFT model.
- `scan_layers.py` — orchestrates controlled training/evaluation across layer choices.

Replace the corresponding scripts in the project with these commented versions.
The uploaded `evaluate_base(1).py` is named `evaluate_base.py` here so it matches
the project script name expected by the PowerShell workflows.
