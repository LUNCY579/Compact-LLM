# Continual skill audit v1

CPU-only tools for verifying skill acquisition and retention without letting newly correct
answers hide lost skills. These tools do not train models, authorize execution, or promote
artifacts. They leave the existing benchmark implementation and historical evidence intact.

- `metrics.py`: paired retention, conditional risk bounds, whole-group sign randomization,
  sequential base/previous/ever-correct comparisons, and component composition audits.
- `audit_lora596.py`: independently re-score the completed historical LoRA run and record
  hashes for every input. It reuses the historical pure typed-answer grammar but recomputes
  source/candidate sets, acquisition JSON scores and long-context raw-answer correctness.
- `audit_pilot.py`: check completed pilot receipts, re-score raw generations, and report
  family-level and whole-world retention for base/control/anchor contrasts.
- `adapter_geometry.py`: compare effective LoRA updates using small Gram matrices, including
  best scalar fit and residual direction. Uses CPU Torch and safetensors, without loading
  a base model. It rejects unsupported adapter variants.
- `constraint_coverage.py`: distinguish a present condition from one that changes the correct
  answer; preserve historical floors and report decisive subsets separately.
- `rule_bank.py` and `bank_shortcuts.py`: build and audit a separate prospective development
  bank for direct rules, explicit-state comparison and jointly required skills. V1 is
  preserved with known shortcuts; v2 corrects the detected cues. Neither is model-qualified.

From the repository root:

```powershell
python -m unittest discover -s experiments/continual_skill_audit_v1 -p "test_*.py" -v
python -m experiments.continual_skill_audit_v1.audit_lora596 --pilot-root C:\Users\clclc\BusinessProject\ChatBubble-retention-pilot-20260917 --records-root C:\Users\clclc\BusinessProject\rineya-dataset-retention-records --output research\continual_skill_audit_v1\NEW_AUDIT.json
```

Use a new output path; existing audits are never overwritten. Historical input repositories
are required for that particular adapter, but `metrics.py` is standard-library only and
independent of model size, architecture, tokenizer and filesystem layout.

`Outcome.fingerprint` must bind the input, target and scoring policy. Comparison rejects
duplicate IDs, missing rows, changed fingerprints/groups and non-boolean outcomes.
Never manufacture a common fingerprint for different instrument versions. When raw records
cannot establish matching inputs, report missing provenance rather than calling them paired.

Risk bounds require iid units and prespecified sampling for population interpretation.
Group sign-flip p-values require exchangeability/sign symmetry. Both are descriptive on
reused development data. A zero-loss bank with no source successes is uninformative.

See [the independent LoRA audit](../../research/continual_skill_audit_v1/LORA596_AUDIT.md)
and [the shared research protocol](../../research/continual_skill_audit_v1/RESEARCH_PROTOCOL.md).
