# Evaluator v4 human-adjudication guide

`quality_v4_adjudication.jsonl` is the mandatory deterministic regression packet for the official evaluator.

Each row contains the prompt, generated reply, explicit response contract, approved scored labels, and any review-only diagnostics that should appear. A scored label is allowed only when it describes a concrete visible defect or an explicitly authored contract violation. Reference mismatch, unusual vocabulary, alternate conversational strategy, and minor grammar are not scored defects.

Run:

```powershell
python .\scripts\common\calibrate_quality_profiles.py --strict
python .\scripts\common\verify_quality_evaluation.py
```

Adding or changing a label requires updating its definition in `configs/quality_issue_taxonomy_v4.json`, adding positive and negative examples here, and obtaining human approval before official Phase 2B evidence is migrated.
