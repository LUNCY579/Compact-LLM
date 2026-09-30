# Stored paired correctness audit

This standard-library helper describes stored CONFIRM-002 answerable outputs on
the spent HSKILL_SOUP_V3 bank. It does not load a model, generate a bank, reconstruct
weights, resample observations, or compute significance. Historical files stay
unchanged. Derived output is create-only and must be outside every input directory.

Existing `composition_confirmation_v2/audit_confirm002.py` checks aggregate
correctness, boundary flags, and attempts, but dictionary construction silently
collapses duplicate IDs and its flag comparison omits answerable correctness.
The centralized MiMo paired analyzer accepts DEV parent/seed booleans and computes
bootstrap intervals; it does not validate this stored-generation/scorer contract.
Neither provides this receipt adapter.

The helper requires unique IDs and complete full-bank coverage for every supplied
arm, matching bank/scorer hashes, schema, full runtime, EOS, and row metadata.
It re-scores text using only the historical scorer whose full SHA256 is pinned in
the code. Stored flag/summary disagreements remain visible in output. They do not
silently replace or edit the original evidence. Missing or incompatible evidence
exits with `AUDIT_FAILED` and creates no report. No missing labels are imputed.

From the repository root, using an installed Python 3.11+ interpreter:

```powershell
python -B -I -S experiments/confirm002_paired_audit_v1/paired_correctness.py `
  --bank experiments/composition_confirmation_v2/HSKILL_SOUP_V3.jsonl `
  --runs experiments/composition_confirmation_v2/B_H_SPARSE.json experiments/composition_confirmation_v2/B_H_COMPOSE.json `
  --scorer C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/boundary_rule.py `
  --historical-result experiments/composition_confirmation_v2/CONFIRM002_RESULT.json `
  --output experiments/confirm002_paired_audit_v1/new-audit.json
python -B -I -S -m unittest discover -s experiments/confirm002_paired_audit_v1 -p 'test_*.py' -v
```

Use a fresh output name. The checked-in derived report is
`CONFIRM002_PAIRED_CORRECTNESS.json`; the human interpretation and provenance are
in `REPORT.md` and `SOURCE_PROVENANCE.json`. Tests use synthetic fixtures, including
a deliberately narrow synthetic scorer, and do not open the historical bank.
The real report separately verifies the complete historical scorer hash.

`audit_documents` is a pure adapter for injected synthetic fixtures.
`audit_paths` is the real-evidence entry point and always verifies the pinned rule.
Reusing a future bank under this receipt schema is descriptive only; it never
grants permission for generation or confirmation.
