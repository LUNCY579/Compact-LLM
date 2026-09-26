# Read-only RESULT and auditor review during smoke

Reviewed the amended `RESULT.md` against the earlier personally recomputed
Phase 0 observations and the raw data-design evidence. No model, training,
source, configuration, or script file was changed in this review.

## RESULT findings

No numerical or interpretation blocker was found in the amended report.
All Phase 0 observation SHA-256 values still match the observations used in my
independent calculation. The report's score tables, contrasts, intervals,
abstention counts, selected-token means, retention rounding, and strict/lenient
contrast are consistent with that recomputation.

A fresh streaming read of the raw demo and LCR corpora also confirmed the report's
supporting-answer counts: demo 168/192 answerable and 144/160 eligible; LCR
553/600 full-dev and 144/150 evaluated. The selected LCR oracle text retains all
507 sampled annotated facts. All 6000 answerable LCR rows have one residual target
present in their original answer, and train/dev have 75 common literal question
strings. Corpus token lengths were checked against the data-design receipt;
the tokenizer measurement was not duplicated during the shared GPU run.

The withdrawn causal-interference, perfect-retrieval, capability-ceiling,
length-causality, checkpoint-equivalence, and significance claims are now
appropriately qualified. The provenance conclusion is accompanied by its
historical-consumption limitation. The upstream mix was not independently
reprocessed by this reviewer; that part was checked against the coordinator's
fresh provenance receipt.

Two documentation updates remain when execution results are filled in:

* `RESULT.md:3` and `:213` still describe repairs/pending execution; repairs are
  complete and smoke is already executing.
* `RESULT.md:222` should explicitly name the bounded budget as **300 optimizer
  attempts per trained arm**, and report applied, scaler-skipped, and
  safety-skipped counts per arm. Matching attempts does not guarantee matching
  applied updates if skips differ.

## Auditor defect found and isolated

`scripts/audit_phase1_results.py:75` assigns the loaded JSON objects in the wrong
order. `required[1:]` is manifest, resource, info, while the assignment was
manifest, info, resource. This affects the independent auditor only; it does not
affect model loading, training, evaluation, or production combination. Initial
missing-output checks returned before reaching the bad assignment, which is why
the earlier empty-stage execution did not reveal it.

Frozen source was left unchanged. Replacing only that assignment **in memory**
validated the actual completed smoke Arm A with no discrepancies: target 20/32,
matched preservation 6/16, strict preservation 3/16. Arm B's live trajectory
field names agree with the auditor's updated applied/scaler/safety checks; no
claim is made about a completed B/C receipt yet.

Exact audit-code evidence is recorded in
`phase1_result_auditor_runtime_patch_receipt.json`:

* Original file SHA-256 before and after:
  `644e5caae318f98490d63fea34bf0ebfb45a448b733ca88d9101a59e486c6467`.
* Executed in-memory corrected code SHA-256:
  `aa08df6e4d37c23b827f62ca9711c3a64d63e8adb2fa59d81f2210c67796c53b`.

No further schema defect was found against the real A receipt and current
training progress schema. The independent auditor covers raw rescoring,
sidecar equality, ID/reference pairing, core/dtype, gate/frozen receipts,
attempt partitions, matched configuration/fingerprint, exact gate thresholds,
full baseline, and resource arithmetic.

It should accompany successful **production combination**, not replace it:
the auditor does not independently validate current source-hash compatibility,
rehash current canonical checkpoints, require a root COMPLETE receipt, or
compare root concatenated observations with all arm observations. The production
combiner supplies the first checks. The root summary/resource discrepancy arrays
must also be inspected; their existence does not currently change the independent
auditor's `decision.status`.

## Corrected in-memory execution after smoke completes

Run from repository root. This executes the frozen auditor with exactly the
recorded assignment correction, writes an independent smoke audit and execution
receipt, and verifies that the frozen source file stayed unchanged. It refuses
to proceed before the production root manifest is COMPLETE. It does not rerun
inference or training.

```powershell
@'
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
lab = Path('experiments/supa_long_context_minimal_intervention_v1').resolve()
script = lab / 'scripts/audit_phase1_results.py'
receipt_path = lab / 'results/audit_20260905/phase1_result_auditor_runtime_patch_receipt.json'
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
root_manifest = json.loads((lab/'results/phase1_smoke/run_manifest.json').read_text(encoding='utf-8'))
assert root_manifest['status'] == 'COMPLETE'
assert root_manifest['canonical_checkpoint_unchanged'] is True
raw = script.read_bytes()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(raw) == receipt['source_file_sha256_before']
source = raw.decode('utf-8')
assert source.count(receipt['old']) == 1
patched = source.replace(receipt['old'], receipt['new'])
assert sha(patched.encode('utf-8')) == receipt['executed_patched_code_sha256']
out = lab/'results/audit_20260905/phase1_smoke_numbers_corrected.json'
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), '--stages', 'phase1_smoke', '--output', str(out)]
namespace = {'__name__': '__main__', '__file__': str(script)}
exec(compile(patched, str(script), 'exec'), namespace)
assert script.read_bytes() == raw
execution = {
    'source_file_sha256_before': sha(raw),
    'source_file_sha256_after': sha(script.read_bytes()),
    'executed_patched_code_sha256': sha(patched.encode('utf-8')),
    'source_file_unchanged': True,
    'output': str(out),
    'output_sha256': sha(out.read_bytes()),
    'root_manifest_status': root_manifest['status'],
}
(lab/'results/audit_20260905/phase1_smoke_audit_execution_receipt.json').write_text(
    json.dumps(execution, indent=2) + '\n', encoding='utf-8')
print(json.dumps(execution, indent=2))
'@ | & '../Chat-Bubble/.venv/Scripts/python.exe' -
```

Before permanently fixing the auditor for bounded runs, preserve its frozen
original bytes under this audit-results directory. A later fixed auditor may
rescore historical smoke observations, but production smoke `--combine` should
not be rerun against changed source hashes as though it were the original run.
