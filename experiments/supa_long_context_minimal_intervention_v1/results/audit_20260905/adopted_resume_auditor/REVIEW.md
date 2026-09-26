# Independent auditor adoption review

The standalone auditor now checks one logical Phase 1 arm across all retained
physical process segments. Phase 1 scientific outcomes remain pending. This
review did not load a canonical model, execute a GPU job, or write canonical
results or checkpoint files.

The original auditor is preserved as `audit_phase1_results.before.py` in this
directory. The adopted source is `scripts/audit_phase1_results.py`; its tests are
`tests/test_phase1_auditor.py`.

The scorer still starts from raw target and preservation predictions. It checks
IDs, duplicates, original target/reference metadata and sidecars, uses the
authoritative target normalization function, and independently recomputes
preservation scoring. Decision thresholds retain exact `Fraction` arithmetic.
Incomplete or invalid arms cannot produce a winner or license; smoke never
licenses Phase 2.

The new checks pin current source, canonical checkpoint, source data and scorer
hashes; the fixed main recipe and total 300 attempts; ordered encoded dataset
hashes and counts; unchanged frozen parameters; and actual adapter topology and
initialization receipts. Every logical attempt must reconcile applied, scaler
skipped and safety skipped counters, cumulative examples and answer tokens,
ordered-data token consumption, and the complete validation schedule.

Recovery checks independently inspect the immutable arm identity, the exact
physical segment inventory and boundaries, each load ledger, per-segment
initialization and frozen-base receipts, completed segment artifacts, latest
checkpoint file hash, committed JSON receipt, and reconstructed logical progress.
Physical counters are checked as deltas from the resumed prefix. The final
logical training dose is counted once. Model loads and durations are summed over
physical segments; memory uses observed maxima. Missing process ends or metrics
remain explicitly incomplete/lower-bound evidence. A single malformed,
unterminated final physical log line is tolerated only for an interrupted
segment; completed and logical logs remain strict.

## Validation

- `final_results.xml`: 44 auditor tests passed in 12.83 seconds using the required
  `../Chat-Bubble/.venv/Scripts/python.exe` from repository root, with
  `CUDA_VISIBLE_DEVICES=-1`. Two tests use the runner's tiny CPU fixture to prove
  independent acceptance after accumulation failure and after completed-training
  evaluation failure. Other tests exercise raw scoring, exact threshold
  boundaries, corrupt IDs/references/source and initializer receipts, missing
  segments/checkpoints/validation, duplicated resumed prefixes, encoded-data
  mutation, cost omissions, and physical dose accounting.
- `final_strict_hardkill_review.json`: read-only review of the runner's final two
  actual child-process hard-exit fixtures, one during accumulation and one after
  checkpoint commit with a torn callback append. All 60 checks passed for each
  recovered arm; each retained two loads and explicitly incomplete costs. The
  standalone audit imported no Torch module.
- `current_phase1_audit.json`: read-only rescoring of existing canonical result
  artifacts. The historical smoke scores had zero numeric discrepancies but
  fail current-source compatibility after adoption. A1.1 Phase 1 is missing B/C;
  A1 Phase 1 is missing A/B/C. All decisions have `winner: null` and no Phase 2
  license.

## Independent integration findings

Review findings sent to and corrected by the runner owner included a recorded
weight-decay default of 0.01 where the actual unchanged loop uses 0.0, cumulative
physical dose counters that needed subtraction of all starting counters,
omitted hard-kill metric fields, incomplete-stage cost refresh before resume,
and preservation of an already committed dose if recovery fails during
preparation. The final runner fixtures passed the independent checks above.

## Limits

The standalone auditor hashes the opaque checkpoint and verifies its committed
JSON receipt; it does not deserialize optimizer or RNG tensors. Safe checkpoint
deserialization, exact restoration and interruption equivalence are separately
tested by the adopted training and runner CPU suites. These checks do not supply
GPU bitwise-equivalence, historical missing-memory measurements, power-loss
durability, or canonical-model integration evidence. No scientific winner or
further-phase license is established by this review.
