# Descriptive fidelity-to-behavior report v1

This separate Track B report addresses whether the frozen transformation-fidelity
ordering is accompanied by useful changes in decisions and reference-answer NLL.
It does not change any operator, dose, selection rule, runtime, data split or
confirmatory gate. It executes no model and fits no parameter. Source and this
protocol must be committed, then its own registration frozen and committed,
before any Stage 2 architecture-generation attempt starts.

The fixed parent is CUDA screen registration
`dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3`.
The parent fixes X0 and XG/XS/XR/XM at gates .03/.10/.20/.40 on the same 128
original Track B architecture scenarios. This report requires the parent's
complete, committed 17-cell audit, with complete X0. A failed cell may have a
qualification or generation failure; it remains in this report, with all
behavioral estimates and costs null. Its retained case count is not a denominator.

The static alignment inputs are the complete 16-pair/96-objective CPU matrix and
its committed, independently audited selection. Use only the selected Supa L19 /
Qwen3-4B L24 pair for the following fixed mapping:

| Screen arm | Matrix objective | Alignment interpretation |
|---|---|---|
| XG | B3 | Selected token-normalized and delta-aware bridges |
| XS | BS | Same-pair shuffled bridge control |
| XR | BR | Same-pair random bridge control |
| X0, XM | Not applicable | No donor bridge-fidelity estimate; never substitute zero |

For each mapped arm report mean token delta cosine, token-relative MSE and global
relative MSE for all, ordinary, special and final-prompt tokens, retaining token
counts. These are held-out calibration measurements, reused unchanged at each
dose. Ordinary-token fidelity is the preregistered directional ranking; preserve
the other strata, including the selected B3's worse special-token fidelity.

For each complete cell report audited accuracy and its difference from X0,
paired wins/losses, NLL gain defined as **mean X0 NLL minus mean cell NLL**,
token-output identity, first-token cosine, KL from X0 to cell, invalidity and
token-cap counts, latency, parameter/FLOP inventory and peak memory. Positive
NLL gain favors the cell. Reconcile it with the parent paired NLL delta, whose
sign is candidate minus baseline. Reconcile counts, denominators and reported
means instead of silently accepting inconsistent summary fields.

At each fixed dose, show the three bridge formulations ordered separately by
ordinary-token delta cosine, accuracy gain and NLL gain. Use descending order
and exact equality of reported values for tie groups. Omit failed behavioral
estimates from behavioral ranks, explicitly listing those omissions; retain
their calibration-fidelity ranks. X0 and XM remain in the full cell table but
are not assigned a donor-fidelity rank. Preserve the parent's selection, status
and ranked eligible gates without recomputing or replacing them.

These are descriptive rankings, not a regression or efficacy test. There are
only three bridge formulations at one selected depth pair; they are not
independent randomized observations of a scalar alignment score. Four doses
reuse each fidelity estimate and do not create twelve independent fidelity
observations. No correlation p-value, fitted slope, additional threshold,
candidate-selection decision or capability claim is produced. A general
across-depth predictive relationship is not identifiable from this selected-pair
screen. Confirmation remains a separate, gated experiment.

The CLI verifies committed static files, all current registered Stage 2 source
hashes, its own sources, exact audit artifact inventory, the unchanged parent
selection, all generation attempts and their hash-bound artifacts. Each saved
generation start must postdate the diagnostic freeze and its recorded Git
commit must already contain the exact diagnostic registration and source bytes.
The report reads only the authorized unsealed Track B calibration matrix and
architecture audit; no Native efficacy rows or confirmation files are opened.
It reconciles saved audit records and bytes, without repeating scoring, decoding
tokens or rerunning numerical tensor calculations. All output files are exclusive.

From this directory with the canonical Python environment:

```text
python -m unittest discover -s . -p "test_*.py" -v
# Commit all source/protocol/test files after review and validation.
python run_report.py freeze
# Commit the returned registration before any B architecture generation.
python run_report.py verify --registration <registration.json>
# Only after the complete Stage 2 screen audit is committed:
python run_report.py report --registration <registration.json> --screen-audit <audit-directory>
```

Any changed report rule requires a separately versioned prospective diagnostic.
A later rule cannot be relabeled as this frozen report. A failed or incomplete
report does not open a science gate or authorize replacement data.
