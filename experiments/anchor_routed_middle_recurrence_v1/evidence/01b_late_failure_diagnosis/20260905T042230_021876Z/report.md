# Isolated late-control failure diagnosis

Status: PASS for failure isolation; L27 repeat-2 remains FAIL.

Question: what triggered the original screen's L27 stop, and did its wrapper
restore the original model? Setup: same first smoke prompt, CPU FP32/eager,
seed 17, unchanged emergency bounds [0.05, 10.0],
one intact forward, one stopped repeat-2 forward, one restored forward.
No generation, threshold change, screen retry, or training.

Results: second-pass RMS / first-pass RMS = 12.176814. Full RMS, norms,
cosines, deltas and checks are in diagnosis.json. Canonical state/files and
restored logits were checked independently. Exact provenance is in run_manifest.json.

Interpretation: this is a late readout control failure, not evidence that the
ten completed middle candidates failed. L27 was excluded from candidate
selection before the screen. Its repeat-3 and remaining repeat-2 cases remain
unrun. The original failed screen directory remains immutable and FAIL.

Non-claims: successful failure isolation does not make L27 safe or establish
recurrence efficacy. It does not authorize routing, PEFT, or sealed evaluation.

Next gate: audit the complete middle-candidate receipts against the unchanged
per-candidate gates before deciding whether any fixed arm is licensed.
