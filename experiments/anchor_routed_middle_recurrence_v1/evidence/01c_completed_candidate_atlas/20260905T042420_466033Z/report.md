# Middle reapplication screen

Status: PASS for the fixed-recurrence mechanical gate.
Original campaign status: FAIL_L27_CONTROL.
The campaign stopped at excluded late control L27; it remains FAIL and is not resumed. All ten prespecified middle layers (both depths, eight cases each) and L25 completed before the stop. The isolated diagnostic reproduced a 12.1768x RMS ratio under unchanged bounds and verified bit-identical restoration. These completed per-candidate gates are recomputed exactly from the immutable raw evidence; no threshold, candidate, score, or selection rule changed. The original per-candidate protocol licenses fixed development qualification for passing middle arms, independently of the already-excluded late control. This is an explicit scope decision after diagnosis, not a passing label for the failed campaign.

Question: which pretrained middle operators tolerate one/two extra applications
at their native site? Setup: Supa-A1.1, CPU FP32 eager, seed 17, 4 threads,
eight historical smoke prompts, frozen configuration; exact hashes/hardware and
commit are in run_manifest.json. Lower/upper layers and abstention head retained.

Mechanical qualification: passing matching baseline receipt required; no-cache
entry guard, exact argument reuse, finite hidden/logit checks, frozen parameter
hash and checkpoint-file comparison. Restoration evidence and any stage stop
must be read with the qualification note above and the source receipts.

Results: see ranking.md, atlas.json, summary.csv, hidden_state_metrics.csv,
logit_metrics.csv and raw_results.jsonl. Selected layer: 12.
Each row is a separately bounded case; stopped cases are recorded and not retried.

Interpretation: smoke gate only; fixed development qualification required. Stable norms, low divergence, and task
scores are separate columns. A removable layer is not assumed reusable.

Non-claims: no reasoning/capability improvement, learned routing, training,
long-context, cached recurrence, or sealed efficacy evidence. Small prompt count
and multiple comparisons preclude efficacy selection claims; task changes are
descriptive only. RMS ratios/rho are not proofs of convergence or contraction.

Next gate: fixed recurrence on unsealed MERGEDEV-fast. Adaptive routing, PEFT and sealed evaluation remain closed.
