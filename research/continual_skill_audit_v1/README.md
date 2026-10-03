# Continual skill research and independent audit

This is the independent review lane collaborating with the active `multi-reasoning-slft`
task. Historical evidence and published cores are read-only. New tools live in
`experiments/continual_skill_audit_v1`; this directory holds analysis and prospective data.

## Verified artifacts

- **[Completed matched-pilot readout](PILOT004_RESULTS.md):** both arms fail the fixed full
  acquisition criterion and still lose old answers; narrow binding signal, no broad win.

- [Historical LoRA audit](LORA596_AUDIT.md): raw-answer rescoring confirms acquisition of
  only one of three transition families and remaining strict-arithmetic/long-context losses.
- [Shared research protocol](RESEARCH_PROTOCOL.md): acquisition, sequential retention,
  component/joint composition, precision, cost and smaller-model evaluation.
- [New pilot baseline](PILOT004_BASELINE_AUDIT.json): completed source outputs independently
  re-scored, with family floors and rule-ablation subsets.
- [Training invariants](PILOT004_TRAINING_AUDIT.json): matched examples, positions, exposure
  counts and actual schedules across both trained arms, with one-pair wall costs.
- [Adapter geometry](PILOT004_ADAPTER_GEOMETRY.json): CPU effective-weight comparison;
  protected/control norm ratio 0.9981 and cosine 0.9936. No behavioral claim follows alone.
- [Rule coverage](SELECT_CONSTRAINT_COVERAGE.json): two prerequisite conditions are never
  decisive in the historical acquisition bank; ledger status changes answers on 11 rows.
- [Prospective rule-bank review](RULE_BANK_REVIEW.md): v1 shortcut failures preserved; v2
  source-bound synthetic development worlds and explicit limitations. No model calls.
- [Richer binding recommendation](BINDING_BANK_RECOMMENDATION.md): reviewed existing
  multi-hop/salience generator, including issues that must be corrected before fresh use.
- [Update-constraint proposal](UPDATE_CONSTRAINTS.md): actual optimizer increments and
  finite-step checks, a mathematical next-method candidate rather than a model result.
- [Smaller checkpoint inventory](CROSS_SCALE_CANDIDATES.json): local 19-layer 454M Aegis
  candidate for later qualification; metadata only, no size/quality generalization claim.

## Verification

24 tests pass across retention statistics, grouped scoring, receipt/ID failure handling,
whole-answer format sensitivity, LoRA geometry and exact rule generation. Exact binomial
upper bounds were separately checked against SciPy on 135 cases (maximum absolute error
1.23e-15). Synthetic tests are mechanics evidence, never model quality evidence.

The current research goal remains active. No release or comprehensive no-forgetting claim
has been established. Both fixed candidate evaluations are complete, independently checked,
and reported together. All failed engineering attempts remain preserved.

## Coordination

This task: `01a0bc14-8123-7172-a81d-08ae467364cd` in the qualification checkout.
Training task: `01a0bc12-24d1-7661-9837-7a7093f23414` in `ChatBubble-slft`.
Current model run: `research/composable_skills_v1/runs/pilot004` in the training checkout.
The training task explicitly released the GPU after pilot004 completed. Coordinate a new
reservation before any further model execution.
