# ArXiv paper evidence and release checklist

A checked box means an immutable artifact exists, not that someone remembers performing the step.

## A. Claim integrity

- [ ] Every abstract claim maps to a preregistered table/figure cell.
- [ ] “Sequential,” “trajectory,” “finite difference,” “momentum,” and “differential” are defined distinctly.
- [ ] The paper does not claim first use of sequential heads, inter-head interaction, momentum, or differences in Transformers.
- [ ] The contribution is stated as a tested coupling of shared-carrier finite differences to later head computation.
- [ ] Negative and null results are retained.
- [ ] Confirmatory and exploratory analyses are labeled.
- [ ] Limitations include serial latency, kernel maturity, scale limits, and telemetry non-faithfulness.

## B. Related work

- [ ] EfficientViT/Cascaded Group Attention compared directly.
- [ ] CODA/Cascaded Head-colliding Attention compared directly.
- [ ] Talking-Heads and other inter-head mixing covered.
- [ ] Momentum ResNet and Momentum Transformer covered.
- [ ] Differential Transformer distinguished mathematically.
- [ ] Dynamic token/head pruning work covered before compression claims.
- [ ] GQA and checkpoint-uptraining constraints covered.
- [ ] Final search date, databases, queries, and citation graph audit archived.
- [ ] An independent reader challenged the novelty table.

## C. Code identity

- [ ] Branch and exact commit recorded.
- [ ] Signed or immutable release tag created.
- [ ] Working tree clean at every paper run.
- [ ] Resolved config hash recorded.
- [ ] Package lock and container/environment recipe archived.
- [ ] Unit/integration tests and logs archived.
- [ ] Disabled-control native equivalence result archived.
- [ ] Full-sequence/cache equivalence result archived.

## D. Model identity

- [ ] Base model repository, revision, and license recorded.
- [ ] Every source tensor hash recorded.
- [ ] Conversion script and mapping table archived.
- [ ] New/copied/transformed/frozen/trainable tensors enumerated.
- [ ] Final checkpoint tensor hashes recorded.
- [ ] Parameter counts independently recomputed.
- [ ] KV-cache bytes and layout recorded.

## E. Data identity and leakage control

- [ ] Data source cards and licenses archived.
- [ ] Raw and processed hashes recorded.
- [ ] Deduplication and contamination scans documented.
- [ ] Grouped/leakage-resistant split procedure archived.
- [ ] Exact tokenizer revision and preprocessing recorded.
- [ ] Training token counts independently checked.
- [ ] Sealed bank authorship and access policy documented.
- [ ] Exclusions decided blinded and listed by ID.
- [ ] Existing diagnostic sets explicitly excluded from primary claims.

## F. Experimental fairness

- [ ] Same token stream/order across arms.
- [ ] Same optimizer/scheduler/precision where applicable.
- [ ] Equal hyperparameter-search budget.
- [ ] Native and matched parameter counts reported.
- [ ] Training FLOPs reported or transparently estimated.
- [ ] Inference FLOPs reported.
- [ ] Prefill and decode latency reported separately.
- [ ] Peak memory and KV-cache memory reported.
- [ ] Deeper and recurrent compute-matched controls included.
- [ ] All five registered seeds included, including failures.

## G. Statistics

- [ ] Primary endpoint frozen before unsealing.
- [ ] Per-example predictions archived.
- [ ] Stratified paired-bootstrap code and seed archived.
- [ ] Effect sizes and confidence intervals reported.
- [ ] Practical significance threshold evaluated.
- [ ] Guardrails evaluated.
- [ ] Multiple-comparison correction applied to registered secondary families.
- [ ] Seed-level results shown, not only pooled mean.
- [ ] Human/LLM scorer agreement and blinding reported if used.

## H. Mechanistic evidence

- [ ] Cascade separates sequential depth from trajectory guidance.
- [ ] Two-state no-delta control separates subtraction from history access.
- [ ] Raw-delta control tests linear-collapse concern.
- [ ] Q, K, and V conditioning eventually ablated separately.
- [ ] Predictor and correction ablated.
- [ ] Soft significance ablated before hard compression.
- [ ] Velocity zero/shuffle/reverse/noise interventions performed.
- [ ] Telemetry predictor evaluated out of sample.
- [ ] Simpler entropy/margin/logit-change baselines included.

## I. Systems evidence

- [ ] Hardware model, firmware/driver, power mode, and software stack recorded.
- [ ] Warm-up and repetition policy frozen.
- [ ] Raw timing samples archived.
- [ ] Batch and sequence lengths reported.
- [ ] Median, p90, throughput, and variance reported.
- [ ] Telemetry disabled during performance timing.
- [ ] Fused and unfused paths labeled.
- [ ] Quality/latency/memory Pareto plot included.

## J. Paper figures

- [ ] Architecture diagram distinguishes token axis, head-depth axis, and layer-depth axis.
- [ ] Main result includes cascade and raw-delta controls.
- [ ] Position-by-context-length heatmap included.
- [ ] Quality versus prefill/decode latency shown.
- [ ] Velocity/gate figure uses held-out examples and avoids cherry-picking.
- [ ] Failure taxonomy shown.
- [ ] Parameter/FLOP/latency table reconciles totals.

## K. Reproduction package

- [ ] One command reproduces Phase-0 tests.
- [ ] One command generates every paper table from raw artifacts.
- [ ] One command validates manifests and hashes.
- [ ] Public configs contain no hidden defaults.
- [ ] Small reproduction path fits accessible hardware.
- [ ] Full-scale resource requirements are disclosed.
- [ ] README distinguishes reproduced facts from expected outcomes.
- [ ] Data/model restrictions and unavailable artifacts are explicit.

## Release decision

Release only when:

1. the primary conclusion follows the preregistered decision table;
2. all load-bearing evidence has an immutable manifest;
3. a clean environment reproduces tests and one representative result;
4. related-work and claim audits are complete;
5. every unresolved limitation is written into the paper rather than hidden in issues or private notes.
