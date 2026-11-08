# TGSA preregistration draft

**Status:** draft. Freeze and timestamp this document before the first paper-primary run. Changes after freeze require a dated amendment that is visible beside the original.

## 1. Research question

Does a learned finite-difference trajectory across sequential attention heads improve causal language-model retrieval/reasoning beyond ordinary cascaded heads, when compared under parameter-, compute-, and latency-aware controls?

## 2. Primary hypothesis

At a frozen model scale and training budget, the full TGSA arm will achieve higher **sealed long-context macro accuracy** than the cascade-only arm, with:

- a stratified paired-bootstrap 95% confidence interval excluding zero;
- an absolute improvement of at least 1 percentage point;
- short-context accuracy regression no worse than 0.5 percentage points;
- validation NLL regression no worse than 1% relative.

The practical thresholds may be amended before any primary data are unsealed, but not after.

## 3. Mechanism hypothesis

The full operator will outperform both:

- a raw finite-difference arm; and
- a parameter-identical history-mean controller with the same accessible two-state history but no explicit finite-difference input.

This tests whether the trajectory representation contributes beyond sequential depth, extra history, and extra parameters.

## 4. Efficiency hypothesis

At least one optimized TGSA configuration will lie on the empirical quality/latency Pareto frontier when compared with:

- cascade-only sequential heads;
- a deeper standard Transformer;
- a recurrent-depth control;
- native attention fine-tuning.

This is a separate hypothesis. Quality improvement alone does not establish efficiency.

## 5. Primary experimental unit

The unit is one sealed evaluation example scored under all arms. Training seed is a replication axis, not a substitute for paired example-level analysis.

## 6. Fixed arms

1. parallel native MHA/GQA;
2. cascade-only sequential heads;
3. parameter-identical history-mean controller without explicit delta;
4. raw finite difference;
5. learned trajectory encoder;
6. learned trajectory plus Q/K/V conditioning;
7. predictor–corrector;
8. full predictor–corrector plus soft significance;
9. recurrent-depth matched control;
10. deeper standard matched control.

Any later adaptive compression or stopping result is a separate experiment family.

## 7. Seeds

Primary small/medium-scale results use seeds:

```text
101, 202, 303, 404, 505
```

The data-order, model-initialization, and sampling seeds must be logged separately even when derived from the master seed.

## 8. Data

### Training

All arms receive the same tokenizer, examples, order, number of tokens, augmentations, and filtering. Generated-task parameters are sampled once and materialized before arm training.

### Development

Visible development data may be used for debugging, hyperparameters, and stopping rules. All such decisions must be frozen before primary unsealing.

### Sealed primary bank

The sealed bank must be independently authored or generated from a separately reviewed specification. Its answers may not be inspected by implementation agents or model developers before final scoring. It must include balanced strata for context length, evidence position, and task family.

### Exclusions

Only malformed examples satisfying a predeclared schema rule may be excluded. Exclusions are determined without access to arm identity or correctness and are reported with IDs and reasons.

## 9. Training controls

Each fairness table freezes:

- base architecture and tokenizer;
- total/trainable parameter targets;
- optimizer, scheduler, clipping, precision;
- batch/effective batch and sequence packing;
- total training tokens and stopping rule;
- checkpoint-selection metric and frequency;
- evaluation decoding settings;
- hardware class and power mode where latency is measured.

Native parameter mismatch is reported, not silently corrected. Matched-capacity controls receive additional parameters through a declared neutral mechanism rather than arbitrary extra training.

## 10. Primary metric

For each example, produce a score in `[0,1]` using the frozen scorer. Macro-average first within each `(task family, context length, evidence position)` stratum, then average strata equally.

Human or LLM adjudication, if required, must be blinded to arm and use a frozen rubric. A deterministic exact/programmatic scorer is preferred.

## 11. Primary analysis

- comparison: full TGSA minus cascade;
- resampling: paired, stratified bootstrap;
- bootstrap samples: 10,000;
- confidence level: 95%;
- aggregation: repeat within each seed, then report seed distribution and a hierarchical aggregate;
- success: CI excludes zero, practical effect threshold met, and both guardrails pass.

A positive mean with a CI including zero is inconclusive, not success.

## 12. Secondary families

### Mechanism family

Raw delta, learned delta, Q/K/V, predictor–corrector, and significance comparisons. Holm correction within this family.

### Position family

Beginning, early-middle, middle, late-middle, end. Holm correction within this family.

### Efficiency family

Prefill latency, decode latency, throughput, peak memory, KV bytes. Report confidence intervals and Pareto plots; do not reduce to one composite score unless preregistered.

### Telemetry family

Velocity convergence, oscillation, gate utilization, importance, attention entropy, intermediate-logit stability. Exploratory unless a predictive rule is frozen on development data.

## 13. Checkpoint selection

Select checkpoints using one frozen development metric that excludes the sealed bank. Apply the same selection rule and evaluation frequency to every arm. Do not select a different checkpoint per test category.

## 14. Hyperparameter policy

Use an equal search budget per arm. Search spaces and trial counts must be recorded. A full operator may not receive more tuning trials merely because it has more controls. Report the default/common configuration and best-within-budget result separately.

## 15. Stopping and failure

A run stops only under the frozen token/step budget, development stopping rule, or numerical/infrastructure rule. Architecture-specific divergence remains an outcome. Infrastructure retries require identical config and a logged reason.

## 16. Blinding and unsealing

Before unsealing:

- tag code and configurations;
- hash checkpoints and data;
- run manifest validation;
- freeze scorers and statistical scripts;
- declare all exclusions;
- record the exact primary command.

Unseal once per frozen checkpoint set. Subsequent changes create a new confirmatory study; they do not overwrite the first result.

## 17. Decision table

| Primary CI | Practical threshold | Guardrails | Conclusion |
|---|---:|---:|---|
| excludes 0 positively | met | pass | primary hypothesis supported |
| excludes 0 positively | not met | pass | statistically positive, practically small |
| includes 0 | any | pass | inconclusive |
| negative | any | any | primary hypothesis contradicted |
| positive | met | fail | quality gain with unacceptable regression |

## 18. Amendments

Every amendment must state:

- date and commit;
- exact changed text;
- reason;
- whether any relevant result was already visible;
- whether the analysis is now confirmatory or exploratory.
