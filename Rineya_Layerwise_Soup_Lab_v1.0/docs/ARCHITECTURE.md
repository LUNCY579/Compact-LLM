# Architecture

## Design goal

The lab separates immutable inputs, numerical candidate construction, generation/evaluation, statistical analysis, and final selection. This prevents a convenience script from silently changing a source model or using test evidence during coefficient tuning.

## Execution graph

```text
CONFIG
  │
  ▼
PREFLIGHT
  ├─ checkpoint compatibility
  ├─ tokenizer fingerprint
  ├─ quantization rejection
  └─ dev/test leakage audit
  │
  ▼
ENDPOINT EVALUATION
  ├─ anchor
  └─ greedy
  │
  ├───────────────┐
  ▼               ▼
UNIFORM 0.5     GLOBAL α SWEEP ──► LOCAL REFINEMENT
                          │
                          ▼
                    GLOBAL BEST
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
 BIDIRECTIONAL LAYER SWAPS       CENTERED SINGLE-LAYER α GRIDS
          │                                │
          └───────────────┬────────────────┘
                          ▼
             PER-RESPONSE INFLUENCE MATRIX
                          │
                          ▼
          BOOTSTRAP RELIABILITY + SHRINKAGE
                          │
                          ▼
                  LAYER-WISE SEED
                          │
                          ▼
         BOUNDED COORDINATE / PAIRWISE SEARCH
                          │
                          ▼
                   DEVELOPMENT FREEZE
                          │
                          ▼
                 ONE SEALED EVALUATION
```

## Numerical candidate identity

A candidate is identified only by the values that change its weights:

- endpoint type;
- global Greedy coefficient;
- per-layer overrides;
- optional parameter-group overrides.

Labels and notes do not change candidate identity. This allows `uniform`, `global_best`, and another alias to point to the same numerical checkpoint without regenerating it.

## Streaming merge

For each tensor key:

1. load the Anchor tensor on CPU;
2. load the Greedy tensor on CPU;
3. verify shape equality;
4. classify it as embedding, Transformer layer `l`, LM head, final norm, or other;
5. select `α`, `α_l`, or a group override;
6. compute `(1-α)A + αG` in FP32;
7. cast to the configured output dtype;
8. append to a bounded safetensors shard;
9. release the tensors before proceeding.
10. fingerprint the completed checkpoint and record its path-independent digest in the merge manifest.

This avoids loading two complete checkpoints plus the output model into RAM simultaneously.

## Candidate lifecycle

An ephemeral candidate can be:

```text
recipe → materialize → evaluate → hash evidence → delete weights
```

Because every merge is reconstructed from the immutable endpoints, deleting an intermediate candidate does not prevent reproducibility. Selected candidates are rematerialized and retained.

## Durable state

`state/run_state.json` stores:

- stage status;
- candidate registry;
- aliases;
- evaluation paths and hashes;
- selection metadata;
- bounded event history.

Every write is atomic. A run lock prevents two pipeline processes from mutating the same run concurrently.

## Evaluation separation

Generation and scoring are separate from coefficient search. The evaluator produces case rows and a summary. The objective module applies predeclared transforms and weights. Statistical modules compare paired case IDs. Search code never parses model text directly to invent a new score.

## Isolation and integrity boundaries

- User-owned endpoint and tokenizer files are read-only inputs. Preflight records size, timestamp, and SHA-256 evidence; later candidate construction fails if those sources change.
- Run-owned candidate caches carry the numerical recipe, endpoint fingerprints, and output-checkpoint digest. A cache with missing files, wrong sources, a different recipe, or a mismatched output digest is discarded and reconstructed.
- Each evaluation directory has a `lab_evaluation_manifest.json` binding the model path, recipe, endpoint fingerprints, dataset hash, evaluator-policy hash, and protocol. Evidence cannot be reused under a different identity.
- Selection freezes the exact materialized checkpoint digest and test-dataset fingerprint. The sealed evaluator verifies both again immediately before access.

## Search interpretation

The search is intentionally hierarchical. Uniform averaging and global interpolation are controls. Layer interventions estimate response-distribution-specific behavioral effects. Reliability shrinkage reduces noisy deviations, and coordinate descent tests interactions that independent single-layer scans cannot capture. This is not a claim that layers act independently or that the resulting coefficient vector is globally optimal.
