# RHF-001 experimental protocol

## Scope and immutable boundaries

Target: test a compact fact-memory mechanism before authorizing any production
architecture change. Main and post-training-Mimo remain untouched. One package,
no automatic merges, no hosted training or Codex tasks, no model uploads, no
checkpoint replacement. Do not reclassify synthetic or mock results as native.

## Stage 0: mathematical and engineering qualification

Run the entire isolated test suite. Required checks: primal/dual equivalence under
mean-loss ridge; visible difference in paper compatibility mode; full-rank versus
rank-deficient reconstruction; both projection objectives against explicit dense
oracles; strict tie/antipodal behavior; correct readout variant after edits;
reversible hooks and alpha=0; output shape and finite-value checks; hash/shape
artifact verification; explicit refusal to load a model without authorization.

Tests qualify these code paths only. They do not establish native quality,
statistical significance, full-repository compatibility, or GPU/Windows behavior.

## Stage 1: development, executed on CPU

Use `configs/development.json`: two seeds, N=16/32/64, width=32/64,
dimension=16, isotropic and correlated geometries, 100-step GD control.
Also run the smaller smoke and 400-step tagged SSFR reader on each seed.
Store raw metrics, source/config hashes, and environment. Preserve failed runs.

Primary measurements: strict clean recall and margins, then recall at relative
query noise 0.1 and 0.3. Secondary: changed-fact success, untouched-fact
repairs/damages, sequential old/new recall, geometry, and storage-rounding loss.
Controls: linear ridge, exact quadratic lookup, nearest-neighbor lookup, random
unfitted memory, shuffled mapping, and matched-architecture GD. Lookup tables
must include retained keys in their scalar counts. GD has a recorded training
budget; no runtime speedup claim without separately matched hardware/timing.

SSFR must report held-out-context A recall, swapped-B recall, stale-A-on-B recall,
and reader query error. A-only training accuracy is insufficient. SSFR's 128
contexts per seed arise from 16 facts, not 128 independent fact acquisitions.

Numerical failures remain in denominators and diagnostics. A subexperiment may
be blocked while another arm is measured; no NaN-to-zero substitution or silent
fallback objective is permitted. Sequential full-bank replay is explicitly not
an online no-rehearsal continual-learning algorithm.

## Stage 2: confirmation, configured but NOT executed

`configs/confirmation_reserved.json` reserves seeds 43/71 with broader
noise/anisotropy stress at N=64 and widths 32/64. Before running, commit the chosen
hypothesis, layer/width/ridge/objective, comparison, and any changed regularizer.
Keep the full grid; do not use the largest passing N from a nonmonotonic sweep as
an interpolated capacity estimate. These are public deterministic data, not a
cryptographically sealed test. No external preregistration is claimed.

Proposed advancement criteria (not measured pass claims): all numerical solves
required for the comparison qualify; strict clean recall >=0.95 on both new
seeds; noisy recall loss <=0.10 at epsilon=0.1; changed-fact success >=0.95;
untouched-fact damage <=0.02. These are screening thresholds, not universal
scientific cutoffs. Report failures rather than adjusting thresholds afterward.
A raw arm may legitimately fail; compare against preselected alternatives.

For real fact panels, aggregate paraphrases at fact ID before paired bootstrap.
Keep calibration, acquisition-check, retention, and semantic-overlap audits
separate. This package provides paired utilities, not an automated semantic
contamination detector. Never tune on acquisition-check or retention labels.

## Stage 3: native reconstruction and reversibility

Prerequisites: exact authorized local checkpoint; independent file-hash manifest;
known tokenizer/template; quiescent host; isolated compatible runtime; reviewed
prompt-only train/check records; explicitly chosen module/alpha before check.
Begin with one layer and one small memory. The example alpha=0.01 is a proposed
starting value, not a calibrated value. Never search alpha on check outputs.

The CLI reconstruction screen fits only train MLP input/output pairs. It requires
exact zero-alpha and restoration parity and no parameter/buffer content changes.
Both baseline and nonzero candidate undergo one-step cached/uncached parity.
It reports held-out MSE and full-vocabulary next-token KL/top-1 agreement.
Default screen gates: maximum per-row KL <=0.01 and agreement >=0.99, plus all
mechanical checks. FP32 cache tolerance is absolute/relative 1e-4; BF16 uses
absolute 0.03 and relative 0.01, explicitly recorded. Tolerances do not convert
an argmax disagreement into agreement. With a tiny check panel, this remains
only an engineering screen.

This is not continuous memory/time containment, multi-step generation testing,
a multilingual retention panel, or proof of unchanged instruction following.
The environment supplied for this delivery has no pretrained checkpoint,
Transformers package, or CUDA device; mock integration tests do not fulfill this
stage. `native_qualified` intentionally stays false even after a screen passes.

## Stage 4: real fact acquisition, requires a new reviewed experiment

Next implement/qualify desired native target-vector estimation and full-answer
scoring on novel facts with paraphrases, neighbors, counterfactual edits,
unknown/abstention probes, and broad instruction/reasoning/role-play retention.
A fixed candidate-value decoder is not a full language-generation benchmark.
Compare frozen baseline, unmodified-MLP reconstruction control, inserted-memory
candidate, shuffled targets, and a matched-budget established adaptation baseline.
Predeclare gains, allowable losses, fact-level uncertainty and resource costs.

Promote only after meaningful acquired-fact gains with no unacceptable regression
on independent retention checks, three or more independent acquisition seeds,
repeatable native receipts, long-generation/cache qualification, and explicit
human review. A failure in acquisition, forgetting, or memory cost is a valid
negative result. This branch does not merge or export a promoted Rineya model.
