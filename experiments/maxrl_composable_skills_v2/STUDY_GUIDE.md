# A controlled MaxRL + composable-skill study

This guide uses the delivered banks. It is not a pretrained launch command or a
replacement for any existing GEN2, INDEP, MAG or retention registration.

## 1. Select a genuinely acquired pair before testing a sum

Start with ONE pair whose skills each pass a native positive-control acquisition
check on independent development parents. R plus S is a useful candidate: reference
binding and constrained selection are different operations and their joint RS
signature is never trained in this corpus. That is an engineering recommendation,
not a measured optimal pair. Inspect source competence first; a source that already
solves everything or almost never produces a valid completion is a different study.

Use `manifests/atomic_R.json` and `atomic_S.json`, not all of train.jsonl. The latter
also contains bridge and answerability tasks. Both donors begin from the same exact
immutable anchor. A common optional warmstart must precede and be identical for BOTH
arms, or neither arm uses it. Warmstarting each donor on different data would change
the anchor and invalidate straightforward same-anchor task arithmetic.

Compare GRPO and MaxRL with the same optimizer-tuning allowance, trainable inventory,
precision, rollout group size and cost ceilings. The policy samples its own online
groups after it diverges. No reused static oracle or other arm's response pool counts
as on-policy training. Fixed groups bind one prompt, one behavior-policy snapshot,
unique rollout IDs and all generated-token costs. Group sizes such as 8/16 are
feasibility candidates, not established optimal budgets.

## 2. Preserve the two degrees of freedom in composition

For independent donors, define deltas against the SAME anchor:

    delta_R = theta_R - theta_0
    delta_S = theta_S - theta_0
    theta_RS = theta_0 + alpha_R * delta_R + alpha_S * delta_S

Choose each coefficient on development data with an equal tuning allowance, then
freeze it before TEST. Preserve theta_0 + alpha_R*delta_R and the corresponding
S-alone control at the EXACT SAME coefficients. Never normalize the sum and then
attribute the resulting weakening of R entirely to adding S. Report actual per-
tensor norms and the total norm/cross term, not only coefficients:

    ||alpha_R*dR + alpha_S*dS||^2
      = alpha_R^2||dR||^2 + alpha_S^2||dS||^2
        + 2*alpha_R*alpha_S*<dR,dS>

Do not substitute flattened-vector cosine for subspace overlap. An optional
orthogonal-projection arm changes a skill delta's direction and must re-pass its
individual acquisition check. Norm-matched random/noise additions and per-donor
attenuation controls help distinguish useful new skill acquisition from merely
adding a harmless ineffective delta. Keep projection, pruning, quantization and
bridge fine-tuning out of the first objective-only comparison.

Inspect per-skill scores AND the actual RS task. Retaining isolated R/S performance
does not establish that their learned functions interact successfully in one answer.
Bridge-trained improvement is adaptation, not spontaneous zero-shot composition.
The larger triple/RUSP banks support later expansion, not an automatic 15-arm launch.

## 3. Separate legacy retention from acquired-skill retention

Legacy retention starts from source-correct membership on guard banks. Acquired-R
retention starts from an R-trained model's correct membership on atomic R evaluation
rows, measured BEFORE further S training or composition. Enrolling only the original
source would miss new R examples that the source could not solve.

For sequential R->S, compare no replay with a declared replay budget. Replay R from
its atomic TRAIN manifest and old legacy tasks from replay_train; never replay guard,
DEV, TEST or CHALLENGE. Interleaved learning and explicit bridge training are separate
controls. Equal rows are not equal supervised/generated tokens; lengths and costs
must be measured with the actual tokenizer. Count prompt prefill, decoding, loss,
failed generations, verifier work, tuning time and memory, not only updates.

Binary mean-centered MaxRL can yield zero gradient on mastered replay prompts. A
separately weighted anchor-KL or supervised replay objective is a plausible retention
intervention but must be registered and matched across objective arms. Do not insert
partial-credit or preservation bonuses into the binary success denominator and
continue to call the resulting estimator the unchanged paper method.

## 4. Freeze actual baseline membership

Actual response JSONL must have exactly these fields per evaluated row:

```json
{"id":"MC2-<actual-row-hash>","prompt_hash":"<actual-prompt-hash>","response":"{\"answer\":[1,3]}","generated_tokens":9}
```

The example above documents shape only; it is not a usable response record.
Run metadata is a separate JSON object with full lowercase SHA-256 strings for
`model_sha256`, `tokenizer_sha256` and `decoding_sha256`. These must identify actual
assets/configuration. This evaluator validates binding consistency, not attestation
that an arbitrary hash proves native model execution. Keep raw generation receipts.

```bash
python -m experiments.maxrl_composable_skills_v2.behaviour enroll --data ./data --role guard_test --responses ./source_guard_responses.jsonl --metadata ./source_run.json --out ./source_enrollment.json
python -m experiments.maxrl_composable_skills_v2.behaviour compare --data ./data --enrollment ./source_enrollment.json --responses ./candidate_guard_responses.jsonl --metadata ./candidate_run.json --out ./guard_comparison.json
```

The source enrollment is established before looking at candidate outcomes. All rows
must be supplied exactly once; missing, extra or differently rendered prompts fail.
The bank and tokenizer/decoding bindings must match. TEST is exposed, not a sealed
instrument; final promotion needs a separately governed independent evaluation.

## 5. Endpoints and interpretation

Report held-out pass@1, per-family and per-signature behavior, both members of each
answerability pair, correct UNKNOWN, false abstention, and actual source-correct
loss counts and denominators. A net accuracy gain cannot cancel a forgotten old
answer. Zero source-correct membership is undefined. No-observed-loss is bounded by
sample size and operating point, not proof of no forgetting.

Pass@k measures whether the verifier finds any correct candidate among k samples;
it is not selected-answer accuracy unless the selection procedure actually runs.
Data calibration should report all-failure/mixed/all-success rates. Difficulty
labels here describe generation settings and do not supply those model probabilities.

The included comparator resamples whole parent clusters for macro-parent accuracy
change. Rowwise paired p-values are explicitly descriptive when sibling views are
correlated. Final per-skill effect thresholds, noninferiority margins, uncertainty,
multiplicity and run-level replication must be bound prospectively. Multiple child
rows, multiple decodes, and postprocessed variants are not independent replications.

## Current stopping boundary

Source/data mechanics are verified; native weights, tokenization, sampling/loss
parity, human review, final historical exclusions, resource contract and margins
remain open. No model quality, no-forgetting or orthogonality benefit is claimed.
