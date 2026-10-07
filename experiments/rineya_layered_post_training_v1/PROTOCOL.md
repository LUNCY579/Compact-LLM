# Preregistered interpretation — RLP-001

## Hypothesis and attribution

Can cyclic RL, short verified-trace SFT and weighted checkpoint interpolation improve
new capability acquisition while preserving earlier behavior better than conventional
post-training on a local 0.6B model? The Limite diagram motivates sequencing; it does
not establish a theorem of monotonic acquisition or a solution to forgetting.

Compare A/control, B/soup_only, C/layered and D/full, then the explicit no-critic/no-soup
and replay-weight-zero ablations. The primary early contrast is A versus B versus C.
D adds both extra staging and a learned baseline; do not credit one component without
its ablation. Use the same base files, tokenizer, trainable dense projections, data
splits, generation limits, verifier and evaluation seeds. Report every training seed.

Actor steps are allocated equally in total across arms, with an even stage allocation
and deterministic remainder. Training order is seeded and suite-balanced. Different
stage sequences still change data exposure and rollout cost. This is intentional and
reported, not magically compute-matched. Match both usable training exposure and
measured compute in the decisive follow-up. Include generation, all forward/reference
passes, failed attempts, critic updates and coefficient selection. Token-forward
counts are operational proxies, not backprop FLOPs or wall-clock equivalence.

## Selection versus confirmation

For development selection, maximize acquisition among checkpoints satisfying a separate
at-most-one-percentage-point loss floor for EACH required protected suite. Tie-break by
protected-score sum then checkpoint name for reproducibility. Include the untouched
base. Selection compares endpoints and every preregistered coefficient candidate.
An alpha=0 or baseline fallback is a null result, not proof of skill composition.

Every stage and soup candidate is retained in the selection archive. Full arms have
more selection opportunities, so both the raw TERMINAL checkpoint and selected
checkpoint are reported. A decisive study should additionally match search budgets or
use a fixed terminal-stage contrast. The current compare command reports cost mismatch;
it does not solve this attribution problem on behalf of the experimenter.

The immutable lock binds config, model identity, selected weights and confirmation data.
Only the separate confirmation command opens held-out inference. Its paired bootstrap
resamples independent task families, not individual repeated completions. Defaults:
acquisition point gain >=5 points, positive acquisition lower 95% bound, and protected
suite lower 95% bounds >=-1 point. `candidate_pass` is explicitly single-seed, not a
claim of cumulative general intelligence.

At least three independent training seeds and powered held-out cohorts are required for
an efficacy claim. The 128-group default is a screening floor, not sufficient power for
every one-point noninferiority claim. Bootstrap intervals can be degenerate with zero
discordant outcomes; use an independently specified exact/conservative damage-rate
analysis or a larger cohort before interpreting that as a population guarantee. Do not
pool seed/example observations as independent. Register multiplicity correction for
claims spanning all protected suites. The v1 output does not implement a multi-seed
hierarchical inference or multiple-testing correction.

## Meaningful positives and negatives

Strong positive: acquisition gain survives confirmation and matched-compute controls,
with retention in every protected suite and repeated across seeds. More modest positive:
consolidation+soup recovers much of acquisition at substantially less forgetting than
SFT alone, at a reported cost. Also report repaired versus damaged examples in external
analysis; macro score alone can hide who changed.

Soup complementarity should be judged against both parent checkpoints on the full
acquisition/retention vector. Merely moving along their tradeoff is interpolation, not
evidence that the soup invented a missing skill. Do not require an impossible guarantee
that every soup dominate both parents, and do not use a post-hoc scalar to assert it did.

Null/negative: endpoint or base repeatedly wins; critic audit is at chance; the full
terminal checkpoint regresses; gains disappear under exposure/compute matching; native
likelihood parity or memory fails; any confirmation contamination occurs. Preserve these
outcomes, including blocked native runs. There is no native success claim in this branch.

## Critic-specific boundary

Train half-MSE on terminal binary correctness over valid preceding-action states; the
head sees detached actor features. Raw estimates may be outside [0,1]. Audit before RL3
using independent selection prompts, AUROC and MSE against a constant predictor. The
default audit uses the state before EOS, which has seen the candidate answer, and does
NOT establish that early reasoning progress is reliably forecast. Long-trace prefix-
position audits, online calibration drift, and learned process rewards remain follow-up
experiments. The fixture bypass is solely for exercising code, never native permission.

## Deferred, not silently emulated

Exact Limite hyperparameters/data, 65K/131K traces, a separate trained critic backbone,
full-weight native training, live teacher generation, custom-architecture Rineya loading,
length-changing curricula, rank-changing LoRA training, vector/reverse-world rewards,
free-form roleplay judging, full benchmark download adapters, and optimizer resume are
outside v1. The implemented experiment is the controlled cyclic post-training core.
