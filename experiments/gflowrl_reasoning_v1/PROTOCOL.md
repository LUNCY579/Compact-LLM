# GFlowRL reasoning posttraining v1

## Scope and independence

This is the Rineya reasoning objective experiment on branch
`research/gflowrl-reasoning-v1`. It is scientifically independent of the Soft Token branch
and does not import or combine Soft Token changes. The branch starts from freshly fetched
`origin/main` at `7d9b4fd450bc56405f6cec41459a7e83df7958ae`.

The question is whether reward-distribution-matching RL preserves multiple successful
reasoning modes in a compact Rineya model better than a matched reward-maximizing control,
without unacceptable pass@1, retention, or stability loss. A qualification harness is not an
efficacy result; no training run is authorized by this file.

## Historical basis and canonical harness

The historical canonical Rineya reasoning posttraining harness is
`experiments/reasoning_posttraining_v1`. The RRP-003 implementation anchor is
`d417e100ac10d225145f9701914854d10c55b43f`, whose files include the controller, engine,
instrument, arm, data contract, preservation, and resource guard layers. That historical
harness is inspected for lineage only; it is not merged wholesale because it is not present on
current `main` and its older launch scripts are not automatically valid.

The current branch uses a small standalone objective module and explicit contracts. The
qualification control R0 is a group-centered reward-maximizing sequence policy gradient. R1
is GFlowRL. A later integration with a native model trainer requires a separate runtime
qualification and exact source/checkpoint authorization.

## Primary source and paper-faithful equations

Primary source: Liu et al., *GFlowRL: Scaling Distribution-Matching RL to Large Language
Models*, arXiv:2607.13394v1 (15 July 2026), Sections 2–3 and Appendix B/D.

For prompt `x`, response `y`, reward `r(x,y)`, reference policy `pi_ref`, current policy
`pi_theta`, and rollout policy `pi_old`, the paper defines the reward-proportional target:

`p*(y|x) = exp(beta r(x,y)) / Z(x)` with
`Z(x) = sum_y exp(beta r(x,y))` (Eq. 2).

The reference-relative trajectory-balance form is

`(log Z_phi(x) + log(pi_theta(y|x)/pi_ref(y|x)) - beta r(x,y))^2` (Eq. 3).

GFlowRL replaces the learned partition function with the in-batch Monte Carlo estimate over
the fixed rollout group of size `G`:

`Z_t(x) = (1/G) sum_i [ beta r(x,y_i) + log pi_ref(y_i|x) - log pi_old(y_i|x) ]` (Eq. 4).

The estimate is a stop-gradient baseline. The length-normalized residual is

`Delta_i(theta) = sg[Z_t(x)] + (1/|y_i|) log(pi_theta(y_i|x)/pi_ref(y_i|x)) - beta r_i`
(Eq. 5).

The flow gap evaluated at the rollout policy is

`g_i = sg[Z_t(x)] + (1/|y_i|) log(pi_old(y_i|x)/pi_ref(y_i|x)) - beta r_i`
(Eq. 6).

The asymmetric flow-gap clip is

`g_tilde_i = clip(g_i, -epsilon_low, +epsilon_high)` (Eq. 7).

The GFlowRL loss is

`L = (1/G) sum_i w_i [ g_tilde_i + (1/|y_i|) log(pi_theta(y_i|x)/pi_old(y_i|x)) ]^2`
(Eq. 8), with sequence-level importance correction

`w_i = min(pi_theta(y_i|x)/pi_old(y_i|x), 1 + epsilon)`.

The implementation uses `importance_clip=2.0`, the paper's reported sequence-level IS
threshold, and computes the ratio as `exp(log pi_theta - log pi_old)` with an overflow-safe
log clamp. Frozen/reference quantities, rewards, lengths, masks, and `Z_t` do not receive
gradients. Invalid/masked rollouts are excluded from the group mean and group loss; an
all-invalid group becomes a zero-cost skipped update. Qualification fixes `G=4` and rejects
other group sizes rather than silently changing it. The paper's production table uses `G=16`;
that is a future registered condition, not a result of this gate.

## Reward transformation and group construction

Qualification uses the verifier reward as-is (`identity`): no centering, standardization,
clipping, or positive shift is applied before `beta*r`. This preserves the paper's reward
distribution target and makes zero/constant reward behavior testable. The paper's Table 9
reports overlong reward shaping enabled with penalty factor 1.0 but does not provide an
executable shaping equation in the paper text. Therefore overlong shaping is explicitly not
enabled for GFRL-QUAL-001. Any future use must bind its exact implementation before selection.

Each prompt contributes exactly four sampled responses from one declared rollout-policy
snapshot. Sequence log-probability sums are recorded for current, old, and frozen reference
policies; response length, EOS/truncation status, raw reward, transformed reward, validity,
and trajectory text/hash are recorded. Truncated responses are invalid by default and remain
in the raw receipt but not the loss. No gold answer or confirmation label is passed to the
policy prompt.

## Controls and preregistration

R0 and R1 match prompt order, rollout group, seed, response length handling, verifier, model,
tokenizer, precision, optimizer, runtime, and resource gates. Only the objective mechanism
changes. The initial qualification config is in `configs/initial_qualification.json`.

Initial pass criteria for a later matched run are: pass@1 no worse than R0 by more than 1
percentage point; at least one preregistered diversity/pass@k metric materially improves; no
runaway KL or gradient instability; no NaN/Inf optimization event; base retention remains in
the preregistered noninferiority range; and the diversity direction agrees across intended
seeds. These are prospective criteria, not evidence from the toy gate.

## GFRL-QUAL-001 gate

The gate is implementation-only and must pass before any full reasoning run. Its tests cover:

1. R0 is unchanged when GFlowRL is disabled.
2. Hand-computable Eq. 4–8 toy loss.
3. Partition estimator and masking.
4. Known importance ratios.
5. Exact asymmetric clipping.
6. Zero/constant reward.
7. All-invalid masking.
8. EOS/truncated termination contract.
9. Frozen/reference gradient isolation.
10. Policy/reference logprob convention.
11. Extreme-ratio numerical stability.
12. NaN/Inf guard.
13. Fixed-seed determinism.
14. Dataset split binding.
15. Sealed asset denial.
16. Bounded objective-only throughput/resource probe.

The gate receipt records exact test output and explicitly does not authorize model training.

## Evaluation contract

Later authorized results use `schemas/result.schema.json` and preserve exact per-example
rollouts. They report pass@1/pass@k, exact reward distribution, correct-solution diversity,
unique successful trajectories/strategies under a frozen metric, entropy/concentration,
reasoning length, base retention, paired repairs/damages, KL/policy drift, gradient norms,
instability events, resources, and all per-example outputs. Sealed/confirmation assets are
denied during selection and tuning; only a frozen arm may later be authorized for final
evaluation.

## Resource and run status

The resource probe is objective-only and does not load a model. The exact RTX 4070 8GB host
gate is not established by this qualification receipt. No full reasoning run, checkpoint
update, remote model load, or training occurred.
