# RGI-001 experimental protocol

## Status and scope

This is a prospective implementation and protocol template, not a preregistered scientific
result. Native execution has not occurred in this delivery. Configuration examples are drafts.
No existing confirmation bank is newly sealed by being copied into this directory.

The program asks whether per-example functional changes reveal useful repairs and interference
that aggregate losses hide. It does not assert that RGI must hold during Rineya post-training.
Uniform improvement can cross a decision boundary; selective change can be harmful. Existing
host-model agreement can dominate correlations even when the adapters differ meaningfully.

## Phase A: discover with existing artifacts

Use already-exposed development evidence to compare Supa with Titan-Sparse, a registered adapter
off with on, and compatible composition arms. Respect the exact host, adapter and precision
specified by the originating program. A Supa-vs-Titan comparison may have different hosts;
a four-arm fixed-delta composition comparison may not.

Create a canonical bank index before collecting or importing losses. Each index binds the
complete context and observed next token. Use independent parent problems as the sampling
unit. Counterfactual variants, horizons, plans and goals derived from one parent stay in the
same cluster and split. Record semantic payload identities independently of prompt wording,
and assign all variants of a semantic lineage to one split.

Measure global offset, unit-slope residual RMS, centered CCC, loss-difference discrepancy,
content/format/mixed profiles, whole-answer averages and operation-specific signed changes.
Use a fixed family-parent-token weighting scheme. Always preserve the absolute loss change
beside the centered residual. Include null models with constant shifts, scale-only changes,
format-only changes, rare content errors and deliberately mismatched contexts.

For world-model diagnostics, distinguish applicable actions from invalid/no-op actions,
copying from state changes, arithmetic from conditional rules, and first-step accuracy from
self-rollout performance. Compute loss profiles on shared reference states; evaluate self-rollout
on each model's own predicted states separately. No cross-model alignment is claimed for
those differing generated contexts.

Historical datasets with only generated text or first-token log probabilities cannot support
all-token RGI. Use the validated importer only where complete identified token losses exist;
otherwise collect them anew on development data. Never invent NLL from accuracy or an aggregate
benchmark. Preserve source bytes and a producer receipt in the conversion record.

Exploratory hypotheses include: rule-coverage training changes conditional content more than
format; a copy shortcut can improve easy/no-op cases while harming applicable transitions;
increasing a composition coefficient can selectively damage protected long-context tasks.
These are falsifiable proposals, not predictions encoded as success conditions.

## Phase B: targeted data versus a matched control

Build a training pool with explicit target-rule labels and a separately frozen development
bank. Choose the coverage policy using training/development only. Exact matching uses family,
frame, horizon, sequence length and number of supervised tokens. Do not use loss-based
selection from confirmation. Preserve unique payloads and record the realized coverage gap.

The control estimand is conditional on the matching buckets. Length can correlate with a rule;
therefore exact matching may reduce or eliminate the coverage contrast. The selector refuses
an inadequate contrast. A refusal is informative, not permission to relax the budget silently.
Revise the design on training data and register the revision before evaluation.

Use the same frozen host and tokenizer, adapter target modules, rank, alpha, dropout, clipping,
optimizer family, updates and accumulation in both arms. Match both processed and supervised
tokens, not merely example counts or nominal steps. `control-plan-matrix` generates paired
plans for initialization seeds 17, 23 and 29. The data selection remains fixed in this first
matrix: these are training-initialization replications, not independent corpus draws.

The default 64-pair pilot is deliberately bounded and is not a reconstruction of historical
RCWM training. Examine its development behavior before authorizing a larger dose. Changed
training budgets or a different optimizer require new plans. The current native training
backend supports AdamW. Other optimizers may be compared through independently qualified
producers and the explicit evidence importer; no Muon-on-LoRA assumption is made.

Evaluate targeted versus control on intended rules and on held-out rule combinations, then on
protected tasks. A relative-loss signature alone cannot establish that targeting caused a skill.
The matched behavioral contrast is essential. Separate the claim about this fixed training
recipe from claims about a distribution of training seeds or a different generator.

## Phase C: functional composition

Register off, A, B and A+B against the same exact host. The components in A and B must be
separately identified; A+B must contain exactly the same components at the same coefficients.
The analyzer refuses a different host or an implicit coefficient change. The native collector
loads compatible LoRA artifacts unmerged and concatenates factors for weighted evaluation.
It never merges an adapter into the Titan-Sparse base.

Compute the interaction `L_AB - L_A - L_B + L_base`, its absolute RMS, signed slices and
parent-sampling uncertainty. Report centered functional-effect cosine only for nondegenerate
effects. This measures behavior through the host, not Euclidean weight orthogonality.

Nonlinear interaction need not be harmful. Test both component capabilities, tasks genuinely
requiring their joint use, and independent protected endpoints. A failed component gate must
not be hidden by better pooled performance. Keep sequential A-then-B training distinct from
adding fixed updates; fixed sums commute, training trajectories need not.

This package analyzes collected composition evidence; it does not assert that its independently
authored mini-worlds certify unrestricted skill stacking or replace established Rineya suites.

## Phase D: fresh confirmation

Before collecting predictions, approve bounded claim wording and explicit endpoints, margins,
coverage counts, seeds and bootstrap settings. Freeze the full training pool, development bank,
fresh confirmation bank, reference/candidate model specs and any separate behavioral bank.
Model files must match their registered fingerprints. Check semantic payload and lineage
separation, including the behavioral bank when distinct from the NLL bank.

Use `collect-frozen` for one registered token-collection attempt. Use the existing, separately
qualified RCWM/BWM evaluation runner for rollout/planning/protected behavior, exporting the
bridge schema with the same freeze identity. Then run `confirm` once. The collection marker and
analysis marker are consumed before their respective outputs are examined. A failed attempt
leaves a failure record. Recovery requires a visible amendment or new preregistration; do not
delete read markers or silently reuse a known bank as fresh.

The reference/candidate artifacts must name the frozen specs and bank. Behavioral records must
name the same model, frozen behavioral bank and freeze, identify registered items, contain all
expected endpoint rows and count invalid parses as failures. Source-file hashes are retained
in the result. Incomplete behavior produces no confirmation.

## Statistical interpretation and stopping rules

The bootstrap resamples parents within family and keeps all dependent observations of each
sampled parent together. Families receive equal weight. Its intervals describe evaluation
sampling conditional on the fixed trained artifacts. They do not quantify training-seed
uncertainty. Report each training-seed contrast separately; do not pool repeated observations
of the same parent as independent model replications.

Confirmation behavioral intervals use Bonferroni-adjusted two-sided percentile bounds across
registered endpoints. Superiority requires the lower improvement bound to exceed the registered
margin. Noninferiority requires the lower bound to exceed minus the allowed degradation.
A clearly inferior bound fails; an overlapping bound remains inconclusive. Fewer than eight
parents overall, or fewer than two per included family, is explicitly underpowered. These
minimum software checks are not a power calculation: select actual sample sizes on development
variance and a meaningful minimum effect before freezing.

Constant parent-level differences are not licensed by a zero-width bootstrap interval. The
behavioral gate returns `INCONCLUSIVE_DEGENERATE`. Preregister a suitable exact or bounded
procedure externally rather than interpreting zero observed losses as proof of zero population
risk. Percentile bootstrap is approximate; sparse rare failures and boundary-valued endpoints
need particular scrutiny. The supplied example margins and sample counts are not validated
power recommendations.

Stop expansion when matched data targeting does not improve the registered rule behavior, when
its gains disappear under counterfactual tests, when protected behavior fails, or when provenance
cannot be established. A negative RGI result can still guide research; a positive correlation
cannot rescue a failed behavior gate. A fixed-artifact confirmation remains bounded to its
registered tasks, frames, horizons, host and data regime. No report promotes a core or permits
release automatically.

## Intentionally separate questions

Loss shifts are scalars; hidden-state offsets are vectors. This package does not identify a
causal circuit from either. Activation injection/ablation, recurrence, model surgery, broader
optimizer research and new planner training remain separate interventions requiring their own
qualification. All changes in this branch are restricted to this experiment directory.
