# REQUEST ONLY: DEVELOPMENT causal replay diagnostic, draft 003

**Design draft only; no native implementation, qualification or dispatch.** A
separately reviewed causal prefix replay addresses original AUTHORITY section 9's
sequence-loss requirement without enabling the old full-gold entrypoint. Keep
original v1/GPU/CPU H3 guards, frozen source receipts and all seven CPU outcomes
unchanged. Root alone may launch a subsequently reviewed complete experiment.

## Question and decision value

Does a gold-independent feature from the original input-only trajectory predict
the local direction of complete canonical answer loss under one globally fixed
recurrent scale? The completed seven-prefix CPU diagnostic established adjoint/FD
mechanics but coherence collapsed to constant increase: 3/6 nonneutral prefixes,
3/4 scenario means, identical to constant increase. First-token targets also
confound accepted `K` versus canonical `k`, and correct `201` versus incorrect
`21`/`247` sharing their first digit. Preserve those results; do not retarget them.

Complete canonical replay remedies first-token truncation and measures a proper
autoregressive canonical-answer PREFIX likelihood. It does **not** exhaust scorer
equivalence: surrounding whitespace and YES/NO/UNKNOWN case variants can be
accepted while having different tokenizations. It remains an offline surrogate,
not semantic accuracy, preservation, on-policy generation or an adaptive-policy
counterfactual. Provisional bulk unit69/TAPS67/oracle70 and failed preservation
still justify no new controller fit. This diagnostic is worthwhile only as the
bounded missing metric/mechanism check; it grants no efficacy or retry authority.

## Frozen data, targets and causal construction

Use all 192 DEVELOPMENT cases/96 parents from
`runs/causal-replay-DEV-export-001`. Six categories retain equal weight, with two
variants per parent. Category6 evidence-transfer variants remain distinct
answerability interventions. No validation rows, outcome-based selection,
replacement, truncation, new nonce or new labels. The metadata packet binds exact
canonical strings, scorer/normalizer, original rendered input IDs and continuations.

For case i with original input IDs x_i and answer IDs y_i[0:m_i], target j uses
exact prefix `x_i + y_i[:j]`; only y_i[j] enters CE after logits. No current or
future target position enters that forward. Equal token values may legitimately
occur in the prompt or earlier answer; test sequence construction, not membership
of a token ID. Every target makes three entire fresh full-prefix rollouts at
alpha=1,1.01,.99 with reset moments, no KV/cache and no saved-state counterfactual.
No single full-future-gold forward or earlier-logit slicing is admitted.
Journal prefix kind, target index and preceding-gold count explicitly. Later
prefixes do contain prior gold: never reuse a blanket "no gold in input" claim.
The enforced firewall is no current/future target positions in the prefix.

The ordinary whole L12-L13 pass is unchanged; raw d2 is measured before the
experimental scale, native gates retained once. Exact unit returns the proposal,
then a detached cloned state leaf. Frozen FP32 CPU parameters stay grad=None.
The actual L14-L27 suffix/final normalization/tied custom UNKNOWN head supply loss
and `dot(G,d2)` summed over the full state. One unit state adjoint per target;
plus/minus rollouts compute values only. No optimizer, teacher pool or fitting.

Freeze predictor feature B_i from the original INPUT-only unit trajectory before
any gold continuation. The first answer-target forward can supply this without
an extra call. Only label-independent raw-update/moment quantities are predictors:
not first-token loss, gradients or gradient norms even though their input is x_i.
Later gold-conditioned features are diagnostic-only, excluded from predictor,
online-policy, A5 and fitting interfaces. Teacher histories differ from the
model's own generated histories; do not describe them as deployable adaptation.

## Endpoints fixed before outcomes

Primary per-case endpoint is mean canonical answer-token CE
`L_i(alpha)=sum_j CE_i,j(alpha)/m_i`; matching local derivative is
`D_i=mean_j dot(G_i,j,d2_i,j)`. Canonical sequence sum/derivative, which relates
to literal answer-prefix probability, is separately descriptive. Equal-case means are
equivalent to averaging variants within parent, then the 16 parents per category,
then six categories. Do not pool 440 token targets as independent examples.

Per-target adjoint/FD gate remains `abs(a-FD)<=.001+.03*abs(a)`. Also compare the
mean derivative against `(L_i(1.01)-L_i(.99))/.02` using that same fixed form at
the mean level; cancellation is not a reason to widen tolerance. Any required
discrepancy/nonfinite/resource/restoration/source failure makes the complete
comparison incomplete, with primary mechanism null and retained raw failures.
Direction is neutral when `abs(D_i)<=.001`; numerical sign disagreement with the
mean finite difference is explicitly uncertain. No outcome-dependent threshold.

Compare input-only coherence direction with both prespecified constant increase
and decrease. Report every case and parent/variant harm, neutral counts, category
and answerability strata, exploratory tied-rank Spearman(B_i,-D_i), and paired
within-world consistency. Shared authored generators and exposed DEV limit
prediction/uncertainty. No fitted coefficients, favorable family selection,
validation peeking, confirmatory significance or endpoint swapping.

EOS is secondary and separate from answer loss. At `x_i + complete_answer_ids`,
the fixed accepted stop set is {151643,151645}. Use
`L_EOS=logsumexp(all logits)-logsumexp(logits[accepted EOS IDs])`, one unit state
adjoint and three recomputed rollouts. This is -log probability of either accepted
stop after the canonical answer. Canonical EOT151645 CE can be a descriptive
scalar from the same logits, with no additional adjoint/call. Never replace the
answer endpoint or hide EOS harm in a composite. EOS after supplied correct text
does not establish actual generation termination or complete-answer correctness.

## Exact counts, shapes and prospective qualification

Metadata-only packet `CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json` binds 440
answer targets: lengths {1:80,2:2,3:84,4:26}. Category target counts are
108/32/94/78/64/64. Add 192 separate EOS prefixes: 632 task target positions,
1,896 task full forwards and 632 task state adjoints. Per alpha, answer replay
processes57,596 prefix positions and EOS25,048; three scales process247,932 task
positions. Base input maximum232, last answer prefix235 and EOS prefix236 all
fit the unchanged max256 contract.
Zero prompt/answer joint-token boundary differences were found; the frozen
convention remains original IDs plus separately bound answer IDs regardless.

Choose the prospective **29 qualification API/3-adjoint** variant. Keep the old
CPU qualification structure (26 APIs/two adjoints), then add one maximum256
accepted-EOS-mass loss FD triplet/adjoint. This new triplet replaces the metadata
feasibility draft's proposed longest-late scalar-target triplet purpose; counts
are unchanged. Teacher-prefix concatenation, past-only target firewall and late
positions require fixture and packet checks. Native qualification still requires
exact noextra/unit/fixed-scale values, scalar CE and new EOS-mass gradient/FD,
finite/RMS/state/parameter/head/assets, masks/positions/L0-L2, cache refusal and
normal/forced-exception restoration. No old native PASS transfers automatically.
Keep the original approved84-token representative and nongold maximum256
extension for qualification:26 completed qualification forwards process4764
positions. The complete planned matrix processes252696 completed positions;
intentional partial work remains null.

Complete total: **1,925 API attempts, 1,923 full-forward attempts, 1,922 expected
complete, one intentional partial, two precompute cache rejections and 635 state
adjoints**. Each target/case/EOS position stays in the required matrix. Counts,
actual positions and partial-null work must be independently checked from raw
journals; no survivor primary comparison. Freeze target-count/prefix packet,
aggregation/feature/neutrality/EOS conventions and statistical plan before native.

## Cost and unchanged safety contract: prospective deadline only

Corrected run002 passed fresh native mechanics but stopped before every task row:
632*(1.7388156999986677+2*.9941931000030308)*1.5=3533.3874012044835 seconds
remaining. Add64.91039009999804 elapsed and180 internal reserve:3778.2977913044815,
exceeding its authorized3600 ceiling by178.2977913044815. Preserve that refusal,
qualification records and empty task outputs. No endpoint or scientific conclusion
changes because no task result was measured.

REQUEST ONLY: propose parent harddeadline4500, body4320 (4500 minus unchanged180
internal reserve), progress950 and global evidence reserve1800. Total ceiling
reservation6300 fits remaining20151.485. Margin over the observed complete-worker
forecast is721.7022086955185 seconds, about19.1percent of that forecast. This is a
prospective allocation, not a measured bound or permission to extend run002.
Factor1.5, all192 cases/632 targets/1925APIs/635adjoints/max256, FD tolerances,
feature/endpoint/statistical rules and every memory/disk/source guard stay fixed.

ParentRAM8.4GiB/postimport-preweight7.4GiB/RSS6GiB/runninghost1GiB; CPU4threads,
FP32/eager and exact canonical runtime/head/state/source remain unchanged. Keep
soleworker lock, suspended-before-Job containment and owned-descendant cleanup.
Do not stop other workloads, lower guards or omit cases/targets/EOS.

The one corrected3600 attempt's grant SHA5397a0b51cdc62891c00c0f6a2147f5e53fafe5e277759b71d28ce3b93948063
is CONSUMED and explicitly granted no further retry. Another directhuman permission
is required for the exact4500 successor before launching any worker. Prepare new
reviewed source/admission/granttemplate, parent+worker preimport/premodel checks,
anti-replay identity and a separate exclusive run. A template/request/source PASS
is not the actual allowance. Do not reuse or extend permit5397.

Version only deadline/body/authority-proof handling. Preserve frozen3600 source,
tests, plan and002 design. The analyzer successor admits exactly reviewed3600 with
design002 or newly approved4500 with design003 and matching source/runtime/grant
proof; arbitrary deadlines are refused. No endpoint, weight or uncertainty edit.
Fresh qualification/resource/source and complete632-target CPU forecast remain
mandatory: measured setup+remaining+180 must fit the new ceiling. Refuse incomplete
if it does not; no automatic retry or model work by this design author.

## Next implementation and stopping rule

Next turn may prepare new `cpu_causal_replay_native.py`, separate entry/supervisor,
admission generator, causal-prefix/EOS/aggregation fixtures and raw analyzer. Reuse
unchanged `DiagnosticBlock` through an explicit new diagnostic loss path; never
enable/monkeypatch the old full-gold guard. Source/native loss, data, EOS set,
CPU forecast and exact final admission need independent resulting review. No
worker is ready or launched by this draft; root owns future dispatch.

If initial coherence again adds no information beyond constants, or any relation
does not survive the declared paired/stratum checks, stop this scale-proxy family
with its restricted evidence and unresolved semantic limits. A positive canonical
prefix-likelihood association is only a mechanism lead: meaningful task headroom and
preservation must still justify a separately registered controller experiment.
This work cannot fill missing bulk validation, solve the original GPU fault,
confirm an independent source benefit or establish deployable quality/time.

## Source bindings

- DEV bank:566e6d6e64f258ff4deaa28a9909841a451fb96cf939ca44533726c59b2a7ff0
- DEV token binding:6529b48b26264123fafc9f81d67cb6cc5a957fe30a0ab4853440790c56bfdd53
- DEV export manifest:278242ad25488a5f7896957aab4e9269dfdf96babe21abf64bc8167838514302
- Metadata feasibility JSON:021ea720f4b644d0e0fff0b975ab460d80e6c02cb401400fd0be58a0ba4cb359
- Metadata helper:7071f731c503e5bc7af01c4e6983ee75b46cd3bfed5672a4d13849601cbc2c6b
- Completed CPU execution math JSON:86305578b651574cad9002932f1b7fe5c751d4b7fc2dbdf7a45e44a324bfe549

No model/tokenizer/optimizer was loaded by this design author. No validation,
protected/native TEST or native source was read or changed for this draft.

## Clarifications frozen in draft 002

Draft001 (SHA739b4ae134aa6e5c88edd17cfc61f44ce0cd77d0cada95d72f790f6d171ba855)
remains unchanged. Counts, targets, resource gates, tolerance and primary endpoint
are unchanged here. Summed answer CE is -log probability of the canonical answer
PREFIX. An incorrect longer response can share that entire prefix (e.g. canonical
201 followed by an extra digit); answer likelihood alone does not discriminate
it. EOS remains separately reported, with no primary composite or endpoint switch.
Spelling, whitespace/case aliases and on-policy behavior remain additional limits.

Reliable directional comparisons use every nonneutral mean adjoint except cases
whose mean adjoint and mean FD have different signs, including an exact-zero FD.
Those are NUMERICALLY_UNCERTAIN, never discarded: report all192 means, neutral,
nonneutral, uncertain and reliable denominator counts explicitly. Neutrality is
still abs(mean_adjoint)<=.001 and the mean FD gate is still .001+.03*abs(mean_adj).
This uncertainty flag does not excuse a failed gate or define a favorable subset.

The phrase 'survive strata' means descriptive paired/category/answerability checks,
not an automated advancement rule. Shared generators and exposed DEV support no
confirmatory population inference, fitting or controller advancement from this
proxy. Positive prefix-likelihood association cannot replace meaningful task
headroom, complete preserved generation evidence or the missing bulk validation.

## Version003 disposition

Draft002 SHA f8b962447f403105a564ec071bbfa89fd3e40843fcfef6800b7ec72a22412492
remains intact. Its scientific population/operator/targets/mean-answer-prefix
endpoint/EOS-secondary/neutrality/uncertainty/shared-parent interpretation are
retained. Only prospective deadline/body and permission/budget-proof handling
change. No active native job, actual new grant or task result is created here.
