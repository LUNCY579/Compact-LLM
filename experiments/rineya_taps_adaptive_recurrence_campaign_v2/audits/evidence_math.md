# Evidence and mathematical audit

Audit owner: evidence_math. Audited inherited HEAD: 7ece5cead285ba65c7d73ab57c0327fae2dc40bb. Integrated base: 03db218ceebc61a2bc04f2482cf7e094045e01c1. This audit loaded no model, ran no native worker, installed nothing, parsed no dataset files, and changed only its two assigned campaign artifacts. Native worker charge: 0 seconds.

The next informative experiment is the complete, freshly qualified reference scale matrix on the exact ARMR source at zero/one extra L12–L13 application. First distinguish useful constant scaling from adaptive headroom. At one extra application there are only two within-prefix observations, so a saturated nearly constant policy is a plausible failure mode. This is a derived hypothesis, not an observed Rineya outcome.

PRIOR_FINDINGS.json has 24 separate actionable entries. Every local source is bound to an immutable commit/path, Git blob, SHA256 of Git blob bytes, SHA256 of actual worktree bytes, and equality check. No referenced local file was missing and all mapped inherited worktree source bytes matched their blobs. Both separately referenced commits were locally reachable. Cross-repository claims remain attributed to the local correction audit and its exact remote commit/path pointers; their original native raw receipts were not independently opened or recomputed here.

## Evidence lineage that changes the experiment

- ARMR source weight SHA256 is a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026. It is the FP16-stored custom Supa-A1.1 checkpoint evaluated after FP32 upcast. The historical CPU run pins Python3.11.15, Torch2.12.1+cu126, Transformers5.14.1 and Qwen implementation fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab. Loaded state before/after is b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551. These receipts do not qualify this host.
- Historical ARMR fixed_cpu_budget_v2 ran at af7c49d432a652c55cbafc5155c378d87497cf24. The preregistration SHA is 7089d1c047eee70bda93b020bd1e291314ae53c2d2beebc9f0daeeff00a0d838; data SHA b75df9368e27cfc7de58e55370090e55bd5bcb4d1bd86aeb079b486555925da0. A0 and L12 unit recurrence each score12/16; L12–L13 supplies13/16 with1 gain/0 loss. Deadline90→900 seconds resolved censoring in a previously inspected example; it did not create independent evidence. Three-total-application qualification remains closed.
- Ordinary ungated Supa in MQ/RRP is weight962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b. Opallo is e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b. Neither can be silently substituted for ARMR a99. Stateless MLP, gates, dense-tail adaptation, readout bias and R2 results have different source/operator/runtime/scorer contracts.
- RRP CL-040 corrects the v2 prompt-copy floor from.2870 to.4391 on the evaluated230-row subset; source remains4/230. Format repair changes validity139→205 and correctness4→4. The strongest honest answer-blind floor must accompany each bank's score.
- The controlling RRP005 context/gold correction invalidates the old768-row preservation bank as a retention gate:192 procedural inputs omitted rules/evidence and31 conversation histories were omitted. Corrected main has757 rows after11 inconsistent procedural golds are excluded. Gold self-scoring checks the scorer, not independently established question answerability. CORRECTION.json SHA b3ef073a3a41df9d1b89c198b7d62b9cd3502c8cd863c7c8e023dc573e40c3f8.
- RRP CL-053's semantic conclusion is superseded. The old recognizer has zero canonical-phrase pair sensitivity, but RRP005 finds9/29 natural abstentions with a correct answer on the corresponding positive half. This is an agent review of exposed DEV outputs, not independent human review. Registered reference scores remain unchanged. SEMANTIC_ABSTENTION_CORRECTION.json SHA fd519cadfd82f88bc9dadc396bf8d17862035b19963ba31e82d550c74a3535a2.
- RRP005 A3 later distinguishes545 unchanged completed native requests from308 changed/new requests;301/301 fresh overlaps match token-for-token. Changed prompts inherit no outputs. Historical reused requests and fresh acquisitions cannot be represented as one newly executed union run or as latency-equivalent. REUSE_AUDIT.json SHA3437a471bf7ce4adb86db1b686f74fa1204f1fc9cfce3263d9d0b613d257a83b.
- Supa audit's corrected FLOOR1 composition improvement coexists with retrieval111→8/120 and abstention15→0/120. Aggregate proxy can favor always-abstain. Matched stateless MLP can beat gate-only changes. Historical carry/memory comparisons include negative promotion and inconclusive contrasts. Gate amplitude can make a manipulation practically absent.
- LoopSpec is exact speculative-scheduling apparatus with an unqualified native recurrent backend. CDE at7a7f7b60e38c18dbeda7e182daf037e993ca8afd is offline depth/width apparatus, not native performance evidence. CED paired-host at627253edde9a0e67334034d252e5d9dd9daead9a has15 host/90 companion fixture tests and zero native feature captures. Dream-RSI admits zero real histories and its qualification is PASS_SYNTHETIC_MECHANICS_ONLY. None supplies permission or qualification for a deeper/cached/TAPS native host.

## Operator and observation indexing

Code identity: scheduler.py SHA eaaa5cd8cab8b390dc932f850f7002dcfc81177c07bef50bcc368bf64ac9d546; integration.py SHA57445e67903309c903655f1de9537658ccf6f8fe83fbc451d2f5062ce61d4b4a, both at inherited HEAD.

Let h0 enter the ordinary two-layer block and let F preserve all allowed layer conditioning u. The ordinary pass yields h1=F(h0;u), with raw observation d1=h1-h0. Observation t=1 seeds EMA under force_unit and preserves the actual normal proposal. One extra block proposal is p2=F(h1;u), raw observation d2=p2-h1; controlled output h2=h1+alpha2*d2. The remaining native layers, final normalization and custom readout operate on h2.

The implementation measures the first observation from the first selected layer's entry to the last selected layer's normal output. It does not seed from the last layer alone. It creates a new scheduler at every complete model.forward, including each generated-token prefix. Controller variation across prefixes is possible, but there is no cross-prefix carry and no multi-step adaptive trajectory within one prefix at this budget.

Disabled scheduler means unit recurrence; disabled host means zero extra applications. Alpha1 returns the actual proposal, bypassing cancellation arithmetic. Native cache objects, padding, training, gradients, compiled/checkpointed hosts and extra_loops>1 are rejected. Scaled arms still require native qualification.

## Moment identity — independently derived from the source recurrence

**Identity in exact arithmetic.** For t observations, nonnegative normalized weights are

    w[t,j]=(1-beta)*beta**(t-j)/(1-beta**t),  j=1,...,t.

They sum to1. With a fixed valid-position mask and denominator N equal to included positions times hidden channels,

    M_t=sum_j w[t,j]*d_j
    V_t=sum_j w[t,j]*mean_mask(d_j**2)
    P_t=mean_mask(M_t**2)
    V_t-P_t=sum_j w[t,j]*mean_mask((d_j-M_t)**2) >= 0.

The subtraction identity follows by expanding the square and using sum(w)=1. The float implementation clamps only negative cancellation error to zero. A static mask is part of trajectory identity; changing mask/shape/device/sequence ID requires reset. Reduction is per sequence, never over batch. Invalid nonfinite delta values are rejected before masking, so masked NaN/Inf are not silently admitted.

**Identity/bounds.** With gamma>0, eps>0 and P,S>=0, |B|<=1 and unclipped alpha lies in[1-rho,1+rho]. Actual alpha lies in[alpha_min,alpha_max]. Warmup observes raw d and uses unit alpha. The default first ordinary pass force_unit guarantees intact first-pass output even for fixed/random modes. All-zero updates imply P=S=B=0 and alpha=1. Tiny updates approach neutrality as their energy becomes small relative to eps. Multiplying a zero actual gated update leaves it zero.

Moments receive the newly proposed RAW update. Feeding the already scaled movement instead would change the estimator and its counterfactual meaning. Every arm must recompute F on its own state; a saved baseline state multiplied afterward is not an alternative model trajectory.

## Two-observation limit and clipping

**Identity at t=2.** For a common normalization denominator,

    M2=(beta*d1+d2)/(1+beta)
    P2=mean((beta*d1+d2)**2)/(1+beta)**2
    S2=beta*mean((d2-d1)**2)/(1+beta)**2.

Therefore P2/S2 measures a weighted two-update alignment/amplitude relation, not signed task progress.

**Identity with default beta=.8,gamma=1.5,rho=.275,bounds[.8,1.2].** Upper saturation alpha=1.2 occurs exactly when

    B >= 8/11
    P >= 9.5*S + (8/3)*eps.

Lower saturation alpha=.8 occurs exactly when

    B <= -8/11
    S >= (38/9)*P + (16/9)*eps.

At energies large compared with eps and d2=c*d1, the ratio is (beta+c)^2/[beta*(c-1)^2]. For nonnegative c, upper saturation spans approximately .520870<=c<=2.024584. Thus aligned raw proposals of broadly comparable size cause an exact fixed1.2 result. This is wide enough to make saturation a useful diagnostic; actual Rineya saturation has not been measured.

**Reproduced arithmetic fixture, not native measurement.** Plain JavaScript binary64 checks stored in PRIOR_FINDINGS.json give:

| Raw updates | P/S interpretation | alpha2 |
|---|---|---:|
| equal nonzero d1,d2 | S=0 |1.2|
| d2=-d1 at beta.8 | P/S=.0125 |.8|
| d1=0,d2 nonzero | P/S=1/beta=1.25 |.975|
| d1 nonzero,d2=0 | P/S=beta=.8 |.91630435|
| both zero | B=0 |1|
| equal components1e-10,2e-10 | eps dominates |1.000000006875|

For the first four nonzero-scale values, eps corrections are negligible rather than mathematically zero. Independent centered-square and subtraction variance expressions agree to at most1.12e-16 in these checks. Positive collinearity saturation interval uses the eps-negligible approximation; exact condition includes eps above.

**Conditional inference.** Two observations contain information, but not enough to establish reliable task-sensitive adaptation by themselves. If nearly all prefixes saturate1.2, A3 will be numerically close to fixed1.2 and the frozen A5 pool may be nearly degenerate. Conversely, substantial alpha variation could still track harmful fluctuations or coherent wrong updates. Distinguish observed variance, useful finite-grid headroom and causal predictive alignment.

## Task sensitivity and stability

**Chain-rule identity, externally specified schedule.** For h[k+1]=h[k]+alpha[k]*(F(h[k];u)-h[k]), define d[k]=F(h[k];u)-h[k]. With differentiable F and terminal loss containing the actual suffix/readout, downstream adjoint G[k+1] includes all subsequent states. Then

    partial L / partial alpha[k] = inner(G[k+1],d[k]).
    D[k]=(1-alpha[k])*I+alpha[k]*J_F(h[k];u).

The identity differentiates a coordinate of an external schedule. It does not differentiate through the online scheduler while pretending future alphas are constants.

**Reproduced finite-difference fixture.** Use h0=(2,-1), F(h)=diag(-.6,.85)*h+(1,-.2), two prescribed alphas(1,1.1), target(.5,.25), L=.5*||h2-target||². Analytic sensitivities(1.3135708125,1.049876875) agree with central differences at step1e-6 to maximum absolute error1.033e-10. The first derivative includes the second step's Jacobian. These are plain arithmetic fixture checks with no Torch/model execution.

**Conditional local stability.** For a fixed-alpha linearization, spectral radius(D)<1 is a local contraction criterion; a global convergence claim needs additional uniform/domain assumptions. For state-dependent alpha, derivative additionally contains d times grad(alpha)^T. EMA mean/energy/previous-update state must be included in the augmented Jacobian. Clipping is piecewise differentiable; its saturation derivative vanishes almost everywhere with respect to balance. A fixed-alpha Jacobian alone cannot establish adaptive stability or correctness.

**Counterexamples.** Coherent updates toward a wrong target have high P yet harmful loss direction. Alternating updates can make useful corrections under the actual readout. Near-zero actual gated updates cannot create computation by scale. A stale EMA under changing conditioning can become irrelevant. A small hidden movement can change answer ranking, while a large movement can change only format. These are mechanisms to discriminate, not universal descriptions of this checkpoint.

Online controller inputs must exclude labels, gold loss, gradients and future states. Causal diagnostic NLL predicts each target from preceding tokens only and must recompute the affected downstream trajectory. Full-gold v1 NLL remains disabled. Generation-derived and teacher-forced alpha journals/pools must remain separate. Final next-token generation on a current prefix can be valid while the same sequence-wide alpha breaks earlier-logit prefix invariance.

## Primary paper distinction

[arXiv2609.36653v1](https://arxiv.org/html/2609.36653v1), sections3/4.2/5.4/C.1, was freshly read through the web tool. Qwen uses three eta/3 Euler stages at layers15–18; its unit setting is a1/3 residual coefficient. Table4 averages: unit59.22, Adam58.71, BB59.37. Adam lowers MMLU-Pro61.40→59.70. Theory needs proxy alignment, controlled temporal mismatch and adequate curvature margin. None is established on Rineya. No author code was freshly read or downloaded. V1's short observed author-code revisions are historical pointers with missing full immutable digests.

## Smallest complete discriminating native question

After exact-source mechanical qualification, hold a99 checkpoint/custom head, L0–L2 first pass, L12–L13 operator, masks/positions, noKV CPUFP32, scorer, data and decoding fixed. Run the existing complete registered A0,A1,fullA2grid,A3,A4constant and frozen-generation-poolA5 comparison at budgets0/1. A4 is retrospective diagnostic; it does not add online labels. A5 requires its own qualification after pool freeze. Do not drop an unfavorable or unsafe arm.

The earliest question is: does the best fixed scale improve unit recurrence, and is per-example constant-grid oracle headroom positive beyond the best global fixed policy? Alongside that question, measure A3 saturation, prefix-weighted alpha distribution, live raw-update energy/alignment and paired gains/losses. If alpha1.2 dominates and oracle headroom is absent, a more complex estimator is unmotivated in this tested family. If headroom exists but capture does not, bounded causal predictive diagnostics can decide whether a successor feature/scale rule is warranted.

DEV16 is an exposed diagnostic surface, not a powered broad-efficacy instrument. Constant-per-example finite-grid null cannot rule out adaptive token-prefix schedules or continuous scales; alpha variation alone is not H3/H4 evidence. Preserve each task family and source-correct case, distinguish semantic answers from format/abstention, and count all calibration/search/smoke/retry work. No speed claim is possible until repeated paired, minimally instrumented end-to-end timings at matched qualified quality are available.

## Explicit remaining uncertainties

1. Native source/runtime qualification on this computer is owned by the coordinator/engineer and unmeasured by this audit.
2. No campaign native alpha, headroom, predictive loss, task preservation, or latency result exists here.
3. Cross-repository model results remain attributed to their local audits; exact underlying raw executions were not independently reproduced.
4. RRP005 corrected final baseline union results were not established by the preparation/reuse records inspected.
5. Author full immutable implementation revisions and byte hashes remain unverified; paper HTML retrieval has version identity but no tool-provided byte hash.
6. The campaign's sole admissible initial native horizon remains zero/one EXTRA block application; fixture/depth/LoopSpec/CED/Dream evidence cannot widen it.

