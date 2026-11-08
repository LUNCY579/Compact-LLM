"""Create-only review publication; source reads and JSON only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def ref(path):
    raw=Path(path).read_bytes()
    return {"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={
 "schema":"CE8-ACTIVATION-METRIC-LORA-INDEPENDENT-MATH-REVIEW-1",
 "reviewer":"/root/review_activation_math",
 "reviewed_utc":datetime.now(timezone.utc).isoformat(),
 "verdict":"PASS_CONDITIONAL_MATHEMATICS_AND_SCOPED_STATIC_MAPPING_ONLY",
 "material_math_defects":[],
 "source_refs":[ref(ROOT/name) for name in ("THEORY.md","SOURCE_FREEZE.json","EXACT_FRACTION_RECEIPT.json","smallFractionexamples.py","EFFICIENCY_AND_GROUP_LIMITS.json","SHARED_MOMENT_EXACT_RECEIPT.json","verify_shared_moment.py","PRIMARY_PAPER_CONNECTIONS.json","SOURCE_MAPPING.json","independent_exact_review.py","INDEPENDENT_EXACT_REVIEW_RECEIPT.json","write_independent_math_review.py")],
 "prior_refs":[ref(ROOT.parent/name) for name in ("lora-factor-gauge-v1/CONDITIONAL_ANALYSIS.json","lora-factor-gauge-v1/WORDING_CLARIFICATION.json","retention-projection-v1/RETENTION_PROJECTION.md","joint-retention-v1/JOINT_RETENTION.md")],
 "exact_execution":{
   "command":"bare CPython 3.11.15 -I -S -B independent_exact_review.py",
   "exit_code":0,"original_fraction_checks":18,"original_shared_moment_checks":5,"additional_independent_checks":11,
   "method":"AST extraction preserves original pure function bodies and shared arithmetic/result expressions while excluding create-only writer/CLI entrypoints. Original results exactly equal saved receipts. Additional independently implemented Fraction matrix routines verify non-diagonal rank-one singular support and mixed-rank nonorthogonal block gauges.",
   "evidence_label":"MECHANICAL_EXACT_SYNTHETIC_ALGEBRA_ONLY",
   "limits":"Finite rational examples validate instances, not a universal theorem or any neural result. General optimum is accepted from the written proof below, not inferred from green examples."
 },
 "proof_review":[
   {"requirement":"Trace orientation, mixed ranks and signed scales","verdict":"PASS","reason":"tr(D_i^T Q D_j X)=s_i s_j tr((B_i^T Q B_j)(A_j X A_i^T)); factor dimensions r_i by r_j and r_j by r_i agree. All scale terms remain, including negative/zero scales. Symmetry follows from real symmetric Q,X."},
   {"requirement":"Gram PSD and common coordinates","verdict":"PASS_CONDITIONAL","reason":"Whitened W_i=Q^(1/2)D_iX^(1/2) supplies a genuine Frobenius Gram. Fixed common PSD X,Q and exact common linear operator/anchor conventions are explicit. PSD means semidefinite; zero diagonal defines no observable cosine. No factor-space cosine is substituted."},
   {"requirement":"Uncentered input second moment","verdict":"PASS","reason":"E[xx^T]=Cov(x)+mu mu^T. Omission changes the bilinear quantity by (D_i mu)^T Q(D_j mu). Deterministic nonzero-input example correctly refutes covariance substitution."},
   {"requirement":"Gauge invariance and shared-basis gauge","verdict":"PASS","reason":"B_i R_i,R_i^-1 A_i preserves the effective delta for invertible R_i. Shared F transforms by G^T F G; H by G^-1 H G^-T. Scalar coefficient blocks commute with within-adapter block G, and cyclic trace cancels it. Mixing different coefficient blocks with unrestricted global gauges would require transforming coefficients too; the note correctly restricts to block gauges."},
   {"requirement":"Metric nullspaces and almost-sure support","verdict":"PASS_CONDITIONAL","reason":"Zero expected nonnegative squared response is equivalent to Q^(1/2)Nx=0 almost surely. It is also equivalent to Q^(1/2)NX^(1/2)=0 via the trace norm. N X^(1/2)=0 implies Nx=0 almost surely; singular Q can hide genuine output changes. No claim for off-support inputs is made."},
   {"requirement":"Input-dependent Q and Kronecker factorization","verdict":"PASS","reason":"Paired E[(xx^T) tensor Q(x)] is the correct column-vectorized quadratic operator. Nonnegative integrands make its Gram PSD when finite. Scalar correlated-support example yields8 versus4, refuting replacement by separate moments. Fixed bilateral whitening is expressly restricted to the separable problem."},
   {"requirement":"Rank-constrained compression lower bound","verdict":"PASS_CONDITIONAL","reason":"For any feasible D, N=Q^(1/2)DX^(1/2) has rank<=r. Orthogonal projection onto range(N) gives residual lower bound, and the projector eigenbasis diagonal majorization yields the singular-value tail. Finite-dimensional real PSD assumptions and integer nonnegative rank are stated."},
   {"requirement":"Singular pseudoinverse lift and attainable support","verdict":"PASS","reason":"C=P_Q C P_X; nonzero singular vectors lie in the respective supported ranges. Truncated C_r retains that property, including ties and zero components. Multiplication by inverse square roots cannot raise rank; lifting gives P_Q C_r P_X=C_r and achieves the lower bound. Null additions require an independent rank feasibility check, correctly stated."},
   {"requirement":"R-squared shared-moment storage and cost","verdict":"PASS_WITH_DECLARED_ASSUMPTIONS","reason":"H=E[(A_stack x)(A_stack x)^T] stores R by R, and scalar C yields tr(C F C H). Per dense token projection O(d_in R) and outer-product O(R^2) are correctly retained. Factors, chosen Q representation, inference, obtaining activations and forming F are outside the O(R^2) statistic storage claim. Finite nonnegative weights with finite positive normalization are required; zero total means unmeasured."},
   {"requirement":"Rare-group obstruction","verdict":"PASS","reason":"X=diag(.99,.01), Q=I, T=I has dominant whitened singular directione1; rank-one D=diag(1,0) has optimal average squared error .01 and conditional rare error1. Simultaneously exact responses on both basis vectors force D=I and rank2. Saved determinant check alone is not the general proof; the linear-column argument and compression theorem prove the stated example."},
   {"requirement":"No empirical retention/acquisition guarantee","verdict":"PASS_SCOPE_DISCIPLINE","reason":"Local fixed-anchor layer response metric is not full-model loss/KL/logits/decoding or groupwise behavior. Input changes, downstream dependence, cross-layer couplings and nonlinear remainders are expressly excluded. No measured covariance, curvature, rank or behavior is claimed."}
 ],
 "novelty_comparison":{
   "prior_lora_note":"Already proves reconstructed-delta Frobenius Gram, gauge invariance, signed concatenation/rank growth and absence of universal retention from compression error.",
   "prior_projection_note":"Already distinguishes PSD functional metric from a loss upper bound and treats singular functional geometry, changed anchors, composition cross terms and valid remainder bounds.",
   "local_extension":"This note specializes fixed-layer functional geometry to explicit input/output moments, mixed-rank LoRA contractions, almost-sure nullspace equivalence, input-dependent output-metric factorization counterexample, singular support-aware optimal weighted low-rank lift, shared R-dimensional moment accumulation and a rare-group rank obstruction.",
   "claim_status":"Useful source-specific conditional extension; whitening, truncated SVD and activation-aware compression are established ideas. No new algorithm or universal retention theorem claimed."
 },
 "primary_source_verification":[
   {"url":"https://arxiv.org/html/2312.05821v5","checked":"ASVD section3.3 equations7-9","supported":"Channel absolute-mean exponent diagonal scaling and Cholesky activation scaling are described. Diagonal magnitudes do not equal full second moment in general."},
   {"url":"https://arxiv.org/html/2403.07378v5","checked":"SVD-LLM section3.1 Theorem3.2/Corollary3.3 and section3.2","supported":"Cholesky whitening maps discarded singular values to reconstruction error under invertibility; subsequent low-rank factors are recovered sequentially."},
   {"url":"https://arxiv.org/html/2605.15626v1","checked":"IO-SVD section3.1 equations2-8 and appendixB.2","supported":"Local KL/Jacobian approximation, moment decoupling, damping and inverse double whitening are explicit. The proposed literature connection accurately preserves these as assumptions."}
 ],
 "source_mapping_review":{
   "hashes":"All30 mapped source references, instruction reference, mapping prior reference and frozen core/prior references revalidated byte-for-byte.",
   "heads":"Mimo19551ddebc87c93161c303865ce6d0f130de0093 and centralized04fe137574cc38c04824b65beea9374f60c9042c reobserved unchanged.",
   "static_findings":"Read relevant source bodies: FP32 reconstructed delta soup, B-only sequential composition, raw Frobenius diagnostics, global alpha/r versus module rank fallback, centered/pooled activation CKA, supplied feature ridge, residual capture, diagonal squared-gradient Fisher and separate behavioral correlations. These support the scoped missing producer/consumer link; no global repository absence claim inferred.",
   "instrumentation":"Suggested pre-linear inputs match inspected streaming_forward_many: normalized q/k/v input, attended o input, post-attention normalized gate/up input and gated down input. Captured residuals/pooled hidden states are not these exact operator inputs.",
   "runtime_limits":"No PEFT, Torch, model, adapters, actual scale/config/layout/dtype or token inputs were loaded. Static formulas require plain compatible LoRA and prospective variant validation. Source manifest language and historical interpretations are not treated as verified model findings.",
   "future_design":"The bounded TRAIN-only measurement is explicitly unregistered and unexecuted, separate from qualification. Proposed IDs/counts/modules remain prospective; no rows selected or read."
 },
 "observations_not_material_defects":[
   "SOURCE_FREEZE.json is the core three-file freeze plus four prior references; later paper/mapping/shared-moment additions are bound by this review and their own saved references, not retroactively asserted as members of that core freeze.",
   "Shared-moment saved example uses two rank-one blocks; added independent example tests mixed ranks1 and2 plus nonorthogonal within-rank2 gauge. Neither finite example substitutes for the symbolic proof.",
   "Moment storage cost is conditional on R=sum ranks being small enough, with finite weights and chosen output metric representation; no empirical resource budget follows from asymptotic notation."
 ],
 "scope":{"models_loaded":0,"optimizer_updates":0,"scientific_rows_read":0,"activation_rows_read":0,"frameworks_loaded":0,"native_api_calls":0,"grants_created":0,"eligibility_changed":False,"existing_artifacts_modified":False,"execution_authority":False},
 "next_requirement":"Any empirical use requires a separate prospective registration and successful applicable qualification, exact source/model/adapter/tokenizer/TRAIN provenance, complete resources/accounting and independently defined acquisition/groupwise retention endpoints. This review authorizes no such execution."
}
out=ROOT/"INDEPENDENT_MATH_REVIEW.json"
with out.open("xb") as stream: stream.write((json.dumps(value,sort_keys=True,indent=2)+"\n").encode())
print(json.dumps(ref(out)))
