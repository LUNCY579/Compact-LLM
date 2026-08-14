# Methods and Mathematical Framework

## Scope and notation

Let (x) denote the prompt, (y_{<t}) the shared accepted prefix before
generation step (t), (	heta) the frozen backbone parameters, and
(Delta_i) the parameters of specialist adapter (i). The committee contains
(K) compatible experts indexed by (i\in\{1,\ldots,K\}), and the tokenizer
vocabulary is (mathcal V).

For a base weight matrix (W_0), a LoRA expert uses

\[
W_i = W_0 + \frac{\alpha_i}{r_i} B_i A_i,
\]

where (A_i\in\mathbb R^{r_i\times d_{\text{in}}}),
(B_i\in\mathbb R^{d_{\text{out}}\times r_i}), (r_i) is the rank, and
(alpha_i/r_i) is the configured scale. In the verified K=7 committee all
adapters are rank-16 LoRAs with a common declared base and target-module set;
their exact hashes are recorded in
`reports/PHASE2_CORNERSTONE_VERIFICATION.md`.

## Expert next-token distribution

At step (t), expert (i) evaluates the same prompt and accepted prefix:

\[
z_{i,t}=f_{\theta+\Delta_i}(x,y_{<t})\in\mathbb R^{|\mathcal V|},
\qquad
p_{i,t}(v)=\frac{\exp z_{i,t}(v)}{\sum_{u\in\mathcal V}\exp z_{i,t}(u)}.
\]

The expert's directional decision is

\[
v_{i,t}=\operatorname*{arg\,max}_{v\in\mathcal V}p_{i,t}(v)
=\operatorname*{arg\,max}_{v\in\mathcal V}z_{i,t}(v).
\]

Exact logit ties use the smallest token ID, matching stable greedy `argmax`
semantics.

## Hard top-1 plurality

Define the vote count for token (v) as

\[
c_t(v)=\sum_{i=1}^{K}\mathbf 1[v_{i,t}=v].
\]

The plurality candidate set is

\[
\mathcal T_t=\operatorname*{arg\,max}_{v\in\mathcal V}c_t(v).
\]

If (|\mathcal T_t|=1), its member is selected. If several tokens tie, the
verified implementation selects

\[
y_t=\operatorname*{arg\,max}_{v\in\mathcal T_t}
\sum_{i=1}^{K}\log p_{i,t}(v),
\]

then uses the smallest token ID if this secondary score also ties. The accepted
prefix becomes

\[
y_{<t+1}=y_{<t}\mathbin\Vert y_t,
\]

and all experts reevaluate that same new prefix. EOS is an ordinary candidate
token in the primary condition; generation stops when EOS wins or the 32-token
limit is reached.

## Soft and logit fusion comparators

### Arithmetic probability mean

\[
\bar p_t(v)=\frac{1}{K}\sum_{i=1}^{K}p_{i,t}(v),
\qquad
y_t^{\text{prob}}=\operatorname*{arg\,max}_{v\in\mathcal V}\bar p_t(v).
\]

### Arithmetic logit mean

\[
\bar z_t(v)=\frac{1}{K}\sum_{i=1}^{K}z_{i,t}(v),
\qquad
y_t^{\text{logit}}=\operatorname*{arg\,max}_{v\in\mathcal V}\bar z_t(v).
\]

Logit averaging is sensitive to additive/scale conventions across experts;
normalizing logits before averaging is therefore a distinct comparator, not an
equivalent implementation.

### Expert-temperature-calibrated probability mean

For (T_i>0),

\[
p^{(T_i)}_{i,t}(v)=\operatorname{softmax}\!\left(z_{i,t}/T_i\right)_v,
\qquad
\bar p^{\text{cal}}_t(v)=\frac1K\sum_i p^{(T_i)}_{i,t}(v).
\]

Each (T_i) is selected by teacher-forced gold-token negative log likelihood
on DEV only. Shared-temperature fusion constrains all (T_i=T).

### Rank-only aggregators

For rank (r_{i,t}(v)), truncated at depth (m):

- top-(m) approval gives (s_i(v)=\mathbf 1[r_{i,t}(v)\le m]);
- Borda gives (s_i(v)=m-r_{i,t}(v)+1) when (r_{i,t}(v)\le m), else 0;
- reciprocal-rank gives (s_i(v)=1/r_{i,t}(v)) when
  (r_{i,t}(v)\le m), else 0.

The committee selects the token maximizing (sum_i s_i(v)), with summed raw
log probability and smallest token ID as deterministic tie breakers. These
comparators test whether ordinal information, rather than top-1 plurality
specifically, supplies the useful bottleneck.

## Positive-temperature invariance of hard voting

For any expert-specific (T_i>0), division by (T_i) is strictly monotone:

\[
z_i(a)>z_i(b) \iff z_i(a)/T_i>z_i(b)/T_i.
\]

Softmax is coordinate-order preserving, so

\[
\operatorname*{arg\,max}_v z_i(v)
=\operatorname*{arg\,max}_v \operatorname{softmax}(z_i/T_i)_v.
\]

Therefore each expert's top-1 proposal and every plurality vote count are
invariant to positive per-expert temperature scaling. If one token has a
unique maximum vote count, the selected hard-plurality token is also invariant.
The deployed rule has an explicit boundary: when vote leaders tie, it compares
their summed raw log probabilities. Expert-specific scaling can change that
secondary ordering even though no proposal or vote count changes. A shared
temperature preserves the secondary ordering. Probability averaging is not
invariant because

\[
\frac1K\sum_i\operatorname{softmax}(z_i/T_i)
\]

changes as any (T_i) changes. Thus the exact theorem covers proposals, vote
counts, and unique-plurality winners; it does not cover the confidence-based
tie-break under unequal temperatures. This robustness property does not
establish that calibration mismatch causes the observed sequence accuracy gap.
That causal claim requires the DEV-fitted and frozen intervention experiments
specified in `paper/EXPERIMENTAL_PROTOCOL.md`.

## Agreement and diversity quantities

Plurality agreement is

\[
A_t=\frac{\max_v c_t(v)}{K}.
\]

With empirical vote distribution (q_t(v)=c_t(v)/K), vote entropy is

\[
H_{\text{vote},t}=-\sum_{v:c_t(v)>0}q_t(v)\log q_t(v).
\]

If (c_{(1),t}\ge c_{(2),t}) are the two largest vote counts, vote margin is

\[
M_t=c_{(1),t}-c_{(2),t}.
\]

For two expert distributions (p_i,p_j), Jensen-Shannon divergence is

\[
\operatorname{JS}(p_i,p_j)=\tfrac12\operatorname{KL}(p_i\Vert m)
+\tfrac12\operatorname{KL}(p_j\Vert m),
\quad m=\tfrac12(p_i+p_j).
\]

Distributional divergence, top-1 disagreement, double-fault rate, and Q-style
error association measure different phenomena and must not be collapsed into a
single unqualified notion of “diversity.”

## Calibration quantities

For a teacher-forced gold token (g_t), expert top-1 correctness is

\[
a_{i,t}=\mathbf 1[v_{i,t}=g_t].
\]

Negative log likelihood and multiclass Brier score are

\[
\operatorname{NLL}_{i,t}=-\log p_{i,t}(g_t),
\qquad
\operatorname{Brier}_{i,t}=\sum_{v\in\mathcal V}
\left(p_{i,t}(v)-\mathbf1[v=g_t]\right)^2.
\]

For confidence bins (B_b), expected calibration error is

\[
\operatorname{ECE}=\sum_b\frac{|B_b|}{n}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
\]

ECE depends on binning and is reported with its bin definition. Sequence-level
correctness conditioned on free-running confidence is not a substitute for
gold-token calibration.

## Rescue and harm

Let (C_A(j),C_B(j)\in\{0,1\}) denote correctness of baseline (A) and
candidate (B) on paired example (j). Define

\[
\operatorname{rescue}(A\to B)=\frac1N\sum_j
\mathbf1[C_A(j)=0\land C_B(j)=1],
\]

\[
\operatorname{harm}(A\to B)=\frac1N\sum_j
\mathbf1[C_A(j)=1\land C_B(j)=0],
\]

and

\[
\operatorname{net\ gain}=\operatorname{rescue}-\operatorname{harm}
=\operatorname{Acc}(B)-\operatorname{Acc}(A).
\]

Exact McNemar testing conditions on the number of discordant pairs; paired
bootstrap intervals resample examples, not unpaired method outputs.

## Committee utility and marginal contribution

For nonempty expert subset (S\subseteq\mathcal N), let

\[
U(S)=\operatorname{performance}(S).
\]

The leave-one-out contribution of expert (i) to the full committee is

\[
\Delta_i^{\text{LOO}}=U(\mathcal N)-U(\mathcal N\setminus\{i\}).
\]

With (U(\varnothing)) explicitly defined (zero in the cached teacher-forced
analysis), the exact Shapley value is

\[
\phi_i=\sum_{S\subseteq\mathcal N\setminus\{i\}}
\frac{|S|!(K-|S|-1)!}{K!}
\left[U(S\cup\{i\})-U(S)\right].
\]

Standalone accuracy and Shapley value need not be aligned: a weak standalone
expert can be valuable if it corrects errors in coalitions where other experts
fail.

## Teacher-forced and free-running objectives

Teacher forcing gives every method the gold prefix and evaluates the local
decision

\[
\hat y_t(x,y_{<t}^{\text{gold}})\stackrel{?}{=}y_t^{\text{gold}}.
\]

Free-running evaluation recursively supplies each method's selected token:

\[
\hat y_t(x,\hat y_{<t})\to\hat y_{<t+1}.
\]

Local token errors can compound or be self-corrected, so teacher-forced token
accuracy and free-running exact-match accuracy answer different questions.
Evidence for trajectory correction requires a same-split, frozen-method
comparison showing a larger hard-vote advantage under free-running inference
than under local gold-prefix decisions.

## Cost

The implementation-independent expert-token forward proxy is

\[
C=\sum_t K_t,
\]

where (K_t) is the number of consulted experts at step (t). It does not
replace measured latency, time to first token, peak allocated VRAM, process RSS,
or adapter storage. For adaptive consultation, (K_t) varies with the routing
decision; both escalation rate and false non-escalation rate are required.

## Limits of classical voting intuition

Condorcet-style majority intuition assumes independent binary voters with
stationary competence above chance. Autoregressive adapter consensus is
multiclass, experts share a backbone and training lineage, their errors are
correlated, competence varies by task and prefix, and the selected token changes
all subsequent decisions. Classical ensemble theory is therefore useful as
intuition, not a literal proof that increasing (K) must improve performance.
