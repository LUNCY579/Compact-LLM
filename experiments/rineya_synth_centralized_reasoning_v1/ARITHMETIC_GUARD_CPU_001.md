# Closed CPU arithmetic guard candidate

The accepted arithmetic retention requirement now has a separate prospective
CPU population and scorer implementation. It adapts pinned central public
bounded-integer semantics without importing central modules or populations.
The original task/quality/statistics policies remain historical and unchanged.

[Policy001](SYNTH_ARITHMETIC_GUARD_POLICY001.json) fixes seed202610091,
128 canonical expression parents,32 at each depth1–4, and four answer-preserving
operand-order/rendering rows per parent. The finite candidate pool is selected
by canonical content hash before solving or baseline outcomes. The declared
grammar uses additive roots, multiplication only at leaf pairs and higher
addition/subtraction; its maximum intermediate bound is20,000. It is a specific
guard subset, not balanced coverage of every permitted root operation.

The compiler and strict visible-output JSON scorer use recursive reference and
independent iterative/postfix paths. Both enforce integer leaves−50…50,
depth≤4 and every intermediate within±100,000. Parent and row IDs bind canonical
expression, fixed allocation, policy, prompt and rendering. Wrong answers,
booleans, extra/duplicate/trailing JSON, overlength responses, overflow and
host-content/target/hash mutations reject; host corruption raises separately.

[Preserved evidence](evidence/arithmetic_guard_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
contains policy-before-generation history, exact code/source snapshots,
128-parent/512-row population,512 deliberately incorrect responses and authored/
Root/independent receipts. Six controls pass, including75 enumerated small
expressions and whole-population independent checks. The independent reviewer
recompiled the saved population exactly and rejected all512 wrong examples.

These exposed CPU candidate labels establish mechanics only. Tokenization/
context fit, genuine native scorer qualification, available training/evaluation
lineage disjointness and M0 source-correct enrollment remain pending. Namespace
does not prove freshness or pretraining exclusion. The128 candidate parents cap
independent support; four renderings do not create512 independent cases. Freeze
the final explicit domain-ID roster before outcomes, then use historical
alpha0.05/(2D) and actual eligible support; no universal94-parent threshold or
post-M0 resampling is claimed. Dialogue authority and separate long-context
qualification still remain open. No model-quality or retention gain is established.
