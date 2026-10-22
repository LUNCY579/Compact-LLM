# Frozen procedure for cheap study 3

This protocol applies only to the assigned abstention worktree and the four
explicitly authorized exposed historical long-context files. No inherited file
is modified. No model runtime or NumPy/Torch import is permitted. New preparation
or evaluation variants require their own worktree and output directory.

1. Freeze all four source byte SHA-256 hashes, their sizes and fixed code/protocol
   hashes before parsing payloads. Preserve the freeze even if validation fails.
   Hashes bind accidental drift; a filesystem owner can rewrite all artifacts,
   so this is ordinary evidence integrity, not an adversarial security sandbox.
2. Require exactly RLC-001 through RLC-064 with B/P/C/U variants in both prompt/key
   files and both prediction/correctness maps. Verify joins, role metadata,
   answerability/gold consistency and base/paraphrase/counterfactual structure.
   A world must stay within one domain and historical split.
3. Check schema 1, labels supa/opallo and their recorded model directory identities,
   shared implementation digest
   `a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e`,
   Transformers 4.51.3, CUDA, greedy seed 17, max context 4096, max completion 24,
   TF32 flags false and float32 matmul precision highest. Scope/license flags
   must remain exploratory/no-efficacy/no-release. Require 256 successful EOS
   terminations per bank, zero overflows and generated token counts strictly
   below the cap. Failed/truncated/missing inputs block rather than disappear.
   Checkpoint hashes and parameter dtype are absent in these banks and must
   remain NOT_RECORDED; runtime precision flags do not qualify checkpoint identity.
4. Prepare all 512 records with exactly opaque ID, original question and original
   response. Do not include context, model name, source row/world metadata,
   answerability, gold or inherited correctness. IDs deterministically hash the
   fixed namespace/seed/source ID and original question/response. Sort by opaque
   ID. This supplies reproducible ordering, not protection against a malicious
   reviewer with access to source artifacts. Separate private joins remain local.
   Optional exact-response groups retain every member opaque ID and question;
   grouped batches contain at most 64 member responses. Grouping is only an
   efficiency aid. It never silently expands group labels or resolves
   question-dependent ambiguity; formal reviews still label all 512 individual IDs.
5. Provide at most 64 responses per batch and the same frozen rubric to two
   independent Codex-assisted reviewers. These are not human-certified and are
   not a qualified semantic grader. No answer-aware rewriting, cross-review
   access, adjudication or outcome-dependent subset selection is permitted.
6. Label communicative acts only: ABSTAIN explicitly lacks evidence/knowledge;
   CLARIFY requests missing necessary information without proposing an answer;
   ANSWER asserts/guesses an answer even when hedged; OTHER is nonresponsive or
   invalid; AMBIGUOUS is unresolved. When a response both asserts an answer and
   asks a question/adds a caveat, ANSWER takes precedence. Literal UNKNOWN is
   communicative abstention. No context is supplied to reviewers, so they cannot
   certify factual correctness or genuine evidence use from this packet.
7. Require two full label sets, exact packet hash binding, distinct reviewer IDs,
   one allowed label and nonempty short rationale per ID. Freeze label/prepared
   input hashes before parsing reviews or the private join; reject label drift.
   Snapshot both original label files byte-for-byte unchanged.
   Any disagreement or either AMBIGUOUS is unresolved. Report each review,
   agreement matrices and conservative bounds: a resolved category contributes
   its known indicator; an unresolved item contributes 0 to lower and 1 to upper.
   These bounds permit any substantive category for unresolved items and do not
   silently resolve even an ABSTAIN-versus-CLARIFY disagreement.
8. Compare literal raw full `UNKNOWN`, whitespace-trimmed case-sensitive full
   `UNKNOWN`, and an outer-Unicode-punctuation-only diagnostic. Never search for
   UNKNOWN as an embedded token and never use punctuation normalization to change
   inherited correctness. Report communicative ABSTAIN and CLARIFY separately
   and combined deferral. Join gold answerability only after reviews are complete.
   Report ANSWER on missing-evidence rows and ABSTAIN on answerables; these are
   annotation-based behavioral rates, not qualified semantic judgments.
9. Preserve historical per_row_correct as the exact correctness estimand. Report
   paired retained source-correct rows, source-correct losses, source-wrong gains
   and both-wrong rows for Supa source versus Opallo candidate. Do not convert
   communicative abstention or punctuation diagnostics into exact correctness.
   Include false-abstention bounds on source-correct answerable rows separately.
10. Group model/world/domain/variant/historical split. Whole-world bootstrap uses
    1,000 deterministic replicates, seed 20260929, and retains all paired models
    and variants. Paired intervals are exploratory descriptive intervals,
    unadjusted for multiple comparisons, with no promotion/causal threshold.
    Single-world intervals are degenerate descriptions and confer no inference.
11. Bound each CPU command by 512 MiB worker-tree RSS, 900 seconds and 50 MiB
    combined preparation/report output. An external supervisor is authoritative
    for the worker tree; package checks self working set/time/storage at boundaries.
    Record actual stage wall/self RSS separately from reviewer-supplied or unknown
    annotation costs and inherited historical generation costs. Zero new local
    model loads/inference tokens/training/paid calls must remain explicit. Preserve
    blocked/partial artifacts. No unattended host retry is permitted.

No protected/sealed record, CHECK/FINAL, new inference, training, model promotion,
human-certification claim or paid grader is within this study. Proceed means only
descriptive instrument inspection; unresolved annotations and provenance limits
remain visible. Stronger grader qualification or independent evidence is separate
authorized work.
