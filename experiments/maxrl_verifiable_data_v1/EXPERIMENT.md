# MAXRL-DATA-001: verifier-first maximum-likelihood post-training

Status: TESTABLE IDEA / DATA CONSTRUCTION. No pretrained execution or efficacy claim.
Repository: LUNCY579/ChatBubble. New branch: research/maxrl-verifiable-data-v1.
Code parent: main@a6ee824584a9e28881826ab92a11db48e124d90c.

## Scientific target

Paper: Tajwar et al., Maximum Likelihood Reinforcement Learning, arXiv:2602.02710v3 (2026-08-21), https://arxiv.org/html/2602.02710v3 . Official project: https://zanette-labs.github.io/MaxRL/ .

The target is probability mass over ALL verifier-accepted completions, not likelihood of one predetermined reference string. Binary terminal feedback and fixed-size, independently sampled, on-policy groups are the primary setup. The practical advantage is (r - group_mean)/(group_mean + epsilon), with all-failure groups producing zero signal. At epsilon=0 under the paper's assumptions, the centered practical estimator targets truncation order N-1, not N. Epsilon, clipping, length normalization, KL, stale-policy reuse, and adaptive retries must not silently inherit the exact theorem.

H1: At matched resource ceilings and equal tuning allowances, MaxRL improves held-out acquisition relative to GRPO on fresh verifier-backed tasks.
H2: Any acquisition improvement does not exceed a prospectively registered preservation-loss margin. H2 is independent of H1 and is not implied by small parameter displacement.
H3 (later): Paired delta sparsification or independently trained skill composition preserves a useful acquisition advantage. Not part of the first primary comparison.

## Why this repository and base

ChatBubble contains the reasoning/data program and the relevant objective, retention, and task-arithmetic research branches. Starting from main deliberately avoids inheriting or amending the uncompleted RoLA-NGU factorial, tangent experiment, or another experiment's authorization. Sibling experiments are context, not merged code or claimed results. No existing files, branches, weights, registrations, or spent instruments are changed.

## Dataset construction contract

Generate original bounded symbolic instances, not benchmark copies or unverified teacher prose. Planned families: countdown expressions, exact-cardinality subset sums, finite-domain linear constraints, Boolean CNF assignments, minimum-cost paths, precedence orders, short affine programs, and register-state tracking. The first seven exercise semantic acceptance (some have multiple valid solutions); state tracking supplies a largely single-answer control. The corpus is a short-answer synthetic reasoning method port, not a replication of GSM8K, natural-language competence, long-context retention, or unrestricted chain-of-thought training.

Policy inputs contain only public problem statements. Construction witnesses and independently recovered reference answers stay outside policy prompts. Solvers must reconstruct answers from visible specifications; production reward verifiers must not import generation or reference-search code. References are audits and an optional disjoint warm start, not fabricated on-policy rollouts. Reward is exactly 0 or 1, with no formatting bonus or shaped reward.

Canonical JSONL, policy-only JSONL, verl-shaped JSONL, optional Parquet conversion, oracle audit sidecars, and a messages-only warm-start view are separate artifacts. Malformed responses receive zero; corrupt verifier metadata raises an error. Avoid eval, executable generated code, unbounded parsers, duplicate JSON keys, nonfinite values, booleans-as-integers, and arbitrary extra answer fields.

Split assignment derives from a canonical public-instance hash, independent of wording and random seed. Enforce disjoint IDs, exact public-instance hashes, and prompt hashes. Challenge uses reserved wording and greater algorithmic complexity; it is NOT an unseen algorithmic family or a sealed test. All included evaluation data are exposed development assets. Do not claim paraphrase/isomorphism-level or historical-corpus exclusion without a separate audit. Difficulty is generator-defined, never a fabricated source-model success probability.

## First native experiment (not opened by this registration)

Start both GRPO and MaxRL from one exact, ungated Supa 1.7 source after source/runtime/tokenizer qualification. Fix trainable scope, precision, optimizer family, reference/KL policy, prompt distribution, EOS/response masks, loss reduction, rollout temperature, group size, tuning allowance, and compute ledger. Candidate group sizes 8 or 16 are feasibility choices, not demonstrated optima. Each diverged learner must collect its own on-policy groups; shared frozen responses do not substitute for online training.

Use an untrained source baseline. A supervised warm-start may precede BOTH arms identically and must remain a separate factor. Do not mix NGU, recurrence, RoLA, sparse-delta processing, or tangent training into the first objective comparison. Charge all failed generations and verifier attempts. Report prefill/decode/loss tokens, elapsed time, memory, updates, and tuning costs; equal optimizer steps or equal tokens alone do not establish matched compute.

Primary acquisition: held-out pass@1. Report actual selected-answer accuracy only when a real selection procedure is executed; pass@k is oracle candidate coverage, not selected-answer accuracy. Diagnostics: pass@4/16 where supported; per-family difficulty-stratified scores; all-failure/mixed/all-success group rates; advantage/gradient magnitudes and effective parameter displacement.

Preservation: fresh eligible source-correct retention, literal abstention on unanswerable cases, false abstention on answerable cases, and separately qualified long-context instruments. This acquisition corpus is NOT a preservation bank. Do not reopen historical/spent instruments or promote net gains as no-forgetting. Use paired gains/losses and prospectively bound meaningful margins and power; non-significance is not equivalence. Do not pool correlated seeds or sibling derivatives as independent replications.

## Remaining native bindings

Model and tokenizer asset hashes; exact local runtime; actual token lengths and chat-template/EOS acceptance; sampler/logprob/gradient parity; external historical exclusion audit; human review of stratified prompts; preservation bank; resource allocation; update ceilings; primary margin/power; and immutable TRAIN/DEV/final instrument hashes. These are technical requirements, not a request for another blanket permission. Dataset generation and CPU mechanics are authorized now; this package does not run or authorize pretrained training.

## Evidence discipline

No claims of free learning, no forgetting, universal composability, paper speedup transfer, model-qualified difficulty, or training readiness from unit tests. Keep locally executed checks, remote publication, native execution, and scientific efficacy as separate receipts. DATA_CARD.md and VALIDATION.json will record actual generated counts and remaining limitations after construction.
