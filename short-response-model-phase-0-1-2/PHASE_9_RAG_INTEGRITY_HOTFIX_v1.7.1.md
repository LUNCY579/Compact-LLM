# Phase 9 RAG integrity and evaluation hotfix — v1.7.1

## Reason for the hotfix

The failed hybrid run showed real architectural compression but no usable RAG behavior: supported-answer correctness was zero and unsupported questions were answered confidently. The old generic opportunity calculation could still report zero quality loss when the teacher was equally weak, and near-zero reference values could produce enormous relative gains. Those results are research evidence, not a Phase 9 PASS.

## Implemented corrections

1. **No hidden prompt truncation.** Phase 9 now budgets factual evidence and trusted procedures before rendering the chat prompt. The system grounding contract and final question are never removed by tokenizer truncation. A fixed prompt that does not fit fails closed.
2. **Prompt provenance.** Every schema-1.2 evaluation row records the prompt hash, full and final prompt-token counts, selected/dropped passage IDs, question/contract presence, prompted support IDs, and whether hidden truncation occurred.
3. **Decode provenance.** Every generated row records raw visible output, raw output with special tokens, blank-output checks, replacement-character checks, and control-token leakage checks.
4. **Training prompt safety.** Completion training rebuilds structured Phase 9 prompts with the same evidence budget and skips a row rather than retaining only the tail of an overlong prompt.
5. **Core RAG outcomes affect opportunity cost.** The gate now combines the generic relative evaluator loss with an explicit Phase 9 task-contract deficiency covering supported correctness, grounding, abstention, conflict handling, injection resistance, and prompt/decode integrity. Core dimensions use a proportional bottleneck cost: a 100% functional failure contributes a 100% cost, while a 30% failure contributes 30%. An equally broken teacher can no longer make a zero-correct student appear lossless or proportionally worthwhile merely because compression is large.
6. **Stable zero-reference comparisons.** Near-zero references use bounded absolute change instead of division by an epsilon. All such gains/losses remain finite and bounded to `[0,1]`.
7. **Teacher preflight.** The official runner evaluates and audits the teacher before any student training. It stops when the teacher, evidence, prompt, or decode path has not demonstrated usable behavior, unless an explicit research-only override is supplied.
8. **Raw-result diagnosis.** `audit_phase_9_results.py` and `run_phase_9_diagnostics.ps1` classify failures as evaluation rerun required, pipeline integrity failure, teacher/data failure, or student training failure.
9. **Schema invalidation.** Evaluation and training schemas are now `1.2`; old evaluations are not silently reused, and old training artifacts are not reused by the v1.7.1 training runner.

## Interpretation policy

Compression is never sufficient by itself. Phase 9 must first demonstrate that the evidence path can answer supported questions and abstain when support is absent. This follows the project order of proving behavior before efficiency.
