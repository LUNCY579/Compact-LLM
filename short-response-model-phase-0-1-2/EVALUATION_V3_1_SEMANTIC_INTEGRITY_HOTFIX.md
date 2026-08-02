> **Historical/superseded:** Evaluator v4.0 is authoritative. Do not use the commands or quality labels in this document for new official evidence. See `EVALUATION_V4_MIGRATION.md`.

# Quality Evaluator v3.1 — Semantic Integrity Hotfix

**Repository release:** v1.3.2  
**Evaluator schema:** 3.1  
**Opportunity-policy schema:** 1.1

## Why this hotfix exists

Evaluator v3.0 correctly stopped treating informal grammar, alternate wording, and natural follow-up questions as automatic failures. However, its task and logic dimensions could remain near-perfect whenever a reply was nonempty and did not match a narrow corruption pattern. Ordinary English words arranged as unrecoverable word salad could therefore score above 0.90.

Evaluator v3.1 preserves conversational permissiveness while separating it from semantic collapse. The governing distinction is:

- **Acceptable imperfection:** the intended meaning and conversational function are recoverable.
- **Semantic failure:** a typical reader cannot recover a useful response to the prompt, or required meaning is lost.

## Evaluation architecture

### 1. Mechanical integrity

The existing evaluator still detects replacement characters, control characters, repeated-token collapse, repeated n-grams, fragmentation, truncation, and reasoning/instruction leakage.

### 2. Semantic structural coherence

`scripts/common/semantic_quality.py` adds deterministic analysis of malformed clause patterns, pronoun/function-word saturation, generic-content loops, punctuation/parenthesis noise, micro-clause fragmentation, and recoverable sentence structure. This catches ordinary-word corruption that contains no Unicode error and does not cross an exact repetition threshold.

### 3. Prompt relevance

The evaluator compares prompt, reply, and reference concepts using corpus-aware inverse-document-frequency weighting and repository-domain paraphrase groups. The reference remains an example rather than a unique answer, but an unrelated nonempty string no longer receives perfect task adequacy.

### 4. Intent fulfillment

Category-aware conversational intent distinguishes acknowledgment, celebration, empathy, clarification, follow-up, uncertainty, safety boundary, factual compression, memory recall, apology repair, de-escalation, and other project behaviors. A natural follow-up can satisfy an acknowledgment case even with low reference wording overlap.

### 5. Required-meaning preservation

Strict factual, context, memory, safety, and explicit-contract cases receive stronger checks for required concepts, numbers/units, polarity, forbidden claims, and content anchors.

### 6. Continuous proportional decision

No new absolute promotion gate was introduced. Case scores, semantic-collapse rate, semantically-unresponsive rate, worst-decile quality, category tails, issue cost, NLL, and other available evidence enter the shared proportional opportunity calculation. Technical invalidity and human approval remain binary prerequisites.

## New evidence fields

Each result row includes:

- `semantic_coherence_score`
- `prompt_relevance_score`
- `intent_fulfillment_score`
- `semantic_integrity_diagnostics`
- new semantic issue labels where applicable

Each summary includes:

- `mean_semantic_coherence_score`
- `mean_prompt_relevance_score`
- `mean_intent_fulfillment_score`
- `semantic_collapse_case_rate`
- `semantic_unresponsive_case_rate`

## Phase 2B migration without repeating completed inference

The migration utility re-scores stored `generation/results.jsonl` files and performs no model inference. In official in-place mode it:

1. creates timestamped backups of every old result and summary;
2. writes v3.1 case fields and summaries;
3. rebuilds candidate-versus-full-reference output deviations;
4. archives stale parent-relative deviations, combined metrics, stage decisions, catalog, and registry;
5. writes exact evidence-lineage manifests; and
6. prepares the original Phase 2B command to run again with `-Resume`.

The resumed runner reconstructs the frontier from Stage 1 using stored generations, NLL, and footprint evidence while parent lineage still matches. If v3.1 selects a different parent at Stage N, evidence downstream of that divergence is invalidated and only Stage N+1 onward must be regenerated. This is necessary: those downstream models were physically created from the old parent and cannot be correctly reinterpreted by rescoring alone.

## Official migration command

From the repository root:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v3_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl
```

Equivalent Python command:

```powershell
python .\scripts\phase_2_layer_selection\rescore_phase_2b_artifacts.py `
  --run-root .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  --benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  --in-place `
  --prepare-resume
```

Then rerun the exact original Phase 2B command with `-Resume`.

## Non-destructive review mode

To inspect v3.1 scores before changing official evidence:

```powershell
.\scripts\phase_2_layer_selection\run_phase_2b_rescore_v3_1.ps1 `
  -RunRoot .\artifacts\phase_2b_quality_v3_thorough_20260801 `
  -Benchmark .\data\phase_2_structural_reduction\probe_150_thorough_v2.jsonl `
  -ReviewOnly
```

This creates sibling `generation_quality_v3_1` directories and does not archive the catalog.

## Full clean rerun

A complete rerun remains supported. Use the normal Phase 2B command without `-Resume` and choose a new `-OutputRoot`. Never overwrite the old experiment when comparing evaluator policies.

## Verification

Run:

```powershell
python .\scripts\common\verify_quality_evaluation.py
python .\scripts\common\calibrate_quality_profiles.py
pytest -q
```

The calibration packet includes understandable informal replies, factual/context errors, unsafe guidance, mechanical collapse, and actual semantic-collapse replies observed in the Phase 2B terminal output.

## Known boundary

The v3.1 semantic layer is deterministic and dependency-light; it is not a general-purpose LLM judge. Its strengths are reproducibility, offline use, low cost, and explicit diagnostics. Human review remains required for frontier promotion, and future calibration should expand the packet when a new false-positive or false-negative family is observed.
