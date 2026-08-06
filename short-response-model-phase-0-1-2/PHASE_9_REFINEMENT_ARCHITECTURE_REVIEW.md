# Phase 9 refinement architecture review

Version 1.6.1 | Behavior Curriculum v2 | August 2026

## Review conclusion

The earlier hybrid Phase 9 run was a useful execution test, but it was not valid evidence that the students learned grounded behavior. Six reviewed rows were too small for a research comparison, direct KD began before the student had learned the task, social rows were counted as unsupported factual cases, several metrics used artificial denominators, and relative comparisons divided by values at or near zero. Those problems explain the combination of zero supported correctness, a reported 100% unsupported-confident-answer rate, and implausible cross-profile changes above 10^11.

The revised Phase 9 keeps the warm inherited student as the production-foundation track and the narrow random students as research controls, but changes the official learning sequence to:

```text
accepted real teacher rows
        ↓
connected topic/document split
        ↓
controlled real curriculum from approved targets only
        +
versioned fictional behavior foundation (training only)
        ↓
deterministic weighted mixture
        ↓
curriculum SFT
        ↓
confidence-filtered progressive KD from the SFT artifact
        ↓
optional on-policy recovery
        ↓
five-condition evaluation and human gate
```

This is a materially stronger teaching design. It does not guarantee that a student will pass; it makes the next run capable of distinguishing genuine learning from smoke-test execution, retrieval failure, evaluator failure, and unstable arithmetic.

## Architecture preserved

The repository still has three student tracks:

1. **Warm depth-copy Qwen student.** Complete selected teacher layers, embeddings, tokenizer, and compatible output head are inherited. This remains the only default production-foundation track.
2. **Narrow same-tokenizer Qwen controls.** The 100M, 130M, and 170M variants remain randomly initialized research controls. Behavior data can teach the prompt/task contract but cannot replace broad language pretraining.
3. **Pretrained compact challenger.** A different-tokenizer model begins with sequence-level SFT. Direct Qwen-vocabulary KL is rejected unless tokenizer fingerprints match exactly.

The accepted Phase 8 teacher remains authoritative. Progressive KD starts from the SFT student; the student does not become its own sole teacher.

## Teaching improvements

### Trusted behavior foundation

A bundled fictional foundation contains 336 authored prompts and no real project facts. It covers:

- supported date, name, stack, and numeric extraction;
- exact evidence-insufficiency abstention;
- irrelevant and distracting passages;
- retrieval prompt injection;
- conflicting evidence;
- semantic-version procedures;
- concise social conversation;
- short completion boundaries.

The foundation is training-only and cannot satisfy the held-out real-domain gate. Its purpose is to give a warm student many repetitions of the behavior pattern while the real reviewed rows remain the higher-weight source.

### Controlled curriculum generation

`build_curriculum_dataset.py` produces only transformations whose approved answer remains known:

- canonical repetitions;
- authored or conservative question variants;
- unrelated distractors;
- counterfactual missing-evidence abstention;
- injected evidence instructions with the original supported answer;
- extra social rehearsals.

It does not ask the model to invent new factual targets. Every transformed row retains the source record and trajectory provenance.

### Weighted source mixture

The default mixture repeats the human-reviewed curriculum three times for each one foundation pass. The manifest hashes both sources and records repeat counts and the seed, preventing `-ReuseExisting` from silently reusing a stale mixture after data or settings change.

### SFT before KD

The primary sequence now trains hard-label curriculum SFT first. Progressive KD loads that SFT artifact or adapter, so KL is used to refine a student that already knows the expected task format. Direct KD from initialization remains available only through the explicit `-RunDirectKD` ablation.

### Confidence-filtered KD

For each completion token, the trainer checks how much probability the teacher assigns to the human-approved target. Tokens below the configured confidence floor do not contribute KL. When no target token is teacher-confident, training falls back completely to hard labels rather than pulling the student away from the approved answer.

The hard-label weight is scheduled from 1.0 during KD warmup toward a configurable start/end mixture. The final completion boundary receives extra loss weight to reinforce concise EOS/end-of-turn behavior. Completion targets that exceed the budget are skipped and reported instead of being mechanically truncated into broken training labels.

### Reuse integrity

`-ReuseExisting` now verifies the pipeline revision, training-data hash, maximum-step setting, student tree hash, initial-adapter tree hash, teacher tree hash, held-out benchmark hash, model/adapter hashes, evaluation condition set, and smoke/research mode. A stale artifact is rebuilt instead of being silently compared under a new dataset or checkpoint.

## Data validity improvements

The official research-ready path now requires, by default:

- at least 30 accepted real rows;
- at least eight independent connected topic/document groups;
- nontrivial development and test sets;
- supported and unsupported coverage in development and test.

Rows are split by connected components across topics and referenced documents. Any transitive document overlap stays inside one split. A tiny packet is accepted only with `-SmokeTest`, is marked `research_ready: false`, and cannot be auto-recommended.

## Evaluation corrections

### Explicit case classes and denominators

Each row is classified as supported, unsupported, conflict, social, or unknown. Social replies no longer count as unsupported factual questions. Every rate contains its numerator and denominator. When a condition has no applicable rows, the rate is `null`, not a fabricated 0.0 or 1.0.

### Functional correctness

Supported correctness is now based on:

- the explicit response contract;
- required and prohibited content;
- grounding support;
- bounded reference overlap only as a fallback when no explicit content anchors exist.

The broad conversational quality score remains a diagnostic and does not silently become a factual correctness gate.

### Adversarial decomposition

Two adversarial modes are reported:

- `adversarial_rag`: the sanitizer can remove prompt-injection passages, measuring complete-system filtering;
- `adversarial_passthrough_rag`: malicious evidence is retained, measuring the student's own instruction resistance.

This prevents sanitizer success from being mistaken for model robustness.

### Numerically stable comparisons

SFT/KD and cross-profile training comparisons use bounded absolute deltas on metrics already bounded to [0,1]. They never divide by a zero reference, so the earlier 10^10–10^11 changes cannot recur. The repository-wide proportional opportunity engine still controls resource-versus-quality promotion; the bounded comparison is the stable training diagnostic.

### Functional readiness blockers

A candidate cannot be recommended for production when evaluation is non-research-ready, applicable case counts are zero, supported cases exist but none are answered correctly, or unsupported cases exist but none receive a correct abstention. Random narrow profiles remain research controls regardless of their apparent memory margin.

## Expected interpretation of the next run

- Poor oracle RAG means the student, prompt construction, checkpoint, or evaluator is failing; retrieval is not the main cause.
- Good oracle RAG with poor real RAG means retrieval, filtering, or gate calibration is failing.
- Good filtered adversarial results with poor passthrough results means the sanitizer works but the student is still vulnerable to evidence-borne instructions.
- SFT improvement followed by KD regression means SFT should be retained; KD is not forced to win.
- A random narrow student that remains incoherent is an expected negative result, not evidence that the behavior curriculum failed to pretrain language.

## Limitations

No CUDA training was performed in the packaging environment. Transformers, PEFT, TRL, and CUDA were unavailable, so the revision is source-, unit-, evaluator-, and data-pipeline-validated rather than model-quality-proven. The real conclusion still requires a fresh Windows/RTX 4070 run, raw-output review, and the Phase 9 human gate.
