# Quality Evaluator v4.5 Integration

The lab includes a vendored copy of the official v4.5 case-quality policy from the supplied Rineya source archive. The source files and checksums are documented in `src/rineya_soup_lab/vendor/official_v45/PROVENANCE.md`.

## Policy characteristics

The evaluator is designed to prioritize understandable, functionally adequate short responses over formal-English perfection. In particular:

- category labels are reporting metadata, not hidden behavioral constraints;
- a reference is not assumed to be the only acceptable answer;
- asking a question is not a failure unless an explicit contract forbids it;
- minor grammar or casual phrasing does not become a major defect merely for being informal;
- explicit factual, context, safety, and response-contract failures remain scoreable;
- candidate promotion is performed by this lab’s separate development objective and uncertainty analysis.

## Bundled backend

`bundled_v45` performs:

1. prompt normalization;
2. Qwen/generic chat-template rendering;
3. deterministic or configured generation;
4. optional thinking extraction;
5. response-control finalization;
6. v4.5 case scoring;
7. durable per-case writes;
8. aggregate summary construction.

The evaluator records both raw generation length and visible final-answer length when thinking extraction is used.

## Official backend

`official_phase_v45` launches:

```text
<official_repo>/scripts/phase_1_dense_sft/evaluate_model.py
```

It verifies that the output declares Quality Evaluator v4.5 and that every dataset ID appears exactly once and in order.

## Objective versus evaluator

The v4.5 score is one input to candidate selection. The lab’s objective also includes tail quality, completeness, observable-defect rate, selected quality dimensions, brevity range, and duplicate-reply rate. This objective is frozen in the resolved configuration before candidate evaluation.

## Raw review

Automated evaluation can miss subtle correctness, social, or safety issues. Before accepting a frozen candidate, review:

- the lowest-quality cases;
- cases with the largest Anchor↔Greedy disagreement;
- cases with the largest layer-wise improvement;
- cases with the largest layer-wise regression;
- every high-stakes category;
- repeated or suspiciously generic replies.

## Provenance contract

Every evaluation writes `lab_evaluation_manifest.json` before generation. It binds the candidate recipe, endpoint fingerprints, dataset SHA-256, v4.5 policy SHA-256, and protocol. A directory containing evidence without this manifest is rejected, as is a manifest whose identity differs from the requested run.

The bundled implementation contains packaging-only path/import adaptations documented with both original-source and packaged checksums. For strict execution through the original repository script, use `official_phase_v45`.
