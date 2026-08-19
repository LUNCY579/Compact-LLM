# Quality Summary

## Release checks

- Total rows: **51,000**
- Symbolically re-solved: **51,000**
- Verification failures: **0**
- Duplicate IDs: **0**
- Duplicate prompt hashes: **0**
- Duplicate structural signatures: **0**
- Template overlap across train/validation/test/challenge: **0**
- Prose-lint findings across user-visible fields: **0**
- Oracle evaluator smoke test: **4,080 / 4,080** exact matches

## Split sizes

- Train: 40,800
- Validation: 4,080
- Test: 4,080
- Challenge: 2,040

Each of the 15 families contributes equally within every split.

## Difficulty mix

- Train: 22,440 easy; 18,360 medium
- Validation: 2,310 easy; 1,770 medium
- Test: 2,310 easy; 1,770 medium
- Challenge: 2,040 challenge

## Response profile

- Mean final-answer length: approximately 1.26 words
- Median final-answer length: 1 word
- 95th percentile: 3 words
- Maximum: 11 words, primarily route or move answers

## Design safeguards

1. Public benchmark rows were not imported.
2. All solutions come from deterministic symbolic procedures or graph search.
3. Test and challenge use held-out surface-language templates.
4. Structured signatures prevent exact puzzle-state reuse across splits.
5. The independent validator reconstructs each family from `structure` rather than trusting the stored answer.
6. Code-breaking uses exact Mastermind feedback; shortest paths require a unique optimum; CSP families require a unique solution; epistemic hats use public-announcement world elimination.

## Recommended interpretation

This release is a strong training and controlled evaluation resource for a puzzle-specialized Rineya branch. It should be combined with newly authored blind puzzles and external untouched benchmarks before making broad claims about general reasoning.
