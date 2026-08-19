# Data Card: Rineya Long-Context Demonstration Suite v1.0

## Summary

The Rineya Long-Context Demonstration Suite is a synthetic, closed-context reading benchmark designed for short-response causal language models. Each example supplies a dense narrative or case record and asks one question whose answer should be returned in a few words.

The release contains 256 rows organized into 64 scenario packs. Each pack contains a base item, a paraphrased-question item, a counterfactual-context item, and an unanswerable item.

## Intended uses

- Demonstrating that a Rineya-family model can use a supplied long context
- Checking whether correct answers remain stable under question paraphrases
- Testing whether the model changes its answer after a decisive context change
- Testing `UNKNOWN` behavior when evidence is insufficient
- Comparing Rineya checkpoints or decoding configurations
- Producing an auditable public showcase after the evaluation is completed

## Out-of-scope uses

- Training on the exact suite and then reporting test performance
- Claiming general long-context mastery from this suite alone
- Measuring factual knowledge of existing books, films, or television series
- High-stakes legal, medical, financial, or safety evaluation
- Measuring retrieval over extremely long contexts
- Evaluating open-ended literary analysis or creative quality

## Dataset composition

### Splits

| Split | Scenario packs | Rows | Purpose |
|---|---:|---:|---|
| `showcase` | 16 | 64 | Publicly readable demonstrations |
| `stability_test` | 48 | 192 | Internal capability and stability measurement |
| **Total** | **64** | **256** | |

### Variants

| Variant | Rows | Expected behavior |
|---|---:|---|
| `base` | 64 | Return the supported short answer |
| `paraphrase` | 64 | Preserve the base answer |
| `counterfactual` | 64 | Change the answer with the changed decisive fact |
| `unanswerable` | 64 | Return `UNKNOWN` or an accepted equivalent |

### Domains

- Original fantasy academy chronicle
- Original detective case summary
- Original science-fiction station log
- Original expedition chronicle
- Original museum mystery and conservation record
- Original research-laboratory chronology
- Original civic and town-event chronology
- Original family-business and estate saga

Each domain contributes 32 rows.

### Task families

- Alias resolution
- Object/state tracking
- Temporal correction
- Rule exception
- Counting with exclusions and duplicates
- Causal disambiguation
- Multi-condition identity resolution
- Latest verified record over provisional accounts

Each task family contributes 32 rows.

### Length and placement

- Context words: 853 minimum, 1020 median, 1012.43 mean, 1183 maximum
- Evidence profiles: early, middle, late, and distributed
- Difficulty: 160 hard rows and 96 medium rows
- Answerability: 192 answerable rows and 64 unanswerable rows

Token counts are intentionally not precomputed because they depend on the tested checkpoint's tokenizer and chat template.

## Record structure

The full JSONL includes:

- `id`
- `scenario_id`
- `split`
- `variant`
- domain and task metadata
- `context`
- `question`
- `prompt`
- chat `messages`
- `answer`
- `accepted_answers`
- `supporting_facts`
- context and prompt word counts
- a SHA-256 hash of the context

Prompt-only files remove the answer, accepted-answer set, supporting facts, and assistant message. Answer-key files contain labels and grouping metadata without the full context.

## Generation and quality controls

The suite uses controlled synthetic generation with fixed seed `20260818`. The following invariants are validated:

- Exactly four variants per scenario
- Base and paraphrase share an identical context
- Base and paraphrase use different question wording
- Base and counterfactual answers differ
- Unanswerable labels are `UNKNOWN`
- No duplicate prompts
- No copied franchise text
- Every answer is at most 12 words
- Context word counts match the stored metadata
- Every prompt ends at the answer-generation boundary

The package includes `scripts/validate_suite.py` for repeatable structural validation.

## Evaluation recommendations

Primary metric: normalized strict exact match.

Pack-level metrics:

1. Base and paraphrase both correct
2. Base and counterfactual both correct with different predictions
3. Unanswerable accuracy
4. All four variants correct

Report coverage and missing outputs. Do not treat skipped context-overflow rows as incorrect predictions without also disclosing the overflow count. Do not truncate contexts.

For deterministic decoding, repeated identical outputs are expected and do not by themselves prove robustness. Paraphrase, counterfactual, and unanswerable pack metrics are the main anti-luck controls. Repeated runs become useful when sampling, stochastic kernels, or different serving stacks are being tested.

## Known limitations

1. **Synthetic structure:** The prose is more controlled and regular than arbitrary books, emails, contracts, or research papers.
2. **Moderate length:** Roughly one thousand words is meaningful for a compact model but is not an extreme long-context benchmark.
3. **One question per row:** The suite does not test sustained multi-turn querying over one persistent context.
4. **Mostly extractive or constrained reasoning:** It emphasizes auditable factual inference rather than open-ended interpretation.
5. **English only:** Multilingual long-context behavior is not measured.
6. **Exact-answer orientation:** Semantically correct but unexpectedly verbose responses may fail strict exact match.
7. **Template-family correlation:** Rows share controlled discourse patterns. Results should later be confirmed on human-written and naturally occurring documents.

## Contamination and publication policy

The scenarios use invented entities and events so answers are not recoverable from famous-story knowledge. Keep the stability split private until the target models have been evaluated. After publication, assume that future models may have seen the suite and create a new sealed release for consequential comparisons.

## Licensing note

The text was generated specifically for this package and does not reproduce franchise source material. The person or organization publishing the dataset should select the final license and verify that it matches the intended distribution and commercial use.
