# Data contract

One UTF-8 JSON object per line. Unknown keys are errors. Example (format illustration,
not a benchmark and far below the sample-size gate):

```json
{"id":"math-001","group":"math-family-001","split":"train","suite":"acquisition","prompt":"Return only the number: 2 + 3.","target":"5","rejected":"6","source":"owned-curated-traces","revision":"frozen-v1","verifier":"exact"}
```

Required strings: id, group, split, suite, prompt, target, source, revision.
`split` is exactly `train`, `select` or `confirm`. IDs must be globally unique. All
variants of a task family share a group. A group or normalized prompt cannot appear
in different splits. Group IDs must also unite paraphrases, translated variants and
shared source problems; exact-hash checks cannot automatically discover semantic leaks.

Use suite `acquisition` for the new capability. Protected suites default to `retention`,
`instruction`, `general_qa`, `roleplay_contract`, `abstention` and `long_context`. Train
replay interleaves suites deterministically so short stages do not use only the first
protected suite. Development and confirmation macro-average within groups, then suites.
For a focused preliminary benchmark, change required suites only BEFORE freezing the
campaign and explicitly narrow the claims. Never imply unmeasured generalism retention.

`rejected` is optional except on DPO examples. It must differ from the preferred target.
Preference provenance is the curator's responsibility: no generated score is silently
represented as a human preference. Targets must pass their declared verifier.

Verifiers:

- `exact`: Unicode-normalized whitespace equality, otherwise case/punctuation-sensitive.
- `json_equal`: parse and compare canonical JSON; malformed JSON fails, booleans and
  numbers remain different. Standard finite JSON only.
- `constraints`: optional `must_contain` and `must_not_contain` string lists; at least
  one constraint is required. These are contract checks, not a semantic quality judge.

No Python eval, executable rubric, remote judge, answer key or private metadata is
passed to generation. Every capped/incomplete generation receives score zero during
evaluation. Training rejects a truncated RL group rather than selectively training
only its completed members. Successful complete TRAIN rollouts alone enter the bounded
consolidation archive. Selection/confirmation outputs are never distilled back in.

`source` and `revision` identify the data lineage, but do not attest its license or
non-contamination. Before using MiMo or teacher traces, verify licensing, collection
permissions, exact dataset revision, task-family decontamination and benchmark overlap.
The v1 JSONL reader fingerprints all splits; confirmation gold is operationally
unused during fitting/selection but is not cryptographically hidden from the process.
Independent evaluator custody is recommended for strong blinded claims.
