# Rineya Invisible-Limit Lab v1

Testing whether a 0.6B-class Rineya configuration can **uniformly** beat
Qwen3-4B — every variant family, on unseen data, at equal or lower inference
compute.

The claim, the analysis, and the failure conditions are fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md), written before any TEST inference
existed. Read that first; this file is only orientation.

## Why this lab exists

The published family's headline numbers cannot support a claim of this kind. All
256 rows of `Rineya_Long_Context_Demo_v1` are spent — showcase64 was selection
data, stability192 was inspected when the S4 threshold was chosen, and the data
card forbids training on any of it. A uniform win tuned against those rows would
be indistinguishable from the real thing.

So this lab builds a fresh substrate and treats one split as sealed.

## Splits

| Split | Scenarios | Rows | Seed | Use |
|---|---:|---:|---:|---|
| `dev` | 64 | 256 | 20260827 | all tuning, probe fitting, thresholds |
| `test` | 128 | 512 | 20260828 | **SEALED** — scored once, Stage 8 only |
| `train` | 256 | 1024 | 20260829 | Stage 5/6 training data |

No context hash is shared between any two splits.

## What makes a scenario hard

Each scenario carries one *decisive binding*, expressed three ways:

- **base** — the binding is stated
- **counterfactual** — the binding is **swapped**; every entity stays present, so
  surface statistics barely move and only the relation flips
- **unanswerable** — the binding is **destroyed**; every entity stays present, so
  the model must notice a missing *link*, not a missing name

`paraphrase` reuses the base context with a reworded question. These properties
are enforced as hard failures by the validator — they are what make the original
suite difficult, and dropping them would quietly turn "unanswerable" into the far
easier "is a name missing?".

## Layout

```
PREREGISTRATION.md      the claim, the decision rule, the sealing protocol
scripts/generate_suite.py   deterministic scenario generator
scripts/validate_suite.py   structural + semantic checks; run before any inference
data/*_answer_key.jsonl     committed ground truth
data/*_full.jsonl           gitignored bulk (regenerable; see .gitignore)
data/*_prompts.jsonl        gitignored bulk (regenerable)
```

Predictions and reports are written under
`artifacts/rineya_invisible_limit_v1/`, which is gitignored like every other
artifact tree in this repository.

## Regenerating the data

The generator is fully seeded; the same seed and count reproduce byte-identical
output. Commands are in [`.gitignore`](.gitignore). Always revalidate afterwards:

```bash
python scripts/validate_suite.py --full data/dev_full.jsonl --key data/dev_answer_key.jsonl
```

A non-zero exit means the substrate is unsound and no downstream number counts.

## Status

Stage 0 complete: substrate built, validated (0 failures, 0 warnings on all three
splits), pre-registration filed. Stages 1–8 are listed in
[`PREREGISTRATION.md` §7](PREREGISTRATION.md). TEST has never been scored.
