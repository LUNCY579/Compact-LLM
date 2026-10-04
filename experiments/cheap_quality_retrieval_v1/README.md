# Cheap retrieval-only evidence retention, v1

This prospectively fixed experiment measures how much required evidence survives
a small retrieval context. It creates **64 fresh, explicitly openDEV synthetic
worlds**: 32 parcel→locker questions and 32 courier→parcel→locker questions. Each
world has 64 irrelevant factual sentences with the same templates as its one or
two required supports. Early, middle and late support placements reuse the same
world/question/facts, giving 192 paired cases.

`config.json` and `protocol.json` fix generation seeds, factual templates,
algorithms, packing, padding, metrics and resources before execution. The worker
first saves **all public bytes and separate evaluator-only support records**,
writes `FROZEN_INPUTS.json`, then verifies and re-reads those exact saved bytes
before retrieval scoring. Generic shuffled sentence IDs do not label supports.

Every public retriever receives only question and sentence-ID/text records. The
schema guard rejects gold support, answer, type and placement fields. Four fixed
methods run: source lead-prefix, seeded random permutation, one-hop BM25 and
two-pass BM25. Random sentence-identity rankings stay fixed across a world's
three placements, so position does not introduce another random draw.
Two-pass BM25 expands its query using the text of its top public
first-pass sentence, retains that seed, then ranks remaining sentences with the
expanded query. Evaluator evidence never enters ranking or packing.

Packing takes the longest whole-sentence ranking prefix that fits, stopping at
the first overflow, then restores selected facts to source order. Lead-prefix
therefore remains a contiguous source prefix. **All methods consume exactly
128/256/512 actual raw native context tokens.** After selection, the same fixed
procedure adds irrelevant whitespace/lexical filler, testing each suffix against
the native tokenizer and requiring precisely one additional token per step.
Its provenance, literal text and net token count are saved. Sentence/chunk/sample
counts differ across methods; they are not used as a token-cost proxy.

The only external source is the read-only local Titan `tokenizer.json`, SHA-256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
The worker requires `tokenizers==0.22.2` and uses only `Tokenizer.from_file` and
raw `encode(add_special_tokens=False).ids`. No Transformers, Torch, NumPy, weights,
models, adapters, protected data or GPU computation is used. The process sets
`TOKENIZERS_PARALLELISM=false`.

Before tokenizer import, the worker rejects already imported model/tensor
packages and installs an import audit hook that refuses Torch, Transformers,
TensorFlow, Flax, JAX, NumPy and safetensors. The source receipt records the guard
and checks those packages remain absent after tokenizer loading. This enforces
the intended ordinary CPU-only path; it is not an adversarial security sandbox.

The token budget covers only selected factual context plus recorded padding.
**Question/chat overhead is excluded; no native total-envelope claim follows.**
Full evidence recall requires every support (both hops for two-hop questions).
Any evidence recall, support precision, selected fact count, exact consumed
tokens, padding and retrieval/packing times are also recorded. One ranking time
per case/method is reused across budgets; packing and native encoding time are
separate. Case means, type/position strata and each world's mean over its three
paired positions are reported; 192 placements are not 192 independent worlds.

These results diagnose synthetic evidence retention. They do not measure answer
correctness, intelligence improvement, native model usage or reproduce the
Lost in the Middle model experiment. Algorithms must not be tuned from resulting
recall. Scientific model tests would need their own prospective qualification.

Run tests and then the benchmark **after experiment 5**, one supervised CPU
worker at a time, with **512 MiB process-tree RSS, 900 seconds and 50 MiB outputs**:

```powershell
python -m unittest discover -s experiments/cheap_quality_retrieval_v1/tests -v
python -m experiments.cheap_quality_retrieval_v1 --out experiments/cheap_quality_retrieval_v1/runs/run-20260929-01
```

Tests use small synthetic fixtures and include mandatory exact-budget native
raw-tokenizer checks. They do not execute the 192-case study. The worker records
host/process memory at phases; the separate campaign supervisor enforces continuous
admission, process-tree memory and timeout. Existing output directories are refused.

`report.html` summarizes results. `analysis.json` and `cases.csv` retain the raw
packed contexts, source indices, padding and support evaluation. `world_clusters.csv`
and `aggregates.csv` expose descriptive clustered and stratified metrics.
`SOURCE.json`, `FROZEN_INPUTS.json`, `resources.jsonl`, `RUN.json`, and
`COMPLETED.json`/`FAILED.json` preserve source, data and hardware evidence.
All attempt outputs stay beneath ignored `runs/`; failures remain intact.
