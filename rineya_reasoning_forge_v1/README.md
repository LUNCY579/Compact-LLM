---
license: cc0-1.0
language:
- en
task_categories:
- text-generation
tags:
- synthetic
- reasoning
- long-context
- small-language-models
pretty_name: Rineya Reasoning Forge v1
---
# Rineya Reasoning Forge v1

**10,000 original, machine-checked training/development/test examples in 16 task families.**
Generated for Cheng Li's Qwen-derived Rineya research on 2026-09-11. This is a **candidate skill-training supplement**, not a demonstrated improvement to Rineya and not a replacement for broad language recovery data.

## Start here

| Artifact | Purpose |
|---|---|
| `pilot/train_512.jsonl` | Balanced short-prompt pilot: 32 training examples per family |
| `pilot/validation_128.jsonl` | 8 development examples per family, for pilot selection only |
| `data/train.jsonl.gz` | All 8,000 training candidates |
| `data/validation.jsonl.gz` | 800 development examples |
| `data/test_iid.jsonl.gz` | 800 held-out examples from the same procedural families |
| `data/test_transfer.jsonl.gz` | 320 harder or shifted task instances |
| `data/test_length.jsonl.gz` | 80 very-long-input stress instances; evaluation only |
| `probes/layout_views.jsonl.gz` | 160 matched views of 32 existing test worlds; evaluation only |
| `metadata/*.jsonl.gz` | Separate audit records and references. Do not put these in a model prompt |
| `PREVIEW.md` | Abbreviated examples from TRAIN, not the held-out sets |
| `DATA_CARD.md` | Construction, limitations, lengths, verification, and intended use |
| `docs/TRAINING_PLAN.md` | Bounded pilot, loss-mask checks, mixed-data use, and comparisons |
| `docs/PUBLIC_DATA_OPTIONS.md` | Rechecked public-data options and licensing cautions |
| `reports/` | Build report, full-corpus validation, tests, probe validation, release hashes |

The pilot is a subset of the main splits; it is **not** an extra 640 independent examples. Likewise, the 160 probe views are **not** 160 new independent worlds.

## What this trains

| Family | Core skill | Validation |
|---|---|---|
| `deduction` | Chained conjunctions; explicit negation; supported/refuted/both/unknown | Forward-closure checker reading visible facts and rules |
| `relational_join` | Follow 2–5 named relations, ignore near matches, detect missing links | Graph join interpreter |
| `detective` | Intersect opportunity, badge-access, and trace evidence in an original fictional case | Independent candidate-set intersection |
| `story_state` | Track an aliased object through dated story events | Timestamp and alias resolution |
| `document_reconciliation` | Use approved versions; reject drafts, retractions, and future revisions | Snapshot-aware version filter |
| `signed_ledger` | Multistep signed arithmetic with void entries and distractor accounts | Independent ledger evaluator |
| `temporal_order` | Unique, ambiguous, or impossible schedules | Exhaustive permutation checking |
| `truth_tellers` | Formal honest/liar statements, ambiguity, and contradictions | Exhaustive satisfying assignments |
| `hanoi` | Minimum moves and a canonical optimal next move from arbitrary legal states | Forward BFS generator versus reverse BFS stack-based checker |
| `weighted_routes` | Lowest-cost directed paths, unreachable cases, tie-breaking | Dijkstra generator versus Bellman–Ford checker |
| `counterfactual` | Intervene on Boolean systems; distinguish replacement from observation | Iterative generator versus recursive evaluator |
| `program_trace` | Execute a small integer instruction language in numbered order | Separate interpreter; no arbitrary code execution |
| `grounded_summary` | Compress goals, obstacles, adopted actions, and confirmed outcomes | Controlled-language reference and factual-slot checks |
| `dialogue_memory` | Update each speaker's preferences without mixing speakers or fields | Turn-number state reconstruction |
| `communication` | Polite decline, one-question clarification, date correction, and honest progress updates | Controlled-language reference and slot checks |
| `source_boundaries` | Read verified records without obeying quoted instruction-like text | Exact item/field extraction |

These are **16 skill families**, not 16 complete fields of human knowledge. Story and communication language is deliberately controlled and partly templated. Broad factual knowledge, genuine literary interpretation, sophisticated mathematics, unrestricted coding, multilingual competence, and open-ended social judgment are not comprehensively covered.

## What was actually checked

The completed build and independent release validator both recomputed every one of the **10,000 reference answers** from the model-visible prompt. **182 CPU tests passed**, with no skips. Tests cover known answers, arbitrary-state Hanoi, lexical shortest-path ties, contradictions, evidence mutations, context-layout invariance, corruption rejection, reproducible builds, and no-truncation length filtering.

Of the canonical rows, **8,750 use exact-answer checks**. The other **1,250 are summary/communication references** checked against structured facts and response constraints. Their checks are not an independent human assessment of prose quality or an exhaustive semantic proof. The free-form scorer therefore reports lexical slot diagnostics separately, never disguising them as general reasoning accuracy.

The release also checks exact prompt disjointness and exact canonical-case disjointness across splits. Duplicate latent cases, including repeated small Hanoi instances, are rejected during generation. This does **not** prove semantic/template independence or absence of overlap with your private benchmarks.

## Lengths: words, not tokens

The training inputs range from **246 to 4,130 whitespace-delimited words**. The separate long-input stress split ranges from **8,182 to 16,414 words**. Exact split statistics are in `reports/build_report.json`.

**No real tokenizer was available in the build environment. These are not certified 2K/4K/8K token windows.** IDs, punctuation, and your modified Rineya vocabulary can change token counts considerably. The token-count field is explicitly `null` until measured locally.

Use the exact model's tokenizer and chat template before training. The following command measures complete rendered training examples, exports fitting rows, and lists overlength rows **without truncating them**:

```bash
python scripts/prepare_for_model.py \
  --input pilot/train_512.jsonl \
  --tokenizer /path/to/your/exact/rineya/tokenizer \
  --max-length 2048 \
  --output prepared/train.jsonl
```

Run the same command separately for `pilot/validation_128.jsonl` with output `prepared/validation.jsonl`. This optional utility requires Transformers in your existing model environment. It defaults to local files and `trust_remote_code=False`; it never downloads a model or starts training unless you separately invoke such actions. Its filtering logic was unit-tested with a stub tokenizer; actual Rineya tokenization remains to be run locally.

Do not silently truncate: an early clue or final question can disappear and make a formerly correct label unsupported. Start with the pilot, not the very-long stress file. No 8 GB VRAM fit or speed claim is made.

## Training format

Each compressed JSONL row has only `id` and `messages`:

```json
{"id":"RRF1-...","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Keep `id` for joins but train on `messages` only. Apply loss to the **final assistant answer and its turn terminator**, not the source records or instructions. The provided preparation script does not create loss masks or configure a trainer: inspect a decoded training batch and its labels before optimization. Use the exact checkpoint's chat template and explicit non-thinking behavior where that model supports it.

Answers intentionally emphasize compact output: most exact tasks use a label, integer, short name, path, or next move. Summaries and polite replies are longer when required. Long chain-of-thought transcripts are not included. Audit fields and core-record annotations are not training targets.

## Why combine this with public data?

This dataset supplies controllable difficulty, unambiguous closed-world targets, missing-information cases, and targeted context tests. It does not supply the breadth or natural variation of a large general-language corpus. Keep broad recovery and compact instruction-following data in your program; test this as an additional skill-SFT component. A starting hypothesis is about 20% custom examples in a mixed SFT pilot, not a proven optimal ratio. Count both input tokens and loss-bearing tokens when comparing mixtures.

An encouraging learning-rate result on a recovery corpus does not establish general non-regression or prove that CED caused a gain. Use the matched architecture/training controls in `docs/TRAINING_PLAN.md`.

## Rebuild and verify

No API keys, GPU, external dataset, or network are needed for the core generator and checkers. Python 3.10+ is intended; the tested runtime is recorded in the release report.

```bash
python -m unittest discover -s tests -v
python scripts/forge.py --output rebuilt
python scripts/verify.py --root rebuilt
python scripts/build_probes.py --root rebuilt
```

The seed is `20260911`. Small smoke builds use `--scale .02`; they intentionally have different counts and distributions from the published full release. A new seed makes another candidate corpus, not a new independent model benchmark. Never use repeated test feedback to turn a held-out test into a training set.

For your own predictions, save `{"id":"...","prediction":"..."}` per line and run:

```bash
python scripts/score_predictions.py \
  --predictions my_test_predictions.jsonl \
  --metadata metadata/test_iid.jsonl.gz \
  --output my_test_scores.json
```

No model evaluation was performed here. Gold-reference self-scoring is not model performance.

## Provenance and rights

All task instances, fictional scenes, and code were constructed in this conversation. No Sherlock Holmes passages, public benchmark questions, private Rineya test questions, or public Hugging Face dataset rows were imported. The original data is offered with **CC0-1.0 license intent**, and the original generator code with MIT terms; see the license files. This does not amount to a blanket legal warranty, a guarantee that AI-generated text is copyrightable, or permission to relicense third-party material added later.

No GitHub branches, model weights, existing recovery runs, or external permissions were modified.
