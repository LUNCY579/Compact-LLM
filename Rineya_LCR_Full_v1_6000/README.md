# Rineya LCR — full 6,000-row candidate

**Version 1.1.0-full-candidate · September 16, 2026 (America/Chicago)**

The actual corpus is populated: **6,000 single-turn chat rows, 6,000 synthetic scenario groups, 4,800 train / 600 dev / 600 holdout**. This is not an allocation-only package.

**Full acceptance has NOT been established.** Native tokenizer/window/band measurements and external-bank exclusions are unavailable. Strict structural separation and universal nontrivial abstention are not certified. Read `reports/acceptance.json` before using the data. No target model was loaded or trained.

## Start here

| Path | Contents |
|---|---|
| `data/train.jsonl` | 4,800 complete rows with annotations |
| `data/dev.jsonl`, `data/holdout.jsonl` | 600 rows each for data development; NOT model benchmarks |
| `data/all.jsonl` | The same 6,000 rows concatenated; not additional examples |
| `sft/*.jsonl` | Messages-only projections; no metadata, worlds or answer keys in user text |
| `worlds/*.jsonl` | Gold symbolic worlds and provenance; do NOT append to model prompts |
| `PREVIEW.md` | Sixteen complete examples taken from the delivered files |
| `DATA_CARD.md` | Composition, methods, measured results and limitations |
| `reports/` | All five requested receipts plus evidence, tests and acceptance status |
| `src/`, `tests/` | Deterministic generator, independent text checker, audits and tests |
| `provenance/`, `allocation/` | Supplied specification, asset lock and pre-generation assignments |
| `MANIFEST.json` | File sizes and SHA-256 integrity manifest |

## Measured checks

All **6,000** answers and UNKNOWN reasons were rederived by a separate text-only parser/solver, with **zero disagreements** over **85,141 parsed facts**. All named answer baselines are below 25% globally and per task. All applicable positive-evidence position checks passed, including the implemented deletion audit for 840 answerable distributed rows. The software test suite passed **235 tests**.

These results do not imply native token fit or complete corpus acceptance. In particular, a cheap selector-plus-validity rule reaches **87.5% balanced accuracy** on answerability. This limitation is retained in the reports.

## Contract and use boundary

This delivery follows the **attached v1 taxonomy and intended context-length bands**, not the different repository v2 addendum. `length_band` is an intended allocation; native token fields are `null`. Word/character counts must not be interpreted as tokenizer counts.

This is rehearsal training material by purpose, but its unclosed acceptance gates mean it is **not a qualified training input for a registered experiment yet**. Do not silently truncate it, pool dev/holdout with train, treat dev/holdout as a benchmark, or modify a frozen arm's dose/window/optimizer to accommodate it.

## Verify and reproduce

```bash
python src/verify_package.py
python -m pytest -q tests
```

The integrity verifier uses the Python standard library. Tests and audit dependencies are in `requirements-audit.txt`. `REPRODUCE.md` documents regeneration into a working copy and the fail-closed native census. The supplied tokenizer lock is not the tokenizer itself; no tokenizer or weight files are bundled.
