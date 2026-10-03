# Reproduce or inspect without changing the delivered corpus

## Integrity only

Python 3.13.5 was used. The integrity checker requires only the standard library:

```bash
python src/verify_package.py
```

This verifies SHA-256 hashes, sizes, row counts, unique ids, message roles, world/gold agreement, all-row receipt alignment, and exact equality of combined and messages-only exports. It does not load a model or certify native fit.

## Tests and data checks

Use a separate environment with `requirements-audit.txt`. The recorded test execution was `235 passed`, no failures/errors/skips. The same author wrote the independent implementations; see the data card.

```bash
python -m pytest -q tests
```

The audit has separate semantic and classifier stages. **Use a working copy**, because audit commands write reports and checked metadata there:

```bash
python src/audit.py --root /path/to/working-copy --skip-shortcuts
python src/audit_shortcuts.py --root /path/to/working-copy
```

The all-row semantic and shortcut stages were completed separately in the delivery environment. Test outputs and actual receipts are retained. Earlier monolithic attempts were interrupted by execution limits; they are not counted as independent successful replications.

## Regenerate all rows

The generator is deterministic and requires no language model, remote calls or model assets. In a working copy, the pre-generation allocation and partition registry must be preserved.

```bash
python src/generate.py --root /path/to/working-copy
python src/audit.py --root /path/to/working-copy --skip-shortcuts
python src/audit_shortcuts.py --root /path/to/working-copy
python src/export.py --root /path/to/working-copy --replace
```

Generation overwrites candidate data and world files within that supplied root; it does not modify the delivered directory unless you pass the delivered root. Do not describe regeneration as byte-verified until the regenerated outputs are compared with the delivered hashes. Classifier results may depend on numerical-library versions. `MANIFEST.json` describes the frozen delivery, not the modified working copy.

## Native tokenizer census, still blocked in this delivery

No exact tokenizer assets or model weights are included. `provenance/TARGET_TOKENIZER_LOCK.json` contains the four required reference hashes. Use a dedicated Transformers 4.57.1 environment and a tokenizer-only directory containing exactly those native assets. Model weights are not needed.

```bash
python src/native_census.py --root /path/to/working-copy   --tokenizer-dir /path/to/exact-tokenizer   --output /path/to/new-native-census
```

The output directory must not exist. Wrong, missing, symlinked, unresolved-LFS or changing inputs are rejected. The tool uses Qwen2TokenizerFast and the exact native chat template with thinking disabled. It measures all 6,000 rows without truncation or filtering. Context-band endpoint convention is explicitly `[300,600)`, `[600,1100)`, `[1100,1600]` for this candidate; no source modification is implied.

Exit 2 is a blocked input/runtime qualification. Exit 0 means the census ran, **not** that native fit or corpus acceptance passed. Inspect every failure count and do not silently discard or truncate failed rows. Even a successful native census does not close external-bank exclusions or structural/abstention limitations.

## Safety boundaries

Never feed `worlds/`, metadata, checker explanations or bank keys to the language model as extra input. Only messages appear in `sft/` projections. These projections do not authorize training. Do not alter a frozen experiment, budget or evaluation authority merely to make this data fit.

For external exclusion, use an authorized blinded process on a content-frozen candidate; expose only aggregate overlaps and hashes back to generation. Do not copy protected benchmark rows into this directory.
