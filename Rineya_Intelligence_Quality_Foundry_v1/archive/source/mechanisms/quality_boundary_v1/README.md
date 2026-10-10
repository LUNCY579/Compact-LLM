# Strict metadata values candidate

This small standard-library module is an implementation candidate, not integrated
with frozen controllers, schemas or row admission. Independent review and an exact
new consumer pin are required before integration.

`strict_decode` rejects duplicate keys, invalid UTF8, BOM, nonstandard nonfinite
constants, floating overflow and surrogate text in both keys and values before
returning a decoded object. It preserves numeric/string values; it does not strip,
normalize Unicode or coerce integers/floats/booleans.

`canonical_bytes` uses the established sorted-key, compact, UTF8, no-LF JSON value
encoding over exact supported Python JSON types. `exact_identity` compares these
typed bytes, so nested `true`/`1` and `0.0`/`0` remain distinct. Dictionary insertion
order is irrelevant; array order stays meaningful. `exact_integer` and
`exact_sha256` enforce exact integer and full-string hash boundaries.

This module does not validate schema shapes, typed artifact domains, path safety,
ancestry, current revisions, permissions, CAS state, evidence applicability or
runtime/model qualification. Each consumer must still perform those checks.
It introduces no new semantic worlds or row votes.

The 21 tests are metadata fragments for the actual repeated decoder/identity
defects. Run with `python -B -m unittest -v test_strict_values.py` from this folder.
No dataset, model library or filesystem role/permission implementation is loaded.
