# Prospective v2 counterdirection mechanics

Only v2 data.py and tests/test_data.py changed. The v1 compiler hash remains unchanged. No bank artifacts, native model calls, model teacher calls, optimizer steps or broad test run occurred.

UTF-8 decoding, AST parsing and in-memory full-corpus generation/validation passed: 2,000 TRAIN rows, 384 PILOT rows, independent teacher agreement throughout, and zero ID/family/content partition overlaps.

All 500 TRAIN program parents now contain identical worlds/programs and matched formats with opposite x/y queries. Both registers have five writes and distinct true answers; nuisance operations occur equally in both variants. Queries are balanced 500/500. A deterministic query-blind raw-answer policy must issue the same prediction for both rows, so it can get at most one row per parent correct.

| Program control | Correct / 1,000 |
|---|---:|
| Most-written destination with lexicographic tie | 500 |
| First destination | 500 |
| Last destination | 254 |
| Lexicographically first register | 500 |
| Lexicographically last register | 4 |

Every PILOT Horn row contains two facts, four rules and a seven-character query atom. Sixteen pairs change support while retaining fact cardinality and query/rules. Sixteen pairs switch the queried conclusion in an identical world. Each pair type has eight SUPPORTED-to-UNKNOWN and eight reverse directions.

| Horn control | Correct / 64 |
|---|---:|
| Two-fact classifier | 32 |
| Question-name hash parity | 33 |
| Support-atom lexical order relative to query | 31 |

These are deterministic instrument diagnostics, not model capability or generalization evidence. The name controls are incomplete. Existing domain counts, family IDs, truth solvers, output contracts and scientific/native gates remain fixed. The prior v1 data/exposure epoch remains intact.

A fresh bound-tokenizer profile and v2 exposure plan are still required; exposure-003 is bound to v1 and cannot qualify this revision. Other domains' nuisance shortcuts remain outside this repair. Independent review and broader fixtures are the next concrete checks.

Stable APIs remain available, with one additive helper: counterdirection_audit(rows). New draft dataset manifests include its diagnostics automatically.
