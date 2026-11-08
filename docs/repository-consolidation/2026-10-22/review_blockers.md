# Blocked consolidation PRs

These source branches remain intact and are excluded from manual retirement. The complete ready-source integrations are recorded separately. Each linked PR is a blocked draft, not an approved merge.

# [PR #33: codex/cheap-quality-abstention-20260929](https://github.com/LUNCY579/ChatBubble/pull/33)

## Status: blocked draft — do not merge

This PR makes the remaining consolidation work reviewable. The branch `codex/cheap-quality-abstention-20260929` at `351e93120c814d719b340bc4e2ddb5bb506e8b20` is retained. Its 145 added files and three inherited-file replacements need a preservation-aware review; the replacements prevent whole-branch integration.

## Confirmed blocker

These two incoming blobs were retrieved and independently matched to their Git blob hashes. Their previous central/source-base versions are valid UTF-8 JSON/JSONL with no NUL bytes.

| Path | Incoming Git blob | NUL bytes | Finding |
|---|---|---:|---|
| `docs/model-quality/audit-2026-09-15/mq-audit.json` | `893fff8c78a2371145d52039411960b9fdc76f27` | 54,851 / 95,760 bytes | UTF-8 decoding fails at byte 5 |
| `Rineya_Long_Context_Demo_v1/data/showcase_base_full.jsonl` | `273af0d114b73b083afce58c890c5be54a9fec16` | 288,136 / 324,148 bytes | UTF-8 decoding fails at byte 4,274 |

UTF-8-SIG and UTF-16/UTF-32 variants also fail, ruling out an ordinary encoding conversion. The earlier showcase file parses as 16 JSONL records.

The third replacement, `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class/boundary_setting.jsonl`, changes blob `29c387b311c4822c3417b64a38e50a6081fab9a4` to `93174c64c67dbc97d940b1f574486434528aa3a6`. Its 4,915,941-byte body could not be retrieved fully; blob retrieval reported a UTF-8 decoding error. Its original bytes and alternate encodings are unresolved, so it is not counted as independently confirmed corruption.

## Validation already performed

The isolated abstention package's 16 existing tests passed. They do not validate these inherited dataset/audit replacements. The complete source delta was inventoried. No source changes, data repairs, native model runs or historical campaign replays were performed.

## Concrete next steps

1. Recover and inspect the intended bytes of all three modified inherited files against their recorded source-base versions.
2. Decide whether those changes were intended data edits or accidental damage; document any restoration in a new source commit. Do not silently discard modified research data.
3. Re-run JSON/JSONL parsing and the existing package tests on the reviewed result; verify the full three-way union against the then-current central branch.
4. Merge with original source history preserved only after those gates pass. A separately scoped import of the 145 additions would also need an explicit review of the omitted replacements.

Keep this source branch until resolved. Its separate `codex/archive-20261004/codex-cheap-quality-abstention-20260929` alias remains a distinct manual-cleanup item because an existing archive tag preserves it.

All nine current repository workflow definitions were inspected at central `ce3d98ca2ce9c2cee91406b6c2347e54693b285f`: no inspected pull-request path filter matches this delta; native-related workflows are push/manual-only. This PR does not change workflows, settings or branch protections.

# [PR #34: codex/cheap-quality-retrieval-20260929](https://github.com/LUNCY579/ChatBubble/pull/34)

## Status: blocked draft — do not merge

Preserve `codex/cheap-quality-retrieval-20260929` at `bdc71a6d58c8c0f6353b1d3ea5291aedba55617e` for consolidation after its mandatory exact-tokenizer validation can complete. The 42-file incoming delta has been inventoried; no source files have been changed.

## Concrete validation blocker

The unchanged suite recorded **10 passing tests and one error** in `test_native_tokenizer_exact_consumed_budgets`. The error is `FileNotFoundError` for the exact tokenizer fixture. The test explicitly disallows a skip or approximate tokenizer substitute.

- Required library: `tokenizers==0.22.2`.
- Required tokenizer SHA-256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- Recorded fixture path: `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze/tokenizer.json`.
- Candidate committed Git blob: `949e1ec83f61520a25c75426edc4a43acc36f29a` (11,422,654 bytes).
- Candidate paths already present in reviewed related sources:
  - `experiments/cheap_quality_token_fit_v1/runs/token-fit-20260929-001/tokenizer_assets/tokenizer.json`
  - `experiments/data_admission_v1/runs/audit-001/tokenizer_assets/tokenizer.json`

The large original asset bytes were not obtained: the contents interface returned an empty body, the exact blob request ended without a body, and an exact pinned public raw URL returned 404. **The candidate's SHA-256 match has not been verified.** No replacement tokenizer or speculative download was used.

## Concrete next steps

1. Obtain the original asset from an authenticated repository checkout or existing trusted local asset.
2. Verify both Git blob ID `949e1ec83f61520a25c75426edc4a43acc36f29a` and SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` before using it.
3. Materialize those exact bytes at the required path inside an isolated fixture environment with the pinned library, keeping source tests and guards unchanged.
4. Re-run the complete existing suite, including `test_native_tokenizer_exact_consumed_budgets`, then recheck source/target tips and the full merge union before promotion.

Keep this source branch. Its separate `codex/archive-20261004/codex-cheap-quality-retrieval-20260929` alias is a different manual-cleanup candidate preserved by an existing archive tag.

All nine current repository workflow definitions were inspected at central `ce3d98ca2ce9c2cee91406b6c2347e54693b285f`: no inspected pull-request path filter matches this delta; native-related workflows are push/manual-only. No workflow, branch protection or repository setting is changed, and no model/training campaign is authorized by this draft.
