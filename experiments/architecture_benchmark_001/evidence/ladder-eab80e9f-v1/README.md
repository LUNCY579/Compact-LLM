# Fresh official-source ladder at implementation `eab80e9f` (first post-HOTFIX-001 execution)

This is the **first** gate-ladder execution bound to the post-hotfix implementation. Every
receipt here is schema 2 and carries
`implementation_sha256 = eab80e9f99494268a113fba381f4178818aab759e3acbc6193ea936d33560fe6`,
verified equal to `implementation_digest()` in this checkout.

It supersedes nothing by deletion: the earlier generations remain untouched (the
`c5b952d8` ladder in git history at commit `2f65f9e`, the `dc212c9f` ladder in the sibling
`evidence/g*-official-supa-1.7.0/` directories). Those are schema 1 and, by
`gates.py`'s own rule, are historical evidence that cannot qualify the current code.

## Runtime

    python 3.11.15 | Windows-10-10.0.26200-SP0 | torch 2.10.0+cpu
    transformers 4.51.3 | safetensors 0.7.0 | numpy 2.3.5 | tokenizers 0.21.4
    device cpu | precision float32 | backend eager

Built in an isolated environment at `C:\Users\clclc\BusinessProject\.venv-rab001`, pinned to
`requirements-native.txt`. The shared `Chat-Bubble/.venv` had drifted to
`torch 2.12.1+cu126 / transformers 5.14.1`; `run.py:117` refuses to run any gate unless
`transformers == 4.51.3`, so the pinned environment was required, not optional.

Full suite in that environment: **181 passed, 0 skipped** (no native skips).

## Result

| stage | status | checks |
|---|---|---|
| G0 | PASS | 8/8 |
| G1_IDENTITY | PASS | 5/5, `max_abs_logit_error = 0.0` |
| G1_WIDTH | PASS | 7/7 |
| G1_CED_C28 | **FAIL** | 12/15 |
| G1_CED_C40 | **FAIL** | 12/15 |

Both CED arms fail exactly `strict_tail_only_parity`, `strict_chunked_cache_parity` and
`causal_prefix_parity`, and nothing else. All structural checks pass on both arms, and
`decode_choices_exact` passes on both — which is precisely why the pre-hotfix
argmax-plus-global-scale rule wrongly recorded these arms as PASS.

`causal_suffix_isolation_exact` is True for every row of every arm. **There is no causality
defect**; `causal_prefix_parity` fails only because it conjoins that (true) result with the
prefix numerical comparison.

## Checkpoint binaries

`*.safetensors` is gitignored, so the converted checkpoints are not committed. They are
retained on disk under `outputs/ladder-eab80e9f-v1/<STAGE>/attempt-0001/`, including both
**failed** CED checkpoints, which are kept deliberately for diagnosis. Artifact hashes in
the receipts therefore verify only against that directory, not against this one.

## Known reporting gap in `audit-evidence`

Running `audit-evidence --root` over these receipts reports
`recorded_strict_parity_failures: {}` for both failing CED arms. `review.py:28` reads
violation counts from `metrics.cache_order_parity`, a key only the legacy schema-1 CED
receipts carry; the schema-2 writer does not emit it. The failures are visible in
`checks` and in each stage's `numerics.json`, but not in the audit summary. Fixing
`review.py` changes the implementation digest and would invalidate these receipts, so it is
queued as the first action of the next cycle (fix, then requalify, then proceed).
