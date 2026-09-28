# prepared-004 evidence is intentionally partial

Only `DATA_CONTRACT.json` is stored here. The four data artefacts

- `microfit-train.jsonl`
- `REVIEW_PACKET.jsonl`
- `SOURCE_BINDINGS.json`
- `TOKENIZER_SURVEY.json`

are **byte-identical** to the already-committed `../prepared-003/` copies, so they are not
duplicated into git. Verified SHA-256 (both directories):

| file | sha256 |
| --- | --- |
| microfit-train.jsonl | `ccd5d04170935874b567961dff4c1ae29771eea203cbc4f4373181b5403f3b2d` |
| REVIEW_PACKET.jsonl | `6bcea7cb74e4ed43...` |
| SOURCE_BINDINGS.json | `e7f9bb8f0cea696b...` |
| TOKENIZER_SURVEY.json | `d3aeb0eb783b6a9c...` |

`DATA_CONTRACT.json` differs from prepared-003 solely in `preparation_code`, which additionally
covers `engine.py`, `overlap.py` and `preflight.py` — files that existed by the time preparation
was re-run at 23:46 local. No training byte changed. `MICROFIT_PLAN.json` and its digest
`9208fc92...` are identical across preflight-001 and preflight-002.

See `CLAIM_LEDGER.json` CL-001 and CL-002.
