# Selected DEV runtime boundaries

Status: **SELECTED_DEV_RUNTIME_BOUNDARIES_CAPTURED**

- Complete unique rows: **16,200**
- Reciprocal-rank block begins at line: **12,151**
- Legacy stable full-sort rows end at line: **12,238**
- Proven-equivalent stable partial top-k begins at line: **12,239**
- Post-example CUDA cache release begins at line: **12,765**
- OOM retry telemetry begins at line: **12783**
- Stored examples with OOM retry: **0** (0 total retries)
- Archived supervisor attempts: **19** (2 failed; 14 clean batch returns; 2 without recorded terminal event)

The two ranking implementations return exactly identical stable top-k indices under the source-bound parity artifact. Memory cleanup and bounded worker lifetime do not alter logits or scoring. A failed allocation attempt is never appended; only a successful exact recomputation can become a result row.
