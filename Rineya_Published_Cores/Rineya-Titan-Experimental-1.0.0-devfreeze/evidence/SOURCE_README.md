# Rineya Supa RRP-015 Control, seed 17 — research candidate

Preserved experimental model; not a published core or release-qualified replacement.

Starts from exact Supa 1.7. Trained on 384 verified Forge RL v2 transition examples,
excluding both scored-parent instruments. Assistant-response plus EOS; 28 updates,
54,172 processed tokens, FP32 training of blocks 8–27. BF16 export.

Measured with FP32 DenseReference evaluation:

| Endpoint | Supa | This candidate |
|---|---:|---:|
| Selection transition skill /384 | 14 | 234 |
| Arithmetic, permissive /150 | 47 | 69 |
| Arithmetic, strict /150 | 41 | 62 |
| Source arithmetic answers retained /47 | 47 | 39 |
| Long context /256 | 161 | 162 |
| Exact UNKNOWN on unanswerables /64 | 13 | 10 |

Passes the limited experimental Gate v2. Both strict/permissive legacy per-class
retention monitors still fail. One training draw does not establish reliable
preservation. The long-context suite is spent and report-only; net +1 hides 11
old successes lost and 12 gained. Abstention still regresses. Native BF16 serving
and broad independent confirmation have not been qualified.

CUSTODY.json binds these bytes to the full retained training checkpoint and source.
PRESERVE_RESEARCH_CANDIDATE.json marks this export for preservation. Keep the full
training states and unsuccessful rehearsal-arm evidence as well. No automatic
promotion, registry update or publication was performed.
