# Sustained state, replanning, and long context

## Three distinct properties

**Long-context selection:** locate authoritative records among many records, recover temporal order, reject nonmatching scopes/drafts/future versions, and use the selected evidence to solve a task. Route, selection, and evidence-chain families mainly emphasize this property.

**Sustained state computation:** update a state repeatedly while preserving earlier consequences. Ledger and register families carry state through dozens to hundreds of relevant events. These are not just one-hop needles with padding.

**Learn-to-plan:** select a legal future action sequence or constrained schedule/subset according to a stated objective. Prerequisite workflows require action preconditions and effects; routes require current versioned connectivity; scheduling and selection require global constraint satisfaction and optimization.

No single word count proves all three. Report them separately by family and by target-record count. Most non-replay planning worlds are small finite searches embedded in longer record contexts.

## Causal decision snapshots

Each continuous-replanning parent has four separately solvable decision views. All earlier records persist, two new approved edge updates arrive between stages, and no future record is exposed in an earlier view. The reported current location is part of the current query.

The teacher generally takes one edge of the current optimal path before the next stage. When the current task is complete or has no route, a subsequent externally assigned starting location may be used. Thus the teacher-forced snapshot collection supports causal current-state replanning and stale-plan rejection, not a claim that these are full on-policy physical navigation trajectories or optimal policies under unknown future updates.

Do not optimize all four views and then evaluate their parent's final canonical question in another split. They are the same group. Raw episode rows contain verifier fields; use the provided prompt-only exporter.

## Action-level environment

`PlanningEnvironment` provides a bounded stateful environment for prerequisite workflows. A named action changes the persistent state only when applicable; `STOP` requests task completion; `IMPOSSIBLE` requests the status decision. Wrong actions or unsupported statuses terminate with zero reward. A correct optimal terminal result receives one. Step-budget exhaustion is separate from task impossibility.

The environment has sparse terminal reward. It does not contain potential-based shaping, a policy optimizer, hidden memory compression, a native tool-calling protocol adapter, or a curriculum controller. Those should be tested as separate interventions. Reference-policy replay proves mechanics only.

## Matched diagnostics

The four long layout probes share the exact same record multiset and reference problem; only record placement changes. The short probe retains all target records while reducing distractors; it can exceed the nominal short target when the relevant task itself is long. The evidence-edit probe changes task evidence/parameters so the original reference is no longer accepted. Some edits modify multiple fields or a query; the report does not call them all minimal causal interventions.

Score each variant and paired wins/losses by parent. A system should tolerate a layout change yet respond appropriately to a task-relevant change. A large count of derived views must not be presented as a large independent sample size.

## Native-length and truncation policy

Record word counts are release metadata, not model capacity claims. IDs, field separators, decimal masks, arithmetic values, and a model's tokenizer vocabulary can substantially alter token lengths. Measure complete native templates and reserve the response channel budget before selecting a context window. Never truncate away evidence and retain the old label.

The first deployment target is not the 16,400-word stress set. Start with the short pilot, measure actual retained rows, and expand the context tier only after native mechanics and held-out behavior are understood.
