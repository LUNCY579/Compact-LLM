# Original INFQ causal-input projection

The bounded CPU projection completed against the original, byte-committed INFQ
records. It reconstructs all 84 reported physical forward inputs and 28627 input
positions: two parity calls/838 positions, 78 repeated greedy-prefix calls/26544
positions and four generated-prefix bulk calls/1245 positions. Each entry retains
the original journal call ID, forward ordinal, row ID, repeat/step, token sequence,
mask, token hash and source-record membership.

The four selected TRAIN rows belong to two shared source parents,
`balanced_parent_023` and `balanced_parent_038`. Membership was projected from
the byte-bound640 candidate records and enrolled320 records per arm/20 parents.
The whole candidate JSONL was parsed, including its prompt/target/spec fields;
only seven identity fields were extracted. No targets were scored or emitted,
and no TRAIN token/student-array files were opened.

All 95 original raw byte commitments and 105 total selected input/source files
were checked before and after execution. The validator joins the actual403-call
journal, including exactly310 nested tensor-copy calls under checkpoint-load3,
to the serial84 forwards and durable token/control publications and cumulative
cost cursors. Five focused controls pass in authored, Root and independent
replays. The actual successful process took0.424 seconds; its final input/output
recheck checkpoint was0.125 seconds. These are cooperative wall observations,
not RSS qualification or a native performance estimate.

Two actual CPU failures remain preserved. The first assumed a modern body seal
on the original preservation receipt; the second incorrectly assumed the entire
loader journal was serial. The additive repair uses the original13-field receipt,
its fixed file hash and exact95-file map, and permits only the documented loader
nesting. Neither failure wrote an output or changed the originals.

The successful output is source-recipe reconstruction from the fixed producer
code, original prompts and durable generated traces. The original prehook did
not independently record every token vector. The projection therefore does not
claim independent per-call token telemetry, global exposure completeness,
R/S-to-arithmetic family equivalence or native admission. Other native scopes and
failed/partial exposures, authenticated owner chronology, qualified grading
providers/ratings and current native qualification remain pending. No model,
tokenizer, GPU, optimizer, scoring or protected-bank action ran.

[Preserved source, failures, actual output and reviews](evidence/INFQ_exposure_projection_cpu_20261009/ROOT_PRESERVATION.json)
provide the exact handoff. The full matched M0/M1/M2 transfer and retention
question remains unanswered under the standing owner authorization.
