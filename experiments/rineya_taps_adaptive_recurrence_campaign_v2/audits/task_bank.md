# Bound DEV/SMOKE task-bank audit

Status: **partial prompt-only positive controls verified against the pinned scorer**.
No model, tokenizer or v1 native entrypoint was executed. No new task bank or
protected dataset was opened. The only task surface read was the already-admitted
runs/binding-001/PROMPT_AUDIT.json and its binding/receipt metadata.

Final source: solver_audit.py SHA256
fc93dbaec6b275684f43fa5be088fb07adefe3f5e2b9b4ed448f275b41f3a48c.
Final run: runs/solver-audit-003, exit0.
Python3.11.15 at C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe,
with -B and no environment mutation. The solver imports standard-library code only.
The admitted scorer is loaded from its exact source path after source hash/blob checks;
it has no model dependency.

## Positive-control coverage

| Surface | Bound rows | Applicable structured rows | Correct / applicable | Format correct / applicable |
|---|---:|---:|---:|---:|
| DEV |16|13|13/13|13/13|
| SMOKE |8|5|5/5|5/5|

These are partial positive controls, not13/16 and5/8 model accuracy claims.
The remaining6 factual/language rows are explicitly NOT_APPLICABLE with null
prediction/score: houseplant consequence, oxygen-carrying cells, DEV greeting,
ice melting, SI electric-current unit, and SMOKE greeting. No target fallback
supplies an answer to unrecognized tasks. All human-review fields remain NOT_REVIEWED.

The solver receives one raw prompt string only. It receives no case ID, split,
family, target, scorer, format scorer or tokenizer metadata. Parsing yields
witnesses and predictions; only then does an external audit function evaluate
the prediction using the pinned inherited scorer. The structured coverage is:

- Transitive color implication: reachability proves N is ochre from yellow→coral→ochre.
  The bounded positive-only solver refuses a query it cannot prove; it does not
  reinterpret absence of proof as a proved negative.
- Age order: transitive closure finds the unique youngest and rejects a cycle.
- Token/map handoffs: each giver must be the current holder, and the query item
  must match the tracked object.
- Signed multiply-plus and three subtraction paraphrases: integer operations use
  prompt operands only, with no eval or executable expression.
- Access-code lookup: exact queried unit matching against parsed context.
- Both archival source-position cases: exact target matching against all100 records,
  with witnesses identifying records95 and50.
- JSON echo and first letter: deterministic transformation of the visible prompt.
- Two missing-code cases: no queried code exists in the fully parsed closed context;
  UNKNOWN comes from that prompt's explicit missing-evidence instruction. It is
  a prompt contract, not a scorer/gold fallback.

Seven CPU self-checks cover signed arithmetic, unknown-language refusal, duplicate
JSON-key refusal, cyclic age refusal, invalid handoff refusal and rejecting a
dictionary containing a scorer. These are apparatus fixtures, not new model tasks.

## Structural floors and source/query cues

All attackers receive prompt text only. Context attackers remove a recognized
query boundary and receive empty context when that separation is unsupported.
They do not receive target fields. Constant/literal/full-prompt copy predictors
are separately named. Report each score beside its own surface; do not combine
the bank denominators or choose the best attack separately for each item.

| Fixed diagnostic | DEV correct /16 | SMOKE correct /8 |
|---|---:|---:|
| Copy full prompt |3|2|
| Last capitalized name in context |2|1|
| Unique explicitly marked target record, query removed |2|0|
| First code in context, query removed |1|0|
| First number in context |1|0|
| Last number in context |0|0|
| Copy last answer literal |1|1|
| Constant YES |1|0|
| Constant UNKNOWN |1|1|
| Constant NO /0 /1 /empty |0 each|0 each|

The strongest tested single aggregate floor is therefore3/16=.1875 for DEV and
2/8=.25 for SMOKE. This is a retrospective diagnostic maximum over the declared
attacker list, not an optimal shortcut bound or a deployable selected policy.

The aggregate floor hides important strata. Both archival cases contain one
record literally named "the Zorin target", so a query-blind marker copier scores
2/2. Their common queried value is128448 and the record position changes95→50.
A native gain on this pair would establish bounded retrieval/output behavior,
not reasoning robust to query identity or long-context generality. The two rows
are related, even if provisional scenario IDs differ.

Age and handoff answers are the final named entity in their evidence. The three
subtraction paraphrases share operands84 and7 and answer77. Treat related prompt
variants as dependent units. JSON echo legitimately exposes its requested output;
first-letter extraction is a simple visible-string task. Both unanswerable prompts
print UNKNOWN in their response contract, and constant UNKNOWN solves that stratum.
These are valid response-contract tasks, but they cannot alone prove semantic
abstention or complex reasoning.

## Pinned scorer vulnerabilities reproduced

The original scorer was preserved unchanged. Attacks generated after prompt-only
solving reproduce these limitations in the final rows journal:

- Choice: "NO YES" receives correctness1 because the last recognized allowed token
  is YES. Strict format rejects it.
- Keywords: "The answer is not Dina.", "not Bex.", "not Faye." and "not UNKNOWN."
  receive correctness1 because keyword matching does not interpret negation.
  Strict format rejects them.
- Numeric: "The answer is not201." and corresponding computed/retrieved numbers
  receive correctness1 because the last parsed number matches. Strict format
  rejects them.
- JSON: replacing visible true with integer1 still receives correctness1 AND
  format1 because Python JSON object equality equates True and1.

Thus correctness must remain beside format validity, raw outputs and a distinct
semantic review. The strict format scorer closes the illustrated choice,
keyword and numeric attacks, but does not close the JSON boolean/type issue.
No inherited scorer was rewritten, and no prospective endpoint was silently changed.

Machine parser/solver verification and agent semantic interpretation are different
statuses. This audit independently computes recognized answers from visible text.
Its interpretation of cue/negation limitations is an agent review of exposed
development material; no human review or broad bank answerability certificate exists.

## Rendered prompt and token-ID audit

The runner verifies the exact raw/rendered prompt hashes and equality to the bound
user/assistant empty-thinking-stub template. It checks that token IDs are nonnegative
integers, their length equals the bound input count, the complete forward-prefix
ceiling equals input_tokens+max_new_tokens−1, and that ceiling stays within4096.
There is no declared truncation. IDs are reused from the earlier exact tokenizer
binding; this run does not retokenize or claim a fresh tokenizer decode check.

The two archival inputs each have2237 input tokens and maximum2248 forward-prefix
tokens. Their input lengths are equal despite source positions95/50. A separately
qualified maximum-shape native check remains upstream of inference.

## Evidence bindings and reproduction

| Artifact | SHA256 |
|---|---|
| binding-001/PROMPT_AUDIT.json |f4b5b21638bc2d8a8da4fac707072208d3a8a1aa2f1918c7574a326cc996bf7d|
| binding-001/MODEL_DATA_BINDING.json |82590a8f76eca27e7b15694b724123477e21ae0cc38ed2b33b56641ab9e70f26|
| inherited scoring.py |5a22e1c1ff7b0a56623931c0782f81dd2b3a9df83ccfe77ce3724d37cf320086|
| solver-audit-003/summary.json |c4be0cf29e6d483f3539c9f9943d36f938985b8934210052b8b92e8dc3aec8cd|
| solver-audit-003/rows.json |6a50bc043d1747bc7e38a3835205e07ca8f0e9e30b4a66d28f5b303c7f9771c0|

The scorer was byte-checked against integrated-base commit
03db218ceebc61a2bc04f2482cf7e094045e01c1. The original binding receipt admits the
two input metadata files by their hashes before any row is inspected.

Reproduction, with a new exclusive campaign-local output directory:

    & 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m experiments.rineya_taps_adaptive_recurrence_campaign_v2.solver_audit --out '<exclusive-campaign-run>'

solver-audit-001 was an auditor implementation failure: its default Windows text
encoding decoded the multiplication sign incorrectly and the raw-hash guard refused.
The explicit UTF8 repair changed only this successor audit. solver-audit-002
completed successfully. solver-audit-003 additionally restricts context-only
attackers to recognized query separation and verifies the exact rendered template
and complete prefix shape. Old runs are preserved; no inherited bytes changed.

## Required review before use

The independent third agent must review solver purity, witness checks, scorer binding,
floor inputs and interpretation. This author does not grant independent-review PASS.
No solved case becomes a native model result. The existing native screen remains a
small exposed diagnostic: its family preservation and response contracts matter,
but these partial controls do not establish broad task discrimination or confirmatory
power. Stronger structural/semantic task coverage requires a separately registered
open extension.

