# Amendment 007: concrete DEV input and scorer binding

Prospective static preparation, 28 September 2026. Native E0 remains below host
RAM admission. This amendment accepts only the finite missing connection between
existing open DEV banks, their measured prompt arrays and pure deterministic
verifiers. No native generation, training, scorer calibration or source-correct
enrollment is authorized by this preparation. Existing gates and budgets stand.

## Fixed population and separation

Bind 256 acquisition rows/256 parents, 256 deterministic-retention rows/192 parents,
256 long-context rows/32 parents and 64 language DEV rows/64 parents. The total is
832 rows/544 parent groups, not independent samples or reasoning mechanisms.
Keep answerability twins and all eight length/answerability/distractor variants
together. No CAL, TRAIN_SELECTION, CHECK or FINAL payload is read or exported.
Existing receipt/registration metadata may be read without opening their other
data files. Do not resample, filter, replace or relabel any source row.

Policy rows have exactly six fields: row_id, messages, input_ids, attention_mask,
prompt_tokens and prompt_token_hash. Row IDs route records; they are never inserted
into model messages. A future collector consumes only the verified prompt IDs and
attention mask, or independently verifies equivalent message encoding. It must not
send the serialized policy record as conversational content.

For short acquisition and retention arrays, slice input_ids[:prompt_tokens] and
verify tokenized_prompt_sha256; the rest is a reference completion including EOS.
Validate the full recorded token/mask/label contract before separating it. Long
context already stores prompt-only input_ids. Language uses prompt_token_ids;
references and CAL controls are never policy inputs. Preserve all prior authored
assistant context in language messages. No truncation or fresh token estimates.

Evaluation rows retain the existing enrollment row fields and a separate
scoring_payload: schema_version=1, panel, source_row_sha256 and the complete bound
original source_row. The payload is evaluator-side evidence, never model input.
Acquisition variants use str(variant_id); retention keeps the exact source variant;
long-context variants prepend prompt_tokens to variant_id; language uses primary,
row_id=parent_id+'-dev', family=general_language and the original stratum.

Pin all allowlisted source/token/receipt/registration bytes and the exact Titan
and tokenizer/template identities. Bind bytes actually parsed, validate joins,
quotas, hashes and complete parent closures, and recheck snapshots before publishing.
Do not import tokenizer-preparation modules that mutate framework environment
variables. A new interpreter is not required for these standard-library checks.

## Primary malformed-response policy

Freeze this distinction before any native response is available. A well-shaped
wrong value is incorrect, not malformed. Malformed means an invalid required
envelope, type, shape or serialization constraint:

- Short tasks retain the existing at-most-4096-character strict JSON object with
  exactly the answer key. JSON whitespace is allowed. Duplicate keys, nonfinite
  literals, extra keys/prose or code fences are malformed.
- Arithmetic requires an integer excluding booleans. Registers require exactly
  three such integers. Finite sets require a sorted, duplicate-free integer list;
  wrong integer values remain semantic errors. Text transformations require a
  string, including an empty string when appropriate.
- Retrieval requires exactly two strings. Instruction-following requires a string.
  Answerability permits either null or a two-string list in both variants, so
  false abstention and a wrong attempted answer remain semantic errors, not format
  errors. Strings of the wrong content are incorrect rather than malformed.
- Long context uses the existing case-sensitive symbol/UNKNOWN grammar and its
  surrounding-whitespace allowance. The relaxed Answer: parser is forbidden for
  primary scoring. A valid wrong symbol or false UNKNOWN is a semantic error.

Use independent bridge/retention verifiers and the long-context visible-text
oracle and strict scorer. Check stored data references against those semantics
without reporting reference checks as model accuracy. Corrupt source, lineage,
panel mapping or dependency identity raises a host-data error; it must not become
a model malformed/error observation. Language remains input-only and its scorer
raises an explicit unqualified-grader error. Do not exact-match language exemplars.

## Dependency and evidence boundary

The callback file hard-pins the current pure bridge_data, retention_data and
long_context_tasks source hashes and checks them at every validation/scoring call.
The existing importer binds the complete callback file, covering those constants
and checks; no caller override is supplied. Publication also binds every imported
local implementation dependency. Legitimate verifier changes need a newly reviewed
and registered scorer, not silent reuse of old bindings.

Export policy inputs, evaluator sidecars and a binding receipt into a new immutable
directory. Keep runtime identity, seeds, decoding and numerical run budgets unresolved
rather than inventing them to pass the enrollment manifest schema. Different panels
may need separate later run registrations and generation caps. This static artifact
is not a native capture receipt or full enrollment manifest.

Synthetic tests may exercise correct, wrong, malformed, identity-drift, missing
variant and target-leak mutations. Do not manufacture production model responses.
Existing enrollment.py and e1_gates.py remain unqualified/no-launch. Actual E0,
capture, language grading, the 2048/4096 extension and native training remain separate
requirements. No recipe promotion or protected-bank access follows.
