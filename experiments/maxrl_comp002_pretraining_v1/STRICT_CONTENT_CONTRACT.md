# COMP-RS-001 strict versus content scoring contract

Fixed before any source model response is collected. The unmodified
`maxrl_composable_skills_v2.verifier.score` is the **strict** endpoint and
remains the binary reward. `content_scoring.py` is diagnostic only.

The content endpoint accepts one complete candidate and rechecks it with that
same verifier after normalization. Allowed surface differences are exactly:

1. A bare JSON answer list, such as `[1,3]`.
2. Bare `UNKNOWN` when the verifier proves no valid list exists.
3. One outer Markdown code fence containing a complete JSON object/list or
   `UNKNOWN`, with no surrounding prose.
4. One trailing period after a complete candidate.
5. A JSON object with an `answer` field and additional fields; only the
   `answer` value is passed to the verifier.

The parser rejects duplicate JSON keys, nonfinite numbers, oversized output,
multiple candidates, surrounding prose and malformed payloads. It does not
extract a plausible answer substring from an explanation. A content pass is
therefore a bounded formatting diagnosis, not evidence that free-form prose
contained a correct thought. The verifier still decides whether the selected
list/UNKNOWN is valid for the actual task, including alternative certificates.

Report `strict_pass`, `content_pass`, and the normalization category per row.
`strict_fail && content_pass` is a format/convention case;
`content_fail` is a task or undecidable-output failure under this contract.
Neither score is a substitute for actual RS composition-only repair counts.

## Amendment 1 addendum (COMP-RS-001 Amendment 1, section 8.4 a-d; 2026-09-26)

Fixed before any S0 read. The text above is not rewritten; where this addendum
narrows it, the addendum governs. The strict endpoint is unchanged:
`corpus.reward` for every row, which is the composition verifier on composition
rows and the legacy verifier on legacy rows. Every rule below can only make
content stricter or add a diagnostic label; none changes a strict outcome.

### A1. Rejection clause for extra object fields (8.4 b)

Rule 5 allows extra fields, and the parser rejects multiple candidates. The two
are reconciled as follows. An answer object fails content, with normalization
`multiple_candidates_in_object`, when any non-`answer` field value contains, at
any depth (through nested objects and lists), a composition candidate:

- a non-empty list whose elements are all ints (bools are not ints); or
- the string `"UNKNOWN"` (case-sensitive).

Only values are inspected, never field names. Every other extra field remains
allowed: `"reason":"ok"`, a scalar int, an empty list, `[true]`, a list of
strings. The rule is applied after the code fence and trailing period are
removed. Fixtures: `{"answer":[1],"alt":[2]}` fails; `{"answer":[1],"reason":"ok"}`
passes.

### A2. Legacy content path (8.4 a)

`score_legacy(family, problem, response)`:

- **strict** = `maxrl_verifiable_data_v1.verifier.score(family, problem, response)`
  on the raw response.
- **content** = the same bounded normalisations, re-scored by the same legacy
  verifier on the compact `{"answer": value}`:
  1. one outer Markdown code fence (opening line exactly three backticks,
     optionally followed by `json`), with no surrounding prose;
  2. one trailing period;
  3. an answer object with extra fields, subject to the legacy candidate rule
     below;
  4. a bare JSON array, only for the seven families whose answer is a JSON
     array: subset_sum, linear_constraints, cnf_assignment, shortest_path,
     precedence_order, finite_program, state_tracking.

Not admitted on the legacy path:

- bare `UNKNOWN`: no legacy family accepts it and the legacy prompt does not
  offer it;
- any bare countdown form (a quoted JSON string, an unquoted expression, or an
  array). The countdown answer is a string, and the section 8.5 canary
  registers "bare list" as the only bare wrapper. Admitting a bare string would
  add a sixth normalisation that the amendment did not register.

**Legacy candidate rule** (A1 adapted to legacy answer types). An extra field
holding, at any depth, a candidate for the family fails content as
`multiple_candidates_in_object`:

- array families: a non-empty int list (bools excluded);
- countdown: a string that the countdown grammar admits with at least one
  operator. The grammar is int leaves 1..100, binary `+ - * /` and
  parentheses, at most 512 characters, as in the verifier's
  `expression_value`. The number multiset is not checked.

An int list in a countdown object is not a candidate, and neither is an
expression string in an array-family object. `"UNKNOWN"` is not a legacy
candidate.

The parser limits are unchanged: at most 4096 characters, and duplicate keys and
nonfinite numbers are rejected. Strict success implies content success, and this
is asserted.

**Basis.** The legacy verifier's `parse_answer` uses the same exact
`{"answer": ...}` envelope. Every guard_dev reference is a compact
`{"answer": ...}` object. The countdown answer is an expression string, and the
other seven families answer with int arrays (state_tracking values may be
negative).

### A3. Row dispatch

`score_row(row, response)` dispatches on `row['kind']`, which must equal
`row['verifier']['kind']`:

- composition: `score(spec, response)`;
- legacy: `score_legacy(spec['family'], spec['problem'], response)`.

It asserts `strict_pass == bool(corpus.reward(row, response))`.

### A4. Strict-failure subtyper (8.4 c)

`failure_taxonomy.subtype_strict_failure(row, response, stop_reason)`.

**Scope.** A strict failure whose content normalization is
`unparseable_under_content_contract`, or whose parsed candidate does not have
the contract's answer type. The answer types are:

- composition: a non-empty int list or `"UNKNOWN"`;
- countdown: a string;
- the other legacy families: a non-empty int list.

Every other response returns None, because its normalization already names it.
That covers strict passes, format-only passes, well-typed wrong answers, and
`multiple_candidates_in_object` objects whose `answer` is well typed. An empty
list is never an int list anywhere in this addendum.

**Types** (first match wins):

1. LENGTH_CAP: `stop_reason == 'length'`.
2. EMPTY: the response is blank.
3. STRING_ENCODED_ANSWER: the answer value is a string that parses as a JSON
   array or object. The answer value is the `answer` field when the text is a
   single JSON object, or the value itself when the text is a single bare JSON
   string.
4. NON_INT_ELEMENTS: the answer value is a list with an element that is
   neither an int nor an alias key.
5. ALIAS_STRINGS: the answer value is a list with at least one element that
   is a key of `spec['lookup']['links']`, and no other non-int element. This
   applies only to composition rows with a lookup.
6. KEY_VARIANT: the text is a single-key object whose key casefolds to
   `answer` but differs from it, or is one of `answers`, `result`, `output`,
   `final_answer`.
7. PROSE_WRAPPED: the text is not a whitespace-separated sequence of JSON
   values, and a parseable JSON array or object starts inside it.
8. MULTI_OBJECT: the text is a whitespace-separated sequence of two or more
   JSON values.
9. OTHER.

"The text" is the response after the content wrappers are removed, or the raw
stripped response when the fence is invalid. Types 3 to 8 detect shape with the
standard permissive JSON decoder, which tolerates duplicate keys and NaN. The
subtyper never changes a score.

### A5. RS response classifier (8.4 d)

`failure_taxonomy.classify_rs(spec, response)`.

**Scope.** Only specs with `lookup` and `selection` and with neither `updates`
nor `precedence`. The classifier uses the content-contract candidate, so
wrappers do not hide R use.

**Classes** (in precedence order):

1. UNPARSEABLE: there is no single content-contract candidate. This includes
   `multiple_candidates_in_object`.
2. UNKNOWN_ANSWER: the candidate is `"UNKNOWN"`.
3. NOT_INT_LIST: the candidate is not a non-empty list of ints (so `[]` lands
   here).
4. RS_PASS: content pass.
5. FULL_RESOLVED_SET: `set(answer)` equals the resolved set from
   `verifier.concrete`.
6. RESOLVED_WRONG: `set(answer)` is a subset of the resolved set and the
   answer does not pass (duplicates are allowed here).
7. R_BLIND_S_VALID: the answer is not a subset of the resolved set, has k
   distinct in-range indices, and its initial values sum to the target.
8. OTHER_WRONG.

An unresolvable lookup (a cycle or a missing alias; boundary rows only)
resolves to the empty set, so classes 5 and 6 cannot match there.

### A6. Fixtures and canary coverage

`tests/test_s0pre_content_taxonomy.py` holds:

- one positive fixture per normalization (composition and legacy), per subtype
  and per RS class;
- the two 8.4 b fixtures;
- a property check over all DEV core R, S and RS oracle references (750 rows,
  2,250 references). Every reference scores strict 1 and content 1. The code
  fence, trailing period, bare list and a non-candidate extra field each score
  strict 0 and content 1, with the expected category. Bare UNKNOWN never applies,
  because no core reference is UNKNOWN.
- on DEV core RS: the full resolved set classifies FULL_RESOLVED_SET. The
  lexicographically first lookup-blind sum-valid subset outside the resolved
  set exists on 250/250 rows, and it classifies R_BLIND_S_VALID.
- all guard_dev legacy references (8 families, 2,419 references) with their
  applicable wrappers.

### A7. Clarifications C4 to C6 (Amendment 1 section 15; 2026-09-27, pre-data)

Registered before any model output exists. Where A1 to A6 say otherwise, A7
governs. No strict or content outcome changes. C6 changes only which diagnostic
label a row carries.

**C4: one definition of "int list".** `content_scoring.is_int_list` is the
package's only definition. An int list is a non-empty Python list whose elements
all have type exactly `int`. Bools, floats, strings, nested lists and int
subclasses are not ints, and a tuple is not a list. Every use of the term in
this addendum and in the Amendment's H3b calls that function; no module
restates it:

- the A1 and A2 candidate rules (an extra field holding an int list fails
  content);
- the A4 answer types (`failure_taxonomy.well_typed`);
- A5 class NOT_INT_LIST;
- H3b's candidate and e (`stats_exact`).

These consequences were already in force:

- `{"answer":[1],"alt":[]}` passes content, because `[]` is not a candidate.
- `{"answer":[]}` is NOT_INT_LIST, not RESOLVED_WRONG, although `[]` is
  vacuously a subset of the resolved set.
- The subtyper treats `{"answer":[]}` as ill typed (OTHER).
- e([]) = 0 under either reading.

The verifiers' range-checked predicates (`contracts.intlist` and the legacy
`specs.ints`) belong to the unchanged strict endpoint and are a different thing.
A test parses every module of the package and fails if a second int-list
predicate is defined.

**C5: H3b's candidate is the content-contract candidate.** H3b takes its
candidate from `content_scoring.parse_candidate(response)` under the composition
contract. This is the parse `classify_rs` uses, so the code fence, trailing
period, bare list and non-candidate extra fields cannot hide lookup use. The
candidate is None, and e = 0, for:

- a multiple-candidate object;
- `UNKNOWN`;
- an unparseable response;
- an answer that is not an int list.

These are exactly the responses that A5 puts in UNPARSEABLE, UNKNOWN_ANSWER or
NOT_INT_LIST, and a test checks this on every RS fixture. `parse_candidate`
suffices, so `content_scoring` has no second candidate helper.

**C6: multiple-candidate rows get no subtype.** A row whose content
normalization is `multiple_candidates_in_object` is named by that category and
gets no subtype. This holds whatever the type of its `answer` value and whatever
the stop reason.

This narrows A4's scope. Its second disjunct ("whose parsed candidate does not
have the contract's answer type") no longer reaches multiple-candidate objects.
Its phrase "`multiple_candidates_in_object` objects whose `answer` is well
typed" now reads "all `multiple_candidates_in_object` objects". The subtyper
already returned None when the `answer` was well typed. An ill-typed `answer`,
however, fell through the type ladder, so the row carried two labels:

- `{"answer":null,"alt":[2]}` was OTHER (LENGTH_CAP when capped);
- `{"answer":"[1]","alt":[2]}` was STRING_ENCODED_ANSWER;
- `{"answer":[1.0],"alt":"UNKNOWN"}` was NON_INT_ELEMENTS.

All three now return None.

Section 8.4(c) says each strict failure outside the five normalisations gets
exactly one type. That sentence counts the 8.4(b) rejection clause as part of
the normalisation layer, so `multiple_candidates_in_object` is itself the row's
one label. Scorer receipts count these rows in
`strict_failures_without_subtype`, with format-only passes and well-typed wrong
answers, and the `normalization` counts separate them.

**Precedence pins.** Some fixtures match two rules at once. There,
`tests/test_s0pre_content_taxonomy.py` pins the registered order and each
defining condition:

- A4: `"   "` with `stop_reason == 'length'` is LENGTH_CAP, not EMPTY.
- A4: each of the four key variants (`answers`, `result`, `output`,
  `final_answer`) has a KEY_VARIANT fixture, and the registered set is asserted
  exactly.
- A5: `{"answer":"UNKNOWN","alt":[1]}` is UNPARSEABLE, because class 1 precedes
  class 2.
- A5: on the RS fixture with k = 4 and target 24, the full resolved set
  `[0,1,3,4]` passes and is RS_PASS, because RS_PASS precedes
  FULL_RESOLVED_SET. `validate()` admits k up to 5, so the two can co-occur,
  although the DEV generator (k in {2, 3}) never produces the case.
- A5 class 7, each condition in turn:
  - `[0,2]` has k distinct in-range indices outside the resolved set but sums
    to 10, not 12, so it is OTHER_WRONG. So is `[2,5,5]`.
  - With target 19, `[2,5,5]` sums to the target but has length 3 for k = 2:
    OTHER_WRONG.
  - With target 10, `[2,2]` has length k and sums to the target but has one
    distinct index: OTHER_WRONG.
