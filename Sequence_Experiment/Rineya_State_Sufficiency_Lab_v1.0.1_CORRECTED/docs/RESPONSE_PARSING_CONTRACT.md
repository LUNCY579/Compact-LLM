# Response Parsing Contract

## Stateful target schema

```json
{
  "next_reasoning_step": "string",
  "updated_compact_state": "string",
  "done": false,
  "final_answer": null
}
```

At a final transition, `done` is `true` and `final_answer` contains the answer. The updated state should summarize the finished state rather than replay the whole trajectory.

## Accepted structured routes

The parser attempts, in order:

1. exact JSON;
2. fenced JSON;
3. balanced JSON object embedded in surrounding prose;
4. trailing-comma repair;
5. `ast.literal_eval` for a dictionary with string keys.

No model output is executed. General `eval`, shell execution, dynamic imports, and code interpretation are prohibited.

## Key aliases

The lenient structured parser recognizes aliases such as `next_step`, `next_state`, `complete`, and `answer`, while preserving the route and warnings. Alias recovery does not make a response strict-contract compliant.

## Final-answer routes

Supported explicit routes include:

- JSON `final_answer`, `answer`, `result`, `final`, or `output`;
- `<final>...</final>`;
- `<answer>...</answer>`;
- `Final answer: ...` and related labels;
- conclusion sentences such as `Therefore, the answer is ...`;
- LaTeX `\boxed{...}`.

A last-line or numeric-tail fallback is lenient only.

## Conflict policy

If two explicit final-answer candidates disagree after normalization, strict extraction fails with `CONFLICTING_FINALS`. The evaluator does not guess which answer the model intended.

## Stateful JSON with null final

A JSON transition object containing `"final_answer": null` is not reinterpreted as a final answer merely because the next-step text contains numbers. This prevents premature rollout termination.

## Raw preservation

Every response is stored verbatim, along with:

- extraction route;
- repair flag;
- candidate list;
- conflict/refusal flags;
- strict and lenient candidates;
- warnings;
- backend error and latency.
