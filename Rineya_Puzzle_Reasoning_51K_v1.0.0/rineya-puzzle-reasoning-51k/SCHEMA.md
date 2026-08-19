# Dataset Schema

Each line in `data/*.jsonl` is a standalone JSON object.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Globally unique stable example ID |
| `dataset_version` | string | Generator/data version |
| `split` | string | `train`, `validation`, `test`, or `challenge` |
| `family` | string | One of the 15 reasoning families |
| `difficulty` | string | `easy`, `medium`, or `challenge` |
| `template_id` | string | Surface-language template group; disjoint across splits |
| `prompt` | string | Natural-language puzzle and output instruction |
| `reasoning_steps` | array[string] | Concise solver-derived reasoning sequence |
| `reasoning_summary` | string | Steps joined into one training-friendly string |
| `compact_state` | string | Reduced current or solved state |
| `final_answer` | string | Canonical short final response |
| `accepted_answers` | array[string] | Equivalent exact-answer variants |
| `answer_type` | string | Entity, Boolean, code, path, relationship, integer, and so on |
| `answer_normalization` | string | Matching normalization expected by the evaluator |
| `solver` | string | Symbolic/search method used to create and verify the row |
| `verification` | object | Family-specific verification statistics |
| `structure` | object | Machine-readable puzzle representation |
| `structural_signature` | string | SHA-256 of family plus structured puzzle state |
| `prompt_sha256` | string | SHA-256 of the exact natural-language prompt |
| `tags` | array[string] | Capability and task tags |

## Example

```json
{
  "id": "rineya-pr51k-code-tra-00012",
  "family": "code_breaking",
  "difficulty": "easy",
  "prompt": "...",
  "reasoning_steps": [
    "Initially, 120 codes satisfy the length and repetition rule.",
    "The first clue reduces 120 to 6 candidates.",
    "The second clue leaves one candidate."
  ],
  "compact_state": "Unique code=130",
  "final_answer": "1-3-0.",
  "accepted_answers": ["130", "1-3-0", "1-3-0."],
  "answer_type": "code",
  "solver": "mastermind_constraint_filter",
  "verification": {"solution_count": 1},
  "structure": {"...": "family-specific fields"}
}
```

The exact `structure` object differs by family. `scripts/validate_dataset.py` is the executable reference for interpreting and independently solving every structure type.
