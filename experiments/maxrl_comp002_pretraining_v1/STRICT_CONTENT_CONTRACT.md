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
