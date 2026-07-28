# Phase 4B calibration data

Place the non-sealed long-form attribution calibration file here as:

```text
long_form_calibration.jsonl
```

Each JSONL record must contain a `messages` array ending in an assistant target. Include both:

- compact product-style replies, generally 1–24 tokens;
- genuinely longer assistant continuations with enough tokens to populate the configured late-position window.

Example:

```json
{"id":"short_001","messages":[{"role":"user","content":"I fixed it."},{"role":"assistant","content":"Nice!! What was wrong?"}]}
{"id":"long_001","messages":[{"role":"user","content":"Explain the full debugging process."},{"role":"assistant","content":"First, reproduce the failure consistently. Then record the exact input, stack trace, environment, and last known working revision..."}]}
```

Do not copy sealed Phase 8 records into this folder. The profile is used to propose pruning candidates, so leakage would directly influence architecture selection.
