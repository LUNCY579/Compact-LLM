# Hugging Face Cache Verification

Tool: `hf cache verify` from `huggingface_hub 1.27.0` with
`--fail-on-missing-files` and an exact revision.

| Source | Revision | Initial result | Corrective action | Final result |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | Failed: `.gitattributes`, `LICENSE`, and `README.md` absent; all model/tokenizer files present | Resumed pinned `hf download` | Passed, 10 files checked |
| Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` | Failed: `.gitattributes`, `LICENSE`, and `README.md` absent; all model/tokenizer files present | Resumed pinned `hf download` | Passed, 13 files checked |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | Passed | None | Passed, 15 files checked |

The initial failures are retained because they delimit metadata completeness;
they did not indicate corrupted or missing weight shards. Architecture evidence
also records an independent SHA-256 for every weight shard and tokenizer file.
