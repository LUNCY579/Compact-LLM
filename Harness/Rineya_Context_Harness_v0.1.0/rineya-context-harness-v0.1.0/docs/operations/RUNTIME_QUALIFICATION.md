# Runtime Qualification Matrix

Complete this matrix for every release candidate.

| Area | Required evidence |
|---|---|
| Model revision | immutable local path or repository revision |
| Tokenizer | exact revision and chat template hash |
| Thinking mode | verified disabled for control outputs |
| JSON schema | valid query/reconstruction/answer rate |
| Stop behavior | no truncated JSON or runaway generation |
| Context limit | tested maximum with output reserve |
| Latency | prefill, model stage, fetch, end-to-end |
| Memory | idle, peak prompt, repeated run |
| Cancellation | model and HTTP request stop cleanly |
| Crash recovery | checkpoint and database reopen |
| Windows | exact OS/build and GPU driver |
| Quantization | rerun harness evals, not file-size assumption |
