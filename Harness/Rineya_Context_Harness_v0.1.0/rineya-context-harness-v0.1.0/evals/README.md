# Evaluation Assets

`cases/rsf_trials.jsonl` is a small contract/metric smoke set, not a model benchmark. It includes one deliberate premature-deletion case so deletion regret is non-zero.

Run:

```bash
rineya-harness evaluate --cases evals/cases/rsf_trials.jsonl
```

A full study should instantiate the frozen baseline conditions described in `docs/evaluation/RSF_EVALUATION_PROTOCOL.md`.
