# Tokenizer fit and training-signal audit

Study 5 prepares 128 deterministic TRAIN parents, 32 in each of four bridge
families, and audits both unchanged public prompt renderers against identical
independently verified JSON targets. It measures direct nonthinking native chat
serialization under total context profiles 512, 1024 and 2048, each with a
128-token completion reserve: 256 rendered rows and 768 fit decisions.

Read [PROTOCOL.md](PROTOCOL.md) and the prospective [config.json](config.json).
Run sequentially through the campaign hardware supervisor: 512 MiB process-tree
RSS, 900 seconds and 50 MiB total study outputs. Use the pinned tokenizer-only
interpreter containing Transformers 4.57.1 and tokenizers 0.22.2.

Supervised child commands, from this worktree's repository root:

```powershell
<python> -B -m unittest discover -s experiments/cheap_quality_token_fit_v1/tests -v
<python> -B experiments/cheap_quality_token_fit_v1/audit.py --output experiments/cheap_quality_token_fit_v1/runs/token-fit-20260929-001
```

The script freezes exact source/asset/config/runtime/generated-row identities
before measuring. It sets USE_TORCH/TF/FLAX=0 and offline controls before tokenizer
imports, loads inherited pure bridge/serialization files directly without their
packages, and imports no model. Only four pinned tokenizer assets are copied into
the new run; tokenizer loading sees that asset-only directory and cannot discover
model configuration or weights. No source tokenizer file is changed.

Successful outputs include freeze.json, parents.json, generated_train.jsonl,
tokens.jsonl, fit.csv, pairs.csv, family.csv, summary.json and report.html.
Failures preserve failure.json and exit nonzero. Existing result directories are
never overwritten. Actual hardware measurements are in the enclosing supervisor
receipts. Tests use synthetic tokenizer fixtures; they do not load model packages.

Results establish serialization/context suitability and cost in tokens only.
Neither cheap rows nor verified targets establish learnability, native model
qualification, model intelligence improvement or training authorization. All
CHECK/FINAL quotas are zero and their parents are never rendered or solved.
