# Local runbook

Run from the root of a full, clean checkout of `research/ced-cross-scale-e28-d36-v1`. Do not use a copied package, sparse checkout or another experiment's receipt for official execution. Start every attempt in a new output directory; an existing directory is an error, not permission to overwrite.

## Weight-free mechanics

```bash
python -m experiments.cross_scale_ced_v1 inspect
python -m pytest experiments/cross_scale_ced_v1/tests/test_xsc.py -q
python -m pytest experiments/cross_scale_ced_v1/tests/test_native_optional.py -q
python -m experiments.cross_scale_ced_v1 tiny-qualify --output experiments/cross_scale_ced_v1/outputs/tiny-q-001
python -m experiments.cross_scale_ced_v1 tiny-smoke --output experiments/cross_scale_ced_v1/outputs/tiny-s-001
```

The two native random-Qwen checks explicitly skip when Transformers is absent. A mechanical suite with those skips is not official qualification. Tiny outputs always say `SYNTHETIC_PASS_ONLY` and license nothing; they cannot be promoted into P1 receipts.

## Official CPU environment and source preparation

Use a dedicated environment with the versions in `requirements.txt` and `protocol.json`; obtain the CPU Torch build from the official PyTorch CPU package index. The runner verifies exact Torch/Transformers/safetensors versions and binds the Python/platform details in the receipt. It does not install packages, change precision, pick a different attention backend or download weights.

Materialize regular, non-symlink local source files. The encoder must be the exact `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze` release, including `CORE.json` and all seven locked assets. The decoder must be `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, with all three safetensors shards, the shard index, config and four tokenizer files listed in `sources.json`. Merely having the right model name, shape or tokenizer vocabulary size is insufficient. A Hugging Face cache snapshot consisting of symlinks must be materialized to regular files first.

Do not copy checkpoint hashes from a different source release or edit the locks to make a failed audit pass. Use an otherwise idle host with at least 64 GiB **available** RAM, not just installed RAM. The official reference keeps multiple large objects during qualification; laptop BF16/4-bit inference experience is not an FP32 training/qualification certificate. Allow disk space for two ~3 GB interface checkpoints per run, in addition to source weights.

## P1, no optimization

Replace the two absolute paths below with the actual source directories:

```bash
python -m experiments.cross_scale_ced_v1 qualify --encoder /absolute/path/to/Supa-Core-1.7 --decoder /absolute/path/to/Qwen3-4B --output experiments/cross_scale_ced_v1/outputs/p1-001
```

A success produces `binding.json`, `qualification.json`, H/P active interface roundtrip artifacts and `receipt.json`. Source, checkout, runtime, protocol and all required artifact hashes are bound. Every invariant must pass on the actual official shapes. Failures preserve `failure.json` and any completed earlier outputs. Check the failure, fix its concrete cause, and use a fresh output directory. Do not repurpose or relabel tiny results.

## Separately authorized P2, exactly 25 steps per arm

P1 does not start training. Only after reviewing a valid P1 and receiving separate authorization:

```bash
python -m experiments.cross_scale_ced_v1 smoke --encoder /absolute/path/to/Supa-Core-1.7 --decoder /absolute/path/to/Qwen3-4B --p1-receipt experiments/cross_scale_ced_v1/outputs/p1-001/receipt.json --confirm-engineering-only --output experiments/cross_scale_ced_v1/outputs/p2-001
```

The current source/runtime/checkout binding must exactly match P1. Committing even a documentation change makes a new checkout identity and requires a new P1. The runner rechecks sources and native parity, runs H then P sequentially, writes append-only step logs and create-only interface checkpoints, verifies frozen weights and checkpoint replay, and exits. No resume, extra steps, outcome-selected settings, external data, DEV, TEST or deployment path exists. Failed H prevents P completion; incomplete paired results cannot be called a paired PASS.

## Continuation checklist

Preserve existing evidence, branches and source assets. Review `verification/LOCAL_VERIFICATION.json`: this implementation session did not have a full local historical checkout, Transformers, or official weights. Re-run the complete new suite in the actual checkout, the explicitly relevant historical suites, and the native checks before asking the official runner for P1. Local test success, remote commit existence and a remote CI result are different kinds of evidence. No remote CI or official execution result is supplied by this commit.
