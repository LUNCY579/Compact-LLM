# XSC-D82-MECH-001: immutable-subject remote mechanical receipt

Subject: `d82e27c48b2a75b3a9459148990127cd9b14a2e9` on `research/ced-cross-scale-e28-d36-v1`.
Tooling lives on the separate `research/ced-cross-scale-remote-receipt-d82e27c-v1` branch.
The Actions run SHA is the tooling revision, **not** the tested subject SHA. The workflow checks out both separately and the supervisor accepts only the exact subject HEAD, clean checkout and unchanged source bytes.

## Scope

Run only the unchanged 134 XSC tests in `experiments/cross_scale_ced_v1/tests`, including both native random-Qwen cases. Pin Python 3.13.5 and all five distributions in the subject requirements. Record installed transitive distributions without changing the subject's dependency policy. Use CPU, eager native comparisons, one BLAS/OpenMP thread, offline Hugging Face settings and no model downloads. Official dimensions are meta fixtures only. The inherited tiny synthetic optimizer fixtures are not official P2 execution or evidence for training or quality.

PASS requires exact collected/JUnit case identity, 134 passes, zero failures/errors/skips, both native cases passed, consistent XML summaries, successful pytest exit, and pre/post source integrity. The supervisor has bounded child-process groups and create-only outputs. Missing runtime, checkout errors, timeouts or absent evidence block PASS. This receipt cannot be consumed as the original package's official P1 receipt. It licenses nothing.

The workflow grants only contents-read permission. It creates no tags, commits, pull requests, model publications or training jobs. Artifact upload stores runner evidence for 30 days. Download and retain a completed artifact before expiry. Source hashes and GitHub metadata are provenance checks, not cryptographic attestation against a malicious runner.

## Local handoff

Use a complete clean checkout of the exact subject and a separate tooling checkout. Install the exact subject requirements in an isolated Python 3.13.5 CPU environment. Run the validator self-tests and then:

```bash
python /path/to/tooling/experiments/cross_scale_remote_receipt_d82_v1/receipt.py \
  --subject /path/to/clean-exact-d82-checkout \
  --output /path/outside-subject/new-attempt-directory
```

A local result remains local evidence. A remote receipt requires a started runner, actual case-level evidence, and the Actions run/attempt metadata. No default-branch merge or workflow_dispatch is required; the bounded workflow triggers on tooling-code pushes to this isolated branch. Documentation/evidence-only commits do not rerun it.

## Source-review finding before execution

At the exact subject, `source.py:native_from_reference` constructs `Qwen3ForCausalLM` under `torch.device("meta")`, assigns state tensors, ties the head and returns. In Transformers v4.57.1, `Qwen3RotaryEmbedding` registers `inv_freq` with `persistent=False` and stores `original_inv_freq`. This is an identified meta/nonpersistent-buffer risk, not a measured native parity failure. Keep the two native tests unchanged and preserve any failure. Do not apply a monkeypatch or import a sibling repair while reporting that the immutable subject passed.

## Next boundary

First obtain a complete no-skip mechanical result or an honestly classified failure. If native execution confirms the RoPE issue, make a separate explicitly scoped successor repair; materialize the correct native-derived RoPE state, preserve tying and parameter sharing, assert no unresolved meta tensors, and rerun both unchanged native cases plus focused regressions. A repaired successor's result never retroactively passes d82.

Only after mechanics pass consider separately authorized official P1 source/runtime/resource qualification: exact Supa and Qwen3-4B shards/tokenizers, complete clean checkout, pinned CPU runtime, and the existing 64-GiB available-RAM guard. Do not run P1 here. Do not open P2, training, recovery, DEV/TEST, efficacy, GPU-fit, quantization or deployment. Full historical regression remains a separate evidence debt.
