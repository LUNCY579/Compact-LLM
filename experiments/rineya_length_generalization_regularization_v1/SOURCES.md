# Source ledger

## Primary upstream research

* Paper identity from the supplied screenshot: Pavlo Vasylenko, Matthias
  Lindemann, Andre F. T. Martins and Marcos Treviso, **Length Generalization Needs
  Proper Regularization**, arXiv:2610.04518v1, 3 October 2026.
  https://arxiv.org/abs/2610.04518v1
* Author implementation, pinned commit:
  https://github.com/deep-spin/len-gen-regularization/tree/d9749d54347f585abfd471d9a71015ac25562780
* Inspected numerical authority:
  https://github.com/deep-spin/len-gen-regularization/blob/d9749d54347f585abfd471d9a71015ac25562780/dropout/vpad_core.py
  Blob `294a6741f57e9092d4a4f24af1785071e15cb5c6`.
* Placement reference:
  https://github.com/deep-spin/len-gen-regularization/blob/d9749d54347f585abfd471d9a71015ac25562780/dropout/norms.py
* Upstream license: GNU GPL version 3, `LICENSE` at the same commit.

The implementation session re-fetched the pinned author core through GitHub.
The arXiv/web fetch was unavailable; no new full-paper quantitative claims or
independent paper reproduction are made in this branch. The preceding discussion
is motivation, not a replacement for these pinned implementation sources.

## Architecture reference and benchmark follow-up

* Qwen3 Transformers reference (query/key normalization versus block-entry norms):
  https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
* Qwen3-0.6B configuration, a reference only, not native Rineya authority:
  https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
* RULER: https://github.com/NVIDIA/RULER
* HELMET: https://github.com/princeton-nlp/HELMET

Benchmark sources are follow-up integration targets, not bundled/scored datasets.
Pin their revisions and review licenses before adopting data or scorer code.

## Rineya motivation and parent provenance

* Parent `LUNCY579/ChatBubble`, `post-training-Mimo`, verified head:
  `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
* Parent tree: `6a1e3b9b16789f7ce211e443654912859dd37ed0`.
* `BROAD_RETENTION.md`, inspected at that commit:
  https://github.com/LUNCY579/ChatBubble/blob/03db218ceebc61a2bc04f2482cf7e094045e01c1/BROAD_RETENTION.md
  Blob `a4e1a3130521d236aa3a907e0cd7a2b43fa0c3c9`.
* Existing `experiments/__init__.py` was read and retained unchanged. Root and
  `experiments/` tree inventories showed no AGENTS.md/CLAUDE.md at those levels.

Historical report-only suite rows were not copied, read as evaluation examples,
used for training, or used for selecting any regularizer/configuration.
