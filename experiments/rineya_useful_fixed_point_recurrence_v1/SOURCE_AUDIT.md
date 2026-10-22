# Sources and pre-implementation audit

Audit date: 2026-10-03. No native checkpoint was loaded. Source inspection used
the authorized GitHub connector and public official release. Shell git access
failed DNS in the isolated execution container; publication uses GitHub Git-data
commits and a non-force ref update, not a successful shell clone/push claim.

## Source pins

- ChatBubble integrated base: `03db218ceebc61a2bc04f2482cf7e094045e01c1`;
  root tree `6a1e3b9b16789f7ce211e443654912859dd37ed0`.
- xLLM-Loop official main: `3af99f493e14fba162625d5dd687abea551a299e`.
  `xllm/models/looped.py` inspected lines 1-230; blob
  `8520d24d1b2b883099ec052ea0bea13583d0ac2a`.
- TAPS current audited tip: `7ece5cead285ba65c7d73ab57c0327fae2dc40bb`;
  `RESULTS.md` blob `8439743777b54aa17cc6d676b9be3df709d5a64c`.
- Existing `research/rineya-evidence-function-equilibrium-v1`:
  `c934589026f7456352d62a83bb102c453e15519a`. Branch existence confirmed;
  not merged or newly revalidated. This experiment does not supersede it.

Official sources:
https://github.com/ifm-ai/xllm-loop
https://github.com/ifm-ai/xllm-loop/blob/3af99f493e14fba162625d5dd687abea551a299e/xllm/models/looped.py
https://github.com/ifm-ai/xllm-loop/blob/3af99f493e14fba162625d5dd687abea551a299e/xllm/models/looped_depth_prior.py
https://github.com/ifm-ai/xllm-loop/blob/3af99f493e14fba162625d5dd687abea551a299e/papers/part2.pdf

The paper is *Towards Looped Models Done Right — Part II: Rethinking at Fixed
Points*, screenshot date September 29, 2026. The public repository still says
arXiv link to be added. Do not invent a numeric arXiv ID. The full PDF was not
retrieved in this execution; no proof/table revalidation is claimed. This
implementation is independently written from inspected formulas and source
interfaces; it does not vendor xLLM or reproduce its pretraining results.

## Inspected native host sources at the integrated base

| File | Blob SHA | Consequence |
|---|---|---|
| ARMR `AGENTS.md` | `1492479028231aec9104165b0f1d3c3bc93b323b` | Preserve CPU FP32, eval/frozen, no-cache, L0-L2 one-shot, no canonical saves or protected data. |
| `src/armr/hooks.py` | `d10e8113619137548888854dc512c48cafdecc7a` | Scoped methods, cache rejection, tensor return, restoration and inherited emergency norm bounds. |
| `src/armr/recurrence.py` | `ddd6bacb07dc63e460e46c26e5118d0165077279` | Repeat the full contiguous middle block, not only its last layer. |
| `src/armr/evaluation.py` | `a0a59122c887567be6c485a4d7b56b1aeaf0d1b2` | Bounded greedy full-prefix generation, scorer and output safety checks; new bridge supplies expected recorder attributes. |
| `scripts/run_experiment.py` | `a6a331ace71b22ae659f4c7fc105e071082410cc` | Custom local-only loader, checkpoint/state/runtime binding, historical qualification requirements. |
| `configs/reapplication_screen.json` | `0a3b883e5f297f0ade94a1fc7160027044c1b636` | Canonical safetensors SHA, 24-token cap, 90-second case check and emergency bounds. |

The earlier TAPS audit records intact/repeated-L12 12/16 and repeated-L12-L13
13/16 on previously inspected DEV, not independent efficacy. The current TAPS
report remains BLOCKED for native scientific qualification and reports only
mathematical fixtures; it is not a trained useful-recurrence checkpoint.
No deeper-depth authorization, CUDA/FP16 qualification or training permission
transfers from those packages to this one.

## Limits of this audit

This is not an exhaustive all-branch audit or a new run of the inherited full
suite. The entire repository was not cloned into the container. Only this new
package was assembled and tested in isolation; source-reuse compatibility is
based on the inspected contracts and fixture tests. Native runtime/assets and
the real custom model remain unexercised. No GitHub Actions success is claimed.

No existing research branches, MiMo integration files, datasets, canonical
checkpoints, game files or schedules are changed. All new files stay under this
experiment directory. Existing nearby branches were checked before branch creation.
