# PEFT Capability Probe

- Environment: Python 3.11.15, torch 2.12.1+cu126, transformers 5.14.1, PEFT 0.19.1.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU.
- Multiple official LoRAs loaded independently: **yes**.
- Per-example adapter batch API: **runs**; it is not enabled for research runs because the probe showed nonzero bf16 logit differences.
- LoRA hotswap API on official adapters: **runs**; it is not enabled for research runs because the probe showed nonzero bf16 logit differences.
- Tiny-Qwen LoRA/DoRA/PiSSA/AdaLoRA forwards: **all pass**.
- Inference mixed LoRA + AdaLoRA construction: **pass**.

These are software-capability checks, not evidence that a parameterization improves this model. The official compatible pool contains only rank-16 LoRAs; fair DoRA/PiSSA/AdaLoRA comparisons therefore remain preregistered until matched adapters are trained inside this lab.

Primary references: [PEFT hotswapping](https://huggingface.co/docs/peft/package_reference/hotswap), [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora), [PEFT mixed adapters](https://huggingface.co/docs/peft/developer_guides/mixed_models), [LoRA](https://arxiv.org/abs/2106.09685), [DoRA](https://arxiv.org/abs/2402.09353), and [PiSSA](https://arxiv.org/abs/2404.02948).
