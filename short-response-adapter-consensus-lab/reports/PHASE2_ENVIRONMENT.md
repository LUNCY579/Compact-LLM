# Phase 2 Execution Environment

Captured UTC: `2026-08-14T20:39:11.565280+00:00` by `experiments/phase2_mechanism/capture_environment.py`.

| Component | Detected value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| CPU | AMD Ryzen 9 7940HX with Radeon Graphics; 16 physical / 32 logical cores |
| RAM | 16,340,398,080 bytes |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU; 8,188 MiB; compute capability 8.9 |
| NVIDIA driver | 610.62 |
| Python | 3.11.15 (main, Jun  2 2026, 22:29:49) [MSC v.1944 64 bit (AMD64)] |
| PyTorch / CUDA / cuDNN | 2.12.1+cu126 / 12.6 / 91002 |
| Transformers / PEFT / Accelerate | 5.14.1 / 0.19.1 / 1.14.0 |
| Safetensors / NumPy / SciPy | 0.8.0 / 2.4.6 / 1.17.1 |
| Paper inference dtype | torch.bfloat16; BF16 supported=True |

Latency, TTFT, VRAM, and RSS are reported per condition because they are decoder- and trajectory-dependent. This environment record does not make latency portable to other hardware.
