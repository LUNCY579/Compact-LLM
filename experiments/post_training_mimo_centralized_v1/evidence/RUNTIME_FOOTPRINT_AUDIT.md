# Runtime footprint feasibility audit

Status: VERIFIED SOURCE metadata inspection; no import-RSS benchmark and no new
environment installation. The native RAM guard remains unchanged.

The pinned `.venv-mq001` has 29 installed distributions. Its required runtime
dependency closure contains 25. The only extras identified are pytest, iniconfig,
pluggy and Pygments. scikit-learn, scipy, pandas, librosa, torchvision, torchaudio,
PIL, accelerate and peft are absent. A stdlib/find_spec check found no external
site paths and did not import torch. Transformers' optional audio/vision/sklearn
branches are consequently not a removable active dependency bundle here.

All 25 exact runtime versions have compatible local uv wheel-index entries pointing
to archive directories. This includes torch 2.10.0+cu126, Transformers 4.57.1,
safetensors 0.7.0, tokenizers 0.22.2 and psutil 7.2.2. Cache presence is not an
installation/integrity test. No environment was created or modified.

INFERRED: removing four unused testing packages is unlikely to materially change
the native preload footprint. Do not spend time cloning this environment merely
to claim a memory fix. A future bounded, CPU-only staged-import probe could separate
import costs, but cannot qualify post-CUDA admission or model fit.

The E0 preload RSS of about 0.67 GiB is measured after CUDA availability/BF16/device
queries and synchronize/mem_get_info, not purely after Python imports. The import
timer and the post-CUDA resource snapshot measure different quantities. The pinned
Torch Windows package also loads its own bundled DLLs; those costs remain in a
same-wheel clone. Current pre-import free RAM is itself below 7.4 GiB, so this audit
does not justify another native launch or any reduction in the guard.

Separately, the reused watchdog has a Linux parent-death helper that the E0 worker
does not install. E0 is now explicitly Windows-only until a new host-containment
qualification is registered. Current Windows Job Object containment is unaffected.
