"""Read CUDA stack/device availability; allocate no CUDA tensors or models."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch

available = torch.cuda.is_available()
gpu = None
if available:
    free, total = torch.cuda.mem_get_info(0)
    gpu = {'name': torch.cuda.get_device_name(0),
           'compute_capability': list(torch.cuda.get_device_capability(0)),
           'free_vram_bytes': free, 'total_vram_bytes': total}
driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
                         '--format=csv,noheader'], capture_output=True, text=True, check=False)
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'CUDA_STACK_INVENTORY_ONLY_NOT_NATIVE_MODEL_OR_TRAINING_FIT',
    'python_executable': sys.executable, 'python_version': platform.python_version(),
    'platform': platform.platform(),
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'compiled_cuda': torch.version.cuda, 'cuda_available': available,
    'cuda_visible_devices_environment': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'gpu': gpu, 'nvidia_smi_exit_code': driver.returncode,
    'nvidia_smi': driver.stdout.strip(),
    'available_host_ram_bytes': psutil.virtual_memory().available,
    'worker_rss_bytes': psutil.Process().memory_info().rss,
    'cuda_tensor_or_model_allocated': False, 'pretrained_weights_loaded': False,
    'native_inference_mechanics_qualified': False, 'training_fit_established': False,
    'shared_environments_modified': False,
}
assert version('torch') == '2.6.0+cu124'
assert version('transformers') == '4.51.3'
assert torch.version.cuda == '12.4'
output = Path(__file__).resolve().parent/'environment.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt, indent=2))
