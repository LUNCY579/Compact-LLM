"""Explicit CPU/CUDA FP32 execution; no automatic offload or reduced precision."""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _driver_version():
    import subprocess
    return subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader','--id=0'],text=True).strip()


def configure_runtime(config):
    from .guards import require_runtime
    import torch
    require_runtime(config)
    # Must precede CUDA context/BLAS handle creation in this process.
    if config['device'].startswith('cuda'):
        current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if current not in (None, ':4096:8'):
            raise RuntimeError('unregistered CUBLAS_WORKSPACE_CONFIG')
        if current is None and torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized before registered cuBLAS determinism settings')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if config['device'].startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('registered CUDA device unavailable')
        torch.cuda.set_device(config['device'])
        torch.cuda.manual_seed_all(config['seed'])


def synchronize(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def cuda_memory(config):
    import torch
    if not config.get('device','cpu').startswith('cuda'):
        return {'cuda_allocated_bytes':0,'cuda_reserved_bytes':0,
                'peak_cuda_allocated_bytes':0,'peak_cuda_reserved_bytes':0}
    synchronize(config['device'])
    info = {'cuda_allocated_bytes':torch.cuda.memory_allocated(config['device']),
            'cuda_reserved_bytes':torch.cuda.memory_reserved(config['device']),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(config['device']),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(config['device'])}
    if max(info['cuda_reserved_bytes'],info['peak_cuda_reserved_bytes']) > config['max_cuda_memory_gib'] * 2**30:
        raise MemoryError('registered CUDA reserved-memory budget exceeded')
    return info


def runtime_identity(config):
    import torch
    import transformers
    device = config['device']
    result = {'device':device,'dtype':config['dtype'],'threads':torch.get_num_threads(),
              'seed':config.get('seed'),
              'torch':str(torch.__version__),'transformers':transformers.__version__,
              'attention':'eager','use_cache':False}
    if device.startswith('cuda'):
        properties = torch.cuda.get_device_properties(device)
        result.update(device_name=properties.name,compute_capability=list(torch.cuda.get_device_capability(device)),
                      device_uuid=str(getattr(properties,'uuid','unavailable')),
                      driver_version=_driver_version(),
                      total_vram_bytes=properties.total_memory,cuda_runtime=torch.version.cuda,
                      tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                      tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                      matmul_precision=torch.get_float32_matmul_precision(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return result


def assert_model_device(model, device):
    import torch
    expected = torch.device(device)
    wrong = [name for name,t in list(model.named_parameters())+list(model.named_buffers()) if t.device != expected]
    if wrong:
        raise RuntimeError(f'CPU/meta/offloaded tensor in single-device model: {wrong[:8]}')
    return True
