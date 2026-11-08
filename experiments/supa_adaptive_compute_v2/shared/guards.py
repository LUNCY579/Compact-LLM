"""Fail closed before reads; no sealed-data authorization API exists in v2."""
from pathlib import Path
import re


def guard_non_sealed_path(path):
    path = Path(path).expanduser().resolve()
    parts = [part.lower() for part in path.parts]
    if any(re.search(r'(^|[^a-z])sealed([^a-z]|$)|test2|do_not_train', p) for p in parts):
        raise ValueError(f'sealed/TEST2 input forbidden: {path}')
    if 'private' in parts:
        raise ValueError(f'private input forbidden: {path}')
    return path


def require_within(path, root):
    path = guard_non_sealed_path(path)
    root = guard_non_sealed_path(root)
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path is outside required root {root}: {path}')
    return path


def require_track(track):
    if track not in ('native_armr', 'cross_scale'):
        raise ValueError('unknown scientific track')
    return track


def require_runtime(config):
    if (config.get('dtype'), config.get('use_cache'), config.get('threads')) != ('float32', False, 4):
        raise ValueError('only registered FP32 / four-thread / no-cache runtime is supported')
    if config.get('device') not in ('cpu','cuda:0'):
        raise ValueError('only explicit CPU or CUDA:0 execution is supported')
    if config['device'] == 'cuda:0':
        if config.get('tf32') is not False or not isinstance(config.get('runtime_qualification'),str) or not config['runtime_qualification']:
            raise ValueError('CUDA requires a prospective qualification ID and TF32 disabled')
        for key in ('min_available_gib','min_available_vram_gib','max_cuda_memory_gib'):
            import math
            if not isinstance(config.get(key),(int,float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'invalid CUDA resource limit {key}')
    if config.get('max_prompt_seconds', 900) <= 0 or config.get('max_rss_gib', 6) <= 0:
        raise ValueError('invalid resource limit')
