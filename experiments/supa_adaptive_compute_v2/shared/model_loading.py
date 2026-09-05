"""Canonical local Supa, with pinned bytes and the audited historical runtime."""
from pathlib import Path
import json
from . import legacy
from .guards import guard_non_sealed_path, require_runtime
from .integrity import frozen_inventory, checkpoint_files, state_hash
from .hashing import sha256_file

CANONICAL_SHA256 = 'a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026'


def require_available_memory(config):
    import psutil
    available = psutil.virtual_memory().available
    minimum = config.get('min_available_gib',3.0) * 2**30
    if available < minimum:
        raise MemoryError(f'insufficient available physical RAM: {available/2**30:.2f} GiB; registered start requires {minimum/2**30:.2f} GiB')
    return {'available_memory_bytes_at_start':available,'minimum_available_gib':minimum/2**30}


def resolve_model(explicit=None):
    from run_experiment import resolve_model as resolve
    return guard_non_sealed_path(resolve(None if explicit is None else Path(explicit)))


def load_supa(config, model_path=None):
    import inspect
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    require_runtime(config)
    if torch.__version__ != '2.12.1+cu126' or transformers.__version__ != '5.14.1':
        raise RuntimeError('unqualified runtime; use Chat-Bubble/.venv/Scripts/python.exe')
    memory = require_available_memory(config)
    path = resolve_model(model_path)
    before = checkpoint_files(path)
    reference_path = legacy.ARMR / 'evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if before != reference['checkpoint_files_before']:
        raise RuntimeError('canonical checkpoint/tokenizer/config sidecars differ from qualified CPU replication')
    implementation_sha = sha256_file(Path(inspect.getfile(Qwen3DecoderLayer)))
    if implementation_sha != reference['qwen_implementation_sha256']:
        raise RuntimeError('installed Qwen3 source differs from qualified runtime')
    if before.get('model.safetensors') != CANONICAL_SHA256 or config.get('checkpoint_sha256', CANONICAL_SHA256) != CANONICAL_SHA256:
        raise RuntimeError('canonical Supa checkpoint mismatch')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=True,
        dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True).eval().requires_grad_(False)
    if type(model).__name__ != 'Qwen3AbstainForCausalLM' or len(model.model.layers) != 28 or model.config.hidden_size != 1024:
        raise RuntimeError('wrong Supa class, depth or width')
    return model, tokenizer, {**frozen_inventory(model), **memory, 'model_path': str(path),
        'checkpoint_files_before': before, 'state_sha256_before': state_hash(model),
        'model_class': type(model).__name__, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'qwen_implementation_sha256': implementation_sha}
