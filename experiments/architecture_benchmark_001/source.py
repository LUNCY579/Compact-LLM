"""Offline official-source audit. Metadata presence is never weight availability."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from safetensors import safe_open
import torch
from .budget import presets
from .model import DenseReference
from .transfer import native_key_map

ROOT = Path(__file__).parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_lock() -> dict:
    return json.loads((ROOT / 'source_lock.json').read_text())


def source_config(raw: dict):
    c = presets()['B28_1024_official_shape']
    checks = {'hidden_size': c.hidden_size, 'intermediate_size': c.intermediate_size,
              'num_hidden_layers': c.encoder_layers, 'num_attention_heads': c.num_attention_heads,
              'num_key_value_heads': c.num_key_value_heads, 'head_dim': c.head_dim,
              'vocab_size': c.vocab_size, 'rope_theta': c.rope_theta, 'rms_norm_eps': c.rms_norm_eps,
              'max_position_embeddings': c.max_position_embeddings, 'model_type': 'qwen3',
              'tie_word_embeddings': True, 'attention_bias': False, 'attention_dropout': 0.0,
              'hidden_act': 'silu', 'rope_scaling': None, 'sliding_window': None,
              'use_sliding_window': False, 'architectures': ['Qwen3ForCausalLM']}
    for key, value in checks.items():
        if key not in raw or raw[key] != value:
            raise ValueError(f'official source configuration mismatch: {key}')
    if any(key in raw for key in ('auto_map', 'quantization_config', 'abstain_logit_bias')):
        raise ValueError('custom/quantized/readout source configuration forbidden')
    return c


def audit_source(path: str | Path) -> dict:
    root = Path(path).resolve()
    lock = source_lock()
    if not root.is_dir():
        raise ValueError('official source directory missing; no download or fallback will occur')
    actual = {}
    for name, expected in lock['assets'].items():
        p = root / name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f'official source asset missing or symlinked: {name}')
        if name == 'model.safetensors' and p.stat().st_size != lock['weight_bytes']:
            raise ValueError('official weight size mismatch; metadata/LFS pointers are not weights')
        actual[name] = sha256_file(p)
        if actual[name] != expected:
            raise ValueError(f'official source SHA-256 mismatch: {name}')
    if (root/'adapter_config.json').exists() or list(root.glob('*.py')):
        raise ValueError('adapter/custom-model directories are not an ungated official core')
    c = source_config(json.loads((root/'config.json').read_text()))
    with torch.device('meta'):
        target = DenseReference(c).state_dict()
    mapping = native_key_map(c)
    with safe_open(str(root/'model.safetensors'), framework='pt', device='cpu') as f:
        if set(f.keys()) != set(mapping) or len(f.keys()) != lock['tensor_count']:
            raise ValueError('official tensor inventory mismatch')
        total = 0
        for name, dest in mapping.items():
            view = f.get_slice(name)
            if list(target[dest].shape) != view.get_shape() or view.get_dtype() != lock['tensor_dtype']:
                raise ValueError(f'official tensor shape/dtype mismatch: {name}')
            total += target[dest].numel()
        if total != lock['parameters']:
            raise ValueError('official parameter count mismatch')
    return {'schema': 1, 'core_id': lock['core_id'], 'assets': actual, 'parameters': total,
            'audit_digest': canonical_hash({'core_id': lock['core_id'], 'assets': actual}),
            'real_weights_present': True, 'evidence_kind': 'source_audit_only',
            'licenses_training': False, 'licenses_efficacy_claim': False}
