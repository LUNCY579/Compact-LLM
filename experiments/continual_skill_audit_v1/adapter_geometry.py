"""CPU LoRA weight-space geometry without loading a model or forming dense deltas.

Scope: standard unmerged, constant-rank/alpha LoRA A/B adapters without biases or
additional saved modules. Unsupported variants fail explicitly. Geometry is not
behavioral evidence and does not certify the base checkpoint behind an adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def effective_inner(a, b, c, d, scale_left=1.0, scale_right=1.0):
    """<scale_left*B@A, scale_right*D@C> via rank-sized Gram matrices."""
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or d.ndim != 2:
        raise ValueError('LoRA factors must be matrices')
    if b.shape[1] != a.shape[0] or d.shape[1] != c.shape[0]:
        raise ValueError('incompatible LoRA rank dimensions')
    if a.shape[1] != c.shape[1] or b.shape[0] != d.shape[0]:
        raise ValueError('different effective weight shapes')
    if not math.isfinite(scale_left) or not math.isfinite(scale_right):
        raise ValueError('nonfinite scale')
    return float(((b.T @ d) * (a @ c.T)).sum()) * scale_left * scale_right


def compare_factors(left: dict, right: dict, left_scale=1., right_scale=1.) -> dict:
    if not left or left.keys() != right.keys():
        raise ValueError('adapter module sets must match exactly')
    by_module = {}
    sum_left = sum_right = cross = 0.
    for name in sorted(left):
        a, b = left[name]
        c, d = right[name]
        aa = effective_inner(a, b, a, b, left_scale, left_scale)
        bb = effective_inner(c, d, c, d, right_scale, right_scale)
        ab = effective_inner(a, b, c, d, left_scale, right_scale)
        if not all(math.isfinite(v) for v in [aa, bb, ab]) or aa < 0 or bb < 0:
            raise ValueError('invalid effective norm')
        sum_left += aa
        sum_right += bb
        cross += ab
        by_module[name] = {'left_delta_l2': math.sqrt(aa), 'right_delta_l2': math.sqrt(bb),
                           'inner_product': ab,
                           'cosine': ab / math.sqrt(aa * bb) if aa and bb else None}
    coefficient = cross / sum_left if sum_left else None
    residual = max(0., sum_right - cross * cross / sum_left) if sum_left else sum_right
    return {
        'modules': len(by_module), 'left_delta_l2': math.sqrt(sum_left),
        'right_delta_l2': math.sqrt(sum_right),
        'right_over_left_norm': math.sqrt(sum_right / sum_left) if sum_left else None,
        'cosine': cross / math.sqrt(sum_left * sum_right) if sum_left and sum_right else None,
        'best_scalar_left_to_right': coefficient,
        'right_residual_after_best_scalar_l2': math.sqrt(residual),
        'right_residual_energy_fraction': residual / sum_right if sum_right else None,
        'difference_l2': math.sqrt(max(0., sum_left + sum_right - 2 * cross)),
        'per_module': by_module,
        'interpretation': 'Direction/norm diagnostic only; no inference, quality or retention claim.',
    }


def load_adapter(directory: Path):
    import torch
    from safetensors.torch import load_file
    config_path = directory / 'adapter_config.json'
    weight_path = directory / 'adapter_model.safetensors'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['peft_type'] != 'LORA' or config.get('bias', 'none') != 'none':
        raise ValueError('only plain bias-free LoRA is supported')
    unsupported = ['rank_pattern', 'alpha_pattern', 'modules_to_save', 'use_dora', 'use_rslora',
                   'use_qalora', 'use_bdlora', 'lora_bias', 'alora_invocation_tokens',
                   'target_parameters', 'trainable_token_indices', 'layer_replication',
                   'fan_in_fan_out']
    if any(config.get(key) for key in unsupported):
        raise ValueError('unsupported LoRA configuration variant')
    tensors = load_file(str(weight_path), device='cpu')
    factors = {}
    consumed = set()
    for name, tensor in tensors.items():
        if name.endswith('.lora_A.weight'):
            prefix = name.removesuffix('.lora_A.weight')
            other = prefix + '.lora_B.weight'
            if other not in tensors:
                raise ValueError('missing B factor')
            if tensor.shape[0] != config['r'] or tensors[other].shape[1] != config['r']:
                raise ValueError('factor rank disagrees with configuration')
            factors[prefix] = tensor.to(torch.float64), tensors[other].to(torch.float64)
            consumed.update([name, other])
    if not factors or consumed != tensors.keys():
        raise ValueError('unsupported or unmatched adapter tensors')
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [config_path, weight_path]}
    return factors, config['lora_alpha'] / config['r'], config, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite geometry report')
    import torch
    torch.set_num_threads(2)
    left, scale_left, cfg_left, hashes_left = load_adapter(args.left)
    right, scale_right, cfg_right, hashes_right = load_adapter(args.right)
    if cfg_left.get('base_model_name_or_path') != cfg_right.get('base_model_name_or_path'):
        raise ValueError('declared base paths differ; establish base identity separately')
    result = compare_factors(left, right, scale_left, scale_right)
    result.update({'input_sha256': {**hashes_left, **hashes_right}, 'device': 'cpu',
                   'base_weights_verified': False, 'model_calls': 0, 'quality_claim': False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['per_module', 'input_sha256']}, indent=2))


if __name__ == '__main__':
    main()
