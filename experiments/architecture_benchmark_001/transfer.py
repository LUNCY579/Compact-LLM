"""Strict Qwen3 tensor mapping, identity depth expansion, and calibrated surgery.

These are research primitives. Calling one is not a G0/G1 receipt and never
licenses recovery training or an efficacy claim. The CLI enforces execution gates.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import copy
import hashlib
import json
import torch
from torch import Tensor
from .budget import Config, budget
from .model import DenseReference


def state_digest(model: DenseReference) -> str:
    """Byte identity for source immutability/lineage, including BF16 tensors."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(memoryview(raw).cast('B'))
    return h.hexdigest()


def native_key_map(c: Config) -> dict[str, str]:
    if c.architecture != 'plain':
        raise ValueError('native Qwen import is only defined for an ordinary control')
    out = {'model.embed_tokens.weight': 'embed_tokens.weight', 'model.norm.weight': 'final_norm.weight'}
    for i in range(c.encoder_layers):
        a, b = f'model.layers.{i}.', f'encoder.{i}.'
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'):
            out[a + 'self_attn.' + n + '.weight'] = b + n + '.weight'
        out[a + 'input_layernorm.weight'] = b + 'input_norm.weight'
        out[a + 'post_attention_layernorm.weight'] = b + 'post_norm.weight'
        for n in ('gate_proj', 'up_proj', 'down_proj'):
            out[a + 'mlp.' + n + '.weight'] = b + 'mlp.' + n + '.weight'
    return out


def import_qwen_state(state: dict[str, Tensor], c: Config) -> DenseReference:
    mapping = native_key_map(c)
    supplied = dict(state)
    if 'lm_head.weight' in supplied:
        if 'model.embed_tokens.weight' not in supplied or not torch.equal(
                supplied['lm_head.weight'], supplied['model.embed_tokens.weight']):
            raise ValueError('untied or changed output head is forbidden')
        supplied.pop('lm_head.weight')
    if set(supplied) != set(mapping):
        raise ValueError(f'native tensor inventory mismatch: missing={set(mapping)-set(supplied)}, '
                         f'unexpected={set(supplied)-set(mapping)}')
    with torch.device('meta'):
        model = DenseReference(c)
    target = model.state_dict()
    for src, dst in mapping.items():
        value = supplied[src]
        if value.shape != target[dst].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'invalid native tensor: {src}')
    # Clones avoid aliasing mutable source tensors. No mismatched-size loading.
    model.load_state_dict({dst: supplied[src].detach().clone() for src, dst in mapping.items()},
                          strict=True, assign=True)
    return model.eval()


def expansion_map(source_depth: int, target_depth: int) -> list[dict]:
    if not 0 < source_depth <= target_depth <= 2 * source_depth:
        raise ValueError('v1 supports retaining every source layer and at most one insertion per layer')
    extra = target_depth - source_depth
    result = []
    for i in range(source_depth):
        result.append({'source_layer': i, 'identity_inserted': False})
        if (i + 1) * extra // source_depth > i * extra // source_depth:
            result.append({'source_layer': i, 'identity_inserted': True})
    return result


def identity_expand(source: DenseReference, target_depth: int) -> tuple[DenseReference, dict]:
    if source.config.architecture != 'plain':
        raise ValueError('identity expansion is qualified only for ordinary blocks')
    mapping = expansion_map(len(source.encoder), target_depth)
    # Build only the destination inventory; deepcopy(source) followed by replacing
    # all blocks needlessly allocates a second copy of the original body first.
    with torch.device('meta'):
        result = DenseReference(replace(source.config, encoder_layers=target_depth))
    src = source.state_dict()
    state = {}
    for name in result.state_dict():
        key = name
        if name.startswith('encoder.'):
            _, idx, tail = name.split('.', 2)
            key = f"encoder.{mapping[int(idx)]['source_layer']}.{tail}"
        state[name] = src[key].detach().clone()
    result.load_state_dict(state, strict=True, assign=True)
    result.train(source.training)
    with torch.no_grad():
        for block, row in zip(result.encoder, mapping):
            block.c = result.config
            if row['identity_inserted']:
                block.o_proj.weight.zero_()
                block.mlp.down_proj.weight.zero_()
    return result, {'operation': 'identity_depth_expansion', 'layers': mapping,
                    'source_config': asdict(source.config), 'target_config': asdict(result.config),
                    'identity_claim': 'same width only, evaluation mode; must check actual logits'}


class ActivationCalibration:
    """Mean-absolute channel/neuron and mean-L2 head scores on CALIBRATION only.

    This is a documented Minitron-inspired estimator, NOT its exact aggregation
    or head-collapse recipe. No gradients, evaluation data, or silent random ranks.
    """
    def __init__(self, model: DenseReference):
        if model.config.architecture != 'plain':
            raise ValueError('calibrate an ordinary source before changing memory topology')
        self.model = model
        self.mask = None
        self.handles = []
        self.sums: dict[str, Tensor] = {}
        self.counts: dict[str, int] = {}
        self.rows = 0

    def _record(self, name: str, value: Tensor, heads: bool = False) -> None:
        value = value.detach()[self.mask]
        if heads:
            value = value.reshape(-1, self.model.config.num_attention_heads, self.model.config.head_dim)
            value = torch.linalg.vector_norm(value.float(), dim=-1)
        else:
            value = value.float().abs()
        total = value.double().sum(0).cpu()
        self.sums[name] = self.sums.get(name, torch.zeros_like(total)) + total
        self.counts[name] = self.counts.get(name, 0) + value.shape[0]

    def run(self, batches: list[dict], *, split: str) -> dict:
        if split != 'calibration' or not batches:
            raise ValueError('nonempty calibration-only data required')
        if self.rows:
            raise ValueError('calibration objects are single-use')
        for i, block in enumerate(self.model.encoder):
            self.handles.append(block.input_norm.register_forward_hook(
                lambda m, a, out: self._record('residual', out)))
            self.handles.append(block.mlp.down_proj.register_forward_pre_hook(
                lambda m, a, i=i: self._record(f'ffn.{i}', a[0])))
            self.handles.append(block.o_proj.register_forward_pre_hook(
                lambda m, a, i=i: self._record(f'heads.{i}', a[0], True)))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for batch in batches:
                    ids = batch['input_ids']
                    self.mask = batch.get('attention_mask', torch.ones_like(ids)).bool()
                    if self.mask.shape != ids.shape or not self.mask.any():
                        raise ValueError('calibration batch has no valid tokens or bad mask')
                    self.model(**batch)
                    self.rows += ids.shape[0]
        finally:
            for handle in self.handles:
                handle.remove()
            self.model.train(was_training)
        scores = {key: (val / self.counts[key]).tolist() for key, val in self.sums.items()}
        if not all(torch.isfinite(torch.tensor(v)).all() for v in scores.values()):
            raise ValueError('nonfinite calibration scores')
        return {'split': split, 'rows': self.rows, 'estimator': 'mean_abs_residual_ffn_mean_l2_query_heads_v1',
                'config': asdict(self.model.config), 'scores': scores}


def _top(scores: list[float], count: int, expected: int) -> list[int]:
    t = torch.tensor(scores, dtype=torch.float64)
    if t.shape != (expected,) or not 0 < count <= expected or not torch.isfinite(t).all():
        raise ValueError('invalid saliency dimensions/count/values')
    if (t < 0).any() or not (t > 0).any():
        raise ValueError('scores must be nonnegative with nonzero signal')
    # Stable ties, retained original ordering. Never rank TEST or DEV responses.
    return sorted(torch.argsort(t, descending=True, stable=True)[:count].tolist())


def select_width(c: Config, target: Config, calibration: dict) -> dict:
    if c.architecture != 'plain' or target.architecture != 'plain' or c.encoder_layers != target.encoder_layers:
        raise ValueError('width selection must not change depth/topology')
    if (c.vocab_size != target.vocab_size or c.head_dim != target.head_dim
            or c.num_attention_heads // c.num_key_value_heads != target.num_attention_heads // target.num_key_value_heads):
        raise ValueError('vocabulary, full head coordinates and GQA group size must be preserved')
    if calibration.get('split') != 'calibration' or calibration.get('config') != asdict(c):
        raise ValueError('calibration is not bound to this source shape')
    scores = calibration['scores']
    group = c.num_attention_heads // c.num_key_value_heads
    layers = []
    for i in range(c.encoder_layers):
        h = torch.tensor(scores[f'heads.{i}'], dtype=torch.float64)
        if h.shape != (c.num_attention_heads,):
            raise ValueError('bad query-head score shape')
        groups = _top(h.reshape(c.num_key_value_heads, group).sum(-1).tolist(),
                      target.num_key_value_heads, c.num_key_value_heads)
        layers.append({'kv_groups': groups,
                       'query_heads': [g * group + j for g in groups for j in range(group)],
                       'ffn': _top(scores[f'ffn.{i}'], target.intermediate_size, c.intermediate_size)})
    return {'residual': _top(scores['residual'], target.hidden_size, c.hidden_size), 'layers': layers}


def _indices(values: list[int], count: int, limit: int, device) -> Tensor:
    if len(values) != count or len(set(values)) != count or any(type(i) is not int or i < 0 or i >= limit for i in values):
        raise ValueError('duplicate, out-of-range, or wrong-size tensor index selection')
    return torch.tensor(values, dtype=torch.long, device=device)


def narrow(source: DenseReference, target: Config, selection: dict) -> tuple[DenseReference, dict]:
    c = source.config
    if (c.architecture != 'plain' or target.architecture != 'plain' or c.encoder_layers != target.encoder_layers
            or c.vocab_size != target.vocab_size or c.head_dim != target.head_dim
            or c.rope_theta != target.rope_theta or c.rms_norm_eps != target.rms_norm_eps
            or c.max_position_embeddings != target.max_position_embeddings):
        raise ValueError('narrowing may only reduce residual/FFN width and whole GQA groups')
    group = c.num_attention_heads // c.num_key_value_heads
    if group != target.num_attention_heads // target.num_key_value_heads or len(selection['layers']) != c.encoder_layers:
        raise ValueError('GQA group/depth mismatch')
    device, dtype = source.embed_tokens.weight.device, source.embed_tokens.weight.dtype
    S = _indices(selection['residual'], target.hidden_size, c.hidden_size, device)
    with torch.device('meta'):
        out = DenseReference(target)
    state = {'embed_tokens.weight': source.embed_tokens.weight.detach().index_select(1, S),
             'final_norm.weight': source.final_norm.weight.detach().index_select(0, S)}
    for i, (block, row) in enumerate(zip(source.encoder, selection['layers'])):
        _indices(row['kv_groups'], target.num_key_value_heads, c.num_key_value_heads, device)
        expected_q = [g * group + j for g in row['kv_groups'] for j in range(group)]
        if row['query_heads'] != expected_q:
            raise ValueError('query heads must retain their complete, ordered KV groups')
        Q = _indices([q*c.head_dim + j for q in row['query_heads'] for j in range(c.head_dim)],
                     target.num_attention_heads*c.head_dim, c.num_attention_heads*c.head_dim, device)
        K = _indices([k*c.head_dim + j for k in row['kv_groups'] for j in range(c.head_dim)],
                     target.num_key_value_heads*c.head_dim, c.num_key_value_heads*c.head_dim, device)
        F = _indices(row['ffn'], target.intermediate_size, c.intermediate_size, device)
        p = f'encoder.{i}.'
        b = block.state_dict()
        state[p+'input_norm.weight'] = b['input_norm.weight'].index_select(0, S)
        state[p+'post_norm.weight'] = b['post_norm.weight'].index_select(0, S)
        for name in ('q_norm', 'k_norm'):
            state[p+name+'.weight'] = b[name+'.weight'].clone()
        for name, rows, cols in [('q_proj', Q, S), ('k_proj', K, S), ('v_proj', K, S), ('o_proj', S, Q),
                                 ('mlp.gate_proj', F, S), ('mlp.up_proj', F, S), ('mlp.down_proj', S, F)]:
            state[p+name+'.weight'] = b[name+'.weight'].index_select(0, rows).index_select(1, cols)
    out.load_state_dict(state, strict=True, assign=True)
    assert sum(p.numel() for p in out.parameters()) == budget(target)['parameters']
    return out.eval(), {'operation': 'calibrated_width_slice', 'selection': selection,
                        'accounted_target_tensors': sorted(state), 'target_config': asdict(target),
                        'function_preserving': False, 'recovery_required': True}


def to_ced(source: DenseReference, target: Config, *, donor_layer: int) -> tuple[DenseReference, dict]:
    """Declared single donor for shared KV; never averages unrelated projections.

    Extra FFN neurons copy cyclic donor gate/up rows and have zero down columns.
    This isolates FFN growth at initialization; changing memory remains nonidentity.
    """
    c = source.config
    if (c.architecture != 'plain' or target.architecture != 'ced' or c.total_layers != target.total_layers
            or any(getattr(c, k) != getattr(target, k) for k in ('vocab_size', 'hidden_size', 'head_dim',
                'num_attention_heads', 'num_key_value_heads', 'rope_theta', 'rms_norm_eps', 'max_position_embeddings'))
            or target.intermediate_size < c.intermediate_size or not target.encoder_layers <= donor_layer < c.total_layers):
        raise ValueError('CED conversion requires same depth/width/heads, nonshrinking FFN, and an upper-layer donor')
    with torch.device('meta'):
        out = DenseReference(target)
    state = {}
    src = source.state_dict()
    for name in out.state_dict():
        if name.startswith('memory.'):
            tail = name[len('memory.'):]
            tail = 'input_norm.weight' if tail == 'norm.weight' else tail
            key = f'encoder.{donor_layer}.' + tail
        elif name.startswith('decoder.'):
            _, idx, tail = name.split('.', 2)
            key = f'encoder.{target.encoder_layers+int(idx)}.' + tail
        else:
            key = name
        value = src[key].clone()
        extra = target.intermediate_size - c.intermediate_size
        if extra and (name.endswith('mlp.gate_proj.weight') or name.endswith('mlp.up_proj.weight')):
            rows = torch.arange(extra, device=value.device) % c.intermediate_size
            value = torch.cat([value, value.index_select(0, rows)], dim=0)
        elif extra and name.endswith('mlp.down_proj.weight'):
            value = torch.cat([value, value.new_zeros(c.hidden_size, extra)], dim=1)
        state[name] = value
    out.load_state_dict(state, strict=True, assign=True)
    return out.eval(), {'operation': 'shared_encoder_memory_conversion', 'donor_layer': donor_layer,
                        'decoder_local_attention': False, 'function_preserving': False,
                        'ffn_growth': 'cyclic_gate_up_donors_zero_down_columns',
                        'accounted_target_tensors': sorted(state), 'target_config': asdict(target),
                        'requires_matched_plain_continuation_control': True}
