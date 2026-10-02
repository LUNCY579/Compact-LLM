"""Local-only standard Qwen3 host. Unsupported Rineya loaders fail, not substitute."""
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from .core import HybridAttention, LoRALinear, cache_bytes
from .protocol import Blocked, resources, read_json, sha, write_json, now


class IdentityAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, *args, **kwargs):
        return self.source(*args, **kwargs)


def load_host(config):
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    torch.manual_seed(config['seed'])
    dtype = getattr(torch, config['dtype'])
    path = config['source']['path']
    model, info = Qwen3ForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise Blocked(f'exact-source loading rejected: {info}')
    model = model.to(config['device']).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if len(tokenizer) > model.config.vocab_size or not tokenizer.chat_template:
        raise Blocked('tokenizer vocabulary/template is incompatible with source')
    resources(config['device'], runtime=True)
    return model, tokenizer


def install(model, config, arm):
    """One selected site; no structural change in A, zero-output LoRA in B."""
    model.requires_grad_(False)
    source = model.model.layers[config['layer']].self_attn
    if arm in ('C', 'D'):
        module = HybridAttention(source, 'local' if arm == 'C' else 'hybrid', config['window'])
        model.model.layers[config['layer']].self_attn = module
    else:
        module = source
    if arm != 'A':
        # Reset identically AFTER hybrid allocation: paired LoRA draws across arms.
        torch.manual_seed(config['seed'] + config['layer']*100)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(module, name, LoRALinear(getattr(module, name), config['rank']))
    return module


def trainables(model):
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def save_delta(model, directory, binding, record):
    tensors = {n: p.detach().cpu().contiguous().clone() for n, p in trainables(model).items()}
    if not tensors:
        raise Blocked('no adaptation tensors to save')
    path = Path(directory)/'delta.safetensors'
    if path.exists():
        raise Blocked('refusing to overwrite delta')
    save_file(tensors, str(path))
    write_json(Path(directory)/'delta.json', {'binding': binding, 'sha256': sha(path),
               'parameters': {n: list(t.shape) for n, t in tensors.items()}, 'record': record})


def load_delta(model, directory, binding):
    if directory is None:
        return
    path = Path(directory)
    manifest = read_json(path/'delta.json')
    for key in ('source_sha256', 'config_sha256', 'code_sha256', 'arm'):
        if manifest['binding'][key] != binding[key]:
            raise Blocked(f'delta has incompatible {key}')
    if sha(path/'delta.safetensors') != manifest['sha256']:
        raise Blocked('delta bytes changed')
    tensors = load_file(str(path/'delta.safetensors'))
    parameters = trainables(model)
    if tensors.keys() != parameters.keys():
        raise Blocked('delta parameter names do not exactly match adaptation')
    with torch.no_grad():
        for name, p in parameters.items():
            t = tensors[name]
            if list(t.shape) != list(p.shape) or not torch.isfinite(t).all():
                raise Blocked(f'bad delta tensor {name}')
            p.copy_(t.to(p))


def model_forward(model, ids, cache=None):
    return model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)


def check_cache(cache, config, arm, length):
    if arm not in ('C', 'D'):
        return
    site = config['layer']
    for name in ('key_cache', 'value_cache'):
        values = getattr(cache, name)
        if len(values) > site and torch.is_tensor(values[site]) and values[site].numel():
            raise Blocked('converted layer illegally retains full-history KV')
    state = cache._sbhm[site]
    if state.seen != length or state.keys.shape[2] != min(length, config['window']):
        raise Blocked('incorrect persistent local cache length')
    if arm == 'D' and (state.matrix is None or not torch.isfinite(state.matrix).all()):
        raise Blocked('missing/nonfinite recurrent matrix')


@torch.no_grad()
def qualify(model, config, arm, binding, delta=None):
    site = config['layer']
    length = config['limits']['max_prompt_tokens']
    ids = (torch.arange(length, device=config['device']) % min(97, model.config.vocab_size-10) + 10)[None]
    baseline = model(input_ids=ids[:, :8], use_cache=False, logits_to_keep=1).logits
    original = model.model.layers[site].self_attn
    model.model.layers[site].self_attn = IdentityAttention(original)
    wrapped = model(input_ids=ids[:, :8], use_cache=False, logits_to_keep=1).logits
    model.model.layers[site].self_attn = original
    torch.testing.assert_close(wrapped, baseline, atol=0, rtol=0)
    install(model, config, arm)
    load_delta(model, delta, binding)
    small = model(input_ids=ids[:, :8], use_cache=False, logits_to_keep=1).logits
    if arm in ('A', 'B') and delta is None:
        torch.testing.assert_close(small, baseline, atol=0, rtol=0)
    full = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
    if not torch.isfinite(full).all():
        raise Blocked('nonfinite full-prefix logits')
    tolerance = 3e-4 if config['dtype'] == 'float32' else 0.05
    errors = {}
    # Maximum-shape chunked path plus tokenwise on a short synthetic prefix.
    for chunk, tokens in [(config['limits']['prefill_chunk'], ids), (1, ids[:, :17])]:
        cache, output = None, None
        for part in tokens.split(chunk, dim=1):
            output = model_forward(model, part, cache)
            cache = output.past_key_values
            resources(config['device'], runtime=True)
        expected = full if tokens.shape[1] == length else model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits
        error = (expected.float()-output.logits.float()).abs().max().item()
        torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
        if not torch.equal(output.logits.argmax(-1), expected.argmax(-1)):
            raise Blocked('full-prefix/incremental greedy-token mismatch')
        check_cache(cache, config, arm, tokens.shape[1])
        errors[f'{chunk}:{tokens.shape[1]}'] = {'max_abs_logit_error': error, 'cache_storage_bytes': cache_bytes(cache)}
    again = model(input_ids=ids[:, :8], use_cache=False, logits_to_keep=1).logits
    torch.testing.assert_close(again, small, atol=0, rtol=0)
    return {'status': 'PASS_NATIVE_MECHANICS_ONLY', 'created_utc': now(), 'binding': binding,
            'identity_wrapper_max_abs_error': 0, 'parity': errors, 'capability_claim': False,
            'native_training_qualified': False, 'max_prompt_tokens_checked': length,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in trainables(model).values())}
