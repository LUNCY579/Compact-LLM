"""S0 native runtime (COMP-RS-001 Amendment 1, section 16; sections 2, 8.1, 8.2 and 12).

The one place S0 stages take their runtime from. It never trains, never builds an optimizer and never
reads a bank; the stage entrypoints do that on top of it.

- Runtime pins. PINS are copied from the asset receipt (evidence/Q1_ASSET_CANDIDATE.json, sha pinned as
  ASSET_RECEIPT_SHA); assert_runtime() checks the interpreter path, the Python version and every pinned
  package version, both installed and imported.
- configure(device) applies the section 16 determinism settings before CUDA initialises:
  CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True), both TF32 flags off,
  torch.set_num_threads(4) and, on CUDA only, torch.cuda.set_per_process_memory_fraction(0.78, 0). It
  refuses on CUDA once CUDA is initialised, because the workspace setting would no longer be guaranteed.
- Pinned imports. gpu_lease.py and hostbudget.py are loaded from their registered paths only after their
  bytes hash to the registered sha, and the verified bytes are what is executed (no re-read). lease()
  holds the shared GPU lease for a block and always releases its own token; host_require/host_guard
  delegate to hostbudget.require/guard unmodified. Every one takes an injected module for tests.
- Loader (section 16 'Loader'). load_model_fp32 checks the weight and config bytes against the asset
  receipt, builds Qwen3ForCausalLM from config.json directly on the target device in FP32 (under a
  torch.device context, so a CUDA build never materialises on the CPU), ties lm_head to the input
  embedding, then streams every stored BF16 tensor with safe_open(device=device) and copies its exact FP32
  upcast into the tensor of the same name. T1 identity (verify_identity) then requires: every stored key
  has a slot and vice versa, the only model tensor not stored is lm_head.weight, and it is the same
  Parameter (same storage) as model.embed_tokens.weight; every loaded tensor cast back to BF16 equals its
  stored tensor bit for bit; the tensor count and dtypes equal the receipt's header. Digests are taken one
  tensor at a time in bounded chunks, so the host transient is at most CHUNK elements.
- Digests. tensor_sha256 is arm_build.tensor_digest computed in chunks (same bytes, same header), so
  identity['stored_sha256'] is arm_build.state_digest of the stored BF16 state, identity['arm_sha256'] is
  the section 2 identity of arm B (= build_arm_streaming(model.safetensors, [])['arm_sha256']), and
  state_fingerprint(model) is the same construction over every parameter and buffer name of the model,
  tied alias and non-persistent buffers included.
- Anchor (section 18, Amendment 2). load_anchor loads Supa with load_model_fp32 and, given a delta, reads it
  with arm_build.read_delta (its data-section digest must reproduce delta_sha256), adds it in FP32 in place
  (add_delta: keys a subset of the model's parameter names, the tied lm_head alias and buffers refused, shapes
  equal, nothing else touched) and requires anchor_digest(model) == arm_sha256 (ANCHOR_DIGEST_RULE), else
  arm_build.ArmRebuildStop. identity['anchor'] holds ANCHOR_KEYS (delta fields None for plain Supa); on an
  anchored model identity['arm_sha256'] and identity['state_fingerprint'] describe the anchored state and
  identity['base'] keeps plain Supa's. anchored_name adds WARM_SUFFIX to a stage file name.
- Tokenizer. load_tokenizer loads local files only, no remote code, and checks the chat-template sha, the
  class and the special IDs against the tokenizer receipt (and, given the asset receipt, the six asset file
  hashes). render() renders policy messages with add_generation_prompt=True, enable_thinking=False,
  tokenize=False, encodes with add_special_tokens=False, and requires the rendered string to end with the
  pre-closed think block THINK_PREFIX (token suffix THINK_PREFIX_IDS), the Q1 delimiter counts, and no other
  special token (PAD = <|endoftext|> included) - the q1_tokenizer.py rule.
- decoding_preimage(device) is the section 8.2 preimage; decoding_sha256 is preflight.digest of it. It
  refuses a runtime that configure() has not set.
- Records (section 16 'Records', section 12). RECORDS is outside the bound checkout; assert_outside_checkout,
  assert_clean_checkout (read-only git) and write_receipt (create-only, LF bytes) enforce it.
- resources() records free physical RAM (hostbudget.snapshot), worker RSS and peak working set, and on
  CUDA the peak allocated/reserved and mem_get_info.

Nothing here touches CUDA unless device == 'cuda' is passed.
"""
from __future__ import annotations

from collections.abc import Mapping
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import types

from .mechanics import parameter_inventory
from .preflight import digest, read_json, require, sha

HERE = Path(__file__).resolve().parent
ASSET_RECEIPT = HERE / 'evidence' / 'Q1_ASSET_CANDIDATE.json'
ASSET_RECEIPT_SHA = '9d9908cbcbb83a7feb75ad45ac38bd1d768f2841def8a6fb7ad9f4b6444798a3'
TOKENIZER_RECEIPT = HERE / 'evidence' / 'Q1_TOKENIZER_PROMPT_LENGTHS.json'
TOKENIZER_RECEIPT_SHA = '6130cea7c36c9466eb51e1365a3740245efce4549c0a7f3fcdac4dba896da30a'
MODEL_DIR = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores'
                 r'\Rineya-Supa-1.7.0-devfreeze')
DATA = Path(r'C:\Users\clclc\BusinessProject\Rineya-Dataset\data')
RECORDS = Path(r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0')

# Copied from ASSET_RECEIPT['runtime_candidate'] (a test re-reads the receipt and compares).
PINS = {'python': '3.11.15',
        'python_executable': r'C:\Users\clclc\BusinessProject\.venv-mq001\Scripts\python.exe',
        'packages': {'safetensors': '0.7.0', 'tokenizers': '0.22.2', 'torch': '2.10.0+cu126',
                     'transformers': '4.57.1'}}

GPU_LEASE_PATH = Path(r'C:\Users\clclc\BusinessProject\rineya-dataset-retention-records'
                      r'\hplat-rehearsal-20260921\gpu_lease.py')
GPU_LEASE_SHA = '049742734eea9bddd7aaf149d2da824c3d9eec1e2157fab6c6c586d033fda00a'
LOCK_NAME = 'rineya-gpu-0.lock'
HOSTBUDGET_PATH = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
                       r'\experiments\rab001_exploration\hostbudget.py')
HOSTBUDGET_SHA = '9f9dcf351f6c9ebfe00bf4e470296cd48db424585f268c205801bd9b0f38acd1'
LEASE_LABEL = re.compile(r'comp-rs-001-[a-z0-9][a-z0-9.-]{2,80}')

DEVICES = {'cpu': 'cpu', 'cuda': 'cuda:0', 'cuda:0': 'cuda:0'}
CUBLAS_WORKSPACE_CONFIG = ':4096:8'
NUM_THREADS = 4
MEMORY_FRACTION = 0.78
ATTN_IMPLEMENTATION = 'sdpa'  # transformers' default for Qwen3, pinned explicitly; in the preimage

# Section 8.1 decoding, IDs from the tokenizer receipt (a test compares).
EOS_IDS = (151645, 151643)
PAD_ID = 151643
MAX_NEW_TOKENS = 128
DECODING = {'batch_size': 1, 'do_sample': False, 'num_beams': 1, 'max_new_tokens': MAX_NEW_TOKENS,
            'eos_token_id': list(EOS_IDS), 'pad_token_id': PAD_ID, 'enable_thinking': False,
            'add_generation_prompt': True, 'add_special_tokens': False, 'greedy': True}

THINK_PREFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
THINK_PREFIX_IDS = (151644, 77091, 198, 151667, 271, 151668, 271)
POLICY_ROLES = ['system', 'user']
TOKENIZER_ASSETS = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'vocab.json', 'merges.txt')
UNBOUND_TOKENIZER_FILES = ('special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
                           'chat_template.json', 'tokenizer.model')
ARCHITECTURE_KEYS = ('architectures', 'model_type', 'num_hidden_layers', 'tie_word_embeddings', 'vocab_size')
TIED = ('lm_head.weight', 'model.embed_tokens.weight')
CHUNK = 1 << 22
GIB = 1024 ** 3
HEX = frozenset('0123456789abcdef')

# Section 18: theta0' = Supa 1.7 + delta_warm, identified by one digest convention (s0_warmstart uses it too).
ANCHOR_KEYS = ('base_weights_sha256', 'delta_path', 'delta_sha256', 'arm_sha256')
ANCHOR_DIGEST_RULE = (
    "anchor arm_sha256 = anchor_digest(model) = state_fingerprint(model): sha256 over the sorted names of EVERY "
    "parameter and buffer of the model (named_parameters and named_buffers, remove_duplicate=False: the tied "
    "lm_head.weight alias and the non-persistent model.rotary_emb.inv_freq included) of name + NUL + "
    "arm_build.tensor_digest(tensor) + LF, i.e. arm_build.state_digest of that full name -> tensor map, FP32, "
    "as the tensors sit on the device. It is taken on the REBUILT model: Supa loaded by the S0 FP32 loader, then "
    "the delta file as read back by arm_build.read_delta added in FP32 in place (param.add_(delta), which is "
    "bitwise arm_build's fl32(arm + fl32(1.0 * delta))). Never on the trained FP32 master weights: "
    "fl32(base + fl32(final - base)) need not equal final.")
WARM_SUFFIX = '_WARM'

_PINNED: dict = {}


# ----------------------------------------------------------------------------------------------- runtime
def _device(device) -> str:
    require(type(device) is str and device in DEVICES, 'device must be cpu or cuda: ' + repr(device))
    return DEVICES[device]


def _receipt(value, stage: str, expected_sha: str | None = None) -> tuple[dict, str | None]:
    """A receipt given as a path (hashed, strict JSON) or as an already-parsed mapping."""
    if isinstance(value, Mapping):
        require(expected_sha is None, 'a pinned receipt must be passed as a path')
        receipt, receipt_sha = dict(value), None
    else:
        path = Path(value)
        receipt_sha = sha(path)
        require(expected_sha is None or receipt_sha == expected_sha, 'receipt drift: ' + str(path))
        receipt = read_json(path)
    require(receipt.get('stage') == stage, f'not a {stage} receipt')
    return receipt, receipt_sha


def installed_versions(names) -> dict:
    return {name: importlib.metadata.version(name) for name in names}


def assert_runtime(pins: Mapping = PINS) -> dict:
    """The registered interpreter and package versions, installed and imported."""
    import safetensors
    import tokenizers
    import torch
    import transformers
    require(Path(sys.executable).resolve() == Path(pins['python_executable']).resolve(),
            'wrong Python runtime: ' + sys.executable)
    require(platform.python_version() == pins['python'], 'Python version drift')
    installed = installed_versions(sorted(pins['packages']))
    require(installed == dict(pins['packages']), f'package version drift: {installed}')
    imported = {'safetensors': safetensors.__version__, 'tokenizers': tokenizers.__version__,
                'torch': torch.__version__, 'transformers': transformers.__version__}
    require(all(imported[name] == pins['packages'][name] for name in pins['packages']),
            f'imported package drift: {imported}')
    return {'python_executable': str(Path(sys.executable).resolve()), 'python': platform.python_version(),
            'packages': installed}


def configure(device) -> dict:
    """Section 16 determinism settings; on CUDA they must land before CUDA initialises."""
    import torch
    dev = _device(device)
    if dev != 'cpu':
        require(not torch.cuda.is_initialized(),
                'CUDA already initialised: CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts')
        require(torch.cuda.is_available(), 'CUDA unavailable')
    before = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(NUM_THREADS)
    fraction = None
    if dev != 'cpu':
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
        fraction = MEMORY_FRACTION
    settings = _settings()
    settings.update({'device': dev, 'memory_fraction': fraction, 'cublas_workspace_config_before': before})
    _require_configured(settings)
    return settings


def _settings() -> dict:
    import torch
    return {'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
            'deterministic_warn_only': bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            'tf32_cuda_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
            'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'num_threads': torch.get_num_threads()}


def _require_configured(settings: Mapping) -> None:
    require(settings['cublas_workspace_config'] == CUBLAS_WORKSPACE_CONFIG, 'CUBLAS_WORKSPACE_CONFIG not set')
    require(settings['deterministic_algorithms'] and not settings['deterministic_warn_only'],
            'deterministic algorithms not enforced')
    require(not settings['tf32_cuda_matmul'] and not settings['tf32_cudnn'], 'TF32 enabled')
    require(settings['float32_matmul_precision'] == 'highest', 'FP32 matmul precision not highest')
    require(settings['num_threads'] == NUM_THREADS, 'thread count not the registered one')


def _cuda_identity() -> dict:
    """GPU name, capability, driver and CUDA/cuDNN builds, and the SDPA backend flags (initialises CUDA)."""
    import torch
    run = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=30, check=True)
    lines = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, 'expected exactly one NVIDIA GPU')
    smi_name, driver = (part.strip() for part in lines[0].rsplit(',', 1))
    return {'device_name': torch.cuda.get_device_name(0), 'nvidia_smi_name': smi_name,
            'driver_version': driver, 'capability': list(torch.cuda.get_device_capability(0)),
            'cuda_build': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
            'sdp_backends': {'flash': bool(torch.backends.cuda.flash_sdp_enabled()),
                             'mem_efficient': bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                             'math': bool(torch.backends.cuda.math_sdp_enabled()),
                             'cudnn': bool(torch.backends.cuda.cudnn_sdp_enabled())}}


def decoding_preimage(device, *, attn_implementation: str = ATTN_IMPLEMENTATION) -> dict:
    """Section 8.2 preimage of decoding_sha256; refuses a runtime configure() has not set."""
    dev = _device(device)
    require(type(attn_implementation) is str and attn_implementation, 'attention implementation')
    settings = _settings()
    _require_configured(settings)
    return {'schema': 'COMP-RS-001-S0-DECODING-1', 'dtype': 'float32', 'device': dev,
            'attn_implementation': attn_implementation, **settings,
            'cuda': _cuda_identity() if dev != 'cpu' else None,
            'python': platform.python_version(), 'packages': installed_versions(sorted(PINS['packages'])),
            'decoding': dict(DECODING)}


def decoding_sha256(preimage: Mapping) -> str:
    require(isinstance(preimage, Mapping) and preimage.get('schema') == 'COMP-RS-001-S0-DECODING-1',
            'not a decoding preimage')
    return digest(dict(preimage))


# ------------------------------------------------------------------------------------------ pinned imports
def load_pinned(path, expected_sha: str, name: str) -> types.ModuleType:
    """Execute a module from `path` only if its bytes hash to expected_sha; the hashed bytes are executed."""
    location = Path(path).resolve(strict=True)
    key = (os.path.normcase(str(location)), expected_sha)
    if key in _PINNED:
        return _PINNED[key]
    source = location.read_bytes()
    require(hashlib.sha256(source).hexdigest() == expected_sha, 'pinned module drift: ' + str(location))
    module_name = '_comp_rs_pinned_' + name
    module = types.ModuleType(module_name)
    module.__file__ = str(location)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(location), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _PINNED[key] = module
    return module


def gpu_lease() -> types.ModuleType:
    module = load_pinned(GPU_LEASE_PATH, GPU_LEASE_SHA, 'gpu_lease')
    require(Path(module.LOCK).name == LOCK_NAME, 'GPU lease lock mismatch')
    return module


def hostbudget() -> types.ModuleType:
    return load_pinned(HOSTBUDGET_PATH, HOSTBUDGET_SHA, 'hostbudget')


@contextlib.contextmanager
def lease(label: str, timeout_s, *, module=None):
    """Hold the shared GPU lease for the block; the own token is released in finally, always."""
    require(type(label) is str and LEASE_LABEL.fullmatch(label) is not None,
            'lease label must name a COMP-RS-001 stage: ' + repr(label))
    require(type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0, 'lease timeout')
    lease_module = gpu_lease() if module is None else module
    require(Path(lease_module.LOCK).name == LOCK_NAME, 'GPU lease lock mismatch')
    token = lease_module.acquire(timeout_s=timeout_s, label=label)
    try:
        require(type(token) is str and token, 'GPU lease returned no token')
        yield token
    finally:
        lease_module.release(token)


def _gib(value, what: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, what + ' GiB must be > 0')
    return float(value)


def host_require(gib, label: str, *, module=None):
    """hostbudget.require, unmodified: refuses to start (SystemExit) below the floor."""
    require(type(label) is str and label, 'host label')
    return (hostbudget() if module is None else module).require(_gib(gib, 'launch floor'), label)


def host_guard(gib, label: str, *, module=None):
    """hostbudget.guard, unmodified: raises its LowMemoryAbort below the floor."""
    require(type(label) is str and label, 'host label')
    return (hostbudget() if module is None else module).guard(_gib(gib, 'unit floor'), label)


# ------------------------------------------------------------------------------------------------ digests
def tensor_sha256(t) -> str:
    """arm_build.tensor_digest, computed in CHUNK-element pieces (host transient <= one chunk)."""
    import torch
    require(isinstance(t, torch.Tensor) and t.device.type != 'meta', 'not a materialised tensor')
    require(sys.byteorder == 'little', 'big-endian host unsupported')
    header = json.dumps({'dtype': str(t.dtype), 'shape': list(t.shape)}, sort_keys=True, separators=(',', ':'))
    result = hashlib.sha256(header.encode('ascii') + b'\n')
    flat = t.detach().contiguous().reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        piece = flat[start:start + CHUNK].to('cpu')
        result.update(memoryview(piece.view(torch.uint8).numpy()))
        del piece
    return result.hexdigest()


def combine(digests: Mapping) -> str:
    """arm_build.state_digest over precomputed per-tensor digests: sorted key + '\\0' + digest + '\\n'."""
    require(isinstance(digests, Mapping) and digests, 'empty digest set')
    result = hashlib.sha256()
    for key in sorted(digests):
        require(type(key) is str and key and '\0' not in key and '\n' not in key, 'invalid tensor key')
        result.update(key.encode('utf-8') + b'\0' + digests[key].encode('ascii') + b'\n')
    return result.hexdigest()


def state_digests(model) -> dict:
    """name -> tensor_sha256 for every parameter and buffer name (aliases hashed once, one at a time)."""
    named = dict(model.named_parameters(remove_duplicate=False))
    for name, buffer in model.named_buffers(remove_duplicate=False):
        require(name not in named, 'parameter/buffer name collision: ' + name)
        named[name] = buffer
    require(named, 'model has no state')
    cache, digests = {}, {}
    for name in sorted(named):
        tensor = named[name]
        if id(tensor) not in cache:
            cache[id(tensor)] = tensor_sha256(tensor)
        digests[name] = cache[id(tensor)]
    return digests


def state_fingerprint(model) -> str:
    """Full-state fingerprint: sorted names of every parameter and buffer + their tensor_sha256."""
    return combine(state_digests(model))


# ------------------------------------------------------------------------------------------------- loader
def _model_assets(model_dir, asset_receipt) -> dict:
    receipt, receipt_sha = _receipt(asset_receipt, 'Q1_ASSET_CANDIDATE')
    root = Path(model_dir).resolve(strict=True)
    require(Path(receipt['model_path']).resolve() == root, 'asset receipt names another model directory')
    weights = root / 'model.safetensors'
    bound = receipt['assets']['model.safetensors']
    require(weights.stat().st_size == bound['bytes'] and sha(weights) == bound['sha256'],
            'model weight bytes drift (sha/size != asset receipt)')
    raw = (root / 'config.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == receipt['assets']['config.json']['sha256'],
            'config.json drift')
    config = json.loads(raw.decode('utf-8'))
    require(all(config.get(key) == receipt['architecture'][key] for key in ARCHITECTURE_KEYS),
            'config.json architecture != asset receipt')
    require(config['architectures'] == ['Qwen3ForCausalLM'] and config['model_type'] == 'qwen3'
            and config['tie_word_embeddings'] is True, 'not a tied Qwen3ForCausalLM')
    header = receipt['model_header']
    require(type(header['tensors']) is int and header['tensors'] > 0 and header['dtypes'] == ['BF16'],
            'asset receipt header')
    return {'root': root, 'weights': weights, 'config': config, 'receipt_sha256': receipt_sha,
            'weights_sha256': bound['sha256'], 'weights_bytes': bound['bytes'],
            'config_sha256': receipt['assets']['config.json']['sha256'], 'tensors': header['tensors']}


def _upcast(stored):
    import torch
    return stored.to(torch.float32)


def _check_keys(model, handle, expected_count: int) -> tuple[list, dict]:
    """Stored keys vs model slots: one-to-one, except the tied lm_head (a model slot, never stored)."""
    import torch
    stored = sorted(handle.keys())
    state = model.state_dict(keep_vars=True)
    require(len(stored) == expected_count, f'stored tensor count {len(stored)} != receipt {expected_count}')
    require(TIED[0] not in stored, 'lm_head.weight is stored: checkpoint is not tied')
    unexpected = sorted(set(stored) - set(state))
    require(not unexpected, 'unexpected stored tensor (no model slot): ' + ', '.join(unexpected[:3]))
    missing = sorted(set(state) - set(stored) - {TIED[0]})
    require(not missing, 'model tensor absent from checkpoint: ' + ', '.join(missing[:3]))
    for name in stored:
        piece = handle.get_slice(name)
        require(piece.get_dtype() == 'BF16', 'stored tensor not BF16: ' + name)
        require(list(piece.get_shape()) == list(state[name].shape), 'stored shape != model shape: ' + name)
        require(state[name].dtype == torch.float32, 'model tensor not FP32: ' + name)
    return stored, state


def _require_tied(model) -> dict:
    head = model.get_output_embeddings().weight
    embed = model.get_input_embeddings().weight
    state = model.state_dict(keep_vars=True)
    require(state[TIED[0]] is state[TIED[1]] and head is embed, 'lm_head is not the input embedding Parameter')
    require(head.untyped_storage().data_ptr() == embed.untyped_storage().data_ptr(),
            'lm_head does not share storage with the input embedding')
    return {'alias': TIED[0], 'of': TIED[1], 'same_parameter': True, 'same_storage': True}


def _identity(model, assets: dict, dev: str) -> dict:
    import torch
    from safetensors import safe_open
    tied = _require_tied(model)
    stored_digests = {}
    with safe_open(str(assets['weights']), framework='pt', device=dev) as handle:
        stored, state = _check_keys(model, handle, assets['tensors'])
        for name in stored:
            expected = handle.get_tensor(name)
            loaded = state[name].detach()
            require(loaded.device == torch.device(dev) and loaded.dtype == torch.float32,
                    'loaded tensor not FP32 on the device: ' + name)
            back = loaded.to(torch.bfloat16)
            require(expected.dtype == torch.bfloat16 and bool(torch.equal(back.view(torch.int16),
                                                                           expected.view(torch.int16))),
                    'loaded tensor differs from stored (BF16 bitwise): ' + name)
            stored_digests[name] = tensor_sha256(expected)
            del expected, back
    for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)):
        require(tensor.device == torch.device(dev), 'tensor off the device: ' + name)
    persistent = set(state)
    buffers = sorted(name for name, _ in model.named_buffers(remove_duplicate=False) if name not in persistent)
    digests = state_digests(model)
    # mechanics.parameter_inventory refuses an all-frozen model; a frozen eval model records None
    inventory = parameter_inventory(model) if any(p.requires_grad for p in model.parameters()) else None
    return {'model_class': type(model).__name__, 'model_path': str(assets['root']), 'device': dev,
            'dtype': 'float32', 'attn_implementation': model.config._attn_implementation,
            'asset_receipt_sha256': assets['receipt_sha256'], 'weights_sha256': assets['weights_sha256'],
            'weights_bytes': assets['weights_bytes'], 'config_sha256': assets['config_sha256'],
            'stored_tensors': len(stored), 'stored_dtype': 'BF16', 'tensors_matched_bitwise': len(stored),
            'stored_tensor_sha256': stored_digests, 'stored_sha256': combine(stored_digests),
            'arm_sha256': combine({name: digests[name] for name in stored}),
            'state_fingerprint': combine(digests), 'state_names': len(digests),
            'non_persistent_buffers': buffers, 'tied': tied, 'parameter_inventory': inventory,
            'digest_format': 'arm_build.tensor_digest / state_digest (chunked)'}


def verify_identity(model, model_dir, device, asset_receipt) -> dict:
    """T1 identity of an already-loaded model against its stored tensors (raises on any difference)."""
    return _identity(model, _model_assets(model_dir, asset_receipt), _device(device))


def load_model_fp32(model_dir, device, asset_receipt, *, attn_implementation: str = ATTN_IMPLEMENTATION):
    """Build on the device in FP32, stream the stored BF16 tensors in exactly, verify T1. (model, identity)."""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM
    dev = _device(device)
    require(attn_implementation in ('sdpa', 'eager'), 'attention implementation must be sdpa or eager')
    require(torch.get_default_dtype() == torch.float32, 'default dtype is not float32')
    assets = _model_assets(model_dir, asset_receipt)
    config = Qwen3Config(**assets['config'])
    config._attn_implementation = attn_implementation
    with torch.device(dev):
        model = Qwen3ForCausalLM(config)
    require(type(model) is Qwen3ForCausalLM and model.config._attn_implementation == attn_implementation,
            'unexpected model class or attention implementation')
    model.tie_weights()
    _require_tied(model)
    with safe_open(str(assets['weights']), framework='pt', device=dev) as handle:
        stored, state = _check_keys(model, handle, assets['tensors'])
        with torch.no_grad():
            for name in stored:
                tensor = handle.get_tensor(name)
                require(tensor.dtype == torch.bfloat16, 'stored tensor not BF16: ' + name)
                state[name].copy_(_upcast(tensor))
                del tensor
    del state
    model.eval()
    return model, _identity(model, assets, dev)


# ------------------------------------------------------------------------------------------------- anchor
def _hex64(value) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - HEX


def anchor_digest(model) -> str:
    """The theta0' arm digest under ANCHOR_DIGEST_RULE (the one convention; s0_warmstart calls this too)."""
    return state_fingerprint(model)


def add_delta(model, delta) -> dict:
    """Add an FP32 delta to the model's parameters in place, in FP32; nothing else is touched.

    Every key must name a model parameter (not a buffer; the tied alias lm_head.weight is refused, its storage
    is model.embed_tokens.weight), every tensor must be FP32, finite and of the parameter's shape. All keys are
    checked before the first add, so a refused delta leaves the model as it was."""
    import torch
    require(isinstance(delta, Mapping) and delta, 'empty delta')
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    for key in sorted(delta, key=str):
        require(type(key) is str and key, 'invalid delta key: ' + repr(key))
        require(key != TIED[0], 'delta key is the tied alias lm_head.weight (its storage is %s)' % TIED[1])
        require(key not in buffers, 'delta key is a buffer, not a parameter: ' + key)
        require(key in params, 'delta key is not a model parameter: ' + key)
        tensor, parameter = delta[key], params[key]
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32, 'delta not FP32: ' + key)
        require(tuple(tensor.shape) == tuple(parameter.shape), 'delta shape != parameter shape: ' + key)
        require(parameter.dtype == torch.float32, 'parameter not FP32: ' + key)
        require(bool(torch.isfinite(tensor).all()), 'nonfinite delta: ' + key)
    numel = 0
    with torch.no_grad():
        for key in sorted(delta):
            parameter = params[key]
            parameter.add_(delta[key].to(parameter.device))
            require(bool(torch.isfinite(parameter).all()), 'nonfinite anchored parameter: ' + key)
            numel += parameter.numel()
    return {'tensors': len(delta), 'numel': numel, 'keys_sha256': digest(sorted(delta)),
            'arithmetic': 'param.add_(delta) in FP32, in place, sorted key order'}


def load_anchor(model_dir, device, asset_receipt, *, delta_path=None, delta_sha256=None, arm_sha256=None,
                attn_implementation: str = ATTN_IMPLEMENTATION):
    """(model, identity) of the section 18 anchor: plain Supa, or Supa + a delta rebuilt and verified.

    Supa is loaded by load_model_fp32 (T1 identity against the stored tensors). With a delta: read_delta
    refuses a file whose data section does not reproduce delta_sha256; add_delta adds it in FP32; then
    anchor_digest(model) must equal arm_sha256 (ANCHOR_DIGEST_RULE), else arm_build.ArmRebuildStop (section 2:
    a rebuild must reproduce the digest, or STOP) and no model is returned. identity['anchor'] =
    {base_weights_sha256, delta_path, delta_sha256, arm_sha256}, the delta fields None for plain Supa. For an
    anchored model identity['arm_sha256'] (the stored names, the arm-builder convention) and
    identity['state_fingerprint'] (= anchor arm_sha256) are recomputed on the anchored state; identity['base']
    keeps plain Supa's two digests, and every other field describes the Supa load."""
    given = [value is not None for value in (delta_path, delta_sha256, arm_sha256)]
    require(all(given) or not any(given), 'an anchor delta needs delta_path, delta_sha256 and arm_sha256 together')
    if any(given):
        require(isinstance(delta_path, (str, Path)) and str(delta_path), 'anchor delta path')
        require(_hex64(delta_sha256), 'anchor delta_sha256 must be a lowercase sha256 hex digest')
        require(_hex64(arm_sha256), 'anchor arm_sha256 must be a lowercase sha256 hex digest')
        require(Path(delta_path).is_file(), 'anchor delta file missing: ' + str(delta_path))
    model, identity = load_model_fp32(model_dir, device, asset_receipt, attn_implementation=attn_implementation)
    anchor = {'base_weights_sha256': identity['weights_sha256'], 'delta_path': None, 'delta_sha256': None,
              'arm_sha256': None}
    if not any(given):
        return model, dict(identity, anchor=anchor)
    from . import arm_build
    try:
        delta = arm_build.read_delta(delta_path, delta_sha256)
        applied = add_delta(model, delta)
        delta.clear()
        del delta
        digests = state_digests(model)
        actual = combine(digests)
        if actual != arm_sha256:
            raise arm_build.ArmRebuildStop(arm_sha256, {'arm_sha256': actual, 'rule': ANCHOR_DIGEST_RULE,
                                                        'delta_path': str(delta_path),
                                                        'delta_sha256': delta_sha256})
    except BaseException:
        del model
        raise
    stored = sorted(identity['stored_tensor_sha256'])
    anchor.update(delta_path=str(delta_path), delta_sha256=delta_sha256, arm_sha256=arm_sha256)
    return model, dict(identity, anchor=anchor, anchor_delta=applied, anchor_digest_rule=ANCHOR_DIGEST_RULE,
                       base={'arm_sha256': identity['arm_sha256'], 'state_fingerprint': identity['state_fingerprint'],
                             'note': 'plain Supa as loaded and T1-verified, before the delta'},
                       arm_sha256=combine({name: digests[name] for name in stored}), state_fingerprint=actual)


def anchored_name(name: str, anchored: bool) -> str:
    """A stage file name for the anchor: 'S0_COMPETENCE.json' -> 'S0_COMPETENCE_WARM.json' when anchored."""
    require(type(name) is str and name.endswith('.json') and len(name) > 5 and '/' not in name and '\\' not in name,
            'stage file name')
    require(type(anchored) is bool, 'anchored flag')
    stem = name[:-5]
    require(not stem.endswith(WARM_SUFFIX), 'the base stage name already carries ' + WARM_SUFFIX)
    return stem + WARM_SUFFIX + '.json' if anchored else name


# ---------------------------------------------------------------------------------------------- tokenizer
def load_tokenizer(model_dir, tokenizer_receipt, *, asset_receipt=None):
    """Local files only, no remote code; template sha, class and IDs must equal the tokenizer receipt."""
    from transformers import AutoTokenizer
    receipt, _ = _receipt(tokenizer_receipt, 'Q1_TOKENIZER_PROMPT_LENGTHS')
    root = Path(model_dir).resolve(strict=True)
    stray = [name for name in UNBOUND_TOKENIZER_FILES if (root / name).exists()]
    require(not stray, 'unbound tokenizer file present: ' + ', '.join(stray))
    if asset_receipt is not None:
        assets, assets_sha = _receipt(asset_receipt, 'Q1_ASSET_CANDIDATE')
        require(assets_sha is not None and assets_sha == receipt['asset_receipt_sha256'],
                'asset receipt is not the one the tokenizer receipt binds')
        require(Path(assets['model_path']).resolve() == root, 'asset receipt names another model directory')
        for name in TOKENIZER_ASSETS:
            require(sha(root / name) == assets['assets'][name]['sha256'], 'tokenizer asset drift: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    template = tokenizer.chat_template
    require(type(template) is str and hashlib.sha256(template.encode('utf-8')).hexdigest()
            == receipt['chat_template_sha256'], 'chat template drift')
    require(type(tokenizer).__name__ == receipt['tokenizer_class'], 'tokenizer class drift')
    require(tokenizer.model_max_length == receipt['tokenizer_model_max_length'], 'tokenizer max length drift')
    require(_token_ids(tokenizer) == receipt['tokenizer_ids'], 'tokenizer special IDs drift')
    require(list(receipt['generation_eos_ids']) == list(EOS_IDS) and receipt['tokenizer_ids']['pad'] == PAD_ID,
            'receipt EOS/PAD != registered decoding')
    require(tokenizer(THINK_PREFIX, add_special_tokens=False)['input_ids'] == list(THINK_PREFIX_IDS),
            'think-prefix tokenization drift')
    return tokenizer


def _token_ids(tokenizer) -> dict:
    return {'bos': tokenizer.bos_token_id, 'eos': tokenizer.eos_token_id, 'pad': tokenizer.pad_token_id,
            'im_start': tokenizer.convert_tokens_to_ids('<|im_start|>'),
            'im_end': tokenizer.convert_tokens_to_ids('<|im_end|>'),
            'think': tokenizer.convert_tokens_to_ids('<think>'),
            'end_think': tokenizer.convert_tokens_to_ids('</think>')}


def render_prompt(tokenizer, messages) -> tuple[str, list]:
    """(rendered text, input_ids) for policy messages, under the Q1 rendering rule."""
    require(isinstance(messages, list) and all(type(m) is dict for m in messages)
            and [m.get('role') for m in messages] == POLICY_ROLES, 'policy messages must be [system, user]')
    require(all(set(m) == {'role', 'content'} and type(m['content']) is str and m['content'] for m in messages),
            'policy message fields')
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    require(type(rendered) is str and rendered.endswith(THINK_PREFIX),
            'rendered prompt does not end with the pre-closed think block')
    ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    require(type(ids) is list and ids and all(type(t) is int for t in ids), 'encoding')
    require(ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False), 'render/encode mismatch')
    require(tuple(ids[-len(THINK_PREFIX_IDS):]) == THINK_PREFIX_IDS, 'think-prefix token suffix')
    marks = _token_ids(tokenizer)
    counts = {key: ids.count(marks[key]) for key in ('im_start', 'im_end', 'think', 'end_think')}
    require(counts == {'im_start': len(messages) + 1, 'im_end': len(messages), 'think': 1, 'end_think': 1},
            f'duplicate/missing chat delimiters: {counts}')
    others = set(tokenizer.all_special_ids) - {marks['im_start'], marks['im_end']}
    others |= {i for i, token in tokenizer.added_tokens_decoder.items() if token.special} - {
        marks['im_start'], marks['im_end']}
    present = sorted(others & set(ids))
    require(not present, f'unexpected special token(s) in prompt: {present}')
    return rendered, ids


def render(tokenizer, messages) -> list:
    return render_prompt(tokenizer, messages)[1]


# ------------------------------------------------------------------------------------------------ records
def assert_outside_checkout(path, repo) -> Path:
    target = Path(path).resolve()
    root = Path(repo).resolve(strict=True)
    folded, base = os.path.normcase(str(target)), os.path.normcase(str(root)).rstrip('\\/')
    require(folded != base and not folded.startswith(base + os.sep),
            'output inside the bound checkout: ' + str(target))
    return target


def _git(root: Path, *args) -> str:
    run = subprocess.run(['git', '--no-optional-locks', '-C', str(root), *args], capture_output=True,
                         text=True, encoding='utf-8', timeout=120)
    require(run.returncode == 0, f'git {args[0]} failed: {run.stderr.strip()}')
    return run.stdout


def assert_clean_checkout(repo) -> dict:
    """`git status --porcelain --untracked-files=all` must be empty (read-only git calls)."""
    root = Path(repo).resolve(strict=True)
    top = Path(_git(root, 'rev-parse', '--show-toplevel').strip()).resolve()
    require(os.path.normcase(str(top)) == os.path.normcase(str(root)), 'repo is not a checkout root')
    status = _git(root, 'status', '--porcelain', '--untracked-files=all')
    require(status == '', 'checkout not clean: ' + '; '.join(status.splitlines()[:5]))
    head = _git(root, 'rev-parse', '--verify', 'HEAD').strip()
    return {'repo': str(root), 'head': head, 'clean': True}


def write_receipt(path, value, *, repo) -> str:
    """Create-only JSON receipt outside the checkout, LF line endings; returns the sha256 of its bytes."""
    target = assert_outside_checkout(path, repo)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------- resources
def resources(device='cpu', *, module=None) -> dict:
    """Free physical RAM, this worker's RSS and peak working set, and CUDA peaks when on CUDA."""
    import psutil
    dev = _device(device)
    snapshot = (hostbudget() if module is None else module).snapshot()
    memory = psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', None)
    out = {'host': snapshot, 'free_phys_gib': snapshot.get('free_phys_gib'),
           'process_rss_gib': memory.rss / GIB, 'process_peak_rss_gib': None if peak is None else peak / GIB,
           'cuda': None}
    if dev != 'cpu':
        import torch
        free, total = torch.cuda.mem_get_info(0)
        out['cuda'] = {'max_memory_allocated_gib': torch.cuda.max_memory_allocated(0) / GIB,
                       'max_memory_reserved_gib': torch.cuda.max_memory_reserved(0) / GIB,
                       'memory_allocated_gib': torch.cuda.memory_allocated(0) / GIB,
                       'memory_reserved_gib': torch.cuda.memory_reserved(0) / GIB,
                       'free_gib': free / GIB, 'total_gib': total / GIB,
                       'fraction_cap_gib': MEMORY_FRACTION * total / GIB}
    return out
