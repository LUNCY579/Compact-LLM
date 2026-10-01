"""S0 runtime (Amendment 1 section 16): pins, determinism settings, pinned imports, lease, FP32 loader
identity on a tiny random Qwen3, digests, tokenizer rendering, decoding preimage, records and resources.

CPU only: no test creates a CUDA tensor or initialises CUDA (CUDA-only calls are replaced by fakes), no
test takes the real GPU lease, and the real Supa weights are never opened. The tokenizer tests read the
real tokenizer files and policy/calibration.jsonl only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.maxrl_comp002_pretraining_v1 import arm_build
from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, read_json, sha

REPO = Path(__file__).resolve().parents[3]
Q2_SELECTION = s0.HERE / 'evidence' / 'Q2_SOURCE_SMOKE_SELECTION.json'
# The real Supa config.json, verbatim (a test compares it with the file); TINY overrides only the sizes.
REAL_CONFIG = {
    'architectures': ['Qwen3ForCausalLM'], 'attention_bias': False, 'attention_dropout': 0.0,
    'bos_token_id': 151643, 'eos_token_id': 151645, 'head_dim': 128, 'hidden_act': 'silu',
    'hidden_size': 1024, 'initializer_range': 0.02, 'intermediate_size': 3072,
    'max_position_embeddings': 40960, 'max_window_layers': 28, 'model_type': 'qwen3',
    'num_attention_heads': 16, 'num_hidden_layers': 28, 'num_key_value_heads': 8, 'rms_norm_eps': 1e-06,
    'rope_scaling': None, 'rope_theta': 1000000, 'sliding_window': None, 'tie_word_embeddings': True,
    'torch_dtype': 'bfloat16', 'transformers_version': '4.51.0', 'use_cache': True,
    'use_sliding_window': False, 'vocab_size': 151936}
TINY = {**REAL_CONFIG, 'head_dim': 8, 'hidden_size': 32, 'intermediate_size': 64, 'max_window_layers': 2,
        'num_attention_heads': 4, 'num_hidden_layers': 2, 'num_key_value_heads': 2}
LABEL = 'comp-rs-001-s0-q1'


# ------------------------------------------------------------------------------------------------ helpers
@pytest.fixture(scope='module')
def tiny_state():
    """Random BF16 tensors for every stored name of the tiny Qwen3 (norms random too, so a skipped copy shows)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.random.fork_rng(devices=[]):
        model = Qwen3ForCausalLM(Qwen3Config(**TINY))
    shapes = {name: tuple(t.shape) for name, t in model.state_dict().items() if name != 'lm_head.weight'}
    del model
    generator = torch.Generator().manual_seed(20260927)
    return {name: torch.randn(shape, generator=generator).to(torch.bfloat16)
            for name, shape in sorted(shapes.items())}


def receipt_for(root: Path, tensors: int) -> dict:
    weights = root / 'model.safetensors'
    return {'stage': 'Q1_ASSET_CANDIDATE', 'model_path': str(root),
            'assets': {'model.safetensors': {'bytes': weights.stat().st_size, 'sha256': sha(weights)},
                       'config.json': {'bytes': (root / 'config.json').stat().st_size,
                                       'sha256': sha(root / 'config.json')}},
            'architecture': {key: TINY[key] for key in s0.ARCHITECTURE_KEYS},
            'model_header': {'tensors': tensors, 'dtypes': ['BF16']}}


def write_tiny(root: Path, state: dict) -> dict:
    root.mkdir(parents=True)
    (root / 'config.json').write_bytes((json.dumps(TINY, indent=2) + '\n').encode('ascii'))
    save_file(state, str(root / 'model.safetensors'), metadata={'format': 'pt'})
    return receipt_for(root, len(state))


@pytest.fixture
def torch_state():
    """Restore the process-global torch settings that configure() changes."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_num_threads(), os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.set_float32_matmul_precision(saved[2])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[3], saved[4]
    torch.set_num_threads(saved[5])
    if saved[6] is None:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = saved[6]


class FakeLease:
    def __init__(self, lock_name=s0.LOCK_NAME, token='token-1'):
        self.LOCK = Path('Z:/fake-temp') / lock_name
        self.token = token
        self.events = []

    def acquire(self, timeout_s, label):
        self.events.append(('acquire', timeout_s, label))
        return self.token

    def release(self, token):
        self.events.append(('release', token))


# ---------------------------------------------------------------------------------------------- runtime pins
def test_pins_equal_the_asset_and_tokenizer_receipts():
    assert sha(s0.ASSET_RECEIPT) == s0.ASSET_RECEIPT_SHA
    assert sha(s0.TOKENIZER_RECEIPT) == s0.TOKENIZER_RECEIPT_SHA
    asset = read_json(s0.ASSET_RECEIPT)
    candidate = asset['runtime_candidate']
    assert s0.PINS == {'python': candidate['python'], 'python_executable': candidate['python_executable'],
                       'packages': candidate['packages']}
    assert str(s0.MODEL_DIR) == asset['model_path']
    tokens = read_json(s0.TOKENIZER_RECEIPT)
    assert tokens['asset_receipt_sha256'] == s0.ASSET_RECEIPT_SHA
    assert tokens['generation_eos_ids'] == list(s0.EOS_IDS) == asset['generation_token_ids']['eos_token_id']
    assert tokens['tokenizer_ids']['pad'] == s0.PAD_ID == asset['generation_token_ids']['pad_token_id']
    assert tokens['response_budget_candidate'] == s0.MAX_NEW_TOKENS == s0.DECODING['max_new_tokens']
    assert s0._receipt(s0.ASSET_RECEIPT, 'Q1_ASSET_CANDIDATE', s0.ASSET_RECEIPT_SHA)[1] == s0.ASSET_RECEIPT_SHA
    with pytest.raises(ValueError, match='receipt drift'):
        s0._receipt(s0.ASSET_RECEIPT, 'Q1_ASSET_CANDIDATE', '0' * 64)
    with pytest.raises(ValueError, match='not a Q1_TOKENIZER_PROMPT_LENGTHS receipt'):
        s0._receipt(s0.ASSET_RECEIPT, 'Q1_TOKENIZER_PROMPT_LENGTHS')


def test_assert_runtime_checks_interpreter_python_and_every_package(tmp_path):
    here = {'python': s0.platform.python_version(), 'python_executable': sys.executable,
            'packages': s0.installed_versions(sorted(s0.PINS['packages']))}
    assert s0.assert_runtime(here)['packages'] == here['packages']
    with pytest.raises(ValueError, match='wrong Python runtime'):
        s0.assert_runtime({**here, 'python_executable': str(tmp_path / 'python.exe')})
    with pytest.raises(ValueError, match='Python version drift'):
        s0.assert_runtime({**here, 'python': '3.13.5'})
    for name in sorted(here['packages']):
        drifted = copy.deepcopy(here)
        drifted['packages'][name] += '.drift'
        with pytest.raises(ValueError, match='package version drift'):
            s0.assert_runtime(drifted)
    if Path(sys.executable).resolve() != Path(s0.PINS['python_executable']).resolve():
        pytest.skip('not running under the registered interpreter')
    assert s0.assert_runtime()['packages'] == s0.PINS['packages']


def test_configure_cpu_applies_the_registered_settings(torch_state, monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match='CUBLAS_WORKSPACE_CONFIG not set'):
        s0.decoding_preimage('cpu')
    settings = s0.configure('cpu')
    assert settings == {'device': 'cpu', 'memory_fraction': None, 'cublas_workspace_config_before': None,
                        'cublas_workspace_config': ':4096:8', 'deterministic_algorithms': True,
                        'deterministic_warn_only': False, 'tf32_cuda_matmul': False, 'tf32_cudnn': False,
                        'float32_matmul_precision': 'highest', 'num_threads': 4}
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 4
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match='device must be cpu or cuda'):
        s0.configure('mps')


def test_configure_cuda_refuses_once_cuda_is_initialised(torch_state, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda *a: calls.append(a))
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError, match='CUDA already initialised'):
        s0.configure('cuda')
    assert calls == [] and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ


def test_configure_cuda_caps_the_memory_fraction_on_device_0(torch_state, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction',
                        lambda fraction, device: calls.append((fraction, device)))
    settings = s0.configure('cuda')
    assert calls == [(0.78, 0)]
    assert settings['device'] == 'cuda:0' and settings['memory_fraction'] == 0.78
    assert settings['cublas_workspace_config'] == ':4096:8' and settings['num_threads'] == 4
    assert not torch.cuda._initialized


# ------------------------------------------------------------------------------------------ decoding preimage
def test_decoding_preimage_is_stable_and_binds_the_settings(torch_state):
    s0.configure('cpu')
    first, second = s0.decoding_preimage('cpu'), s0.decoding_preimage('cpu')
    assert first == second and s0.decoding_sha256(first) == s0.decoding_sha256(second)
    assert s0.decoding_sha256(json.loads(json.dumps(first))) == s0.decoding_sha256(first)
    assert first['decoding'] == {'batch_size': 1, 'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128,
                                 'eos_token_id': [151645, 151643], 'pad_token_id': 151643,
                                 'enable_thinking': False, 'add_generation_prompt': True,
                                 'add_special_tokens': False, 'greedy': True}
    assert (first['dtype'], first['num_threads'], first['cublas_workspace_config']) == ('float32', 4, ':4096:8')
    assert first['deterministic_algorithms'] is True and first['cuda'] is None
    assert first['tf32_cuda_matmul'] is False and first['tf32_cudnn'] is False
    assert first['packages'] == s0.installed_versions(sorted(s0.PINS['packages']))
    assert first['attn_implementation'] == 'sdpa'
    eager = s0.decoding_preimage('cpu', attn_implementation='eager')
    assert s0.decoding_sha256(eager) != s0.decoding_sha256(first)
    longer = copy.deepcopy(first)
    longer['decoding']['max_new_tokens'] = 256
    assert s0.decoding_sha256(longer) != s0.decoding_sha256(first)
    torch.set_num_threads(2)
    with pytest.raises(ValueError, match='thread count'):
        s0.decoding_preimage('cpu')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    with pytest.raises(ValueError, match='TF32'):
        s0.decoding_preimage('cpu')
    with pytest.raises(ValueError, match='not a decoding preimage'):
        s0.decoding_sha256({'dtype': 'float32'})


def test_decoding_preimage_on_cuda_binds_the_device_identity(torch_state, monkeypatch):
    s0.configure('cpu')
    fake = {'device_name': 'GPU-X', 'nvidia_smi_name': 'GPU-X', 'driver_version': '999.1',
            'capability': [8, 9], 'cuda_build': '12.6', 'cudnn_version': 91002,
            'sdp_backends': {'flash': True, 'mem_efficient': True, 'math': True, 'cudnn': True}}
    monkeypatch.setattr(s0, '_cuda_identity', lambda: dict(fake))
    cuda = s0.decoding_preimage('cuda')
    assert cuda['cuda'] == fake and cuda['device'] == 'cuda:0'
    assert s0.decoding_sha256(cuda) != s0.decoding_sha256(s0.decoding_preimage('cpu'))
    monkeypatch.setattr(s0, '_cuda_identity', lambda: {**fake, 'driver_version': '999.2'})
    assert s0.decoding_sha256(s0.decoding_preimage('cuda')) != s0.decoding_sha256(cuda)


# --------------------------------------------------------------------------------------------------- loader
def test_tiny_config_is_the_real_config_format():
    real = s0.MODEL_DIR / 'config.json'
    if not real.is_file():
        pytest.skip('Supa model directory absent')
    assert read_json(real) == REAL_CONFIG
    assert sha(real) == read_json(s0.ASSET_RECEIPT)['assets']['config.json']['sha256']


def test_tiny_bf16_round_trip_passes_t1_identity(tmp_path, tiny_state):
    from transformers import Qwen3ForCausalLM
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    model, identity = s0.load_model_fp32(root, 'cpu', receipt)
    assert identity['stored_tensors'] == identity['tensors_matched_bitwise'] == len(tiny_state) == 24
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
    assert not model.training and identity['attn_implementation'] == 'sdpa'
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert identity['tied'] == {'alias': 'lm_head.weight', 'of': 'model.embed_tokens.weight',
                                'same_parameter': True, 'same_storage': True}
    assert identity['non_persistent_buffers'] == ['model.rotary_emb.inv_freq']
    stored = load_file(str(root / 'model.safetensors'))
    state = model.state_dict()
    assert all(torch.equal(state[name], tensor.to(torch.float32)) for name, tensor in stored.items())
    # stored digests are the arm_build digests of the stored BF16 tensors, in sorted key order
    assert identity['stored_tensor_sha256'] == {name: arm_build.tensor_digest(t) for name, t in stored.items()}
    assert list(identity['stored_tensor_sha256']) == sorted(stored)
    assert identity['stored_sha256'] == arm_build.state_digest(stored)
    # the loaded FP32 state over the stored keys is the section 2 identity of arm B
    assert identity['arm_sha256'] == arm_build.build_arm_streaming(root / 'model.safetensors', [])[1]['arm_sha256']
    assert identity['state_fingerprint'] == s0.state_fingerprint(model)
    inventory = identity['parameter_inventory']
    assert inventory['unique_numel'] == sum(t.numel() for t in stored.values())
    assert ['lm_head.weight', 'model.embed_tokens.weight'] in [e['names'] for e in inventory['parameters']]
    _, again = s0.load_model_fp32(root, 'cpu', receipt)
    assert again == identity
    # an independent reference: transformers' own loader, FP32, same attention, same logits bit for bit
    reference = Qwen3ForCausalLM.from_pretrained(str(root), dtype=torch.float32, local_files_only=True,
                                                 attn_implementation='sdpa').eval()
    ids = torch.tensor([[151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]])
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, reference(input_ids=ids).logits)


def test_corrupted_weight_bytes_fail_the_receipt_sha(tmp_path, tiny_state):
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    path = root / 'model.safetensors'
    data = bytearray(path.read_bytes())
    data[-3] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='model weight bytes drift'):
        s0.load_model_fp32(root, 'cpu', receipt)


def test_a_corrupted_upcast_fails_t1_identity(tmp_path, tiny_state, monkeypatch):
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    real, calls = s0._upcast, []

    def corrupt_fifth(stored):
        out = real(stored)
        calls.append(1)
        if len(calls) == 5:
            out = out.clone()
            out.view(-1)[0] += 1.0
        return out

    monkeypatch.setattr(s0, '_upcast', corrupt_fifth)
    with pytest.raises(ValueError, match='loaded tensor differs from stored'):
        s0.load_model_fp32(root, 'cpu', receipt)


def test_verify_identity_detects_post_load_changes(tmp_path, tiny_state):
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    model, identity = s0.load_model_fp32(root, 'cpu', receipt)
    assert s0.verify_identity(model, root, 'cpu', receipt) == identity
    model.requires_grad_(False)
    frozen = s0.verify_identity(model, root, 'cpu', receipt)
    assert frozen['parameter_inventory'] is None
    assert {k: v for k, v in frozen.items() if k != 'parameter_inventory'} == {
        k: v for k, v in identity.items() if k != 'parameter_inventory'}
    model.requires_grad_(True)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 0] += 1.0
    with pytest.raises(ValueError, match='loaded tensor differs from stored'):
        s0.verify_identity(model, root, 'cpu', receipt)
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[0, 0] -= 1.0
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match='lm_head is not the input embedding'):
        s0.verify_identity(model, root, 'cpu', receipt)


@pytest.mark.parametrize('edit, match', [
    ('extra', 'unexpected stored tensor'),
    ('missing', 'model tensor absent from checkpoint'),
    ('lm_head', 'lm_head.weight is stored'),
    ('fp32', 'stored tensor not BF16'),
    ('shape', 'stored shape != model shape')])
def test_stored_key_set_and_dtypes_are_one_to_one(tmp_path, tiny_state, edit, match):
    state = dict(tiny_state)
    if edit == 'extra':
        state['model.layers.0.extra.weight'] = torch.zeros(2, dtype=torch.bfloat16)
    elif edit == 'missing':
        del state['model.norm.weight']
    elif edit == 'lm_head':
        state['lm_head.weight'] = state['model.embed_tokens.weight'].clone()
    elif edit == 'fp32':
        state['model.norm.weight'] = state['model.norm.weight'].to(torch.float32)
    else:
        state['model.norm.weight'] = torch.zeros(33, dtype=torch.bfloat16)
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, state)
    with pytest.raises(ValueError, match=match):
        s0.load_model_fp32(root, 'cpu', receipt)


def test_the_receipt_binds_count_directory_config_and_device(tmp_path, tiny_state):
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    with pytest.raises(ValueError, match='stored tensor count 24 != receipt 310'):
        s0.load_model_fp32(root, 'cpu', {**receipt, 'model_header': {'tensors': 310, 'dtypes': ['BF16']}})
    with pytest.raises(ValueError, match='asset receipt names another model directory'):
        s0.load_model_fp32(root, 'cpu', {**receipt, 'model_path': str(tmp_path / 'elsewhere')})
    with pytest.raises(ValueError, match='architecture'):
        s0.load_model_fp32(root, 'cpu', {**receipt, 'architecture': {**receipt['architecture'],
                                                                      'num_hidden_layers': 28}})
    with pytest.raises(ValueError, match='not a Q1_ASSET_CANDIDATE receipt'):
        s0.load_model_fp32(root, 'cpu', {**receipt, 'stage': 'Q1_TOKENIZER_PROMPT_LENGTHS'})
    with pytest.raises(ValueError, match='device must be cpu or cuda'):
        s0.load_model_fp32(root, 'meta', receipt)
    (root / 'config.json').write_bytes((json.dumps({**TINY, 'rms_norm_eps': 1e-05}, indent=2) + '\n').encode())
    with pytest.raises(ValueError, match='config.json drift'):
        s0.load_model_fp32(root, 'cpu', receipt)


# -------------------------------------------------------------------------------------------------- digests
def test_tensor_sha256_is_the_chunked_arm_build_digest(monkeypatch):
    generator = torch.Generator().manual_seed(1)
    cases = [torch.randn(7, 5, generator=generator), torch.randn(7, 5, generator=generator).to(torch.bfloat16),
             torch.randn(4, 6, generator=generator).t(), torch.tensor(3.0), torch.zeros(0, 3)]
    for chunk in (1, 3, 1 << 22):
        monkeypatch.setattr(s0, 'CHUNK', chunk)
        assert [s0.tensor_sha256(t) for t in cases] == [arm_build.tensor_digest(t) for t in cases]
    x = torch.arange(6, dtype=torch.float32)
    assert s0.tensor_sha256(x) != s0.tensor_sha256(x.reshape(2, 3))
    assert s0.tensor_sha256(x) != s0.tensor_sha256(x.view(torch.int32))
    assert s0.combine({'a': s0.tensor_sha256(x), 'b': s0.tensor_sha256(x[:2])}) == arm_build.state_digest(
        {'b': x[:2], 'a': x})


def test_state_fingerprint_covers_every_parameter_buffer_and_alias(tmp_path, tiny_state):
    root = tmp_path / 'tiny'
    model, identity = s0.load_model_fp32(root, 'cpu', write_tiny(root, tiny_state))
    digests = s0.state_digests(model)
    names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    names |= {n for n, _ in model.named_buffers(remove_duplicate=False)}
    assert set(digests) == names and 'model.rotary_emb.inv_freq' in digests
    assert digests['lm_head.weight'] == digests['model.embed_tokens.weight']
    base = s0.state_fingerprint(model)
    assert base == identity['state_fingerprint'] != identity['arm_sha256']
    for tensor in (model.model.rotary_emb.inv_freq, model.model.layers[0].self_attn.q_norm.weight):
        with torch.no_grad():
            tensor.view(-1)[0] += 1.0
        assert s0.state_fingerprint(model) != base
        with torch.no_grad():
            tensor.view(-1)[0] -= 1.0
        assert s0.state_fingerprint(model) == base


# --------------------------------------------------------------------------------------- anchor (section 18)
DELTA_KEYS = ('model.embed_tokens.weight', 'model.layers.1.mlp.down_proj.weight', 'model.norm.weight')


def _delta(tiny_state, keys=DELTA_KEYS, seed=7) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {key: torch.randn(tiny_state[key].shape, generator=generator) * 1e-2 for key in keys}


def _full_state(model) -> dict:
    named = dict(model.named_parameters(remove_duplicate=False))
    named.update(model.named_buffers(remove_duplicate=False))
    return {name: tensor.detach().clone() for name, tensor in named.items()}


@pytest.fixture
def anchor_case(tmp_path, tiny_state):
    """A tiny Supa, an FP32 delta over three of its parameters (the tied embedding included), and the expected
    anchor digest computed independently: plain FP32 state + delta, hashed with arm_build.state_digest."""
    root = tmp_path / 'tiny'
    receipt = write_tiny(root, tiny_state)
    delta = _delta(tiny_state)
    (tmp_path / 'delta').mkdir()
    path = tmp_path / 'delta' / 'delta_warm.safetensors'
    delta_sha = arm_build.write_delta(path, delta)
    plain, plain_identity = s0.load_model_fp32(root, 'cpu', receipt)
    expected = _full_state(plain)
    for name in expected:
        source = s0.TIED[1] if name == s0.TIED[0] else name
        if source in delta:
            expected[name] = expected[name] + delta[source]
    return SimpleNamespace(root=root, receipt=receipt, delta=delta, path=path, delta_sha=delta_sha, plain=plain,
                           plain_identity=plain_identity, expected=expected,
                           arm=arm_build.state_digest(expected))


def test_load_anchor_rebuilds_supa_plus_delta_with_both_digests(anchor_case):
    case = anchor_case
    model, identity = s0.load_anchor(case.root, 'cpu', case.receipt, delta_path=case.path,
                                     delta_sha256=case.delta_sha, arm_sha256=case.arm)
    assert identity['anchor'] == {'base_weights_sha256': case.receipt['assets']['model.safetensors']['sha256'],
                                  'delta_path': str(case.path), 'delta_sha256': case.delta_sha,
                                  'arm_sha256': case.arm}
    assert set(s0.ANCHOR_KEYS) == set(identity['anchor'])
    # one convention: the anchor digest is the full-state fingerprint of the rebuilt model
    assert s0.anchor_digest(model) == case.arm == identity['state_fingerprint'] == s0.state_fingerprint(model)
    assert case.arm != case.plain_identity['state_fingerprint']
    assert identity['anchor_digest_rule'] == s0.ANCHOR_DIGEST_RULE
    # the stored-name digest is the section 2 arm the streaming builder makes from the same files
    built = arm_build.build_arm_streaming(case.root / 'model.safetensors', [('W', case.path, 1.0)], order=('W',))
    assert identity['arm_sha256'] == built[1]['arm_sha256'] != case.plain_identity['arm_sha256']
    assert identity['base'] == {'arm_sha256': case.plain_identity['arm_sha256'],
                                'state_fingerprint': case.plain_identity['state_fingerprint'],
                                'note': 'plain Supa as loaded and T1-verified, before the delta'}
    assert identity['anchor_delta']['tensors'] == 3
    assert identity['anchor_delta']['keys_sha256'] == digest(sorted(DELTA_KEYS))
    # the delta tensors moved by exactly delta, the tied head followed its embedding, nothing else changed
    state = _full_state(model)
    assert set(state) == set(case.expected)
    assert all(torch.equal(state[name], case.expected[name]) for name in state)
    plain_digests, digests = s0.state_digests(case.plain), s0.state_digests(model)
    changed = sorted(name for name in digests if digests[name] != plain_digests[name])
    assert changed == sorted(set(DELTA_KEYS) | {s0.TIED[0]})
    assert model.lm_head.weight is model.model.embed_tokens.weight and not model.training
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # every other identity field is the Supa load's
    skip = {'anchor', 'anchor_delta', 'anchor_digest_rule', 'base', 'arm_sha256', 'state_fingerprint'}
    assert {k: v for k, v in identity.items() if k not in skip} == {
        k: v for k, v in case.plain_identity.items() if k not in skip}


def test_load_anchor_without_a_delta_is_plain_supa(anchor_case):
    case = anchor_case
    model, identity = s0.load_anchor(case.root, 'cpu', case.receipt)
    assert identity == dict(case.plain_identity, anchor={
        'base_weights_sha256': case.plain_identity['weights_sha256'], 'delta_path': None, 'delta_sha256': None,
        'arm_sha256': None})
    assert s0.state_fingerprint(model) == case.plain_identity['state_fingerprint']
    for partial in ({'delta_path': case.path}, {'delta_path': case.path, 'delta_sha256': case.delta_sha},
                    {'arm_sha256': case.arm}):
        with pytest.raises(ValueError, match='together'):
            s0.load_anchor(case.root, 'cpu', case.receipt, **partial)


@pytest.mark.parametrize('edit, error, match', [
    ('delta_sha', ValueError, 'data-section digest mismatch'),
    ('arm_sha', arm_build.ArmRebuildStop, 'STOP'),
    ('unknown', ValueError, 'not a model parameter'),
    ('buffer', ValueError, 'is a buffer'),
    ('alias', ValueError, 'tied alias'),
    ('shape', ValueError, 'delta shape != parameter shape'),
    ('missing', ValueError, 'delta file missing'),
    ('digest_format', ValueError, 'lowercase sha256')])
def test_load_anchor_refuses(anchor_case, tmp_path, edit, error, match):
    case = anchor_case
    kwargs = {'delta_path': case.path, 'delta_sha256': case.delta_sha, 'arm_sha256': case.arm}
    stray = {'unknown': ('model.layers.0.extra.weight', torch.zeros(4)),
             'buffer': ('model.rotary_emb.inv_freq', torch.zeros(4)),
             'alias': ('lm_head.weight', torch.zeros(case.delta['model.embed_tokens.weight'].shape)),
             'shape': ('model.norm.weight', torch.zeros(33))}
    if edit == 'delta_sha':
        kwargs['delta_sha256'] = '0' * 64
    elif edit == 'arm_sha':
        kwargs['arm_sha256'] = case.plain_identity['state_fingerprint']
    elif edit == 'missing':
        kwargs['delta_path'] = tmp_path / 'absent.safetensors'
    elif edit == 'digest_format':
        kwargs['arm_sha256'] = case.arm.upper()
    else:
        key, tensor = stray[edit]
        path = tmp_path / ('delta_%s.safetensors' % edit)
        kwargs['delta_sha256'] = arm_build.write_delta(path, dict(case.delta, **{key: tensor}))
        kwargs['delta_path'] = path
    with pytest.raises(error, match=match) as raised:
        s0.load_anchor(case.root, 'cpu', case.receipt, **kwargs)
    if edit == 'arm_sha':
        assert raised.value.actual == case.arm and raised.value.expected == kwargs['arm_sha256']


def test_add_delta_checks_every_key_before_the_first_add(anchor_case):
    case = anchor_case
    before = s0.state_fingerprint(case.plain)
    good = case.delta['model.norm.weight']              # sorts before the bad key: it would be added first
    for bad in ({'model.norm.weight': good, 'model.zz.weight': torch.zeros(3)},
                {'model.norm.weight': good, 'model.zz.weight': torch.zeros(3, dtype=torch.float64)},
                {'model.norm.weight': good, 'model.zz.weight': torch.full((3,), float('nan'))},
                {'model.norm.weight': good, 'model.norm.weight2': torch.zeros(3)},
                {'model.norm.weight': good.to(torch.bfloat16)},
                {'model.norm.weight': torch.full(good.shape, float('inf'))}, {}):
        with pytest.raises(ValueError):
            s0.add_delta(case.plain, bad)
        assert s0.state_fingerprint(case.plain) == before


def test_anchored_name_adds_the_warm_suffix():
    assert s0.anchored_name('S0_COMPETENCE.json', True) == 'S0_COMPETENCE_WARM.json'
    assert s0.anchored_name('S0_COMPETENCE.json', False) == 'S0_COMPETENCE.json'
    assert s0.anchored_name('S0_ENROLL_RUN1.json', True) == 'S0_ENROLL_RUN1_WARM.json'
    for bad in (('S0_COMPETENCE_WARM.json', True), ('S0_COMPETENCE.jsonl', True), ('x/y.json', False),
                ('S0.json', 1)):
        with pytest.raises(ValueError):
            s0.anchored_name(*bad)


# ------------------------------------------------------------------------------------ pinned imports, lease
def test_load_pinned_checks_the_bytes_before_executing(tmp_path):
    good = tmp_path / 'good_mod.py'
    good.write_bytes(b'VALUE = 41 + 1\n')
    module = s0.load_pinned(good, hashlib.sha256(b'VALUE = 41 + 1\n').hexdigest(), 'test_good')
    assert module.VALUE == 42 and module.__file__ == str(good.resolve())
    bad = tmp_path / 'bad_mod.py'
    bad.write_bytes(b'raise SystemExit("executed")\n')
    with pytest.raises(ValueError, match='pinned module drift'):
        s0.load_pinned(bad, '0' * 64, 'test_bad')
    assert '_comp_rs_pinned_test_bad' not in sys.modules


def test_registered_pinned_modules_verify_without_touching_the_lock():
    if not (s0.GPU_LEASE_PATH.is_file() and s0.HOSTBUDGET_PATH.is_file()):
        pytest.skip('pinned programme modules absent')
    lease_module = s0.gpu_lease()
    assert Path(lease_module.LOCK).name == 'rineya-gpu-0.lock'
    assert callable(lease_module.acquire) and callable(lease_module.release)
    budget = s0.hostbudget()
    assert all(callable(getattr(budget, name)) for name in ('require', 'guard', 'snapshot'))


def test_lease_releases_its_own_token_on_exception():
    fake = FakeLease()
    with pytest.raises(RuntimeError, match='boom'):
        with s0.lease(LABEL, 60, module=fake) as token:
            assert token == 'token-1'
            raise RuntimeError('boom')
    assert fake.events == [('acquire', 60, LABEL), ('release', 'token-1')]
    fake = FakeLease()
    with s0.lease(LABEL, 7.5, module=fake):
        pass
    assert fake.events == [('acquire', 7.5, LABEL), ('release', 'token-1')]
    fake = FakeLease(token='')
    with pytest.raises(ValueError, match='no token'):
        with s0.lease(LABEL, 60, module=fake):
            pytest.fail('entered the block without a token')
    assert fake.events == [('acquire', 60, LABEL), ('release', '')]


@pytest.mark.parametrize('label, timeout, lock', [
    ('', 60, s0.LOCK_NAME), ('q1', 60, s0.LOCK_NAME), ('comp-rs-001-', 60, s0.LOCK_NAME),
    ('COMP-RS-001-S0', 60, s0.LOCK_NAME), (LABEL, 0, s0.LOCK_NAME), (LABEL, float('nan'), s0.LOCK_NAME),
    (LABEL, True, s0.LOCK_NAME), (LABEL, 60, 'other-gpu.lock')])
def test_lease_refuses_before_acquiring(label, timeout, lock):
    fake = FakeLease(lock_name=lock)
    with pytest.raises(ValueError):
        with s0.lease(label, timeout, module=fake):
            pytest.fail('lease entered')
    assert fake.events == []


def test_host_budget_calls_delegate_unmodified():
    class Standdown(RuntimeError):
        pass

    calls = []

    def guard(gib, label):
        calls.append(('guard', gib, label))
        raise Standdown('below floor')

    fake = SimpleNamespace(require=lambda gib, label: calls.append(('require', gib, label)) or {'ok': True},
                           guard=guard)
    assert s0.host_require(2.5, LABEL, module=fake) == {'ok': True}
    with pytest.raises(Standdown, match='below floor'):
        s0.host_guard(1, LABEL, module=fake)
    assert calls == [('require', 2.5, LABEL), ('guard', 1.0, LABEL)]
    for bad in (0, -1.0, float('nan'), '2.5', True):
        with pytest.raises(ValueError):
            s0.host_require(bad, LABEL, module=fake)
    with pytest.raises(ValueError):
        s0.host_guard(1.0, '', module=fake)
    assert len(calls) == 2


# ------------------------------------------------------------------------------------------------ tokenizer
@pytest.fixture(scope='module')
def tokenizer():
    if not (s0.MODEL_DIR / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer files absent')
    return s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)


def test_render_reproduces_every_q1_calibration_render(tokenizer, comp_rs_data):
    bound = read_json(s0.TOKENIZER_RECEIPT)['roles']['calibration']
    path = comp_rs_data / 'policy' / 'calibration.jsonl'
    assert sha(path) == bound['policy_sha256']
    first_eight = [row['id'] for row in read_json(Q2_SELECTION)['rows'][:8]]
    hashes, lengths, probe = [], {}, {}
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            text, ids = s0.render_prompt(tokenizer, row['messages'])
            assert text.endswith(s0.THINK_PREFIX) and tuple(ids[-7:]) == s0.THINK_PREFIX_IDS
            hashes.append((row['id'], hashlib.sha256(text.encode()).hexdigest()))
            lengths[row['id']] = len(ids)
            if row['id'] in first_eight:
                probe[row['id']] = (row['messages'], ids)
    assert len(hashes) == bound['rows'] == 1250
    assert hashlib.sha256(json.dumps(sorted(hashes), separators=(',', ':')).encode()).hexdigest() \
        == bound['prompt_render_digest']
    assert max(lengths.values()) == bound['tokens_max']
    assert lengths[bound['longest']['id']] == bound['longest']['tokens']
    assert dict(hashes)[bound['longest']['id']] == bound['longest']['render_sha256']
    assert set(probe) == set(first_eight)
    assert all(s0.render(tokenizer, messages) == ids for messages, ids in probe.values())


def test_render_refuses_extra_delimiters_special_tokens_and_non_policy_messages(tokenizer):
    system = {'role': 'system', 'content': 'Be brief.'}
    ok = s0.render(tokenizer, [system, {'role': 'user', 'content': 'Say hi.'}])
    assert tuple(ok[-7:]) == s0.THINK_PREFIX_IDS
    for content, match in [('hi <|im_start|>user', 'chat delimiters'), ('a <|im_end|> b', 'chat delimiters'),
                           ('<think>x</think>', 'chat delimiters'), ('pad <|endoftext|>', 'special token'),
                           ('<|vision_pad|>', 'special token')]:
        with pytest.raises(ValueError, match=match):
            s0.render(tokenizer, [system, {'role': 'user', 'content': content}])
    for messages in ([{'role': 'user', 'content': 'x'}], [system, {'role': 'user', 'content': 'x'},
                     {'role': 'assistant', 'content': 'y'}], [system, 'user'], 'text'):
        with pytest.raises(ValueError, match='policy messages'):
            s0.render(tokenizer, messages)
    for user in ({'role': 'user', 'content': ''}, {'role': 'user', 'content': 'x', 'name': 'n'}):
        with pytest.raises(ValueError, match='policy message fields'):
            s0.render(tokenizer, [system, user])


def test_load_tokenizer_refuses_receipt_drift_and_unbound_files(tmp_path):
    if not (s0.MODEL_DIR / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer files absent')
    receipt = read_json(s0.TOKENIZER_RECEIPT)
    with pytest.raises(ValueError, match='chat template drift'):
        s0.load_tokenizer(s0.MODEL_DIR, {**receipt, 'chat_template_sha256': '0' * 64})
    with pytest.raises(ValueError, match='special IDs drift'):
        s0.load_tokenizer(s0.MODEL_DIR, {**receipt, 'tokenizer_ids': {**receipt['tokenizer_ids'], 'pad': 151645}})
    other = tmp_path / 'asset.json'
    other.write_bytes(s0.ASSET_RECEIPT.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='asset receipt is not the one the tokenizer receipt binds'):
        s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=other)
    stray = tmp_path / 'model'
    stray.mkdir()
    (stray / 'chat_template.jinja').write_bytes(b'{{ messages }}')
    with pytest.raises(ValueError, match='unbound tokenizer file'):
        s0.load_tokenizer(stray, s0.TOKENIZER_RECEIPT)


# -------------------------------------------------------------------------------------------------- records
def _git(root: Path, *args) -> str:
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout


def test_assert_clean_checkout_on_a_tmp_repo(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git absent')
    repo = tmp_path / 'repo'
    (repo / 'sub').mkdir(parents=True)
    _git(repo, 'init', '-q')
    (repo / 'a.txt').write_bytes(b'a\n')
    (repo / 'sub' / 'b.txt').write_bytes(b'b\n')
    _git(repo, 'add', 'a.txt', 'sub/b.txt')
    _git(repo, '-c', 'user.name=S0 Test', '-c', 'user.email=s0@example.invalid', 'commit', '-q', '-m', 'fixture')
    head = _git(repo, 'rev-parse', 'HEAD').strip()
    assert s0.assert_clean_checkout(repo) == {'repo': str(repo.resolve()), 'head': head, 'clean': True}
    index = repo / '.git' / 'index'
    before = index.read_bytes(), index.stat().st_mtime_ns
    os.utime(repo / 'a.txt', ns=(before[1] + 5_000_000_000,) * 2)  # stat-dirty, same content
    s0.assert_clean_checkout(repo)
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before  # read-only: the index is not refreshed
    (repo / 'sub' / 'untracked.txt').write_bytes(b'u\n')
    with pytest.raises(ValueError, match='checkout not clean'):
        s0.assert_clean_checkout(repo)
    (repo / 'sub' / 'untracked.txt').unlink()
    (repo / 'a.txt').write_bytes(b'changed\n')
    with pytest.raises(ValueError, match='checkout not clean'):
        s0.assert_clean_checkout(repo)
    (repo / 'a.txt').write_bytes(b'a\n')
    assert s0.assert_clean_checkout(repo)['clean'] is True
    with pytest.raises(ValueError, match='not a checkout root'):
        s0.assert_clean_checkout(repo / 'sub')
    with pytest.raises(ValueError, match='git rev-parse failed'):
        s0.assert_clean_checkout(tmp_path)


def test_assert_outside_checkout(tmp_path):
    repo = tmp_path / 'Repo'
    (repo / 'sub').mkdir(parents=True)
    for inside in (repo, repo / 'sub' / 'x.json', repo / 'new' / 'y.json', repo / 'sub' / '..' / 'z.json'):
        with pytest.raises(ValueError, match='inside the bound checkout'):
            s0.assert_outside_checkout(inside, repo)
    if os.name == 'nt':
        with pytest.raises(ValueError, match='inside the bound checkout'):
            s0.assert_outside_checkout(Path(str(repo).upper()) / 'x.json', repo)
    sibling = tmp_path / 'Repo-records' / 'x.json'
    assert s0.assert_outside_checkout(sibling, repo) == sibling.resolve()
    assert s0.assert_outside_checkout(s0.RECORDS / 'x.json', REPO) == (s0.RECORDS / 'x.json').resolve()
    with pytest.raises(ValueError, match='inside the bound checkout'):
        s0.assert_outside_checkout(s0.HERE / 'evidence' / 'x.json', REPO)


def test_write_receipt_is_create_only_lf_and_outside_the_checkout(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    target = tmp_path / 'records' / 's0' / 'r.json'
    value = {'b': 1, 'a': [1, 2], 'text': 'line\nbreak'}
    digest_hex = s0.write_receipt(target, value, repo=repo)
    raw = target.read_bytes()
    assert b'\r' not in raw and raw.endswith(b'}\n') and raw.count(b'\n') > 3
    assert hashlib.sha256(raw).hexdigest() == digest_hex and json.loads(raw) == value
    with pytest.raises(FileExistsError):
        s0.write_receipt(target, {'other': True}, repo=repo)
    assert target.read_bytes() == raw
    with pytest.raises(ValueError, match='inside the bound checkout'):
        s0.write_receipt(repo / 'r.json', value, repo=repo)
    with pytest.raises(ValueError):
        s0.write_receipt(tmp_path / 'nan.json', {'x': float('nan')}, repo=repo)
    assert not (repo / 'r.json').exists() and not (tmp_path / 'nan.json').exists()


# ------------------------------------------------------------------------------------------------ resources
def test_resources_on_cpu_record_host_and_worker_memory():
    fake = SimpleNamespace(snapshot=lambda: {'available': True, 'free_phys_gib': 3.25})
    out = s0.resources('cpu', module=fake)
    assert out['host'] == {'available': True, 'free_phys_gib': 3.25} and out['free_phys_gib'] == 3.25
    assert out['cuda'] is None and out['process_rss_gib'] > 0
    if os.name == 'nt':
        assert out['process_peak_rss_gib'] >= out['process_rss_gib']
    if s0.HOSTBUDGET_PATH.is_file() and os.name == 'nt':
        real = s0.resources('cpu')
        assert real['host']['available'] is True and real['free_phys_gib'] > 0


def test_resources_on_cuda_read_the_peaks(monkeypatch):
    values = {'max_memory_allocated': 3, 'max_memory_reserved': 4, 'memory_allocated': 1, 'memory_reserved': 2}
    for name, gib in values.items():
        monkeypatch.setattr(torch.cuda, name, lambda device, gib=gib: gib * s0.GIB)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (5 * s0.GIB, 8 * s0.GIB))
    out = s0.resources('cuda', module=SimpleNamespace(snapshot=lambda: {'available': False}))
    assert out['cuda'] == {'max_memory_allocated_gib': 3.0, 'max_memory_reserved_gib': 4.0,
                           'memory_allocated_gib': 1.0, 'memory_reserved_gib': 2.0, 'free_gib': 5.0,
                           'total_gib': 8.0, 'fraction_cap_gib': 0.78 * 8}
    assert out['free_phys_gib'] is None and not torch.cuda._initialized
