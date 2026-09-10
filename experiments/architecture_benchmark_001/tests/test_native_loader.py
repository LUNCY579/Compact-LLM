"""The actual run._load native path on a tiny RANDOM fixture, never official Supa qualification.

A tiny random Qwen3 checkpoint and a real tiny tokenizer are written to a temporary
directory. Native construction under `torch.device('meta')`, state assignment, weight
tying and the device transfer are the real ones from `run._load`; the only substitution
is the official B28 shape validator, because the registered official shape is 596M
parameters and cannot be a test fixture. No fixture fallback is added to the CLI, and
`source.source_config`/`audit_source` keep validating official sources unchanged.

Without the nonpersistent-buffer reconstruction in `run._load` these cases fail with
"Cannot copy out of meta tensor": `Qwen3RotaryEmbedding.inv_freq` is registered
`persistent=False`, so no checkpoint can supply it and `assign=True` leaves it on meta.
"""
import hashlib
import json
import pytest
import torch
transformers = pytest.importorskip('transformers', reason='native dependency absent; official G0 remains blocked')
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from experiments.architecture_benchmark_001 import run
from experiments.architecture_benchmark_001.budget import Config
from experiments.architecture_benchmark_001.gates import GateClosed, protocol
from experiments.architecture_benchmark_001.transfer import native_key_map

TINY = Config(vocab_size=37, hidden_size=24, intermediate_size=48, num_attention_heads=6,
              num_key_value_heads=3, head_dim=8, encoder_layers=4, decoder_layers=0,
              architecture='plain')
ATOL, RTOL = protocol()['G0']['native_reference_atol'], protocol()['G0']['native_reference_rtol']


def raw_config() -> dict:
    """A tiny config that still satisfies every official-shape invariant except size."""
    return {'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM'],
            'vocab_size': TINY.vocab_size, 'hidden_size': TINY.hidden_size,
            'intermediate_size': TINY.intermediate_size, 'num_hidden_layers': TINY.encoder_layers,
            'num_attention_heads': TINY.num_attention_heads,
            'num_key_value_heads': TINY.num_key_value_heads, 'head_dim': TINY.head_dim,
            'rms_norm_eps': TINY.rms_norm_eps, 'rope_theta': TINY.rope_theta,
            'max_position_embeddings': TINY.max_position_embeddings, 'tie_word_embeddings': True,
            'attention_bias': False, 'attention_dropout': 0.0, 'hidden_act': 'silu',
            'rope_scaling': None, 'sliding_window': None, 'use_sliding_window': False}


def native_config() -> Qwen3Config:
    cfg = Qwen3Config(**raw_config())
    cfg._attn_implementation = 'eager'
    return cfg


def build_source(source, dtype):
    """Write a tiny random source directory; return the same model built the ordinary way."""
    from safetensors.torch import save_file
    torch.manual_seed(17)
    ordinary = Qwen3ForCausalLM(native_config()).eval()
    state = ordinary.state_dict()
    # Tied checkpoints store no lm_head, exactly like the registered official inventory.
    save_file({k: state[k].detach().clone().to(dtype).contiguous() for k in native_key_map(TINY)},
              str(source/'model.safetensors'))
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
    vocab.update({f't{i}': i for i in range(3, TINY.vocab_size)})
    inner = Tokenizer(models.WordLevel(vocab=vocab, unk_token='t3'))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=inner, pad_token='<pad>', bos_token='<bos>',
                            eos_token='<eos>').save_pretrained(str(source))
    (source/'config.json').write_text(json.dumps(raw_config()))
    return ordinary


@pytest.fixture(scope='module')
def fixture_source(tmp_path_factory):
    source = tmp_path_factory.mktemp('tiny_native_source')
    return source, build_source(source, torch.float32)


@pytest.fixture
def loaded(fixture_source, monkeypatch):
    source, ordinary = fixture_source
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    reference, tokenizer, native = run._load(source, 'cpu', native=True)
    return source, ordinary, reference, tokenizer, native


def digests(folder) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_load_materializes_required_buffers_not_only_parameters(loaded):
    """Regression: `assign=True` reports "all keys matched" yet leaves inv_freq on meta."""
    _, _, _, _, native = loaded
    named = list(native.named_parameters()) + list(native.named_buffers())
    assert named, 'expected the native model to expose parameters and buffers'
    assert [n for n, t in named if t.is_meta] == []
    assert {str(t.device) for _, t in named} == {'cpu'}
    assert [n for n, _ in native.named_buffers()] == ['model.rotary_emb.inv_freq']
    # A plain attribute, so Module.to() never touches it; the rebuild must still set it.
    assert not native.model.rotary_emb.original_inv_freq.is_meta


def test_rebuilt_rotary_state_is_exactly_the_native_state(loaded):
    _, ordinary, _, _, native = loaded
    got, want = native.model.rotary_emb, ordinary.model.rotary_emb
    assert torch.equal(got.inv_freq, want.inv_freq)
    assert got.inv_freq.dtype == want.inv_freq.dtype
    assert torch.equal(got.original_inv_freq, want.original_inv_freq)
    assert got.attention_scaling == want.attention_scaling
    assert (got.rope_type, got.max_seq_len_cached, got.original_max_seq_len) == (
        want.rope_type, want.max_seq_len_cached, want.original_max_seq_len)
    assert 'model.rotary_emb.inv_freq' not in native.state_dict(), 'inv_freq must stay nonpersistent'


def test_loaded_weights_and_logits_match_ordinary_native_construction(loaded):
    """Assigned storage is not reallocated, so CPU BLAS may differ by ~1 ULP, never more."""
    _, ordinary, _, _, native = loaded
    assert set(native.state_dict()) == set(ordinary.state_dict())
    for key, value in native.state_dict().items():
        assert torch.equal(value, ordinary.state_dict()[key]), key
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(native(ids, use_cache=False).logits,
                                   ordinary(ids, use_cache=False).logits, atol=ATOL, rtol=RTOL)


def test_loaded_native_matches_the_dense_reference_unpadded_and_padded(loaded):
    _, _, reference, _, native = loaded
    for ids in (torch.tensor([[5, 6, 7, 8, 9]]), torch.tensor([[0, 0, 5, 6, 7]])):
        mask = (ids != 0).long()
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            a = native(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
            b = reference(ids, attention_mask=mask, position_ids=pos).logits
        torch.testing.assert_close(a[mask.bool()], b[mask.bool()], atol=ATOL, rtol=RTOL)


def test_native_reference_parity_in_the_exact_g0_call_shape(loaded):
    """qualify_source passes neither attention_mask nor position_ids; the two implicit
    conventions (HF cache_position vs the reference's cumsum) must already agree."""
    _, _, reference, _, native = loaded
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(native(ids, use_cache=False).logits,
                                   reference(ids).logits, atol=ATOL, rtol=RTOL)


def test_loaded_cached_continuation_matches_full_reforward(loaded):
    """The decode loop shape qualify_source actually runs, compared on logits.

    Comparing only the argmax ids is a vacuous oracle here: this tiny random fixture
    emits one constant token, so zeroed, frozen or dropped cache positions all keep the
    id sequence green. Those same faults move the step logits by 4e-2 to 3e-1, three
    orders of magnitude above this tolerance, so the logits are the real oracle.
    """
    _, _, _, _, native = loaded
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.inference_mode():
        past, current, sequence = None, ids, ids
        for _ in range(6):
            step = native(current, past_key_values=past, use_cache=True)
            torch.testing.assert_close(step.logits[:, -1],
                                       native(sequence, use_cache=False).logits[:, -1],
                                       atol=ATOL, rtol=RTOL)
            current = step.logits[:, -1].argmax(-1, keepdim=True)
            past = step.past_key_values
            sequence = torch.cat((sequence, current), 1)


def test_load_ties_embeddings_and_never_writes_the_source(fixture_source, monkeypatch):
    source, _ = fixture_source
    before = digests(source)
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    _, _, native = run._load(source, 'cpu', native=True)
    assert native.lm_head.weight.data_ptr() == native.model.embed_tokens.weight.data_ptr()
    assert digests(source) == before


def test_reference_only_load_needs_no_buffer_reconstruction(fixture_source, monkeypatch):
    """DenseReference registers no buffers at all, so the native=False path is unaffected."""
    source, _ = fixture_source
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    reference, _, native = run._load(source, 'cpu', native=False)
    assert native is None
    assert list(reference.named_buffers()) == []
    assert [n for n, p in reference.named_parameters() if p.is_meta] == []


def test_unknown_unmaterialized_buffer_fails_closed_instead_of_being_filled():
    """No zero fill, no to_empty(), no guess: an unrecognized meta buffer must stop the load."""
    class Mystery(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            with torch.device('meta'):
                self.register_buffer('mystery', torch.zeros(4), persistent=False)

    with pytest.raises(GateClosed, match='no declared initializer'):
        run._materialize_native_buffers(Mystery(), 'cpu')


def test_materialization_is_a_noop_when_nothing_is_left_on_meta():
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(native_config()).eval()
    before = model.model.rotary_emb.inv_freq
    assert run._materialize_native_buffers(model, 'cpu') == []
    assert model.model.rotary_emb.inv_freq is before


def test_materialization_reports_every_rebuilt_buffer():
    with torch.device('meta'):
        model = Qwen3ForCausalLM(native_config())
    assert run._materialize_native_buffers(model, 'cpu') == ['model.rotary_emb.inv_freq']
    assert not model.model.rotary_emb.inv_freq.is_meta


def test_load_upcasts_a_bfloat16_checkpoint_like_the_registered_source(tmp_path, monkeypatch):
    """source_lock.json pins the official tensors as BF16; the loader must upcast both models."""
    from safetensors import safe_open
    source = tmp_path/'bf16_source'
    source.mkdir()
    build_source(source, torch.bfloat16)
    with safe_open(str(source/'model.safetensors'), framework='pt', device='cpu') as f:
        assert {f.get_slice(k).get_dtype() for k in f.keys()} == {'BF16'}
    monkeypatch.setattr(run, 'source_config', lambda raw: TINY)
    reference, _, native = run._load(source, 'cpu', native=True)
    assert {t.dtype for _, t in native.named_parameters()} == {torch.float32}
    assert [n for n, t in native.named_buffers() if t.is_meta] == []
    assert {p.dtype for p in reference.parameters()} == {torch.float32}
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(native(ids, use_cache=False).logits,
                                   reference(ids).logits, atol=ATOL, rtol=RTOL)


def test_reference_and_native_rope_agree_across_the_registered_context():
    """The reference builds `theta**(-i/h)`; pinned Transformers builds `1/(theta**(i/h))`.

    Equal in real arithmetic, but they differ by up to 1.18e-7 relative in float32 and the
    angle error scales with absolute position. At the registered 512-token decoding context
    the gap stays about 3e-5, inside the G0 native/reference tolerance; it crosses that
    tolerance around position 4096. Pin the registered-context guarantee so that raising
    `decoding.max_context` cannot silently move G0 parity onto the wrong side of the gate.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from experiments.architecture_benchmark_001.budget import presets
    from experiments.architecture_benchmark_001.model import rope
    official = presets()['B28_1024_official_shape']
    assert official.head_dim == 128, 'the registered arms keep an explicit 128-dimensional head'
    half = official.head_dim // 2
    cfg = Qwen3Config(hidden_size=official.hidden_size, num_attention_heads=official.num_attention_heads,
                      num_key_value_heads=official.num_key_value_heads, head_dim=official.head_dim,
                      rope_theta=official.rope_theta, rope_scaling=None,
                      max_position_embeddings=official.max_position_embeddings)
    positions = torch.arange(protocol()['decoding']['max_context'])[None]
    cos, sin = Qwen3RotaryEmbedding(cfg)(torch.zeros(1, dtype=torch.float32), positions)
    # rope() on an all-ones input returns cos-sin in the first half and cos+sin in the second.
    out = rope(torch.ones(1, 1, positions.shape[1], official.head_dim), positions, official.rope_theta)[0, 0]
    torch.testing.assert_close((out[:, half:] + out[:, :half]) / 2, cos[0][:, :half], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close((out[:, half:] - out[:, :half]) / 2, sin[0][:, :half], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('damage', ['meta', 'device'])
def test_require_materialized_rejects_meta_and_foreign_devices(damage):
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(native_config()).eval()
    run._require_materialized(model, 'cpu')
    if damage == 'meta':
        with torch.device('meta'):
            model.register_buffer('broken', torch.zeros(2), persistent=False)
        with pytest.raises(GateClosed, match='unmaterialized on meta'):
            run._require_materialized(model, 'cpu')
    else:
        with pytest.raises(GateClosed, match='expected cuda'):
            run._require_materialized(model, 'cuda')
