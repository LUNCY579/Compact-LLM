"""CPU-only loss/gradient parity for supervised-position language-model logits."""
import copy

import pytest
import torch

from sbhm import native, pipeline
from sbhm.protocol import Blocked


@pytest.mark.parametrize('arm', ['B', 'C', 'D'])
@pytest.mark.parametrize('mask', ['answer_and_eos', 'discontiguous', 'eos_only'])
def test_selected_recovery_matches_pinned_full_loss_and_all_gradients(arm, mask):
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '4.51.3':
        pytest.skip('requires pinned Transformers, no version fallback')
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(701)
    torch.set_num_threads(1)
    config = Qwen3Config(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, num_hidden_layers=5,
                        vocab_size=67, max_position_embeddings=128)
    config._attn_implementation = 'sdpa'
    source = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    native.install(source, {'layer': 3, 'window': 4, 'rank': 2, 'seed': 17}, arm)
    # Zero LoRA b would hide a-factor gradient differences. Test recovered factors.
    with torch.no_grad():
        for name, parameter in native.trainables(source).items():
            if name.endswith('.b'):
                parameter.copy_(torch.randn_like(parameter)*0.03)
    reference, selected = copy.deepcopy(source), copy.deepcopy(source)
    ids = torch.tensor([[9, 12, 7, 21, 32, 11, 5, 17, 22, 2]], dtype=torch.long)
    labels = ids.clone()
    labels[:, :5] = -100
    if mask == 'discontiguous':
        labels[:, 6:8] = -100
    elif mask == 'eos_only':
        labels[:, :-1] = -100
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    observed_head_shapes = []
    observed_logits = []
    handle = selected.lm_head.register_forward_pre_hook(
        lambda _module, arguments: observed_head_shapes.append(tuple(arguments[0].shape)))
    output_handle = selected.lm_head.register_forward_hook(
        lambda _module, _arguments, output: observed_logits.append(output.detach()))
    full = reference(input_ids=ids, labels=labels, use_cache=False)
    full.loss.backward()
    reduced_loss = pipeline.recovery_loss(selected, ids, labels)
    reduced_loss.backward()
    handle.remove()
    output_handle.remove()

    # The full teacher-forced prefix is unchanged; the vocabulary projection is
    # materialized only at predictors of the supervised next-token labels.
    assert observed_head_shapes == [(1, positions.numel(), config.hidden_size)]
    assert positions.numel() < ids.shape[1]
    torch.testing.assert_close(observed_logits[0], full.logits[:, positions, :], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reduced_loss, full.loss, atol=1e-6, rtol=1e-6)
    full_parameters, selected_parameters = native.trainables(reference), native.trainables(selected)
    assert full_parameters.keys() == selected_parameters.keys()
    for name, expected_parameter in full_parameters.items():
        actual_parameter = selected_parameters[name]
        assert expected_parameter.grad is not None and actual_parameter.grad is not None, name
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad,
                                   atol=2e-6, rtol=2e-5, msg=lambda message: f'{name}: {message}')
        assert torch.isfinite(actual_parameter.grad).all(), name
        assert actual_parameter.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in selected.parameters() if not parameter.requires_grad)
    assert all(parameter.grad is None for parameter in reference.parameters() if not parameter.requires_grad)
    # Match the experiment's clip and AdamW recipe, not merely a scalar objective.
    full_norm = torch.nn.utils.clip_grad_norm_(list(full_parameters.values()), 1.0, error_if_nonfinite=True)
    selected_norm = torch.nn.utils.clip_grad_norm_(list(selected_parameters.values()), 1.0, error_if_nonfinite=True)
    torch.testing.assert_close(selected_norm, full_norm, atol=1e-6, rtol=1e-6)
    recipe = dict(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False)
    torch.optim.AdamW(list(full_parameters.values()), **recipe).step()
    torch.optim.AdamW(list(selected_parameters.values()), **recipe).step()
    for name, expected_parameter in full_parameters.items():
        torch.testing.assert_close(selected_parameters[name], expected_parameter,
                                   atol=2e-7, rtol=2e-6, msg=lambda message: f'{name}: {message}')


class MustNotForward:
    def __call__(self, **_kwargs):
        raise AssertionError('invalid labels must be rejected before model allocation')


@pytest.mark.parametrize('invalid', ['no_supervision', 'multibatch', 'mismatched_shape', 'rank_one'])
def test_selected_recovery_rejects_invalid_native_label_contract(invalid):
    ids = torch.tensor([[4, 5, 2]], dtype=torch.long)
    labels = torch.tensor([[-100, 5, 2]], dtype=torch.long)
    if invalid == 'no_supervision':
        labels.fill_(-100)
    elif invalid == 'multibatch':
        ids, labels = ids.repeat(2, 1), labels.repeat(2, 1)
    elif invalid == 'mismatched_shape':
        labels = labels[:, :-1]
    else:
        ids, labels = ids[0], labels[0]
    with pytest.raises(Blocked):
        pipeline.recovery_loss(MustNotForward(), ids, labels)
