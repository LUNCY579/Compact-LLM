"""Native Qwen3 parity on tiny RANDOM tensors, never official Supa qualification."""
import pytest
import torch
transformers = pytest.importorskip('transformers', reason='native dependency absent; official G0 remains blocked')
from transformers import Qwen3Config, Qwen3ForCausalLM
from experiments.architecture_benchmark_001.transfer import import_qwen_state
from experiments.architecture_benchmark_001.budget import Config


@pytest.mark.parametrize('padded', [False, True])
def test_native_random_qwen3_logits(padded):
    c=Config(vocab_size=37,hidden_size=24,intermediate_size=48,num_attention_heads=6,
             num_key_value_heads=3,head_dim=8,encoder_layers=4,decoder_layers=0,architecture='plain')
    cfg=Qwen3Config(vocab_size=c.vocab_size,hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,
        num_hidden_layers=c.encoder_layers,num_attention_heads=c.num_attention_heads,num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,rms_norm_eps=c.rms_norm_eps,rope_theta=c.rope_theta,tie_word_embeddings=True,
        attention_dropout=0.0,attention_bias=False,use_sliding_window=False,pad_token_id=0)
    cfg._attn_implementation='eager'
    torch.manual_seed(17); native=Qwen3ForCausalLM(cfg).eval()
    reference=import_qwen_state(native.state_dict(),c)
    ids=torch.tensor([[0,0,1,2,3]]) if padded else torch.tensor([[1,2,3,4,5]])
    mask=(ids!=0).long(); pos=(mask.cumsum(-1)-1).clamp_min(0)
    with torch.inference_mode():
        a=native(ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
        b=reference(ids,attention_mask=mask,position_ids=pos).logits
    torch.testing.assert_close(a[mask.bool()],b[mask.bool()],atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('position_start',[0,10000])
def test_native_rope_order_at_registered_length_and_long_absolute_positions(position_start):
    """Random weights: exercise frequency rounding, not official 32K quality."""
    c=Config(vocab_size=37,hidden_size=24,intermediate_size=48,num_attention_heads=6,
             num_key_value_heads=3,head_dim=8,encoder_layers=2,decoder_layers=0,architecture='plain')
    cfg=Qwen3Config(vocab_size=c.vocab_size,hidden_size=c.hidden_size,intermediate_size=c.intermediate_size,
        num_hidden_layers=c.encoder_layers,num_attention_heads=c.num_attention_heads,num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,rms_norm_eps=c.rms_norm_eps,rope_theta=c.rope_theta,tie_word_embeddings=True,
        attention_dropout=0.0,attention_bias=False,use_sliding_window=False,pad_token_id=0)
    cfg._attn_implementation='eager'
    torch.manual_seed(17);native=Qwen3ForCausalLM(cfg).eval();reference=import_qwen_state(native.state_dict(),c)
    ids=torch.randint(1,37,(1,512)); pos=torch.arange(position_start,position_start+512)[None]
    with torch.inference_mode():
        a=native(ids,position_ids=pos,use_cache=False).logits
        b=reference(ids,position_ids=pos).logits
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
