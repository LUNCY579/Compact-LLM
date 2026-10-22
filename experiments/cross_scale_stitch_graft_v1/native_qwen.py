"""Explicit *unqualified* adapter candidate for already-loaded stock Qwen3, 4.51.3.

No downloads or from_pretrained. No custom Supa-A1.1 wrapper support. Do not use
this adapter as a native receipt: Transformers is absent in the delivery runtime.
"""
from __future__ import annotations

from torch import nn
from .contracts import Identity, require
from .modules import Decoder, ResidualBlock, frozen


class QwenBlock(ResidualBlock):
    def __init__(self, block: nn.Module, rotary: nn.Module):
        super().__init__()
        frozen(block)
        frozen(rotary)
        self.block, self.rotary, self.width = block, rotary, block.hidden_size
        self.eval()

    def attention_residual(self, h, mask, positions):
        pe = self.rotary(h, positions)
        delta, _ = self.block.self_attn(
            hidden_states=self.block.input_layernorm(h), attention_mask=mask,
            position_embeddings=pe, position_ids=positions, cache_position=positions[0],
            past_key_value=None, output_attentions=False, use_cache=False)
        return h + delta

    def ffn_delta(self, post_attention_raw):
        return self.block.mlp(self.block.post_attention_layernorm(post_attention_raw))

    def forward(self, h, mask, positions):
        return self.block(h, attention_mask=mask, position_ids=positions,
                          position_embeddings=self.rotary(h, positions), cache_position=positions[0],
                          past_key_value=None, use_cache=False, output_attentions=False)[0]


def from_loaded_stock_qwen3(model: nn.Module, identity: Identity, *, max_tokens: int = 512) -> Decoder:
    """Accept caller-owned eval/frozen CPU FP32 modules; never silently unwrap A1.1.

    The caller must separately verify asset hashes, template/IDs, provenance,
    source parity and resource fit. Identity hashes here are metadata, not asset
    verification. This helper deliberately does not load or cast anything.
    """
    try:
        import transformers
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except ImportError as error:
        raise RuntimeError("native adapter unavailable: requires Transformers 4.51.3") from error
    require(transformers.__version__ == "4.51.3", "unreviewed Transformers version")
    require(type(model) is Qwen3ForCausalLM, "only exact stock Qwen3 class; custom/wrapped A1.1 is NOT interchangeable")
    require(not identity.fixture and identity.logit_policy == "dense_unmodified", "native dense identity required")
    require(identity.precision == "float32", "this candidate supports CPU float32 only")
    frozen(model)
    cfg = model.config
    require(cfg._attn_implementation == "eager", "only native eager attention reviewed")
    require(not cfg.use_sliding_window and cfg.rope_scaling is None, "SWA/scaled RoPE unsupported")
    require(cfg.attention_dropout == 0.0, "dropout must be zero")
    for value in list(model.parameters()) + list(model.buffers()):
        require(value.device.type == "cpu", "all tensors including nonpersistent RoPE buffers must be materialized on CPU")
        require(not value.is_floating_point() or str(value.dtype) == "torch.float32", "no implicit dtype conversion")
    require(0 < max_tokens <= cfg.max_position_embeddings, "position budget outside config")
    core = model.model
    blocks = [QwenBlock(block, core.rotary_emb) for block in core.layers]
    return Decoder(identity, core.embed_tokens, blocks, core.norm, model.lm_head, max_tokens=max_tokens)
