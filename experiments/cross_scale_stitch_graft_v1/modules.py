"""Composable decoder parts. No loader, hooks, cache, or implicit backbone updates."""
from __future__ import annotations

import hashlib
from typing import Sequence
from dataclasses import asdict
import torch
from torch import nn

from .alignment import AffineFit
from .contracts import Identity, digest, finite, positive_int, require, same_token_grid


def frozen(module: nn.Module) -> None:
    require(not module.training, "caller must supply eval-mode frozen components")
    require(all(not p.requires_grad for p in module.parameters()), "backbone is not frozen")


def fingerprint(module: nn.Module) -> str:
    """Fixture/small-module integrity helper; native full-weight hashing is separately budgeted."""
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        v = value.detach().cpu().contiguous()
        require(v.device.type != "meta", "cannot fingerprint meta state")
        h.update(key.encode() + str((tuple(v.shape), v.dtype)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Bridge(nn.Linear):
    @classmethod
    def from_fit(cls, fit: AffineFit, *, delta: bool = False,
                 dtype: torch.dtype = torch.float32) -> "Bridge":
        require(dtype in {torch.float32, torch.float64}, "mechanics bridges require float32/64")
        require(fit.weight.ndim == 2 and fit.bias.shape == (fit.weight.shape[0],), "invalid affine fit")
        require(torch.isfinite(fit.weight).all().item() and torch.isfinite(fit.bias).all().item(), "nonfinite fit")
        result = cls(fit.weight.shape[1], fit.weight.shape[0], bias=not delta, dtype=dtype)
        with torch.no_grad():
            result.weight.copy_(fit.weight)
            if result.bias is not None:
                result.bias.copy_(fit.bias)
        return result


class ResidualBlock(nn.Module):
    """Implement attention_residual and ffn_delta without hidden caches.

    attention_residual returns raw residual AFTER attention addition and BEFORE
    post-attention normalization. ffn_delta includes that native normalization.
    """
    width: int

    def attention_residual(self, h: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ffn_delta(self, post_attention_raw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        z = self.attention_residual(h, mask, positions)
        return z + self.ffn_delta(z)


class FrozenMode(nn.Module):
    def train(self, mode: bool = True):
        super().train(mode)
        # Only interface weights train. Frozen components remain deterministic.
        for module in self.frozen_components():
            module.eval()
        return self

    def frozen_components(self) -> Sequence[nn.Module]:
        return ()


class FFNGraft(FrozenMode):
    """Additional donor FFN, parallel to the intact host FFN at a raw post-attention site.

    Input bridge is affine; return bridge acts on a DELTA, hence has NO bias.
    No double-zero initialization: bypass uses a distinct explicit control flag.
    """
    def __init__(self, donor_norm: nn.Module, donor_ffn: nn.Module, input_bridge: Bridge,
                 return_bridge: Bridge, *, donor_identity: Identity, donor_block_index: int,
                 alpha: float = 0.05, train_alpha: bool = False):
        super().__init__()
        frozen(donor_norm)
        frozen(donor_ffn)
        require(isinstance(input_bridge, Bridge) and isinstance(return_bridge, Bridge), "explicit bridges required")
        require(input_bridge.in_features == return_bridge.out_features and
                input_bridge.out_features == return_bridge.in_features, "bridge dimensions disagree")
        require(return_bridge.bias is None, "delta return bridge must not include affine offset")
        finite(alpha, "alpha")
        require(alpha >= 0, "initial alpha must be nonnegative")
        require(type(train_alpha) is bool, "train_alpha must be bool")
        require(not train_alpha or alpha > 0, "trainable zero gate blocks bridge gradients at initialization")
        require(isinstance(donor_identity, Identity), "explicit donor identity required")
        require(type(donor_block_index) is int and donor_block_index >= 0, "invalid donor block index")
        self.donor_identity, self.donor_block_index = donor_identity, donor_block_index
        self.donor_norm, self.donor_ffn = donor_norm, donor_ffn
        self.input_bridge, self.return_bridge = input_bridge, return_bridge
        scalar = torch.tensor(float(alpha), dtype=input_bridge.weight.dtype, device=input_bridge.weight.device)
        if train_alpha:
            self.alpha = nn.Parameter(scalar)
        else:
            self.register_buffer("alpha", scalar)
        self.train()

    def frozen_components(self):
        return self.donor_norm, self.donor_ffn

    def forward(self, post_attention_raw: torch.Tensor, *, bypass: bool = False) -> torch.Tensor:
        if bypass:
            return torch.zeros_like(post_attention_raw)
        if not self.alpha.requires_grad and self.alpha.item() == 0:
            return torch.zeros_like(post_attention_raw)  # literal no-op, do not evaluate donor
        u = self.input_bridge(post_attention_raw)
        delta = self.return_bridge(self.donor_ffn(self.donor_norm(u)))
        require(delta.shape == post_attention_raw.shape, "donor delta shape mismatch")
        require(torch.isfinite(delta).all().item() and torch.isfinite(self.alpha).item(), "nonfinite graft")
        return self.alpha * delta


class OrdinaryAdapter(FrozenMode):
    """No-donor control: frozen host norm + affine up / SiLU / bias-free down.

    At hidden width donor_width, trainables exactly equal the two graft bridges
    (excluding an optional learned scalar). Same supervision is a run contract,
    NOT implied by constructing a parameter-matched module. FLOPs are not equal.
    """
    def __init__(self, host_norm: nn.Module, width: int, inner: int, *,
                 dtype: torch.dtype = torch.float64, seed: int = 17, alpha: float = 0.05):
        super().__init__()
        frozen(host_norm)
        positive_int(width, "width")
        positive_int(inner, "inner")
        finite(alpha, "alpha")
        require(alpha > 0, "ordinary adapter needs a nonzero fixed scale")
        self.host_norm = host_norm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_bridge = Bridge(width, inner, dtype=dtype)
            self.return_bridge = Bridge(inner, width, bias=False, dtype=dtype)
            nn.init.zeros_(self.return_bridge.weight)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=dtype))
        self.train()

    def frozen_components(self):
        return (self.host_norm,)

    def forward(self, post_attention_raw, *, bypass=False):
        if bypass:
            return torch.zeros_like(post_attention_raw)
        delta = self.return_bridge(nn.functional.silu(self.input_bridge(self.host_norm(post_attention_raw))))
        require(torch.isfinite(delta).all().item(), "nonfinite ordinary adapter")
        return self.alpha * delta


class GraftedBlock(ResidualBlock, FrozenMode):
    def __init__(self, host: ResidualBlock, graft: FFNGraft | OrdinaryAdapter, *, bypass: bool = False):
        super().__init__()
        frozen(host)
        require(host.width == graft.input_bridge.in_features, "graft/host width mismatch")
        require(type(bypass) is bool, "bypass must be an explicit bool")
        self.host, self.graft, self.bypass = host, graft, bypass
        self.width = host.width
        self.train()

    def frozen_components(self):
        return (self.host,)

    def attention_residual(self, h, mask, positions):
        return self.host.attention_residual(h, mask, positions)

    def ffn_delta(self, post_attention_raw):
        return self.host.ffn_delta(post_attention_raw) + self.graft(post_attention_raw, bypass=self.bypass)

    def forward(self, h, mask, positions):
        # Keep baseline addition order exact; avoid (host_delta + zero) regrouping.
        z = self.host.attention_residual(h, mask, positions)
        baseline = z + self.host.ffn_delta(z)
        if self.bypass or (not self.graft.alpha.requires_grad and self.graft.alpha.item() == 0):
            return baseline
        return baseline + self.graft(z)


class Decoder(FrozenMode):
    """Unpadded batch-one full-prefix runner, intentionally uncached.

    Supply only retained modules. Does not retain an owning full donor model.
    Identity describes provenance, not native qualification or permission.
    """
    def __init__(self, identity: Identity, embedding: nn.Embedding, blocks: Sequence[ResidualBlock],
                 norm: nn.Module, head: nn.Linear, *, max_tokens: int = 512):
        super().__init__()
        positive_int(max_tokens, "max_tokens")
        require(len(blocks) > 0, "decoder needs blocks")
        for component in (embedding, *blocks, norm, head):
            frozen(component)
        require(all(b.width == embedding.embedding_dim for b in blocks), "heterogeneous plain decoder")
        require(head.in_features == embedding.embedding_dim and head.out_features == embedding.num_embeddings,
                "embedding/head geometry mismatch")
        require(identity.logit_policy == "dense_unmodified", "custom Supa-A1.1 logits require separate integration")
        self.identity = identity
        self.embedding, self.blocks, self.norm, self.head = embedding, nn.ModuleList(blocks), norm, head
        self.max_tokens = max_tokens
        self.eval()

    def frozen_components(self):
        return (self.embedding, *self.blocks, self.norm, self.head)

    def layout(self, input_ids: torch.Tensor):
        require(input_ids.ndim == 2 and input_ids.shape[0] == 1 and input_ids.dtype == torch.long,
                "only unpadded batch-one int64 IDs supported")
        length = input_ids.shape[1]
        require(0 < length <= self.max_tokens, "token budget exceeded or empty input")
        require(input_ids.device == self.embedding.weight.device, "input/embedding device mismatch")
        require(int(input_ids.min()) >= 0 and int(input_ids.max()) < self.embedding.num_embeddings, "token ID out of range")
        h = self.embedding(input_ids)
        mask = torch.full((length, length), float("-inf"), dtype=h.dtype, device=h.device).triu(1)[None, None]
        positions = torch.arange(length, device=h.device)[None]
        return h, mask, positions

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool = False,
                past_key_values=None) -> torch.Tensor:
        require(not use_cache and past_key_values is None, "cache unsupported; no implicit fallback")
        h, mask, positions = self.layout(input_ids)
        for block in self.blocks:
            h = block(h, mask, positions)
        return self.head(self.norm(h))

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, block_index: int, site: str) -> torch.Tensor:
        require(type(block_index) is int and 0 <= block_index < len(self.blocks), "bad block index")
        require(site in {"block_input_raw", "post_attention_raw", "block_output_raw"}, "bad capture site")
        h, mask, positions = self.layout(input_ids)
        for index, block in enumerate(self.blocks):
            if index == block_index:
                if site == "block_input_raw":
                    return h.detach().cpu().clone()
                z = block.attention_residual(h, mask, positions)
                result = z if site == "post_attention_raw" else z + block.ffn_delta(z)
                return result.detach().cpu().clone()
            h = block(h, mask, positions)
        raise AssertionError("unreachable")


class StitchedDecoder(Decoder):
    def __init__(self, prefix: Decoder, suffix: Decoder, prefix_count: int,
                 suffix_start: int, bridge: Bridge):
        # Intentionally bypass Decoder's homogeneous-width constructor.
        FrozenMode.__init__(self)
        same_token_grid(prefix.identity, suffix.identity)
        require(type(prefix_count) is int and 0 < prefix_count < len(prefix.blocks), "interior prefix count required")
        require(type(suffix_start) is int and 0 < suffix_start < len(suffix.blocks), "interior suffix start required")
        require(bridge.in_features == prefix.embedding.embedding_dim and
                bridge.out_features == suffix.head.in_features, "stitch bridge geometry mismatch")
        require(prefix.embedding.num_embeddings == suffix.head.out_features, "vocabulary mismatch")
        self.identity = suffix.identity  # output-policy identity; pair recorded separately
        self.prefix_identity, self.suffix_identity = prefix.identity, suffix.identity
        self.embedding = prefix.embedding
        self.prefix_blocks = nn.ModuleList(list(prefix.blocks[:prefix_count]))
        self.suffix_blocks = nn.ModuleList(list(suffix.blocks[suffix_start:]))
        self.bridge, self.norm, self.head = bridge, suffix.norm, suffix.head
        self.max_tokens = min(prefix.max_tokens, suffix.max_tokens)
        self.prefix_count, self.suffix_start = prefix_count, suffix_start
        for component in self.frozen_components():
            frozen(component)
        self.eval()

    def frozen_components(self):
        return (self.embedding, *self.prefix_blocks, *self.suffix_blocks, self.norm, self.head)

    def forward(self, input_ids, *, use_cache=False, past_key_values=None):
        require(not use_cache and past_key_values is None, "cache unsupported; no implicit fallback")
        h, mask, positions = self.layout(input_ids)
        for block in self.prefix_blocks:
            h = block(h, mask, positions)
        h = self.bridge(h)
        for block in self.suffix_blocks:
            h = block(h, mask, positions)
        return self.head(self.norm(h))

    def capture(self, *args, **kwargs):
        raise ValueError("capture the unmodified sources, not the stitched candidate")


def interface_parameters(model: nn.Module, allowed: Sequence[nn.Parameter]) -> list[nn.Parameter]:
    """Refuse extra trainables, duplicate ownership, and missing interface parameters."""
    params = list(allowed)
    ids = [id(p) for p in params]
    require(len(ids) > 0 and len(ids) == len(set(ids)), "empty/duplicate optimizer parameters")
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    require(set(ids) == actual, "trainable set differs from explicit interface allowlist")
    require(all(p.is_leaf and torch.isfinite(p).all().item() for p in params), "invalid interface parameter")
    return params


def masked_next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                           target_mask: torch.Tensor) -> torch.Tensor:
    """Mask refers to target-token positions, including EOS; position zero is never predicted."""
    require(logits.ndim == 3 and input_ids.shape == target_mask.shape == logits.shape[:2], "loss shape mismatch")
    require(input_ids.dtype == torch.long and target_mask.dtype == torch.bool, "loss IDs/mask dtype mismatch")
    require(logits.device == input_ids.device == target_mask.device, "loss device mismatch")
    require(not target_mask[:, 0].any().item(), "position zero cannot be a causal target")
    require(target_mask[:, 1:].any().item(), "no supervised next-token targets")
    require(torch.isfinite(logits).all().item(), "nonfinite logits")
    selected = logits[:, :-1][target_mask[:, 1:]]
    targets = input_ids[:, 1:][target_mask[:, 1:]]
    return nn.functional.cross_entropy(selected, targets)


class GraftedDecoder(Decoder):
    def __init__(self, host: Decoder, block_index: int, graft: FFNGraft | OrdinaryAdapter, *, bypass: bool = False):
        FrozenMode.__init__(self)
        require(type(block_index) is int and 0 <= block_index < len(host.blocks), "invalid graft index")
        for component in host.frozen_components():
            frozen(component)
        if isinstance(graft, FFNGraft):
            same_token_grid(host.identity, graft.donor_identity)
        self.identity, self.max_tokens = host.identity, host.max_tokens
        self.embedding, self.norm, self.head = host.embedding, host.norm, host.head
        blocks = list(host.blocks)
        blocks[block_index] = GraftedBlock(blocks[block_index], graft, bypass=bypass)
        self.blocks = nn.ModuleList(blocks)
        self.graft_index = block_index
        self.eval()

    def frozen_components(self):
        return (self.embedding, self.norm, self.head,
                *(b for i, b in enumerate(self.blocks) if i != self.graft_index))

    def capture(self, *args, **kwargs):
        raise ValueError("capture the unmodified host, not a grafted candidate")


def composition_descriptor(model: StitchedDecoder | GraftedDecoder) -> dict:
    """Explicit candidate identity, distinct from its sources; no native qualification.

    Recompute AFTER each interface update. `.identity` on a composed decoder is
    the output-policy source, not a candidate checkpoint identity.
    """
    if isinstance(model, StitchedDecoder):
        details = {"kind": "prefix_suffix_stitch", "prefix": asdict(model.prefix_identity),
                   "suffix": asdict(model.suffix_identity), "prefix_count": model.prefix_count,
                   "suffix_start": model.suffix_start, "bridge_state_sha256": fingerprint(model.bridge)}
    elif isinstance(model, GraftedDecoder):
        block = model.blocks[model.graft_index]
        sidecar = block.graft
        details = {"kind": "ffn_graft" if isinstance(sidecar, FFNGraft) else "ordinary_adapter_control",
                   "host": asdict(model.identity), "host_block_index": model.graft_index,
                   "site": "post_attention_raw", "bypass": block.bypass,
                   "alpha": float(sidecar.alpha.detach()),
                   "input_bridge_sha256": fingerprint(sidecar.input_bridge),
                   "return_bridge_sha256": fingerprint(sidecar.return_bridge)}
        if isinstance(sidecar, FFNGraft):
            details.update(donor=asdict(sidecar.donor_identity), donor_block_index=sidecar.donor_block_index)
    else:
        raise ValueError("not a composed candidate")
    return {"schema": "xsg-composition-v1", "details": details,
            "composition_sha256": digest(details), "native_qualification": False}
