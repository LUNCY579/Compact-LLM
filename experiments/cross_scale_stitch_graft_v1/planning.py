"""Analytic inventory, not allocation, GPU-fit or latency measurements."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import finite, positive_int, require


@dataclass(frozen=True)
class Shape:
    width: int
    layers: int
    intermediate: int
    q_heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_head: bool

    def __post_init__(self):
        for field in (self.width, self.layers, self.intermediate, self.q_heads,
                      self.kv_heads, self.head_dim, self.vocab):
            positive_int(field, "shape dimension")
        require(self.q_heads % self.kv_heads == 0 and self.head_dim % 2 == 0, "invalid GQA/RoPE geometry")
        require(type(self.tied_head) is bool, "tied_head must be bool")

    @property
    def ffn_parameters(self):
        return 3 * self.width * self.intermediate

    @property
    def block_parameters(self):
        return (2 * self.width * (self.q_heads + self.kv_heads) * self.head_dim +
                self.ffn_parameters + 2 * self.width + 2 * self.head_dim)

    @property
    def model_parameters(self):
        return self.layers * self.block_parameters + self.width + (
            1 if self.tied_head else 2) * self.vocab * self.width


SMALL = Shape(1024, 28, 3072, 16, 8, 128, 151936, True)
Q4 = Shape(2560, 36, 9728, 32, 8, 128, 151936, True)
Q8 = Shape(4096, 36, 12288, 32, 8, 128, 151936, False)


def bridge_parameters(source: int, target: int, *, bias: bool = True) -> int:
    positive_int(source, "source width")
    positive_int(target, "target width")
    require(type(bias) is bool, "bias must be bool")
    return source * target + (target if bias else 0)


def graft_budget(host: Shape, donor: Shape, *, train_alpha: bool = False) -> dict:
    require(type(train_alpha) is bool, "train_alpha must be bool")
    # One donor post-attention norm + intact FFN; no donor embeddings/attention.
    frozen_added = donor.ffn_parameters + donor.width
    interfaces = bridge_parameters(host.width, donor.width) + bridge_parameters(donor.width, host.width, bias=False)
    total = host.model_parameters + frozen_added + interfaces + int(train_alpha)
    return {"total_parameters": total, "added_frozen_parameters": frozen_added,
            "interface_parameters": interfaces + int(train_alpha),
            "bf16_weights_bytes_lower_bound": 2 * total,
            "float32_weights_bytes_lower_bound": 4 * total, "extra_attention_kv_banks": 0,
            "allocation_or_speed_measured": False}


def stitch_budget(prefix: Shape, suffix: Shape, prefix_count: int, suffix_start: int,
                  *, same_source_identity_control: bool = False) -> dict:
    require(type(prefix_count) is int and 0 < prefix_count < prefix.layers, "invalid prefix count")
    require(type(suffix_start) is int and 0 < suffix_start < suffix.layers, "invalid suffix start")
    require(prefix.vocab == suffix.vocab, "vocabulary sizes differ")
    require(type(same_source_identity_control) is bool, "identity-control flag must be bool")
    if same_source_identity_control:
        require(prefix == suffix and prefix_count == suffix_start, "identity control must rejoin at same cut")
    tables = prefix.vocab * prefix.width + suffix.vocab * suffix.width
    if same_source_identity_control and prefix.tied_head:
        tables -= prefix.vocab * prefix.width
    interfaces = bridge_parameters(prefix.width, suffix.width)
    total = (tables + prefix_count * prefix.block_parameters +
             (suffix.layers - suffix_start) * suffix.block_parameters + suffix.width + interfaces)
    return {"total_parameters": total, "interface_parameters": interfaces,
            "retained_blocks": prefix_count + suffix.layers - suffix_start,
            "bf16_weights_bytes_lower_bound": 2 * total,
            "attention_kv_banks_if_cached_in_future": prefix_count + suffix.layers - suffix_start,
            "cache_implemented": False, "allocation_or_speed_measured": False}


def plan() -> dict:
    return {"study": "XSG-001", "status": "OFFLINE_MECHANICS_ONLY",
            "historical_host": "Supa-A1.1 FP16 custom wrapper; native registration still required",
            "4b_graft": graft_budget(SMALL, Q4), "8b_graft": graft_budget(SMALL, Q8),
            "4b_front_small_back": stitch_budget(Q4, SMALL, 18, 14),
            "small_front_4b_back": stitch_budget(SMALL, Q4, 14, 18),
            "stages": ["register_exact_host_and_open_panels", "native_donor_headroom",
                       "native_source_parity_and_capture", "cached_affine_fit",
                       "nonzero_graft_or_stitch_screen", "separately_authorized_bridge_only_training"],
            "native_ready": False, "backbone_recovery_authorized": False,
            "donor_download_authorized": False, "sealed_access_authorized": False}
