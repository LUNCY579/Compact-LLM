"""Identity-initialized augmentation. Native attention and KV cache are retained."""
from dataclasses import replace
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .core import MemoryConfig, MemoryState, normalize, delta_step, read_state


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class MemorySidecar(nn.Module):
    """Reference triadic GDN with a width-4 streaming causal convolution.

    Default FP32 parameters/state; FP64 is supported for numerical tests. Do not
    cast this module to FP16/BF16: the backbone can independently use BF16.
    gate=0 with NONZERO output weights preserves identity without a dead gate.
    """
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = c = config
        self.norm = RMSNorm(c.d_model)
        self.project = nn.Linear(c.d_model, c.channels, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(c.channels, c.conv_width))
        nn.init.uniform_(self.conv_weight, -c.conv_width ** -0.5, c.conv_width ** -0.5)
        self.gates = nn.Linear(c.d_model, c.heads * (c.slices + 1), bias=True)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[:c.heads * c.slices].fill_(-4.0)
        self.out_norm = RMSNorm(c.value_dim)
        self.output = nn.Linear(c.heads * c.value_dim, c.d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(c.gate_init))

    def initial_state(self, batch: int, device=None) -> MemoryState:
        c, dtype = self.config, self.project.weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("reference state requires FP32/FP64; keep sidecar in FP32")
        if c.state_bytes(batch, torch.tensor([], dtype=dtype).element_size()) > c.max_state_bytes:
            raise MemoryError("recurrent-state allocation exceeds configured limit")
        device = device or self.project.weight.device
        return MemoryState(torch.zeros(batch, c.heads, c.slices, c.key_dim, c.value_dim, device=device, dtype=dtype),
                           torch.zeros(batch, c.channels, c.conv_width - 1, device=device, dtype=dtype))

    def _validate_state(self, state, b, device, dtype):
        c = self.config
        for t, shape in ((state.tensor, (b, c.heads, c.slices, c.key_dim, c.value_dim)),
                         (state.history, (b, c.channels, c.conv_width - 1))):
            if t.shape != shape or t.dtype != dtype or t.device != device:
                raise ValueError("memory/history cache shape, dtype, or device mismatch")
        if state.nbytes > c.max_state_bytes:
            raise MemoryError("supplied memory exceeds allocation limit")

    def forward(self, hidden: Tensor, state: MemoryState | None = None,
                valid: Tensor | None = None, reset: Tensor | None = None,
                write: Tensor | None = None, ablation: str = "none"):
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.d_model or hidden.shape[1] == 0:
            raise ValueError("hidden must be nonempty [B,T,d_model]")
        if ablation not in ("none", "constant_second_keys", "erase_reads", "read_only"):
            raise ValueError("unsupported causal ablation")
        b, t, _ = hidden.shape
        c = self.config
        masks = []
        for name, mask, default in (("valid", valid, True), ("reset", reset, False), ("write", write, True)):
            mask = torch.full((b, t), default, dtype=torch.bool, device=hidden.device) if mask is None else mask
            if mask.shape != (b, t) or mask.dtype != torch.bool or mask.device != hidden.device:
                raise ValueError(f"{name} must be boolean [B,T] on the hidden device")
            masks.append(mask)
        valid, reset, write = masks
        if torch.any(reset & ~valid):
            raise ValueError("document reset cannot be attached to padding")
        dtype = self.project.weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("keep reference sidecar parameters in FP32/FP64")
        state = self.initial_state(b, hidden.device) if state is None else state
        self._validate_state(state, b, hidden.device, dtype)
        x = self.norm(hidden.to(dtype))
        projected, gates = self.project(x), self.gates(x)
        recurrent, history = state.tensor, state.history
        outputs = []
        for i in range(t):
            keep = (~reset[:, i]).to(dtype)
            recurrent = recurrent * keep[:, None, None, None, None]
            history = history * keep[:, None, None]
            window = torch.cat((history, projected[:, i, :, None]), -1)
            z = (window * self.conv_weight).sum(-1)
            history_next = window[..., 1:]
            history = torch.where(valid[:, i, None, None], history_next, history)
            sizes = [c.heads*c.key_dim, c.heads*c.key_dim, c.heads*c.value_dim]
            if c.slices > 1:
                sizes += [c.heads*c.slices, c.heads*c.slices]
            q, k, v, *second = z.split(sizes, dim=-1)
            q = normalize(F.silu(q).reshape(b, c.heads, c.key_dim))
            k = normalize(F.silu(k).reshape(b, c.heads, c.key_dim))
            v = F.silu(v).reshape(b, c.heads, c.value_dim)
            if second:
                k2, q2 = (normalize(F.softplus(s.reshape(b, c.heads, c.slices))) for s in second)
                if ablation == "constant_second_keys":
                    k2 = torch.ones_like(k2) / c.slices**0.5
                    q2 = torch.ones_like(q2) / c.slices**0.5
            else:
                k2 = q2 = torch.ones(b, c.heads, 1, device=x.device, dtype=dtype)
            decay, beta = gates[:, i].split([c.heads*c.slices, c.heads], -1)
            alpha = torch.exp(-F.softplus(decay.reshape(b, c.heads, c.slices)))
            beta = torch.sigmoid(beta.reshape(b, c.heads))
            updated = delta_step(recurrent, k, k2, v, alpha, beta)
            do_write = valid[:, i] & write[:, i] & (ablation != "read_only")
            recurrent = torch.where(do_write[:, None, None, None, None], updated, recurrent)
            out = read_state(recurrent, q, q2) * c.key_dim**-0.5
            if ablation == "erase_reads":
                out = out * 0
            out = self.output(self.out_norm(out).reshape(b, -1))
            outputs.append(out * valid[:, i, None])
        correction = torch.stack(outputs, 1)
        # Read-only disables state writes AND decay, but query-convolution history advances.
        result = hidden + (torch.tanh(self.gate) * correction).to(hidden.dtype)
        return result, MemoryState(recurrent, history)


class StatelessSidecar(nn.Module):
    """Active low-rank MLP control, matched by rank rather than dummy parameters."""
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        if min(d_model, rank) <= 0:
            raise ValueError("positive model width and rank required")
        self.norm = RMSNorm(d_model)
        self.up = nn.Linear(d_model, rank, bias=False)
        self.down = nn.Linear(rank, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, state=None, valid=None, reset=None, write=None, ablation="none"):
        if state is not None or ablation != "none":
            raise ValueError("stateless control has no memory or memory ablations")
        out = self.down(F.silu(self.up(self.norm(hidden.to(self.up.weight.dtype)))))
        if valid is not None:
            out = out * valid[..., None]
        return hidden + (self.gate.tanh() * out).to(hidden.dtype), None


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_arm(arm: str, config: MemoryConfig):
    if arm == "baseline":
        return None
    if arm == "triadic_e4":
        if config.slices != 4:
            raise ValueError("triadic_e4 requires slices=4")
        return MemorySidecar(config)
    if arm == "gdn_e1":
        return MemorySidecar(replace(config, slices=1))
    if arm == "matrix_state_matched":
        return MemorySidecar(replace(config, key_dim=config.key_dim*config.slices, slices=1))
    if arm == "stateless":
        # Derive count without creating a temporary model or consuming model RNG.
        c = config
        target = (c.d_model + c.d_model*c.channels + c.channels*c.conv_width
                  + (c.d_model+1)*c.heads*(c.slices+1) + c.value_dim
                  + c.d_model*c.heads*c.value_dim + 1)
        rank = max(1, round((target-c.d_model-1)/(2*c.d_model)))
        return StatelessSidecar(c.d_model, rank)
    raise ValueError(f"unknown arm: {arm}")
