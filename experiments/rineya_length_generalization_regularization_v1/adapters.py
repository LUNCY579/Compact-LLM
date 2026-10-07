"""Small explicit Q/V LoRA reference; no adapter dropout or PEFT dependency."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if type(base) is not nn.Linear or rank <= 0 or alpha <= 0:
            raise ValueError("Only ordinary Linear projections are qualified")
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        residual = F.linear(F.linear(x.float(), self.A), self.B)
        return self.base(x) + (residual * self.scale).to(x.dtype)

    @torch.no_grad()
    def merged(self):
        """New ordinary Linear, with one explicit cast to the serving dtype."""
        result = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None, device=self.base.weight.device,
                           dtype=self.base.weight.dtype)
        result.weight.copy_((self.base.weight.float() + self.scale * (self.B @ self.A)).to(result.weight.dtype))
        if result.bias is not None:
            result.bias.copy_(self.base.bias)
        result.requires_grad_(False)
        result.train(self.training)
        return result


def inject_lora(model, plan):
    from .regularization import norm_targets
    norm_targets(model, plan.expected_layers)
    names = [f"model.layers.{i}.self_attn.{kind}_proj" for i in range(plan.expected_layers)
             for kind in ("q", "v")]
    modules = dict(model.named_modules())
    if any(type(modules.get(n)) is not nn.Linear for n in names):
        raise ValueError("Unexpected Q/V projection layout or preexisting adapters")
    if len({id(modules[n]) for n in names}) != len(names):
        raise ValueError("Aliased projections are not supported")
    model.requires_grad_(False)
    for name in names:
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, LoRALinear(modules[name], plan.rank, plan.lora_alpha))
    return names


def adapter_state(model):
    return {f"{name}.{key}": getattr(mod, key).detach().cpu().clone()
            for name, mod in model.named_modules() if isinstance(mod, LoRALinear) for key in ("A", "B")}


def load_adapters(model, state):
    expected = adapter_state(model)
    if expected.keys() != state.keys() or not expected:
        raise ValueError("Adapter key mismatch")
    if any(not isinstance(state[k], torch.Tensor) or state[k].shape != v.shape or
           not torch.isfinite(state[k]).all() for k, v in expected.items()):
        raise ValueError("Invalid adapter shape/value")
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                for key in ("A", "B"):
                    getattr(mod, key).copy_(state[f"{name}.{key}"])


def merge_adapters(model):
    """Explicit destructive model transformation; never done by train/eval mode."""
    names = [name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    for name in names:
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, model.get_submodule(name).merged())
    return names
