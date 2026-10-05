"""Injected-Torch FP32 activation helpers; no framework import or launch.

The resident frozen parameters remain BF16. Temporary linear weight casts stay
on the parameter's device and may be retained by autograd until backward. This
is an unqualified prospective compute profile, not a memory-fit assertion.
"""
from __future__ import annotations

import math
import re

VERSION = "synth-bf16-storage-fp32-activations-1"
QKV_NAME = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _base_snapshot(model, torch):
    named = dict(model.named_parameters(remove_duplicate=False))
    _require(bool(named), "empty base parameter inventory")
    _require(all(p.dtype == torch.bfloat16 and not p.requires_grad and p.grad is None
                 and p.device.type in ("cpu", "cuda") for p in named.values()),
             "base must be resident frozen BF16 with no gradients")
    return {name: (p, p.data_ptr(), tuple(p.shape), str(p.device)) for name, p in named.items()}


def _verify_base(model, torch, snapshot, *, allow_adapters=False):
    named = dict(model.named_parameters(remove_duplicate=False))
    extras = set(named) - set(snapshot)
    _require(not (set(snapshot) - set(named)), "base parameter name disappeared")
    _require(not extras or (allow_adapters and all(n.endswith((".lora_A", ".lora_B")) for n in extras)),
             "unexpected parameter registration")
    for name, (parameter, pointer, shape, device) in snapshot.items():
        observed = named[name]
        _require(observed is parameter and observed.data_ptr() == pointer and tuple(observed.shape) == shape
                 and str(observed.device) == device and observed.dtype == torch.bfloat16
                 and not observed.requires_grad and observed.grad is None,
                 "frozen base object/pointer/shape/dtype changed: " + name)
    return {"base_parameter_aliases": len(snapshot),
            "unique_base_tensors": len({id(item[0]) for item in snapshot.values()}),
            "unique_base_parameters": sum(p.numel() for p in {id(item[0]): item[0] for item in snapshot.values()}.values()),
            "base_parameter_objects_and_pointers_preserved": True,
            "base_storage_dtype": "bfloat16", "activation_dtype": "float32"}


def _replace(model, name, replacement):
    parent_name, separator, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if separator else model
    setattr(parent, child, replacement)


def _linear(torch, inputs, weight, bias=None):
    _require(inputs.is_floating_point() and inputs.device == weight.device,
             "linear requires floating activations on the weight device")
    # Explicitly disable an outer autocast context, which otherwise could cast
    # this float-input linear back to BF16. No original parameter is modified.
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        output = torch.nn.functional.linear(inputs.float(), weight.float(),
                                             None if bias is None else bias.float())
    _require(output.dtype == torch.float32, "linear output must remain FP32")
    return output


def install_fp32_compute(model, torch, expected_linear_sites=197):
    """Wrap every linear and embedding without renaming/replacing parameters.

    Counts and source parameter requirements are checked before mutation. The
    caller separately hashes base tensor content; pointer preservation alone is
    not proof of byte identity. This function constructs no model or optimizer.
    """
    _require(type(expected_linear_sites) is int and expected_linear_sites > 0, "linear site count")
    snapshot = _base_snapshot(model, torch)
    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)]
    embeddings = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Embedding)]
    _require(len(linears) == expected_linear_sites and bool(embeddings), "linear/embedding site inventory differs")
    all_sites = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
                 if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))]
    _require(len(all_sites) == len(linears) + len(embeddings),
             "shared linear/embedding module aliases are unsupported")
    _require(all(not getattr(module, "_synth_fp32_compute", False) for _, module in linears + embeddings),
             "compute helpers already installed")
    _require(all(module.max_norm is None for _, module in embeddings),
             "embedding max_norm may mutate frozen weights")

    class FP32Linear(torch.nn.Linear):
        def __init__(self, original):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = original.in_features, original.out_features
            self.weight, self.bias = original.weight, original.bias
            self._synth_fp32_compute = True
            self.training = original.training

        def forward(self, inputs):
            return _linear(torch, inputs, self.weight, self.bias)

    class FP32Embedding(torch.nn.Embedding):
        def __init__(self, original):
            torch.nn.Module.__init__(self)
            for field in ("num_embeddings", "embedding_dim", "padding_idx", "max_norm",
                          "norm_type", "scale_grad_by_freq", "sparse"):
                setattr(self, field, getattr(original, field))
            self.weight = original.weight
            self._synth_fp32_compute = True
            self.training = original.training

        def forward(self, inputs):
            return torch.nn.functional.embedding(inputs, self.weight, self.padding_idx, self.max_norm,
                                                  self.norm_type, self.scale_grad_by_freq, self.sparse).float()

    for name, module in linears:
        _replace(model, name, FP32Linear(module))
    for name, module in embeddings:
        _replace(model, name, FP32Embedding(module))
    summary = _verify_base(model, torch, snapshot)
    return {"profile": VERSION, "qualification": "UNQUALIFIED_COMPUTE_HELPER",
            "linear_site_count": len(linears), "linear_sites": [name for name, _ in linears],
            "embedding_site_count": len(embeddings), "embedding_sites": [name for name, _ in embeddings],
            **summary}


def adapter_inventory(model, torch):
    """Return actual registered adapter metadata, never weight values."""
    adapters = [(name, p) for name, p in model.named_parameters()
                if name.endswith((".lora_A", ".lora_B"))]
    _require(all(p.requires_grad and p.dtype == torch.float32 and p.device.type in ("cpu", "cuda")
                 for _, p in adapters), "adapter dtype/device/trainability differs")
    _require({name for name, p in model.named_parameters() if p.requires_grad} == {name for name, _ in adapters},
             "non-adapter trainable parameter")
    return [{"name": name, "shape": list(p.shape), "dtype": "float32", "device": str(p.device),
             "requires_grad": True, "parameters": p.numel()} for name, p in adapters]


def install_qv_lora(model, torch, rank=8, alpha=16, expected_layers=28):
    """Add FP32 rank-8 q/v factors to an already wrapped FP32 compute path.

    Original weight/bias parameters keep their original names and objects. B is
    zero initially; callers must physically compare zero-adapter outputs rather
    than infer native parity from that algebraic identity.
    """
    _require(type(rank) is int and rank == 8 and type(alpha) is int and alpha == 16, "rank/alpha contract")
    _require(type(expected_layers) is int and expected_layers > 0, "layer count")
    snapshot = _base_snapshot(model, torch)
    sites = [(name, module) for name, module in model.named_modules()
             if name.endswith((".q_proj", ".v_proj"))]
    expected = {f"model.layers.{layer}.self_attn.{projection}"
                for layer in range(expected_layers) for projection in ("q_proj", "v_proj")}
    _require({name for name, _ in sites} == expected and len(sites) == 2 * expected_layers,
             "q/v sites must cover every registered layer exactly once")
    _require(all(QKV_NAME.fullmatch(name) and isinstance(module, torch.nn.Linear)
                 and getattr(module, "_synth_fp32_compute", False)
                 and not getattr(module, "_synth_qv_lora", False) for name, module in sites),
             "q/v requires installed FP32 linear path")

    class QVLoRA(torch.nn.Linear):
        def __init__(self, original):
            torch.nn.Module.__init__(self)
            self.in_features, self.out_features = original.in_features, original.out_features
            self.weight, self.bias = original.weight, original.bias
            self.lora_A = torch.nn.Parameter(torch.empty(rank, self.in_features,
                                                        dtype=torch.float32, device=self.weight.device))
            self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, rank,
                                                        dtype=torch.float32, device=self.weight.device))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self._synth_fp32_compute, self._synth_qv_lora = True, True
            self.rank, self.alpha, self.dropout, self.scale = rank, alpha, 0.0, alpha / rank
            self.training = original.training

        def forward(self, inputs):
            base = _linear(torch, inputs, self.weight, self.bias)
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                delta = torch.nn.functional.linear(
                    torch.nn.functional.linear(inputs.float(), self.lora_A), self.lora_B)
                output = base + delta * self.scale
            _require(output.dtype == torch.float32, "LoRA output must remain FP32")
            return output

    for name, module in sites:
        _replace(model, name, QVLoRA(module))
    summary = _verify_base(model, torch, snapshot, allow_adapters=True)
    inventory = adapter_inventory(model, torch)
    _require(len(inventory) == 4 * expected_layers, "adapter tensor inventory differs")
    return {"profile": VERSION, "qualification": "UNQUALIFIED_COMPUTE_HELPER", "rank": rank,
            "alpha": alpha, "dropout": 0.0, "qv_site_count": len(sites), "layer_count": expected_layers,
            "trainable_tensors": len(inventory),
            "trainable_parameters": sum(item["parameters"] for item in inventory),
            "inventory": inventory, **summary}


def zero_lora_status(model, torch):
    inventory = adapter_inventory(model, torch)
    factors = [p for name, p in model.named_parameters() if name.endswith(".lora_B")]
    _require(bool(factors) and len(inventory) == 2 * len(factors), "missing/partial LoRA inventory")
    return {"all_B_zero": all(bool((p.detach() == 0).all()) for p in factors),
            "qv_site_count": len(factors), "trainable_tensors": len(inventory),
            "qualification": "ZERO_FACTOR_QUERY_ONLY_NOT_OUTPUT_PARITY"}


def install_fp32_activation_assertions(model, torch):
    """Attach bounded dtype checks to norms, RoPE, attention, MLP and layers.

    Return hook handles and per-module observation counts. No activations or
    weight values are retained. Callers remove handles through handle.remove().
    """
    watched = [(name, module) for name, module in model.named_modules() if
               name.endswith((".input_layernorm", ".post_attention_layernorm", ".q_norm", ".k_norm",
                              ".norm", ".rotary_emb", ".self_attn", ".mlp"))
               or re.fullmatch(r"model\.layers\.\d+", name)]
    _require(bool(watched), "no activation sites to watch")
    observations = {name: 0 for name, _ in watched}

    def floating_tensors(value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                yield value
        elif isinstance(value, (list, tuple)):
            for part in value:
                yield from floating_tensors(part)
        elif isinstance(value, dict):
            for part in value.values():
                yield from floating_tensors(part)

    def make_hook(name):
        def hook(module, inputs, output):
            tensors = list(floating_tensors(inputs)) + list(floating_tensors(output))
            _require(bool(tensors) and all(t.dtype == torch.float32 for t in tensors),
                     "non-FP32 hidden activation at " + name)
            observations[name] += 1
        return hook

    return {"handles": [module.register_forward_hook(make_hook(name)) for name, module in watched],
            "observations": observations, "profile": VERSION}
