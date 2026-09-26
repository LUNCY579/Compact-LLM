"""Phase 1 adapters: attention-only LoRA on late layers, and NoRA-init.

Trainable set:

    theta_train = {q_proj, k_proj, v_proj, o_proj}_{L17, L18, L19}

Layers 0, 1 and 2 -- the catastrophic stem in the identity-skip screen -- are
never touched. Neither are MLP projections, embeddings, the LM head or norms.

NoRA-init differs from ordinary LoRA at initialization ONLY:

1. initialise the LoRA down-projection ``lora_A`` in the matched random manner
   (identical distribution and identical seed to the LoRA arm);
2. rescale each of its columns to unit L2 norm;
3. from there, train exactly as ordinary LoRA.

``lora_B`` stays at zero in both arms, so both start as an exact functional copy
of frozen Supa. ``verify_adapter`` checks that and the rest of the invariants.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

import torch
from peft import LoraConfig, get_peft_model

PROTECTED_LAYERS = (0, 1, 2)
DEFAULT_LAYERS = (17, 18, 19)
EXPANDED_LAYERS = (17, 18, 19, 25, 27)
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

DEFAULT_RANK = 4
DEFAULT_ALPHA = 8
DEFAULT_DROPOUT = 0.0


def build_lora_config(
    layers: Sequence[int] = DEFAULT_LAYERS,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
) -> LoraConfig:
    forbidden = sorted(set(layers) & set(PROTECTED_LAYERS))
    if forbidden:
        raise ValueError(
            f"layers {forbidden} are the protected stem and must never be trained"
        )
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(ATTENTION_TARGETS),
        layers_to_transform=list(layers),
        layers_pattern="layers",
        init_lora_weights=True,
    )


def _lora_a_modules(model: Any) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("lora_A.default")
    ]


def _lora_b_modules(model: Any) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("lora_B.default")
    ]


@torch.no_grad()
def apply_nora_init(model: Any, eps: float = 1e-12) -> dict:
    """Rescale every ``lora_A`` column to unit L2 norm, in place.

    ``lora_A`` has shape (r, in_features); its columns are the ``in_features``
    vectors in R^r, one per input direction. Normalising them makes every input
    direction enter the low-rank code with equal gain, instead of with the
    spread that Kaiming-uniform sampling happens to produce.
    """
    report = {"modules": 0, "columns": 0, "mean_norm_before": 0.0}
    norms_before: list[float] = []
    for _, module in _lora_a_modules(model):
        weight = module.weight  # (r, in_features)
        col_norms = weight.norm(dim=0, keepdim=True)  # (1, in_features)
        norms_before.extend(col_norms.flatten().tolist())
        weight.div_(col_norms.clamp_min(eps))
        report["modules"] += 1
        report["columns"] += weight.shape[1]
    report["mean_norm_before"] = (
        round(sum(norms_before) / len(norms_before), 6) if norms_before else 0.0
    )
    return report


def build_adapter(
    base_model: Any,
    arm: str,
    layers: Sequence[int] = DEFAULT_LAYERS,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = 17,
) -> tuple[Any, dict]:
    """Wrap ``base_model`` with the requested arm's adapter.

    ``arm`` is ``"lora"`` or ``"nora"``. Both draw ``lora_A`` from the same
    distribution under the same seed, so the two arms start from the same random
    numbers and differ only by the normalisation step.
    """
    if arm not in ("lora", "nora"):
        raise ValueError(f"unknown arm {arm!r}; expected 'lora' or 'nora'")

    config = build_lora_config(layers, rank, alpha, dropout)
    torch.manual_seed(seed)
    model = get_peft_model(base_model, config)

    info: dict[str, Any] = {
        "arm": arm,
        "layers": list(layers),
        "targets": list(ATTENTION_TARGETS),
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "init_seed": seed,
    }
    if arm == "nora":
        info["nora_init"] = apply_nora_init(model)
    info.update(count_trainable(model))
    return model, info


def count_trainable(model: Any) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": round(trainable / total, 8) if total else 0.0,
        "trainable_tensor_count": len(names),
        "trainable_param_names": names,
    }


def trained_layer_indices(model: Any) -> set[int]:
    """Decoder layer indices that own at least one trainable parameter."""
    found: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                found.add(int(parts[i + 1]))
    return found


# --------------------------------------------------------------------------
# Invariant checks
# --------------------------------------------------------------------------


@torch.no_grad()
def verify_adapter(
    model: Any,
    arm: str,
    expected_layers: Iterable[int],
    reference_logits: torch.Tensor | None = None,
    probe_inputs: dict | None = None,
    tol: float = 1e-3,
) -> dict:
    """Mechanical checks the brief requires before either arm is trained."""
    checks: dict[str, Any] = {}

    a_norms: list[float] = []
    for _, module in _lora_a_modules(model):
        a_norms.extend(module.weight.norm(dim=0).flatten().tolist())
    max_dev = max((abs(n - 1.0) for n in a_norms), default=float("nan"))
    checks["lora_A_column_norm_max_deviation_from_1"] = round(max_dev, 8)
    checks["lora_A_column_norm_mean"] = round(sum(a_norms) / len(a_norms), 6) if a_norms else 0.0

    if arm == "nora":
        checks["nora_columns_normalised"] = bool(max_dev < 1e-5)
    else:
        # Ordinary LoRA must be left alone: Kaiming-uniform columns are not unit
        # norm, so seeing unit norms here would mean the NoRA step leaked.
        checks["lora_columns_left_unnormalised"] = bool(max_dev > 1e-3)

    b_max = max(
        (module.weight.abs().max().item() for _, module in _lora_b_modules(model)),
        default=0.0,
    )
    checks["lora_B_all_zero_at_init"] = bool(b_max == 0.0)

    trained = trained_layer_indices(model)
    checks["trained_layers"] = sorted(trained)
    checks["expected_layers"] = sorted(set(expected_layers))
    checks["trained_layers_as_expected"] = trained == set(expected_layers)
    checks["protected_stem_untouched"] = not (trained & set(PROTECTED_LAYERS))

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    checks["only_attention_projections_trainable"] = all(
        any(f".{t}." in n for t in ATTENTION_TARGETS) for n in trainable_names
    )
    checks["no_mlp_embed_or_head_trainable"] = not any(
        k in n
        for n in trainable_names
        for k in ("mlp", "embed_tokens", "lm_head", "norm.weight")
    )

    if reference_logits is not None and probe_inputs is not None:
        out = model(**probe_inputs).logits
        delta = (out.float() - reference_logits.float()).abs().max().item()
        checks["zero_adapter_reproduces_frozen_max_logit_delta"] = round(delta, 8)
        checks["zero_adapter_reproduces_frozen"] = bool(delta <= tol)

    checks["all_passed"] = all(
        v for k, v in checks.items() if isinstance(v, bool)
    )
    return checks


@torch.no_grad()
def verify_serialisation_and_merge(
    model: Any,
    tmpdir: str,
    probe_inputs: dict,
    tol: float = 1e-2,
) -> dict:
    """Save/reload the adapter and, if merging is supported, check merge parity."""
    from peft import PeftModel

    checks: dict[str, Any] = {}
    before = model(**probe_inputs).logits.float().clone()

    model.save_pretrained(tmpdir)
    checks["adapter_serialised"] = True

    base = model.get_base_model()
    reloaded = PeftModel.from_pretrained(copy.deepcopy(base), tmpdir)
    after = reloaded(**probe_inputs).logits.float()
    checks["reload_max_logit_delta"] = round((before - after).abs().max().item(), 8)
    checks["serialisation_round_trip_ok"] = bool(
        checks["reload_max_logit_delta"] <= tol
    )

    try:
        merged = reloaded.merge_and_unload()
        merged_logits = merged(**probe_inputs).logits.float()
        checks["merge_max_logit_delta"] = round(
            (before - merged_logits).abs().max().item(), 8
        )
        checks["merge_preserves_function"] = bool(checks["merge_max_logit_delta"] <= tol)
        del merged
    except Exception as exc:  # pragma: no cover - merge support is optional
        checks["merge_supported"] = False
        checks["merge_error"] = f"{type(exc).__name__}: {exc}"

    del reloaded
    checks["all_passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks
