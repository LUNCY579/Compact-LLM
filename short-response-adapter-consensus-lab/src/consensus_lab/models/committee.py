from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from consensus_lab.io import read_json, sha256_file
from consensus_lab.paths import LabPaths


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    adapter_path: Path | None
    validation_weight: float = 1.0
    temperature: float = 1.0
    logit_scale: float = 1.0

    @property
    def is_backbone(self) -> bool:
        return self.adapter_path is None


def _safe_adapter_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    if not value:
        raise ValueError(f"Invalid empty adapter name derived from {name!r}")
    return value


def _safetensor_parameter_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return sum(
            int(torch.tensor(handle.get_slice(key).get_shape()).prod().item())
            for key in handle.keys()
        )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision not in mapping:
        raise ValueError(f"Unknown precision: {precision}")
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("fp16 CPU inference is intentionally unsupported")
    return mapping[precision]


def validate_expert_compatibility(
    *,
    paths: LabPaths,
    backbone: Path,
    experts: list[ExpertSpec],
) -> dict[str, Any]:
    backbone = paths.assert_official_source(backbone)
    backbone_config = read_json(backbone / "config.json")
    backbone_tokenizer = backbone / "tokenizer.json"
    if not backbone_tokenizer.is_file():
        raise FileNotFoundError(f"Backbone tokenizer is missing: {backbone_tokenizer}")
    tokenizer_sha = sha256_file(backbone_tokenizer)
    target_modules: set[tuple[str, ...]] = set()
    packets: list[dict[str, Any]] = []
    for expert in experts:
        if expert.temperature <= 0 or expert.logit_scale <= 0 or expert.validation_weight < 0:
            raise ValueError(f"Expert {expert.name} has invalid calibration/weight values")
        if expert.is_backbone:
            packets.append({"name": expert.name, "kind": "backbone", "compatible": True})
            continue
        adapter = paths.assert_official_source(Path(expert.adapter_path))
        config_path = adapter / "adapter_config.json"
        weights = adapter / "adapter_model.safetensors"
        if not config_path.is_file() or not weights.is_file():
            raise FileNotFoundError(f"Incomplete PEFT adapter: {adapter}")
        config = read_json(config_path)
        if config.get("peft_type") != "LORA":
            raise ValueError(f"Unsupported PEFT type for {expert.name}: {config.get('peft_type')}")
        declared = paths.resolve_declared_source(str(config.get("base_model_name_or_path") or ""))
        if declared != backbone.resolve():
            raise ValueError(
                f"Adapter {expert.name} was declared against {declared}, not selected backbone {backbone.resolve()}"
            )
        adapter_tokenizer = adapter / "tokenizer.json"
        adapter_tokenizer_sha = sha256_file(adapter_tokenizer) if adapter_tokenizer.is_file() else tokenizer_sha
        if adapter_tokenizer_sha != tokenizer_sha:
            raise ValueError(f"Tokenizer mismatch for adapter {expert.name}")
        modules = tuple(sorted(config.get("target_modules") or []))
        target_modules.add(modules)
        packets.append(
            {
                "name": expert.name,
                "kind": "adapter",
                "path": str(adapter),
                "peft_type": config.get("peft_type"),
                "rank": config.get("r"),
                "adapter_parameter_count": _safetensor_parameter_count(weights),
                "adapter_storage_bytes": sum(path.stat().st_size for path in adapter.iterdir() if path.is_file()),
                "use_dora": bool(config.get("use_dora")),
                "target_modules": list(modules),
                "declared_base_resolved": str(declared),
                "tokenizer_sha256": adapter_tokenizer_sha,
                "compatible": True,
            }
        )
    if len(target_modules) > 1:
        raise ValueError("Adapters target different module sets; explicit mixed-target validation is required")
    return {
        "backbone": str(backbone),
        "architecture": (backbone_config.get("architectures") or [None])[0],
        "model_type": backbone_config.get("model_type"),
        "vocabulary_size": backbone_config.get("vocab_size"),
        "hidden_size": backbone_config.get("hidden_size"),
        "layer_count": backbone_config.get("num_hidden_layers"),
        "tokenizer_sha256": tokenizer_sha,
        "experts": packets,
    }


class ExpertCommittee:
    """One frozen backbone with independently activated PEFT experts."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        experts: list[ExpertSpec],
        device: torch.device,
        dtype: torch.dtype,
        compatibility: dict[str, Any],
    ) -> None:
        if not experts:
            raise ValueError("At least one expert is required")
        if len({expert.name for expert in experts}) != len(experts):
            raise ValueError("Expert names must be unique")
        self.model = model
        self.tokenizer = tokenizer
        self.experts = experts
        self.device = device
        self.dtype = dtype
        self.compatibility = compatibility
        self._adapter_names = {
            expert.name: _safe_adapter_name(expert.name) for expert in experts if not expert.is_backbone
        }

    @classmethod
    def load(
        cls,
        *,
        paths: LabPaths,
        backbone: Path,
        experts: list[ExpertSpec],
        device: str = "auto",
        precision: str = "auto",
    ) -> "ExpertCommittee":
        compatibility = validate_expert_compatibility(paths=paths, backbone=backbone, experts=experts)
        resolved_device = resolve_device(device)
        dtype = resolve_dtype(precision, resolved_device)
        tokenizer = AutoTokenizer.from_pretrained(str(backbone), trust_remote_code=True, local_files_only=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            str(backbone),
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base.to(resolved_device)
        adapter_experts = [expert for expert in experts if not expert.is_backbone]
        model: Any = base
        adapter_names: dict[str, str] = {}
        if adapter_experts:
            first = adapter_experts[0]
            first_name = _safe_adapter_name(first.name)
            adapter_names[first.name] = first_name
            model = PeftModel.from_pretrained(
                base,
                str(first.adapter_path),
                adapter_name=first_name,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
            for expert in adapter_experts[1:]:
                adapter_name = _safe_adapter_name(expert.name)
                adapter_names[expert.name] = adapter_name
                model.load_adapter(
                    str(expert.adapter_path),
                    adapter_name=adapter_name,
                    is_trainable=False,
                    low_cpu_mem_usage=True,
                )
        model.eval()
        model.config.use_cache = True
        committee = cls(
            model=model,
            tokenizer=tokenizer,
            experts=experts,
            device=resolved_device,
            dtype=dtype,
            compatibility=compatibility,
        )
        committee._adapter_names = adapter_names
        return committee

    @contextlib.contextmanager
    def activate(self, expert: ExpertSpec) -> Iterator[None]:
        if expert.is_backbone:
            if not isinstance(self.model, PeftModel):
                yield
            else:
                with self.model.disable_adapter():
                    yield
            return
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Adapter expert requested from a non-PEFT model")
        self.model.set_adapter(self._adapter_names[expert.name])
        yield

    def encode_messages(self, messages: list[dict[str, Any]], *, max_sequence_length: int) -> torch.Tensor:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[1]) > max_sequence_length:
            raise RuntimeError(
                f"Prompt requires {input_ids.shape[1]} tokens > max_sequence_length={max_sequence_length}"
            )
        return input_ids.to(self.device)

    def forward(
        self,
        expert: ExpertSpec,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, Any | None]:
        with self.activate(expert), torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                logits_to_keep=1,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].detach().to(dtype=torch.float32)
        if expert.logit_scale != 1.0:
            logits = logits * expert.logit_scale
        return logits.squeeze(0), output.past_key_values if use_cache else None

    def teacher_forced_logits(
        self,
        expert: ExpertSpec,
        input_ids: torch.Tensor,
        *,
        target_length: int,
    ) -> torch.Tensor:
        """Return logits for the final target positions of an unpadded sequence."""
        if target_length <= 0 or target_length > int(input_ids.shape[1]):
            raise ValueError("target_length must be within the supplied sequence")
        attention_mask = torch.ones_like(input_ids)
        with self.activate(expert), torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=target_length,
                return_dict=True,
            )
        return output.logits[0, -target_length:, :].detach().float()

    def metadata(self) -> dict[str, Any]:
        selected = {expert.name for expert in self.experts}
        compatibility_by_name = {
            packet["name"]: packet for packet in self.compatibility["experts"] if packet["name"] in selected
        }
        adapter_parameter_count = sum(
            int(packet.get("adapter_parameter_count") or 0) for packet in compatibility_by_name.values()
        )
        adapter_storage_bytes = sum(
            int(packet.get("adapter_storage_bytes") or 0) for packet in compatibility_by_name.values()
        )
        loaded_parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        loaded_adapter_parameter_count = sum(
            parameter.numel()
            for name, parameter in self.model.named_parameters()
            if "lora_" in name or "modules_to_save" in name
        )
        backbone_parameter_count = loaded_parameter_count - loaded_adapter_parameter_count
        largest_active_adapter = max(
            (int(packet.get("adapter_parameter_count") or 0) for packet in compatibility_by_name.values()),
            default=0,
        )
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "expert_names": [expert.name for expert in self.experts],
            "backbone_parameter_count": backbone_parameter_count,
            "loaded_parameter_count": loaded_parameter_count,
            "loaded_adapter_parameter_count": loaded_adapter_parameter_count,
            "committee_adapter_parameter_count": adapter_parameter_count,
            "active_parameter_count_per_expert_forward": backbone_parameter_count + largest_active_adapter,
            "adapter_storage_bytes": adapter_storage_bytes,
            "compatibility": self.compatibility,
        }

    def subset(self, names: list[str]) -> "ExpertCommittee":
        by_name = {expert.name: expert for expert in self.experts}
        missing = sorted(set(names) - set(by_name))
        if missing:
            raise KeyError(f"Unknown committee experts: {missing}")
        selected = [by_name[name] for name in names]
        view = ExpertCommittee(
            model=self.model,
            tokenizer=self.tokenizer,
            experts=selected,
            device=self.device,
            dtype=self.dtype,
            compatibility=self.compatibility,
        )
        view._adapter_names = {name: self._adapter_names[name] for name in names if name in self._adapter_names}
        return view
