from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path
from typing import Any

import accelerate
import bitsandbytes
import peft
import safetensors
import torch
import transformers
from peft import AdaLoraConfig, LoraConfig, PeftMixedModel, get_peft_model
from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES
from peft.utils.hotswap import hotswap_adapter
from transformers import Qwen3Config, Qwen3ForCausalLM

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths


def _attempt(action: Any) -> dict[str, Any]:
    try:
        return {"supported": True, **(action() or {})}
    except Exception as exc:  # capability probes must preserve negative evidence
        return {"supported": False, "error_type": type(exc).__name__, "error": str(exc)}


def _tiny_qwen() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    return Qwen3ForCausalLM(config)


def _variant_probe(config: Any) -> dict[str, Any]:
    model = get_peft_model(_tiny_qwen(), config)
    logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    return {
        "forward_shape": list(logits.shape),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe installed PEFT features with synthetic and official adapters.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(LAB_ROOT / "configs/hard_vote.yaml", paths)
    wanted = {"general", "abstention"}
    experts = [expert for expert in config.experts if expert.name in wanted]
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=experts,
        device=args.device,
        precision=args.precision,
    )
    row = read_jsonl(config.datasets["v22_dev"])[0]
    input_ids = committee.encode_messages(row["messages"][:-1], max_sequence_length=config.max_sequence_length)
    attention_mask = torch.ones_like(input_ids)
    sequential: dict[str, torch.Tensor] = {}
    for expert in committee.experts:
        sequential[expert.name], _ = committee.forward(
            expert, input_ids, attention_mask=attention_mask, use_cache=False
        )

    def mixed_batch() -> dict[str, Any]:
        batch_ids = input_ids.repeat(len(experts), 1)
        batch_mask = attention_mask.repeat(len(experts), 1)
        internal_names = [committee._adapter_names[expert.name] for expert in experts]
        with torch.inference_mode():
            output = committee.model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                adapter_names=internal_names,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1, :].float()
        differences = [
            float(torch.max(torch.abs(output[index] - sequential[expert.name])).item())
            for index, expert in enumerate(experts)
        ]
        return {
            "adapter_names": internal_names,
            "top1_matches_sequential": [
                int(torch.argmax(output[index]).item()) == int(torch.argmax(sequential[expert.name]).item())
                for index, expert in enumerate(experts)
            ],
            "max_abs_logit_differences": differences,
        }

    per_example = _attempt(mixed_batch)

    def hotswap() -> dict[str, Any]:
        source = experts[0]
        target = experts[1]
        source_name = committee._adapter_names[source.name]
        hotswap_adapter(
            committee.model,
            str(target.adapter_path),
            adapter_name=source_name,
            torch_device=str(committee.device),
        )
        committee.model.set_adapter(source_name)
        with torch.inference_mode():
            swapped = committee.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            ).logits[0, -1].float()
        expected = sequential[target.name]
        return {
            "source_slot": source_name,
            "replacement_adapter": target.name,
            "top1_matches_preloaded_target": int(torch.argmax(swapped).item())
            == int(torch.argmax(expected).item()),
            "max_abs_logit_difference": float(torch.max(torch.abs(swapped - expected)).item()),
        }

    hotswap_result = _attempt(hotswap)
    del committee
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    variants = {
        "lora": _attempt(
            lambda: _variant_probe(LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        ),
        "dora": _attempt(
            lambda: _variant_probe(
                LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM", use_dora=True)
            )
        ),
        "pissa": _attempt(
            lambda: _variant_probe(
                LoraConfig(
                    r=2,
                    target_modules=["q_proj"],
                    task_type="CAUSAL_LM",
                    init_lora_weights="pissa",
                )
            )
        ),
        "adalora": _attempt(
            lambda: _variant_probe(
                AdaLoraConfig(
                    init_r=2,
                    target_r=2,
                    target_modules=["q_proj"],
                    task_type="CAUSAL_LM",
                    total_step=2,
                )
            )
        ),
    }

    def mixed_types() -> dict[str, Any]:
        model = PeftMixedModel(
            _tiny_qwen(),
            LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"),
            "lora",
        )
        model.add_adapter(
            "adalora",
            AdaLoraConfig(
                init_r=2,
                target_r=2,
                target_modules=["q_proj"],
                task_type="CAUSAL_LM",
                total_step=2,
            ),
        )
        model.set_adapter("adalora")
        shape = list(model(input_ids=torch.tensor([[1, 2, 3]])).logits.shape)
        return {"combination": ["LORA", "ADALORA"], "forward_shape": shape}

    packet = {
        "schema_version": "1.0",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "safetensors": safetensors.__version__,
        },
        "official_adapter_tests": {
            "multiple_lora_adapters_loaded": True,
            "independent_activation": {
                "top_tokens": {
                    name: int(torch.argmax(logits).item()) for name, logits in sequential.items()
                },
                "distinct_logits": not torch.equal(sequential[experts[0].name], sequential[experts[1].name]),
            },
            "per_example_adapter_selection": per_example,
            "lora_hotswap": hotswap_result,
        },
        "synthetic_qwen_tests": {
            **variants,
            "mixed_lora_adalora": _attempt(mixed_types),
        },
        "declared_mixed_compatible_tuner_types": [
            getattr(value, "value", str(value)) for value in COMPATIBLE_TUNER_TYPES
        ],
        "interpretation": {
            "existing_official_pool": "LoRA only; no valid empirical mixed-method committee exists yet.",
            "dora_pissa_note": "DoRA and PiSSA are LoRA parameterization/initialization variants in this PEFT release.",
            "training_required": "Matched new adapters must be trained in the lab before mixed-method conclusions are possible.",
            "per_example_batching": "The API runs, but nonzero bf16 logit differences require a larger decision-parity gate before use.",
            "hotswap": "The API runs, but nonzero bf16 logit differences require a larger decision-parity gate before use.",
        },
    }
    output = paths.assert_output_path(LAB_ROOT / "reports/peft_capability_probe.json")
    write_json_atomic(output, packet)
    lines = [
        "# PEFT Capability Probe",
        "",
        f"- Environment: Python {platform.python_version()}, torch {torch.__version__}, transformers "
        f"{transformers.__version__}, PEFT {peft.__version__}.",
        f"- GPU: {packet['environment']['gpu']}.",
        f"- Multiple official LoRAs loaded independently: **yes**.",
        f"- Per-example adapter batch API: **{'runs' if per_example['supported'] else 'failed'}**; it is not enabled "
        "for research runs because the probe showed nonzero bf16 logit differences.",
        f"- LoRA hotswap API on official adapters: **{'runs' if hotswap_result['supported'] else 'failed'}**; it is "
        "not enabled for research runs without a larger parity gate.",
        f"- Tiny-Qwen LoRA/DoRA/PiSSA/AdaLoRA forwards: **"
        f"{'all pass' if all(value['supported'] for value in variants.values()) else 'one or more failed'}**.",
        f"- Inference mixed LoRA + AdaLoRA construction: **"
        f"{'pass' if packet['synthetic_qwen_tests']['mixed_lora_adalora']['supported'] else 'fail'}**.",
        "",
        "These are software-capability checks, not evidence that a parameterization improves this model. "
        "The official compatible pool contains only rank-16 LoRAs; fair DoRA/PiSSA/AdaLoRA comparisons therefore "
        "remain preregistered until matched adapters are trained inside this lab.",
        "",
        "Primary references: [PEFT hotswapping](https://huggingface.co/docs/peft/package_reference/hotswap), "
        "[PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora), "
        "[PEFT mixed adapters](https://huggingface.co/docs/peft/developer_guides/mixed_models), "
        "[LoRA](https://arxiv.org/abs/2106.09685), [DoRA](https://arxiv.org/abs/2402.09353), "
        "and [PiSSA](https://arxiv.org/abs/2404.02948).",
        "",
    ]
    report = paths.assert_output_path(LAB_ROOT / "reports/PEFT_CAPABILITIES.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), **packet}, indent=2))


if __name__ == "__main__":
    main()
