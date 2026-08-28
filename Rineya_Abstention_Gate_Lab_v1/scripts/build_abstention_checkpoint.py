#!/usr/bin/env python3
"""Bake nimbus_low + the abstention gate into one dense checkpoint directory.

Two things are folded in:

1. the halo Stage-A LoRA, merged at its evaluation scale, so no adapter is
   needed at load time;
2. the abstention decision, as a constant logit offset on the ``UNKNOWN`` token
   applied by a small Qwen3 subclass registered through ``auto_map``.

Step 2 is exact only for the gate family whose decision boundary is
``z_UNKNOWN - z_top1 >= threshold``: the softmax normaliser cancels, so the rule
is *literally* "add ``b = -threshold`` to one logit and take the argmax". Gates
that compare a normalised probability against a threshold (S1, S3) are not
expressible this way and still need a logits processor at inference time.

Caveat, measured rather than assumed: the subclass applies the offset at every
decode step, while the gate is defined on the first generated token only.
Whether that matters is an empirical question -- run the checkpoint through
the published runner and compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core

SCRIPT_PATH = Path(__file__).resolve()
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)

MODELING_ABSTAIN = '''\
"""Qwen3 with a constant abstention-token logit offset."""

from __future__ import annotations

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class Qwen3AbstainForCausalLM(Qwen3ForCausalLM):
    """Qwen3 plus the configured offset on one abstention token."""

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        token_id = getattr(self.config, "abstain_token_id", None)
        bias = float(getattr(self.config, "abstain_logit_bias", 0.0) or 0.0)
        logits = getattr(outputs, "logits", None)
        if token_id is not None and bias and logits is not None:
            logits[..., int(token_id)] = logits[..., int(token_id)] + bias
        return outputs
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def configure_abstention(config: Any, token_id: int, logit_bias: float) -> None:
    """Register the portable custom-model route on a Transformers config."""

    config.architectures = ["Qwen3AbstainForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_abstain.Qwen3AbstainForCausalLM"
    }
    config.abstain_token_id = int(token_id)
    config.abstain_logit_bias = float(logit_bias)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-scale", type=float, required=True)
    parser.add_argument(
        "--gate-spec",
        type=Path,
        required=True,
        help="Gate whose threshold becomes the logit offset. Must be a "
        "single-signal gate on unknown_exact_minus_top1.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--merge-dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="Precision the LoRA delta is added in before the BF16 cast.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.loads(args.gate_spec.read_text(encoding="utf-8"))
    spec = core.GateSpec.from_dict(payload["gate"])
    if spec.combine != "single" or spec.signal != "unknown_exact_minus_top1":
        raise SystemExit(
            "Only a single-signal gate on unknown_exact_minus_top1 has an exact "
            f"logit-offset form; got {core.describe(spec)}"
        )
    logit_bias = -spec.threshold

    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to write into a non-empty directory: {out}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    encoded = tokenizer.encode(core.ABSTAIN_TEXT, add_special_tokens=False)
    if len(encoded) != 1:
        raise SystemExit(
            f"{core.ABSTAIN_TEXT!r} is {len(encoded)} tokens in this tokenizer; the "
            "logit-offset form requires a single token"
        )
    unknown_token_id = int(encoded[0])

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.merge_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(args.adapter.expanduser().resolve()))

    scaled = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            active = module.active_adapters
            for name in (active,) if isinstance(active, str) else tuple(active):
                module.set_scale(str(name), args.adapter_scale)
                scaled += 1
    if scaled == 0:
        raise SystemExit("No active LoRA adapters to scale")

    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.config.torch_dtype = "bfloat16"
    configure_abstention(merged.config, unknown_token_id, logit_bias)

    generation = GenerationConfig.from_pretrained(args.base)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    generation.save_pretrained(out)
    (out / "modeling_abstain.py").write_text(
        MODELING_ABSTAIN, encoding="utf-8", newline="\n"
    )
    for name in TOKENIZER_FILES:
        source = args.base / name
        if source.is_file():
            shutil.copy2(source, out / name)

    weights = sorted(out.glob("*.safetensors"))
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "abstention_checkpoint",
        "name": args.model_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": {
            "base": str(args.base.resolve()),
            "base_config_sha256": sha256_file(args.base / "config.json"),
            "adapter": str(args.adapter.resolve()),
            "adapter_weights_sha256": sha256_file(
                args.adapter / "adapter_model.safetensors"
            ),
            "adapter_scale": args.adapter_scale,
            "scaled_lora_slots": scaled,
            "merge_dtype": args.merge_dtype,
            "saved_dtype": "bfloat16",
        },
        "abstention": {
            "form": (
                "constant logit offset on a single token, applied inside the "
                "model forward pass by modeling_abstain.Qwen3AbstainForCausalLM "
                "(requires trust_remote_code=True)"
            ),
            "gate_spec": str(args.gate_spec.resolve()),
            "gate_spec_sha256": sha256_file(args.gate_spec),
            "rule": core.describe(spec),
            "unknown_token_id": unknown_token_id,
            "unknown_surface": core.ABSTAIN_TEXT,
            "logit_bias": logit_bias,
            "equivalence": (
                "signal = logP(UNKNOWN) - logP(top1) = z_UNK - z_top1, so "
                "'signal >= threshold' is exactly 'argmax(z + b*e_UNKNOWN) == UNKNOWN' "
                "with b = -threshold"
            ),
            "known_deviation": (
                "the model-class offset applies at every decode step; the gate is defined on "
                "the first generated token only"
            ),
            "generation_config_route_rejected": (
                "Transformers 5.14 serialises GenerationConfig.sequence_bias tuple "
                "keys as strings and then refuses to load them, so the offset is "
                "implemented by the auto_map model class instead"
            ),
        },
        "weights": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weights
        ],
        "implementation": {
            "script": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
        },
    }
    core.write_json(out / "CHECKPOINT.json", manifest)
    print(
        json.dumps(
            {
                "output": str(out),
                "unknown_token_id": unknown_token_id,
                "logit_bias": logit_bias,
                "scaled_lora_slots": scaled,
                "weights": [entry["name"] for entry in manifest["weights"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
