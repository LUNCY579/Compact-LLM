#!/usr/bin/env python3
"""Measure conversion damage on a stacked decoder at the real Rineya architecture.

Development inputs only. No Phase-1 confirmatory data is read.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import torch
from torch import nn
from rineya_tgsa import NativeReferenceAttention, RineyaTGSAAttention, RineyaTGSAConfig
from rineya_tgsa.damage import DAMAGE_SCHEMA_VERSION, measure_conversion_damage, render_markdown


class TinyDecoder(nn.Module):
    """Minimal stacked decoder so damage is measured through a real depth."""

    def __init__(self, config: RineyaTGSAConfig, layers: int, vocab: int, retrofit: set[int]):
        super().__init__()
        self.config, self.retrofit = config, retrofit
        self.embed = nn.Embedding(vocab, config.hidden_size)
        self.blocks = nn.ModuleList(
            (RineyaTGSAAttention(config, layer_index=i) if i in retrofit
             else NativeReferenceAttention(config))
            for i in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(config.hidden_size) for _ in range(layers))
        self.head = nn.Linear(config.hidden_size, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        T = input_ids.shape[1]
        mask = torch.full((T, T), float("-inf"), dtype=x.dtype, device=x.device).triu(1)[None, None]
        hidden = []
        for block, norm in zip(self.blocks, self.norms):
            out = block(norm(x), attention_mask=mask)
            x = x + (out[0] if isinstance(out, tuple) and len(out) == 4 else out[0])
            hidden.append(x)
        return self.head(x), hidden


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--convert-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260829)
    args = p.parse_args()

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    config = RineyaTGSAConfig.from_source_config(source, trajectory_rank=32)
    retrofit_layers = set(range(0, args.layers, args.convert_every))

    reports = []
    for dtype, name, lengths in ((torch.float64, "float64", [64]),
                                 (torch.float32, "float32", [64, 256]),
                                 (torch.bfloat16, "bfloat16", [64, 256])):
        for length in lengths:
            torch.manual_seed(args.seed)
            native = TinyDecoder(config, args.layers, args.vocab, retrofit=set()).eval().to(dtype)
            torch.manual_seed(args.seed)
            tgsa = TinyDecoder(config, args.layers, args.vocab, retrofit=retrofit_layers).eval().to(dtype)
            # transplant every native tensor into the retrofit
            tgsa.embed.load_state_dict(native.embed.state_dict())
            tgsa.head.load_state_dict(native.head.state_dict())
            for i in range(args.layers):
                tgsa.norms[i].load_state_dict(native.norms[i].state_dict())
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    getattr(tgsa.blocks[i], proj).load_state_dict(
                        getattr(native.blocks[i], proj).state_dict())
            torch.manual_seed(args.seed + 1)
            ids = torch.randint(0, args.vocab, (args.batch, length))
            labels = torch.randint(0, args.vocab, (args.batch, length))
            report = measure_conversion_damage(
                native_forward=native, retrofit_forward=tgsa,
                input_ids=ids, labels=labels, dtype_name=name,
                exact_in_float64=(dtype is torch.float64),
            )
            reports.append(report.to_dict())
            print(f"  {name:9s} seq={length:4d}  KL_max={report.token_kl.get('max',0):.3e}  "
                  f"top1={report.top1_agreement:.4f}  |dlogit|max={report.output_absolute_drift.get('max',0):.3e}",
                  flush=True)

    context = {
        "architecture": f"hidden={config.hidden_size} heads={config.num_attention_heads} "
                        f"kv={config.num_key_value_heads} head_dim={config.head_dim} "
                        f"rope_theta={config.rope_theta:g}",
        "converted_layers": sorted(retrofit_layers),
        "conditioning": list(config.conditioning_targets),
        "total_layers": args.layers,
    }
    payload = {"schema_version": DAMAGE_SCHEMA_VERSION, "context": context, "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(reports, context=context), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
