"""CPU mechanics demo, synthetic tiny causal model. Never downloads a checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from .data import Row
from .diagnostics import qualify
from .study import TrainSpec, run_study
from .tangent import CallBudget, FrozenAnchor


class TinyCausalLM(nn.Module):
    """Causal cumulative context + nonlinear block + tied LM head; not Rineya."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        lengths = torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        return self.head(torch.tanh(self.proj(x.cumsum(1) / lengths)))


def fixture_rows() -> list[Row]:
    rows = []
    for task, token, answer in (("A", 1, 6), ("B", 2, 7)):
        for split, offset in (("train", 0), ("dev", 3), ("test", 6)):
            for j in range(3):
                key = f"fixture:{task}:{split}:{j}"
                ids = (token, 3 + offset + j, answer, 0)
                rows.append(Row(key, key, task, split, ids, (-100, -100, answer, 0)))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(1); torch.manual_seed(17)
    anchor = FrozenAnchor(TinyCausalLM().double(), budget=CallBudget(10000))
    rows = fixture_rows()
    q = qualify(anchor, rows[0], max_relative_error=0.01)
    result = run_study(anchor, rows, ("A", "B"),
                       TrainSpec(steps=12, batch_size=3, lr=0.03), alphas=(0.0, 0.5, 1.0))
    result.update({"evidence_class": "SYNTHETIC_CPU_MECHANICS_ONLY", "qualification": q,
                   "pretrained_model_loads": 0, "native_titan_optimizer_steps": 0,
                   "fixture_optimizer_steps": 48, "fixture_sequence_disjointness": True,
                   "warning": "Tiny synthetic fixtures are not an efficacy experiment; do not publish their scores as Titan results"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"status": q["status"], "evidence_class": result["evidence_class"],
                      "pretrained_model_loads": 0, "fixture_optimizer_steps": 48}))
    return 0 if q["status"] == "QUALIFICATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
