"""Hard-exit fixture driver; CPU tiny adapters only, never loads a real model."""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
from test_resume_equivalence import DATA, StochasticAdapter, options, seed, train
import torch

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["uninterrupted", "interrupt", "resume", "mid_accumulation"])
parser.add_argument("directory", type=Path)
args = parser.parse_args()
seed()


class HardExitAdapter(StochasticAdapter):
    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        if args.mode == "mid_accumulation" and torch.is_grad_enabled() and self.train_calls == 4:
            os._exit(37)
        return out


def callback(record):
    if args.mode == "interrupt" and record["event"] == "optimizer_attempt" and record["step"] == 2:
        os._exit(37)


model = HardExitAdapter().eval()
result = train(model, DATA, **options(args.directory, resume_from_checkpoint=args.mode == "resume", progress_callback=callback))
torch.save({"result": result, "weight": model.lora_A.weight.detach()}, args.directory / "final_result.pt")
