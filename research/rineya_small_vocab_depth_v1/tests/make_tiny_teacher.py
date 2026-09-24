"""Build a TINY random-weight Qwen3 with the REAL teacher tokenizer and sidecars, for pipeline dry
runs of run_layer_study.py. Random weights: every number it produces is meaningless; only the
plumbing is being tested. 6 blocks, hidden 64 -> ~9.8M parameters (the vocabulary dominates)."""
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

teacher, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
cfg = json.loads((teacher / "config.json").read_text(encoding="utf-8"))
cfg.update(num_hidden_layers=6, hidden_size=64, intermediate_size=128, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_window_layers=6)
torch.manual_seed(0)
m = Qwen3ForCausalLM(Qwen3Config(**cfg)).to(torch.bfloat16)
m.save_pretrained(str(out), safe_serialization=True)
for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
    shutil.copyfile(teacher / n, out / n)
print(sum(p.numel() for p in m.parameters()))
