"""Synthetic CpuAdamW.step cost at the real tail-20 shapes.

No model weights are loaded, no model forward/backward runs, and no optimizer step
touches a real parameter. This is a pure allocation/transfer/arithmetic benchmark whose
only purpose is to size the watchdog timeout from measurement instead of a guess.
"""
import json, math, struct, time
from pathlib import Path
import torch

from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
from experiments.model_quality_gpu_v1.resources import lease, snapshot

SRC = Path("C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Supa")

with SRC.joinpath("model.safetensors").open("rb") as s:
    n = struct.unpack("<Q", s.read(8))[0]
    header = json.loads(s.read(n).decode("utf-8"))
tail = {k: v for k, v in header.items()
        if k != "__metadata__" and any(k.startswith(f"model.layers.{i}.") for i in range(8, 28))}
shapes = sorted((k, tuple(v["shape"])) for k, v in tail.items())
total = sum(math.prod(sh) for _, sh in shapes)
print(f"tail tensors={len(shapes)} params={total}")
assert (len(shapes), total) == (220, 314618880), "shape inventory differs from the lock"

torch.set_num_threads(1)          # exactly what native.configure_runtime does
torch.manual_seed(17)

results = {}
for device in ("cpu", "cuda:0"):
    named = []
    for name, sh in shapes:
        p = torch.nn.Parameter(torch.zeros(sh, dtype=torch.float32, device=device))
        p.requires_grad_(True)
        named.append((name, p))
    opt = CpuAdamW(named, Design(processed_budget=167256))
    timings = []
    for i in range(4):
        for _, p in named:
            p.grad = torch.full_like(p, 1e-3)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.monotonic()
        opt.step(1e-5)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.monotonic() - t0)
        opt.zero_grad()
    results[device] = timings
    print(f"{device}: per-step seconds {[round(t,2) for t in timings]}  median={sorted(timings)[len(timings)//2]:.2f}")
    del named, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

med = sorted(results["cuda:0"])[len(results["cuda:0"]) // 2]
print()
print(f"MEASURED cuda-resident median step = {med:.2f} s")
for updates, label in ((105, "one full attempt (105 updates)"), (103, "resume attempt (103 updates)")):
    print(f"  {label}: {updates * med:.0f} s optimizer-only")
print(f"  both attempts optimizer-only: {(105 + 103) * med:.0f} s")
print()
print("resources after:", json.dumps(snapshot()["gpu0"]))
