"""v4 — fit a per-layer orthogonal basis map from Qwen3-8B to Supa-A1.1.

v3.1 fitted maps at five relative depths. P6 rebases *every* layer, so it needs a
map per target layer. Capturing all 36 donor layers over all 9,013 tokens would
be 5.3 GiB, so this subsamples token positions during capture and keeps the donor
activation set under 1 GiB.

Orthogonal Procrustes, not ridge. An orthogonal map preserves norms and therefore
commutes with RMSNorm up to a scale; ridge does not, and using it to rebase
weights would be a category error. Amendment v4 freezes this choice.

The map for target layer `t` is fitted between the residual streams *entering*
that layer and the donor layer it corresponds to under the frozen D2 uniform
relative-depth map, which v3.1's E5 validated functionally.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.architecture import uniform_depth_map  # noqa: E402
from rineya_cross_scale.hashing import atomic_write_json  # noqa: E402
from rineya_cross_scale.streaming_forward import (  # noqa: E402
    StreamingWeights,
    streaming_forward_many,
)

SEED = 17
TOKENIZER = Path("C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
                 "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
DONOR = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores"  # placeholder, unused
TARGET = ("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
          "Rineya-Supa-A1.1-1.1.0-devfreeze")


def free_ram_gib() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return float("nan")


def rss_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float("nan")


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def procrustes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthogonal map A minimising ||x A - y||, plus its held-out-free fit cosine."""
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    a = (u @ vt).astype(np.float32)
    predicted = x @ a
    numerator = (predicted * y).sum(-1)
    denominator = np.maximum(np.linalg.norm(predicted, axis=-1)
                             * np.linalg.norm(y, axis=-1), 1e-30)
    return a, float((numerator / denominator).mean())


def capture(path: Path, prompts: list[torch.Tensor], select: list[torch.Tensor],
            label: str, floor: float,
            wanted_layers: tuple[int, ...] | None = None,
            ceiling: float = 5.0) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    with StreamingWeights(path) as weights:
        layers = (wanted_layers if wanted_layers is not None
                  else tuple(range(int(weights.config["num_hidden_layers"]))))
        width = int(weights.config["hidden_size"])

        def report(done: int, total: int) -> None:
            available = free_ram_gib()
            if done % 6 == 0 or done == total:
                print(f"  [{label}] layer {done}/{total} "
                      f"({time.perf_counter() - started:.0f}s, rss {rss_gib():.2f} GiB, "
                      f"free {available:.2f} GiB)", flush=True)
            # Guard PROCESS RSS, not system free RAM. Streaming a 15 GiB
            # checkpoint fills the OS file cache, which drives "free" to zero
            # while remaining fully reclaimable -- it is not a crash risk and
            # earlier v3.1 passes completed in exactly that state. Anonymous
            # memory held by this process is the quantity that can actually OOM.
            resident = rss_gib()
            if resident > ceiling:
                raise SystemExit(
                    f"ABORT: process RSS {resident:.2f} GiB exceeded the {ceiling} GiB "
                    "ceiling frozen in amendment v4. Re-run with a smaller --tokens.")
            if available < floor:
                print(f"  [{label}] note: free RAM {available:.2f} GiB is OS file cache "
                      f"from streaming; process RSS is {resident:.2f} GiB", flush=True)

        with torch.inference_mode():
            produced = streaming_forward_many(
                weights, prompts, capture_layers=layers, want_logits=False,
                pooling="select", select_positions=select, progress=report,
                recycle_handles_every=4,
            )
    slots = ["embedding"] + [f"layer_{i}" for i in layers]
    slot_index = {"embedding": 0, **{f"layer_{i}": n + 1 for n, i in enumerate(layers)}}
    stacked = np.stack(
        [np.concatenate([p["captured"][s].numpy() for p in produced], axis=0)
         for s in slots], axis=1)          # [tokens, slots, width]
    del produced
    gc.collect()
    return stacked, {"label": label, "path": str(path), "width": width,
                     "layers": len(layers), "slots": len(slots),
                     "slot_index": slot_index, "captured_layers": list(layers),
                     "tokens": int(stacked.shape[0]),
                     "wall_seconds": time.perf_counter() - started,
                     "rss_gib_after": rss_gib(), "free_gib_after": free_ram_gib()}


def main() -> int:
    parser = argparse.ArgumentParser(description="v4 per-layer basis map fitting.")
    parser.add_argument("--donor", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=Path(TARGET))
    parser.add_argument("--dataset", type=Path,
                        default=LAB_ROOT / "data" / "evaluation" / "calibration.jsonl")
    parser.add_argument("--tokens", type=int, default=1200)
    parser.add_argument("--holdout-prompts", type=int, default=15)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ram-floor-gib", type=float, default=0.5)
    parser.add_argument("--rss-ceiling-gib", type=float, default=5.0)
    parser.add_argument("--output", type=Path,
                        default=LAB_ROOT / "local_models" / "v4_basis_maps")
    args = parser.parse_args()

    if args.donor is None:
        record = json.loads(
            (LAB_ROOT / "cross_scale_evidence/02_architecture/qwen8b.json").read_text("utf-8"))
        args.donor = Path(record["resolved_path"])

    torch.set_num_threads(args.threads)
    print(f"[env] threads {args.threads}, free RAM {free_ram_gib():.2f} GiB, "
          f"floor {args.ram_floor_gib} GiB", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = [json.loads(l) for l in args.dataset.read_text("utf-8").splitlines() if l.strip()]
    encoded = [tokenizer(render_prompt(tokenizer, r["prompt"]), return_tensors="pt",
                         truncation=True, max_length=4096)["input_ids"][0] for r in rows]

    # Deterministic token subsample, spread proportionally across prompts, with a
    # prompt-disjoint holdout so the map's fit can be reported on unseen prompts.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    holdout = set(order[: args.holdout_prompts].tolist())
    total = sum(int(e.shape[0]) for e in encoded)
    select, prompt_of = [], []
    for index, e in enumerate(encoded):
        length = int(e.shape[0])
        take = max(1, round(args.tokens * length / total))
        take = min(take, length)
        picks = np.linspace(0, length - 1, take).round().astype(int)
        picks = np.unique(picks)
        select.append(torch.from_numpy(picks))
        prompt_of.append(np.full(len(picks), index, dtype=np.int32))
    prompt_of = np.concatenate(prompt_of)
    print(f"[plan] {len(prompt_of)} sampled token positions from {len(rows)} prompts; "
          f"{len(holdout)} prompts held out", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    target_config = json.loads((args.target / "config.json").read_text("utf-8"))
    donor_config = json.loads((args.donor / "config.json").read_text("utf-8"))
    target_layer_count = int(target_config["num_hidden_layers"])
    donor_layer_count = int(donor_config["num_hidden_layers"])
    depth_map = uniform_depth_map(target_layer_count, donor_layer_count)

    # The residual ENTERING layer L is the output of layer L-1, or the embedding
    # for L = 0. Capture only those slots: for the donor that is 27 layers rather
    # than 36, which is a quarter less activation memory for free.
    donor_needed = tuple(sorted({d - 1 for d in depth_map if d > 0}))
    target_needed = tuple(sorted({t - 1 for t in range(target_layer_count) if t > 0}))

    target_states, target_meta = capture(args.target, encoded, select, "target",
                                         args.ram_floor_gib, target_needed,
                                         args.rss_ceiling_gib)
    print(f"[done] target {target_states.shape} {target_meta['wall_seconds']:.0f}s", flush=True)
    donor_states, donor_meta = capture(args.donor, encoded, select, "donor",
                                       args.ram_floor_gib, donor_needed,
                                       args.rss_ceiling_gib)
    print(f"[done] donor {donor_states.shape} {donor_meta['wall_seconds']:.0f}s", flush=True)

    def slot_of(meta: dict[str, Any], entering_layer: int) -> int:
        key = "embedding" if entering_layer == 0 else f"layer_{entering_layer - 1}"
        return meta["slot_index"][key]
    train = ~np.isin(prompt_of, list(holdout))
    test = ~train

    maps, rows_out = {}, []
    for target_layer in range(target_layer_count):
        donor_layer = depth_map[target_layer]
        x = donor_states[:, slot_of(donor_meta, donor_layer), :].astype(np.float64)
        y = target_states[:, slot_of(target_meta, target_layer), :].astype(np.float64)
        a, fit_cosine = procrustes(x[train], y[train])
        predicted = x[test] @ a
        held = float((( predicted * y[test]).sum(-1) / np.maximum(
            np.linalg.norm(predicted, axis=-1) * np.linalg.norm(y[test], axis=-1), 1e-30)).mean())
        maps[f"layer_{target_layer}"] = a
        rows_out.append({"target_layer": target_layer, "donor_layer": donor_layer,
                         "train_cosine": fit_cosine, "heldout_cosine": held})
        print(f"  map t{target_layer:02d} <- d{donor_layer:02d}: "
              f"train {fit_cosine:+.4f} heldout {held:+.4f}", flush=True)
        del x, y, predicted
        if target_layer % 8 == 7:
            gc.collect()

    np.savez_compressed(args.output / "basis_maps_qwen8b_to_supa_a11.npz", **maps)
    atomic_write_json(args.output / "basis_maps_manifest.json", {
        "schema_version": "1.0",
        "experiment": "v4_basis_map_fitting",
        "protocol": "amendment_v4_aligned_transport.md 4.0.0-exploratory",
        "method": "orthogonal Procrustes per target layer",
        "donor": str(args.donor), "target": str(args.target),
        "depth_map": depth_map,
        "sampled_tokens": int(len(prompt_of)),
        "holdout_prompts": sorted(holdout),
        "train_tokens": int(train.sum()), "test_tokens": int(test.sum()),
        "layers": rows_out,
        "mean_heldout_cosine": float(np.mean([r["heldout_cosine"] for r in rows_out])),
        "capture": {"target": target_meta, "donor": donor_meta},
        "resources": {"free_gib_at_end": free_ram_gib(), "rss_gib_at_end": rss_gib()},
    })
    print(f"\nmean held-out cosine across {len(rows_out)} layers: "
          f"{np.mean([r['heldout_cosine'] for r in rows_out]):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
