"""Resource scaling across context length, per arm.

Fixes three measurement defects in the P2 profiling:

1. no warmup pass, so arm A's first measurement included CUDA context creation
   (which is why its recorded tokens/sec were non-monotonic across lengths);
2. full-vocabulary logits were materialised for every position, which is 5 GB at
   16K tokens and makes long-context profiling impossible -- we ask for the last
   position only;
3. generation latency was measured from a 64-token prefix regardless of the
   context length being profiled, so it did not measure long-context decoding.

It also records the consequence of the wrapper's KV-cache refusal: with
use_cache=False every decode step recomputes the whole prefix, so generation is
quadratic in context length. That is a real cost of this design and is reported
rather than hidden.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm, count_fwa_tensors, count_trainable  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.profile import kv_cache_bytes, recurrent_state_bytes  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter  # noqa: E402

P2 = ARTIFACTS / "p2_abcd"
FILLER = (
    "The harbour ledger notes routine weather, ordinary cargo, and no change to the "
    "standing watch roster for this interval. "
)


def cpu_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def make_ids(tokenizer: Any, n: int, device: str) -> torch.Tensor:
    text = FILLER * (n // 16 + 32)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    while ids.numel() < n:
        ids = torch.cat([ids, ids])
    return ids[:n].unsqueeze(0).to(device)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_forward(model: Any, ids: torch.Tensor, repeats: int) -> float:
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
    sync()
    return (time.perf_counter() - t0) / repeats


def profile_one(
    model: Any,
    tokenizer: Any,
    handle: Any,
    seq_len: int,
    device: str,
    *,
    repeats: int,
    gen_tokens: int,
    gen_max_len: int,
) -> dict[str, Any]:
    ids = make_ids(tokenizer, seq_len, device)
    # warmup (excluded from every timing below)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fwd = timed_forward(model, ids, repeats)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    row: dict[str, Any] = {
        "seq_len": int(seq_len),
        "forward_seconds": fwd,
        "prefill_tokens_per_sec": seq_len / fwd if fwd > 0 else None,
        "ttft_seconds": fwd,  # use_cache=False: TTFT == one full forward
        "peak_vram_bytes": peak,
        "cpu_rss_bytes": cpu_rss_bytes(),
        "recurrent_state_bytes": recurrent_state_bytes(handle),
        "kv_cache_bytes_if_used": kv_cache_bytes(seq_len),
        "trainable_parameters": count_trainable(handle),
        "fwa_parameters": count_fwa_tensors(handle),
    }
    if seq_len <= gen_max_len:
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids=ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        sync()
        gen = time.perf_counter() - t0
        row["generate_seconds"] = gen
        row["generate_tokens"] = gen_tokens
        row["decode_seconds_per_token"] = gen / gen_tokens
    else:
        row["generate_seconds"] = None
        row["generate_note"] = f"skipped above {gen_max_len} tokens (no KV cache: cost is O(new_tokens * T))"
    del ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--arms", type=str, nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=4)
    ap.add_argument("--gen-max-len", type=int, default=2048)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "scaling.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    model = load_supa(device=device, eval_mode=True)
    total_params = sum(p.numel() for p in model.parameters())

    results: dict[str, Any] = {}
    for arm in args.arms:
        handle = apply_arm(model, arm, layers=[args.layer], rank=args.rank, init_alpha=0.0)
        adapter = P2 / f"arm_{arm}" / f"seed_{args.seed}" / "fwa_adapter.pt"
        if handle is not None and adapter.is_file():
            load_adapter(handle, adapter)
        rows = []
        try:
            for n in args.lengths:
                try:
                    row = profile_one(
                        model, tokenizer, handle, n, device,
                        repeats=args.repeats, gen_tokens=args.gen_tokens,
                        gen_max_len=args.gen_max_len,
                    )
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    row = {"seq_len": int(n), "oom": True}
                rows.append(row)
                tps = row.get("prefill_tokens_per_sec")
                pv = row.get("peak_vram_bytes")
                print(
                    f"arm {arm} T={n:6d} "
                    f"prefill={tps:10.1f} tok/s  " if tps else f"arm {arm} T={n:6d} OOM  ",
                    end="",
                )
                if pv:
                    print(f"peak={pv/2**30:.2f} GiB  state={row['recurrent_state_bytes']}B "
                          f"kv_if_used={row['kv_cache_bytes_if_used']/2**20:.0f} MiB")
                else:
                    print()
        finally:
            if handle is not None:
                handle.uninstall()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[arm] = rows

    payload = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_parameters": total_params,
        "layer": args.layer,
        "rank": args.rank,
        "adapter_seed": args.seed,
        "arms": results,
    }
    write_json(Path(args.out), payload, config=vars(args))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
