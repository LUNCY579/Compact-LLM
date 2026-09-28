"""OSIDE-001 per-token O-side scorer (PREREG.md sections 2-3).

Same rows, prompt, answer and tokenisation as titan_opallo_skill_vector_v1/full_scale/o_reply_nll.py, whose
make_rows() is imported, and the same teacher-forced log-softmax readout. The difference is that it keeps every
reply token's NLL, so the terminal '.'/'?'/'!' token can be separated from the content tokens. It also stores the
row mean exactly as o_reply_nll computed it (`mean_nll_gpu`), which is the canary against O_REPLY_NLL.json.

--cast-bf16 rounds every parameter to BF16 in memory and computes in FP32: the cast-noise arm (DG2). For
OPALLO that equals the published Opallo checkpoint bit for bit (titan_opallo_skill_vector_v1/GEOMETRY.json,
opallo_roundtrip 596,049,920 of 596,049,920 elements), and three tensors are spot-checked against it here.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FS = EXP / "titan_opallo_skill_vector_v1" / "full_scale"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
import o_reply_nll as O  # noqa: E402
from seekst import SeekST  # noqa: E402

MANIFEST = HERE / "ROWS_MANIFEST.json"
MANIFEST_SHA = "19834d340aff2df12dc5f2333b7e9354832f3c916989c1d9ab459986586dd63b"
PUBLISHED_OPALLO = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/"
                        r"Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/model.safetensors")
CAST_SPOT = ("model.layers.27.mlp.down_proj.weight", "model.layers.20.self_attn.q_proj.weight", "model.embed_tokens.weight")
VRAM_FRACTION = 0.78
REGISTERED_RUNTIME = ("3.11.15", "2.6.0+cu124", "5.14.1")   # PREREG section 10; enforced by AMENDMENT_1
REGISTERED_DEVICE = "NVIDIA GeForce RTX 4070 Laptop GPU"      # as recorded in O_REPLY_NLL.json; AMENDMENT_1


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cast_spot_check(model, torch) -> dict:
    """The in-memory BF16 cast of OPALLO against the published BF16 bytes, on three tensors."""
    import numpy as np
    pub = SeekST(PUBLISHED_OPALLO)
    params = dict(model.named_parameters())
    out = {}
    try:
        for k in CAST_SPOT:
            shape = pub.info(k)[1]
            want = np.frombuffer(pub.raw(k, 0, shape[0]), dtype="<u2")
            got = params[k].detach().to(torch.bfloat16).cpu().view(torch.int16).numpy().view("<u2").reshape(-1)
            out[k] = int((got != want).sum())
    finally:
        pub.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--expect-sha", required=True, help="sha256 of model.safetensors; checked before loading")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cast-bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: first N rows; the analysis rejects such files")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise SystemExit("ROWS_MANIFEST.json does not match its registered hash")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # AMENDMENT_1: the prompt/answer source must still be the one the manifest was built from
    inputs = {"o_reply_nll_py_sha256": sha256(Path(O.__file__)), "source_train_sha256": sha256(O.TRAIN),
              "consumed_arm_dose_sha256": sha256(O.ARM_DOSE)}
    for k, v in inputs.items():
        if v != man[k]:
            raise SystemExit(f"{k}: the imported source differs from the manifest ({v[:12]} vs {man[k][:12]})")
    rows = O.make_rows()
    if [r["id"] for r in rows] != [m["id"] for m in man["rows"]]:
        raise SystemExit("make_rows() order differs from the manifest")
    if a.limit is not None:
        if a.output.resolve().parent == (HERE / "scores").resolve():
            raise SystemExit("--limit output may not go to scores/")
        rows, man["rows"] = rows[:a.limit], man["rows"][:a.limit]
    if sha256(a.model_dir / "tokenizer.json") != man["tokenizer_json_sha256"]:
        raise SystemExit("tokenizer.json differs from the manifest's")
    wsha = sha256(a.model_dir / "model.safetensors")
    if wsha != a.expect_sha:
        raise SystemExit(f"{a.arm}: weights hash {wsha} != expected {a.expect_sha}")

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if (platform.python_version(), torch.__version__, transformers.__version__) != REGISTERED_RUNTIME:
        raise SystemExit(f"runtime {(platform.python_version(), torch.__version__, transformers.__version__)} "
                         f"differs from the registered {REGISTERED_RUNTIME}")
    if torch.cuda.get_device_name(0) != REGISTERED_DEVICE:
        raise SystemExit(f"device {torch.cuda.get_device_name(0)!r} differs from the registered {REGISTERED_DEVICE!r}")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)   # past VRAM, Windows pages into host RAM
    started = time.time()
    tok = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model_dir), torch_dtype=torch.float32, local_files_only=True,
                                                 low_cpu_mem_usage=True, device_map={"": "cuda"})
    model.eval()
    spot = None
    if a.cast_bf16:
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
        spot = cast_spot_check(model, torch)
        if any(spot.values()):
            raise SystemExit(f"BF16 cast does not reproduce the published Opallo bytes: {spot}")
    device = next(model.parameters()).device
    out_rows = []
    for ix, (row, m) in enumerate(zip(rows, man["rows"]), 1):
        pids = tok(row["prompt"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        tids = tok(row["answer"], add_special_tokens=False)["input_ids"]
        if len(tids) != m["n_tokens"] or tids[-1] != m["final_token_id"]:
            raise SystemExit(f"row {row['id']}: tokenisation differs from the manifest")
        ids = torch.tensor([pids.tolist() + tids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0].float()
        pos = torch.arange(pids.numel() - 1, pids.numel() + len(tids) - 1, device=device)
        targets = torch.tensor(tids, dtype=torch.long, device=device)
        logp = torch.log_softmax(logits[pos], dim=-1).gather(1, targets[:, None]).squeeze(1)
        if not bool(torch.isfinite(logp).all()):             # AMENDMENT_1: a crash, never a stored NaN
            raise SystemExit(f"{a.arm} row {row['id']}: non-finite NLL")
        out_rows.append({"id": row["id"], "nll": (-logp).tolist(), "mean_nll_gpu": float(-logp.mean().item())})
        if ix % 1000 == 0:
            print(json.dumps({"arm": a.arm, "rows": ix, "seconds": round(time.time() - started, 1)}), flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    res = {"schema": 1, "experiment": "OSIDE-001", "instrument": "OPALLO-HELDOUT-REPLY-PER-TOKEN-NLL-001",
           "arm": a.arm, "model_dir": str(a.model_dir), "weight_sha256": wsha, "cast_bf16": a.cast_bf16, "limit": a.limit,
           "cast_spot_check_mismatches": spot, "manifest_sha256": MANIFEST_SHA, "input_sha256": inputs,
           "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
                       "dtype": "float32", "tf32": False, "use_cache": False,
                       "readout": "teacher-forced final assistant reply tokens, per-token NLL"},
           "seconds": round(time.time() - started, 1), "rows": out_rows,
           "quality_claim": False, "release_permitted": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".partial")
    with open(tmp, "wb") as f:                                # AMENDMENT_1: durable before the rename
        f.write((json.dumps(res) + "\n").encode("utf-8"))
        f.flush(); os.fsync(f.fileno())
    tmp.rename(a.output)
    print(f"wrote {a.output} ({len(out_rows)} rows, {res['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
