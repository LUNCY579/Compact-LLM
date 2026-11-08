"""OSIDE-001 arm builder (PREREG.md section 1). CPU only, create-only, seek/read reader, fsync every 64 MB.

Subcommands
  rebuild ARM       ARM in {ADD-O50, H1, H1c}. Re-creates CONVENTION-LOC-001's step-1 arm byte for byte:
                    SPARSE + c(tensor)*(ADD-100 - SPARSE), with convention_localization_v1/build_beta_arms.py's
                    own chunks() and coef_fn(), and its header metadata. The result must hash to the sha256
                    recorded in convention_localization_v1/arms/ARM/BUILD.json (committed). On a mismatch the
                    weights are deleted, REBUILD.json records it, and the exit code is 4.
  seed N            N in {23, 29, 41}: Supa (stored BF16, upcast once) with the trained tensors of Opallo's
                    seed-N replicate (same recipe as Opallo, B28last20 lr1e-05 b5200 wsd0.2-0.2 b2 0.95), by
                    titan_opallo_skill_vector_v1/compose.py's recipe for its OPALLO endpoint.
  seed-control      Positive control for `seed`: the same stream built from seed 17 with compose.py's header must
                    hash to the stored OPALLO arm (5ff1c9a5...) without writing anything.
  delete ARM        After scoring: removes the weights and copied assets, keeps BUILD.json (and REBUILD.json),
                    writes WEIGHTS_DELETED.json. Refuses unless scores/ARM.json carries the same weight hash.

Every source is opened read-only. Outputs go only to o_side_module_vs_magnitude_v1/arms/.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, shutil, struct, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from seekst import SeekST  # noqa: E402

SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
CL = EXP / "convention_localization_v1"
ARMS = HERE / "arms"
SCORES = HERE / "scores"
ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "config.json")
CHUNK = 4_000_000
FSYNC_EVERY = 64 << 20

REBUILT = {"ADD-O50": ("beta:0.5", "bc69a4ec403b74fb09de815ec55738a7126da345802a7b45e520f205e3c68051"),
           "H1": ("H1", "3ff2aa596ccdc8234dae053362c1cabc8f5c3e274f3ea78591a9a7b219729fc1"),
           "H1c": ("H1c", "ea46496c1a713c971589f3575532d0dd7f64e85cf7eaf8dd410c4b206c82043e")}
SUPA = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/model.safetensors")
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
CKPT_ROOT = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/"
                 r"architecture_benchmark_001/outputs/arm-checkpoints")
TRAINED_SHA = {17: "915af77f4ad2a1813df46a5b5a65b9dddd138b820e2e9e53ad8845bdf8d0db0d",
               23: "22a69c163ad039720aa8d7ef22f2e23e238d8a9193d4672df75ca9023d091c5e",
               29: "085ed7d8ff4e8466f79fb6dc9e6e0cb8cab969c8704ac98962b4280d5f71a216",
               41: "b9da8dbcbdcc58a1fe99491a9179c28aa7e84650230af29a205258c2ca1e9666"}
OPALLO_SHA, OPALLO_BYTES = "5ff1c9a5cf3674ace09eaf33bfc21431ea63718cff3cbf5b52701fd3de1416f8", 2384235032


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ckpt(seed: int) -> Path:
    return CKPT_ROOT / f"B28last20_lr1e-05_b5200_seed{seed}_wsd0.2-0.2_b20.95" / "trainable.safetensors"


def header_bytes(meta: dict, keys, shape_of) -> tuple[bytes, int]:
    header, off = {"__metadata__": meta}, 0
    for k in keys:
        shape = shape_of(k)
        nb = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    enc = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    enc += b" " * (-len(enc) % 8)
    return struct.pack("<Q", len(enc)) + enc, off


def rebuild_stream(arm: str):
    """CONVENTION-LOC-001's build(): same key order, header, chunks() arithmetic and float32 output."""
    bba = _load("cl_build_beta_arms", CL / "build_beta_arms.py")
    spec, _ = REBUILT[arm]
    cf = bba.coef_fn(spec)
    sp, ad = SeekST(SV / "SPARSE" / "model.safetensors"), SeekST(SV / "ADD-100" / "model.safetensors")
    try:
        keys = sp.keys()
        if keys != ad.keys():
            raise SystemExit("SPARSE and ADD-100 key sets differ")
        prefix, nbytes = header_bytes({"format": "pt", "experiment": "CONVENTION-LOC-001", "arm": arm}, keys,
                                      lambda k: sp.info(k)[1])
        yield prefix
        for k in keys:
            for _, _, w in bba.chunks(sp, ad, k, cf(k)):
                if not np.isfinite(w).all():
                    raise RuntimeError(f"nonfinite in {k}")
                yield w.astype("<f4", copy=False).tobytes(order="C")
    finally:
        sp.close(); ad.close()


def seed_stream(seed: int, experiment: str, arm: str):
    """compose.py's OPALLO endpoint for a given trained file: trained FP32 tensors where Opallo's harness trained
    (CED names mapped by compose.qwen_to_ced), exact FP32 upcast of Supa's stored BF16 everywhere else."""
    comp = _load("sv_compose", EXP / "titan_opallo_skill_vector_v1" / "compose.py")
    base, tr = SeekST(SUPA), SeekST(ckpt(seed))
    try:
        keys = [k for k in base.keys() if k != "lm_head.weight"]
        if len(keys) != 310:
            raise SystemExit(f"unexpected Supa tensor count {len(keys)}")
        tmap = {k: comp.qwen_to_ced(k) for k in keys}
        tmap = {k: c for k, c in tmap.items() if c in tr.header}
        if len(tmap) != 220 or set(tmap.values()) != set(tr.header):
            raise SystemExit(f"seed {seed}: {len(tmap)} mapped trained tensors, {len(tr.header)} in the file (want 220 = all)")
        for k, c in tmap.items():
            if tr.info(c)[0] != "F32" or tr.info(c)[1] != base.info(k)[1]:
                raise SystemExit(f"seed {seed}: {c} dtype/shape does not match Supa {k}")
        prefix, _ = header_bytes({"format": "pt", "experiment": experiment, "arm": arm}, keys, lambda k: base.info(k)[1])
        yield prefix
        for k in keys:
            src, name = (tr, tmap[k]) if k in tmap else (base, k)
            shape = base.info(k)[1]
            per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
            step = max(1, CHUNK // per_row)
            for lo in range(0, shape[0], step):
                w = src.rows(name, lo, min(lo + step, shape[0]))
                if not np.isfinite(w).all():
                    raise RuntimeError(f"nonfinite in {k}")
                yield w.astype("<f4", copy=False).tobytes(order="C")
    finally:
        base.close(); tr.close()


def write_arm(stream, out: Path, assets_from: Path) -> tuple[str, int]:
    """Create-only. Hashes while writing; fsyncs every 64 MB so dirty pages never pile up in host RAM."""
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    tmp = out / "model.safetensors.partial"
    h, n, since = hashlib.sha256(), 0, 0
    with open(tmp, "xb") as f:
        for blk in stream:
            f.write(blk); h.update(blk); n += len(blk); since += len(blk)
            if since >= FSYNC_EVERY:
                f.flush(); os.fsync(f.fileno()); since = 0
        f.flush(); os.fsync(f.fileno())
    for a in ASSETS:
        shutil.copyfile(assets_from / a, out / a)
    tmp.rename(out / "model.safetensors")
    return h.hexdigest(), n


def write_json(p: Path, obj) -> None:
    p.write_bytes((json.dumps(obj, indent=2) + "\n").encode("utf-8"))


def cmd_rebuild(arm: str) -> int:
    spec, want = REBUILT[arm]
    committed = json.loads((CL / "arms" / arm / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
    if committed != want:
        raise SystemExit(f"{arm}: committed BUILD.json hash {committed} != the hash registered here {want}")
    out = ARMS / arm
    t0 = time.time()
    got, n = write_arm(rebuild_stream(arm), out, SV / "SPARSE")
    rec = {"experiment": "OSIDE-001", "arm": arm, "spec": spec,
           "construction": "rebuild of CONVENTION-LOC-001 step 1: SPARSE + c(tensor)*(ADD-100 - SPARSE), FP32, streaming",
           "weight_sha256": got, "weight_bytes": n, "recorded_sha256": want, "match": got == want,
           "sources_sha256_from": "titan_opallo_skill_vector_v1/arms/{SPARSE,ADD-100}/BUILD.json",
           "seconds": round(time.time() - t0, 1), "quality_claim": False}
    write_json(out / "BUILD.json", rec)
    print(json.dumps(rec), flush=True)
    if got != want:
        (out / "model.safetensors").unlink()
        write_json(out / "REBUILD_MISMATCH.json", rec)
        print(f"REBUILD MISMATCH for {arm}: weights deleted; this arm must not be scored", flush=True)
        return 4
    return 0


def cmd_seed(seed: int) -> int:
    arm = f"SEED{seed}"
    t0 = time.time()
    got, n = write_arm(seed_stream(seed, "OSIDE-001", arm), ARMS / arm, SV / "OPALLO")
    rec = {"experiment": "OSIDE-001", "arm": arm, "seed": seed,
           "construction": "compose.py OPALLO-endpoint recipe: trained FP32 tensors of the seed replicate on the "
                           "220 trained tensors, FP32(Supa_BF16) elsewhere",
           "trained_file": str(ckpt(seed)), "trained_sha256_registered": TRAINED_SHA[seed],
           "supa_sha256_registered": SUPA_SHA, "weight_sha256": got, "weight_bytes": n,
           "seconds": round(time.time() - t0, 1), "quality_claim": False}
    write_json(ARMS / arm / "BUILD.json", rec)
    print(json.dumps(rec), flush=True)
    return 0


def seed_control() -> dict:
    h, n = hashlib.sha256(), 0
    for blk in seed_stream(17, "TITAN-OPALLO-SKILL-VECTOR-001", "OPALLO"):
        h.update(blk); n += len(blk)
    return {"sha256": h.hexdigest(), "bytes": n, "want_sha256": OPALLO_SHA, "want_bytes": OPALLO_BYTES,
            "PASS": h.hexdigest() == OPALLO_SHA and n == OPALLO_BYTES}


def cmd_delete(arm: str) -> int:
    out = ARMS / arm
    build = json.loads((out / "BUILD.json").read_text(encoding="utf-8"))
    score = SCORES / f"{arm}.json"
    if not score.is_file():
        raise SystemExit(f"{arm}: no score file; refusing to delete unscored weights")
    if json.loads(score.read_text(encoding="utf-8"))["weight_sha256"] != build["weight_sha256"]:
        raise SystemExit(f"{arm}: the score file's weight hash differs from BUILD.json; refusing")
    (out / "model.safetensors").unlink()
    for a in ASSETS:
        (out / a).unlink(missing_ok=True)
    write_json(out / "WEIGHTS_DELETED.json", {"deleted_after_scoring": time.strftime("%Y-%m-%d %H:%M:%S"),
                                              "weight_sha256": build["weight_sha256"],
                                              "reason": "disk: one rebuilt arm on disk at a time; rebuild with "
                                                        "build_o_arms.py and check this hash"})
    print(f"deleted {arm} weights ({build['weight_sha256'][:12]})", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild").add_argument("arm", choices=sorted(REBUILT))
    sub.add_parser("seed").add_argument("seed", type=int, choices=(23, 29, 41))
    sub.add_parser("seed-control")
    sub.add_parser("delete").add_argument("arm")
    a = ap.parse_args()
    if a.cmd == "rebuild":
        return cmd_rebuild(a.arm)
    if a.cmd == "seed":
        return cmd_seed(a.seed)
    if a.cmd == "seed-control":
        r = seed_control()
        print(json.dumps(r), flush=True)
        return 0 if r["PASS"] else 5
    return cmd_delete(a.arm)


if __name__ == "__main__":
    raise SystemExit(main())
