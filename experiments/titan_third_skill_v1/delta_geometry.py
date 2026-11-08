"""Read-only FP32 delta geometry for the third skill, streamed one tensor at a time.

The effective H update is TT - SPARSE = 0.25*P16(delta_H) up to FP32 rounding.
This measures vector norms and cross terms; it does not infer functional retention
or substitute for the Stage A/B model-output comparisons.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OTHER = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-titan-model-soup-mini-v1/experiments/titan_opallo_skill_vector_v1/arms")
BASE = OTHER / "BASE" / "model.safetensors"
SPARSE = OTHER / "SPARSE" / "model.safetensors"
TT = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Transition-1.0.0-devfreeze/model.safetensors")
C = HERE / "trained_C_fp32.safetensors"
OUT = HERE / "DELTA_C_GEOMETRY.json"
EXPECTED_SHA = {
    "BASE": "75b0a92cd28eb90d121d72aa7618c497a6ff1b4522b73bdf01cabf515c30ad93",
    "SPARSE": "6ae85f5bdab9437edc06b20716e6e7008d8aa402ec95a0a65ebc7bf8da1e94ed",
    "TT": "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567",
    "C": "e42afe6a89231188cf03879647267b2040d2a6fccc1ff1fc3a85448d28c9e447",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


class TensorFile:
    """Plain seek/read avoids growing a persistent memory map working set."""

    def __init__(self, path: Path):
        self.file = path.open("rb")
        n = struct.unpack("<Q", self.file.read(8))[0]
        header = json.loads(self.file.read(n))
        header.pop("__metadata__", None)
        self.header = header
        self.start = 8 + n

    def tensor(self, key: str) -> np.ndarray:
        info = self.header[key]
        if info["dtype"] != "F32":
            raise SystemExit(f"{key}: expected F32, got {info['dtype']}")
        first, last = info["data_offsets"]
        self.file.seek(self.start + first)
        raw = self.file.read(last - first)
        if len(raw) != last - first:
            raise SystemExit(f"{key}: short read")
        return np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).astype(np.float64)

    def close(self) -> None:
        self.file.close()


def kind(key: str) -> str:
    if ".self_attn." in key:
        return "attention"
    if ".mlp." in key:
        return "mlp"
    return "norm_or_other"


def cosine(dot: float, x2: float, y2: float) -> float | None:
    return dot / math.sqrt(x2 * y2) if x2 > 0 and y2 > 0 else None


def main() -> int:
    if OUT.exists():
        raise SystemExit("DELTA_C_GEOMETRY.json exists (create-only)")
    paths = {"BASE": BASE, "SPARSE": SPARSE, "TT": TT, "C": C}
    for label, path in paths.items():
        actual = sha256(path)
        if actual != EXPECTED_SHA[label]:
            raise SystemExit(f"{label} sha256 mismatch: {actual}")
    files = {label: TensorFile(path) for label, path in paths.items()}
    try:
        keys = sorted(files["C"].header)
        if len(keys) != 220 or any(not set(keys) <= set(f.header) for f in files.values()):
            raise SystemExit("C trained-key inventory mismatch")
        sums = defaultdict(float)
        by_kind = defaultdict(lambda: defaultdict(float))
        tensor_rows = []
        for key in keys:
            base = files["BASE"].tensor(key)
            sparse = files["SPARSE"].tensor(key)
            dc = files["C"].tensor(key) - base
            dt = sparse - base
            dh = files["TT"].tensor(key) - sparse
            cc = float(np.vdot(dc, dc)); tt = float(np.vdot(dt, dt)); hh = float(np.vdot(dh, dh))
            tc = float(np.vdot(dt, dc)); hc = float(np.vdot(dh, dc)); th = float(np.vdot(dt, dh))
            measures = {"cc": cc, "tt": tt, "hh": hh, "tc": tc, "hc": hc, "th": th}
            group = kind(key)
            for measure, value in measures.items():
                sums[measure] += value
                by_kind[group][measure] += value
            tensor_rows.append({"tensor": key, "module": group, "norm_C": math.sqrt(cc),
                                "cos_C_T": cosine(tc, cc, tt), "cos_C_H_effective": cosine(hc, cc, hh),
                                "norm_T": math.sqrt(tt), "norm_H_effective": math.sqrt(hh)})
            del base, sparse, dc, dt, dh
        norm_c = math.sqrt(sums["cc"])
        norm_t = math.sqrt(sums["tt"])
        norm_h = math.sqrt(sums["hh"])
        if abs(norm_c - 0.8575171663087725) > 2e-6:
            raise SystemExit(f"C norm canary failed: {norm_c}")
        if abs(norm_t - 0.6940550846962167) > 2e-6:
            raise SystemExit(f"T norm canary failed: {norm_t}")
        if abs(norm_h - 0.17747358888627796) > 2e-5:
            raise SystemExit(f"H effective norm canary failed: {norm_h}")
        parent2 = sums["tt"] + sums["hh"] + 2 * sums["th"]
        doses = {}
        for beta in (0.25, 0.50, 1.00):
            total2 = parent2 + beta**2 * sums["cc"] + 2 * beta * (sums["tc"] + sums["hc"])
            doses[str(beta)] = {"C_update_norm": beta * norm_c, "composite_delta_norm": math.sqrt(total2),
                                "parent_delta_norm": math.sqrt(parent2),
                                "premium_pct": 100 * (math.sqrt(total2 / parent2) - 1),
                                "cross_term_with_C": 2 * beta * (sums["tc"] + sums["hc"])}
        result = {"experiment": "TRISKILL-001", "analysis": "delta C FP32 vector geometry, read-only",
                  "source_sha256": EXPECTED_SHA, "trained_tensors": len(keys),
                  "norms": {"C": norm_c, "T_dSparse": norm_t, "H_effective_0p25": norm_h,
                            "TT_minus_BASE": math.sqrt(parent2)},
                  "cosines": {"C_T": cosine(sums["tc"], sums["cc"], sums["tt"]),
                              "C_H_effective": cosine(sums["hc"], sums["cc"], sums["hh"]),
                              "T_H_effective": cosine(sums["th"], sums["tt"], sums["hh"])},
                  "inner_products": dict(sums),
                  "by_module": {k: {"norm_C": math.sqrt(v["cc"]), "norm_T": math.sqrt(v["tt"]),
                                    "norm_H_effective": math.sqrt(v["hh"]),
                                    "cos_C_T": cosine(v["tc"], v["cc"], v["tt"]),
                                    "cos_C_H_effective": cosine(v["hc"], v["cc"], v["hh"])}
                                for k, v in by_kind.items()},
                  "doses": doses, "per_tensor": tensor_rows,
                  "interpretation_limit": "Vector geometry only; no activation covariance, functional skill or retention verdict."}
        OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: result[k] for k in ("norms", "cosines", "doses")}, indent=1))
    finally:
        for file in files.values():
            file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
