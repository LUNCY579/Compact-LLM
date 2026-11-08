"""Descriptive n=47 retention panel added by TRISKILL Amendment 1.

Run only after Stage B. BASE must reproduce 47/47 before TT or TTC is measured.
The exact recorded retention_probe.py is used with an HF-layout sibling and
receipt for each arm. No retention number changes C1-C4 or the headline.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stage_b_guard import selected

HERE = Path(__file__).resolve().parent
SB = HERE / "stage_b"
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
SUPA = CORES / "Rineya-Supa-1.7.0-devfreeze"
TT = CORES / "Rineya-Titan-Transition-1.0.0-devfreeze"
SUPA_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TT_SHA = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")
R_REPO = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification")
R_PROBE = R_REPO / "experiments/rab001_exploration/retention_probe.py"
R_EVID = R_REPO / "experiments/architecture_benchmark_001/evidence/retention-design-v1"
R_PY = Path(r"C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe")
PROBE_SHA = "5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266"
IMPL = "a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e"
RECEIPT_CONTENT = "ec3fca1e8b1eb418a6b848dadfea06235cf7c4317ddf6930061c84147afa9a8f"
sys.path.insert(0, str(HERE.parent / "fresh_dual_skill_instrument_v1"))
from gpu_lease import lease  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sibling(name: str, arm: Path, weight_sha: str) -> Path:
    target = SB / "retention_hf" / name
    if target.exists():
        raise SystemExit(f"retention sibling exists; inspect before reuse: {target}")
    target.mkdir(parents=True)
    os.link(arm / "model.safetensors", target / "model.safetensors")
    for sidecar in SIDECARS:
        shutil.copyfile(SUPA / sidecar, target / sidecar)
    receipt = {"record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
               "source_weight_sha256": SUPA_SHA, "out_weight_sha256": weight_sha,
               "checkpoint_manifest_sha256": None,
               "construction": "hardlink to the exact registered arm weight; source-identical HF sidecars",
               "arm": name, "quality_claim": False, "release_permitted": False}
    (target / "MATERIALIZATION.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    return target


def verify(path: Path, weight_sha: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    reference = Path(result["reference_receipt"])
    checks = {"probe_sha": result.get("probe_sha256") == PROBE_SHA,
              "reference_digest": result.get("reference_digest") == IMPL,
              "reference_content": sha256(reference) == RECEIPT_CONTENT,
              "hf_target": str(result.get("target", "")).startswith("hf:"),
              "weight_sha": result.get("hf_candidate_identity", {}).get("weight_sha256") == weight_sha,
              "device": result.get("device") == "cuda",
              "frozen_n": result.get("frozen_reference_sizes", {}).get("permissive") == 47,
              "source_correct_n": result.get("by_bare_policy", {}).get("permissive", {}).get("source_correct_total") == 47}
    if not all(checks.values()):
        raise SystemExit(f"retention comparability tuple failed: {checks}")
    lost = result["by_bare_policy"]["permissive"]["lost_total"]
    return {"retained": 47 - lost, "reference": 47, "lost": lost,
            "tuple": checks, "artifact_sha256": sha256(path),
            "lost_by_class": {k: v.get("lost") for k, v in result["by_bare_policy"]["permissive"]["per_class"].items() if v.get("lost")}}


def probe(name: str, arm: Path, weight_sha: str) -> dict:
    kept = SB / "retention" / f"{name}.json"
    if kept.exists():
        return verify(kept, weight_sha)
    label = f"triskill001_{name.lower()}_{weight_sha[:8]}"
    produced = R_EVID / f"RETENTION_PROBE_{label}.json"
    if produced.exists():
        raise SystemExit(f"external probe artifact exists without local record: {produced}")
    hf = sibling(name, arm, weight_sha)
    cmd = [str(R_PY), str(R_PROBE), label, "hf:" + hf.as_posix(), "cuda"]
    log = SB / "retention" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(" ".join(cmd) + "\n")
        stream.flush()
        rc = subprocess.call(cmd, cwd=str(R_REPO), stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"exit rc={rc}\n")
    if rc != 0 or not produced.exists():
        raise SystemExit(f"retention probe {name} failed rc={rc}; inspect {log}")
    shutil.copyfile(produced, kept)
    info = verify(kept, weight_sha)
    (hf / "model.safetensors").unlink()  # hardlink only; the registered arm remains
    print(f"  retention {name}: {info['retained']}/47, tuple OK", flush=True)
    return info


def build_composite(name: str) -> tuple[Path, str]:
    import build_arm
    build_arm.ARMS_DIR = SB / "retention_build"
    arm = build_arm.ARMS_DIR / name
    if not arm.exists():
        build_arm.build(name, None)
    if not (arm / "model.safetensors").exists():
        raise SystemExit(f"retention composite weights missing: {arm}")
    receipt = json.loads((arm / "BUILD.json").read_text(encoding="utf-8"))
    return arm, receipt["weight_sha256"]


def main() -> int:
    composite, _, beta = selected()
    if not (HERE / "STAGE_B_RESULT.json").exists():
        raise SystemExit("retention deferred until Stage B scoring and analysis complete")
    output = HERE / "RETENTION_RESULT.json"
    if output.exists():
        raise SystemExit("RETENTION_RESULT.json exists (create-only)")
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage("C:/").free / 2**30
    if ram < 5.0 or disk < 6.0:
        raise SystemExit(f"retention not started: RAM {ram:.2f} GiB, disk {disk:.2f} GiB")
    result = {"experiment": "TRISKILL-001", "endpoint": "references_retained (47)",
              "selected_beta": beta, "descriptive_only": True,
              "status": "NOT_MEASURED", "arms": {}, "quality_claim": False, "release_permitted": False}
    with lease(label="triskill001:retention", job="descriptive 47-row panel on BASE, TT, TTC"):
        result["arms"]["BASE"] = probe("BASE", SUPA, SUPA_SHA)
        if result["arms"]["BASE"]["retained"] == 47:
            result["arms"]["TT"] = probe("TT", TT, TT_SHA)
            arm, weight_sha = build_composite(composite)
            result["arms"][composite] = probe(composite, arm, weight_sha)
            if sha256(arm / "model.safetensors") != weight_sha:
                raise SystemExit("retention composite hash mismatch before deletion")
            (arm / "model.safetensors").unlink()
            (arm / "WEIGHTS_DELETED.json").write_text(json.dumps({"record": "WEIGHTS_DELETED",
                "weight_sha256": weight_sha, "deleted": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rebuild": f"build_arm.build('{composite}', None) with ARMS_DIR=stage_b/retention_build"},
                indent=1) + "\n", encoding="utf-8", newline="\n")
            result["status"] = "MEASURED"
        else:
            result["status"] = "NOT_MEASURED_BASE_CANARY_FAILED"
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "arms": result["arms"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
