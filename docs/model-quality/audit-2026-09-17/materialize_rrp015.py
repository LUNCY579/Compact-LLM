"""Create a source-bound legacy evaluation export; preserve all checkpoint evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path.cwd().resolve()))
from experiments.model_quality_gpu_v1.integrity import file_hash, unseal, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--research-format", action="store_true")
    args = p.parse_args()
    manifest = args.checkpoint / "manifest.json"
    body = unseal(json.loads(manifest.read_text(encoding="utf-8")))
    if body.get("status") != "COMPLETE" or len(body.get("entries", [])) != 220:
        raise ValueError("expected complete 220-tensor Supa tail checkpoint")
    assets = body["binding"]["source"]["assets"]
    if assets["model.safetensors"] != "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b":
        raise ValueError("checkpoint was not trained from the registered Supa source")
    for name, expected in assets.items():
        if file_hash(args.source / name) != expected:
            raise ValueError(f"checkpoint source binding differs: {name}")
    # Preserve the same floor even though no optimizer or CUDA model is allocated.
    # Same tensor names, shapes and stored BF16 dtype as the bound source. Use the
    # measured write inventory plus a 2MiB header/receipt allowance, not an arbitrary
    # 64MiB estimate. The 14GiB floor itself is unchanged.
    sidecar_bytes = sum((args.source / n).stat().st_size for n in assets if n != "model.safetensors")
    expected_bytes = ((args.source / "model.safetensors").stat().st_size
                      * (2 if args.research_format else 1)
                      + (0 if args.research_format else sidecar_bytes) + 2 * 2**20)
    if shutil.disk_usage(args.out.parent).free - expected_bytes < 14 * 2**30:
        raise ValueError("export would cross the unchanged 14 GiB free-space floor")
    import psutil
    if psutil.virtual_memory().available < 5 * 2**30:
        raise ValueError("insufficient host RAM for bounded CPU materialization")
    from experiments.rab001_exploration.materialize_checkpoint import materialize
    receipt = materialize(args.source, args.checkpoint, args.out,
                          research=args.research_format)
    if receipt["source_weight_sha256"] != assets["model.safetensors"]:
        raise ValueError("source changed during materialization")
    written_bytes = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    if written_bytes > expected_bytes or shutil.disk_usage(args.out.parent).free < 14 * 2**30:
        raise ValueError("export exceeded its measured byte reservation or free-space floor")
    from safetensors import safe_open
    with safe_open(str(args.out / "model.safetensors"), framework="pt", device="cpu") as f:
        dtypes = sorted({str(f.get_slice(name).get_dtype()) for name in f.keys()})
    expected_dtype = "F32" if args.research_format else "BF16"
    if dtypes != [expected_dtype]:
        raise ValueError(f"unexpected export dtypes: {dtypes}")
    write_new(args.out / "CUSTODY.json", {
        "source_binding_verified_against_checkpoint": True,
        "reserved_export_bytes": expected_bytes,
        "written_bytes_before_custody_receipt": written_bytes,
        "source_weight_sha256": assets["model.safetensors"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_manifest_sha256": file_hash(manifest),
        "materialization_sha256": file_hash(args.out / "MATERIALIZATION.json"),
        "output_weight_sha256": receipt["out_weight_sha256"],
        "stored_dtype_inventory": dtypes,
        "stored_dtype": "float32" if args.research_format else "bfloat16",
        "evaluation_dtype": "float32",
        "precision_path": "FP32 trained tensors rounded to source BF16 then FP32 DenseReference",
        "native_bf16_serving_qualified": False,
        "model_constructed_on_cpu_for_format_conversion": bool(args.research_format),
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "legacy_receipt_correction": (
            "Research-format export constructs a CPU DenseReference; the legacy receipt's "
            "generic no-model-constructed sentence does not apply to this layout. No "
            "model forward or optimizer call is made." if args.research_format else None),
        "quality_claim": False,
    })
    print(json.dumps({"output": str(args.out), "dtypes": dtypes,
                      "weight_sha256": receipt["out_weight_sha256"]}))


if __name__ == "__main__":
    main()
