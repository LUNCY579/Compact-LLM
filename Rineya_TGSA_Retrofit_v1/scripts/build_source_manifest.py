#!/usr/bin/env python3
"""Hash every real source artifact and verify against the publisher's manifest.

Fails closed: a file whose hash disagrees with the published FILES.json is a
corrupt or substituted artifact, not a warning.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

MANIFEST_SCHEMA = "rineya.tgsa-source-checkpoint-manifest.v1"
REQUIRED = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--published-files", type=Path, help="the repo's FILES.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    published = {}
    if args.published_files and args.published_files.is_file():
        published = json.loads(args.published_files.read_text(encoding="utf-8")).get("files", {})

    artifacts: dict[str, dict] = {}
    errors: list[str] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(model_dir).as_posix()
        digest = sha256_file(path)
        entry = {"bytes": path.stat().st_size, "sha256": digest}
        declared = published.get(name)
        if declared is not None:
            entry["published_sha256"] = declared["sha256"]
            entry["matches_published"] = declared["sha256"] == digest
            if not entry["matches_published"]:
                errors.append(f"{name}: sha256 {digest} != published {declared['sha256']}")
            if declared["bytes"] != entry["bytes"]:
                errors.append(f"{name}: {entry['bytes']} bytes != published {declared['bytes']}")
        artifacts[name] = entry

    missing = [name for name in REQUIRED if name not in artifacts]
    if missing:
        errors.append(f"missing required artifacts: {missing}")

    config = {}
    config_path = model_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    # Shard-level hashes of the weight file's tensors, so a later conversion can
    # prove which exact tensors it consumed.
    tensor_hashes: dict[str, str] = {}
    weights = model_dir / "model.safetensors"
    if weights.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(weights), framework="pt") as handle:
                for key in sorted(handle.keys()):
                    tensor = handle.get_tensor(key)
                    digest = hashlib.sha256()
                    digest.update(key.encode("utf-8"))
                    digest.update(str(tensor.dtype).encode("ascii"))
                    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
                    digest.update(tensor.contiguous().view(-1).view(torch_uint8(tensor)).numpy().tobytes())
                    tensor_hashes[key] = digest.hexdigest()
        except Exception as exc:  # pragma: no cover
            errors.append(f"could not hash tensors: {exc.__class__.__name__}: {exc}")

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "model_dir": str(model_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "verified_against_published_manifest": bool(published),
        "published_mismatches": errors,
        "passed": not errors,
        "architecture": {
            k: config.get(k)
            for k in ("architectures", "hidden_size", "num_hidden_layers",
                      "num_attention_heads", "num_key_value_heads", "head_dim",
                      "vocab_size", "dtype", "rope_parameters", "tie_word_embeddings")
        },
        "tensor_count": len(tensor_hashes),
        "tensor_sha256": tensor_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"artifacts hashed : {len(artifacts)}")
    print(f"tensors hashed   : {len(tensor_hashes)}")
    print(f"published check  : {'verified' if published else 'no published manifest supplied'}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"passed           : {manifest['passed']}")
    return 0 if manifest["passed"] else 1


def torch_uint8(tensor):
    import torch
    return torch.uint8


if __name__ == "__main__":
    raise SystemExit(main())
