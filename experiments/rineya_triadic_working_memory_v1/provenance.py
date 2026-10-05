"""Local-only checkpoint locks, source identities, and explicit resource guards."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash() -> str:
    root = Path(__file__).parent
    return canonical_hash({p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.py"))})


def environment():
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return dict(python=platform.python_version(), torch=torch.__version__,
                transformers=transformers_version, cuda=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                source_sha256=source_hash())


def _safe_root(path):
    path = Path(path).expanduser().resolve()
    if any("sealed" in p.lower() or "test2" in p.lower() for p in path.parts):
        raise ValueError("sealed/TEST2 paths are excluded")
    if not path.is_dir():
        raise FileNotFoundError(f"local checkpoint directory not found: {path}")
    return path


def _safe_file(root, name):
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("checkpoint manifest paths must remain inside the checkpoint")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or escaped checkpoint file: {name}")
    return path


def create_checkpoint_lock(checkpoint, checkpoint_id: str):
    root = _safe_root(checkpoint)
    if not checkpoint_id.strip():
        raise ValueError("an explicit checkpoint identity is required")
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for name in required:
        _safe_file(root, name)
    if (root / "model.safetensors.index.json").exists():
        index = json.loads(_safe_file(root, "model.safetensors.index.json").read_text())
        weights = sorted(set(index["weight_map"].values()))
        required.append("model.safetensors.index.json")
    elif (root / "model.safetensors").exists():
        weights = ["model.safetensors"]
    else:
        raise FileNotFoundError("actual local safetensors weights are required")
    if not weights or any(not w.endswith(".safetensors") for w in weights):
        raise ValueError("only safetensors checkpoints are supported")
    for name in weights:
        path = _safe_file(root, name)
        with path.open("rb") as f:
            if f.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError("weights are Git LFS pointers, not resident weights")
    optional = [n for n in ("generation_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt") if (root / n).exists()]
    result = dict(schema="twm-checkpoint-lock-v1", checkpoint_id=checkpoint_id,
                  files={n: sha256_file(_safe_file(root, n)) for n in sorted(required+weights+optional)})
    result["lock_sha256"] = canonical_hash(result)
    return result


def verify_checkpoint_lock(checkpoint, lock):
    root = _safe_root(checkpoint)
    if lock.get("schema") != "twm-checkpoint-lock-v1":
        raise ValueError("unsupported checkpoint lock schema")
    expected = canonical_hash({k: v for k, v in lock.items() if k != "lock_sha256"})
    if lock.get("lock_sha256") != expected:
        raise ValueError("checkpoint lock was edited or corrupted")
    actual = create_checkpoint_lock(root, lock["checkpoint_id"])
    if actual != lock:
        raise ValueError("checkpoint/tokenizer files changed after locking")
    return root


def source_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def check_device(device: str, min_free_gib: float = 6.5):
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("only CPU or CUDA is qualified")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no native GPU run performed")
        free, total = torch.cuda.mem_get_info(selected)
        if free < min_free_gib * 1024**3:
            raise MemoryError(f"free GPU memory {free/1024**3:.2f} GiB is below {min_free_gib:.2f} GiB guard")
        return dict(free_bytes=free, total_bytes=total)
    return dict(free_bytes=None, total_bytes=None)


def check_peak(device: str, budget_gib: float):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        if peak > budget_gib * 1024**3:
            raise MemoryError(f"native allocated peak exceeded {budget_gib} GiB")
        return peak
    return None


def save_json(path, data, exclusive: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
