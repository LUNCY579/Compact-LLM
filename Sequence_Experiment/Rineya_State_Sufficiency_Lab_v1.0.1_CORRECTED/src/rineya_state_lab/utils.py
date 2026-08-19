from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    p = ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True, default=_json_default) + "\n",
    )


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = ensure_parent(path)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    with p.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: str | Path, *, tolerate_truncated_tail: bool = False) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                if tolerate_truncated_tail:
                    remainder = handle.read()
                    if not remainder:
                        break
                raise ValueError(f"Invalid JSON at {p}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {p}:{line_number}")
            yield value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_int_hash(text: str, *, bits: int = 64) -> int:
    value = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest(), "big")
    return value


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_casefold(text: Any) -> str:
    value = normalize_ws(text)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", "*").replace("÷", "/")
    return value.casefold()


def load_completed_ids(path: str | Path, key: str = "request_id") -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    completed: set[str] = set()
    for row in iter_jsonl(p, tolerate_truncated_tail=True):
        value = row.get(key)
        if value is not None:
            completed.add(str(value))
    return completed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict[str, Any]:
    packages = [
        "rineya-state-sufficiency-lab",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "PyYAML",
        "safetensors",
    ]
    manifest: dict[str, Any] = {
        "captured_at": utc_now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: package_version(name) for name in packages},
        "environment": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
                "TOKENIZERS_PARALLELISM",
                "RINEYA_BASE_MODEL",
            ]
            if os.environ.get(key) is not None
        },
    }
    try:
        import torch

        manifest["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        }
    except Exception as exc:
        manifest["torch_error"] = f"{type(exc).__name__}: {exc}"
    return manifest


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def deep_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def parse_scalar(text: str) -> Any:
    lowered = text.strip().casefold()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def human_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(str(x) for x in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return "\n".join([header, separator, *body])


def html_page(title: str, body_html: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{safe_title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1f2937}}
h1,h2,h3{{color:#163a63}} table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{border:1px solid #cbd5e1;padding:.45rem;text-align:left;vertical-align:top}} th{{background:#e8eef6}} code,pre{{background:#f3f4f6;border-radius:4px}} code{{padding:.1rem .25rem}} pre{{padding:1rem;overflow:auto}} .ok{{color:#166534;font-weight:700}} .bad{{color:#991b1b;font-weight:700}} .muted{{color:#64748b}}
</style>
</head>
<body>{body_html}</body>
</html>"""


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def copytree_atomic(source: str | Path, destination: str | Path) -> None:
    src = Path(source)
    dst = Path(destination)
    tmp = dst.with_name(dst.name + ".tmp-copy")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
