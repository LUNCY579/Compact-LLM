from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
PHASE1_TRAINER = (
    REPO_ROOT
    / "short-response-model-phase-0-1-2"
    / "scripts"
    / "phase_1_dense_sft"
    / "train_dense_sft.py"
)
CORPUS_CORE = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "scripts" / "corpus_core.py"
SEALED_SCORER = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "scripts" / "score_sealed.py"

STRATUM_BOUNDS = {
    "2k": (1843, 2252),
    "4k": (3768, 4423),
    "8k": (7536, 8847),
    "16k": (15073, 17694),
    "32k": (30146, 35389),
}
_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def proxy_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def normalize_stratum(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    aliases = {"2k": "2k", "4k": "4k", "8k": "8k", "16k": "16k", "32k": "32k"}
    return aliases.get(text)


def first_user_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def infer_stratum(row: dict[str, Any], lineage: dict[str, dict[str, Any]] | None = None) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("context_stratum", "stratum"):
        found = normalize_stratum(metadata.get(key))
        if found:
            return found
    row_id = row.get("id")
    if lineage and isinstance(row_id, str) and row_id in lineage:
        found = normalize_stratum(lineage[row_id].get("stratum"))
        if found:
            return found
    count = proxy_count(first_user_content(list(row.get("messages") or [])))
    for stratum, (low, high) in STRATUM_BOUNDS.items():
        if low <= count <= high:
            return stratum
    return None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def iter_shards(directory: Path) -> Iterator[dict[str, Any]]:
    shards = sorted(directory.glob("shard-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.jsonl files under {directory}")
    for shard in shards:
        yield from iter_jsonl(shard)


def load_lineage(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = row.get("sft_id") or row.get("id")
        if isinstance(key, str):
            mapping[key] = row
    return mapping


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _parse_scalar(text: str) -> Any:
    raw = text.strip()
    if raw in {"true", "True", "yes"}:
        return True
    if raw in {"false", "False", "no"}:
        return False
    if raw in {"null", "None", "~"}:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        return raw[1:-1]
    try:
        if raw.startswith("0") and len(raw) > 1 and not raw.startswith("0.") and "e" not in raw.lower():
            raise ValueError
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        mapping: dict[str, Any] = {}
        sequence: list[Any] | None = None
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"invalid YAML indent near {content!r}")
            if content.startswith("- "):
                if mapping:
                    raise ValueError("mixed mapping/list in YAML block")
                if sequence is None:
                    sequence = []
                item = content[2:].strip()
                index += 1
                if item and ":" in item and not item.startswith("{") and not item.startswith("["):
                    key, _, rest = item.partition(":")
                    nested = {key.strip(): _parse_scalar(rest)}
                    if index < len(lines) and lines[index][0] > current_indent:
                        child, index = parse_block(index, lines[index][0])
                        if isinstance(child, dict):
                            nested.update(child)
                    sequence.append(nested)
                elif index < len(lines) and lines[index][0] > current_indent:
                    child, index = parse_block(index, lines[index][0])
                    sequence.append(child)
                else:
                    sequence.append(_parse_scalar(item) if item else None)
                continue
            key, sep, rest = content.partition(":")
            if not sep:
                raise ValueError(f"invalid YAML line: {content!r}")
            key = key.strip()
            rest = rest.strip()
            index += 1
            if rest:
                mapping[key] = _parse_scalar(rest)
            elif index < len(lines) and lines[index][0] > current_indent:
                child, index = parse_block(index, lines[index][0])
                mapping[key] = child
            else:
                mapping[key] = None
        if sequence is not None:
            return sequence, index
        return mapping, index

    parsed, _ = parse_block(0, lines[0][0] if lines else 0)
    if not isinstance(parsed, dict):
        raise ValueError("YAML root must be a mapping")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def stage_paths(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = resolve_repo_path(config["paths"]["output_dir"])
    mix_dir = output_dir / "mix"
    eval_dir = output_dir / "eval"
    return {
        "output_dir": output_dir,
        "mix_dir": mix_dir,
        "eval_dir": eval_dir,
        "train_mix": mix_dir / "train_stage_a.jsonl",
        "val_mix": mix_dir / "val_stage_a.jsonl",
        "val_trainer": mix_dir / "val_stage_a_trainer.jsonl",
        "test_mix": mix_dir / "test_stage_a.jsonl",
        "mix_manifest": mix_dir / "mix_manifest.json",
        "adapter_dir": output_dir / "adapter",
        "training_manifest": output_dir / "training_manifest.json",
    }


def list_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]

    def step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return sorted(checkpoints, key=step)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def validate_adapter_directory(path: Path) -> None:
    config = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not config.exists():
        raise RuntimeError(f"Adapter save is incomplete; missing {config}")
    if not any(candidate.exists() and candidate.stat().st_size > 0 for candidate in weights):
        raise RuntimeError(
            f"Adapter save is incomplete; no non-empty adapter_model.safetensors/bin exists under {path}"
        )


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        counter[str(metadata.get(key, "unknown"))] += 1
    return dict(sorted(counter.items()))


def python_executable() -> str:
    venv = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv)
    return shutil.which("python") or "python"


def trim_text(text: str) -> str:
    return str(text).strip()


def exact_match(observed: str, expected: str, mode: str) -> bool:
    if mode == "strict":
        return observed == expected
    if mode == "trim":
        return trim_text(observed) == trim_text(expected)
    if mode == "casefold":
        return " ".join(trim_text(observed).split()).casefold() == " ".join(trim_text(expected).split()).casefold()
    raise ValueError(mode)
