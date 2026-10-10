"""Byte identities and write-once artifacts; no model-framework imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank line {number}")
            yield strict_json(line)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")

def source_identity():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_digest(p)
            for p in sorted((ROOT / "scripts").glob("*.py"))}

def child_path(path):
    """Refuse mutations outside this new workspace, including symlink escapes."""
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    return path
