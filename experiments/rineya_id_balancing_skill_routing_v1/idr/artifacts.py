"""Read-only input custody, write-once outputs and split-overlap checks."""
import hashlib
import json
from pathlib import Path
from .control import digest


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant, object_pairs_hook=unique_keys)


def write_json(path, data):
    p = Path(path)
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def tree_manifest(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("symlinks are not admitted to artifact manifests")
        if p.is_file():
            files[p.relative_to(root).as_posix()] = sha256_file(p)
    if not files:
        raise ValueError("empty artifact")
    return {"files": files, "tree_sha256": digest(files)}


def verify_tree(root, expected):
    actual = tree_manifest(root)
    if actual != expected:
        raise ValueError("artifact tree differs from registered manifest")
    return actual["tree_sha256"]


def freeze(paths, output):
    entries = {str(Path(p).resolve(strict=True)): sha256_file(p) for p in paths}
    if not entries:
        raise ValueError("cannot freeze empty inputs")
    write_json(output, {"schema": 1, "files": entries, "sha256": digest(entries)})


def verify_freeze(path):
    manifest = load_json(path)
    if manifest["schema"] != 1 or digest(manifest["files"]) != manifest["sha256"]:
        raise ValueError("invalid freeze manifest")
    for p, expected in manifest["files"].items():
        if sha256_file(p) != expected:
            raise ValueError(f"frozen input changed: {p}")
    return manifest


def validate_rows(rows):
    """Parent AND canonical input must be disjoint across all named splits.

    Rows: id, parent_id, family, split, payload; no oracle fields in payload.
    Callers may store outputs separately, but never use them as router features.
    """
    ids, parents, payloads = set(), {}, {}
    for r in rows:
        if r["split"] not in {"train", "calibration", "dev", "confirm"}:
            raise ValueError("unknown split")
        if not r["id"] or not r["parent_id"] or not r["family"] or r["id"] in ids:
            raise ValueError("missing identity or duplicate example id")
        ids.add(r["id"])
        for table, key in ((parents, (r["family"], r["parent_id"])),
                           (payloads, digest(r["payload"]))):
            if key in table and table[key] != r["split"]:
                raise ValueError("parent/payload leakage across splits")
            table[key] = r["split"]
    return {"rows": len(ids), "parents": len(parents), "payloads": len(payloads)}
