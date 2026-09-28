"""Read-only adapter for the existing MAXRL-COMP-002 canonical/atomic manifests.

No generator, oracle import, bridge training, or evaluation-answer access.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

DATA_COMMIT = "a97df24c2c7dd062985942ead8c3375bf20c66b9"
DATA_MANIFEST = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


@dataclass(frozen=True)
class TrainingRow:
    row_id: str
    parent_id: str
    skill: str
    role: str
    manifest_sha256: str
    prompt_json: str
    verifier_json: str

    @property
    def messages(self):
        # Only these messages may be sent to the model, never the verifier specification.
        return json.loads(self.prompt_json)


def validate_training_rows(rows, skill, manifest_sha256):
    if not rows or any(not isinstance(r, TrainingRow) for r in rows) or len({r.row_id for r in rows}) != len(rows):
        raise ValueError("empty/duplicate training rows")
    for r in rows:
        if not isinstance(r, TrainingRow) or r.role != "train" or r.skill != skill or r.manifest_sha256 != manifest_sha256:
            raise ValueError("only bound atomic TRAIN rows are allowed")
        if not r.row_id or not r.parent_id:
            raise ValueError("explicit row and parent IDs required")
        messages = r.messages
        if not isinstance(messages, list) or not messages or any(set(m) != {"role", "content"} or m["role"] not in ("system", "user") or not isinstance(m["content"], str) for m in messages):
            raise ValueError("policy input must contain only system/user messages")


def load_atomic(root, skill, expected_manifest=DATA_MANIFEST):
    root = Path(root).resolve()
    if skill not in ("R", "U", "S", "P") or file_hash(root / "manifest.json") != expected_manifest:
        raise ValueError("unbound dataset/skill")
    manifest = read_json(root / "manifest.json")
    if manifest.get("experiment") != "MAXRL-COMP-002":
        raise ValueError("wrong corpus")

    def checked(relative):
        path = root / relative
        receipt = manifest["files"][relative]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != receipt["bytes"] or file_hash(path) != receipt["sha256"]:
            raise ValueError(f"dataset file drift: {relative}")
        return path

    bank = read_json(checked(f"manifests/atomic_{skill}.json"))
    ids = bank["ids"]
    if bank["name"] != f"atomic_{skill}" or bank["rows"] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("atomic manifest inconsistency")
    selected = {}
    wanted = set(ids)
    with checked("canonical/train.jsonl").open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if row["id"] not in wanted:
                continue
            if row["id"] in selected or row["role"] != "train" or row["tag"] != "core" or row["kind"] != "composition" or row["skills"] != [skill]:
                raise ValueError("invalid atomic membership")
            if digest_json(row["prompt"]) != row["prompt_hash"]:
                raise ValueError("prompt hash mismatch")
            selected[row["id"]] = TrainingRow(row["id"], row["parent_id"], skill, "train", expected_manifest,
                                               json.dumps(row["prompt"]), json.dumps(row["verifier"]))
    if set(selected) != wanted:
        raise ValueError("missing atomic rows")
    result = tuple(selected[r] for r in ids)
    validate_training_rows(result, skill, expected_manifest)
    return result
