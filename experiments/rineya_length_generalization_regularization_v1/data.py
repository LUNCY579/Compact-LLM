"""Token-exact, no-truncation cases and disjointness checks.

Generated banks are synthetic diagnostics, NOT RULER/HELMET reproductions.
External verified banks may use the same schema; quarantine eligibility must
be established by the repository's existing data-governance process.
"""
import hashlib
import json
from pathlib import Path
import random

SPLITS = {"train", "dev", "confirmation"}
ROLES = {"long_context", "acquisition", "retention", "abstention"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_cases(rows, *, split=None, vocabulary=None):
    if not rows:
        raise ValueError("Empty bank")
    seen = set()
    for row in rows:
        required = {"id", "parent_id", "split", "task", "role", "prompt_ids", "target_ids",
                    "answer", "answerable", "synthetic", "fixture", "length", "depth"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("Incomplete case schema")
        if any(not isinstance(row[k], str) or not row[k] for k in ("id", "parent_id", "task", "answer")):
            raise ValueError("Missing string identity/answer")
        if row["id"] in seen:
            raise ValueError("Duplicate case id")
        seen.add(row["id"])
        if row["split"] not in SPLITS or (split is not None and row["split"] != split):
            raise ValueError("Wrong or unrecognized split")
        if row["role"] not in ROLES:
            raise ValueError("Unknown endpoint")
        if any(type(row[k]) is not bool for k in ("answerable", "synthetic", "fixture")):
            raise ValueError("Case flags must be boolean")
        for key in ("prompt_ids", "target_ids"):
            values = row[key]
            if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Token ids must be nonempty nonnegative integer lists")
            if vocabulary is not None and max(values) >= vocabulary:
                raise ValueError("Token id outside model vocabulary")
        if type(row["length"]) is not int or row["length"] != len(row["prompt_ids"]):
            raise ValueError("Recorded length differs from actual token length")
        if type(row["depth"]) not in (int, float) or not 0 <= row["depth"] <= 1:
            raise ValueError("Invalid evidence depth")
        if not row["answerable"] and (row["answer"] != "UNKNOWN" or row["role"] != "abstention"):
            raise ValueError("Unanswerable cases require UNKNOWN and abstention endpoint")
    return rows


def read_bank(path, *, split=None):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Bank is symlinked or exceeds the bounded reader")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return validate_cases(rows, split=split)


def write_bank(path, rows):
    validate_cases(rows)
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def audit_disjoint(*banks):
    """Detect cross-split parents AND prompt tokens, not just family names."""
    parent_splits, prompt_splits = {}, {}
    for rows in banks:
        validate_cases(rows)
        for row in rows:
            for key, index in ((row["parent_id"], parent_splits), (digest(row["prompt_ids"]), prompt_splits)):
                previous = index.setdefault(key, row["split"])
                if previous != row["split"]:
                    raise ValueError("Cross-split parent/prompt contamination")
    return {"parents": len(parent_splits), "prompts": len(prompt_splits)}


def make_bank(tokenizer, *, split, seed, parents=4, lengths=(1024,), depths=(0.1, 0.5, 0.9), fixture=False):
    """Exact prompt lengths; same semantic parent across lengths and positions.

    Inputs are constructed at token level; logical instruction/needle/answer
    is never truncated. This intentionally uses plain-completion-v1, not a
    hidden chat-template transformation. Train lengths must leave target room.
    """
    if split not in SPLITS or type(parents) is not int or parents <= 0:
        raise ValueError("Invalid generation request")
    if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("Invalid length grid")
    if not depths or len(set(depths)) != len(depths) or any(not 0 <= d <= 1 for d in depths):
        raise ValueError("Invalid depth grid")
    rng = random.Random(digest([split, seed]))
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    filler = encode(" Irrelevant record. Nothing requested here.")
    if not filler:
        raise ValueError("Tokenizer produced empty filler")
    rows = []
    for task in ("retrieve", "multi_retrieve", "state_update", "absent"):
        for _ in range(parents):
            key, other = rng.sample(range(100000, 999999), 2)
            first, second = rng.sample(range(1000, 9999), 2)
            semantic = [task, key, other, first, second]
            parent = digest(semantic)
            if task == "state_update":
                needle = f" Account K{key}: initial {first}; add {second}."
                question = f"Final balance of K{key}?"
                answer = str(first + second)
            else:
                needle = f" K{key}={first}. K{other}={second}."
                question = f"Value of K{key}?"
                answer = str(first)
                if task == "multi_retrieve":
                    question = f"Values of K{key} and K{other}, separated by one space?"
                    answer = f"{first} {second}"
                if task == "absent":
                    question = f"Value of M{key}?"
                    answer = "UNKNOWN"
            prefix = encode("Use the records. Give only the answer; if missing, give UNKNOWN.\nRecords:\n")
            evidence = encode(needle)
            suffix = encode("\nQuestion: " + question + "\nAnswer:")
            target = encode(answer)
            if getattr(tokenizer, "eos_token_id", None) is not None:
                target.append(tokenizer.eos_token_id)
            for length in lengths:
                padding = length - len(prefix) - len(evidence) - len(suffix)
                if padding < 0:
                    raise ValueError("Requested context cannot fit the untruncated problem")
                noise = (filler * ((padding + len(filler) - 1) // len(filler)))[:padding]
                for depth in depths:
                    index = int(padding * depth)
                    prompt = prefix + noise[:index] + evidence + noise[index:] + suffix
                    rows.append({"id": digest([parent, length, depth]), "parent_id": parent, "split": split,
                                 "task": task, "role": "abstention" if task == "absent" else "long_context",
                                 "prompt_ids": prompt, "target_ids": target, "answer": answer,
                                 "answerable": task != "absent", "synthetic": True, "fixture": fixture,
                                 "length": len(prompt), "depth": depth})
    return validate_cases(rows)
