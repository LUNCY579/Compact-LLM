"""Rebuild the open calibration selection with stdlib and pinned source oracle only.

No tokenizer/model imports, no native execution, and no writes outside the requested
create-only output. Example from the experiment root:
  python build_calibration.py --spec evidence/CALIBRATION_BINDING.json \
      --output evidence/CALIBRATION_BINDING.v2.json --supersedes
The original specification freezes input hashes and the deterministic selection.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

SALT = "cross-scale-composable-001/calibration-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pinned_oracle(directory):
    # Avoid running an unneeded package __init__, and never emit caches in old worktrees.
    sys.dont_write_bytecode = True
    name = "_cross_scale_calibration_oracle"
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    for stem in ("common", "oracle"):
        spec = importlib.util.spec_from_file_location(name + "." + stem,
                                                     directory / (stem + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.replay_transition


def build(specification):
    require(specification["selection"]["salt"] == SALT, "wrong registered salt")
    inputs = specification["source_bindings"]
    paths = {Path(item["path"]).as_posix(): item["sha256"] for item in inputs}
    require(len(paths) == len(inputs), "duplicate input binding")
    for path, expected in paths.items():
        require(file_sha(path) == expected, "input bytes changed: " + path)
    source = Path(specification["selection"]["source_path"])
    train = Path(specification["prompt_contract"]["system_rules_source"])
    require(source.as_posix() in paths and train.as_posix() in paths, "unbound data")
    oracle_dir = source.parent.parent / "forge_rl"
    require(all((oracle_dir / (stem + ".py")).as_posix() in paths
                for stem in ("common", "oracle")), "unbound oracle code")
    replay = load_pinned_oracle(oracle_dir)
    evaluation = [Path(item["path"]) for item in inputs
                  if Path(item["path"]).name in ("SELECT_VIEWS.jsonl", "CANDIDATE_VIEWS.jsonl")]
    require(len(evaluation) == 2, "both evaluation parent inventories required")
    training = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    train_parents = {row["parent_id"] for row in training}
    require(len(train_parents) == 384, "donor training-parent count changed")
    rules = {}
    for row in training:
        require(row["source_split"] == "train", "non-open training row")
        rule = row["prompt"][0]["content"]
        require(rules.setdefault(row["family"], rule) == rule, "rule inconsistency")
    eval_parents = set()
    for path in evaluation:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    # Only metadata is used; no evaluation prompt/reference is inspected.
                    eval_parents.add(json.loads(line)["parent_id"])
    require(len(eval_parents) == 768, "evaluation-parent count changed")
    require(not train_parents & eval_parents, "donor/evaluation overlap")
    exclude = eval_parents | train_parents
    pool = {}
    total = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            require(row["source_split"] == "train" and row["group_id"] == row["parent_id"],
                    "non-TRAIN or broken parent identity")
            expected = replay(row["family"], row["before"], row["event"])
            require((expected == row["proposed_after"]) == row["valid"], "oracle mismatch")
            if not row["valid"] or row["parent_id"] in exclude:
                continue
            previous = pool.get(row["parent_id"])
            key = lambda item: (digest([SALT, "step", item["id"]]), item["id"])
            if previous is None or key(row) < key(previous):
                pool[row["parent_id"]] = row
    chosen = sorted(pool.values(), key=lambda row:
                    (digest([SALT, "parent", row["parent_id"]]), row["parent_id"]))[:32]
    require(len(chosen) == 32, "insufficient eligible parents")
    rows = []
    for index, row in enumerate(chosen):
        prompt = [{"role": "system", "content": rules[row["family"]]},
                  {"role": "user", "content": json.dumps(
                      {"before": row["before"], "event": row["event"]},
                      sort_keys=True, separators=(",", ":"))}]
        rows.append({"id": row["id"], "parent_id": row["parent_id"],
                     "split": "fit" if index < 24 else "map_check", "open": True,
                     "prompt_sha256": digest(prompt)})
    require(not {row["parent_id"] for row in rows} & exclude, "selection overlaps exclusion")
    require(rows == specification["rows"], "reproduced row identities/prompt digests differ")
    require(digest(rows) == specification["calibration_sha256"], "row digest mismatch")
    require(digest(rules) == specification["prompt_contract"]["system_rules_sha256"],
            "system-rule binding differs")
    require(file_sha(source) == specification["dataset_sha256"], "dataset digest mismatch")
    require(total == specification["selection"]["source_label_count_replayed"], "label count differs")
    require(len(pool) == specification["selection"]["eligible_parent_count"], "pool count differs")
    require(dict(collections.Counter(row["family"] for row in chosen)) ==
            specification["selection"]["family_counts"], "family counts differ")
    require(digest(sorted(exclude)) == specification["selection"]["excluded_parent_ids_sha256"],
            "exclusion identity differs")
    # Recheck source bytes after parsing; changed files never yield a valid receipt.
    for path, expected in paths.items():
        require(file_sha(path) == expected, "input changed during selection: " + path)
    output = dict(specification)
    output["rows"] = rows
    output["role"] = "open_calibration"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "create-only output already exists")
    raw = args.spec.read_bytes()
    result = build(json.loads(raw))
    if args.supersedes:
        result["supersedes_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"path": args.output.as_posix(), "sha256": file_sha(args.output),
                      "calibration_sha256": result["calibration_sha256"],
                      "native_model_calls": 0, "native_tokenizer_calls": 0}))


if __name__ == "__main__":
    main()
