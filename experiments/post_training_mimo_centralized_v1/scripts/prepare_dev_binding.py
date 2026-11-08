"""Publish the fixed DEV prompt/evaluator binding; no native run or enrollment."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import time
from . import dev_inputs as inputs
from . import dev_scorer as scorer
from . import enrollment


CONFIG = "configs/DEV_BINDING_V1.json"
AMENDMENT = "AMENDMENT_007.md"
UNQUALIFIED = ("native_qualified", "native_launch_authorized", "native_execution",
    "training_eligibility", "training_authorized", "grading_qualified", "language_scoring",
    "reference_labels_qualified", "source_correct_membership", "e1_launch_ready")


def read(path):
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    inputs.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                   (after.st_size, after.st_mtime_ns, after.st_ino), "file changed while reading")
    return raw


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def clean(root, path):
    relative = path.relative_to(root).as_posix()
    subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
                   check=True, capture_output=True)
    inputs.require(not git(root, "status", "--porcelain", "--", relative).strip(),
                   "critical file uncommitted: " + relative)


def critical_paths(package):
    return [Path(__file__).resolve(), Path(inputs.__file__).resolve(), Path(scorer.__file__).resolve(),
            Path(enrollment.__file__).resolve(), Path(scorer.bridge.__file__).resolve(),
            Path(scorer.retention.__file__).resolve(), Path(scorer.long_context.__file__).resolve(),
            package / CONFIG, package / AMENDMENT]


def prepare(config_path, output):
    started = time.perf_counter()
    package = Path(__file__).resolve().parents[1]; root = package.parents[1]
    config_path = Path(config_path); output = Path(output)
    inputs.require(not output.exists(), "write-once output exists")
    inputs.require(config_path.absolute() == package / CONFIG and config_path.resolve() == config_path.absolute(),
                   "fixed config path without redirects required")
    config_path = config_path.absolute()
    revision = git(root, "rev-parse", "HEAD").strip()
    inputs.require(inputs.re.fullmatch("[0-9a-f]{40}", revision), "full source commit required")
    paths = critical_paths(package); snapshots = {}
    for path in paths:
        inputs.require(path.is_absolute() and path.resolve() == path, "critical source redirect")
        clean(root, path)
        snapshots[path.relative_to(root).as_posix()] = inputs.identity(read(path))
    config_raw = read(config_path)
    inputs.require(inputs.identity(config_raw) == snapshots[config_path.relative_to(root).as_posix()], "config changed before parsing")
    config = inputs._parse(config_raw)
    bound = inputs.bind(package, config)
    inputs.require(bound["status"] == "BOUND_DEV_INPUTS_NOT_NATIVE_ADMISSION"
                   and all(bound.get(field) is False for field in UNQUALIFIED), "static binding cannot grant authority")
    inputs.require(bound["data_sha256"] == inputs.digest({k: v for k, v in bound.items() if k != "data_sha256"}),
                   "bound data digest mismatch")
    policy, evaluator = bound["policy_rows"], bound["evaluator_rows"]
    inputs.require(len(policy) == len(evaluator) == 832 and
                   len({r["parent_id"] for r in evaluator}) == 544, "fixed DEV population")
    inputs.require(len({r["row_id"] for r in policy}) == 832 and
                   [r["row_id"] for r in policy] == [r["row_id"] for r in evaluator], "policy/evaluator row join")
    dependencies = scorer.dependency_binding()
    callback = enrollment.scorer_binding(scorer.score_response, inputs.PANELS[:-1])
    validated = Counter(); tokens = Counter()
    for p, e in zip(policy, evaluator):
        inputs.require(set(p) == inputs.POLICY_FIELDS and set(e) == inputs.EVALUATOR_FIELDS, "output field boundary")
        inputs.require(p["prompt_token_hash"] == e["prompt_token_hash"] == inputs.digest(p["input_ids"])
                       and p["prompt_tokens"] == len(p["input_ids"])
                       and p["attention_mask"] == [1] * len(p["input_ids"]), "policy prefix binding")
        scorer.validate_evaluator_row(e)  # Source/reference validation, never a generated response.
        validated[e["panel"]] += 1; tokens[e["panel"]] += p["prompt_tokens"]
        if validated[e["panel"]] == inputs.COUNTS[e["panel"]][0]:
            print(json.dumps({"event": "source_panel_validated", "panel": e["panel"], "rows": validated[e["panel"]]}), flush=True)
    inputs.require(dict(validated) == {panel: counts[0] for panel, counts in inputs.COUNTS.items()}, "panel coverage")
    metadata = {k: v for k, v in bound.items() if k not in ("policy_rows", "evaluator_rows")}
    artifacts = {
        "policy/DEV_INPUTS.jsonl": b"".join(inputs.canonical(row) + b"\n" for row in policy),
        "evaluation/DEV_ROWS.jsonl": b"".join(inputs.canonical(row) + b"\n" for row in evaluator),
        "STATIC_BINDING.json": inputs.canonical(metadata) + b"\n",
    }
    scorer.validate_dependency_binding(dependencies)
    for label, expected in bound["bindings"].items():
        inputs.require(inputs.identity(inputs._read(root, label)) == expected, "data changed before publication: " + label)
    for label, expected in snapshots.items():
        path = root / label
        inputs.require(path.resolve() == path and inputs.identity(read(path)) == expected, "source/config changed: " + label)
        clean(root, path)
    inputs.require(git(root, "rev-parse", "HEAD").strip() == revision, "HEAD changed before publication")
    dirty = git(root, "status", "--porcelain")
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy").mkdir(); (output / "evaluation").mkdir()
    files = {}
    for label, raw in artifacts.items():
        with (output / label).open("xb") as f: f.write(raw)
        files[label] = inputs.identity(read(output / label))
        inputs.require(files[label] == inputs.identity(raw), "published output drift")
    receipt = {
        "schema_version": 1, "status": "STATIC_DEV_INPUT_SCORER_BINDING_NO_NATIVE_ADMISSION",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "code_commit": revision,
        "overall_dirty_status": dirty, "source_files": snapshots, "source_inventory_sha256": inputs.digest(snapshots),
        "config_file_sha256": inputs.identity(config_raw)["sha256"], "config_canonical_sha256": inputs.digest(config),
        "data_sha256": bound["data_sha256"], "counts": bound["counts"], "rows": 832, "parent_groups": 544,
        "prompt_tokens_by_panel": dict(tokens), "prompt_tokens_total": sum(tokens.values()),
        "source_rows_validated": dict(validated), "deterministic_data_references_validated": 768,
        "language_identity_only_rows": 64, "scorer_dependencies": dependencies, "callback_binding": callback,
        "files": files, **dict.fromkeys(UNQUALIFIED, False),
        "unresolved_native_registration": {"runtime_identity": None, "seeds": None, "decoding": None, "run_budgets": None},
        "actual_response_records": 0, "native_model_loads": 0, "model_calls": 0, "teacher_calls": 0,
        "optimizer_steps": 0, "training_tokens": 0, "generated_model_tokens": 0, "tokenizer_calls": 0,
        "held_out_access": {"CAL": False, "CHECK": False, "FINAL": False},
        "framework_modules_present": sorted(set(sys.modules).intersection({"torch", "transformers", "tokenizers", "tensorflow", "jax"})),
        "wall_seconds": time.perf_counter() - started,
        "scope": "Static sources, prompt separation and verifier mechanics only; no source enrollment, language calibration, native resource fit or model accuracy. Row IDs are routing metadata, never model messages.",
    }
    with (output / "BINDING_RECEIPT.json").open("xb") as f: f.write(inputs.canonical(receipt) + b"\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = prepare(args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "rows", "parent_groups", "prompt_tokens_total", "wall_seconds")}))
