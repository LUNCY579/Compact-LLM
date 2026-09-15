"""Strict source/asset/approval contracts. No model imports or network access."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CLOSED = {"licenses_training": False, "licenses_efficacy_claim": False,
          "licenses_retrieval_claim": False, "licenses_release": False}
COLLECTOR = "OPALLO-NATIVE-001"


class Blocked(RuntimeError):
    """An execution prerequisite failed; never a scientific negative result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise Blocked(f"nonfinite JSON constant: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    canonical(value)  # Also rejects overflowed JSON numbers such as 1e999.
    return value


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Exclusive creation; a partial write is deliberately NOT resumable.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        import os
        os.fsync(handle.fileno())


def append_event(path: Path, value: dict) -> None:
    import os
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(value).decode() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def valid_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def finite(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def safe_child(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "empty relative path")
    require("\\" not in relative and ":" not in relative, "portable POSIX path required")
    p = Path(relative)
    require(not p.is_absolute() and not any(v in ("..", ".git") for v in p.parts),
            "unsafe relative path")
    resolved = (root / p).resolve()
    require(resolved.is_relative_to(root.resolve()), "path escapes root")
    # No symlink aliases in evidence or source paths.
    cursor = root
    for part in p.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "symlink evidence/source is unsupported")
    return resolved


def pins() -> dict:
    return read(ROOT / "PINS.json")


def source_manifest() -> dict[str, str]:
    paths = list(ROOT.rglob("*.py")) + [ROOT / "PINS.json", ROOT / "README.md", ROOT / ".gitattributes"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def verify_source() -> dict[str, str]:
    manifest = source_manifest()
    require(read(ROOT / "REGISTRATION.json")["source_sha256"] == manifest,
            "collector source registration mismatch")
    for name, expected in pins()["source_sha256"].items():
        require(sha(ROOT.parent / "opallo_behavior_screen_v1" / name) == expected,
                f"frozen offline subject changed: {name}")
    return manifest


def screen():
    file = ROOT.parent / "opallo_behavior_screen_v1" / "screen.py"
    require(sha(file) == pins()["source_sha256"]["screen.py"], "offline scorer drift")
    spec = importlib.util.spec_from_file_location("opallo_frozen_screen", file)
    require(spec is not None and spec.loader is not None, "cannot load frozen scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset_manifest(model_dir: Path) -> dict:
    require(model_dir.is_dir() and not model_dir.is_symlink(), "local model directory required")
    expected = pins()["assets"]
    # HF must not discover unregistered overrides, adapters, alternate weights or code.
    metadata = {"CORE.json", "FILES.json", "README.md", "LICENSE", "LICENSE.txt", ".gitattributes"}
    actual = {p.name for p in model_dir.iterdir()}
    require(actual <= set(expected) | metadata,
            f"unregistered model directory entries: {sorted(actual - set(expected) - metadata)}")
    result = {}
    for name, item in expected.items():
        path = safe_child(model_dir, name)
        require(path.is_file(), f"missing official asset: {name}")
        require(path.stat().st_size == item["bytes"], f"asset size mismatch: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        require(result[name] == item, f"asset hash mismatch: {name}")
    gen = read(model_dir / "generation_config.json")
    require(gen.get("eos_token_id") == pins()["official_eos_ids"], "official EOS list changed")
    config = read(model_dir / "config.json")
    require(config.get("model_type") == "qwen3", "not a dense Qwen3 artifact")
    require(config.get("max_position_embeddings", 0) >= 4096, "context capacity too small")
    require(not config.get("auto_map") and not config.get("quantization_config"),
            "remote code/quantization not allowed")
    return result


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            timeout=20, check=False)
    require(result.returncode == 0, "Git receipt unavailable: " + result.stderr.decode(errors="replace")[:400])
    return result.stdout


def committed_file(repo: Path, commit: str, relative: str) -> Path:
    require(valid_sha(commit, 40), "full Git commit SHA required")
    path = safe_child(repo, relative)
    require(path.is_file(), f"missing committed input: {relative}")
    require(git(repo, "cat-file", "-t", commit).strip() == b"commit", "not a Git commit")
    require(git(repo, "show", f"{commit}:{relative}") == path.read_bytes(),
            f"working bytes differ from Git receipt: {relative}")
    return path


def reviewed_inputs(repo: Path, commit: str, auth_relative: str) -> tuple[dict, dict]:
    """Validate source-bound human attestations, not manufacture review approval."""
    auth = read(committed_file(repo, commit, auth_relative))
    p = pins()
    require(auth.get("protocol") == p["protocol"] and auth.get("subject_commit") == p["subject_commit"],
            "authorization subject mismatch")
    require(auth.get("collector") == COLLECTOR and auth.get("status") == "APPROVED_FOR_QUALIFICATION",
            "native qualification not approved")
    require(isinstance(auth.get("run_id"), str) and re.fullmatch(r"[A-Za-z0-9_-]{8,80}", auth["run_id"]),
            "registered run identifier required")
    require(auth.get("host_quiescent") is True, "quiescent-host attestation required")
    require(all(auth.get(k) is False for k in CLOSED), "closed authority boundary changed")
    require(auth.get("source_sha256") == verify_source(), "authorization source digest mismatch")
    git(repo, "merge-base", "--is-ancestor", p["subject_commit"], commit)
    for name in source_manifest():
        committed_file(repo, commit, f"experiments/opallo_native_collector_v1/{name}")
    committed_file(repo, commit, "experiments/opallo_native_collector_v1/REGISTRATION.json")
    for name in p["source_sha256"]:
        committed_file(repo, commit, f"experiments/opallo_behavior_screen_v1/{name}")
    inputs = {}
    for name in ("recipe", "human_review", "overlap_audit"):
        item = auth.get(name, {})
        require(isinstance(item, dict) and valid_sha(item.get("sha256")), f"missing {name} binding")
        path = committed_file(repo, commit, item.get("path", ""))
        require(sha(path) == item["sha256"], f"{name} byte digest mismatch")
        inputs[name] = read(path)
    human, audit = inputs["human_review"], inputs["overlap_audit"]
    for value in (human, audit):
        require(value.get("fixture_canonical_sha256") == p["fixture_canonical_sha256"], "review fixture drift")
        require(isinstance(value.get("reviewer_name"), str) and bool(value["reviewer_name"].strip()), "reviewer required")
        from datetime import datetime
        try:
            stamp = datetime.fromisoformat(value["reviewed_at_utc"].replace("Z", "+00:00"))
            require(stamp.utcoffset() is not None, "review time requires timezone")
        except (KeyError, ValueError, TypeError) as exc:
            raise Blocked("valid review time required") from exc
    validate_human(human)
    validate_audit(audit, repo)
    validate_recipe(inputs["recipe"])
    return auth, inputs["recipe"]


def validate_human(human: dict) -> None:
    require(human.get("status") == "APPROVED", "192-target human review pending")
    expected = pins()["review_rows"]  # IDs/scenarios only; no target/key access.
    rows = human.get("rows")
    require(isinstance(rows, list) and len(rows) == 192, "all 192 target decisions required")
    for actual, wanted in zip(rows, expected):
        require(isinstance(actual, dict) and actual.get("id") == wanted["id"] and
                actual.get("scenario_id") == wanted["scenario_id"], "human review ID/order mismatch")
        require(actual.get("decision") == "APPROVED" and all(actual.get(k) is True for k in
                ("complete_value_correct", "field_presence_correct", "unambiguous_contract")),
                "unapproved or unresolved target")


def validate_audit(audit: dict, repo: Path) -> None:
    require(audit.get("status") == "PASS_REVIEWED_UNION" and
            audit.get("unresolved_findings") == 0 and audit.get("exclusions") == [],
            "union overlap review incomplete or exclusions require new registration")
    required = {"training", "historical_demo", "exposed_panel", "repair_selection"}
    sources = audit.get("sources")
    require(isinstance(sources, list) and bool(sources), "actual pinned audit inputs required")
    covered = set()
    for source in sources:
        require(isinstance(source, dict), "audit source object required")
        require(source.get("category") in required, "unknown audit source category")
        covered.add(source["category"])
        require(valid_sha(source.get("source_commit"), 40) and valid_sha(source.get("sha256")),
                "audit input source commit and hash required")
        require(type(source.get("rows")) is int and source["rows"] > 0, "audit input row count required")
        require(type(source.get("normalized_full_prompt_matches")) is int and
                source["normalized_full_prompt_matches"] == 0 and
                type(source.get("normalized_latest_user_matches")) is int and
                source["normalized_latest_user_matches"] == 0, "exact overlap is unresolved")
        require(source.get("near_template_review") == "REVIEWED_NO_UNRESOLVED_FINDINGS",
                "near-template human review pending")
        path = safe_child(repo, source.get("path", ""))
        require(path.is_file() and sha(path) == source["sha256"], "audited source missing or changed")
    require(covered == required, "union source categories incomplete")


EFFECTIVE = {
    "forward_path": "stock_transformers_Qwen3ForCausalLM_manual_greedy",
    "attention_backend": "eager", "do_sample": False, "num_beams": 1,
    "max_new_tokens": 24, "max_context": 4096, "use_cache": True,
    "enable_thinking": False, "add_generation_prompt": True,
    "add_special_tokens": False, "truncation": False, "batch_size": 1,
    "logits_to_keep": 1, "logit_processors": [], "adapters": [],
    "decode_skip_special_tokens": False, "clean_up_tokenization_spaces": False,
    "probability": "exp(full_vocabulary_float64_log_softmax_raw_first_step_UNKNOWN)",
    "signal_storage": "full_vocabulary_little_endian_float64",
    "seed": 17, "tf32": False, "deterministic_algorithms": True,
    "sampling_defaults": "NOT_USED_manual_argmax", "optimizer_steps": 0,
}


def validate_recipe(recipe: dict) -> None:
    require(recipe.get("collector") == COLLECTOR and recipe.get("schema") == 1, "recipe schema mismatch")
    require(recipe.get("effective") == EFFECTIVE, "unregistered generation recipe")
    require(recipe.get("official_eos_ids") == pins()["official_eos_ids"], "recipe EOS mismatch")
    require(recipe.get("device") in ("cpu", "cuda:0"), "only CPU or one CUDA device supported")
    require(recipe.get("compute_dtype") in ("float32", "bfloat16"), "unsupported compute dtype")
    require(recipe["device"] != "cpu" or recipe["compute_dtype"] == "float32", "CPU candidate is FP32 only")
    require(type(recipe.get("threads")) is int and 1 <= recipe["threads"] <= 8, "invalid thread count")
    limits = recipe.get("limits", {})
    require(limits == {"per_capture_seconds": 900, "screen_seconds": 3600, "generation_max": 400,
                       "min_free_host_gib": 3.0, "max_process_tree_rss_gib": 8.0,
                       "min_free_cuda_gib_before_load": 3.0}, "resource limits changed")
    require(isinstance(recipe.get("runtime"), dict) and
            recipe["runtime"].get("packages", {}).get("transformers") == "4.51.3", "Transformers 4.51.3 required in isolated environment")
    require(recipe.get("runtime_sha256") == digest(recipe["runtime"]), "runtime digest mismatch")
    require(recipe.get("source_sha256") == source_manifest(), "recipe source mismatch")
    require(all(recipe.get(k) is False for k in CLOSED), "recipe authority boundary changed")
