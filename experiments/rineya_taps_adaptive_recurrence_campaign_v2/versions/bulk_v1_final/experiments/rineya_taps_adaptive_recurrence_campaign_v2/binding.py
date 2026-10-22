"""Exact allowlisted DEV/SMOKE tokenizer and shape binding; never load weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
BASE = "03db218ceebc61a2bc04f2482cf7e094045e01c1"
ARMR = REPO / "experiments/anchor_routed_middle_recurrence_v1"
LAB = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1"
HISTORY = ARMR / "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json"
SMOKE_HISTORY = ARMR / "evidence/00_baseline_requalification/20260905T040646_352570Z/run_manifest.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def admitted_source(source):
    source = Path(source).resolve(strict=True)
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    if source != Path(history["model_path"]).resolve(strict=True):
        raise PermissionError("source must resolve to the exact historical canonical directory")
    paths = [HISTORY, SMOKE_HISTORY, ARMR / "configs/reapplication_screen.json"]
    for root in (ARMR / "src", ARMR / "scripts", LAB / "src"):
        if not root.is_dir():
            raise RuntimeError(f"missing exact inherited source tree: {root}")
        paths.extend(root.rglob("*.py"))
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(REPO).as_posix()
        expected = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASE}:{relative}"],
                                           stderr=subprocess.PIPE)
        if path.read_bytes() != expected:
            raise RuntimeError(f"source-bound inherited bytes changed: {relative}")
        hashes[relative] = sha(path)
    for name, expected in history["checkpoint_files_before"].items():
        if not (source / name).is_file() or sha(source / name) != expected:
            raise RuntimeError(f"historical model asset mismatch: {name}")
    if any(path.relative_to(source).as_posix() not in history["checkpoint_files_before"]
           for path in source.rglob("*.py")):
        raise RuntimeError("unverified Python in model source")
    if history["status"] != "PASS":
        raise RuntimeError("controlling historical receipt is not PASS")
    return history, hashes


def header_only(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError("unexpected oversized safetensors metadata header")
        header = json.loads(stream.read(length))
    tensors = {name: row for name, row in header.items() if name != "__metadata__"}
    count = sum(math.prod(row["shape"]) for row in tensors.values())
    return {"header_bytes": length, "metadata": header.get("__metadata__"),
            "tensor_count": len(tensors), "stored_parameter_count": count,
            "stored_dtypes": dict(Counter(row["dtype"] for row in tensors.values())),
            "fp32_tensor_bytes": count * 4,
            "embedding_metadata": tensors.get("model.embed_tokens.weight"),
            "stored_lm_head_metadata": tensors.get("lm_head.weight"), "weights_loaded": False}


def targets(scorer):
    kind = scorer["type"]
    if kind == "exact":
        return [str(x) for x in scorer["answers"]]
    if kind in ("choice", "numeric"):
        return [str(scorer["value"])]
    if kind == "keywords":
        return [str(scorer["phrase"])]
    if kind == "contains_all":
        return [str(x) for x in scorer["values"]]
    if kind == "json_exact":
        return [json.dumps(scorer["value"], sort_keys=True)]
    return []


def bind_cases(tokenizer, config, scorer):
    """Read only the two exact current-worktree allowlist paths, never TEST."""
    historical = json.loads(HISTORY.read_text(encoding="utf-8"))
    smoke_historical = json.loads(SMOKE_HISTORY.read_text(encoding="utf-8"))
    expected = {"DEV": ("mergedev_fast.jsonl", historical["dataset_sha256"]),
                "SMOKE": ("supa_layer_ablation_smoke.jsonl", smoke_historical["dataset_sha256"])}
    rows, datasets = [], {}
    normalize = lambda value: " ".join(value.casefold().split())
    for split, (name, expected_hash) in expected.items():
        path = (LAB / "data/evaluation" / name).resolve(strict=True)
        if sha(path) != expected_hash:
            raise RuntimeError(f"allowlisted {split} data changed")
        bank = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not bank or len({case["case_id"] for case in bank}) != len(bank):
            raise ValueError("empty/duplicate case ids in allowlisted bank")
        datasets[split] = {"path": str(path), "sha256": expected_hash,
                           "items": len(bank), "exposure": "previously inspected development",
                           "scenario_clusters": dict(Counter(case["scenario_id"] for case in bank))}
        for case in bank:
            prompt = case["prompt"]
            if case["prompt_sha256"] != text_sha(prompt):
                raise RuntimeError(f"prompt digest mismatch: {case['case_id']}")
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            # Match inherited encode(): render, then tokenizer with its default specials.
            ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist()
            cap = min(config["max_new_tokens"], case.get("max_new_tokens", config["max_new_tokens"]))
            if type(cap) is not int or cap < 1 or not ids or len(ids) + cap - 1 > 4096:
                raise ValueError("case violates initial native shape contract")
            target_strings = targets(case["scorer"])
            normalized = normalize(prompt)
            mentioned = [target for target in target_strings if target and normalize(target) in normalized]
            options = case["scorer"].get("allowed", [])
            option_positions = {str(value): [m.start() for m in re.finditer(
                r"\b" + re.escape(str(value)) + r"\b", prompt, re.IGNORECASE)] for value in options}
            blind = {answer: scorer.score_response(answer, case["scorer"])["score"]
                     for answer in ("", "YES", "NO", "UNKNOWN", "0", "1")}
            rows.append({"case_id": case["case_id"], "split": split, "family": case["family"],
                         "independent_cluster": case["scenario_id"],
                         "cluster_status": "scenario_id provisional; semantic/generator independence unverified",
                         "source_split": case.get("split"), "transformation_id": case.get("transformation_id"),
                         "raw_prompt": prompt, "rendered_prompt": rendered,
                         "prompt_sha256": text_sha(prompt), "rendered_prompt_sha256": text_sha(rendered),
                         "input_token_ids": ids, "input_tokens": len(ids), "max_new_tokens": cap,
                         "max_forward_prefix_tokens": len(ids) + cap - 1,
                         "maximum_model_calls": cap,
                         "maximum_processed_prefix_positions": cap * len(ids) + cap * (cap - 1) // 2,
                         "scorer": case["scorer"], "format_scorer": case.get("format_scorer"),
                         "answerability_status": "requires semantic review; scorer alone does not prove answerability",
                         "audit": {"target_strings_found_in_prompt": mentioned,
                                   "target_copy_interpretation": "surface flag; choices/retrieval may legitimately contain target",
                                   "choice_option_positions": option_positions,
                                   "answer_position_cue_human_review": "NOT_REVIEWED",
                                   "constant_answer_scores": blind,
                                   "prompt_copy_score": scorer.score_response(prompt, case["scorer"])["score"],
                                   "query_blind_status": "NOT_IMPLEMENTED",
                                   "solver_verified": False, "human_review": "NOT_REVIEWED",
                                   "truncation": False}})
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[text_sha(normalize(row["raw_prompt"]))].append(row["case_id"])
    exact = [names for names in duplicates.values() if len(names) > 1]
    near = []
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            a, b = normalize(first["raw_prompt"]), normalize(second["raw_prompt"])
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if a != b and ratio >= .85:
                near.append({"case_ids": [first["case_id"], second["case_id"]], "ratio": ratio})
    return rows, datasets, {"normalized_exact_prompt_duplicates": exact,
                            "near_duplicate_method": "casefold/whitespace-normalized SequenceMatcher, autojunk=False, ratio>=.85",
                            "near_duplicate_pairs": near,
                            "scorer_caution": "choice uses last recognized allowed token; numeric uses final parsed number; prompt-copy floors are reported",
                            "semantic_review_status": "NOT_REVIEWED"}


def run(source, out):
    source, out = Path(source).resolve(strict=True), Path(out).resolve()
    campaign = Path(__file__).resolve().parent
    if campaign not in out.parents or out == source or source in out.parents:
        raise PermissionError("output must be an exclusive child of campaign_v2 outside assets")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "native_model_loads": 0, "training": False, "native_test": False,
               "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        historical, source_hashes = admitted_source(source)
        cache = out / "hf_cache"
        for name, path in {"HF_HOME": cache, "HF_HUB_CACHE": cache / "hub",
                           "HF_MODULES_CACHE": cache / "modules"}.items():
            os.environ[name] = str(path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        qwen_file = Path(inspect.getfile(Qwen3DecoderLayer))
        runtime = {name: importlib.metadata.version(name) for name in
                   ("torch", "transformers", "psutil", "safetensors", "tokenizers", "huggingface_hub")}
        if (platform.python_version() != historical["python"]
                or runtime["torch"] != historical["torch"]
                or runtime["transformers"] != historical["transformers"]
                or sha(qwen_file) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("exact historical tokenizer/Qwen runtime mismatch")
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True,
                                                  trust_remote_code=False, cache_dir=cache / "hub")
        if tokenizer.encode("UNKNOWN", add_special_tokens=False) != [51378]:
            raise RuntimeError("custom head UNKNOWN token identity mismatch")
        sys.path.insert(0, str(LAB / "src"))
        from rineya_cross_scale.evaluation import scoring
        if Path(inspect.getfile(scoring)).resolve() != (LAB / "src/rineya_cross_scale/evaluation/scoring.py").resolve():
            raise RuntimeError("loaded scorer module is not the source-bound current-worktree file")
        rows, datasets, audits = bind_cases(tokenizer, historical["config"], scoring)
        smoke = [row for row in rows if row["split"] == "SMOKE"]
        dev = [row for row in rows if row["split"] == "DEV"]
        if len(smoke) != 8 or len(dev) != 16:
            raise RuntimeError("historical complete-matrix count changed")
        forecast = {"physical_evaluations": 13 * len(smoke) + 18 * len(dev),
                    "smoke_evaluations_per_item": 13, "dev_evaluations_per_item": 18,
                    "maximum_model_calls": 13 * sum(row["maximum_model_calls"] for row in smoke)
                                           + 18 * sum(row["maximum_model_calls"] for row in dev),
                    "maximum_processed_prefix_positions": 13 * sum(row["maximum_processed_prefix_positions"] for row in smoke)
                                                          + 18 * sum(row["maximum_processed_prefix_positions"] for row in dev),
                    "maximum_generated_tokens": 13 * sum(row["max_new_tokens"] for row in smoke)
                                              + 18 * sum(row["max_new_tokens"] for row in dev),
                    "wall_seconds_forecast": None, "forecast_status": "shape/count only; coordinator must bind historical-cost or qualified-pilot forecast",
                    "oracle_physical_evaluations": 0, "note": "includes repeated deterministic seed rows as historical layout; not independent tasks"}
        own_source = {str(path.relative_to(REPO)): sha(path) for path in
                      (Path(__file__), campaign / "control.py")}
        binding = {"status": "TOKENIZER_METADATA_BOUND_NATIVE_NOT_QUALIFIED",
                   "source_path": str(source), "historical_receipt_sha256": sha(HISTORY),
                   "checkpoint_files": historical["checkpoint_files_before"],
                   "inherited_source_sha256": source_hashes, "campaign_source_sha256": own_source,
                   "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                   "runtime": {"interpreter": sys.executable, "python": platform.python_version(), **runtime,
                               "qwen_file": str(qwen_file), "qwen_sha256": sha(qwen_file),
                               "hf_cache": str(cache), "network_allowed": False},
                   "model_config": json.loads((source / "config.json").read_text()),
                   "safetensors_header": header_only(source / "model.safetensors"),
                   "tokenizer_class": type(tokenizer).__name__, "tokenizer_length": len(tokenizer),
                   "loaded_model_state_sha256": None,
                   "historical_expected_loaded_fp32_state_sha256": historical["state_hash_before"],
                   "loaded_custom_code_sha256": None, "native_qualified": False,
                   "datasets": datasets, "scorer_path": inspect.getfile(scoring),
                   "scorer_sha256": sha(inspect.getfile(scoring)),
                   "shapes": {"batch": 1, "padding": False, "cache": False,
                              "max_input_tokens": max(row["input_tokens"] for row in rows),
                              "max_new_tokens": max(row["max_new_tokens"] for row in rows),
                              "max_forward_prefix_tokens": max(row["max_forward_prefix_tokens"] for row in rows),
                              "max_shape_native_mechanics": "NOT_RUN"},
                   "forecast": forecast, "audits": audits}
        write(out / "MODEL_DATA_BINDING.json", binding)
        write(out / "PROMPT_AUDIT.json", rows)
        if any(sha(source / name) != expected for name, expected in historical["checkpoint_files_before"].items()):
            raise RuntimeError("canonical assets changed during metadata binding")
        receipt.update(status=binding["status"], source_unchanged=True,
                       binding_sha256=sha(out / "MODEL_DATA_BINDING.json"),
                       prompt_audit_sha256=sha(out / "PROMPT_AUDIT.json"))
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "TOKENIZER_METADATA_BOUND_NATIVE_NOT_QUALIFIED" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
