#!/usr/bin/env python3
"""Transactionally promote the evaluated abstention checkpoint as Supa-A1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VAULT = Path(__file__).resolve().parent
CATALOG = VAULT / "CATALOG.json"
SOURCE_ROOT = (
    REPO
    / "artifacts"
    / "rineya_abstention_gate_v1"
    / "checkpoint"
    / "Rineya-Nimbus-Abstain-1.0"
)
CONTROL_ROOT = (
    REPO
    / "artifacts"
    / "rineya_abstention_gate_v1"
    / "checkpoint"
    / "Rineya-Nimbus-Dense-1.0-nobias"
)
EVAL_ROOT = REPO / "artifacts" / "rineya_abstention_gate_v1" / "checkpoint_eval"
GATE_SPEC = (
    REPO
    / "artifacts"
    / "rineya_abstention_gate_v1"
    / "tuning_showcase64"
    / "showcase64_gate_S4_logit_bias.json"
)
CURRENT_BUILDER = (
    REPO
    / "Rineya_Abstention_Gate_Lab_v1"
    / "scripts"
    / "build_abstention_checkpoint.py"
)
DESTINATION_NAME = "Rineya-Supa-A1-1.0.0-devfreeze"
DESTINATION = VAULT / DESTINATION_NAME
RUNTIME_MODULE_NAME = "modeling_supa_a1.py"
ROOT_RUNTIME_MODULE = VAULT / RUNTIME_MODULE_NAME
TRANSACTION_JOURNAL = VAULT / ".supa-a1-promotion-transaction.json"
TRANSACTION_LOCK = VAULT / ".supa-a1-promotion.lock"
CORE_ID = (
    "rineya.core.supa-a1.v1.0.0-devfreeze.28L.qwen3-nimbus-lcr-"
    "step200-scale-0p8125-abstain-b5p15625"
)
MODEL_SHA256 = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
MODEL_BYTES = 1_192_135_096
SOURCE_CONFIG_SHA256 = (
    "630ec3b22b272439d441069eee1e1425f743c5f0986e314064883c1d8ea689e2"
)
SOURCE_MODELING_SHA256 = (
    "a56a2022ccbe45da003e3d7e695e9a86442b89966fd24b042e0e9299387f74a2"
)
SOURCE_MANIFEST_SHA256 = (
    "c613d0fd6c0da0ee19d0367331d8fc97b4d988b041a6faaf02c0ee196df7c389"
)
GATE_SPEC_SHA256 = "5f6cc42e9c4baa57a83ef0fbe6c0b924abb881246480e8b2b84d457ba240279d"
BASE_WEIGHT_SHA256 = "7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c"
ADAPTER_SHA256 = "517966d7564c083f12504e4c3ff1ac3e82a0abb97d2845ea3dd116174b01efb4"
ORIGINAL_BUILDER_SHA256 = (
    "0f7f2b573827da0f695cc5b1303135d9050c85d3bfb2d1a9fc186e1b6b9155c1"
)

SOURCE_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "modeling_abstain.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)
PROMOTED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    RUNTIME_MODULE_NAME,
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)
SOURCE_FILE_PINS = {
    "config.json": SOURCE_CONFIG_SHA256,
    "generation_config.json": (
        "9a464692406690ab0d54da69ee1b1a88ef80ea5f5cfad9d84180f2bc7f53259c"
    ),
    "model.safetensors": MODEL_SHA256,
    "modeling_abstain.py": SOURCE_MODELING_SHA256,
    "tokenizer_config.json": (
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    ),
    "tokenizer.json": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
    "vocab.json": ("ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": ("8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
}
EVIDENCE_FILES = {
    "Rineya-Nimbus-Abstain-1.0_showcase.jsonl": "supa_a1_showcase.jsonl",
    "Rineya-Nimbus-Abstain-1.0_showcase_report.json": "supa_a1_showcase_report.json",
    "Rineya-Nimbus-Abstain-1.0_showcase_report.md": "supa_a1_showcase_report.md",
    "Rineya-Nimbus-Abstain-1.0_stability.jsonl": "supa_a1_stability.jsonl",
    "Rineya-Nimbus-Abstain-1.0_stability_report.json": "supa_a1_stability_report.json",
    "Rineya-Nimbus-Abstain-1.0_stability_report.md": "supa_a1_stability_report.md",
    "Rineya-Nimbus-Dense-1.0-nobias_showcase.jsonl": "no_bias_control_showcase.jsonl",
    "Rineya-Nimbus-Dense-1.0-nobias_showcase_report.json": (
        "no_bias_control_showcase_report.json"
    ),
    "Rineya-Nimbus-Dense-1.0-nobias_showcase_report.md": (
        "no_bias_control_showcase_report.md"
    ),
    "Rineya-Nimbus-Dense-1.0-nobias_stability.jsonl": (
        "no_bias_control_stability.jsonl"
    ),
    "Rineya-Nimbus-Dense-1.0-nobias_stability_report.json": (
        "no_bias_control_stability_report.json"
    ),
    "Rineya-Nimbus-Dense-1.0-nobias_stability_report.md": (
        "no_bias_control_stability_report.md"
    ),
}
EVIDENCE_PINS = {
    "Rineya-Nimbus-Abstain-1.0_showcase.jsonl": "bd489f7fff1a68de24427ef53d87483f95353cb8279a016cdda52906fafee8ac",
    "Rineya-Nimbus-Abstain-1.0_showcase_report.json": "428b97203cf27303d20f677e939e672beecbe4f8dae48f57d10f9b01f54bc60d",
    "Rineya-Nimbus-Abstain-1.0_showcase_report.md": "39fa9feee6cd351600148a19a50f7037348146bd17cba43f067b5b4e18feacc6",
    "Rineya-Nimbus-Abstain-1.0_stability.jsonl": "406718cf253a487696d56ac9bcfc104ef7f293a7a51f47223b5f9590b6b2357f",
    "Rineya-Nimbus-Abstain-1.0_stability_report.json": "afa8f18b6bbc6a9edf104b3cafb5f249505abeb643c7ab888feb11dc5987fd7f",
    "Rineya-Nimbus-Abstain-1.0_stability_report.md": "7d65946e37db251746b8a02cd407090e07e92383868e1d950dea404506ff42c9",
    "Rineya-Nimbus-Dense-1.0-nobias_showcase.jsonl": "aa060280954a2f9004d9af7f17dd912a667c3e29848bb866ff7e502a83fddd3a",
    "Rineya-Nimbus-Dense-1.0-nobias_showcase_report.json": "cc54d121df6e0fc397ce74b585b222ee5825bd72acc956c52ae45d1d00641e70",
    "Rineya-Nimbus-Dense-1.0-nobias_showcase_report.md": "1134708f50b5caf04c23e3daae13ffa51d0aa1cc3dcec6ec40ad2468e89214d1",
    "Rineya-Nimbus-Dense-1.0-nobias_stability.jsonl": "8c2617bcc0daa84785e294e6271221494a7a32d604259fc5cffbe42794eb2d52",
    "Rineya-Nimbus-Dense-1.0-nobias_stability_report.json": "6e58e3dbd6316067ea6fb1665d1fbc8b58acb3ac0594295b0b710c16d844716f",
    "Rineya-Nimbus-Dense-1.0-nobias_stability_report.md": "039b0b2ee8c1d6a3233f01ff2ec9cb479f33ba69d03249df904d3f803a24e5ff",
}
EVALUATION_INPUTS = {
    REPO / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_prompts.jsonl": (
        "showcase_prompts.jsonl",
        "7fbf6a061613fa8e5f3a6ff3709b00247daac815b84b9559a90723c165182c6d",
    ),
    REPO / "Rineya_Long_Context_Demo_v1" / "data" / "stability_prompts.jsonl": (
        "stability_prompts.jsonl",
        "7dc1454cd8fc9e22f536f3acb10a4e302343896ce1837ff9a0ad4f163dc73901",
    ),
    REPO / "Rineya_Long_Context_Demo_v1" / "data" / "showcase_answer_key.jsonl": (
        "showcase_answer_key.jsonl",
        "dc1541b6053b7a945774e3d87ca02cd3f06adb0a27a2f707f61d655001b6d53b",
    ),
    REPO / "Rineya_Long_Context_Demo_v1" / "data" / "stability_answer_key.jsonl": (
        "stability_answer_key.jsonl",
        "248bdc5592ba9eac05737e6cf160d5e6ec85d03590ee4ba8778f28e4098d01d9",
    ),
}
EVALUATOR = REPO / "Rineya_Long_Context_Demo_v1" / "scripts" / "evaluate_predictions.py"
EVALUATOR_SHA256 = "92c9e17762ff5979ac6c0657be26ee0a99bfce0620453886a2e2bc81632af1b5"
RUNNER = REPO / "Rineya_Long_Context_Demo_v1" / "scripts" / "run_local_hf.py"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def acquire_transaction_lock(transaction_id: str) -> int:
    try:
        descriptor = os.open(
            TRANSACTION_LOCK,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another Supa-A1 transaction owns {TRANSACTION_LOCK}"
        ) from exc
    payload = {
        "schema_version": "1.0",
        "transaction_id": transaction_id,
        "owner_pid": os.getpid(),
        "acquired_at_utc": utc_now(),
    }
    os.write(
        descriptor,
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
    )
    return descriptor


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied still implies alive.
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return getattr(exc, "winerror", None) not in {87, 1168}
    return True


def restore_catalog_or_fail(
    backup: Path,
    original_sha256: str,
    new_sha256: str | None,
) -> None:
    current_hash = sha256_file(CATALOG) if CATALOG.is_file() else None
    if current_hash == original_sha256:
        return
    if new_sha256 is None or current_hash != new_sha256:
        raise RuntimeError(
            "Catalog is neither the recorded original nor this transaction's new state"
        )
    if not backup.is_file():
        raise RuntimeError(
            "Catalog differs from the original transaction state and its backup is missing"
        )
    if sha256_file(backup) != original_sha256:
        raise RuntimeError("Catalog backup does not match the recorded original hash")
    os.replace(backup, CATALOG)
    if sha256_file(CATALOG) != original_sha256:
        raise RuntimeError("Catalog restoration did not reproduce the original hash")


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def report_hits(path: Path) -> tuple[int, int]:
    report = load_json(path)
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError(f"Expected exactly one report run: {path}")
    run = runs[0]
    items = int(run["key_items"])
    if int(run.get("prediction_rows", -1)) != items:
        raise ValueError(f"Prediction row count is incomplete: {path}")
    if int(run.get("covered_items", -1)) != items or float(run.get("coverage", 0)) != 1:
        raise ValueError(f"Evaluation coverage is incomplete: {path}")
    if run.get("duplicate_prediction_ids") or run.get("unexpected_prediction_ids"):
        raise ValueError(f"Evaluation identity mismatch: {path}")
    hits = round(float(run["strict_exact"]) * items)
    return hits, items


def validate_sources() -> dict[str, Any]:
    required = [SOURCE_ROOT / name for name in SOURCE_MODEL_FILES]
    required += [
        SOURCE_ROOT / "CHECKPOINT.json",
        CONTROL_ROOT / "model.safetensors",
        CONTROL_ROOT / "config.json",
        CONTROL_ROOT / "generation_config.json",
        GATE_SPEC,
        CURRENT_BUILDER,
        EVALUATOR,
        RUNNER,
    ]
    required += [EVAL_ROOT / name for name in EVIDENCE_FILES]
    required += list(EVALUATION_INPUTS)
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing promotion inputs: " + ", ".join(missing))

    pins = {
        CONTROL_ROOT / "model.safetensors": (MODEL_BYTES, MODEL_SHA256),
        SOURCE_ROOT / "CHECKPOINT.json": (None, SOURCE_MANIFEST_SHA256),
        GATE_SPEC: (None, GATE_SPEC_SHA256),
        CONTROL_ROOT / "config.json": (
            None,
            "f4cf2e6b25181578265205d4373303769702851a8b88cc9eb2b45a250595bbc9",
        ),
        CONTROL_ROOT / "generation_config.json": (
            None,
            "0a6ef79013177858dd5820698b12270863e03baf60ec36521e5c6a34da63dbdc",
        ),
        EVALUATOR: (None, EVALUATOR_SHA256),
    }
    pins.update(
        {
            SOURCE_ROOT / name: (None, expected)
            for name, expected in SOURCE_FILE_PINS.items()
        }
    )
    pins.update(
        {EVAL_ROOT / name: (None, expected) for name, expected in EVIDENCE_PINS.items()}
    )
    pins.update(
        {path: (None, expected) for path, (_, expected) in EVALUATION_INPUTS.items()}
    )
    for path, (expected_bytes, expected_hash) in pins.items():
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected size for {path}: {path.stat().st_size}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Unexpected SHA-256 for {path}: {actual_hash}")

    config = load_json(SOURCE_ROOT / "config.json")
    if config.get("auto_map") != {
        "AutoModelForCausalLM": "modeling_abstain.Qwen3AbstainForCausalLM"
    }:
        raise ValueError("Source checkpoint does not register the abstention model")
    if config.get("abstain_token_id") != 51378:
        raise ValueError("Unexpected abstention token id")
    if float(config.get("abstain_logit_bias")) != 5.156249985098839:
        raise ValueError("Unexpected abstention logit bias")

    observed = {
        "supa_a1_showcase": report_hits(
            EVAL_ROOT / "Rineya-Nimbus-Abstain-1.0_showcase_report.json"
        ),
        "supa_a1_stability": report_hits(
            EVAL_ROOT / "Rineya-Nimbus-Abstain-1.0_stability_report.json"
        ),
        "no_bias_showcase": report_hits(
            EVAL_ROOT / "Rineya-Nimbus-Dense-1.0-nobias_showcase_report.json"
        ),
        "no_bias_stability": report_hits(
            EVAL_ROOT / "Rineya-Nimbus-Dense-1.0-nobias_stability_report.json"
        ),
    }
    expected = {
        "supa_a1_showcase": (43, 64),
        "supa_a1_stability": (125, 192),
        "no_bias_showcase": (40, 64),
        "no_bias_stability": (121, 192),
    }
    if observed != expected:
        raise ValueError(f"Evaluation evidence changed: {observed!r}")
    return {"source_config": config, "evaluation": observed}


def source_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "former_identity": "Rineya-Nimbus-Abstain-1.0",
        "source_manifest": {
            "path": relative(SOURCE_ROOT / "CHECKPOINT.json"),
            "sha256": SOURCE_MANIFEST_SHA256,
            "sanitization": (
                "The exact source shape is bundled as ORIGINAL_CHECKPOINT.sanitized.json "
                "with machine-specific paths replaced by repository-relative paths."
            ),
        },
        "recipe": {
            "base": "Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze",
            "base_weight_sha256": BASE_WEIGHT_SHA256,
            "adapter": "artifacts/halo_long_context_sft_v1/stage_a/checkpoint-200",
            "adapter_weight_sha256": ADAPTER_SHA256,
            "adapter_scale": 0.8125,
            "scaled_lora_slots": 196,
            "merge_dtype": "float32",
            "saved_dtype": "bfloat16",
        },
        "gate": {
            "spec": relative(GATE_SPEC),
            "spec_sha256": GATE_SPEC_SHA256,
            "signal": "unknown_exact_minus_top1",
            "threshold": -5.156249985098839,
            "unknown_surface": "UNKNOWN",
            "unknown_token_id": 51378,
            "logit_bias": 5.156249985098839,
            "selection_status": (
                "POST-HOC fourth gate chosen after the S1 confirmatory Stability run; "
                "its threshold itself was selected on Showcase only"
            ),
        },
        "implementation": {
            "original_builder_sha256": ORIGINAL_BUILDER_SHA256,
            "current_builder": file_record(CURRENT_BUILDER),
            "note": (
                "The current builder replaces the original non-round-tripping "
                "GenerationConfig.sequence_bias route with the evaluated auto_map "
                "model-class route."
            ),
        },
    }


def sanitized_original_manifest() -> dict[str, Any]:
    manifest = load_json(SOURCE_ROOT / "CHECKPOINT.json")
    manifest["recipe"]["base"] = "Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze"
    manifest["recipe"]["adapter"] = (
        "artifacts/halo_long_context_sft_v1/stage_a/checkpoint-200"
    )
    manifest["abstention"]["gate_spec"] = "evidence/showcase64_gate_S4_logit_bias.json"
    manifest["implementation"]["script"] = (
        "Rineya_Abstention_Gate_Lab_v1/scripts/build_abstention_checkpoint.py"
    )
    manifest["source_manifest_sha256"] = SOURCE_MANIFEST_SHA256
    manifest["sanitized_fields"] = [
        "recipe.base",
        "recipe.adapter",
        "abstention.gate_spec",
        "implementation.script",
    ]
    return manifest


def directory_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def evaluation_manifest(evidence: Path) -> dict[str, Any]:
    outputs = {
        destination_name: directory_file_record(evidence / destination_name, evidence)
        for destination_name in EVIDENCE_FILES.values()
    }
    inputs = {
        destination_name: directory_file_record(evidence / destination_name, evidence)
        for destination_name, _ in EVALUATION_INPUTS.values()
    }
    return {
        "schema_version": "1.0",
        "checkpoint": {
            "weight_sha256": MODEL_SHA256,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_modeling_sha256": SOURCE_MODELING_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "protocol": {
            "context_tokens": 4096,
            "max_new_tokens": 24,
            "seed": 17,
            "decoding": "greedy",
            "thinking": False,
            "showcase_role": "threshold selection and checkpoint evaluation",
            "stability_role": (
                "previously inspected validation; S4 was added after the S1 "
                "confirmatory run, although the S4 threshold used Showcase only"
            ),
        },
        "inputs": inputs,
        "outputs": outputs,
        "scorer": file_record(EVALUATOR),
        "runner": {
            "current_reproduction_runner": file_record(RUNNER),
            "exact_evaluation_revision": "not durably recorded",
            "limitation": (
                "The frozen predictions and reports are hash-pinned, but the exact "
                "runner source revision used to generate them was not recorded at "
                "execution time; this manifest does not retroactively prove causation."
            ),
        },
        "assertions": {
            "supa_a1_showcase": {"hits": 43, "items": 64, "coverage": 1.0},
            "supa_a1_stability": {"hits": 125, "items": 192, "coverage": 1.0},
            "no_bias_showcase": {"hits": 40, "items": 64, "coverage": 1.0},
            "no_bias_stability": {"hits": 121, "items": 192, "coverage": 1.0},
        },
    }


def release_manifest(directory: Path, published_at: str) -> dict[str, Any]:
    files = {
        path.relative_to(directory).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "FILES.json"
    }
    return {
        "schema_version": "1.0",
        "core_id": CORE_ID,
        "published_at_utc": published_at,
        "files": files,
    }


def core_payload(published_at: str) -> dict[str, Any]:
    return {
        "schema_version": "1.2",
        "core_id": CORE_ID,
        "codename": "Supa-A1",
        "family_name": "Supa",
        "directory_name": DESTINATION_NAME,
        "version": "1.0.0-devfreeze",
        "tracking_label": "Supa-A1",
        "tracking": {
            "scheme": "family letter/sequence release label",
            "generation": "A",
            "sequence": 1,
            "label": "A1",
        },
        "kind": "dense_lora_merge_with_baked_abstention",
        "status": "showcase_selected_posthoc_stability_evaluated_private_core",
        "classification": "single_pass_posthoc_abstention_core",
        "public_claim": (
            "Single-pass Supa A-series core: 43/64 Showcase (tuning-split checkpoint "
            "evaluation), 125/192 previously inspected Stability, 168/256 combined. "
            "Requires greedy decoding and trust_remote_code=True. Sealed TEST not run."
        ),
        "release_manifest": "FILES.json",
        "published_at_utc": published_at,
        "former_identity": {
            "name": "Rineya-Nimbus-Abstain-1.0",
            "kind": "local checkpoint name",
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "source": source_manifest()["recipe"],
        "abstention": source_manifest()["gate"]
        | {
            "form": "constant offset applied inside Qwen3AbstainForCausalLM.forward",
            "loader_requirement": "trust_remote_code=True",
            "runtime_module": RUNTIME_MODULE_NAME,
            "decoding_requirement": "greedy argmax",
            "known_deviation": (
                "The model class applies the offset at every decode step; the tuned "
                "gate was defined on the first generated token. No mid-answer UNKNOWN "
                "was observed on the 256-row evaluation."
            ),
        },
        "evaluation": {
            "protocol": {
                "suite": "Rineya Long-Context Demonstration Suite",
                "context_tokens": 4096,
                "max_new_tokens": 24,
                "decoding": "greedy",
                "seed": 17,
                "thinking": False,
            },
            "checkpoint_results": {
                "showcase64": {"hits": 43, "items": 64, "role": "tuning split"},
                "stability192": {
                    "hits": 125,
                    "items": 192,
                    "role": (
                        "previously inspected validation; S4 was added after the S1 "
                        "confirmatory run, threshold selected on Showcase only"
                    ),
                },
                "all256": {"hits": 168, "items": 256, "role": "combined report"},
            },
            "offline_gate_reference": {
                "showcase64": {"hits": 46, "items": 64},
                "stability192": {"hits": 125, "items": 192},
                "caveat": (
                    "Do not inherit 46/64 to this checkpoint. Two rows drifted during "
                    "BF16 LoRA folding and one additional showcase row differs because "
                    "the checkpoint applies the offset at every decode step."
                ),
            },
            "no_bias_control": {
                "former_name": "Rineya-Nimbus-Dense-1.0-nobias",
                "classification": "abstention_ablation_control_not_a_named_core",
                "showcase64": {"hits": 40, "items": 64},
                "stability192": {"hits": 121, "items": 192},
                "all256": {"hits": 161, "items": 256},
                "weight_sha256": MODEL_SHA256,
                "note": (
                    "This control is weight-identical to Supa Core 1.7 and Supa-A1. "
                    "It is evidence, not a distinct Nimbus model."
                ),
            },
        },
        "checks": {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "vocab_size": 151936,
            "model_type": "qwen3",
            "architecture": "Qwen3AbstainForCausalLM",
            "weight_bytes": MODEL_BYTES,
            "weight_sha256": MODEL_SHA256,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_modeling_sha256": SOURCE_MODELING_SHA256,
            "has_tokenizer": True,
        },
        "limits": [
            "Showcase64 selected the threshold; it is not untouched test data.",
            "S4 was added after Stability had been inspected for S1; its threshold used Showcase only.",
            "The 125/192 gain over the 121/192 no-bias control has exact McNemar p=0.2188.",
            "The threshold is calibrated to this model, adapter scale, prompt, and class mix.",
            "Serving stacks that reject remote code need an equivalent logits processor.",
        ],
    }


def model_card() -> str:
    return f"""# Rineya Supa-A1

Supa-A1 is the first A-series tracked abstention core in the Rineya family. It
promotes the evaluated checkpoint formerly called
`Rineya-Nimbus-Abstain-1.0` without changing its weights or behavior.

## Results

| Split | Role | Strict exact |
|---|---|---:|
| Showcase64 | threshold selection / checkpoint evaluation | **43/64** |
| Stability192 | previously inspected validation; S4 added post-confirmation | **125/192** |
| All256 | combined, not independent | **168/256** |

The offline S4 gate scored 46/64 on Showcase and 125/192 on Stability. The
published checkpoint itself is 43/64 and 125/192; it does not inherit the
offline tuning score. Sealed TEST has not been run.

## What changed

The dense tensor is Nimbus 1.6 plus the Stage-A step-200 LoRA at scale 0.8125,
merged in FP32 and stored in BF16. A custom Qwen3 class adds
`5.156249985098839` to token 51378 (`UNKNOWN`) before greedy argmax. The rule is
exact for `logP(UNKNOWN) - logP(top1)` because the shared softmax normalizer
cancels. The weight SHA-256 is `{MODEL_SHA256}` and is byte-identical to Supa
Core 1.7; the behavioral difference is the model class and config.

## Load

Local standalone-directory load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "{DESTINATION_NAME}"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype="bfloat16", trust_remote_code=True
)
```

Private family-vault load (Transformers 5.14 resolves custom code at repository
root, so the root runtime module is intentionally mirrored):

```python
repo = "Luncy1978/rineya-published-cores"
subfolder = "{DESTINATION_NAME}"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    subfolder=subfolder,
    dtype="bfloat16",
    trust_remote_code=True,
)
```

Use greedy decoding (`do_sample=False`). The custom code requirement and the
fact that the bias is applied at every decode step are material compatibility
constraints, not optional implementation details.

## No-bias control and naming

The former `Rineya-Nimbus-Dense-1.0-nobias` checkpoint is retained under
`evidence/` as Supa-A1's ablation control. It is not published as `Nimbus-A1`:
its weights are identical to Supa Core 1.7, so that name would collide with the
distinct Nimbus 1.6 lineage. `Nimbus-A1` is instead a tracking label for the
actual Nimbus 1.6 core.

See `CORE.json`, `FILES.json`, `SOURCE_CHECKPOINT.json`, and `evidence/` for
machine-readable provenance and frozen predictions. `EVALUATION_MANIFEST.json`
explicitly records that the exact runner revision used to generate the frozen
predictions was not durably captured; their files and scoring inputs are still
hash-pinned.
"""


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_staging(staging: Path, published_at: str) -> None:
    staging.mkdir(parents=False)
    for name in SOURCE_MODEL_FILES:
        source = SOURCE_ROOT / name
        destination_name = (
            RUNTIME_MODULE_NAME if name == "modeling_abstain.py" else name
        )
        destination = staging / destination_name
        if name == "model.safetensors":
            link_or_copy(source, destination)
        elif name == "config.json":
            config = load_json(source)
            config["auto_map"] = {
                "AutoModelForCausalLM": ("modeling_supa_a1.Qwen3AbstainForCausalLM")
            }
            atomic_json(destination, config)
        else:
            shutil.copy2(source, destination)
    shutil.copy2(VAULT / "Rineya-Supa-1.7.0-devfreeze" / ".gitattributes", staging)
    shutil.copy2(VAULT / "Rineya-Supa-1.7.0-devfreeze" / "LICENSE", staging)

    evidence = staging / "evidence"
    evidence.mkdir()
    for source_name, destination_name in EVIDENCE_FILES.items():
        shutil.copy2(EVAL_ROOT / source_name, evidence / destination_name)
    for source, (destination_name, _) in EVALUATION_INPUTS.items():
        shutil.copy2(source, evidence / destination_name)
    shutil.copy2(GATE_SPEC, evidence / "showcase64_gate_S4_logit_bias.json")
    shutil.copy2(CONTROL_ROOT / "config.json", evidence / "no_bias_control_config.json")
    shutil.copy2(
        CONTROL_ROOT / "generation_config.json",
        evidence / "no_bias_control_generation_config.json",
    )
    atomic_json(
        evidence / "ORIGINAL_CHECKPOINT.sanitized.json",
        sanitized_original_manifest(),
    )
    atomic_json(evidence / "EVALUATION_MANIFEST.json", evaluation_manifest(evidence))

    atomic_json(staging / "CORE.json", core_payload(published_at))
    atomic_json(staging / "SOURCE_CHECKPOINT.json", source_manifest())
    atomic_write(staging / "README.md", model_card())
    atomic_json(staging / "FILES.json", release_manifest(staging, published_at))


def verify_directory(directory: Path) -> None:
    for name in PROMOTED_MODEL_FILES + (
        ".gitattributes",
        "LICENSE",
        "CORE.json",
        "FILES.json",
        "SOURCE_CHECKPOINT.json",
        "README.md",
    ):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    for destination_name in EVIDENCE_FILES.values():
        if not (directory / "evidence" / destination_name).is_file():
            raise FileNotFoundError(directory / "evidence" / destination_name)

    manifest = load_json(directory / "FILES.json")
    if manifest.get("core_id") != CORE_ID:
        raise ValueError("Release manifest has the wrong core identity")
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise TypeError("Release manifest has no file map")
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "FILES.json"
    }
    if set(records) != actual_paths:
        raise ValueError("Release manifest file inventory does not match the directory")
    for relative_path, record in records.items():
        path = directory / relative_path
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Release file size changed: {relative_path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Release file hash changed: {relative_path}")

    weight = directory / "model.safetensors"
    if weight.stat().st_size != MODEL_BYTES or sha256_file(weight) != MODEL_SHA256:
        raise ValueError("Promoted model weight does not match the pinned checkpoint")
    config = load_json(directory / "config.json")
    if config.get("abstain_token_id") != 51378:
        raise ValueError("Promoted config lost the abstention token")
    if config.get("auto_map") != {
        "AutoModelForCausalLM": "modeling_supa_a1.Qwen3AbstainForCausalLM"
    }:
        raise ValueError("Promoted config lost its auto_map")
    compile(
        (directory / RUNTIME_MODULE_NAME).read_text(encoding="utf-8"),
        str(directory / RUNTIME_MODULE_NAME),
        "exec",
    )
    if sha256_file(directory / RUNTIME_MODULE_NAME) != SOURCE_MODELING_SHA256:
        raise ValueError("Promoted custom runtime differs from the evaluated source")
    core = load_json(directory / "CORE.json")
    if core.get("core_id") != CORE_ID or core.get("tracking_label") != "Supa-A1":
        raise ValueError("Promoted CORE.json identity is incorrect")
    evaluation = load_json(directory / "evidence" / "EVALUATION_MANIFEST.json")
    if evaluation.get("assertions", {}).get("supa_a1_stability") != {
        "hits": 125,
        "items": 192,
        "coverage": 1.0,
    }:
        raise ValueError("Evaluation manifest lost the Stability assertion")


def catalog_with_supa_a1(
    catalog: dict[str, Any], core: dict[str, Any]
) -> dict[str, Any]:
    updated = json.loads(json.dumps(catalog))
    cores = updated.get("cores")
    if not isinstance(cores, list):
        raise TypeError("CATALOG.json has no cores list")
    collisions = [
        entry
        for entry in cores
        if entry.get("core_id") == CORE_ID or entry.get("directory") == DESTINATION_NAME
    ]
    if collisions:
        raise FileExistsError("Supa-A1 is already present in CATALOG.json")
    cores.append(
        {
            "core_id": CORE_ID,
            "codename": "Supa-A1",
            "tracking_label": "Supa-A1",
            "directory": DESTINATION_NAME,
            "present": True,
            "core": core,
        }
    )
    updated["updated_at_utc"] = core["published_at_utc"]
    updated["last_written"] = [DESTINATION_NAME]
    labels = updated.setdefault("tracking_labels", {})
    labels["Supa-A1"] = {
        "kind": "published_core",
        "core_id": CORE_ID,
        "directory": DESTINATION_NAME,
    }
    labels["Nimbus-A1"] = {
        "kind": "tracking_alias",
        "core_id": (
            "rineya.core.nimbus.v1.6.0-devfreeze.28L.qwen3-x-halo-uncensored-alpha-0.40"
        ),
        "directory": "Rineya-Nimbus-1.6.0-devfreeze",
        "note": (
            "Assigned to the actual Nimbus lineage; not the weight-identical "
            "Rineya-Nimbus-Dense-1.0-nobias ablation control."
        ),
    }
    return updated


def apply() -> None:
    if TRANSACTION_JOURNAL.exists():
        raise RuntimeError("Unfinished Supa-A1 transaction exists; run recover first")
    transaction_id = uuid.uuid4().hex
    lock_descriptor = acquire_transaction_lock(transaction_id)
    published_at = utc_now()
    staging = VAULT / f".{DESTINATION_NAME}.staging-{uuid.uuid4().hex}"
    catalog_temporary = CATALOG.with_name(f".{CATALOG.name}.tmp-{uuid.uuid4().hex}")
    catalog_backup = CATALOG.with_name(f".{CATALOG.name}.backup-{uuid.uuid4().hex}")
    runtime_temporary = VAULT / f".{RUNTIME_MODULE_NAME}.tmp-{uuid.uuid4().hex}"
    runtime_was_absent = not ROOT_RUNTIME_MODULE.exists()
    journal_written = False
    transaction_complete = False
    rollback_complete = False
    original_catalog_sha256: str | None = None
    journal: dict[str, Any] = {}
    try:
        original_catalog_sha256 = sha256_file(CATALOG)
        journal = {
            "schema_version": "1.0",
            "transaction_id": transaction_id,
            "owner_pid": os.getpid(),
            "destination": str(DESTINATION),
            "staging": str(staging),
            "catalog_backup": str(catalog_backup),
            "runtime_module": str(ROOT_RUNTIME_MODULE),
            "runtime_was_absent": runtime_was_absent,
            "original_catalog_sha256": original_catalog_sha256,
            "new_catalog_sha256": None,
        }
        atomic_json(TRANSACTION_JOURNAL, journal)
        journal_written = True
        validate_sources()
        if DESTINATION.exists():
            raise FileExistsError(f"Refusing to overwrite {DESTINATION}")
        if (
            ROOT_RUNTIME_MODULE.exists()
            and sha256_file(ROOT_RUNTIME_MODULE) != SOURCE_MODELING_SHA256
        ):
            raise FileExistsError(f"Runtime module collision: {ROOT_RUNTIME_MODULE}")
        build_staging(staging, published_at)
        verify_directory(staging)
        new_catalog = catalog_with_supa_a1(
            load_json(CATALOG), load_json(staging / "CORE.json")
        )
        catalog_temporary.write_text(
            json.dumps(new_catalog, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        journal["new_catalog_sha256"] = sha256_file(catalog_temporary)
        atomic_json(TRANSACTION_JOURNAL, journal)
        shutil.copy2(CATALOG, catalog_backup)
        if sha256_file(catalog_backup) != original_catalog_sha256:
            raise RuntimeError(
                "Catalog backup changed while the transaction was staged"
            )
        shutil.copy2(SOURCE_ROOT / "modeling_abstain.py", runtime_temporary)
        if sha256_file(CATALOG) != original_catalog_sha256:
            raise RuntimeError("Catalog changed concurrently before commit")
        os.replace(staging, DESTINATION)
        if not ROOT_RUNTIME_MODULE.exists():
            os.replace(runtime_temporary, ROOT_RUNTIME_MODULE)
        os.replace(catalog_temporary, CATALOG)
        verify()
        smoke_load()
        transaction_complete = True
    except BaseException:
        try:
            if original_catalog_sha256 is None:
                raise RuntimeError(
                    "Transaction never recorded the original catalog hash"
                )
            restore_catalog_or_fail(
                catalog_backup,
                original_catalog_sha256,
                journal.get("new_catalog_sha256"),
            )
            if runtime_was_absent and ROOT_RUNTIME_MODULE.exists():
                ROOT_RUNTIME_MODULE.unlink()
            if DESTINATION.exists():
                shutil.rmtree(DESTINATION)
            if staging.exists():
                shutil.rmtree(staging)
            rollback_complete = True
        except BaseException as rollback_error:
            raise RuntimeError(
                "Supa-A1 rollback failed; preserve the journal and run recover"
            ) from rollback_error
        raise
    finally:
        safe_to_clean = transaction_complete or rollback_complete or not journal_written
        os.close(lock_descriptor)
        if safe_to_clean:
            if staging.exists():
                shutil.rmtree(staging)
            if catalog_temporary.exists():
                catalog_temporary.unlink()
            if runtime_temporary.exists():
                runtime_temporary.unlink()
            if catalog_backup.exists():
                catalog_backup.unlink()
            if TRANSACTION_JOURNAL.exists():
                TRANSACTION_JOURNAL.unlink()
            if TRANSACTION_LOCK.exists():
                TRANSACTION_LOCK.unlink()


def verify() -> None:
    validate_sources()
    verify_directory(DESTINATION)
    if not ROOT_RUNTIME_MODULE.is_file():
        raise FileNotFoundError(ROOT_RUNTIME_MODULE)
    if sha256_file(ROOT_RUNTIME_MODULE) != SOURCE_MODELING_SHA256:
        raise ValueError("Root custom-code module differs from the evaluated source")
    catalog = load_json(CATALOG)
    matches = [entry for entry in catalog["cores"] if entry.get("core_id") == CORE_ID]
    if len(matches) != 1:
        raise ValueError("CATALOG.json does not resolve Supa-A1 exactly once")
    match = matches[0]
    expected_outer = {
        "core_id": CORE_ID,
        "codename": "Supa-A1",
        "tracking_label": "Supa-A1",
        "directory": DESTINATION_NAME,
        "present": True,
    }
    if {key: match.get(key) for key in expected_outer} != expected_outer:
        raise ValueError("CATALOG.json has inconsistent Supa-A1 outer metadata")
    if match.get("core") != load_json(DESTINATION / "CORE.json"):
        raise ValueError("Catalog-embedded Supa-A1 core differs from CORE.json")
    labels = catalog.get("tracking_labels", {})
    if labels.get("Supa-A1", {}).get("core_id") != CORE_ID:
        raise ValueError("Supa-A1 tracking label is missing")
    nimbus = next(
        entry for entry in catalog["cores"] if entry.get("codename") == "Nimbus"
    )
    if labels.get("Nimbus-A1", {}).get("core_id") != nimbus.get("core_id"):
        raise ValueError("Nimbus-A1 does not resolve to the actual Nimbus core")
    nimbus_core = load_json(VAULT / nimbus["directory"] / "CORE.json")
    if (
        nimbus.get("tracking_label") != "Nimbus-A1"
        or nimbus_core.get("tracking_label") != "Nimbus-A1"
    ):
        raise ValueError("Nimbus-A1 metadata is incomplete")


def smoke_load() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(VAULT, subfolder=DESTINATION_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        VAULT,
        subfolder=DESTINATION_NAME,
        dtype="bfloat16",
        device_map="cpu",
        trust_remote_code=True,
    )
    if model.__class__.__name__ != "Qwen3AbstainForCausalLM":
        raise TypeError(f"Unexpected loaded class: {model.__class__.__name__}")
    if tokenizer.encode("UNKNOWN", add_special_tokens=False) != [51378]:
        raise ValueError("Tokenizer no longer maps UNKNOWN to the pinned token")
    encoded = tokenizer("promotion smoke test", return_tensors="pt")
    original_bias = float(model.config.abstain_logit_bias)
    with torch.inference_mode():
        model.config.abstain_logit_bias = 0.0
        plain = model(**encoded).logits[:, -1, :].to(torch.float32)
        model.config.abstain_logit_bias = original_bias
        biased = model(**encoded).logits[:, -1, :].to(torch.float32)
    delta = biased - plain
    target_delta = float(delta[0, 51378])
    delta[0, 51378] = 0
    if target_delta < 4.5 or float(delta.abs().max()) != 0.0:
        raise ValueError("Loaded custom class does not isolate the UNKNOWN-logit bias")


def recover() -> None:
    if not TRANSACTION_JOURNAL.is_file():
        if not TRANSACTION_LOCK.is_file():
            raise FileNotFoundError("No unfinished Supa-A1 transaction exists")
        lock = load_json(TRANSACTION_LOCK)
        owner_pid = int(lock.get("owner_pid", -1))
        if process_is_alive(owner_pid):
            raise RuntimeError(f"Supa-A1 transaction owner PID {owner_pid} is alive")
        TRANSACTION_LOCK.unlink()
        return
    journal = load_json(TRANSACTION_JOURNAL)
    if not TRANSACTION_LOCK.is_file():
        raise RuntimeError("Transaction journal exists without its ownership lock")
    lock = load_json(TRANSACTION_LOCK)
    transaction_id = journal.get("transaction_id")
    owner_pid = int(journal.get("owner_pid", -1))
    if (
        not isinstance(transaction_id, str)
        or lock.get("transaction_id") != transaction_id
        or int(lock.get("owner_pid", -2)) != owner_pid
    ):
        raise RuntimeError("Transaction journal and ownership lock do not match")
    if process_is_alive(owner_pid):
        raise RuntimeError(f"Supa-A1 transaction owner PID {owner_pid} is alive")
    allowed_root = VAULT.resolve()
    destination = Path(journal["destination"]).resolve()
    staging = Path(journal["staging"]).resolve()
    backup = Path(journal["catalog_backup"]).resolve()
    runtime = Path(journal["runtime_module"]).resolve()
    if destination != DESTINATION.resolve():
        raise ValueError(f"Journal has an unexpected destination: {destination}")
    if runtime != ROOT_RUNTIME_MODULE.resolve():
        raise ValueError(f"Journal has an unexpected runtime module: {runtime}")
    if staging.parent != allowed_root or not staging.name.startswith(
        f".{DESTINATION_NAME}.staging-"
    ):
        raise ValueError(f"Journal has an unsafe staging path: {staging}")
    if backup.parent != allowed_root or not backup.name.startswith(
        f".{CATALOG.name}.backup-"
    ):
        raise ValueError(f"Journal has an unsafe catalog backup: {backup}")
    original_catalog_sha256 = journal.get("original_catalog_sha256")
    new_catalog_sha256 = journal.get("new_catalog_sha256")
    if (
        not isinstance(original_catalog_sha256, str)
        or len(original_catalog_sha256) != 64
    ):
        raise ValueError("Journal has no valid original catalog hash")
    if new_catalog_sha256 is not None and (
        not isinstance(new_catalog_sha256, str) or len(new_catalog_sha256) != 64
    ):
        raise ValueError("Journal has an invalid new catalog hash")
    restore_catalog_or_fail(backup, original_catalog_sha256, new_catalog_sha256)
    if bool(journal.get("runtime_was_absent")) and runtime.is_file():
        runtime.unlink()
    if destination.is_dir():
        shutil.rmtree(destination)
    if staging.is_dir():
        shutil.rmtree(staging)
    TRANSACTION_JOURNAL.unlink()
    if TRANSACTION_LOCK.exists():
        TRANSACTION_LOCK.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "apply", "verify", "recover"))
    parser.add_argument("--smoke-load", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        validate_sources()
    elif args.mode == "apply":
        apply()
    elif args.mode == "verify":
        verify()
    else:
        recover()
    if args.smoke_load and args.mode != "recover":
        smoke_load()
    print(json.dumps({"mode": args.mode, "core_id": CORE_ID, "ok": True}, indent=2))


if __name__ == "__main__":
    main()
