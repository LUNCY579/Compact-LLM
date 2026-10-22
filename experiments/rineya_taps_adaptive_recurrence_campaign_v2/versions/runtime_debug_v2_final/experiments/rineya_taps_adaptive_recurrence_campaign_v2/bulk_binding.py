"""No-model tokenizer/shape binding for the larger reviewed prospective bank."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys

from .binding import admitted_source, sha

PACKAGE = Path(__file__).resolve().parent
PACKET = PACKAGE / "runs/bulk-bank-apparatus-001"


def verify_packet(packet):
    packet = packet.resolve(strict=True)
    if PACKAGE not in packet.parents or packet != PACKET.resolve(strict=True):
        raise PermissionError("only the exact larger campaign bank is admitted")
    manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("generator_version") != "BULK-TAPS-001" or manifest.get("training") is not False:
        raise RuntimeError("larger bank provenance differs")
    if sha(PACKAGE / "bulk_bank.py") != manifest["source_sha256"]:
        raise RuntimeError("larger generator/scorer changed after export")
    if sha(PACKAGE / "open_bank.py") != manifest.get("normalizer_source_sha256"):
        raise RuntimeError("larger imported strict normalizer changed after export")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha(packet / name) != expected:
            raise RuntimeError("larger source-bound packet changed: " + name)
    from . import bulk_bank
    bank = json.loads((packet / "bank.json").read_text(encoding="utf-8"))
    if len(bank) != 288 or len({row["id"] for row in bank}) != 288 or len({row["parent_id"] for row in bank}) != 144:
        raise RuntimeError("larger complete case/parent identity differs")
    dev = [row for row in bank if row["split"] == "DEVELOPMENT"]
    validation = [row for row in bank if row["split"] == "OPEN_VALIDATION"]
    if len(dev) != 192 or len(validation) != 96:
        raise RuntimeError("larger prospective split count differs")
    if {row["parent_id"] for row in dev} & {row["parent_id"] for row in validation}:
        raise PermissionError("larger parent worlds cross selection/validation")
    for row in bank:
        if row["provenance"].get("historical_parents") != [] or row["shape"]["max_output_tokens"] != 8:
            raise PermissionError("larger shape/protected-parent authority differs")
        if not bulk_bank.score_response(bulk_bank.solve_prompt(row["prompt"]), row)["correct"]:
            raise RuntimeError("larger prompt-only solver/gold mismatch")
    return bank, manifest


def run(source, out):
    source, out = source.resolve(strict=True), out.resolve()
    if PACKAGE not in out.parents:
        raise PermissionError("binding output must be campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "native_model_loads": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        historical, inherited = admitted_source(source)
        bank, manifest = verify_packet(PACKET)
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        from transformers import AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"]
                or importlib.metadata.version("torch") != historical["torch"]
                or importlib.metadata.version("transformers") != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("larger tokenizer exact-runtime binding differs")
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, local_files_only=True, cache_dir=cache / "hub")
        rows = []
        for row in bank:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist()
            if not ids or len(ids) > 350:
                raise RuntimeError("larger measured input exceeds planned350; no truncation")
            rows.append({"case_id": row["id"], "parent_id": row["parent_id"], "split": row["split"],
                "generator_family": row["generator_family"], "category": row["category"],
                "difficulty_stratum": row["difficulty_stratum"], "variant_id": row["variant_id"],
                "raw_prompt": row["prompt"], "rendered_prompt": rendered, "input_token_ids": ids,
                "input_tokens": len(ids), "max_new_tokens": 8, "max_forward_prefix_tokens": len(ids) + 7,
                "max_processed_prefix_positions": 8 * len(ids) + 28,
                "machine_solver_verified": True, "human_review": "NOT_REVIEWED"})
        binding = {"status": "BULK_TOKENIZER_BOUND_NATIVE_NOT_MEASURED", "bank_path": str(PACKET / "bank.json"),
            "bank_sha256": sha(PACKET / "bank.json"), "manifest_sha256": sha(PACKET / "manifest.json"),
            "generator_scorer_sha256": sha(PACKAGE / "bulk_bank.py"), "binding_source_sha256": sha(Path(__file__)),
            "normalizer_source_sha256": sha(PACKAGE / "open_bank.py"),
            "canonical_source": str(source), "checkpoint_files": historical["checkpoint_files_before"],
            "inherited_source_sha256": inherited,
            "runtime": {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"), "interpreter": sys.executable,
                "qwen_sha256": sha(inspect.getfile(Qwen3DecoderLayer))},
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_length": len(tokenizer),
            "shapes": {"batch": 1, "padding": False, "cache": False,
                "max_input_tokens": max(row["input_tokens"] for row in rows), "max_output_tokens": 8,
                "max_forward_prefix_tokens": max(row["max_forward_prefix_tokens"] for row in rows)},
            "world_parents": 144, "development_parents": 96, "validation_parents": 48,
            "partition": "parent-world holdout; shared generators/authors; no held-out generator claim",
            "rows": rows, "native_model_loads": 0, "independent_confirmation": False,
            "exposure": "author-visible prospective larger OPEN screen; no native outcomes or policy fitting"}
        with (out / "BULK_TOKENIZER_BINDING.json").open("x", encoding="utf-8") as handle:
            json.dump(binding, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        receipt.update(status=binding["status"], binding_sha256=sha(out / "BULK_TOKENIZER_BINDING.json"), shapes=binding["shapes"])
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out / "receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
            handle.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "BULK_TOKENIZER_BOUND_NATIVE_NOT_MEASURED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out))
