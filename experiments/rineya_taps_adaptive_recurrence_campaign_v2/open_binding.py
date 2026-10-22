"""No-model tokenizer and shape binding for the exact reviewed OPEN packet."""
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
from .open_bank import solve_prompt, score_response

PACKAGE = Path(__file__).resolve().parent
PACKET = PACKAGE / "runs/open-bank-apparatus-002"


def run(source, out):
    source, out = source.resolve(strict=True), out.resolve()
    if PACKAGE not in out.parents:
        raise PermissionError("binding output must be campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FAIL", "native_model_loads": 0, "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        historical, inherited = admitted_source(source)
        manifest = json.loads((PACKET / "manifest.json").read_text())
        if sha(PACKAGE / "open_bank.py") != manifest["source_sha256"]:
            raise RuntimeError("OPEN generator/scorer source changed")
        for name, expected in manifest["files"].items():
            if Path(name).name != name or sha(PACKET / name) != expected:
                raise RuntimeError("OPEN source-bound packet changed")
        bank = json.loads((PACKET / "bank.json").read_text())
        if len(bank) != 24 or len({row["id"] for row in bank}) != 24:
            raise RuntimeError("OPEN complete bank count/identity changed")
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        from transformers import AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"]
                or importlib.metadata.version("torch") != historical["torch"]
                or importlib.metadata.version("transformers") != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("OPEN tokenizer exact-runtime binding differs")
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, local_files_only=True, cache_dir=cache / "hub")
        rows = []
        for row in bank:
            if row["provenance"]["historical_parents"] or row["shape"]["max_output_tokens"] != 8:
                raise RuntimeError("OPEN provenance/shape authority differs")
            prediction = solve_prompt(row["prompt"])
            result = score_response(prediction, row)
            if result["correct"] is not True:
                raise RuntimeError("OPEN independent prompt-only solver/gold mismatch")
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(rendered, return_tensors="pt")["input_ids"][0].tolist()
            if not ids or len(ids) > 350:
                raise RuntimeError("OPEN measured input shape exceeds prospectively allowed350; no truncation")
            rows.append({"case_id": row["id"], "parent_id": row["parent_id"], "split": row["split"],
                         "generator_family": row["generator_family"], "category": row["category"],
                         "raw_prompt": row["prompt"], "rendered_prompt": rendered, "input_token_ids": ids,
                         "input_tokens": len(ids), "max_new_tokens": 8,
                         "max_forward_prefix_tokens": len(ids) + 7,
                         "max_processed_prefix_positions": 8 * len(ids) + 28,
                         "machine_solver_verified": True, "human_review": "NOT_REVIEWED"})
        binding = {"status": "OPEN_TOKENIZER_BOUND_NATIVE_NOT_MEASURED", "bank_path": str(PACKET / "bank.json"),
                   "bank_sha256": sha(PACKET / "bank.json"), "manifest_sha256": sha(PACKET / "manifest.json"),
                   "generator_scorer_sha256": sha(PACKAGE / "open_bank.py"), "binding_source_sha256": sha(Path(__file__)),
                   "canonical_source": str(source), "checkpoint_files": historical["checkpoint_files_before"],
                   "inherited_source_sha256": inherited,
                   "runtime": {"python": platform.python_version(), "torch": importlib.metadata.version("torch"),
                               "transformers": importlib.metadata.version("transformers"), "interpreter": sys.executable,
                               "qwen_sha256": sha(inspect.getfile(Qwen3DecoderLayer))},
                   "tokenizer_class": type(tokenizer).__name__, "tokenizer_length": len(tokenizer),
                   "shapes": {"batch": 1, "padding": False, "cache": False,
                              "max_input_tokens": max(row["input_tokens"] for row in rows),
                              "max_output_tokens": 8, "max_forward_prefix_tokens": max(row["max_forward_prefix_tokens"] for row in rows)},
                   "rows": rows, "native_model_loads": 0, "independent_confirmation": False,
                   "exposure": "author-visible OPEN development/validation; no model outcomes or policy fitting on bank yet"}
        with (out / "OPEN_TOKENIZER_BINDING.json").open("x", encoding="utf-8") as f:
            json.dump(binding, f, indent=2, ensure_ascii=False)
            f.write("\n")
        receipt.update(status=binding["status"], binding_sha256=sha(out / "OPEN_TOKENIZER_BINDING.json"), shapes=binding["shapes"])
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
        with (out / "receipt.json").open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "OPEN_TOKENIZER_BOUND_NATIVE_NOT_MEASURED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out))
