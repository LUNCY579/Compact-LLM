"""Measure the authored open DEV/CAL candidates with Titan's tokenizer only.

This preparation module sets framework-off guards through tokenize_bridge. Never
import it in a native model worker. Counts and arrays are evaluation inputs, not
reference-label qualification or evidence of a generated student response.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from . import tokenize_bridge as common
from . import e1_data as data

PROMPT_CAP = 896
TARGET_CAP = 128
STATUS = "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED"


def references(row):
    common.require(row.get("role") in ("DEV", "CAL"), "open DEV/CAL only")
    if row["role"] == "DEV":
        common.require(type(row.get("dev_exemplar")) is str and row["dev_exemplar"], "DEV exemplar missing")
        common.require(row.get("cal_controls") == [], "DEV cannot contain CAL controls")
        return [("dev_exemplar", row["dev_exemplar"])]
    common.require(row.get("dev_exemplar") is None, "CAL has no DEV exemplar")
    controls = row.get("cal_controls")
    expected = {"positive_a", "positive_b", "adequacy", "relevance", "fluency", "role_consistency"}
    common.require(type(controls) is list and len(controls) == 6 and
                   {x["control_id"] for x in controls} == expected, "CAL control inventory")
    common.require(all(x.get("qualified_reference_scores") is None for x in controls), "unqualified controls only")
    return [(x["control_id"], x["response"]) for x in controls]


def tokenize_world(row, tokenizer):
    refs = references(row)
    messages = row.get("messages")
    common.require(type(messages) is list and messages and len(messages) % 2 == 1, "conversation boundary")
    common.require(all(type(m) is dict and set(m) == {"role", "content"} and
                       m["role"] == ("user" if i % 2 == 0 else "assistant") and
                       type(m["content"]) is str and m["content"]
                       for i, m in enumerate(messages)), "conversation roles/content")
    common.require(tokenizer.eos_token_id == common.EOS_ID and tokenizer.eos_token == "<|im_end|>", "EOS identity")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **common.TEMPLATE_KWARGS)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **common.TEMPLATE_KWARGS)
    common.require(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "generation prefix drift")
    common.require(ids == tokenizer.encode(prompt, add_special_tokens=False), "prompt tokenization drift")
    common.require(0 < len(ids) <= PROMPT_CAP, "prompt over cap; no truncation")
    measured = []
    for reference_id, response in refs:
        common.require(type(response) is str and response, "empty reference/control")
        target = tokenizer.encode(response, add_special_tokens=False)
        common.require(not set(target).intersection((151643, 151644, 151645)), "embedded target control token")
        completion = target + [common.EOS_ID]
        common.require(len(completion) <= TARGET_CAP, "reference/control over cap; no truncation")
        common.require(tokenizer.decode(completion, skip_special_tokens=False) == response + tokenizer.eos_token,
                       "reference/control roundtrip drift")
        whole = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        common.require(whole == prompt + response + tokenizer.eos_token + "\n", "full template boundary drift")
        whole_ids = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": response}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False)
        common.require(whole_ids == ids + completion + tokenizer.encode("\n", add_special_tokens=False),
                       "full template token prefix drift")
        measured.append({"reference_id": reference_id, "response_sha256": data.hashlib.sha256(response.encode()).hexdigest(),
                         "completion_token_ids": completion, "completion_tokens_including_EOS": len(completion),
                         "completion_token_ids_sha256": data.digest(completion), "qualified_reference_labels": False})
    return {"parent_id": row["parent_id"], "role": row["role"], "stratum": row["stratum"],
            "source_row_sha256": data.digest(row), "messages_sha256": data.digest(messages),
            "prompt_token_ids": ids, "prompt_tokens": len(ids), "prompt_token_ids_sha256": data.digest(ids),
            "prior_authored_assistant_turns": sum(m["role"] == "assistant" for m in messages),
            "references": measured, "truncation": False, "native_qualified": False,
            "training_eligibility": False, "e1_launch_ready": False}


def profile(bank, tokenizer_receipt, output):
    started = time.perf_counter()
    common.require(not output.exists(), "write-once output exists")
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    common.require(tuple(sys.version_info[:3]) == (3, 11, 15) and
                   Path(sys.executable).resolve() == (root.parent / ".venv-mq001/Scripts/python.exe").resolve(),
                   "pinned Python 3.11.15 interpreter required before tokenizer import")
    bank = data.safe_path(root, bank.resolve().relative_to(root).as_posix())
    common.require(tokenizer_receipt.resolve() == package / "receipts/TOKENIZER_RECEIPT.json", "fixed tokenizer receipt required")
    def bank_file(name):
        return data.safe_path(root, (bank / name).relative_to(root).as_posix())
    receipt, receipt_id = data.read_json(bank_file("DATA_RECEIPT.json"))
    common.require(receipt["status"] == STATUS, "unbound candidate bank")
    source_files = {}
    rows_by_role = {}
    for role, count in (("DEV", 64), ("CAL", 16)):
        path = bank_file(role + ".jsonl")
        common.require(data.identity(path) == receipt["files"][path.name], "bank file drift")
        raw = path.read_bytes()
        common.require({"sha256": data.hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == receipt["files"][path.name],
                       "bank changed while reading")
        rows = [data.parse(line) for line in raw.splitlines()]
        common.require(len(rows) == count and all(r["role"] == role for r in rows), "population count/role drift")
        common.require(len({r["parent_id"] for r in rows}) == count, "duplicate parent")
        rows_by_role[role] = rows
        source_files[path.name] = receipt["files"][path.name]
    common.require(len({r["parent_id"] for rs in rows_by_role.values() for r in rs}) == 80, "cross-role parent reuse")
    pin, pin_id = data.read_json(tokenizer_receipt)
    model_dir = Path(pin["checkpoint_path"])
    assets = {name: common.file_identity(model_dir / name) for name in common.TOKENIZER_FILES}
    common.require(assets == pin["tokenizer_files"], "tokenizer asset drift")
    template = common.read_json(model_dir / "tokenizer_config.json")["chat_template"]
    common.require(data.hashlib.sha256(template.encode()).hexdigest() == pin["chat_template_sha256"], "template drift")
    tokenizer = common.load_tokenizer(model_dir)
    common.require(tokenizer.chat_template == template, "runtime template drift")
    measured = {role: [tokenize_world(r, tokenizer) for r in rows] for role, rows in rows_by_role.items()}
    common.require({name: common.file_identity(model_dir / name) for name in common.TOKENIZER_FILES} == assets, "tokenizer changed")
    common.require(all(data.identity(bank_file(name)) == ident for name, ident in source_files.items()), "bank changed during profile")
    common.require("torch" not in sys.modules, "unexpected Torch import")
    output.mkdir(parents=True)
    for role, rows in measured.items():
        with (output / (role + ".prompts.jsonl")).open("xb") as f:
            for row in rows:
                f.write(data.canonical(row) + b"\n")
    prompts = [r["prompt_tokens"] for rs in measured.values() for r in rs]
    targets = [t["completion_tokens_including_EOS"] for rs in measured.values() for r in rs for t in r["references"]]
    result = {"status": "MEASURED_LANGUAGE_DEV_CAL_TOKENIZER_ONLY_UNQUALIFIED_GRADING",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "bank_receipt": receipt_id,
        "source_files": source_files, "tokenizer_receipt": pin_id, "tokenizer_assets": assets,
        "chat_template_sha256": pin["chat_template_sha256"], "template_kwargs": common.TEMPLATE_KWARGS,
        "script": data.identity(Path(__file__)), "tokenizer_helper": data.identity(Path(common.__file__)),
        "python_executable": sys.executable, "python_version": list(sys.version_info[:3]),
        "parents": {role: len(rows) for role, rows in measured.items()}, "prompt_tokens_total": sum(prompts),
        "prompt_tokens": common.distribution(prompts), "reference_control_tokens": common.distribution(targets),
        "reference_control_count": len(targets), "reference_control_tokens_total": sum(targets),
        "prior_authored_assistant_turns": sum(r["prior_authored_assistant_turns"] for rs in measured.values() for r in rs),
        "files": {f.name: data.identity(f) for f in output.iterdir()}, "truncation": False,
        "qualified_reference_labels": False, "qualified_graders": 0, "native_qualified": False,
        "training_eligibility": False, "e1_launch_ready": False, "CHECK_FINAL_read": False,
        "torch_imported": False, "cost": {"wall_seconds": time.perf_counter() - started,
        "model_calls": 0, "teacher_calls": 0, "training_tokens": 0, "native_generated_tokens": 0}}
    with (output / "TOKENIZATION_RECEIPT.json").open("xb") as f:
        f.write(data.canonical(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("status", "parents", "prompt_tokens", "reference_control_tokens")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.bank, args.tokenizer_receipt, args.output)
