"""Offline Titan token profiling; no model weights or model frameworks loaded."""
import os
for name, value in {"USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[name] = value
import argparse
from collections import Counter
import hashlib
import importlib.metadata
from pathlib import Path
import sys

from common import ROOT, canonical, child_path, digest, file_digest, strict_json, write_json, write_rows, source_identity
from pipeline import load_batch

def profile(batch, out):
    batch, manifest, rows = load_batch(batch)
    out = child_path(out)
    if out.exists():
        raise ValueError("profile exists; use a new version")
    policy = strict_json((ROOT / "config/quality_policy.json").read_text(encoding="utf-8"))
    model = Path(policy["tokenizer_path"])
    before = {}
    for filename, expected in [("tokenizer.json", policy["tokenizer_sha256"]),
                               ("tokenizer_config.json", policy["tokenizer_config_sha256"])]:
        before[filename] = file_digest(model / filename)
        if before[filename] != expected:
            raise ValueError("tokenizer identity mismatch: " + filename)
    config = strict_json((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    if hashlib.sha256(config["chat_template"].encode()).hexdigest() != policy["chat_template_sha256"]:
        raise ValueError("template mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    if "torch" in sys.modules or "tensorflow" in sys.modules or "flax" in sys.modules:
        raise ValueError("model backend imported")
    if tokenizer.eos_token_id != policy["eos_token_id"] or tokenizer.eos_token != "<|im_end|>":
        raise ValueError("EOS mismatch")
    measured = []
    for row in rows:
        messages = row["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        if prompt != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("rendered prompt differs")
        target = tokenizer.encode(row["target"], add_special_tokens=False)
        if tokenizer.eos_token_id in target:
            raise ValueError("target contains EOS")
        completion = target + [tokenizer.eos_token_id]
        if tokenizer.decode(completion, skip_special_tokens=False) != row["target"] + tokenizer.eos_token:
            raise ValueError("target roundtrip mismatch")
        full_messages = messages + [{"role": "assistant", "content": row["target"]}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
        if full_text != text + row["target"] + tokenizer.eos_token + "\n":
            raise ValueError("assistant template boundary mismatch")
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        tail = tokenizer.encode("\n", add_special_tokens=False)
        if full_ids != prompt + completion + tail:
            raise ValueError("completion boundary mismatch")
        if len(completion) > policy["max_completion_tokens_including_eos"]:
            raise ValueError("completion exceeds frozen budget")
        ids = prompt + completion
        labels = [-100] * len(prompt) + completion
        if len(ids) != len(labels) or sum(x != -100 for x in labels) != len(completion):
            raise ValueError("mask mismatch")
        record = {"row_id": row["row_id"], "row_sha256": row["row_sha256"],
                  "family": row["family"], "split": row["split"], "profile": row["profile"],
                  "prompt_tokens": len(prompt), "completion_tokens_including_eos": len(completion),
                  "total_tokens": len(ids), "reserved_response_budget": 128,
                  "ce512_length_fit_only": len(ids) <= 512,
                  "profile_bucket": next((n for n in [512,1024,2048,4096,8192,16384,32768,40960] if len(ids) <= n), "over_declared"),
                  "prompt_ids_sha256": digest(prompt), "completion_ids_sha256": digest(completion),
                  "input_ids_sha256": digest(ids), "labels_sha256": digest(labels),
                  "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "padding_tokens": 0, "packing": False, "truncated": False,
                  "native_training_eligible": False}
        measured.append(record)
    for filename, expected in before.items():
        if file_digest(model / filename) != expected:
            raise ValueError("tokenizer changed during profiling")
    write_rows(out / "rows.jsonl", measured)
    summary = {"schema_version": 1, "status": "PASS_TOKENIZATION_ONLY",
               "canonical_sha256": manifest["canonical_sha256"], "rows": len(rows),
               "rows_sha256": file_digest(out / "rows.jsonl"), "source_identity": source_identity(),
               "policy_sha256": file_digest(ROOT / "config/quality_policy.json"),
               "tokenizer_files": before, "chat_template_sha256": policy["chat_template_sha256"],
               "runtime": {k: importlib.metadata.version(k) for k in ["transformers", "tokenizers"]},
               "python": sys.version, "model_framework_imported": False, "model_weights_read": False,
               "template_args": {"enable_thinking": False, "add_generation_prompt": True},
               "ce512_length_fit_rows": sum(r["ce512_length_fit_only"] for r in measured),
               "length_bucket_counts": dict(Counter(str(r["profile_bucket"]) for r in measured)),
               "minimum_total_tokens": min(r["total_tokens"] for r in measured),
               "maximum_total_tokens": max(r["total_tokens"] for r in measured),
               "input_tokens": sum(r["total_tokens"] for r in measured),
               "supervised_tokens": sum(r["completion_tokens_including_eos"] for r in measured),
               "native_long_context_qualified": False, "training_ready": False}
    write_json(out / "summary.json", summary)
    print(canonical(summary))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", required=True); p.add_argument("--out", required=True)
    profile(**vars(p.parse_args()))
