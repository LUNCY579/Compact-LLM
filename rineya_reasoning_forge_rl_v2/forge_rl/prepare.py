"""Optional exact-local-tokenizer qualification, never truncation or training.

Native tokenization has NOT been run for this release. Unit tests exercise this
boundary with stubs. Use the precise checkpoint tokenizer and chat template.
"""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
from typing import Any
from .common import read_jsonl, Writer, digest, sha_file, loads, write_json, dumps, integer


def tokenizer_inventory(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError("Tokenizer must be an existing local directory")
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model", "sentencepiece.bpe.model", "config.json"}
    files = {str(p.relative_to(directory)).replace("\\", "/"): sha_file(p)
             for p in sorted(directory.rglob("*")) if p.is_file() and (p.name in names or p.suffix == ".jinja")}
    if not files:
        raise ValueError("No recognized tokenizer/template assets found")
    return {"files": files, "sha256": digest(files)}


def prepare_record(row: dict[str, Any], tokenizer: Any, *, kind: str, max_length: int,
                   completion_reserve: int, terminator_id: int | None,
                   template_kwargs: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if kind not in ("rl", "sft") or not integer(max_length) or max_length <= 0:
        raise ValueError("Invalid preparation configuration")
    if not integer(completion_reserve) or completion_reserve < 1:
        raise ValueError("A positive generation reserve is required")
    kwargs = dict(template_kwargs or {})
    if set(kwargs) & {"tokenize", "add_generation_prompt", "return_tensors", "return_dict", "truncation", "max_length"}:
        raise ValueError("Template kwargs cannot override tokenization or safety gates")
    if kind == "sft" and row.get("split") not in ("train", "validation"):
        raise ValueError("Held-out SFT preparation is forbidden")
    messages = row["prompt"] if kind == "rl" else row["messages"]
    if kind == "sft" and (len(messages) != 3 or messages[-1].get("role") != "assistant"):
        raise ValueError("Only single-answer SFT examples are supported")
    prefix_messages = messages if kind == "rl" else messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(prefix_messages, tokenize=True, add_generation_prompt=True, **kwargs)
    if not isinstance(prompt_ids, list) or not prompt_ids or any(not integer(x) or x < 0 for x in prompt_ids):
        raise ValueError("Tokenizer must return one nonempty list of integer token IDs")
    if kind == "rl":
        report = {"id": row["id"], "prompt_tokens": len(prompt_ids), "completion_reserve": completion_reserve,
                  "required_context": len(prompt_ids) + completion_reserve, "truncated": False}
        if report["required_context"] > max_length:
            return None, {**report, "status": "OVERLENGTH_EXCLUDED"}
        return {"id": row["id"], "group_id": row["group_id"], "input_ids": prompt_ids,
                "attention_mask": [1]*len(prompt_ids)}, {**report, "status": "PASS"}
    if not integer(terminator_id) or terminator_id < 0:
        raise ValueError("SFT requires the exact native assistant turn-terminator ID")
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **kwargs)
    if not isinstance(full, list) or any(not integer(x) or x < 0 for x in full):
        raise ValueError("Invalid full token sequence")
    if full[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Native prompt/full prefix mismatch; manual loss-boundary qualification required")
    suffix = full[len(prompt_ids):]
    if len(suffix) < 2 or terminator_id not in suffix or suffix.index(terminator_id) == 0:
        raise ValueError("No answer followed by the declared turn terminator")
    if suffix.count(terminator_id) != 1 or suffix.index(terminator_id) < len(suffix) - 8:
        raise ValueError("Turn terminator is repeated or unexpectedly far from the end")
    report = {"id": row["id"], "prompt_tokens": len(prompt_ids), "full_tokens": len(full),
              "loss_tokens": len(suffix), "truncated": False, "prefix_parity": True, "terminator_id": terminator_id}
    if len(full) > max_length:
        return None, {**report, "status": "OVERLENGTH_EXCLUDED"}
    labels = [-100]*len(prompt_ids) + suffix
    if all(x == -100 for x in labels):
        raise ValueError("No supervised tokens")
    return {"id": row["id"], "group_id": row["group_id"], "input_ids": full,
            "attention_mask": [1]*len(full), "labels": labels}, {**report, "status": "PASS"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True); p.add_argument("--expected-tokenizer-sha256")
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--kind", choices=["rl", "sft"], required=True)
    p.add_argument("--max-length", type=int, required=True); p.add_argument("--completion-reserve", type=int, default=512)
    p.add_argument("--terminator-id", type=int); p.add_argument("--template-kwargs", type=Path)
    a = p.parse_args()
    if a.output.exists() or Path(str(a.output) + ".report.json").exists():
        raise FileExistsError("Refusing to overwrite prepared output/report")
    assets = tokenizer_inventory(a.tokenizer)
    if a.expected_tokenizer_sha256 and assets["sha256"] != a.expected_tokenizer_sha256:
        raise ValueError("Tokenizer assets do not match the expected hash")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Transformers is optional and absent. Run this utility in your separately qualified native model environment.") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True, trust_remote_code=False)
    kwargs = loads(a.template_kwargs.read_text(encoding="utf-8")) if a.template_kwargs else {}
    reports = []; accepted = 0
    try:
        with Writer(a.output) as writer:
            for row in read_jsonl(a.input):
                prepared, report = prepare_record(row, tokenizer, kind=a.kind, max_length=a.max_length,
                                                  completion_reserve=a.completion_reserve, terminator_id=a.terminator_id,
                                                  template_kwargs=kwargs)
                reports.append(report)
                if prepared is not None:
                    writer.write(prepared); accepted += 1
        if accepted == 0:
            raise ValueError("No fitting qualified examples; output removed")
    except BaseException:
        a.output.unlink(missing_ok=True)
        raise
    write_json(Path(str(a.output) + ".report.json"), {"checkpoint_label_unverified": a.checkpoint_label,
               "input_sha256": sha_file(a.input), "tokenizer_assets": assets,
               "transformers_version": importlib.metadata.version("transformers"), "template_kwargs": kwargs,
               "max_length": a.max_length, "accepted": accepted, "excluded_overlength": len(reports)-accepted,
               "rows": reports, "model_weights_loaded": False, "training_started": False,
               "attention_mask_scope": "ordinary token-presence mask; not a custom architecture attention mask"})
    print(dumps({"accepted": accepted, "excluded": len(reports)-accepted}))


if __name__ == "__main__":
    main()
