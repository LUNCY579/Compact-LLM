"""Torch/Transformers-free Qwen3 training serialization and token checks.

Tokenizers is imported only when an explicitly pinned tokenizer is loaded.
The byte renderer is fixed for system/user messages and thinking disabled.
Full Hugging Face AutoTokenizer/template equivalence remains unqualified here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import schema

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
CONTROL_MARKERS = (IM_START, IM_END, "<|endoftext|>", "<think>", "</think>")
IGNORE_INDEX = -100
MAX_PROMPT_TOKENS = schema.MAX_SEQUENCE_LENGTH - schema.MAX_OUTPUT_TOKENS


def render_prompt(system: str, user: str) -> str:
    if not isinstance(system, str) or not isinstance(user, str) or not system.strip() or not user.strip():
        raise ValueError("STOP_EMPTY_OR_NONSTRING_CHAT_INPUT")
    if any(marker in value for value in (system, user) for marker in CONTROL_MARKERS):
        raise ValueError("STOP_EMBEDDED_CHAT_CONTROL_MARKER")
    return (f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{user}{IM_END}\n"
            f"{IM_START}assistant\n<think>\n\n</think>\n\n")


class PinnedTokenizer:
    def __init__(self, backend, path: Path, digest: str):
        self.backend = backend
        self.binding = {"tokenizer_json": str(path), "sha256": digest,
                        "loader": "tokenizers.Tokenizer.from_str", "transformers_imported": False}
    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens)
    def token_to_id(self, token):
        return self.backend.token_to_id(token)
    @property
    def truncation(self):
        return self.backend.truncation
    @property
    def padding(self):
        return self.backend.padding


def load_tokenizer(path: Path, expected_sha256: str) -> PinnedTokenizer:
    """Read/hash/parse the exact same tokenizer bytes; never downloads assets."""
    path = Path(path).resolve()
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or actual != expected_sha256:
        raise ValueError("STOP_TOKENIZER_DIGEST_MISMATCH")
    from tokenizers import Tokenizer
    return PinnedTokenizer(Tokenizer.from_str(source.decode("utf-8")), path, actual)


def _ids(tokenizer, text: str) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=False)
    ids = list(value.ids if hasattr(value, "ids") else value)
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("STOP_EMPTY_OR_INVALID_TOKEN_IDS")
    return ids


def _target(row: dict, target_fn=None) -> str:
    if target_fn is None:
        from .generator import target_reply
        target_fn = target_reply
    target = target_fn(row)
    if not isinstance(target, str) or not target.strip():
        raise ValueError("STOP_EMPTY_OR_NONSTRING_TRAINING_TARGET")
    if any(marker in target for marker in CONTROL_MARKERS):
        raise ValueError("STOP_TARGET_HAS_CHAT_CONTROL_MARKER")
    return target


def encode_training_row(row: dict, tokenizer, *, target_fn=None) -> dict:
    schema.validate_row(row)
    if getattr(tokenizer, "truncation", None) is not None:
        raise ValueError("STOP_TOKENIZER_TRUNCATION_ENABLED")
    if getattr(tokenizer, "padding", None) is not None:
        raise ValueError("STOP_TOKENIZER_PADDING_ENABLED")
    eos = tokenizer.token_to_id(IM_END)
    if type(eos) is not int or eos < 0:
        raise ValueError("STOP_QWEN3_EOS_UNAVAILABLE")
    prompt, target = render_prompt(row["system"], row["user"]), _target(row, target_fn)
    prompt_ids = _ids(tokenizer, prompt)
    # Tokenize the full concatenation. Never assume separately encoded prompt
    # and answer concatenate without a BPE merge at the supervision boundary.
    full_ids = _ids(tokenizer, prompt + target + IM_END)
    count = len(prompt_ids)
    if full_ids[:count] != prompt_ids:
        raise ValueError("STOP_PROMPT_TARGET_TOKEN_BOUNDARY_MERGE")
    labels = [IGNORE_INDEX] * count + full_ids[count:]
    target_ids = labels[count:]
    if not target_ids or not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("STOP_ALL_MASKED_OR_EMPTY_TARGET")
    if full_ids[-1] != eos or target_ids[-1] != eos:
        raise ValueError("STOP_TARGET_EOS_MISSING_OR_TRUNCATED")
    # Only the final target EOS is permitted: an extra EOS inside the target is
    # not harmless padding or a shortened answer. Prompt role EOS is expected.
    if eos in target_ids[:-1]:
        raise ValueError("STOP_EARLY_TARGET_EOS")
    if count > MAX_PROMPT_TOKENS:
        raise ValueError("STOP_PROMPT_TOKEN_CAP")
    if len(target_ids) > schema.MAX_OUTPUT_TOKENS:
        raise ValueError("STOP_TARGET_TOKEN_CAP_NO_TRUNCATION")
    if len(full_ids) > schema.MAX_SEQUENCE_LENGTH:
        raise ValueError("STOP_SEQUENCE_TOKEN_CAP_NO_TRUNCATION")
    if len(labels) != len(full_ids) or any(label != IGNORE_INDEX for label in labels[:count]):
        raise AssertionError("STOP_PROMPT_LABEL_MASK")
    return {"row_id": row["row_id"], "world_id": row["world_id"], "split": row["split"],
            "family": row["family"], "variant_id": row["variant_id"],
            "input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids),
            "prompt_tokens": count, "target_tokens_including_eos": len(target_ids),
            "sequence_tokens": len(full_ids), "loss_tokens": len(target_ids),
            "eos_token_id": eos, "final_eos_label_positive": True,
            "serialization_sha256": schema.digest({"input_ids": full_ids, "labels": labels}),
            "thinking_enabled": False, "target_truncation": False, "all_masked": False}


def audit_rows(rows, tokenizer, *, target_fn=None) -> dict:
    """Return counts/digests; do not retain the full 2000-row tensor payload.

    Any bad row aborts the complete audit. A failure is never permission to drop
    or shorten that row; the root supervisor retains its blocked/error receipt.
    """
    summaries, ids, total_prompt, total_loss = [], set(), 0, 0
    family_mass, split_mass = {}, {}
    for row in rows:
        if row.get("row_id") in ids:
            raise ValueError("STOP_DUPLICATE_TOKEN_AUDIT_ROW_ID")
        encoded = encode_training_row(row, tokenizer, target_fn=target_fn)
        ids.add(row["row_id"])
        summary = {key: value for key, value in encoded.items() if key not in {"input_ids", "labels", "attention_mask"}}
        summaries.append(summary)
        total_prompt += encoded["prompt_tokens"]
        total_loss += encoded["loss_tokens"]
        family_mass[row["family"]] = family_mass.get(row["family"], 0) + encoded["loss_tokens"]
        split_mass[row["split"]] = split_mass.get(row["split"], 0) + encoded["loss_tokens"]
    if not summaries:
        raise ValueError("STOP_EMPTY_TOKEN_AUDIT")
    return {"experiment": schema.EXPERIMENT, "passed": True, "rows": len(summaries),
            "row_audits": summaries, "total_prompt_tokens": total_prompt, "total_loss_tokens": total_loss,
            "loss_token_mass_by_family": family_mass, "loss_token_mass_by_split": split_mass,
            "example_fraction_by_split": {split: sum(r["split"] == split for r in summaries) / len(summaries)
                                          for split in split_mass},
            "loss_token_fraction_by_split": {split: mass / total_loss for split, mass in split_mass.items()},
            "max_prompt_tokens": max(r["prompt_tokens"] for r in summaries),
            "max_target_tokens_including_eos": max(r["target_tokens_including_eos"] for r in summaries),
            "max_sequence_tokens": max(r["sequence_tokens"] for r in summaries),
            "audit_sha256": schema.digest(summaries), "tokenizer_binding": getattr(tokenizer, "binding", None),
            "masking": "PROMPT_-100_TARGET_AND_FINAL_EOS_POSITIVE",
            "thinking_enabled": False, "truncation_used": False,
            "full_hf_auto_tokenizer_equivalence": "NOT_RUN_SEPARATE_QUALIFIED_PROFILE_REQUIRED",
            "native_model_loads": 0, "native_model_forwards": 0, "capability_evidence": False}
