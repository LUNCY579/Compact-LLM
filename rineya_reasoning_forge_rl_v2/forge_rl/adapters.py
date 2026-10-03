"""Framework-independent callable matching TRL's documented custom reward shape.

Native TRL/model integration is intentionally not executed or certified by this
package. This boundary is tested using synthetic batches and exact prompt hashes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .common import read_jsonl, digest, loads, sha_file
from .rewards import score_response


class VerifiableReward:
    def __init__(self, root: str | Path, split: str = "train", mode: str = "final_only"):
        if split not in ("train", "validation"):
            raise ValueError("RL training adapter will not load held-out tests")
        if mode not in ("final_only", "reason_then_final"):
            raise ValueError("Invalid mode")
        self.mode = mode
        self.__name__ = "rineya_verifiable_outcome"
        self.rows = {}
        root = Path(root)
        metadata = root / "metadata" / f"{split}.jsonl.gz"
        relative = f"metadata/{split}.jsonl.gz"
        manifest = loads((root / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
        expected_file = manifest["files"].get(relative)
        if expected_file is None or metadata.stat().st_size != expected_file["bytes"] or sha_file(metadata) != expected_file["sha256"]:
            raise ValueError("Training references do not match the frozen manifest")
        for row in read_jsonl(metadata):
            if row["id"] in self.rows:
                raise ValueError("Duplicate reference ID")
            # Large teacher traces and diagnostic spans are not needed by rewards.
            self.rows[row["id"]] = {key: row[key] for key in
                ("world", "certificate", "prompt_sha256", "reason_prompt_sha256")}

    def __call__(self, completions: list[Any], id: list[str], prompts: list[Any], **kwargs: Any) -> list[float]:
        if not (len(completions) == len(id) == len(prompts)):
            raise ValueError("Reward batch columns are not aligned")
        values = []
        for completion, row_id, prompt in zip(completions, id, prompts):
            if not isinstance(row_id, str):
                raise ValueError("Training ID must be text")
            row = self.rows.get(row_id)
            if row is None:
                raise ValueError("Unknown training ID")
            expected = row["prompt_sha256"] if self.mode == "final_only" else row["reason_prompt_sha256"]
            if digest(prompt) != expected:
                raise ValueError("Prompt/reference mismatch; refusing to score")
            if isinstance(completion, str):
                text = completion
            elif isinstance(completion, list) and len(completion) == 1 and isinstance(completion[0], dict) and completion[0].get("role") == "assistant" and isinstance(completion[0].get("content"), str):
                text = completion[0]["content"]
            else:
                raise ValueError("Expected text or one assistant completion")
            values.append(score_response(row["world"], text, mode=self.mode, certificate=row["certificate"])["reward"])
        return values
