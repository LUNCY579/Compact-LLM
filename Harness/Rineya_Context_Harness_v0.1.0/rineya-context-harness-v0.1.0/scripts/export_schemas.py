from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_harness import __version__
from rineya_harness.domain.models import (
    ClaimRecord,
    ConversationTurn,
    DeleteMemoryRequest,
    DeleteMemoryResult,
    EvidenceRecord,
    HarnessAnswer,
    HarnessRunRequest,
    MemoryItem,
    RestoreMemoryRequest,
    StatePatch,
)
from rineya_harness.orchestration.contracts import AnswerDraft, QueryPlan, ReconstructionOutcome
from rineya_harness.tools.memory import ModelDeleteMemoryArgs, UpdateMemoryArgs

MODELS = {
    "answer_draft": AnswerDraft,
    "claim_record": ClaimRecord,
    "conversation_turn": ConversationTurn,
    "delete_memory_request": DeleteMemoryRequest,
    "delete_memory_result": DeleteMemoryResult,
    "evidence_record": EvidenceRecord,
    "harness_answer": HarnessAnswer,
    "harness_run_request": HarnessRunRequest,
    "memory_item": MemoryItem,
    "model_delete_memory_args": ModelDeleteMemoryArgs,
    "query_plan": QueryPlan,
    "restore_memory_request": RestoreMemoryRequest,
    "reconstruction_outcome": ReconstructionOutcome,
    "state_patch": StatePatch,
    "update_memory_args": UpdateMemoryArgs,
}


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def build_outputs() -> dict[str, str]:
    outputs = {
        f"{name}.schema.json": _render(model.model_json_schema())
        for name, model in MODELS.items()
    }
    try:
        from rineya_harness.api.app import create_app
        from rineya_harness.config import DatabaseConfig, HarnessConfig, TraceConfig

        with tempfile.TemporaryDirectory(prefix="rineya-openapi-") as temporary:
            temporary_path = Path(temporary)
            config = HarnessConfig(
                database=DatabaseConfig(path=temporary_path / "schema.sqlite3"),
                trace=TraceConfig(directory=temporary_path / "traces"),
            )
            outputs["openapi.json"] = _render(create_app(config=config).openapi())
    except ImportError:
        # The HTTP API is an optional dependency. Core schemas remain exportable.
        pass
    hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in sorted(outputs.items())
    }
    outputs["schema_manifest.json"] = _render(
        {"schema_version": __version__, "files": hashes}
    )
    return outputs


def export(target: Path, *, check: bool = False) -> bool:
    outputs = build_outputs()
    mismatches: list[str] = []
    if not check:
        target.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        path = target / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                mismatches.append(name)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    if check:
        actual = {path.name for path in target.glob("*.json")}
        expected = set(outputs)
        mismatches.extend(sorted(actual - expected))
    if mismatches:
        print("Schema drift:", ", ".join(sorted(set(mismatches))))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("schemas/generated"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return 0 if export(args.output, check=args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
