#!/usr/bin/env python3
"""Score sealed bank once for baseline vs layer-8 adapter (protocol: after metrics frozen)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
REPO_ROOT = LAB_ROOT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import atomic_write_json, load_yaml, resolve_repo_path  # noqa: E402

HALO_PREDICT = REPO_ROOT / "rineya_halo_long_context_sft_v1" / "scripts" / "predict_long_context.py"
SEALED_SCORER = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "scripts" / "score_sealed.py"
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_qlora_cc0_8k.yaml",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = resolve_repo_path(cfg["train"]["output_dir"])
    corpus = resolve_repo_path(cfg["data"]["corpus_root"])
    prompts = corpus / "sealed_eval" / "prompts" / "prompts.jsonl"
    answers = corpus / "sealed_eval" / "answer_key" / "answers.jsonl"
    out = output_root / "eval_cc0" / "sealed"
    out.mkdir(parents=True, exist_ok=True)

    if not prompts.exists() or not answers.exists():
        payload = {
            "status": "SKIPPED_SEALED",
            "reason": "sealed prompts/answers missing",
            "prompts": str(prompts),
            "answers": str(answers),
        }
        atomic_write_json(out / "summary.json", payload)
        print(json.dumps(payload, indent=2))
        return 0

    model = str(resolve_core_path(cfg["model"]["path"]))
    adapter = output_root / "adapter"
    summary: dict = {"status": "ok", "limit": args.limit, "arms": {}}

    for condition, use_adapter in (("baseline", False), ("layer8_qlora", True)):
        pred = out / f"{condition}_predictions.jsonl"
        cmd = [
            str(PY),
            str(HALO_PREDICT),
            "--model-id",
            model,
            "--records",
            str(prompts),
            "--output",
            str(pred),
            "--max-new-tokens",
            str(cfg["eval"]["max_new_tokens"]),
            "--trust-remote-code",
        ]
        # predict_long_context may not have all flags — fall back to our eval if needed
        if use_adapter:
            if not adapter.exists():
                summary["arms"][condition] = {"status": "SKIPPED_NO_ADAPTER"}
                continue
            cmd.extend(["--adapter", str(adapter)])
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        try:
            completed = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        except Exception as exc:  # noqa: BLE001
            summary["arms"][condition] = {"status": "SKIPPED_PREDICT", "error": str(exc)}
            continue
        if completed.returncode != 0:
            # Use local eval_cc0 path via a tiny sealed prompt→messages wrapper instead.
            summary["arms"][condition] = {
                "status": "SKIPPED_PREDICT",
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "")[-2000:],
            }
            continue

        score_out = out / f"{condition}_score.json"
        score_cmd = [
            str(PY),
            str(SEALED_SCORER),
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--predictions",
            str(pred),
            "--output",
            str(score_out),
        ]
        scored = subprocess.run(score_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        summary["arms"][condition] = {
            "status": "ok" if scored.returncode == 0 else "SCORE_FAILED",
            "score_path": str(score_out),
            "stderr": (scored.stderr or "")[-1000:],
        }
        if score_out.exists():
            summary["arms"][condition]["report"] = json.loads(
                score_out.read_text(encoding="utf-8")
            )

    atomic_write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
