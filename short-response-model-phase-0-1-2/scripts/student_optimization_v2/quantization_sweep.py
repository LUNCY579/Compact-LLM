from __future__ import annotations

"""DEV-only deployment quantization sweep with dense fallback.

Backends are selected on non-sealed data. A failing backend is recorded and the
sweep continues rather than aborting the entire student-optimization workflow.
Only the selected backend is later allowed into the final Phase 11 sealed run.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction


def run(command: list[str], *, allow_failure: bool = False) -> tuple[bool, str | None]:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=False)
    if result.returncode != 0:
        message = f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}"
        if allow_failure:
            print("WARNING:", message, flush=True)
            return False, message
        raise RuntimeError(message)
    return True, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep student deployment quantization backends on DEV.")
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--knowledge-db", type=Path, required=True)
    p.add_argument("--answerability-calibration", type=Path, required=True)
    p.add_argument("--snapshot-manifest", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--backends", nargs="+", default=["bitsandbytes_nf4", "bitsandbytes_int8"])
    p.add_argument("--retrieval-top-k", type=int, required=True)
    p.add_argument("--selected-top-k", type=int, required=True)
    p.add_argument("--maximum-evidence-tokens", type=int, required=True)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--response-control", choices=("eos_only", "one_sentence"), default="one_sentence")
    p.add_argument("--memory-reconstruction-mode", choices=("off", "prompted", "trained"), default="off")
    p.add_argument("--reconstruction-max-new-tokens", type=int, default=128)
    p.add_argument("--llama-cpp-convert-script", type=Path, default=None)
    p.add_argument("--llama-cpp-quantize-exe", type=Path, default=None)
    p.add_argument("--llama-cli", type=Path, default=None)
    p.add_argument("--gguf-type", default="Q4_K_M")
    p.add_argument("--reuse-existing", action="store_true")
    return p.parse_args()


def memory_bytes(summary: dict[str, Any]) -> float | None:
    for key in ("total_process_memory_bytes", "process_rss_bytes", "model_weight_bytes"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    return None


def benchmark(
    *,
    model: str,
    runtime_backend: str,
    tokenizer_source: str | None,
    output: Path,
    args: argparse.Namespace,
) -> tuple[bool, str | None]:
    if args.reuse_existing and (output / "summary.json").exists():
        return True, None
    command = [
        sys.executable,
        str(ROOT / "scripts/phase_11_production_rag/benchmark_production_runtime.py"),
        "--model", model,
        "--runtime-backend", runtime_backend,
        "--benchmark", str(args.benchmark),
        "--knowledge-db", str(args.knowledge_db),
        "--answerability-calibration", str(args.answerability_calibration),
        "--output-dir", str(output),
        "--retrieval-top-k", str(args.retrieval_top_k),
        "--selected-top-k", str(args.selected_top_k),
        "--maximum-evidence-tokens", str(args.maximum_evidence_tokens),
        "--max-sequence-length", str(args.max_sequence_length),
        "--max-new-tokens", str(args.max_new_tokens),
        "--response-control", args.response_control,
        "--memory-reconstruction-mode", args.memory_reconstruction_mode,
        "--reconstruction-max-new-tokens", str(args.reconstruction_max_new_tokens),
    ]
    if tokenizer_source:
        command += ["--tokenizer-source", tokenizer_source]
    if runtime_backend == "llama_cpp":
        if args.llama_cli is None:
            return False, "llama_cpp benchmark requested without --llama-cli"
        command += ["--llama-cli", str(args.llama_cli)]
    if args.reuse_existing:
        command.append("--resume")
    return run(command, allow_failure=True)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    for path in (args.benchmark, args.knowledge_db, args.answerability_calibration, args.snapshot_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if "test" in args.benchmark.name.casefold() or "sealed" in str(args.benchmark).casefold():
        raise ValueError("Quantization backend selection must use DEV/non-sealed data")
    valid_backends = {"bitsandbytes_nf4", "bitsandbytes_int8", "llamacpp_gguf"}
    unknown = sorted(set(args.backends) - valid_backends)
    if unknown:
        raise ValueError(f"Unsupported quantization backends: {unknown}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    dense_eval = args.output_root / "dense_reference"
    ok, err = benchmark(model=args.model, runtime_backend="transformers", tokenizer_source=None, output=dense_eval, args=args)
    if not ok:
        raise RuntimeError(err or "Dense reference benchmark failed")
    reference = read_json(dense_eval / "summary.json")
    reference_memory = memory_bytes(reference)

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for backend in args.backends:
        artifact = args.output_root / "artifacts" / backend
        manifest = artifact / "phase_11_quantization_manifest.json"
        if backend == "llamacpp_gguf" and (
            args.llama_cpp_convert_script is None
            or args.llama_cpp_quantize_exe is None
            or args.llama_cli is None
        ):
            failures.append({"backend": backend, "error": "GGUF tool paths were not all supplied; backend skipped."})
            continue
        if not (args.reuse_existing and manifest.exists()):
            command = [
                sys.executable,
                str(ROOT / "scripts/phase_11_production_rag/quantize_student.py"),
                "--model", args.model,
                "--output", str(artifact),
                "--backend", backend,
                "--gguf-type", args.gguf_type,
            ]
            if backend == "llamacpp_gguf":
                command += [
                    "--llama-cpp-convert-script", str(args.llama_cpp_convert_script),
                    "--llama-cpp-quantize-exe", str(args.llama_cpp_quantize_exe),
                ]
            if artifact.exists():
                command.append("--overwrite")
            success, error = run(command, allow_failure=True)
            if not success:
                failures.append({"backend": backend, "error": error})
                continue
        if not manifest.exists():
            failures.append({"backend": backend, "error": "Quantization manifest missing after backend execution."})
            continue
        quant = read_json(manifest)
        model_path = str(quant["model_path"])
        runtime_backend = "llama_cpp" if backend == "llamacpp_gguf" else "transformers"
        eval_root = args.output_root / "runtime" / backend
        success, error = benchmark(
            model=model_path,
            runtime_backend=runtime_backend,
            tokenizer_source=args.model if runtime_backend == "llama_cpp" else None,
            output=eval_root,
            args=args,
        )
        if not success or not (eval_root / "summary.json").exists():
            failures.append({"backend": backend, "error": error or "Runtime summary missing."})
            continue
        summary = read_json(eval_root / "summary.json")
        candidate_memory = memory_bytes(summary)
        gain = resource_reduction(reference_memory, candidate_memory)
        assessment = evaluate_opportunity(
            reference,
            summary,
            resource_gain=gain,
            resource_name="total_process_memory_bytes",
        )
        candidates.append({
            "backend": backend,
            "artifact": model_path,
            "quantization_manifest": str(manifest),
            "summary": str(eval_root / "summary.json"),
            "total_process_memory_bytes": candidate_memory,
            "assessment": assessment,
            "eligible": bool(assessment.get("eligible")),
        })

    eligible = [item for item in candidates if item["eligible"]]
    selected = max(
        eligible,
        key=lambda item: (
            float(item["assessment"].get("opportunity_margin", float("-inf"))),
            -float((read_json(Path(item["summary"])).get("runtime") or {}).get("median_total_seconds") or float("inf")),
        ),
        default=None,
    )
    selected_backend = selected["backend"] if selected else "dense_copy"
    payload = {
        "schema_version": "2.0",
        "stage": "student_quantization_sweep",
        "model": args.model,
        "benchmark": str(args.benchmark),
        "sealed_test_used": False,
        "dense_reference_summary": str(dense_eval / "summary.json"),
        "reference_total_process_memory_bytes": reference_memory,
        "selected_backend": selected_backend,
        "selected_candidate": selected,
        "candidates": candidates,
        "failures": failures,
        "frozen_context": {
            "retrieval_top_k": args.retrieval_top_k,
            "selected_top_k": args.selected_top_k,
            "maximum_evidence_tokens": args.maximum_evidence_tokens,
            "answerability_calibration": str(args.answerability_calibration),
        },
        "final_test_policy": "Only selected_backend may be carried into the final Phase 11 sealed runtime experiment; dense remains rollback.",
    }
    write_json(args.output_root / "quantization_selection.json", payload)

    lines = [
        "# Student quantization/runtime DEV sweep",
        "",
        f"- Selected backend: `{selected_backend}`",
        "- Sealed TEST used: `NO`",
        "- Dense fallback retained: `YES`",
        "",
        "| Backend | Total process memory | Resource gain | Quality loss | Opportunity margin | Eligible |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in candidates:
        a = item["assessment"]
        lines.append(
            f"| {item['backend']} | {item['total_process_memory_bytes']} | {fmt(a.get('resource_gain_fraction'))} | "
            f"{fmt(a.get('quality_opportunity_loss_fraction'))} | {fmt(a.get('opportunity_margin'))} | {item['eligible']} |"
        )
    if failures:
        lines.extend(["", "## Backend failures / skips", ""])
        lines.extend([f"- `{item['backend']}`: {item['error']}" for item in failures])
    lines.extend(["", payload["final_test_policy"]])
    (args.output_root / "QUANTIZATION_SWEEP_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(args.output_root / "quantization_selection.json")


if __name__ == "__main__":
    main()
