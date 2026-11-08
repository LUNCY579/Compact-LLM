"""CPU-only entry points. No hidden native, training, confirmation or network calls."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from .integrity import code_digest, digest, file_hash, new_directory, read_json, read_jsonl, write_new
from .workflow import status, freeze_binding, outside_checkout, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("status")
    p.add_argument("--binding", type=Path)
    p.add_argument("--receipts", type=Path, help="JSON object mapping stage names to receipt objects")
    p = commands.add_parser("selftest")
    p.add_argument("--out", type=Path, required=True, help="new directory outside checkout")
    p = commands.add_parser("freeze")
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("paired")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("factorial")
    for key in ("a", "b", "c", "d"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        result = status(read_json(args.binding) if args.binding else None,
                        read_json(args.receipts) if args.receipts else None)
    elif args.command == "selftest":
        root = Path(__file__).resolve().parents[2]
        out = new_directory(outside_checkout(root, args.out))
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "experiments/sequential_complementary_v1/tests"]
        run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        result = {"test_returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                  "code_sha256": code_digest(), "python": platform.python_version(),
                  "tests_sha256": digest({p.name: file_hash(p) for p in sorted(Path(__file__).with_name("tests").glob("*.py"))}),
                  "classification": "SOFTWARE_FIXTURES_ONLY", "pretrained_model_calls": 0, "optimizer_steps": 0}
        write_new(out / "TESTS.json", result)
        binding = {"experiment": "SCP-001", "scope": "MECHANICS_ONLY"}
        write_new(out / "BINDING.json", binding)
        r = receipt("mechanics", "PASS" if run.returncode == 0 else "FAIL", binding, {"tests": out / "TESTS.json"})
        write_new(out / "RECEIPTS.json", {"mechanics": r})
    elif args.command == "freeze":
        result = {"binding_file_sha256": freeze_binding(read_json(args.binding), args.out)}
    elif args.command == "paired":
        from .analysis import paired, cluster_interval
        before, after = read_jsonl(args.before), read_jsonl(args.after)
        result = {"paired": paired(before, after), "cluster_interval": cluster_interval(before, after)}
        write_new(args.out, result)
    else:
        from .analysis import factorial
        result = factorial(*(read_jsonl(getattr(args, key)) for key in ("a", "b", "c", "d")))
        write_new(args.out, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("test_returncode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
