# Purpose: Creates either a lightweight manifest package or a copied local model package.
from __future__ import annotations

"""Create a deployable release directory or a lightweight manifest package."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the Phase 8 release candidate.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase_8_release/package")
    parser.add_argument("--mode", choices=("manifest", "copy"), default="manifest")
    parser.add_argument("--routers", type=Path, default=ROOT / "artifacts/phase_7_router/routers.pt")
    parser.add_argument("--response-policy", type=Path, default=ROOT / "configs/response_control.json")
    parser.add_argument(
        "--response-contract",
        type=Path,
        default=ROOT / "scripts/phase_7_sparse_runtime/RESPONSE_CONTROL_CONTRACT.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_json = args.audit.with_suffix(".json")
    audit = read_json(audit_json)
    if audit.get("status") != "PASS_PENDING_HUMAN_RELEASE_DECISION":
        raise SystemExit("Release audit is blocked; packaging is not allowed yet.")
    if (audit.get("response_control") or {}).get("observed_mode") != "one_sentence":
        raise SystemExit("Release audit did not validate the strict one-sentence response policy.")
    for required in (args.response_policy, args.response_contract):
        if not required.exists():
            raise SystemExit(f"Missing response-control release artifact: {required}")
    # A previous copy-mode package can leave a stale model or router behind if
    # a later run switches to manifest/dense mode. Recreate this dedicated
    # package directory so its contents always match the current audit.
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_location = str(args.model_dir.resolve())
    if args.mode == "copy":
        copied_model = args.output_dir / "model"
        if copied_model.exists():
            shutil.rmtree(copied_model)
        shutil.copytree(args.model_dir, copied_model)
        model_location = str(copied_model.resolve())

    shutil.copy2(args.audit, args.output_dir / args.audit.name)
    shutil.copy2(audit_json, args.output_dir / audit_json.name)
    shutil.copy2(args.reference_manifest, args.output_dir / args.reference_manifest.name)
    packaged_policy = args.output_dir / args.response_policy.name
    packaged_contract = args.output_dir / args.response_contract.name
    shutil.copy2(args.response_policy, packaged_policy)
    shutil.copy2(args.response_contract, packaged_contract)

    deployment_mode = audit.get("deployment_mode")
    router_location = None
    if deployment_mode == "SPARSE_FEATURE_FLAG":
        if not args.routers.exists():
            raise SystemExit("Sparse deployment was selected but routers.pt is missing.")
        if args.mode == "copy":
            shutil.copy2(args.routers, args.output_dir / "routers.pt")
            router_location = str((args.output_dir / "routers.pt").resolve())
        else:
            router_location = str(args.routers.resolve())

    write_json(
        args.output_dir / "release_manifest.json",
        {
            "package_version": 1,
            "package_mode": args.mode,
            "model": model_location,
            "model_fingerprint": sha256_tree(Path(model_location)),
            "deployment_mode": deployment_mode,
            "routers": router_location,
            "response_control": audit.get("response_control"),
            "response_policy": str(packaged_policy.resolve()),
            "response_policy_sha256": sha256(packaged_policy),
            "response_contract": str(packaged_contract.resolve()),
            "audit": str((args.output_dir / args.audit.name).resolve()),
            "reference_manifest": str((args.output_dir / args.reference_manifest.name).resolve()),
            "launch_rule": (
                "Use the dense model by default. Enable routers only when deployment_mode is SPARSE_FEATURE_FLAG, "
                "keep a runtime switch that immediately falls back to dense execution, and enforce the packaged "
                "one_sentence response policy after model-native EOS handling."
            ),
        },
    )
    print(args.output_dir / "release_manifest.json")


if __name__ == "__main__":
    main()
