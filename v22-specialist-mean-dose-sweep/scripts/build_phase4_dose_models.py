from __future__ import annotations

"""Build Phase-4 specialist-mean soup and capability doses on CPU."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or ROOT), env=env)


def tag(dose: float) -> str:
    return f"{dose:.2f}".replace(".", "p")


def build_phase4_mean(cfg: dict[str, Any], official: Path, v22: Path, out_root: Path, force: bool) -> Path:
    dest = out_root / "soups" / "phase4_specialist_mean_d1p0"
    if dest.exists() and not force:
        print(f"reuse {dest}")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = v22 / cfg["ingredients"]["frozen_anchor"]
    adapters = [str(v22 / rel) for rel in cfg["ingredients"]["specialist_adapters"]]
    # Equal weights omitted => merge_delta_soup normalizes to 1/N each.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(official)
    run(
        [
            sys.executable,
            str(official / "scripts" / "phase_4_model_soups" / "merge_delta_soup.py"),
            "--model-id",
            str(base),
            "--adapters",
            *adapters,
            "--output-dir",
            str(dest),
            "--overwrite",
        ],
        cwd=official,
        env=env,
    )
    return dest


def apply_doses(cfg: dict[str, Any], v22: Path, mean_soup: Path, out_root: Path, force: bool) -> dict[str, str]:
    anchor = v22 / cfg["ingredients"]["frozen_anchor"]
    dose_root = out_root / "doses"
    dose_root.mkdir(parents=True, exist_ok=True)
    built: dict[str, str] = {}
    for dose in cfg["dose_calibration"]["coarse_weights"]:
        dose = float(dose)
        name = f"dose_{tag(dose)}"
        dest = dose_root / name
        if dose <= 0.0:
            # Symlink/copy pointer to anchor without duplicating weights when possible.
            if dest.exists() and force:
                shutil.rmtree(dest)
            if not dest.exists():
                dest.mkdir(parents=True)
                # Lightweight pointer manifest; evaluation uses the anchor path directly.
                (dest / "dose_manifest.json").write_text(
                    json.dumps(
                        {
                            "dose": 0.0,
                            "model_path": str(anchor),
                            "note": "zero dose reuses frozen_anchor path",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            built[name] = str(anchor)
            continue
        if abs(dose - 1.0) < 1e-12:
            built[name] = str(mean_soup)
            continue
        if dest.exists() and not force:
            built[name] = str(dest)
            continue
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "interpolate_dose_cpu.py"),
                "--anchor",
                str(anchor),
                "--soup",
                str(mean_soup),
                "--dose",
                str(dose),
                "--output-dir",
                str(dest),
                "--force",
            ]
        )
        built[name] = str(dest)
    return built


def maybe_verify_prior(cfg: dict[str, Any], mean_soup: Path, out_root: Path) -> dict[str, Any] | None:
    prior = (ROOT / cfg["prior_micro_experiment"]).resolve()
    left = prior / "artifacts" / "soups" / "specialists_lora_uniform"
    if not left.exists():
        return {"skipped": True, "reason": "prior specialists_lora_uniform missing"}
    report = out_root / "PHASE4_EQUIVALENCE_CHECK.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "verify_phase4_equivalence.py"),
        "--left",
        str(left),
        "--right",
        str(mean_soup),
        "--output-json",
        str(report),
    ]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"equivalent": False}
    data["exit_code"] = proc.returncode
    data["prior_specialists_lora_uniform"] = str(left)
    data["phase4_mean"] = str(mean_soup)
    report.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "dose_sweep.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-verify-prior", action="store_true")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    official = (ROOT / cfg["official_root"]).resolve()
    v22 = (ROOT / cfg["v22_run_root"]).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    mean_soup = build_phase4_mean(cfg, official, v22, args.output_root, args.force)
    doses = apply_doses(cfg, v22, mean_soup, args.output_root, args.force)
    verify = None if args.skip_verify_prior else maybe_verify_prior(cfg, mean_soup, args.output_root)
    summary = {
        "merge_backend": "official phase_4_model_soups/merge_delta_soup.py",
        "formula_unit_soup": "W = W_anchor + mean_i(Delta_i)",
        "formula_dose": "W(d) = (1-d)*W_anchor + d*W_unit = W_anchor + d*mean(Delta)",
        "phase4_specialist_mean": str(mean_soup),
        "doses": doses,
        "prior_equivalence": verify,
    }
    out = args.output_root / "DOSE_BUILD_SUMMARY.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
