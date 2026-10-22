from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import checkpoint_fingerprint
from .hashing import atomic_write_json, sha256_file
from .promotion import evaluate_two_control
from .registry import ExperimentSpec, alpha_label


LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LAB_ROOT.parent
GRID_PATH = LAB_ROOT / "cross_scale_evidence" / "12_v2_followup" / "protocol" / "frozen_grids.json"

EXPECTED_ARTIFACTS = {
    "delta4": {
        "relative": "local_models/deltas/P4_D2_W4_4B_s010",
        "digest": "8edf2d4fd5173dfb583b2cc8cbf6768d039981a6be72f81ff7a3bca2ad31243c",
    },
    "delta8": {
        "relative": "local_models/deltas/P4_D2_W4_8B_s010",
        "digest": "2951457e28c7ce503f7fdf836dd5488aa888f821cb009cb8ac735df39dc53c5e",
    },
    "projected4": {
        "relative": "local_models/projected/P4_D2_W4/qwen4b_spectral_rank8_s010",
        "weight_sha256": "43d2d2e7fea550f0f79511ba4739a254dfe609b88eb5ab6e93ce6285fddd0168",
    },
    "projected8": {
        "relative": "local_models/projected/P4_D2_W4/qwen8b_spectral_rank8_s010",
    },
    "v1_consensus": {
        "relative": "local_models/candidate_scratch",
        "weight_sha256": "5758b7af3334af517b6b38c63605e6d74b62617c75bb9e69544e815f8f713a51",
        "digest": "b30894ce690b3e387bc628dce2dec118f18a1bb6d5b976022bcba0f603141299",
    },
}

DEFAULT_MODEL_CANDIDATES = {
    "supa": [
        REPO_ROOT / "Rineya_Published_Cores" / "Rineya-Supa-A1.1-1.1.0-devfreeze",
        REPO_ROOT.parent / "Chat-Bubble" / "Rineya_Published_Cores" / "Rineya-Supa-A1.1-1.1.0-devfreeze",
    ],
    "qwen06": [
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / "c1899de289a04d12100db370d81485cdf75e47ca",
    ],
    "qwen4b": [
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-4B"
        / "snapshots"
        / "1cfa9a7208912126459214e8b04321603b3df60c",
    ],
    "qwen8b": [
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-8B"
        / "snapshots"
        / "b968826d9c46dd6066d109eabc6255188de91218",
    ],
}

ENV_KEYS = {
    "supa": "RINEYA_SUPA_PATH",
    "qwen06": "RINEYA_QWEN06_PATH",
    "qwen4b": "RINEYA_QWEN4B_PATH",
    "qwen8b": "RINEYA_QWEN8B_PATH",
}

METHOD_COMPLEXITY = {
    "CONSENSUS_SIGN": 10,
    "TASK_VECTOR": 15,
    "CONSENSUS_COSINE": 20,
    "TIES": 30,
    "DARE": 40,
    "DARETIES": 50,
    "FISHER": 60,
    "LAYER": 70,
    "MODULE": 80,
}


class V2Error(RuntimeError):
    pass


def load_frozen_grids(path: Path = GRID_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path(name: str) -> Path:
    env_value = os.environ.get(ENV_KEYS[name])
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(DEFAULT_MODEL_CANDIDATES[name])
    for candidate in candidates:
        if candidate.is_dir() and (
            (candidate / "model.safetensors").is_file()
            or (candidate / "model.safetensors.index.json").is_file()
        ):
            return candidate.resolve()
    raise V2Error(f"could not resolve {name}; set {ENV_KEYS[name]}")


def gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        busy = False
        if completed.returncode == 0:
            for line in completed.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3:
                    utilization = float(parts[1])
                    memory_used = float(parts[2])
                    if utilization >= 10.0 or memory_used >= 1024.0:
                        busy = True
        return {
            "available": completed.returncode == 0,
            "busy": busy,
            "summary": completed.stdout.strip(),
            "returncode": completed.returncode,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"available": False, "busy": False, "error": repr(error)}


def _weight_sha(path: Path) -> str | None:
    single = path / "model.safetensors"
    if single.is_file():
        return sha256_file(single)
    return None


def verify_cached_artifacts(lab_root: Path = LAB_ROOT) -> dict[str, Any]:
    rows = []
    mismatches = []
    for name, expected in EXPECTED_ARTIFACTS.items():
        path = (lab_root / expected["relative"]).resolve()
        row: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "exists": path.is_dir(),
        }
        if not path.is_dir():
            mismatches.append(f"missing:{name}")
            rows.append(row)
            continue
        fingerprint = checkpoint_fingerprint(path)
        row["digest"] = fingerprint["digest"]
        row["weight_sha256"] = _weight_sha(path)
        if "digest" in expected and fingerprint["digest"] != expected["digest"]:
            mismatches.append(f"digest:{name}")
        if (
            "weight_sha256" in expected
            and row["weight_sha256"] != expected["weight_sha256"]
        ):
            mismatches.append(f"weight:{name}")
        rows.append(row)
    return {
        "schema_version": "2.0",
        "status": "ok" if not mismatches else "mismatch",
        "mismatches": mismatches,
        "artifacts": rows,
    }


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    merge_family: str
    coefficient: float
    seed: int
    parameters: dict[str, Any]
    wave: str = "first"

    def spec(self, source_hashes: dict[str, str]) -> ExperimentSpec:
        return ExperimentSpec(
            donor=str(self.parameters.get("donor", "4B8B")),
            projection_strategy="P4",
            depth_mapping="D2",
            width_mapping="W4",
            merge_family=self.merge_family,
            coefficient=self.coefficient,
            seed=self.seed,
            output_dtype="float16",
            source_hashes=source_hashes,
            layer_mask=tuple(self.parameters.get("layers", ())),
            module_mask=tuple(self.parameters.get("modules", ())),
            parameters=self.parameters,
        )


def _consensus_plans(alphas: Iterable[float], *, wave: str) -> list[CandidatePlan]:
    plans = []
    for alpha in alphas:
        plans.append(
            CandidatePlan(
                name=f"CONS_SIGN_{alpha_label(alpha)}",
                merge_family="CONSENSUS_SIGN",
                coefficient=alpha,
                seed=17,
                parameters={"method": "sign_agree", "donor": "4B8B"},
                wave=wave,
            )
        )
        for threshold in load_frozen_grids()["cosine_thresholds"]:
            label = f"CONS_COS{int(threshold * 10):02d}_{alpha_label(alpha)}"
            plans.append(
                CandidatePlan(
                    name=label,
                    merge_family="CONSENSUS_COSINE",
                    coefficient=alpha,
                    seed=17,
                    parameters={
                        "method": "tensor_cosine",
                        "cosine_threshold": threshold,
                        "donor": "4B8B",
                    },
                    wave=wave,
                )
            )
    return plans


def _ties_plans(alphas: Iterable[float], *, wave: str) -> list[CandidatePlan]:
    grids = load_frozen_grids()
    plans = []
    for alpha in alphas:
        for keep in grids["ties_keep_fractions"]:
            for delta_set in grids["ties_delta_sets"]:
                donor = "+".join(delta_set)
                plans.append(
                    CandidatePlan(
                        name=f"TIES_K{int(keep * 100):02d}_{donor}_{alpha_label(alpha)}",
                        merge_family="TIES",
                        coefficient=alpha,
                        seed=17,
                        parameters={
                            "keep_fraction": keep,
                            "delta_set": list(delta_set),
                            "donor": donor,
                        },
                        wave=wave,
                    )
                )
    return plans


def _dare_plans(alphas: Iterable[float], *, wave: str) -> list[CandidatePlan]:
    grids = load_frozen_grids()
    plans = []
    for alpha in alphas:
        for drop in grids["dare_drop_probabilities"]:
            for seed in grids["dare_seeds"]:
                plans.append(
                    CandidatePlan(
                        name=f"DARE8_P{int(drop * 100):02d}_S{seed}_{alpha_label(alpha)}",
                        merge_family="DARE",
                        coefficient=alpha,
                        seed=seed,
                        parameters={
                            "drop_probability": drop,
                            "delta_set": ["delta8"],
                            "donor": "8B",
                        },
                        wave=wave,
                    )
                )
                plans.append(
                    CandidatePlan(
                        name=(
                            f"DARETIES_P{int(drop * 100):02d}_S{seed}_{alpha_label(alpha)}"
                        ),
                        merge_family="DARETIES",
                        coefficient=alpha,
                        seed=seed,
                        parameters={
                            "drop_probability": drop,
                            "keep_fraction": grids["dare_ties_keep_fraction"],
                            "delta_set": ["delta4", "delta8"],
                            "donor": "4B8B",
                        },
                        wave=wave,
                    )
                )
    return plans


def first_wave_plans() -> list[CandidatePlan]:
    alpha = float(load_frozen_grids()["first_wave_alpha"])
    return _consensus_plans([alpha], wave="first") + _ties_plans(
        [alpha], wave="first"
    ) + _dare_plans([alpha], wave="first")


def expand_alpha_plans(families: set[str]) -> list[CandidatePlan]:
    extras = [
        alpha
        for alpha in load_frozen_grids()["alphas"]
        if abs(alpha - 0.15) > 1e-12
    ]
    plans: list[CandidatePlan] = []
    if "CONSENSUS_SIGN" in families or "CONSENSUS_COSINE" in families:
        plans.extend(_consensus_plans(extras, wave="expand"))
    if "TIES" in families:
        plans.extend(_ties_plans(extras, wave="expand"))
    if "DARE" in families or "DARETIES" in families:
        plans.extend(_dare_plans(extras, wave="expand"))
    return plans


def one_layer_plans() -> list[CandidatePlan]:
    grids = load_frozen_grids()
    alpha = float(grids["one_layer_alpha"])
    plans = []
    for donor in grids["one_layer_donors"]:
        for layer in range(28):
            plans.append(
                CandidatePlan(
                    name=f"LAYER_{donor}_{layer:02d}_{alpha_label(alpha)}",
                    merge_family="LAYER",
                    coefficient=alpha,
                    seed=17,
                    parameters={"donor": donor, "layers": [layer]},
                    wave="localization",
                )
            )
    return plans


def rank_two_control(
    supa: dict[str, Any],
    consensus: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, summary in candidates.items():
        decision = evaluate_two_control(supa, consensus, summary)
        rows.append(
            {
                "name": name,
                "complexity": METHOD_COMPLEXITY.get(
                    summary.get("merge_family", name.split("_")[0]), 100
                ),
                **decision,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["eligible"],
            -row["worst_family_score"],
            -row["scenario_macro_delta_vs_supa"],
            -row["overall_delta_vs_supa"],
            row["complexity"],
            row["name"],
        )
    )
    return rows


def successive_halving_names(ranked: list[dict[str, Any]], keep: int) -> list[str]:
    eligible = [row["name"] for row in ranked if row["eligible"]]
    if eligible:
        return eligible[:keep]
    return [row["name"] for row in ranked[:keep]]


def write_verification_report(report: dict[str, Any], destination: Path) -> None:
    atomic_write_json(destination, report)
