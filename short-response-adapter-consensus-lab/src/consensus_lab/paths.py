from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


OFFICIAL_DIRNAME = "short-response-model-phase-0-1-2"
LAB_DIRNAME = "short-response-adapter-consensus-lab"
EXCLUDED_DIRNAMES = {
    "v22-model-soup-micro-experiment",
    "v22-specialist-mean-dose-sweep",
}


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(parent))
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class LabPaths:
    workspace_root: Path
    lab_root: Path
    official_root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "LabPaths":
        here = _resolved(start or Path(__file__))
        candidates = [here, *here.parents]
        for candidate in candidates:
            if candidate.name == LAB_DIRNAME:
                workspace = candidate.parent
                official = workspace / OFFICIAL_DIRNAME
                if official.is_dir():
                    return cls(workspace, candidate, official)
            official = candidate / OFFICIAL_DIRNAME
            lab = candidate / LAB_DIRNAME
            if official.is_dir() and lab.is_dir():
                return cls(candidate, lab, official)
        raise RuntimeError(
            f"Could not discover sibling {OFFICIAL_DIRNAME!r} and {LAB_DIRNAME!r} from {here}"
        )

    @classmethod
    def from_config(cls, path: Path) -> "LabPaths":
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        workspace = _resolved(Path(payload["workspace_root"]))
        lab = _resolved(Path(payload.get("lab_root", workspace / LAB_DIRNAME)))
        official = _resolved(Path(payload.get("official_root", workspace / OFFICIAL_DIRNAME)))
        obj = cls(workspace, lab, official)
        obj.validate_layout()
        return obj

    def validate_layout(self) -> None:
        if self.lab_root == self.official_root or is_relative_to(self.lab_root, self.official_root):
            raise RuntimeError("Lab root must be a sibling of, not inside, the official repository")
        if not self.official_root.is_dir():
            raise FileNotFoundError(f"Official repository not found: {self.official_root}")
        if not self.lab_root.is_dir():
            raise FileNotFoundError(f"Lab root not found: {self.lab_root}")
        if self.lab_root.parent != self.official_root.parent:
            raise RuntimeError("Lab and official repository must share a parent directory")

    def assert_official_source(self, path: Path) -> Path:
        resolved = _resolved(path)
        if not is_relative_to(resolved, self.official_root):
            raise ValueError(f"Expected an official read-only source path, got: {resolved}")
        return resolved

    def assert_output_path(self, path: Path, *, must_be_in_lab: bool = True) -> Path:
        resolved = _resolved(path)
        forbidden = [self.official_root, *(self.workspace_root / name for name in EXCLUDED_DIRNAMES)]
        for root in forbidden:
            if resolved == _resolved(root) or is_relative_to(resolved, root):
                raise ValueError(f"Refusing output beneath protected path: {resolved}")
        if must_be_in_lab and not is_relative_to(resolved, self.lab_root):
            raise ValueError(f"Experimental output must remain inside the lab: {resolved}")
        return resolved

    def resolve_declared_source(self, value: str | Path) -> Path:
        """Resolve historical manifests whose absolute paths predate the sibling layout."""
        raw = Path(value)
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([self.official_root / raw, self.workspace_root / raw])
        parts = list(raw.parts)
        lowered = [part.casefold() for part in parts]
        if "artifacts" in lowered:
            index = lowered.index("artifacts")
            candidates.append(self.official_root.joinpath(*parts[index:]))
        if "data" in lowered:
            index = lowered.index("data")
            candidates.append(self.official_root.joinpath(*parts[index:]))
        for candidate in candidates:
            if candidate.exists():
                return _resolved(candidate)
        return _resolved(candidates[-1] if candidates else raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "lab_root": str(self.lab_root),
            "official_root": str(self.official_root),
        }

