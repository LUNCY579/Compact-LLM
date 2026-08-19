from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    root: Path

    @property
    def full_context_dir(self) -> Path:
        return self.root / "data" / "full_context"

    @property
    def split_transitions_dir(self) -> Path:
        return self.root / "data" / "split_transitions"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "data" / "manifests"

    @property
    def schemas_dir(self) -> Path:
        return self.root / "schemas"

    def full_file(self, split: str) -> Path:
        return self.full_context_dir / f"{split}.jsonl"

    def transition_file(self, split: str) -> Path:
        return self.split_transitions_dir / f"{split}.jsonl"

    def assert_exists(self) -> None:
        missing = []
        for split in ("train", "dev", "test"):
            for path in (self.full_file(split), self.transition_file(split)):
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError("Dataset layout is incomplete:\n" + "\n".join(missing))


def discover_dataset_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    direct = DatasetLayout(candidate)
    try:
        direct.assert_exists()
        return candidate
    except FileNotFoundError:
        pass

    matches: list[Path] = []
    if candidate.is_dir():
        for child in candidate.rglob("data/full_context/train.jsonl"):
            root = child.parents[2]
            try:
                DatasetLayout(root).assert_exists()
            except FileNotFoundError:
                continue
            matches.append(root)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(f"Could not find a prepared Rineya dataset under {candidate}")
    raise RuntimeError(f"Found multiple dataset roots under {candidate}: {unique}")
