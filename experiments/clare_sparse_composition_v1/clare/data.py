"""MAXRL-COMP-002 adapter: pinned, role-gated access to the R/U/S/P composable-skills corpus.

The corpus is NOT copied. Rows are read from the materialised dataset (default
``C:/Users/clclc/BusinessProject/Rineya-Dataset``) and every file is checked against the
dataset's own ``data/manifest.json`` (itself pinned by sha256) before use.

Access is gated by PURPOSE, so held-out data cannot reach mask discovery or training:

    train / discovery -> role ``train``, atomic single-skill manifests only
    calibration       -> role ``calibration`` (TRAIN signatures only; format/sanity checks)
    evaluation        -> role ``dev`` (the registered CLARE-RINEYA-001 endpoint)

``test``, ``challenge`` and the guard banks are refused in CLARE-RINEYA-001: a concurrent study
(COMP-RS-001) registered TEST/guard_test as one-read-per-arm, and CHALLENGE is a distribution
shift reserved for later. Scoring uses the dataset's own verifier (any valid certificate is
accepted); it is loaded as an isolated package with bytecode writing disabled, so importing it
never writes into the dataset tree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .io import sha256_file

DATASET_ID = "MAXRL-COMP-002"
DEFAULT_ROOT = Path("C:/Users/clclc/BusinessProject/Rineya-Dataset")
MANIFEST_SHA256 = "86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44"
PACKAGE_REL = Path("experiments/maxrl_composable_skills_v2")

SKILLS = ("R", "U", "S", "P")
TRAIN_SIGNATURES = ("R", "U", "S", "P", "RU", "US", "SP")
HELDOUT_SIGNATURES = ("RS", "RP", "UP", "RUS", "RUP", "RSP", "USP", "RUSP")
ALL_SIGNATURES = TRAIN_SIGNATURES + HELDOUT_SIGNATURES

PURPOSE_ROLES = {"train": {"train"}, "discovery": {"train"}, "calibration": {"calibration"},
                 "evaluation": {"dev"}}
REFUSED_ROLES_001 = ("test", "challenge", "guard_dev", "guard_test", "guard_challenge",
                     "warmstart", "replay_train")


class AccessRefused(PermissionError):
    pass


@dataclass(frozen=True)
class Row:
    id: str
    parent_id: str
    role: str
    signature: str
    tag: str
    difficulty: int
    template: int
    messages: tuple[dict, ...]
    spec: dict
    target: str | None  # SFT target (oracle references[0]) for train rows; None elsewhere

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(self.signature)


def _load_package(root: Path):
    """Load maxrl_composable_skills_v2 under a private name without writing bytecode."""
    name = "_clare_maxrl_comp_v2"
    if name in sys.modules:
        return sys.modules[name]
    pkg_dir = root / PACKAGE_REL
    spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py",
                                                  submodule_search_locations=[str(pkg_dir)])
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
        for sub in ("contracts", "verifier"):
            sub_spec = importlib.util.spec_from_file_location(f"{name}.{sub}", pkg_dir / f"{sub}.py")
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[f"{name}.{sub}"] = sub_mod
            sub_spec.loader.exec_module(sub_mod)
            setattr(module, sub, sub_mod)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _readjsonl(path: Path) -> Iterable[dict]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line.decode("utf-8"))


class Corpus:
    def __init__(self, root: str | Path = DEFAULT_ROOT, verify: bool = True,
                 refused_roles: Sequence[str] = REFUSED_ROLES_001):
        self.root = Path(root)
        self.data = self.root / "data"
        manifest_path = self.data / "manifest.json"
        if verify and sha256_file(manifest_path) != MANIFEST_SHA256:
            raise ValueError("MAXRL-COMP-002 manifest hash mismatch")
        self.manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
        self.verify = verify
        self.refused_roles = tuple(refused_roles)
        self._verified: dict[str, str] = {}
        self._pkg = None

    # -- integrity ------------------------------------------------------------------------
    def file(self, rel: str) -> Path:
        path = self.data / rel
        if self.verify and rel not in self._verified:
            entry = self.manifest["files"].get(rel)
            if entry is None:
                raise ValueError(f"{rel} is not a manifest-listed file")
            got = sha256_file(path)
            if got != entry["sha256"]:
                raise ValueError(f"{rel}: sha256 {got} != manifest {entry['sha256']}")
            self._verified[rel] = got
        return path

    def curriculum(self, name: str) -> list[str]:
        entry = self.manifest["curricula"][name]
        path = self.root / "data" / entry["path"]
        if self.verify and sha256_file(path) != entry["sha256"]:
            raise ValueError(f"curriculum {name} hash mismatch")
        body = json.loads(path.read_bytes().decode("utf-8"))
        if body["name"] != name or len(body["ids"]) != entry["rows"]:
            raise ValueError(f"curriculum {name} content mismatch")
        return list(body["ids"])

    def receipt(self) -> dict:
        return {"dataset": DATASET_ID, "root": str(self.root), "manifest_sha256": MANIFEST_SHA256,
                "verified_files": dict(sorted(self._verified.items()))}

    # -- gating ---------------------------------------------------------------------------
    def _gate(self, purpose: str, role: str) -> None:
        if role in self.refused_roles:
            raise AccessRefused(f"role {role!r} is not readable in this experiment")
        allowed = PURPOSE_ROLES.get(purpose)
        if allowed is None:
            raise AccessRefused(f"unknown purpose {purpose!r}")
        if role not in allowed:
            raise AccessRefused(f"purpose {purpose!r} may not read role {role!r}")

    # -- rows -----------------------------------------------------------------------------
    def _rows(self, role: str, with_targets: bool) -> list[Row]:
        canon = self.file(f"canonical/{role}.jsonl")
        targets = {}
        if with_targets:
            for o in _readjsonl(self.file(f"oracles/{role}.jsonl")):
                refs = o["references"]
                if not refs:
                    raise ValueError(f"{o['id']}: no oracle reference")
                targets[o["id"]] = refs[0]
        out = []
        for r in _readjsonl(canon):
            if r["role"] != role:
                raise ValueError(f"{r['id']}: role {r['role']!r} inside {role}.jsonl")
            out.append(Row(r["id"], r["parent_id"], r["role"], "".join(r["skills"]), r["tag"],
                           int(r["difficulty"]), int(r["template"]), tuple(r["prompt"]),
                           r["verifier"]["spec"], targets.get(r["id"]) if with_targets else None))
        return out

    def atomic_bank(self, skill: str, purpose: str = "train") -> list[Row]:
        """The 3,000-row atomic donor bank for one skill, in manifest order, with SFT targets."""
        if skill not in SKILLS:
            raise ValueError(skill)
        self._gate(purpose, "train")
        ids = self.curriculum(f"atomic_{skill}")
        by_id = {r.id: r for r in self._train_cache()}
        rows = [by_id[i] for i in ids]
        for r in rows:
            if r.signature != skill or r.tag != "core" or r.role != "train" or r.target is None:
                raise ValueError(f"{r.id}: not an atomic {skill} train row")
        return rows

    @lru_cache(maxsize=1)
    def _train_cache(self) -> tuple[Row, ...]:
        return tuple(self._rows("train", with_targets=True))

    def rows(self, role: str, purpose: str, signatures: Sequence[str] | None = None,
             tag: str = "core", with_targets: bool = False) -> list[Row]:
        """``with_targets`` attaches oracle references[0] (needed only for teacher-forced
        EVALUATION scoring; never for allocation or training outside role train)."""
        self._gate(purpose, role)
        if with_targets and purpose in ("train", "discovery") and role != "train":
            raise AccessRefused("targets outside TRAIN cannot feed training or discovery")
        rows = [r for r in self._rows(role, with_targets=with_targets) if r.tag == tag]
        if signatures is not None:
            want = set(signatures)
            rows = [r for r in rows if r.signature in want]
        return rows

    def sample(self, role: str, purpose: str, signatures: Sequence[str], n_per_signature: int,
               selection_key: str, with_targets: bool = False) -> list[Row]:
        """Registered stratified subset: per signature, the first ``n`` rows in sha256(key|id) order."""
        rows = self.rows(role, purpose, signatures, with_targets=with_targets)
        by_sig: dict[str, list[Row]] = {s: [] for s in signatures}
        for r in rows:
            by_sig[r.signature].append(r)
        out = []
        for s in signatures:
            ranked = sorted(by_sig[s], key=lambda r: hashlib.sha256(f"{selection_key}|{r.id}".encode()).hexdigest())
            if len(ranked) < n_per_signature:
                raise ValueError(f"{role}/{s}: only {len(ranked)} rows for n={n_per_signature}")
            out.extend(ranked[:n_per_signature])
        return out

    # -- scoring --------------------------------------------------------------------------
    @property
    def package(self):
        if self._pkg is None:
            self._pkg = _load_package(self.root)
        return self._pkg

    def score(self, row: Row, response: str) -> int:
        """1 iff ``response`` is exactly one strict-JSON certificate valid for the row's spec."""
        return int(self.package.verifier.score(row.spec, response))

    def render_check(self, row: Row) -> None:
        """The stored prompt must equal the dataset renderer's output for its own spec."""
        spec = self.package.contracts.parse_visible(list(row.messages))
        if spec != row.spec:
            raise ValueError(f"{row.id}: visible spec differs from verifier spec")


def assert_parent_disjoint(a: Iterable[Row], b: Iterable[Row]) -> None:
    pa = {r.parent_id for r in a}
    clash = pa & {r.parent_id for r in b}
    if clash:
        raise ValueError(f"{len(clash)} parent worlds shared between the two row sets")


def shortcut_floors(corpus: Corpus, rows: Sequence[Row]) -> dict:
    """Pass rates of content-free responses, per signature: the floor any accuracy must clear.

    ``all_ascending`` returns every index 0..n-1; ``first_k`` returns 0..k-1 when a selection
    size k exists (else all indices); ``unknown`` returns the abstention.
    """
    out: dict[str, dict] = {}
    for r in rows:
        n = len(r.spec["initial"])
        k = r.spec.get("selection", {}).get("k", n)
        cands = {"all_ascending": json.dumps({"answer": list(range(n))}, separators=(",", ":")),
                 "first_k": json.dumps({"answer": list(range(k))}, separators=(",", ":")),
                 "unknown": '{"answer":"UNKNOWN"}'}
        d = out.setdefault(r.signature, {"rows": 0, **{c: 0 for c in cands}})
        d["rows"] += 1
        for c, resp in cands.items():
            d[c] += corpus.score(r, resp)
    for d in out.values():
        for c in ("all_ascending", "first_k", "unknown"):
            d[f"{c}_rate"] = d[c] / d["rows"]
    return out
