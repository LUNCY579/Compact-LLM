from __future__ import annotations

import json
from pathlib import Path

import pytest

from rineya_significance_lab.config import ConfigError, config_hash, load_config
from rineya_significance_lab.manifest import atomic_write_json, build_manifest, sha256_file

BASE = """
protocol:
  execution_mode: open_loop_trace_simulation
significance:
  enabled: true
  exponent: 1.0
  score_pooling:
    enabled: false
    width: 1
query:
  mode: agnostic
cache:
  total_budget_slots: 16
  sink_tokens: 2
  recent_tokens: 2
attention_prior:
  enabled: false
memory_slots:
  enabled: false
controller:
  enabled: false
"""


def test_config_hash_is_stable_and_overrides_are_resolved(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text(BASE, encoding="utf-8")
    relocated = tmp_path / "elsewhere" / "base.yaml"
    relocated.parent.mkdir()
    relocated.write_text(BASE, encoding="utf-8")
    first = load_config(path, overrides={"significance.exponent": "0.5"})
    second = load_config(path, overrides={"significance.exponent": "0.5"})
    third = load_config(relocated, overrides={"significance.exponent": "0.5"})
    assert first == second
    assert first["significance"]["exponent"] == 0.5
    assert first["_meta"]["config_sha256"] == config_hash(first, omit_self_hash=True)
    assert first["_meta"]["config_sha256"] == third["_meta"]["config_sha256"]


def test_zero_heavy_slots_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(BASE.replace("total_budget_slots: 16", "total_budget_slots: 4"), encoding="utf-8")
    with pytest.raises(ConfigError, match="zero heavy slots"):
        load_config(path)


def test_open_loop_rejects_feedback_architectures(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(BASE.replace("attention_prior:\n  enabled: false", "attention_prior:\n  enabled: true"), encoding="utf-8")
    with pytest.raises(ConfigError, match="closed-loop"):
        load_config(path)


def test_attention_prior_requires_explicit_no_leakage_declaration(tmp_path: Path) -> None:
    enabled = BASE.replace(
        "execution_mode: open_loop_trace_simulation",
        "execution_mode: closed_loop_model_execution",
    ).replace("attention_prior:\n  enabled: false", "attention_prior:\n  enabled: true")
    path = tmp_path / "bad-access.yaml"
    path.write_text(enabled, encoding="utf-8")
    with pytest.raises(ConfigError, match="access declaration"):
        load_config(path)

    valid = enabled.replace(
        "attention_prior:\n  enabled: true",
        """attention_prior:
  enabled: true
  mode: normalized
  lambda: 0.01
  clip: 2.0
  epsilon: 0.000001
  max_abs_lambda: 0.25
  access:
    query: agnostic
    answer: none
    oracle_evidence: none
    future_tokens: none
    future_hidden_states: none""",
    )
    path.write_text(valid, encoding="utf-8")
    assert load_config(path)["attention_prior"]["lambda"] == 0.01


def test_attention_prior_rejects_unbounded_or_mismatched_access(tmp_path: Path) -> None:
    canonical = (Path(__file__).resolve().parents[1] / "configs" / "halo_attention_prior_smoke.yaml")
    with pytest.raises(ConfigError, match="max_abs_lambda"):
        load_config(canonical, overrides={"attention_prior.lambda": "0.5"})
    with pytest.raises(ConfigError, match="must match"):
        load_config(canonical, overrides={"attention_prior.access.query": "conditioned"})


def test_atomic_json_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert sha256_file(path) == sha256_file(path)


def test_manifest_builder_matches_committed_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    lab = Path(__file__).resolve().parents[1]
    config = load_config(lab / "configs" / "tier_a.yaml")
    manifest = build_manifest(
        repo=lab.parent,
        config=config,
        run_id="builder-test",
        execution_mode="open_loop_trace_simulation",
        architecture_variant="B3",
        seed=17,
    )
    schema = json.loads((lab / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(manifest)
