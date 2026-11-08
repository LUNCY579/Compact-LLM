"""Label boundary, full-prompt teacher, mmap-slice and scalar arithmetic fixtures."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from cross_scale.generation_stage.v1.src import control_calibration as c
from shared.hashing import canonical_sha256, sha256_file
from shared.hashing import write_json
from shared.legacy import legacy_graft


def prompt(index=0, partition="fit"):
    scenario = f"cross_scale_calibration_symbolic_0123456789ab_{index:04d}"
    return {"track": "cross_scale", "split": "calibration", "variant": "original",
            "example_id": scenario + "_original", "scenario_id": scenario,
            "family": "symbolic", "template_id": "toy_1", "prompt": "A implies B. A. Is B true?",
            "prompt_sha256": canonical_sha256("A implies B. A. Is B true?"),
            "alignment_partition": partition}


class ForbidLabels(dict):
    def __getitem__(self, key):
        if key in ("answer", "facts", "oracle", "scorer"):
            pytest.fail("label-bearing field accessed")
        return super().__getitem__(key)


def test_prompt_projection_reads_only_b_calibration_identity_and_prompt():
    row = ForbidLabels(prompt())
    row.update(answer=object(), facts=object(), oracle=object(), scorer=object())
    projected = c.prompt_projection(row)
    assert set(projected) == set(c.PROMPT_FIELDS)
    assert not set(projected) & {"answer", "facts", "oracle", "scorer"}


@pytest.mark.parametrize("field,value", [("track", "native_armr"), ("split", "architecture_dev"),
    ("split", "confirmatory_dev"), ("variant", "paraphrase"), ("alignment_partition", "holdout"),
    ("prompt_sha256", "0" * 64), ("example_id", "native_armr_calibration_0")])
def test_other_track_role_variant_hash_or_partition_is_rejected(field, value):
    row = prompt()
    row[field] = value
    with pytest.raises(ValueError):
        c.prompt_projection(row)


def test_scenario_suffix_three_is_held_out():
    assert c.prompt_projection(prompt(3, "holdout"))["alignment_partition"] == "holdout"


def test_ordinary_mask_excludes_union_special_and_added_vocabulary():
    ids = torch.tensor([7, 3, 8, 4])
    assert c.ordinary_mask(ids, [7, 8]).tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match="no ordinary"):
        c.ordinary_mask(torch.tensor([7, 8]), [7, 8])
    with pytest.raises(ValueError, match="int64"):
        c.ordinary_mask(torch.tensor([1., 2.]), [7])


def test_equal_token_relative_energy_and_fp64_reduction_ignore_special_scale():
    h = torch.tensor([[1000., 1000.], [1., 1.], [10., 10.]])
    teacher = torch.tensor([[9000., 9000.], [2., 2.], [10., 10.]])
    control = torch.tensor([[.1, .1], [1., 1.], [5., 5.]])
    values = c.scenario_energies(h, teacher, control, torch.tensor([False, True, True]))
    assert values["ordinary"]["teacher_energy"] == 2.5
    assert values["ordinary"]["control_energy"] == .625
    assert values["special_control"]["teacher_energy"] == 81.
    assert values["final_prompt"]["teacher_energy"] == 1.
    assert c.fit_alpha(values["ordinary"]) == 2.


def test_energy_floor_is_fixed_and_empty_special_stratum_is_explicit():
    h = torch.zeros(2, 3)
    delta = torch.full((2, 3), 1e-4)
    values = c.scenario_energies(h, delta, 2 * delta, torch.tensor([True, True]))
    expected = float(delta.double().square().mean(-1).div(1e-8).mean())
    assert values["ordinary"]["teacher_energy"] == expected
    assert values["special_control"] is None
    assert c.fit_alpha(values["ordinary"]) == .5


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "mask", "empty"])
def test_bad_calibration_tensors_fail_closed(kind):
    h, teacher, control = (torch.ones(3, 2) for _ in range(3))
    mask = torch.tensor([True, True, False])
    if kind == "shape":
        teacher = teacher[:, :1]
    elif kind == "dtype":
        teacher = teacher.double()
    elif kind == "nonfinite":
        control[0, 0] = float("inf")
    elif kind == "mask":
        mask = mask[:1]
    else:
        mask[:] = False
    with pytest.raises(ValueError):
        c.scenario_energies(h, teacher, control, mask)


def case(index, teacher, control, tokens):
    return {"example_id": "example_" + str(index), "scenario_id": "scenario_" + str(index),
            "metrics": {"ordinary": {"teacher_energy": teacher, "control_energy": control,
                "hidden_rms": 1., "teacher_delta_rms": 1., "control_delta_rms": 1., "token_count": tokens}}}


def test_aggregation_weights_scenarios_equally_not_by_token_count():
    values = c.aggregate_partition([case(0, 4., 1., 2), case(1, 1., 1., 200)])
    assert values["teacher_energy"] == 2.5
    assert values["ordinary_tokens"] == 202 and values["scenarios"] == 2
    assert c.fit_alpha(values) == math.sqrt(2.5)


def test_duplicate_scenarios_rejected_and_holdout_is_not_an_alpha_input():
    fit = [case(0, 4., 1., 3), case(1, 4., 1., 4)]
    holdout = [case(2, 999999., .000001, 3)]
    assert c.fit_alpha(c.aggregate_partition(fit)) == 2.
    holdout[0]["metrics"]["ordinary"]["teacher_energy"] = 1e12
    assert c.fit_alpha(c.aggregate_partition(fit)) == 2.
    with pytest.raises(ValueError, match="independent"):
        c.aggregate_partition([fit[0], fit[0]])


@pytest.mark.parametrize("value", [0., -1., float("inf"), float("nan"), True])
@pytest.mark.parametrize("key", ["teacher_energy", "control_energy"])
def test_zero_negative_or_nonfinite_fit_never_gets_clipped_or_fallback(key, value):
    fit = {"teacher_energy": 1., "control_energy": 1.}
    fit[key] = value
    with pytest.raises(ValueError):
        c.fit_alpha(fit)


def test_actual_donor_full_prompt_original_positions_causality_and_down_bias_cancellation():
    torch.manual_seed(2702)
    architecture = Qwen3Config(hidden_size=12, intermediate_size=24, num_hidden_layers=36,
        num_attention_heads=2, num_key_value_heads=1, head_dim=6, max_position_embeddings=64)
    architecture._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(architecture, 24).eval().requires_grad_(False)
    rope = Qwen3RotaryEmbedding(architecture).eval().requires_grad_(False)
    bridges = legacy_graft().Bridges(torch.randn(8, 12) * .1, torch.randn(12) * .02,
                                    torch.randn(12, 8) * .1, torch.ones(8) * 1000).eval()
    hidden = torch.randn(1, 4, 8)
    seen = []
    real = layer.forward
    def capture(z, **kwargs):
        seen.append(kwargs)
        return real(z, **kwargs)
    layer.forward = capture
    with torch.inference_mode():
        delta, receipt = c.teacher_delta(hidden, layer, rope, bridges)
        positions, mask = legacy_graft().positions_mask(hidden)
        z = bridges.project(hidden)
        expected = (real(z, attention_mask=mask, position_ids=positions,
                         position_embeddings=rope(z, positions), use_cache=False, past_key_values=None) - z) @ bridges.down
        altered = hidden.clone()
        altered[:, 2:] += 20
        changed, _ = c.teacher_delta(altered, layer, rope, bridges)
        bridges.down_bias.add_(900)
        rebias, _ = c.teacher_delta(hidden, layer, rope, bridges)
    assert torch.equal(delta, expected) and torch.equal(delta, rebias)
    assert torch.equal(delta[:, :2], changed[:, :2])
    assert all(torch.equal(row["position_ids"], positions) for row in seen)
    assert all(torch.equal(row["attention_mask"], mask) for row in seen)
    assert all(row["use_cache"] is False and row["past_key_values"] is None for row in seen)
    assert receipt["original_positions_sha256"] == c.tensor_sha256(positions)
    assert receipt["original_mask_sha256"] == c.tensor_sha256(mask)


class Tokenizer:
    all_special_ids = [7]
    def get_added_vocab(self):
        return {"added": 8}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return messages[0]["content"]
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[7, 3]])}


def mmap_fixture(tmp_path, monkeypatch):
    root = tmp_path / "programme"
    cache = root / "cross_scale/local/test/activations.pt"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"opaque fixture cache hashed before mocked mmap")
    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(c, "REPO", tmp_path)
    rows = [c.prompt_projection(prompt(0)), c.prompt_projection(prompt(3, "holdout"))]
    context = {"supa_identity": {"model_path": "unused"}, "supa_files": {"model": "unused"},
               "donor_identity": {"revision": "unused"}, "stage1_sources": {"source": "digest"},
               "stage1_config_sha256": "config"}
    contract = {"activation_path": cache.relative_to(tmp_path).as_posix(), "activation_sha256": sha256_file(cache),
                "upstream_context": context, "config": {"max_input_tokens": 256}}
    class OnlyS19(dict):
        def __getitem__(self, key):
            assert key == 19, "unused Supa activation slice was touched"
            return super().__getitem__(key)
        def __iter__(self):
            pytest.fail("unused Supa slices were enumerated")
    bundle = {"schema_version": 1, "track": "cross_scale", "role": "calibration",
              "canonical_fixture_bit_identical": True, "supa_identity": context["supa_identity"],
              "supa_files": context["supa_files"], "donor_identity": context["donor_identity"],
              "sources": context["stage1_sources"], "source_sha256": canonical_sha256(context["stage1_sources"]),
              "config_sha256": context["stage1_config_sha256"], "special_ids": [7, 8],
              "rows": [{key: row[key] for key in ("example_id", "scenario_id", "family", "prompt")} for row in rows],
              "train_prompt_mask": [True, False], "heldout_scenarios": [rows[1]["scenario_id"]],
              "small": OnlyS19({19: [torch.ones(2, 1024), torch.ones(2, 1024)]}),
              "input_ids": [torch.tensor([[7, 3]]), torch.tensor([[7, 3]])], "donor": object(),
              "lengths": [{"example_id": row["example_id"], "full_tokens": 2, "retained_tokens": 2} for row in rows]}
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        assert kwargs == {"weights_only": True, "map_location": "cpu", "mmap": True}
        return bundle
    monkeypatch.setattr(torch, "load", load)
    return contract, rows, bundle, calls


def test_verified_cache_uses_mmap_and_only_s19_ids_and_partition(tmp_path, monkeypatch):
    contract, rows, bundle, calls = mmap_fixture(tmp_path, monkeypatch)
    selected, special = c.load_used_capture(contract, rows, Tokenizer())
    assert set(selected) == {row["example_id"] for row in rows}
    assert special == [7, 8] and len(calls) == 1
    assert selected[rows[0]["example_id"]][0].data_ptr() == bundle["small"][19][0].data_ptr()


@pytest.mark.parametrize("mutation", ["hash", "role", "row", "partition", "ids", "length", "special", "state", "source"])
def test_mmap_reuse_fails_on_wrong_boundary_or_used_slice(tmp_path, monkeypatch, mutation):
    contract, rows, bundle, calls = mmap_fixture(tmp_path, monkeypatch)
    if mutation == "hash":
        contract["activation_sha256"] = "0" * 64
    elif mutation == "role":
        bundle["role"] = "confirmatory_dev"
    elif mutation == "row":
        bundle["rows"][0]["prompt"] = "changed prompt"
    elif mutation == "partition":
        bundle["train_prompt_mask"][0] = False
    elif mutation == "ids":
        bundle["input_ids"][0][0, 1] = 4
    elif mutation == "length":
        bundle["lengths"][0]["full_tokens"] = 3
    elif mutation == "special":
        bundle["special_ids"] = [7]
    elif mutation == "state":
        bundle["small"][19][0][0, 0] = float("nan")
    else:
        bundle["source_sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        c.load_used_capture(contract, rows, Tokenizer())
    if mutation == "hash":
        assert not calls


def test_calibration_config_is_separate_fixed_b_design_and_no_labels():
    config = c.load_config()
    assert config["control"]["seed"] == 2702 and config["control"]["intermediate_size"] == 34560
    assert config["track"] == "cross_scale" and config["supa_layer"] == 19 and config["donor_layer"] == 24
    assert config["min_available_gib"] == 2 and config["max_rss_gib"] == 6
    assert config["label_free"] and not config["holdout_selects_alpha"] and not config["directional_fit"]
    assert config["formula"] == c.FORMULA


def test_frozen_b_bank_projects_128_own_scenarios_without_scoring():
    rows, binding = c.calibration_prompts()
    assert len(rows) == 128
    assert sum(row["alignment_partition"] == "fit" for row in rows) == 96
    assert all(set(row) == set(c.PROMPT_FIELDS) for row in rows)
    assert "cross_scale" in binding["calibration_data_path"]


def artifact_fixture(tmp_path, monkeypatch):
    """Persist a small synthetic receipt graph; no tensor/model deserialization."""
    monkeypatch.setattr(c, "REPO", tmp_path)
    root = tmp_path / "stage"
    monkeypatch.setattr(c, "ARTIFACT_ROOT", root / "calibration")
    monkeypatch.setattr(c, "REGISTRATION_ROOT", root / "registration")
    out = c.ARTIFACT_ROOT / "run"
    rows = [c.prompt_projection(prompt(index, "holdout" if index % 4 == 3 else "fit")) for index in range(128)]
    monkeypatch.setattr(c, "calibration_prompts", lambda path: (rows, {}))
    hashes = {name: "1" * 64 for name in ("up", "up_bias", "down", "down_bias")}
    export = root / "export.json"
    write_json(export, {"tensor_sha256": {"XG": hashes}})
    contract = {"schema_version": c.SCHEMA, "track": "cross_scale", "source_files": {},
                "calibration_config_path": "config.json", "dataset_manifest_path": "manifest.json",
                "export_manifest_path": export.relative_to(tmp_path).as_posix(), "export_manifest_sha256": sha256_file(export),
                "config": {"max_input_tokens": 256, "donor_output_input_rms_bounds": [.05, 10.],
                           "max_stage_seconds": 172800, "max_rss_gib": 6.0},
                "fit_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "fit"],
                "holdout_example_ids": [r["example_id"] for r in rows if r["alignment_partition"] == "holdout"],
                "upstream_context": {"supa_files": {"model.safetensors": "1" * 64}, "donor_identity": {"revision": "toy"}}}
    monkeypatch.setattr(c, "_contract", lambda *args: deepcopy(contract))
    registration_sha = canonical_sha256(contract)
    registration_path = c.REGISTRATION_ROOT / registration_sha / "registration.json"
    write_json(registration_path, {"status": "FROZEN", "registration_sha256": registration_sha, "contract": contract})
    cases = []
    for row in rows:
        metrics = {name: {"token_count": count, "teacher_energy": 1., "control_energy": 4., "hidden_rms": 1.,
                          "teacher_delta_rms": 1., "control_delta_rms": 2.}
                   for name, count in (("all", 3), ("ordinary", 2), ("special_control", 1), ("final_prompt", 1))}
        value = {key: row[key] for key in c.PROMPT_FIELDS if key != "prompt"}
        value.update(token_ids=[7, 3, 4], full_prompt_tokens=3, retained_prompt_tokens=3,
                     hidden_sha256_before="1" * 64, hidden_sha256_after="1" * 64,
                     teacher_delta_sha256="1" * 64, control_delta_sha256="1" * 64,
                     original_positions_sha256="1" * 64, original_mask_sha256="1" * 64,
                     projected_hidden_sha256="1" * 64, donor_rope_sha256=["1" * 64, "1" * 64],
                     donor_output_input_rms_ratio=1., metrics=metrics)
        write_json(out / "cases" / (canonical_sha256(row["example_id"]) + ".json"), value)
        cases.append(value)
    fit = c.aggregate_partition([row for row in cases if row["alignment_partition"] == "fit"])
    holdout = c.aggregate_partition([row for row in cases if row["alignment_partition"] == "holdout"])
    write_json(out / "fit_scale.json", {"alpha": .5, "fit": fit, "formula": c.FORMULA,
                                        "heldout_transformations_at_write": 0})
    module_receipts = {name: {"parameters": count, "trainable_parameters": 0, "backbone_gradients": 0,
                             "state_sha256_before": "1" * 64, "state_sha256_after": "1" * 64,
                             "buffer_sha256_before": {}, "buffer_sha256_after": {}}
                       for name, count in (("donor", 100930816), ("rotary", 0), ("bridges", 0), ("control", c.PARAMETERS))}
    report = {**{key: value for key, value in contract.items() if key != "config"},
              "status": "COMPLETE", "control": c.CONTROL, "formula": c.FORMULA, "label_free": True,
              "answer_labels_used": False, "generated_tokens": 0, "holdout_selects_alpha": False,
              "alpha_search": False, "alpha_clipping": False, "directional_fit": False,
              "trainable_parameters": 0, "backbone_gradients": 0, "canonical_unchanged": True,
              "device": "cpu", "dtype": "float32", "threads": 4, "use_cache": False,
              "runtime_identity": {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 2701,
                  "torch": "2.12.1+cu126", "transformers": "5.14.1", "attention": "eager", "use_cache": False},
              "activation_mmap": True, "accessed_supa_layers": [19], "accessed_donor_activation_layers": [],
              "full_supa_model_loaded": False, "donor_layer_index": 24, "donor_hidden_size": 2560,
              "additional_parameters": c.PARAMETERS, "additional_active_parameters": c.PARAMETERS,
              "additional_matmul_flops_per_token": c.MATMUL_FLOPS_PER_TOKEN,
              "alpha": .5, "fit": {**fit, "scaled_control_energy": 1.},
              "holdout": {**holdout, "scaled_control_energy": 1.},
              "registration_path": registration_path.relative_to(tmp_path).as_posix(),
              "registration_sha256": registration_sha, "special_control_token_ids": [7],
              "case_files": {path.name: sha256_file(path) for path in (out / "cases").glob("*.json")},
              "module_state_receipts": module_receipts, "control_state_sha256": "1" * 64,
              "bridge_source_hashes": hashes, "bridge_copy_hashes": hashes,
              "supa_checkpoint_files_before": contract["upstream_context"]["supa_files"],
              "supa_checkpoint_files_after": contract["upstream_context"]["supa_files"],
              "donor_identity_before": contract["upstream_context"]["donor_identity"],
              "donor_identity_after": contract["upstream_context"]["donor_identity"],
              "fit_scale_sha256": sha256_file(out / "fit_scale.json"), "wall_seconds": 1., "peak_rss_bytes": 1000}
    write_json(out / "started_manifest.json", {**contract, "status": "STARTED", "failure_receipt": None,
                                               "registration_sha256": registration_sha})
    return out, report


def save_report(out, report):
    write_json(out / "report.json", report)
    started = c._read(out / "started_manifest.json")
    write_json(out / "run_manifest.json", {**started, "status": "COMPLETE", "report_sha256": sha256_file(out / "report.json"),
                                           "registration_sha256": report["registration_sha256"]})
    return out / "report.json"


def test_artifact_verifier_recomputes_128_case_fit_holdout_aggregates(tmp_path, monkeypatch):
    out, report = artifact_fixture(tmp_path, monkeypatch)
    path = save_report(out, report)
    verified = c.verify_calibration_artifact(path, sha256_file(path), require_committed=False)
    assert verified["alpha"] == .5 and verified["fit"]["scenarios"] == 96 and verified["holdout"]["scenarios"] == 32


@pytest.mark.parametrize("mutation", ["alpha", "aggregate", "source", "coverage", "module", "parameters", "teacher", "budget", "labels", "runtime"])
def test_artifact_verifier_rejects_rehashed_but_inconsistent_receipts(tmp_path, monkeypatch, mutation):
    out, report = artifact_fixture(tmp_path, monkeypatch)
    if mutation == "alpha":
        report["alpha"] = .6
    elif mutation == "aggregate":
        report["holdout"]["teacher_energy"] += .1
    elif mutation == "source":
        report["source_files"] = {"unexpected.py": "1" * 64}
    elif mutation == "coverage":
        report["case_files"].pop(next(iter(report["case_files"])))
    elif mutation == "module":
        report["module_state_receipts"]["donor"]["state_sha256_after"] = "2" * 64
    elif mutation == "parameters":
        report["module_state_receipts"]["control"]["parameters"] -= 1
    elif mutation == "teacher":
        report["bridge_copy_hashes"] = {**report["bridge_copy_hashes"], "up": "2" * 64}
    elif mutation == "budget":
        report["peak_rss_bytes"] = 7 * 2**30
    elif mutation == "runtime":
        report["runtime_identity"]["dtype"] = "bfloat16"
    else:
        report["answer_labels_used"] = True
    path = save_report(out, report)
    with pytest.raises(RuntimeError):
        c.verify_calibration_artifact(path, sha256_file(path), require_committed=False)
