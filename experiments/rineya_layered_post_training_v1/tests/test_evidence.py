from dataclasses import replace
import copy
import pytest
from experiments.rineya_layered_post_training_v1.evaluation import choose, eligible, paired_confirmation, macro_scores, value_metrics
from experiments.rineya_layered_post_training_v1.data import fixtures, partition
from experiments.rineya_layered_post_training_v1.storage import native_preflight, native_runtime, write_json


def report(split="select", acquisition=.5, retention=.5):
    records = []
    for suite, score in (("acquisition", acquisition), ("retention", retention)):
        for i in range(4):
            records.append({"id": f"{suite}{i}", "group": f"g{i}", "suite": suite, "score": score})
    return {"split": split, "data_hash": "sealed-hash", "records": records, "scores": macro_scores(records)}


def test_retention_is_hard_constraint(cfg):
    baseline = report()
    dangerous = report(acquisition=1., retention=.4)
    better = report(acquisition=.6)
    assert not eligible(dangerous, baseline, cfg)
    assert choose({"base": baseline, "dangerous": dangerous, "better": better}, baseline, cfg) == "better"
    assert choose({"base": baseline, "dangerous": dangerous}, baseline, cfg) == "base"


@pytest.mark.parametrize("mutation", ["confirm", "hash", "missing", "few"])
def test_selection_guards(cfg, mutation):
    base = report()
    candidate = copy.deepcopy(base)
    if mutation == "confirm":
        candidate["split"] = "confirm"
    elif mutation == "hash":
        candidate["data_hash"] = "different"
    elif mutation == "missing":
        del candidate["scores"]["retention"]
    else:
        candidate["records"] = candidate["records"][:1]
    with pytest.raises(ValueError):
        choose({"candidate": candidate}, base, cfg)


def test_group_macro_and_duplicate_guard():
    rs = [{"id": str(i), "group": "same", "suite": "x", "score": 1.} for i in range(100)]
    rs.append({"id": "other", "group": "other", "suite": "x", "score": 0.})
    assert macro_scores(rs)["x"] == .5
    with pytest.raises(ValueError):
        macro_scores(rs + [rs[0]])


def test_paired_bootstrap_and_pair_guards(cfg):
    base = report("confirm")
    candidate = report("confirm", acquisition=.75)
    result = paired_confirmation(candidate, base, cfg)
    assert result["candidate_pass"]
    assert result["paired_cluster_bootstrap"]["acquisition"]["ci95"] == [.25, .25]
    candidate["records"][0]["group"] = "other"
    with pytest.raises(ValueError):
        paired_confirmation(candidate, base, cfg)


@pytest.mark.parametrize("values,rewards,auc", [([0., 1.], [0., 1.], 1.), ([1., 0.], [0., 1.], 0.), ([1., 1.], [0., 1.], .5), ([1., 2.], [1., 1.], None)])
def test_value_audit(values, rewards, auc):
    metrics = value_metrics(values, rewards)
    assert metrics["auroc"] == auc and metrics["raw_value_not_probability"]


def test_missing_native_and_explicit_permissions(cfg, tmp_path):
    preflight = native_preflight(tmp_path / "missing", cfg, tmp_path)
    assert preflight["status"] == "BLOCKED_NATIVE_UNQUALIFIED"
    assert "local checkpoint directory missing" in preflight["blockers"]
    with pytest.raises(ValueError, match="allow-model-load"):
        native_runtime(tmp_path, cfg, tmp_path)
    write_json(tmp_path / "immutable.json", {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(tmp_path / "immutable.json", {"a": 2})
