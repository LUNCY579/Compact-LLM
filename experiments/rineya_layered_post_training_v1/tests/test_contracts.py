from dataclasses import asdict, replace
import json
import pytest
from experiments.rineya_layered_post_training_v1.config import ARMS, Config, canonical
from experiments.rineya_layered_post_training_v1.data import fixtures, validate, score, load_rows, fingerprint
from experiments.rineya_layered_post_training_v1.runtime import Budget


@pytest.mark.parametrize("arm", ARMS)
def test_schedule_matches_actor_budget(cfg, arm):
    c = replace(cfg, arm=arm)
    assert sum(n for s, n in c.schedule() if s != "critic_fit") == c.actor_steps
    assert all(n >= 2 for s, n in c.schedule() if s in c.train_stages)


@pytest.mark.parametrize("key,value", [("arm", "bad"), ("seed", -1), ("actor_steps", 0), ("actor_steps", 1), ("group_size", 1), ("learning_rate", float("nan")), ("dpo_beta", 0), ("clip_epsilon", 1.), ("max_retention_drop", -.1), ("soup_alphas", (.25, .5)), ("soup_alphas", (0., 1., 1.)), ("required_suites", ("acquisition",)), ("tail_checkpoints", 1), ("min_eval_groups", True), ("trainable_fraction_cap", 0.)])
def test_invalid_configs(cfg, key, value):
    with pytest.raises(ValueError):
        replace(cfg, **{key: value})


def test_config_roundtrip(cfg):
    assert Config.from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg
    with pytest.raises(TypeError):
        Config.from_dict({"unknown": True})
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})


def test_leakage_and_duplicate_ids():
    rows = fixtures()
    with pytest.raises(ValueError):
        validate([rows[0], rows[0]])
    for changes in ({"group": rows[0].group}, {"prompt": rows[0].prompt}, {"prompt": " sum  0 0 "}):
        with pytest.raises(ValueError):
            validate([rows[0], replace(rows[8], **changes)])


def test_verifiers_and_provenance(tmp_path):
    r = fixtures()[0]
    assert score(r, " 0 ") == 1
    assert score(r, "0", False) == 0
    j = replace(r, verifier="json_equal", target='{"a":1}')
    assert score(j, '{ "a" : 1 }') == 1
    assert score(j, '{"a":true}') == 0
    assert score(j, "not json") == 0
    c = replace(r, verifier="constraints", target="Hello captain", must_contain=("captain",), must_not_contain=("BANANA",))
    assert score(c, "captain BANANA") == 0
    for changes in ({"source": ""}, {"revision": ""}, {"split": "test"}, {"verifier": "python_eval"}, {"rejected": "0"}, {"verifier": "constraints"}):
        with pytest.raises(ValueError):
            replace(r, **changes)
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(asdict(x)) for x in fixtures()))
    assert fingerprint(load_rows(path)) == fingerprint(fixtures())


@pytest.mark.parametrize("counter", ["processed_tokens", "generated_tokens", "rollout_calls", "actor_steps", "critic_steps"])
def test_budget_rejects_before_mutation(cfg, counter):
    b = Budget(cfg)
    before = dict(b.counts)
    with pytest.raises(ValueError):
        b.charge(**{counter: 10**15})
    assert b.counts == before
    with pytest.raises(ValueError):
        b.charge(**{counter: -1})


def test_replay_interleaves_protected_suites():
    from experiments.rineya_layered_post_training_v1.data import balanced_order
    train = [r for r in fixtures() if r.split == "train"]
    ordered = balanced_order(train, 17)
    assert ordered == balanced_order(train, 17)
    assert {r.suite for r in ordered[:2]} == {"acquisition", "retention"}
    assert {r.id for r in ordered} == {r.id for r in train}
