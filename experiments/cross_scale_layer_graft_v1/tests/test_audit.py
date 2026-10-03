from copy import deepcopy
from audit import primary_gate


def fixture():
    rows = [{"case_id": str(i), "safety_stopped": False, "trajectory_min_rms_ratio": 1.,
             "trajectory_max_rms_ratio": 1., "hidden": [], "logit_cosine": 1.,
             "kl_base_to_arm": 0., "score": 1., "invalid": False} for i in range(8)]
    config = {"hidden_rms_ratio_bounds": [.25, 4], "minimum_mean_logit_cosine": .9,
              "maximum_mean_kl": .5, "maximum_score_drop": 0, "maximum_invalid_increase": 0}
    return rows, deepcopy(rows), config


def test_audit_rejects_missing_duplicate_or_reordered_cases():
    rows, base, c = fixture()
    assert primary_gate(rows, base, c)
    assert not primary_gate(rows[:-1], base, c)
    assert not primary_gate(rows[:-1]+[rows[0]], base, c)
    assert not primary_gate(list(reversed(rows)), base, c)


def test_audit_rejects_hidden_instability_score_loss_or_timeout():
    for key, value in (("trajectory_max_rms_ratio", 4.001), ("safety_stopped", True),
                       ("score", 0.), ("invalid", True), ("kl_base_to_arm", 4.01)):
        rows, base, c = fixture()
        rows[0][key] = value
        assert not primary_gate(rows, base, c)
