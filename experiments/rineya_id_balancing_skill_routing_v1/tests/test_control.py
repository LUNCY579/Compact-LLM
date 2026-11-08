import copy
import numpy as np
import pytest
from idr.control import ControlConfig, CountController, center_bound, digest, quantile_bias


def step(c, n, m=None):
    c.begin()
    c.observe(n, [sum(n) / len(n)] * len(n) if m is None else m)
    return c.commit()


def reference(method="id", ki=0.01, kd=0.02):
    return CountController(["a", "b", "c", "d"], ControlConfig(method, ki, kd, None, 0))


def test_exact_paper_equation_and_worsening_gate():
    c = reference()
    first = step(c, [10, 20, 30, 40])
    assert first["gate"] == [False] * 4
    np.testing.assert_allclose(c.bias, [.006, .002, -.002, -.006], atol=1e-12)
    second = step(c, [5, 20, 30, 45])
    assert second["gate"] == [True, False, False, True]
    np.testing.assert_allclose(c.bias, [.018, .004, -.004, -.018], atol=1e-12)


def test_zero_crossing_and_improving_errors_close_gate():
    c = reference()
    step(c, [10, 20, 30, 40])
    result = step(c, [45, 25, 25, 5])
    assert result["gate"] == [False] * 4


@pytest.mark.parametrize("method", ["none", "sign", "integral", "id"])
def test_balanced_fixed_point(method):
    c = reference(method)
    for _ in range(5):
        step(c, [20] * 4)
    np.testing.assert_array_equal(c.bias, np.zeros(4))


def test_integral_only_matches_id_with_kd_zero():
    a, b = reference("integral"), reference("id", kd=0)
    for n in ([0, 10, 20, 30], [20, 10, 10, 20], [10] * 4):
        step(a, n)
        step(b, n)
    np.testing.assert_array_equal(a.bias, b.bias)


def test_sign_discards_magnitude():
    c = reference("sign")
    step(c, [10, 20, 30, 40])
    np.testing.assert_array_equal(c.bias, [.01, .01, -.01, -.01])


@pytest.mark.parametrize("bad", [{"method": "pid"}, {"ki": -1}, {"kd": float("nan")},
                                  {"min_expected": -1}, {"bias_limit": 0}, {"bias_limit": float("inf")}])
def test_config_invalid(bad):
    with pytest.raises(ValueError):
        ControlConfig(**bad)


@pytest.mark.parametrize("phase", ["dev", "confirm", "eval", "online"])
def test_evaluation_cannot_update(phase):
    with pytest.raises(ValueError):
        reference().begin(phase)


def test_window_lifecycle_and_abort():
    c = reference()
    with pytest.raises(RuntimeError):
        c.observe([1]*4, [1]*4)
    c.begin()
    with pytest.raises(RuntimeError):
        c.begin()
    with pytest.raises(RuntimeError):
        c.state_dict()
    c.observe([0, 0, 2, 2], [1]*4)
    c.abort()
    assert c.updates == 0
    with pytest.raises(RuntimeError):
        c.commit()
    np.testing.assert_array_equal(c.bias, np.zeros(4))


def test_microbatches_update_only_at_completed_step():
    whole, micro = reference(), reference()
    step(whole, [10, 20, 30, 40])
    micro.begin()
    micro.observe([5, 10, 15, 20], [12.5]*4)
    np.testing.assert_array_equal(micro.bias, [0]*4)
    micro.observe([5, 10, 15, 20], [12.5]*4)
    micro.commit()
    np.testing.assert_array_equal(whole.bias, micro.bias)


@pytest.mark.parametrize("counts,targets", [([1,2,3],[1]*4), ([1,2,3,float("nan")],[1]*4),
    ([1,1,1,1],[2]*4), ([-1,1,1,3],[1]*4), ([.5,.5,1,2],[1]*4), ([1,1,1,1],[0,1,1,2])])
def test_invalid_counts_and_mass(counts, targets):
    c = reference()
    c.begin()
    with pytest.raises(ValueError):
        c.observe(counts, targets)
    c.abort()


def test_nonuniform_and_zero_target():
    c = reference()
    result = step(c, [80, 20, 0, 0], [80, 20, 0, 0])
    assert result["error"] == [0]*4
    np.testing.assert_array_equal(c.bias, np.zeros(4))


def test_rare_windows_accumulate_without_mass_loss():
    c = CountController(["a", "b"], ControlConfig(min_expected=2))
    for _ in range(7):
        assert step(c, [1, 0], [.75, .25])["reason"] == "rare_target"
    assert step(c, [1, 0], [.75, .25])["updated"]
    assert c.updates == 1
    np.testing.assert_allclose(c.pending_counts, 0)


def test_empty_batch_no_update():
    c = reference()
    assert step(c, [0]*4)["reason"] == "empty"
    assert c.updates == 0


@pytest.mark.parametrize("seed", range(10))
def test_projection_centered_bounded(seed):
    x = np.random.default_rng(seed).normal(size=16) * 100
    y = center_bound(x, .1)
    assert abs(y.mean()) < 1e-12
    assert max(abs(y)) <= .1 + 1e-12
    np.testing.assert_allclose(y, center_bound(y, .1), atol=1e-12)


def test_saturation_has_no_hidden_integral_windup():
    c = CountController(["a", "b"], ControlConfig(ki=1, kd=0, bias_limit=.1, min_expected=0))
    for _ in range(100):
        step(c, [0, 10])
    step(c, [10, 0])
    assert c.bias[0] < 0 < c.bias[1]


def test_checkpoint_exact_resume_including_rare_buffer():
    a = CountController(["a","b"], ControlConfig(min_expected=2), "artifact")
    step(a, [1,0], [.75,.25])
    b = CountController(["a","b"], a.config, "artifact")
    b.load_state_dict(a.state_dict())
    for _ in range(9):
        step(a,[1,0],[.75,.25]); step(b,[1,0],[.75,.25])
    assert a.state_dict() == b.state_dict()


@pytest.mark.parametrize("change", ["checksum", "identity", "order", "config", "shape", "center", "limit", "counter"])
def test_checkpoint_rejects_mismatch(change):
    c = reference()
    saved = copy.deepcopy(c.state_dict())
    d = saved["payload"]
    if change == "checksum":
        saved["sha256"] = "0" * 64
    else:
        if change == "identity": d["identity"] = "different"
        if change == "order": d["expert_ids"].reverse()
        if change == "config": d["config"]["ki"] = 1
        if change == "shape": d["bias"] = [0]
        if change == "center": d["bias"] = [1]*4
        if change == "limit":
            c = CountController(c.expert_ids, ControlConfig())
            saved = copy.deepcopy(c.state_dict()); d = saved["payload"]
            d["bias"] = [1,-1,0,0]
        if change == "counter": d["updates"] = -1
        saved["sha256"] = digest(d)
    with pytest.raises(ValueError):
        c.load_state_dict(saved)


def test_quantile_matches_manual_formula():
    scores = np.array([[.1,.3,.6],[.8,.4,.2],[.5,.3,.1],[.1,.2,.8]])
    bias = np.array([.1,-.1,0])
    cutoff = np.sort(scores + bias, axis=1)[:, -2]
    wanted = -np.quantile(scores - cutoff[:,None], 2/3, axis=0)
    wanted -= wanted.mean()
    np.testing.assert_allclose(quantile_bias(scores, bias, 1), wanted)


@pytest.mark.parametrize("k", [0,3,4,1.5])
def test_quantile_invalid_k(k):
    with pytest.raises(ValueError):
        quantile_bias(np.ones((2,3)), [0]*3, k)
