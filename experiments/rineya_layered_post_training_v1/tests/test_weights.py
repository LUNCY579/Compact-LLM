from dataclasses import replace
import json
import pytest
import torch
from experiments.rineya_layered_post_training_v1.soup import weighted_soup, compose_deltas, exact_lora_sum, state_hash
from experiments.rineya_layered_post_training_v1.storage import save_snapshot, load_snapshot


@pytest.mark.parametrize("alpha", [0., .1, .25, .5, .75, 1.])
def test_soup_exact(alpha):
    a = {"w": torch.tensor([1., 3.])}
    b = {"w": torch.tensor([5., 7.])}
    out = weighted_soup([a, b], [1-alpha, alpha], ["base", "base"])
    torch.testing.assert_close(out["w"], a["w"] * (1-alpha) + b["w"] * alpha)
    assert a["w"].tolist() == [1., 3.]


@pytest.mark.parametrize("weights,identities", [([1, 1], ["a", "a"]), ([-.1, 1.1], ["a", "a"]), ([.5, .5], ["a", "b"]), ([float("nan"), .5], ["a", "a"]), ([1.], ["a", "a"]), ([.5, .5], ["", ""])])
def test_bad_soup(weights, identities):
    with pytest.raises(ValueError):
        weighted_soup([{"w": torch.ones(2)}] * 2, weights, identities)


@pytest.mark.parametrize("bad", [{"z": torch.ones(2)}, {"w": torch.ones(3)}, {"w": torch.ones(2, dtype=torch.float64)}, {"w": torch.tensor([1., float("nan")])}, {"w": torch.ones(2, dtype=torch.long)}])
def test_bad_tensor_compatibility(bad):
    with pytest.raises(ValueError):
        weighted_soup([{"w": torch.ones(2)}, bad], [.5, .5], ["a", "a"])


def test_lora_factor_trap_and_exact_sum():
    a1, b1 = torch.tensor([[1., 2.]]), torch.tensor([[3.], [4.]])
    a2, b2 = a1 * 3, b1 * 5
    a, b = exact_lora_sum([(a1, b1, 2.), (a2, b2, .5)], [.2, .8])
    expected = .2 * 2. * (b1.double() @ a1.double()) + .8 * .5 * (b2.double() @ a2.double())
    torch.testing.assert_close(b @ a, expected)
    assert not torch.allclose((.2*b1 + .8*b2) @ (.2*a1 + .8*a2), expected.float())
    with pytest.raises(ValueError):
        weighted_soup([{"lora_A.weight": a1}, {"lora_A.weight": a2}], [.5, .5], ["a", "a"])


def test_independent_deltas():
    base = {"w": torch.tensor([3., 4.])}
    da, db = {"w": torch.tensor([2., 0.])}, {"w": torch.tensor([0., 8.])}
    result = compose_deltas(base, [da, db], [1., .25], ["anchor"] * 3)
    assert result["w"].tolist() == [5., 6.]
    with pytest.raises(ValueError):
        compose_deltas(base, [da], [3.], ["anchor"] * 2)


def test_checkpoint_integrity_and_restore(rt, tmp_path):
    state = rt.state()
    directory = tmp_path / "checkpoint"
    save_snapshot(directory, rt, state)
    loaded = load_snapshot(directory, rt)
    assert state_hash(loaded) == state_hash(state)
    old_version = rt.version
    rt.restore(loaded)
    assert rt.version == old_version + 1 and not rt.optimizer.state
    with pytest.raises(FileExistsError):
        save_snapshot(directory, rt, state)
    data = directory / "weights.safetensors"
    data.write_bytes(data.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tampering"):
        load_snapshot(directory, rt)


def test_checkpoint_identity_guard(rt, tmp_path):
    directory = tmp_path / "c"
    save_snapshot(directory, rt, rt.state())
    meta = json.loads((directory / "metadata.json").read_text())
    meta["identity_hash"] = "other"
    (directory / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_snapshot(directory, rt)
