"""CPU correctness tests, including independent distribution/cache oracles."""
import gc
import math
import random
import weakref
from dataclasses import dataclass, replace

import pytest
import torch

from experiments.loopspec_v1.core import (
    Plan, Proposal, Sampling, decode, draw, fallback_distribution, pmf, residual, verify,
)
from experiments.loopspec_v1.tiny import TinyLoopBackend

torch.set_num_threads(1)


class Uniforms:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


@pytest.mark.parametrize("bad", [(), (0.2, 0.2), (-0.1, 1.1), (math.nan, 1), (math.inf, 0)])
def test_invalid_pmf(bad):
    with pytest.raises(ValueError):
        pmf(bad)


@pytest.mark.parametrize("kw", [dict(temperature=-1), dict(temperature=math.nan), dict(top_k=True),
                                    dict(top_k=-1), dict(top_p=0), dict(top_p=1.01)])
def test_invalid_sampling(kw):
    with pytest.raises(ValueError):
        Sampling(**kw)


@pytest.mark.parametrize("bad", [[], [-math.inf, -math.inf], [math.nan, 0], [math.inf, 0]])
def test_invalid_logits(bad):
    with pytest.raises(ValueError):
        Sampling().distribution(bad)


def test_filters_and_tie_break():
    assert Sampling().distribution([1, 1, 0]) == (1, 0, 0)
    assert Sampling(top_k=1, temperature=1).distribution([1, 1, 0]) == (1, 0, 0)
    assert Sampling(top_p=0.7, temperature=1).distribution([0, 0, 0]) == pytest.approx((1/3,)*3)
    assert Sampling(top_p=0.6, temperature=1).distribution([0, 0, 0]) == (0.5, 0.5, 0)
    assert Sampling(temperature=1).distribution([-math.inf, 0]) == (0, 1)
    assert Sampling(temperature=1e-20).distribution([10000, -10000]) == (1, 0)
    assert draw((0, 1, 0), Uniforms(0)) == 1
    with pytest.raises(ValueError):
        draw((0, 1), Uniforms(1))


def test_residual_zero_mass_and_immutability():
    assert residual((0.5, 0.5), (0.5, 0.5)) is None
    assert residual((1, 0), (0, 1)) == (1, 0)
    assert residual((0.5 + 1e-14, 0.5 - 1e-14), (0.5, 0.5)) == (1, 0)
    source = [0.7, 0.3]
    p = Proposal(0, source)
    source[0] = 0
    assert p.distribution == (0.7, 0.3)
    with pytest.raises(ValueError):
        Proposal(1, (1, 0))
    with pytest.raises(ValueError):
        residual((1,), (1, 0))


def test_gate_is_withdrawal_not_confidence_threshold():
    first = Proposal(0, (0.6, 0.3, 0.1))
    assert fallback_distribution(first, (0.7, 0.1, 0.2), True) is None
    assert fallback_distribution(first, (0.7, 0.1, 0.2), False) == pytest.approx((0.5, 0, 0.5))
    assert fallback_distribution(first, (0.5, 0.4, 0.1), True) == (0, 1, 0)


def test_cascade_uses_residual_target_not_original_target():
    first = Proposal(0, (0.6, 0.3, 0.1))
    # First rejection leaves rho=(0, 1/3, 2/3). The second acceptance
    # probability is (1/3)/0.8, NOT p[1]/0.8=0.4/0.8.
    second = Proposal(1, (0, 0.8, 0.2))
    assert verify((0.3, 0.4, 0.3), [first, second], Uniforms(0.99, 0.45, 0.5)) == (2, None)
    assert verify((0.3, 0.4, 0.3), [first, second], Uniforms(0.99, 0.1)) == (1, 1)
    assert verify((0.3, 0.4, 0.3), [first], Uniforms(0.1)) == (0, 0)


def enumerated_law(p, q, deeper, gated):
    """Integrate every primary, gate, fallback, acceptance and residual path.

    Independent vector arithmetic; no calls to production verify/residual.
    """
    n, out = len(p), [0.0] * len(p)
    def normdiff(a, b):
        w = [max(x-y, 0) for x, y in zip(a, b)]
        z = sum(w)
        return [x/z for x in w] if z else None
    rho = normdiff(p, q)
    q2 = normdiff(deeper, q)
    for x, mass in enumerate(q):
        if mass == 0:
            continue
        accepted = min(mass, p[x])
        out[x] += accepted
        rejected = mass - accepted
        if rejected == 0:
            continue
        enabled = q2 is not None and (not gated or deeper[x] < q[x])
        if not enabled:
            for v in range(n):
                out[v] += rejected * rho[v]
            continue
        last = normdiff(rho, q2)
        for y, mass2 in enumerate(q2):
            a2 = min(mass2, rho[y])
            out[y] += rejected * a2
            if mass2 > a2:
                for v in range(n):
                    out[v] += rejected * (mass2-a2) * last[v]
    return out


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("gated", [True, False])
def test_exhaustive_probability_law(seed, gated):
    rng = random.Random(seed)
    cases = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((0.2, 0.3, 0.5), (0.2, 0.3, 0.5), (0.2, 0.3, 0.5))]
    for _ in range(20):
        vectors = []
        for _ in range(3):
            w = [rng.random() if rng.random() > 0.35 else 0 for _ in range(5)]
            if not sum(w):
                w[0] = 1
            vectors.append(tuple(x / sum(w) for x in w))
        cases.append(vectors)
    for p, q, d in cases:
        assert enumerated_law(p, q, d, gated) == pytest.approx(p, abs=1e-12)


def test_implemented_sampler_distribution():
    p, q, d = (0.12, 0.33, 0.55), (0.6, 0.25, 0.15), (0.3, 0.3, 0.4)
    rng = random.Random(20260926)
    counts = [0, 0, 0]
    for _ in range(16000):
        first = Proposal(draw(q, rng), q)
        proposals = [first]
        q2 = fallback_distribution(first, d, True)
        if q2 is not None:
            proposals.append(Proposal(draw(q2, rng), q2))
        token, _ = verify(p, proposals, rng)
        counts[token] += 1
    assert [x/16000 for x in counts] == pytest.approx(p, abs=0.014)


@dataclass(eq=False)
class ScriptState:
    prefix: tuple[int, ...]
    parent: object = None
    depth: int = 0


class ScriptBackend:
    """State-machine fixture only, never described as model evidence."""
    total_depth, vocab_size, max_context = 4, 5, 64
    def __init__(self, mode="primary", target=1, draft=2):
        self.mode, self.target, self.draft = mode, target, draft
        self.states = []
    def prefill(self, prompt, depths):
        state = ScriptState(prompt, depth=4)
        return state, self.z(self.target)
    def start(self, parent, token):
        state = ScriptState(parent.prefix + (token,), parent)
        self.states.append(weakref.ref(state))
        return state
    def advance(self, states):
        for s in states:
            s.depth += 1
    def z(self, token):
        return [0 if i == token else -20 for i in range(self.vocab_size)]
    def readout(self, states):
        def token(s):
            if s.depth == 4 or self.mode == "primary":
                return self.target
            if s.depth == 2 and self.mode == "secondary":
                return self.target
            return self.draft if s.depth == 1 else 3
        return [self.z(token(s)) for s in states]
    def cache_bytes(self, states):
        return 0


@pytest.mark.parametrize("mode,source", [("primary", "primary"), ("secondary", "secondary"), ("neither", "residual")])
@pytest.mark.parametrize("cap", [1, 2, 8])
def test_forced_commit_paths_and_cleanup(mode, source, cap):
    model = ScriptBackend(mode)
    result = decode(model, [0], Plan(max_new_tokens=12, max_branches=cap))
    assert result.tokens == [1]*12
    assert all(x["source"] == source for x in result.commits[1:])
    assert result.stats["peak_active"] <= cap
    gc.collect()
    assert all(ref() is None for ref in model.states)
    if cap == 1:
        assert result.stats["recurrent_batches"] == 44
        assert result.stats["suppressed_children"] > 0


def test_steady_state_pipeline_and_r1():
    m = ScriptBackend()
    result = decode(m, [0], Plan(second=None, max_new_tokens=12))
    assert result.stats["recurrent_batches"] == 4 + 10*1
    assert result.stats["branch_steps"] > result.stats["recurrent_batches"]
    m.total_depth = 1
    assert decode(m, [0], Plan(first=None, second=None, max_new_tokens=4)).tokens == [1]*4


def test_eos_only_after_commit_and_length():
    m = ScriptBackend("secondary", draft=2)
    result = decode(m, [0], Plan(eos_ids=(2,), max_new_tokens=6))
    assert result.tokens == [1]*6 and result.stop_reason == "length"
    result = decode(m, [0], Plan(eos_ids=(1,), max_new_tokens=6))
    assert result.tokens == [1] and result.stop_reason == "eos"
    assert result.stats["recurrent_batches"] == 0
    assert decode(m, [0], Plan(max_new_tokens=0)).tokens == []
    assert decode(m, [0], Plan(max_new_tokens=1)).tokens == [1]


@pytest.mark.parametrize("kw", [dict(first=3), dict(first=2, second=3), dict(first=None),
    dict(max_branches=0), dict(max_new_tokens=-1), dict(seed=1.2), dict(eos_ids=(99,)),
    dict(max_new_tokens=100), dict(gated=1), dict(eos_ids=[1])])
def test_invalid_plan(kw):
    with pytest.raises(ValueError):
        decode(ScriptBackend(), [0], Plan(**kw))


def test_backend_shape_fail_closed():
    m = ScriptBackend()
    m.readout = lambda states: []
    with pytest.raises(ValueError, match="readout"):
        decode(m, [0], Plan())
    with pytest.raises(ValueError):
        decode(ScriptBackend(), [], Plan())


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-11), (torch.float32, 3e-6)])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_native_tiny_cache_against_full_prefix(dtype, atol, seed):
    m = TinyLoopBackend(seed=seed, dtype=dtype)
    prefix = (1, 2, 5, 3, 2)
    state = None
    for i, token in enumerate(prefix):
        state = m.start(state, token)
        oracle = m.full_logits(prefix[:i+1])
        for depth in range(1, 5):
            m.advance([state])
            assert m.readout([state])[0] == pytest.approx(oracle[depth], abs=atol)
        assert set(state.loops) == {1, 2, 3, 4}
        assert set(state.coda) == {1, 2, 3, 4}
        assert len({kv[0].data_ptr() for kv in state.coda.values()}) == 4


def test_cache_wrong_depth_duplicate_state_and_request_isolation():
    m = TinyLoopBackend()
    parent = m.start(None, 1)
    child = m.start(parent, 2)
    with pytest.raises(KeyError):
        m.advance([child])
    with pytest.raises(ValueError):
        m.advance([parent, parent])
    with pytest.raises(ValueError):
        m.readout([parent])
    before = m.weight_digest()
    a = decode(m, [1, 2], Plan(max_new_tokens=8))
    decode(m, [6, 9], Plan(max_new_tokens=8, seed=3))
    assert decode(m, [1, 2], Plan(max_new_tokens=8)).tokens == a.tokens
    assert m.weight_digest() == before
    assert all(not p.requires_grad for p in m.parameters())


def full_greedy(m, prompt, count):
    output = []
    for _ in range(count):
        z = m.full_logits(tuple(prompt) + tuple(output))[m.total_depth]
        output.append(max(range(len(z)), key=lambda i: z[i]))
    return output


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cap", [1, 3, 8])
def test_tiny_all_arms_match_independent_greedy(seed, cap):
    m = TinyLoopBackend(seed=seed)
    oracle = full_greedy(m, [2, 5, 1], 10)
    for first, second, gated in [(None, None, True), (1, None, True), (1, 2, True), (1, 2, False), (2, None, True)]:
        plan = Plan(first=first, second=second, gated=gated, max_branches=cap, max_new_tokens=10)
        result = decode(m, [2, 5, 1], plan)
        assert result.tokens == oracle
        assert result.stop_reason == "length"
        assert result.stats["peak_active"] <= cap
        assert result.stats["recurrent_batches"] <= 36
    if cap >= 3:
        assert m.mixed_depth_batches > 0 and m.max_batch_size > 1


def test_sampling_replay_does_not_mutate_global_rng():
    m = TinyLoopBackend(seed=31)
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    plan = Plan(max_new_tokens=16, sampling=Sampling(temperature=0.8, top_k=8, top_p=0.9), seed=122)
    a, b = decode(m, [3, 2, 8], plan), decode(m, [3, 2, 8], plan)
    assert a.tokens == b.tokens and a.commits == b.commits
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_all_mixed_batch_logits_audited_against_full_prefix():
    class Audited(TinyLoopBackend):
        def readout(self, states):
            output = super().readout(states)
            for s, z in zip(states, output):
                prefix = [a.token for a in s.ancestors()] + [s.token]
                assert z == pytest.approx(self.full_logits(prefix)[s.depth], abs=1e-11)
            return output
    m = Audited(seed=19)
    result = decode(m, [1, 2], Plan(max_new_tokens=12, gated=False,
        sampling=Sampling(temperature=1, top_p=0.9), seed=19))
    assert len(result.tokens) == 12 and m.mixed_depth_batches > 0


@pytest.mark.parametrize("depth,first,second", [(1, None, None), (2, 1, None), (8, 2, 4)])
def test_other_recurrent_depths(depth, first, second):
    m = TinyLoopBackend(seed=4, total_depth=depth)
    result = decode(m, [1, 4], Plan(first=first, second=second, max_new_tokens=7))
    assert result.tokens == full_greedy(m, [1, 4], 7)


def test_late_committed_eos_prunes_speculation():
    class LaterEOS(ScriptBackend):
        def readout(self, states):
            return [self.z(4 if len(s.prefix) >= 4 else 1) for s in states]
    m = LaterEOS()
    result = decode(m, [0], Plan(max_new_tokens=12, eos_ids=(4,)))
    assert result.tokens == [1, 1, 1, 4] and result.stop_reason == "eos"
    gc.collect()
    assert all(ref() is None for ref in m.states)


def test_stalled_backend_is_rejected():
    m = ScriptBackend()
    m.advance = lambda states: None
    with pytest.raises(ValueError, match="exactly one"):
        decode(m, [0], Plan())


def test_native_preflight_cannot_be_self_certified(tmp_path):
    from experiments.loopspec_v1.__main__ import preflight, main
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"qwen3"}')
    value = preflight(config)
    assert value["status"] == "BLOCKED" and not value["weights_loaded"]
    assert any("STANDARD_QWEN3" in x for x in value["reasons"])
    config.write_text('{"model_type":"looped", "qualified":true}')
    assert preflight(config)["status"] == "BLOCKED"
    output = tmp_path / "preflight.json"
    assert main(["native-preflight", "--config", str(config), "--out", str(output)]) == 2
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(["native-preflight", "--config", str(config), "--out", str(output)])
    assert output.read_bytes() == original
    assert preflight(tmp_path / "missing.json")["status"] == "BLOCKED"


def test_receipt_write_is_append_only(tmp_path):
    from experiments.loopspec_v1.__main__ import write_new
    out = tmp_path / "receipt.json"
    write_new(out, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(out, {"a": 2})
