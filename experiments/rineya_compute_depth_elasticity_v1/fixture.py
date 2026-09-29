"""Scripted binary-answer traces: deliberately NOT a learned model or benchmark."""
from .core import Observation, Trace, digest, nested_seed


def bank(split: str, parents: int = 12, width: int = 8, depth: int = 8) -> list[Trace]:
    from .core import positive_int
    for name, n in (("parents", parents), ("width", width), ("depth", depth)):
        positive_int(n, name)
    traces = []
    for p in range(parents):
        parent = f"fixture-{split}-{p}"
        gold = str(p % 2)
        for k in range(1, width + 1):
            # Explicit failure modes: immediate success, late rescue, reversion,
            # stable wrong then rescue, width-only opportunity, permanent failure.
            kind = p % 6
            observations, correct = [], []
            offset = nested_seed(17, parent, k) % 2
            for r in range(1, depth + 1):
                ok = (kind == 0 or (kind == 1 and r >= 4 + offset)
                      or (kind == 2 and r in {2, 3})
                      or (kind == 3 and r >= 7)
                      or (kind == 4 and k % 2 == 0))
                answer = gold if ok else str(1 - int(gold))
                probs = (0.9, 0.1) if answer == "0" else (0.1, 0.9)
                observations.append(Observation(r, digest(parent), answer, probs, float(r)))
                correct.append(ok)
            traces.append(Trace(parent, split, k, "SCRIPTED_FIXTURE_NO_WEIGHTS",
                                tuple(observations), tuple(correct)))
    return traces
