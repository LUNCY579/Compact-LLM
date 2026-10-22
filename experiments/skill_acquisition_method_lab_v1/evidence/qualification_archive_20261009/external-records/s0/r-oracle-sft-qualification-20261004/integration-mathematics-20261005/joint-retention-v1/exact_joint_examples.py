"""Exact prospective mathematics; no model or scientific input reads."""
from fractions import Fraction as F
import json


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def project_halfspace(u, normal):
    excess = max(F(0), dot(normal, u))
    return tuple(x - excess * n / dot(normal, normal)
                 for x, n in zip(u, normal))


def main():
    u = (F(1), F(1))
    n1, n2 = (F(1), F(0)), (F(-1), F(1))
    first = project_halfspace(project_halfspace(u, n1), n2)
    reverse = project_halfspace(project_halfspace(u, n2), n1)
    assert first == (F(1, 2), F(1, 2)) and dot(n1, first) > 0
    assert reverse == (F(0), F(1)) and dot(n2, reverse) > 0
    joint = (F(0), F(0))
    assert dot(n1, joint) <= 0 and dot(n2, joint) <= 0
    assert dot(tuple(x - y for x, y in zip(joint, u)),
               tuple(x - y for x, y in zip(joint, u))) / 2 == 1
    # For every feasible x<=0,y<=x, (1-x)^2+(1-y)^2 >= 2.
    # The document proves the universal inequality; this row checks its equality.
    raw, d, lam, g, sigma, smoothness = F(2), F(1), F(1), F(-2), F(1), F(2)
    q = lambda x: g * x + sigma * abs(x) + smoothness * x * x / 2
    assert q(d) == 0
    assert d - raw + lam * (g + smoothness * d + sigma) == 0
    implicit = (raw - lam * g) / (1 + lam * smoothness + lam * sigma / abs(d))
    assert implicit == d
    assert q(F(1, 3)) == F(-2, 9) and q(F(1)) == 0 and q(F(0)) == 0
    assert q(F(2)) == 2
    x, y = F(2, 3), F(-1, 4)
    norm2 = x * x + y * y
    assert (x + norm2) + (-x + norm2) == 2 * norm2 > 0
    assert (F(0) + F(0)) == 0
    print(json.dumps({
        "classification": "EXACT_MATHEMATICAL_EXAMPLES_ONLY",
        "status": "PASS", "exact_examples": 6,
        "projection_order_1": [str(v) for v in first],
        "projection_order_2": [str(v) for v in reverse],
        "joint_projection": [str(v) for v in joint],
        "robust_boundary": str(d), "robust_multiplier": str(lam),
        "convex_mix_upper": str(q(F(1, 3))),
        "extrapolated_upper": str(q(F(2))),
        "models_loaded": 0, "optimizer_updates": 0,
        "scientific_inputs_read": 0, "protocol_changed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
