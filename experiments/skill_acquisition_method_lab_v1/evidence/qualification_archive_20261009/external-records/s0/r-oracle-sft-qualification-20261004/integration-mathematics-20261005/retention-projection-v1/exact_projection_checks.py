"""Create-only exact mathematical checks; no model/data/runtime execution."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json


def vec(*entries):
    return tuple(F(x) for x in entries)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, k):
    return tuple(F(k) * x for x in a)


def matvec(matrix, a):
    return tuple(dot(row, a) for row in matrix)


def q(g, curvature, delta):
    return dot(g, delta) + F(curvature, 2) * dot(delta, delta)


def encoded(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, tuple):
        return [encoded(x) for x in value]
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    return value


def check_examples():
    findings = []
    zero = vec(0, 0)

    # Ball c=(1/2,0), radius=1/2. Distance u-c is exactly one.
    g, curvature = vec(-1, 0), 2
    u, delta = vec(F(1, 2), 1), vec(F(1, 2), F(1, 2))
    c, radius = vec(F(1, 2), 0), F(1, 2)
    assert dot(sub(u, c), sub(u, c)) == 1
    assert delta == add(c, scale(sub(u, c), radius))
    assert q(g, curvature, delta) == 0
    lagrange = F(1, 2)
    assert add(sub(delta, u), scale(add(g, scale(delta, curvature)), lagrange)) == zero
    assert dot(vec(1, 0), vec(0, 1)) == 0
    assert q(g, curvature, vec(0, 1)) == 1
    findings.append({"id": "BALL_PROJECTION", "g": g, "L": curvature,
                     "u": u, "projection": delta, "lambda": lagrange,
                     "constraint_value": q(g, curvature, delta),
                     "objective": dot(sub(delta, u), sub(delta, u)) / 2,
                     "orthogonal_task_vector_update_constraint": F(1)})

    # ell=-x+2(x*x+y*y) has true L=4, not the assumed 2.
    assert q(g, 2, delta) == 0
    assert q(g, 4, delta) == F(1, 2)
    findings.append({"id": "UNDERSTATED_CURVATURE", "assumed_L": 2,
                     "actual_L": 4, "claimed_upper": q(g, 2, delta),
                     "actual_loss_drift": q(g, 4, delta)})

    matrix = (vec(2, 1), vec(1, 2))  # SPD: leading minor 2, determinant 3.
    g_h, u_h = vec(1, 0), vec(1, 1)
    weighted, euclidean = vec(0, F(3, 2)), vec(0, 1)
    lam_h = F(3, 2)
    assert add(matvec(matrix, sub(weighted, u_h)), scale(g_h, lam_h)) == zero
    assert dot(g_h, weighted) == 0
    cost_weighted = dot(sub(weighted, u_h), matvec(matrix, sub(weighted, u_h))) / 2
    cost_euclidean = dot(sub(euclidean, u_h), matvec(matrix, sub(euclidean, u_h))) / 2
    assert cost_weighted == F(3, 4) < cost_euclidean == 1
    findings.append({"id": "METRIC_HALFSPACE", "M": matrix,
                     "weighted_projection": weighted, "euclidean_projection": euclidean,
                     "lambda": lam_h, "weighted_cost": cost_weighted,
                     "euclidean_cost_in_M": cost_euclidean})

    # Euclidean projection onto y<=1/4 already lies in the first ball.
    grouped = vec(F(1, 2), F(1, 4))
    group2_g, group2_eps = vec(0, 1), F(1, 4)
    assert q(g, 2, grouped) == F(-3, 16)
    assert dot(group2_g, grouped) == group2_eps
    assert add(sub(grouped, u), scale(group2_g, F(3, 4))) == zero
    hidden_increase, hidden_decrease = dot(vec(1, 0), vec(1, 0)), dot(vec(-1, 0), vec(1, 0))
    assert (hidden_increase + hidden_decrease) / 2 == 0
    assert hidden_increase == 1
    findings.append({"id": "GROUPWISE_CONSTRAINTS", "projection": grouped,
                     "first_group_drift": q(g, 2, grouped),
                     "second_group_drift": dot(group2_g, grouped),
                     "averaging_counterexample": {"group_1": hidden_increase,
                     "group_2": hidden_decrease, "mean": F(0)}})

    # Positive definite old loss at its minimizer has singleton feasible set.
    assert q(zero, 2, zero) == 0
    assert q(zero, 2, vec(1, 0)) == 1
    findings.append({"id": "ZERO_TOLERANCE_AT_OLD_OPTIMUM", "g": zero,
                     "L": 2, "epsilon": F(0), "unique_feasible_update": zero,
                     "acquisition_movement": F(0),
                     "proof": "q(delta)=||delta||^2 <= 0 iff delta=0; finite KKT multiplier can fail for nonzero u."})

    # Raw new-loss descent becomes ascent after retention halfspace projection.
    raw, retained = vec(-1, 1), vec(0, 1)
    old_g, new_g = vec(-1, 0), vec(1, F(1, 2))
    assert dot(old_g, retained) == 0
    assert add(sub(retained, raw), old_g) == zero
    assert dot(new_g, raw) == F(-1, 2)
    assert dot(new_g, retained) == F(1, 2)
    findings.append({"id": "RETENTION_PROJECTION_CAN_LOSE_ACQUISITION",
                     "raw": raw, "retained": retained,
                     "raw_new_linear_loss_drift": dot(new_g, raw),
                     "projected_new_linear_loss_drift": dot(new_g, retained)})

    a, b = vec(1, 0), vec(F(1, 2), F(1, 2))
    assert q(g, 2, a) == q(g, 2, b) == 0
    total = add(a, b)
    cross = F(2) * dot(a, b)
    assert q(g, 2, total) == cross == 1
    findings.append({"id": "SEPARATE_CERTIFICATES_DO_NOT_COMPOSE",
                     "a": a, "b": b, "component_constraint_values": (F(0), F(0)),
                     "sum": total, "sum_constraint_value": q(g, 2, total),
                     "cross_term": cross})
    return findings


def main():
    findings = check_examples()
    directory = Path(__file__).resolve().parent
    references = []
    for name in ("RETENTION_PROJECTION.md", "exact_projection_checks.py"):
        data = (directory / name).read_bytes()
        references.append({"name": name, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()})
    receipt = {"schema": "SKILL-RETENTION-PROJECTION-EXACT-1",
               "classification": "EXACT_MECHANICAL_MATHEMATICS_ONLY",
               "status": "PASS", "examples_checked": len(findings),
               "examples": encoded(tuple(findings)), "references": references,
               "scientific_banks_read": 0, "models_loaded": 0,
               "optimizer_updates": 0, "framework_imports": [],
               "execution_authority": False, "study_protocol_modified": False,
               "real_model_bounds": "UNMEASURED",
               "real_model_skill_gain": "UNMEASURED"}
    output = directory / "EXACT_PROJECTION_RECEIPT.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "examples": len(findings),
                      "receipt": str(output), "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
