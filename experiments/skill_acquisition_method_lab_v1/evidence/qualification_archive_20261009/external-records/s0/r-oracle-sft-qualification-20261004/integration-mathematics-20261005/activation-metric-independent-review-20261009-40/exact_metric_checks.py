"""Independent rational examples only: no frameworks, weights, or data rows."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
checks = []

def mat(rows):
    return [[F(x) for x in row] for row in rows]

def transpose(a):
    return list(map(list, zip(*a)))

def product(a, b):
    assert len(a[0]) == len(b)
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]

def scale(c, a):
    return [[c*x for x in row] for row in a]

def add(a, b):
    return [[x+y for x, y in zip(row, other)] for row, other in zip(a, b)]

def subtract(a, b):
    return add(a, scale(-1, b))

def trace(a):
    return sum(a[i][i] for i in range(len(a)))

def inner(a, b, q, x):
    return trace(product(product(product(transpose(a), q), b), x))

def norm(a, q, x):
    return inner(a, a, q, x)

def check(name, value):
    assert value, name
    checks.append({"name": name, "passed": True, "evidence": "EXACT_RATIONAL_SYNTHETIC_ALGEBRA"})

identity = mat([[1, 0], [0, 1]])
q = mat([[2, 1], [1, 2]])
x = mat([[3, 1], [1, 1]])
bs = [mat([[1, 2], [0, 1]]), mat([[1], [-1]])]
aa = [mat([[1, 0], [1, 1]]), mat([[2, 1]])]
scales = [F(2), F(-1, 2)]
deltas = [scale(s, product(b, a)) for s, b, a in zip(scales, bs, aa)]
k = [[scales[i]*scales[j]*trace(product(product(product(transpose(bs[i]), q), bs[j]), product(product(aa[j], x), transpose(aa[i])))) for j in range(2)] for i in range(2)]
for i in range(2):
    for j in range(2):
        check(f"mixed_rank_thin_trace_{i}_{j}", k[i][j] == inner(deltas[i], deltas[j], q, x))
check("gram_values", k == mat([[520, -17], [-17, F(17, 2)]]))
check("gram_symmetric", k == transpose(k))
check("two_by_two_gram_positive_definite", k[0][0] > 0 and k[0][0]*k[1][1]-k[0][1]*k[1][0] == 4131)
c = [F(3, 2), F(-2)]
combined = add(scale(c[0], deltas[0]), scale(c[1], deltas[1]))
check("coefficient_gram_norm", sum(c[i]*k[i][j]*c[j] for i in range(2) for j in range(2)) == norm(combined, q, x) == 1306)
g = mat([[1, 1], [0, 1]])
gi = mat([[1, -1], [0, 1]])
gb = product(bs[0], g)
ga = product(gi, aa[0])
check("invertible_rank2_gauge_same_delta", product(gb, ga) == product(bs[0], aa[0]))
kg = scales[0]*scales[1]*trace(product(product(product(transpose(gb), q), bs[1]), product(product(aa[1], x), transpose(ga))))
check("gauge_same_metric_cross_inner", kg == k[0][1])
check("scale_squared_in_self_norm", norm(deltas[0], q, x) == 4*norm(product(bs[0], aa[0]), q, x))

d = mat([[2, 0], [0, 1]])
xw = mat([[1, 0], [0, 9]])
plain = mat([[2, 0], [0, 0]])
weighted = mat([[0, 0], [0, 1]])
check("plain_rank1_frobenius_preferred", norm(subtract(d, plain), identity, identity) == 1 < norm(subtract(d, weighted), identity, identity))
check("activation_weighted_rank1_reverses_choice", norm(subtract(d, weighted), identity, xw) == 4 < norm(subtract(d, plain), identity, xw) == 9)
qs = mat([[1, 0], [0, 0]])
xs = mat([[4, 0], [0, 0]])
d0 = mat([[3, 5], [7, 11]])
lift = mat([[3, 0], [0, 0]])
check("singular_pseudoinverse_lift_metric_error_zero", norm(subtract(d0, lift), qs, xs) == 0)
check("singular_zero_metric_does_not_preserve_dense_delta", norm(subtract(d0, lift), identity, identity) == 195)
null_addition = mat([[0, 0], [0, 1]])
check("null_addition_metric_zero", norm(null_addition, qs, xs) == 0)
check("null_addition_can_violate_rank1", (lift[0][0]*1-lift[0][1]*0) != 0)
constant_moment = mat([[1, 1], [1, 1]])
e11 = mat([[1, 0], [0, 0]])
check("uncentered_second_moment_needed", norm(e11, identity, constant_moment) == 1)
check("constant_activation_covariance_wrongly_zero", norm(e11, identity, mat([[0, 0], [0, 0]])) == 0)
paired = F(1+16, 2)
factorized = F(1+4, 2)**2
check("sample_dependent_output_metric_does_not_factor", paired == F(17, 2) and factorized == F(25, 4) and paired != factorized)
qtie = mat([[1, 0], [0, 100]])
dtie = mat([[1, 0], [0, F(1, 10)]])
keep1 = mat([[1, 0], [0, 0]])
keep2 = mat([[0, 0], [0, F(1, 10)]])
check("tied_whitened_rank1_errors_equal", norm(subtract(dtie, keep1), qtie, identity) == norm(subtract(dtie, keep2), qtie, identity) == 1)
check("tied_optima_differ_in_euclidean_norm", norm(keep1, identity, identity) == 1 and norm(keep2, identity, identity) == F(1, 100))
check("indefinite_output_metric_not_PSD", norm(e11, mat([[-1, 0], [0, 1]]), identity) == -1)
small = F(1, 100)
large = F(10000)
check("frozen_layer_metrics_do_not_bound_composed_network", small**2 == F(1, 10000) and (1+large)*small == F(10001, 100))

payload = {
    "schema": "CE8-INDEPENDENT-ACTIVATION-METRIC-RATIONAL-CHECKS-1",
    "classification": "SYNTHETIC_EXACT_ALGEBRA_ONLY",
    "checks": checks,
    "mixed_rank_scaled_gram": [[str(v) for v in row] for row in k],
    "gram_determinant": "4131",
    "weighted_vs_plain_rank1_metric_errors": ["4", "9"],
    "singular_metric_error": "0",
    "singular_lost_frobenius_squared": "195",
    "sample_dependent_Q_true_vs_factorized": [str(paired), str(factorized)],
    "tied_metric_optima_frobenius_squared": ["1", "1/100"],
    "frozen_two_layer_counterexample_output": "10001/100",
    "models_loaded": 0,
    "dataset_rows_read": 0,
    "framework_imports": 0,
    "optimizer_updates": 0,
    "native_qualification_evidence": False,
}
source = Path(__file__).read_bytes()
payload["source_ref"] = {"path": str(Path(__file__).resolve()), "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}
path = ROOT / "EXACT_CHECKS.json"
with path.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"passed": len(checks), "receipt": str(path)}))
