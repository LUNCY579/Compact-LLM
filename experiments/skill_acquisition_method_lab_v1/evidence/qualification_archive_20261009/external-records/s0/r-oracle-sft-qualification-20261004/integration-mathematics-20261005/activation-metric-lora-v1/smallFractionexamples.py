"""Exact synthetic layer algebra; stdlib only, no model/data/runtime APIs."""
from fractions import Fraction as F
from math import isqrt
import hashlib
import json
from pathlib import Path


def mat(rows):
    assert rows and rows[0] and all(len(row) == len(rows[0]) for row in rows)
    return [[F(x) for x in row] for row in rows]


def tr(a):
    return [list(row) for row in zip(*a)]


def mul(a, b):
    assert len(a[0]) == len(b)
    return [[sum((x * y for x, y in zip(row, col)), F(0))
             for col in tr(b)] for row in a]


def plus(a, b):
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scaled(c, a):
    return [[F(c) * x for x in row] for row in a]


def minus(a, b):
    return plus(a, scaled(-1, b))


def trace(a):
    assert len(a) == len(a[0])
    return sum((a[i][i] for i in range(len(a))), F(0))


def metric(a, b, x, q):
    return trace(mul(mul(mul(tr(a), q), b), x))


def thin(bi, ai, si, bj, aj, sj, x, q):
    return F(si) * F(sj) * trace(mul(mul(mul(tr(bi), q), bj),
                                    mul(mul(aj, x), tr(ai))))


def ident(n):
    return mat([[int(i == j) for j in range(n)] for i in range(n)])


def cosine(a, b, x, q):
    aa, bb = metric(a, a, x, q), metric(b, b, x, q)
    if aa == 0 or bb == 0:
        return None
    product = aa * bb
    numerator, denominator = isqrt(product.numerator), isqrt(product.denominator)
    assert numerator * numerator == product.numerator
    assert denominator * denominator == product.denominator
    return metric(a, b, x, q) / F(numerator, denominator)


def det2(a):
    assert len(a) == len(a[0]) == 2
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def normalized(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def run_examples():
    checks, examples = [], {}

    def check(name, condition):
        assert condition, name
        checks.append({"name": name, "passed": True,
                       "evidence": "EXACT_FRACTION_SYNTHETIC_ALGEBRA"})

    # Mixed ranks, signed scales, non-diagonal positive definite input/output
    # metrics. Every arithmetic operation below is rational.
    bi, ai, si = mat([[1], [2]]), mat([[3, -1]]), F(2)
    bj, aj, sj = mat([[1, 0], [1, 1]]), mat([[1, 2], [0, 1]]), F(-1, 2)
    x, q = mat([[2, 1], [1, 3]]), mat([[3, 1], [1, 2]])
    donors = [scaled(si, mul(bi, ai)), scaled(sj, mul(bj, aj))]
    factors = [(bi, ai, si), (bj, aj, sj)]
    gram = [[metric(a, b, x, q) for b in donors] for a in donors]
    factor_gram = [[thin(*first, *second, x, q) for second in factors]
                   for first in factors]
    check("mixed_rank_scaled_thin_Gram_equals_dense_functional_Gram",
          factor_gram == gram)
    check("toy_PSD_metrics_positive_principal_minors",
          x[0][0] > 0 and det2(x) > 0 and q[0][0] > 0 and det2(q) > 0)
    check("two_by_two_Gram_PSD_principal_minors",
          gram[0][0] >= 0 and gram[1][1] >= 0 and det2(gram) >= 0)
    weights = [F(3, 2), F(-2)]
    combined = plus(scaled(weights[0], donors[0]), scaled(weights[1], donors[1]))
    quadratic = sum((weights[i] * gram[i][j] * weights[j]
                     for i in range(2) for j in range(2)), F(0))
    check("weighted_layer_response_norm_equals_Gram_quadratic",
          metric(combined, combined, x, q) == quadratic)
    r, rinv = mat([[1, 1], [0, 1]]), mat([[1, -1], [0, 1]])
    gauged = (mul(bj, r), mul(rinv, aj), sj)
    check("nonorthogonal_gauge_preserves_functional_Gram",
          mul(r, rinv) == ident(2) and
          thin(bi, ai, si, *gauged, x, q) == gram[0][1] and
          thin(*gauged, *gauged, x, q) == gram[1][1])
    examples["mixed_rank_Gram"] = {"X": x, "Q": q, "K": gram,
                                  "det_K": det2(gram), "mixture_norm_squared": quadratic}

    # Uncentered second moment carries a nonzero mean response.
    scalar_d = mat([[3]])
    second, covariance, mean = F(5), F(1), F(2)  # x in {1,3}, equally likely
    whole = metric(scalar_d, scalar_d, mat([[second]]), mat([[1]]))
    centered = metric(scalar_d, scalar_d, mat([[covariance]]), mat([[1]]))
    check("second_moment_equals_covariance_plus_mean_response",
          whole == 45 and centered == 9 and whole - centered == (3 * mean) ** 2)
    check("deterministic_nonzero_input_covariance_is_insufficient",
          metric(mat([[1]]), mat([[1]]), mat([[1]]), mat([[1]])) == 1 and
          metric(mat([[1]]), mat([[1]]), mat([[0]]), mat([[1]])) == 0)
    examples["uncentered_second_moment"] = {"full": whole, "centered_only": centered,
                                          "mean_response_squared": whole - centered}

    i2, one = ident(2), mat([[1]])
    input_first, input_second = mat([[1, 0], [0, 0]]), mat([[0, 0], [0, 1]])
    d1, d2 = mat([[2, 1]]), mat([[2, -1]])
    check("positive_Euclidean_cosine_but_functional_opposition",
          cosine(d1, d2, i2, one) == F(3, 5) and
          cosine(d1, d2, input_second, one) == -1)
    orth1, orth2 = mat([[1, 1]]), mat([[1, -1]])
    check("Euclidean_orthogonality_but_functional_opposition",
          metric(orth1, orth2, i2, one) == 0 and
          cosine(orth1, orth2, input_second, one) == -1)
    neg1, neg2 = mat([[1, 2]]), mat([[1, -2]])
    check("negative_Euclidean_cosine_but_functional_alignment",
          cosine(neg1, neg2, i2, one) == F(-3, 5) and
          cosine(neg1, neg2, input_first, one) == 1)
    invisible = mat([[0, 1]])
    check("nonzero_delta_functionally_irrelevant_cosine_undefined",
          metric(invisible, invisible, i2, one) == 1 and
          metric(invisible, invisible, input_first, one) == 0 and
          cosine(mat([[1, 0]]), invisible, input_first, one) is None)
    examples["cosine_counterexamples"] = {
        "positive_Frobenius": F(3, 5), "its_functional": F(-1),
        "orthogonal_Frobenius_dot": F(0), "its_functional": F(-1),
        "negative_Frobenius": F(-3, 5), "its_aligned_functional": F(1),
        "metric_null_cosine": None}

    n = mat([[0, 5], [0, -7]])
    check("input_nullspace_update_is_identically_unobserved_on_support",
          mul(n, input_first) == mat([[0, 0], [0, 0]]) and
          all(mul(n, point) == mat([[0], [0]])
              for point in (mat([[1], [0]]), mat([[-1], [0]]))))
    output_null = mat([[0, 0], [9, 11]])
    check("output_nullspace_metric_zero_need_not_mean_actual_response_zero",
          metric(output_null, output_null, i2, input_first) == 0 and
          mul(output_null, mat([[1], [0]])) == mat([[0], [9]]))
    examples["nullspaces"] = {"input_null_update": n, "output_null_update": output_null,
                              "ignored_output_response": mat([[0], [9]])}

    # Exact correlated variable-Q counterexample. No empirical samples are read.
    support = [(F(1, 2), F(0)), (F(1, 2), F(2))]
    paired = sum((prob * value ** 4 for prob, value in support), F(0))
    eq = sum((prob * value ** 2 for prob, value in support), F(0))
    exx = eq
    check("input_dependent_Q_moment_decoupling_is_not_exact",
          paired == 8 and eq * exx == 4 and paired != eq * exx)
    examples["input_dependent_output_metric"] = {"paired_true_Gram": paired,
                                                "product_of_moments": eq * exx}

    # Diagonal positive metrics make whitening/SVD/lifting exactly rational.
    target = mat([[2, 0], [0, 1]])
    x_weight = mat([[1, 0], [0, 9]])
    x_root = mat([[1, 0], [0, 3]])
    x_invroot = mat([[1, 0], [0, F(1, 3)]])
    whitened = mul(target, x_root)
    whitened_rank1 = mat([[0, 0], [0, 3]])
    weighted_best = mul(whitened_rank1, x_invroot)
    euclidean_best = mat([[2, 0], [0, 0]])
    err_weighted = metric(minus(weighted_best, target), minus(weighted_best, target), x_weight, i2)
    err_euclidean = metric(minus(euclidean_best, target), minus(euclidean_best, target), x_weight, i2)
    check("PSD_weighted_whitening_and_rank1_lift",
          mul(x_root, x_root) == x_weight and whitened == mat([[2, 0], [0, 3]]) and
          weighted_best == mat([[0, 0], [0, 1]]) and det2(weighted_best) == 0)
    check("weighted_rank1_tail_error_beats_plain_Frobenius_truncation",
          err_weighted == 4 and err_euclidean == 9 and err_weighted == 2 ** 2)
    examples["weighted_compression"] = {"whitened_target": whitened,
        "weighted_rank1": weighted_best, "weighted_error": err_weighted,
        "plain_Frobenius_rank1": euclidean_best, "its_weighted_error": err_euclidean}

    singular_target = mat([[2, 7], [11, 13]])
    projector = input_first
    c = mul(mul(projector, singular_target), projector)
    lifted = mul(mul(projector, c), projector)
    singular_error = metric(minus(lifted, singular_target), minus(lifted, singular_target), projector, projector)
    frob_error = metric(minus(lifted, singular_target), minus(lifted, singular_target), i2, i2)
    check("singular_PSD_pseudoinverse_lift_attains_zero_metric_error",
          c == lifted == mat([[2, 0], [0, 0]]) and singular_error == 0 and frob_error == 339)
    rank_raising_null = mat([[0, 0], [0, 1]])
    check("metric_null_addition_can_violate_rank_constraint",
          metric(rank_raising_null, rank_raising_null, projector, projector) == 0 and
          det2(plus(lifted, rank_raising_null)) == 2)
    examples["singular_compression"] = {"canonical_lift": lifted,
        "metric_error": singular_error, "Frobenius_error": frob_error,
        "null_addition_result_determinant": det2(plus(lifted, rank_raising_null))}

    return {"schema": "ACTIVATION-METRIC-LORA-EXACT-FRACTION-EXAMPLES-1",
            "classification": "EXACT_SYNTHETIC_ALGEBRA_ONLY",
            "arithmetic": "fractions.Fraction and integer isqrt; no floating-point approximation",
            "checks": checks, "check_count": len(checks), "examples": normalized(examples),
            "scope": {"model_loads": 0, "scientific_rows_read": 0,
                      "activation_rows_read": 0, "optimizer_updates": 0,
                      "frameworks_loaded": 0, "Native_API_calls": 0,
                      "grants_created": 0, "dispatch_eligibility_changed": False},
            "limitations": ["Toy identities do not establish neural retention, logits, accuracy or acquisition.",
                            "The general singular-compression optimum follows the written proof; these diagonal examples instantiate it.",
                            "Input-dependent Q requires its joint moment or paired expectation; no empirical moment-decoupling claim."]}


def file_ref(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    root = Path(__file__).resolve().parent
    value = run_examples()
    value["source_refs"] = [file_ref(root / "THEORY.md"), file_ref(Path(__file__).resolve())]
    output = root / "EXACT_FRACTION_RECEIPT.json"
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        assert stream.write(raw) == len(raw)
    print(json.dumps({"receipt": file_ref(output), "check_count": value["check_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
