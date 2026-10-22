"""Finite independent exact algebra review; synthetic rational matrices only."""
import ast
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ref(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def mm(a, b):
    assert len(a[0]) == len(b)
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def t(a):
    return list(map(list, zip(*a)))


def mat(rows):
    return [[F(x) for x in row] for row in rows]


def sub(a, b):
    return [[x-y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def trace(a):
    return sum((a[i][i] for i in range(len(a))), F(0))


def inner(a, b, x, q):
    return trace(mm(mm(mm(t(a), q), b), x))


def same_ref(record):
    return ref(record["path"]) == record


freeze = json.loads((ROOT / "SOURCE_FREEZE.json").read_text())
assert all(same_ref(v) for v in freeze["files"].values())
assert all(same_ref(v) for v in freeze["prior_context_read"].values())
mapping = json.loads((ROOT / "SOURCE_MAPPING.json").read_text())
assert all(same_ref(s["source_ref"]) for s in mapping["sources"])
assert all(same_ref(v) for v in mapping["instruction_refs"])
assert same_ref(mapping["prior_gauge_analysis_ref"])
shared_saved = json.loads((ROOT / "SHARED_MOMENT_EXACT_RECEIPT.json").read_text())
assert all(same_ref(v) for v in shared_saved["source_refs"])
fraction_saved = json.loads((ROOT / "EXACT_FRACTION_RECEIPT.json").read_text())
assert all(same_ref(v) for v in fraction_saved["source_refs"])

# Load only source-defined pure functions/imports: exclude create-only CLI writers.
src = (ROOT / "smallFractionexamples.py").read_text()
tree = ast.parse(src)
tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
namespace = {}
exec(compile(tree, str(ROOT / "smallFractionexamples.py"), "exec"), namespace)
rerun = namespace["run_examples"]()
assert rerun == {k: v for k, v in fraction_saved.items() if k != "source_refs"}

shared_src = (ROOT / "verify_shared_moment.py").read_text()
shared_tree = ast.parse(shared_src)
# Execute exact arithmetic body through result creation, never the receipt writer.
shared_tree.body = [node for node in shared_tree.body
                    if not isinstance(node, ast.With)
                    and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Name) and node.value.func.id == "print")]
shared_ns = {"__file__": str(ROOT / "verify_shared_moment.py")}
exec(compile(shared_tree, str(ROOT / "verify_shared_moment.py"), "exec"), shared_ns)
assert shared_ns["result"] == shared_saved

checks = []


def check(name, condition):
    assert condition, name
    checks.append({"name": name, "passed": True, "evidence": "INDEPENDENT_EXACT_SYNTHETIC_FRACTION_ALGEBRA"})


# Non-diagonal singular supports: Q=25P_u, X=4P_v; roots/inverse-roots rational.
u = mat([[F(3,5)], [F(4,5)]]); v = mat([[F(5,13)], [F(12,13)]])
pq = mm(u,t(u)); px = mm(v,t(v))
q = [[25*z for z in row] for row in pq]
x = [[4*z for z in row] for row in px]
qr = [[5*z for z in row] for row in pq]
xr = [[2*z for z in row] for row in px]
qi = [[z/5 for z in row] for row in pq]
xi = [[z/2 for z in row] for row in px]
target = mat([[2,7],[11,13]])
c = mm(mm(qr,target),xr)
lift = mm(mm(qi,c),xi)
check("non_diagonal_singular_square_roots", mm(qr,qr)==q and mm(xr,xr)==x)
check("pseudoinverse_root_support_projectors",mm(qr,qi)==pq and mm(xr,xi)==px)
check("supported_whitened_target",mm(mm(pq,c),px)==c)
check("rank_one_supported_lift", lift[0][0]*lift[1][1]-lift[0][1]*lift[1][0]==0)
check("non_diagonal_singular_lift_zero_weighted_error",inner(sub(lift,target),sub(lift,target),x,q)==0)
check("pseudoinverse_lift_reconstructs_whitened_map",mm(mm(qr,lift),xr)==c)
check("rank_zero_tail_energy",inner(target,target,x,q)==trace(mm(t(c),c)))

# Shared basis with mixed ranks 1 and 2; non-orthogonal within rank-2 gauge.
a = mat([[1,2],[2,-1],[0,3]]); b = mat([[1,3,-2],[2,1,1]])
cc = mat([[2,0,0],[0,F(-3,2),0],[0,0,F(-3,2)]])
xx = mat([[4,1],[1,2]]); qq = mat([[2,1],[1,3]])
h = mm(mm(a,xx),t(a)); f = mm(mm(t(b),qq),b)
d = mm(mm(b,cc),a)
energy = trace(mm(mm(mm(cc,f),cc),h))
check("mixed_rank_shared_basis_energy",energy==inner(d,d,xx,qq))
g = mat([[2,0,0],[0,1,2],[0,0,1]])
gi = mat([[F(1,2),0,0],[0,1,-2],[0,0,1]])
hg=mm(mm(gi,h),t(gi)); fg=mm(mm(t(g),f),g)
check("nonorthogonal_mixed_rank_block_gauge_inverse",mm(g,gi)==mat([[1,0,0],[0,1,0],[0,0,1]]))
check("scalar_block_coefficient_commutes_with_gauge",mm(cc,g)==mm(g,cc))
check("nonorthogonal_mixed_rank_block_gauge_energy",trace(mm(mm(mm(cc,fg),cc),hg))==energy)

result = {
    "schema": "CE8-ACTIVATION-METRIC-INDEPENDENT-EXACT-REVIEW-1",
    "classification": "MECHANICAL_EXACT_SYNTHETIC_ALGEBRA_ONLY",
    "source_refs": [ref(ROOT/name) for name in ("SOURCE_FREEZE.json","THEORY.md","smallFractionexamples.py","EXACT_FRACTION_RECEIPT.json","EFFICIENCY_AND_GROUP_LIMITS.json","verify_shared_moment.py","SHARED_MOMENT_EXACT_RECEIPT.json","SOURCE_MAPPING.json","PRIMARY_PAPER_CONNECTIONS.json","independent_exact_review.py")],
    "reference_revalidation": {"core":3,"prior":4,"mapping_sources":30,"instruction":1,"mapping_prior":1,"receipt_refs":4,"all_match":True},
    "original_fraction_checks_rerun":len(rerun["checks"]),
    "original_shared_moment_checks_rerun":len(shared_ns["checks"]),
    "additional_independent_checks":checks,
    "additional_independent_check_count":len(checks),
    "models_loaded":0,"optimizer_updates":0,"scientific_rows_read":0,"activation_rows_read":0,
    "frameworks_loaded":0,"native_api_calls":0,"grants_created":0,"eligibility_changed":False
}
out=ROOT/"INDEPENDENT_EXACT_REVIEW_RECEIPT.json"
with out.open("xb") as stream:
    stream.write((json.dumps(result,sort_keys=True,indent=2)+"\n").encode())
print(json.dumps({"receipt":ref(out),"fraction_rerun":18,"shared_rerun":5,"additional":len(checks)}))
