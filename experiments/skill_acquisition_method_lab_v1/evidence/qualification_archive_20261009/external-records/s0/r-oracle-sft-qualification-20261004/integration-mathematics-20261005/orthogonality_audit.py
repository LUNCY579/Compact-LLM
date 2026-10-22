"""Exact rational counterexamples for skill composition; no ML/data/runtime."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
def dot(a, b): return sum((x*y for x, y in zip(a, b)), F(0))
def mv(a, x): return [dot(row, x) for row in a]
def add(a, b): return [x+y for x, y in zip(a, b)]
def scale(a, k): return [x*k for x in a]
def sub(a, b): return add(a, scale(b, -1))
def show(value):
    if isinstance(value, F): return str(value)
    if isinstance(value, list): return [show(item) for item in value]
    if isinstance(value, dict): return {key:show(item) for key,item in value.items()}
    return value
def f32(value): return struct.unpack('<f', struct.pack('<f', value))[0]

def main():
    findings = []
    a,b,j = [F(1),F(0)], [F(0),F(1)], [F(1),F(1)]
    assert dot(a,b)==0 and dot(j,a)==dot(j,b)==1
    assert dot(j,add(a,b))-dot(j,a)==1
    findings.append({'id':'ORTHOGONAL_WEIGHTS_FUNCTIONAL_INTERFERENCE',
        'updates':[a,b], 'jacobian':j, 'euclidean_inner_product':dot(a,b),
        'individual_effects':[dot(j,a),dot(j,b)], 'merge_effect':dot(j,add(a,b)),
        'proof':'For linear f(theta)=theta1+theta2, task A target1 is matched by deltaA; adding orthogonal deltaB changes output to2. Euclidean orthogonality is not a universal retention guarantee.'})
    p=[[F(1),F(1)],[F(0),F(1)]]
    pinv=[[F(1),F(-1)],[F(0),F(1)]]
    ap,bp=mv(p,a),mv(p,b)
    jp=[dot(j,[pinv[row][col] for row in range(2)]) for col in range(2)]
    assert dot(ap,bp)==1 and dot(jp,ap)==dot(jp,bp)==1
    findings.append({'id':'COORDINATE_DEPENDENCE','parameter_map':p,
        'new_updates':[ap,bp],'new_jacobian':jp,'new_inner_product':dot(ap,bp),
        'proof':'Invertible reparameterization changes Euclidean update overlap while preserving both functional effects. Functional diagnostics must transform with the Jacobian/metric.'})
    x1,x2=[F(1),F(-1)],[F(-1),F(1)]
    cov=[[sum(x[i]*x[k] for x in (x1,x2))/2 for k in range(2)] for i in range(2)]
    assert cov==[[F(1),F(-1)],[F(-1),F(1)]]
    assert all(sum(x)==0 and dot(x,x)/2==1 for x in (x1,x2))
    findings.append({'id':'NORMALIZATION_IS_NOT_WHITENING','normalized_samples':[x1,x2],
        'covariance':cov, 'proof':'Each sample already has mean0 and variance1; the dataset covariance is singular and has off-diagonal -1. Centering/scaling does not imply covariance identity. An epsilon merely rescales this example.'})
    d=F(1,10)
    f=lambda t:t[0]*t[1]
    assert f(scale(a,d))==f(scale(b,d))==0 and f(scale(add(a,b),d))==F(1,100)
    findings.append({'id':'ZERO_FIRST_ORDER_EFFECT_NONZERO_INTERACTION',
        'function':'f(theta)=theta1*theta2','anchor':[0,0], 'step':d,
        'individual_effects':[0,0],'merged_effect':F(1,100),
        'proof':'The anchor Jacobian is zero; the mixed Hessian contribution is nonzero. First-order compatibility alone cannot certify finite nonlinear compositions.'})
    ha=[[F(1),F(0)],[F(0),F(0)]]
    hb=[[F(1),F(1,2)],[F(1,2),F(1)]]
    ga=lambda t:sub(mv(ha,t),a)
    gb=lambda t:sub(mv(hb,t),b)
    ua=lambda t:sub(t,scale(ga(t),d))
    ub=lambda t:sub(t,scale(gb(t),d))
    origin=[F(0),F(0)]
    difference=sub(ub(ua(origin)),ua(ub(origin)))
    bracket=sub(mv(hb,ga(origin)),mv(ha,gb(origin)))
    assert dot(ga(origin),gb(origin))==0
    assert difference==scale(bracket,d*d)==[F(-1,100),F(-1,200)]
    findings.append({'id':'TRAINING_PERMUTATION_COMMUTATOR',
        'quadratic_A_hessian':ha,'quadratic_B_hessian':hb,'gradients_at_anchor':[ga(origin),gb(origin)],
        'B_after_A_minus_A_after_B':difference,'eta_squared_bracket':scale(bracket,d*d),
        'formula':'U_B(U_A(theta))-U_A(U_B(theta)) = eta^2*(H_B*g_A-H_A*g_B)+O(eta^3)',
        'proof':'This quadratic example is exact for all eta. Orthogonal initial gradients do not make training updates commute. Adam comparisons additionally need the full optimizer state.'})
    c=F(1,2)
    h=[[F(1),c],[c,F(1)]]
    loss=lambda t:dot(t,mv(h,t))/2
    cross=loss(add(a,b))-loss(a)-loss(b)
    assert cross==c and dot(a,b)==0
    findings.append({'id':'CURVATURE_METRIC','retention_hessian':h,
        'euclidean_overlap':0,'loss_cross_term':cross,
        'formula':'Delta_A^T H_R Delta_B; first-order loss drift is g_R^T Delta, not Delta_A^T Delta_B',
        'proof':'Even at a retention optimum, anisotropic curvature creates a nonzero cross-loss term between Euclidean-orthogonal updates.'})
    margin=F(1,5); epsilon=F(9,100)
    assert 2*epsilon<margin
    baseline=[margin,F(0)]; changed=[margin-epsilon,epsilon]
    assert changed[0]>changed[1]
    findings.append({'id':'GREEDY_MARGIN_RETENTION','margin':margin,'logit_inf_bound':epsilon,
        'remaining_margin':changed[0]-changed[1],
        'proof':'If each logit changes by at most epsilon and the baseline correct-token margin exceeds2epsilon, that token remains the unique greedy choice. Sequence preservation follows by induction only when the bound holds at every identical baseline prefix; average similarity is insufficient.'})
    base=float(2**24); da=-base; db=1.0
    lhs=f32(f32(base+da)+db); rhs=f32(f32(base+db)+da)
    assert lhs==1.0 and rhs==0.0
    findings.append({'id':'FINITE_PRECISION_COMPOSITION_ORDER','base':base,'delta_A':da,'delta_B':db,
        'A_then_B':lhs,'B_then_A':rhs,
        'proof':'Weight addition commutes in exact arithmetic, but nested FP32 rounding does not. A registered summation/rounding rule and exact anchor are part of the scientific intervention.'})
    body={
        'schema':'SKILL-COMPOSITION-EXACT-MATHEMATICS-1',
        'classification':'MATHEMATICAL_COUNTEREXAMPLES_NOT_NATIVE_OR_LEARNING_EVIDENCE',
        'checked_counterexamples':len(findings),'findings':show(findings),
        'papers':[
            {'title':'Multi-Task Model Merging via Adaptive Weight Disentanglement',
             'url':'https://arxiv.org/html/2411.18729v2', 'sections':'3.2 equations5-10',
             'source_summary':'The merging-gap derivation uses a first-order loss approximation and substitutes a scaled task vector for the anchor task gradient. Orthogonality is sufficient within those approximations; the derivation does not establish an unconditional guarantee for Adam-trained nonlinear models.',
             'local_inference':'Record gradient alignment and finite-step curvature as assumptions requiring validation; do not treat low global cosine as acquired or preserved skill.'},
            {'title':'Understanding and Enforcing Weight Disentanglement in Task Arithmetic',
             'url':'https://arxiv.org/html/2604.17078v1','sections':'3.3,4.1,AppendixE.2 equations38-41',
             'source_summary':'The analysis uses NTK linearization and task-feature specialization. Its covariance-to-Euclidean-Gram step additionally assumes approximately whitened inputs.',
             'local_inference':'The exact normalization counterexample shows why whitening must be measured or explicitly assumed; normalization alone does not establish that step.'},
            {'title':'Task Singular Vectors: Reducing Task Interference in Model Merging',
             'url':'https://arxiv.org/abs/2412.00081v3','sections':'abstract',
             'source_summary':'The authors study layer task matrices through singular vectors and propose compression and interference reduction in that representation.',
             'local_inference':'Layer subspaces are a prospective diagnostic alternative; their efficacy on the local models remains unmeasured.'}
        ],
        'study_implications':[
            'Require genuine acquisition evidence before attempting combinations.',
            'Preserve STOP_NO_SIGNAL and failed retention results; geometry cannot overturn them.',
            'Separate weight addition, sequential training order and function composition as different interventions.',
            'Use common exact anchor, frozen arithmetic/runtime and paired per-case retention outcomes.',
            'Prospective function/curvature diagnostics need fresh source-bound budgets and data review; none are dispatched here.'
        ],
        'native_models_loaded':0,'native_optimizer_updates':0,'scientific_banks_read':0,
        'scientific_effect_measured':False,'execution_authority':False,
    }
    raw=json.dumps(body,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    with (ROOT/'EXACT_COUNTEREXAMPLES.json').open('xb') as stream:stream.write(raw)
    print(json.dumps({'checks':len(findings),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
        'native_optimizer_updates':0}))

if __name__=='__main__':main()
