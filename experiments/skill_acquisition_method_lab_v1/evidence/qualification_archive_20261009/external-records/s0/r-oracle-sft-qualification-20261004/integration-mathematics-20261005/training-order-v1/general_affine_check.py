"""Exact check of the general affine Euler commutator; no model operations."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).parent
def plus(x,y):return tuple(a+b for a,b in zip(x,y))
def minus(x,y):return tuple(a-b for a,b in zip(x,y))
def times(a,x):return tuple(a*v for v in x)
def mv(h,x):return tuple(sum(a*b for a,b in zip(row,x)) for row in h)
def mm(a,b):return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)) for i in range(2))
def mminus(a,b):return tuple(minus(x,y) for x,y in zip(a,b))
def step(t,h,g,e):return minus(t,times(e,plus(mv(h,t),g)))
ha=((F(1),F(0)),(F(0),F(2)))
hb=((F(3),F(1)),(F(1),F(4)))
a=(F(1),F(0));b=(F(0),F(1))
bracket=mminus(mm(hb,ha),mm(ha,hb))
assert bracket==((F(0),F(1)),(F(-1),F(0)))
translation=minus(mv(hb,a),mv(ha,b))
assert translation==(F(3),F(-1))
checks=[]
for theta in ((F(0),F(0)),(F(2,3),F(-5,7)),(F(-1),F(4))):
    for alpha,beta in ((F(1,4),F(1,4)),(F(2,9),F(3,8)),(F(0),F(1,3))):
        actual=minus(step(step(theta,ha,a,alpha),hb,b,beta),step(step(theta,hb,b,beta),ha,a,alpha))
        predicted=times(alpha*beta,plus(mv(bracket,theta),translation))
        assert actual==predicted
        checks.append({'theta':[str(v) for v in theta],'alpha':str(alpha),'beta':str(beta),
                       'difference':[str(v) for v in actual],'identity_exact':True})
analysis=ROOT/'CONDITIONAL_ANALYSIS.json';raw_analysis=analysis.read_bytes()
out={'schema':'EXACT-GENERAL-AFFINE-COMMUTATOR-CHECK-1',
     'classification':'EXACT_RATIONAL_TOY_MATHEMATICS_ONLY',
     'original_analysis_ref':{'path':str(analysis),'bytes':len(raw_analysis),'sha256':hashlib.sha256(raw_analysis).hexdigest()},
     'noncommuting_Hessian_bracket':[[str(v) for v in row] for row in bracket],
     'affine_translation':[str(v) for v in translation],'checks':checks,
     'proof':'For quadratics write T_i(theta)=(I-eta_i*H_i)theta-eta_i*a_i. Multiplying and subtracting the two affine maps yields alpha*beta*((H_B*H_A-H_A*H_B)theta+H_B*a-H_A*b). Equality for every theta at nonzero alpha*beta is equivalent to both displayed coefficients vanishing.',
     'scope':'Checks corroborate the algebra at finite rational points; the coefficient identity provides the general proof. No neural performance, Adam or many-step claim.',
     'physical_optimizer_updates':0,'model_loads':0}
raw=json.dumps(out,sort_keys=True,indent=2).encode()+b'\n'
with (ROOT/'GENERAL_AFFINE_CHECK.json').open('xb') as handle:handle.write(raw)
print(json.dumps({'checks':len(checks),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}))
