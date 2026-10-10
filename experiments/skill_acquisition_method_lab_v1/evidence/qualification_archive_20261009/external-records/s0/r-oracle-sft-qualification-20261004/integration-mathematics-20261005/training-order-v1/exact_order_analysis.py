"""Exact mathematical examples; no models, scientific banks or evaluations."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).parent
def add(x,y):return tuple(a+b for a,b in zip(x,y))
def sub(x,y):return tuple(a-b for a,b in zip(x,y))
def scale(a,x):return tuple(a*v for v in x)
def mv(h,x):return tuple(sum(a*b for a,b in zip(row,x)) for row in h)
def dot(x,y):return sum(a*b for a,b in zip(x,y))
HA=((F(1),F(0)),(F(0),F(2)))
HB=((F(3),F(0)),(F(0),F(4)))
a=(F(1),F(0));b=(F(0),F(1));zero=(F(0),F(0))
eta=F(1,4)
def step(theta,h,g,rate):return sub(theta,scale(rate,add(mv(h,theta),g)))
def LA(theta):return (theta[0]+1)**2/2+theta[1]**2
def LB(theta):return 3*theta[0]**2/2+2*(theta[1]+F(1,4))**2
deltaA=step(zero,HA,a,eta);deltaB=step(zero,HB,b,eta)
merged=add(deltaA,deltaB)
ab=step(deltaA,HB,b,eta);ba=step(deltaB,HA,a,eta)
commutator=scale(eta*eta,sub(mv(HB,a),mv(HA,b)))
assert dot(a,b)==dot(a,mv(HA,b))==dot(a,mv(HB,b))==0
assert sub(ab,ba)==commutator==(F(3,16),F(-1,8))
assert merged==(F(-1,4),F(-1,4)) and add(deltaB,deltaA)==merged
assert (LA(zero),LB(zero))==(F(1,2),F(1,8))
assert (LA(merged),LB(merged))==(F(11,32),F(3,32))
assert (LA(ab),LB(ab))==(F(257,512),F(3,512))
assert (LA(ba),LB(ba))==(F(19,64),F(1,8))
assert LA(ab)>LA(zero) and LB(merged)<LB(zero) and LA(merged)<LA(zero)
checks=[]
# Exact affine identity at several anchors/rates, not floating-point fits.
for theta in (zero,(F(2,3),F(-5,7)),(F(-1),F(4))):
    for alpha,beta in ((F(1,4),F(1,4)),(F(2,9),F(3,8)),(F(0),F(1,3))):
        first=step(step(theta,HA,a,alpha),HB,b,beta)
        second=step(step(theta,HB,b,beta),HA,a,alpha)
        predicted=scale(alpha*beta,sub(mv(HB,add(mv(HA,theta),a)),mv(HA,add(mv(HB,theta),b))))
        assert sub(first,second)==predicted
        checks.append({'theta':[str(v) for v in theta],'alpha':str(alpha),'beta':str(beta),'exact_identity':True})
def row(name,theta):return {'construction':name,'theta':[str(v) for v in theta],
    'L_A':str(LA(theta)),'L_B':str(LB(theta)),
    'delta_L_A_from_anchor':str(LA(theta)-LA(zero)),
    'delta_L_B_from_anchor':str(LB(theta)-LB(zero))}
output={
    'schema':'EXACT-CONDITIONAL-TRAINING-ORDER-ANALYSIS-1',
    'classification':'EXACT_RATIONAL_TOY_MATHEMATICS; NO_CAPABILITY_OR_MODEL_EVIDENCE',
    'primary_paper_context':[
        {'title':'Editing Models with Task Arithmetic','url':'https://arxiv.org/abs/2212.04089',
         'relevance':'Defines fixed task vectors by subtracting shared pretrained weights from fine-tuned weights; addition of fixed vectors is the composition considered here.'},
        {'title':'Model soups','url':'https://arxiv.org/abs/2203.05482',
         'relevance':'Studies averaging separately fine-tuned model weights; its empirical results do not prove training-order invariance.'}],
    'original_derivation':{
        'definitions':'g_i=gradient L_i(theta), H_i=Hessian L_i(theta), T_i^eta(theta)=theta-eta*g_i(theta)',
        'order_convention':'AB means A first, then B: T_B^beta(T_A^alpha(theta)).',
        'leading_difference':'AB-BA = alpha*beta*(H_B*g_A-H_A*g_B)+R',
        'remainder_assumptions':'H_A,H_B are Lipschitz with constants rho_A,rho_B along their respective straight update segments; alpha,beta are nonnegative.',
        'remainder_bound':'norm(R) <= (alpha*beta^2*rho_A*norm(g_B)^2 + beta*alpha^2*rho_B*norm(g_A)^2)/2',
        'proof_steps':[
            'Subtract the two exact Euler compositions; the two anchor-gradient terms cancel.',
            'Expand g_A(theta-beta*g_B) and g_B(theta-alpha*g_A) with the integral Hessian remainder.',
            'A Lipschitz Hessian bounds each gradient remainder by rho*norm(update)^2/2; multiply by the outer step size and use the triangle inequality.'],
        'quadratic_exactness':'For quadratic losses rho_A=rho_B=0; the displayed difference is exact at every theta.',
        'quadratic_global_commutation_conditions':'For nonzero alpha*beta, equality for every theta requires H_B*H_A=H_A*H_B and H_B*a=H_A*b.',
        'fixed_delta_addition':'theta0+alpha*delta_A+beta*delta_B is permutation invariant in exact arithmetic when both deltas are fixed at the same anchor. This identity makes no performance guarantee and is separate from finite-precision accumulation.'},
    'counterexample':{
        'L_A':'(theta_1+1)^2/2 + theta_2^2',
        'L_B':'3*theta_1^2/2 + 2*(theta_2+1/4)^2',
        'H_A':[['1','0'],['0','2']],'H_B':[['3','0'],['0','4']],
        'anchor_gradients':{'g_A':['1','0'],'g_B':['0','1']},
        'orthogonal_under':'Euclidean, H_A, H_B and H_A+H_B metrics at the anchor.',
        'Hessians_commute':True,'both_losses_strictly_convex':True,
        'step_sizes':'alpha=beta=1/4','order_difference':['3/16','-1/8'],
        'outcomes':[row('anchor',zero),row('fixed_delta_sum',merged),row('A_then_B',ab),row('B_then_A',ba)],
        'interpretation':'Even commuting positive Hessians and orthogonal anchor gradients do not imply order invariance. A then B worsens objective A relative to the anchor, while the fixed anchored delta sum improves both toy objectives.'},
    'exact_identity_checks':checks,
    'scope_limits':['Toy quadratic objectives only; no neural model/data/quality measurements.',
                    'One Euler update per task; no claim for Adam, clipping, momentum, many-step trajectories or stochastic data order.',
                    'No acquisition registration or experiment executed.'],
    'model_loads':0,'physical_optimizer_updates':0,'scientific_banks_read':0,
}
raw=json.dumps(output,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
with (ROOT/'CONDITIONAL_ANALYSIS.json').open('xb') as handle:handle.write(raw)
print(json.dumps({'path':str(ROOT/'CONDITIONAL_ANALYSIS.json'),'bytes':len(raw),
    'sha256':hashlib.sha256(raw).hexdigest(),'exact_checks':len(checks),
    'outcomes':output['counterexample']['outcomes']}))
