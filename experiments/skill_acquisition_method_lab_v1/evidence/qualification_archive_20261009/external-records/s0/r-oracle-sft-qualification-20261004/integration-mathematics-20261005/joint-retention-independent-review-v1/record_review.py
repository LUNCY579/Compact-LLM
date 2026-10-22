"""Exact small arithmetic and local proof review; no study data/model work."""
import datetime
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'joint-retention-v1'


def ref(path):
    blob=path.read_bytes()
    return {'path':str(path),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}


document_ref=ref(ROOT/'JOINT_RETENTION.md')
source_ref=ref(ROOT/'exact_joint_examples.py')
receipt_ref=ref(ROOT/'EXACT_JOINT_EXAMPLES_RECEIPT.json')
assert document_ref['bytes']==4695 and document_ref['sha256']=='9653deac12749a75fd2e18f92240cbf68e1a52d1f78758cb8496ccad8ba4a122'
assert source_ref['bytes']==2215 and source_ref['sha256']=='5697f88641908c5047d9557c0ef73f5af9a828c395e22b1dd8efaf9e23dda9d2'
assert receipt_ref['bytes']==1140 and receipt_ref['sha256']=='597d2a7f5c1fae25bc87a364f9b12ba00da77cdc263c2bdf253e004c08cdc254'

# Independent exact rational checks of the numerical statements.
u=(F(1),F(1));first=(F(1,2),F(1,2));reverse=(F(0),F(1));joint=(F(0),F(0))
assert first[0]>0 and reverse[1]>reverse[0]
assert sum((a-b)**2 for a,b in zip(joint,u))/2==1
q=lambda x:F(-2)*x+abs(x)+x*x
assert q(F(1))==0 and q(F(1,3))==F(-2,9) and q(F(2))==2
assert F(1)-F(2)+F(1)*(F(-2)+F(2)*F(1)+F(1))==0
assert (F(2)-F(1)*F(-2))/(F(1)+F(1)*F(2)+F(1)*F(1)/abs(F(1)))==1
x,y=F(2,3),F(-1,4);norm2=x*x+y*y
assert (x+norm2)+(-x+norm2)==2*norm2>0

review={
    'schema':'CE8-JOINT-RETENTION-INDEPENDENT-MATHEMATICS-REVIEW-1',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'VALID_PROOFS_WITH_EXPLICIT_KKT_PRIMAL_FEASIBILITY_CORRECTION',
    'document_ref':document_ref,'exact_example_source_ref':source_ref,'exact_example_receipt_ref':receipt_ref,
    'method':'Read-only local algebra and small Fraction arithmetic; no candidate optimizer/model/bank/native API.',
    'verified_claims':[
        'With finite real parameters L_j,sigma_j,epsilon_j>=0 andR>=0, everyq_j isconvexcontinuous; C isclosedconvexnonempty becauseq_j(0)=0. FiniteR makes C boundedandcompact in finite dimensional space.',
        'Symmetric positive definiteM gives a continuous strictlyconvex objective. CompactnonemptyC givesexistence; strictconvexity onconvexC givesuniqueness.',
        'For a finite group family withallepsilon_j>0 andR>0, zero isstrictlyfeasible. Convex Slater qualifies necessity of finite nonnegative multipliers; primal feasibility plus stationarity/dualfeasibility/complementarity is sufficient even without Slater.',
        'Subgradient ofnorm atzero isclosedunitball; eachz_j canbechosenindependently inthatball. At nonzerod the stated implicitmatrixformula follows exactly bycollectingscalarI terms; it isnot a closedformsolver.',
        'Halfspace projections in one sweep give(1/2,1/2) or(0,1), eachviolating theotherconstraint. Jointprojection(0,0) isunique because everyfeasible x,y<=0 hasdistance-square>=2.',
        'For commonC/commonanchor/commonbounds andallowances, nonnegativeweights sum<=1 combinefeasibledonors withzero andstayfeasible byconvexity. This saysnothing aboutacquisition orsequentialanchors/optimizerstate.',
        'Opposingquadraticconstraints addto2||d||^2<=0, soonlyzero isfeasible. The onedimensional KKTexample, mix-2/9 andextrapolation2 areexactlycorrect.',
    ],
    'required_display_correction':{
        'location':'JOINT_RETENTION.md KKTblock',
        'issue':'Displayed equations omitprimalfeasibility. Takenalone, d=u withlambda=nu=0 satisfiesstationarity/complementarity evenwhenu isoutsideC.',
        'add':'d in C; equivalently q_j(d)<=epsilon_j for everyj and||d||<=R.',
        'assumption_precision':'State finite groupfamily andfinite realbounds,R>=0, andMsymmetricpositive definite explicitly. The existing wording naturally intends these assumptions, butKKTanduniquenessclaims should notleave them implicit.',
    },
    'empirical_limitations_verified':[
        'SmoothnessL_j andgradienterrorsigma_j areassumptions, notmeasuredmodelquantities.',
        'Specified loss-group constraints do notcertify allbehavior ordecodingoutcomes; epsilon/R/groups/solvercertification/endpoints requireprospective registration.',
        'Exact arithmetic examples and proof validity establishno meaningfulskillgain, learningqualification, modelpromotion orprotocolchange.',
    ],
    'no_literature_search_required':'Allconclusions derivedlocally fromthe specified functions.',
    'models_loaded':0,'optimizer_updates':0,'scientific_inputs_read':0,'native_api_calls':0,
    'protocol_changed':False,'original_document_or_receipts_modified':False,
    'next_action':'Addthe explicit primalfeasibility line and assumptionprecision to a create-only clarification/corrected mathematics document; retain empirical quantities asunmeasured andseparate any future experiment registration.',
}
path=HERE/'INDEPENDENT_MATHEMATICS_REVIEW.json'
blob=json.dumps(review,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False).encode()+b'\n'
with path.open('xb') as stream:assert stream.write(blob)==len(blob)
print(json.dumps(ref(path),sort_keys=True))
