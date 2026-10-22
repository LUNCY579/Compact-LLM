"""Exact rational counterexamples; mathematical evidence, no model execution."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def two(a, b, k):
    return (a - k*a*b, b - k*a*b)


def mixed(a, b, k):
    joint, axis_a, axis_b, base = two(a,b,k),two(a,0,k),two(0,b,k),two(0,0,k)
    return tuple(joint[i]-axis_a[i]-axis_b[i]+base[i] for i in range(2))


def triple(a,b,c,k):
    return a+b+c-k*a*b*c


def main():
    cases=[]
    k=Q(100)
    # Exact teacher/student matching on every point of both axes, even when
    # reference inputs cover both toy task families. No interpolation error.
    for a in (Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1)):
        assert two(a,0,k)==(a,Q(0))
        assert two(0,a,k)==(Q(0),a)
    assert two(Q(1),Q(1),k)==(Q(-99),Q(-99))
    cases.append({'name':'zero_along_axis_distillation_loss_unbounded_mixed_damage',
        'axis_identity':'holds algebraically for every real scale, not only checked grid',
        'joint_output':two(Q(1),Q(1),k),'mixed_error':mixed(Q(1),Q(1),k)})
    a,b=Q(1,2),Q(1,4)
    residual=mixed(a,b,k)
    assert residual==(Q(-25,2),Q(-25,2))
    assert max(map(abs,residual))==abs(a*b)*k
    cases.append({'name':'rectangle_integral_bound_attained',
        'scales':[a,b],'kappa_infinity_norm':k,'residual':residual,'bound':abs(a*b)*k})
    # Anchor Jacobians are J_task1=(1,0), J_task2=(0,1). Their equally
    # weighted exact GGN is diag(1/2,1/2); there is no cross Gram term.
    gram=((Q(1,2),Q(0)),(Q(0),Q(1,2)))
    u,v=(Q(1),Q(0)),(Q(0),Q(1))
    euclidean=sum(x*y for x,y in zip(u,v))
    metric_cross=sum(u[i]*gram[i][j]*v[j] for i in range(2) for j in range(2))
    assert euclidean==metric_cross==0 and mixed(Q(1),Q(1),k)!=(0,0)
    cases.append({'name':'exact_anchor_gram_and_weight_orthogonality_do_not_control_curvature',
        'gram':gram,'euclidean_cross':euclidean,'GGN_cross':metric_cross,'mixed_error':mixed(Q(1),Q(1),k)})
    forward=two(Q(0),Q(1),k)[0]-two(Q(0),Q(0),k)[0]
    after_other=two(Q(1),Q(1),k)[0]-two(Q(1),Q(0),k)[0]
    assert forward==0 and after_other==-100 and after_other-forward==-100
    cases.append({'name':'layer_marginal_depends_on_other_layer_context',
        'base_marginal':forward,'after_other_layer_marginal':after_other,'difference':after_other-forward})
    for a,b,c in ((1,1,0),(1,0,1),(0,1,1)):
        assert triple(Q(a),Q(b),Q(c),k)==a+b+c
    joint=triple(Q(1),Q(1),Q(1),k)
    assert joint==-97 and joint-3==-100
    # Full-box mixed second derivatives have magnitude at most100, yielding
    # a valid three-pair bound300. Pair faces at the anchor alone see zero.
    assert abs(joint-3)<=3*k
    cases.append({'name':'all_pair_faces_exact_yet_three_way_composition_fails',
        'joint_output':joint,'interaction_error':joint-3,'full_box_pair_bound':3*k,
        'anchor_pair_face_interaction_errors':[Q(0),Q(0),Q(0)]})
    own_logits=(Q(1),Q(0));composite_logits=(Q(-99),Q(0))
    margin=own_logits[0]-own_logits[1];eta=Q(100)
    assert margin>0 and composite_logits[0]<composite_logits[1] and not margin>2*eta
    cases.append({'name':'classification_requires_margin_over_mixed_drift',
        'individual_logits':own_logits,'merged_logits':composite_logits,
        'individual_margin':margin,'drift_bound':eta,'sufficient_preservation_condition':False})
    def convert(value):
        if isinstance(value,Q):return str(value)
        if isinstance(value,(list,tuple)):return [convert(x) for x in value]
        if isinstance(value,dict):return {key:convert(item) for key,item in value.items()}
        return value
    raw_source=Path(__file__).read_bytes()
    receipt={'schema':'CE8-EXACT-MIXED-INTERACTION-MATHEMATICS-1',
        'status':'PASS_SIX_EXACT_RATIONAL_EXAMPLES','source_ref':{'path':str(Path(__file__).resolve()),
        'bytes':len(raw_source),'sha256':hashlib.sha256(raw_source).hexdigest()},
        'cases':convert(cases),'classification':'MATHEMATICAL_MECHANICAL_EVIDENCE_ONLY',
        'models_loaded':0,'optimizer_updates':0,'data_or_evaluation_banks_read':False,
        'empirical_Rineya_or_Titan_claim':False}
    blob=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    assert len(blob)<16384
    with (ROOT/'EXACT_EXAMPLES_RECEIPT.json').open('xb') as stream:
        assert stream.write(blob)==len(blob)
    print(json.dumps({'cases_passed':len(cases),'receipt_path':str(ROOT/'EXACT_EXAMPLES_RECEIPT.json'),
        'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'models_loaded':0,'optimizer_updates':0}))


if __name__=='__main__':main()
