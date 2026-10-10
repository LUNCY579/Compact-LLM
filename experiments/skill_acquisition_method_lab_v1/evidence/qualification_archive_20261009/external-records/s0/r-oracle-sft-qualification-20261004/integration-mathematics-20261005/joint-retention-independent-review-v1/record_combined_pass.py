"""Combined conditional mathematics review, no empirical/runtime work."""
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'joint-retention-v1'


def ref(path):
    data=path.read_bytes()
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


doc_ref=ref(ROOT/'JOINT_RETENTION.md');addendum_ref=ref(ROOT/'KKT_FEASIBILITY_ADDENDUM.md')
assert doc_ref['sha256']=='9653deac12749a75fd2e18f92240cbf68e1a52d1f78758cb8496ccad8ba4a122'
assert addendum_ref['bytes']==1376 and addendum_ref['sha256']=='4ab85f87824fc265daa737930e48ce578e6fab8469edb5b91d96f9180d331c69'
assert ref(ROOT/'EXACT_JOINT_EXAMPLES_RECEIPT.json')['sha256']=='597d2a7f5c1fae25bc87a364f9b12ba00da77cdc263c2bdf253e004c08cdc254'
prior_ref=ref(HERE/'INDEPENDENT_MATHEMATICS_REVIEW.json')
assert prior_ref['sha256']=='c00e2903c435e147c8f31ed816b2f86fb5e82a194be93222f9cd834ff26fc1b9'
review={
    'schema':'CE8-JOINT-RETENTION-COMBINED-INDEPENDENT-MATHEMATICS-DECISION-1',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'PASS_CONDITIONAL_MATHEMATICAL_STATEMENTS',
    'original_note_ref':doc_ref,'primal_feasibility_addendum_ref':addendum_ref,
    'prior_independent_review_ref':prior_ref,
    'exact_example_source_ref':ref(ROOT/'exact_joint_examples.py'),
    'exact_example_receipt_ref':ref(ROOT/'EXACT_JOINT_EXAMPLES_RECEIPT.json'),
    'corrected_claims':[
        'Combined KKT system explicitly includes primalfeasibility, stationarity, nonnegative multipliers and complementarity.',
        'Finite-dimensional/finite-group/finiteparameter/R>=0/symmetricSPD assumptions now explicit; convexclosednonemptycompactC givesexistence andstrictconvexobjective givesuniqueness.',
        'Strictallowances andR>0 implyzeroSlater andnecessary/sufficientKKT; degenerateboundary cases do notclaimnecessaryfinite multipliers. PrimalfeasibleKKT retains sufficiency.',
        'Zero-norm subgradient unitball andnonzero implicitSPDformula, exactprojection-order/jointminimum, common-anchorconvexmixture andopposingquadraticzeromovement conclusions remaincorrect.',
    ],
    'empirical_limits':[
        'L_j, sigma_j andothermodel/behavioral quantities remainunmeasured assumptions.',
        'Specified loss-group feasibility doesnotprove everybehavior/decoding outcome ormeaningfulacquisition.',
        'Sixexact examples are mathematical evidence only; no experiment, protocol, modelpromotion orqualification resultis claimed.',
    ],
    'models_loaded':0,'optimizer_updates':0,'scientific_inputs_read':0,'native_api_calls':0,
    'registered_protocol_changed':False,'original_note_and_receipt_preserved':True,
    'additional_tests_or_literature_search_performed':False,
    'next_step':'Useonlythe combined note+addendum forprospective mathematical reasoning; anyempirical solver orlearning study still needs separate measured/justified bounds andregistration.',
}
path=HERE/'COMBINED_CONDITIONAL_MATHEMATICS_DECISION.json'
data=json.dumps(review,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False).encode()+b'\n'
with path.open('xb') as stream:assert stream.write(data)==len(data)
print(json.dumps(ref(path),sort_keys=True))
