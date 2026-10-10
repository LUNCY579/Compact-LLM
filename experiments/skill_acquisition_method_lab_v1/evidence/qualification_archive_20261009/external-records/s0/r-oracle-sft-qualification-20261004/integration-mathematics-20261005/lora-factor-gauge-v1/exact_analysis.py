"""Exact toy algebra; no model/data/framework/native qualification imports."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parent
checks = []
def mat(rows): return tuple(tuple(F(x) for x in row) for row in rows)
def transpose(a): return tuple(zip(*a))
def multiply(a,b):
    assert len(a[0]) == len(b)
    return tuple(tuple(sum(x*y for x,y in zip(row,col)) for col in transpose(b)) for row in a)
def scale(c,a): return tuple(tuple(c*x for x in row) for row in a)
def add(a,b): return tuple(tuple(x+y for x,y in zip(ar,br)) for ar,br in zip(a,b))
def dot(a,b): return sum(x*y for ar,br in zip(a,b) for x,y in zip(ar,br))
def trace(a): return sum(a[i][i] for i in range(len(a)))
def check(name,actual,expected):
    assert actual == expected, (name,actual,expected)
    checks.append({'name':name,'passed':True,'evidence':'EXACT_RATIONAL_TOY_ALGEBRA'})
def encode(a): return [[str(x) for x in row] for row in a]

b1,a1 = mat([[1],[0]]),mat([[1,0]])
b2,a2 = mat([[0],[1]]),mat([[0,1]])
d1,d2 = multiply(b1,a1),multiply(b2,a2)
mean_delta = scale(F(1,2),add(d1,d2))
factor_mean = multiply(scale(F(1,2),add(b1,b2)),scale(F(1,2),add(a1,a2)))
check('reconstructed_delta_mean',mean_delta,mat([[F(1,2),0],[0,F(1,2)]]))
check('factor_mean_has_cross_terms',factor_mean,mat([[F(1,4),F(1,4)],[F(1,4),F(1,4)]]))
error = add(factor_mean,scale(-1,mean_delta))
check('factor_mean_squared_frobenius_error',dot(error,error),F(1,4))
gauge_b2,gauge_a2 = scale(-1,b2),scale(-1,a2)
check('rank1_gauge_preserves_delta',multiply(gauge_b2,gauge_a2),d2)
gauge_mean = multiply(scale(F(1,2),add(b1,gauge_b2)),scale(F(1,2),add(a1,gauge_a2)))
check('factor_mean_changes_under_equivalent_gauge',gauge_mean,mat([[F(1,4),-F(1,4)],[-F(1,4),F(1,4)]]))
check('same_delta_factor_dot_can_reverse',dot(b1,scale(-1,b1)),-1)
check('same_delta_reconstructed_dot_is_positive',dot(d1,multiply(scale(-1,b1),scale(-1,a1))),1)
check('actual_deltas_are_frobenius_orthogonal',dot(d1,d2),0)

# Exact concatenation also permits signed coefficients and differing ranks.
c1,c2 = F(3,2),F(-2,3)
bcat = tuple(tuple(c1*x for x in row1)+tuple(c2*x for x in row2) for row1,row2 in zip(b1,b2))
acat = a1+a2
weighted = add(scale(c1,d1),scale(c2,d2))
check('signed_weighted_concatenation_exact',multiply(bcat,acat),weighted)
check('orthogonal_rank1_sum_has_nonzero_determinant',weighted[0][0]*weighted[1][1]-weighted[0][1]*weighted[1][0],-1)

# Trace identity: <BiAi,BjAj>F = tr((Bi^T Bj)(Aj Ai^T)).
b3,a3 = mat([[1,2],[3,-1]]),mat([[2,0,-1],[1,4,3]])
b4,a4 = mat([[2],[-3]]),mat([[1,2,5]])
d3,d4 = multiply(b3,a3),multiply(b4,a4)
check('mixed_rank_trace_identity',trace(multiply(multiply(transpose(b3),b4),multiply(a4,transpose(a3)))),dot(d3,d4))
check('trace_identity_self_norm',trace(multiply(multiply(transpose(b3),b3),multiply(a3,transpose(a3)))),dot(d3,d3))
g = ((dot(d3,d3),dot(d3,d4)),(dot(d4,d3),dot(d4,d4)))
check('weighted_gram_norm_identity',c1*c1*g[0][0]+2*c1*c2*g[0][1]+c2*c2*g[1][1],dot(add(scale(c1,d3),scale(c2,d4)),add(scale(c1,d3),scale(c2,d4))))

# A fixed rank1 cannot retain I2 exactly. This explicit truncation has error1.
full,compressed = add(d1,d2),d1
rank_error = add(compressed,scale(-1,full))
check('rank_restriction_explicit_error',dot(rank_error,rank_error),1)
curvature = F(1000)
retention_loss = lambda d: curvature*(d[1][1]-1)**2/2
check('toy_full_update_retention_loss',retention_loss(full),0)
check('toy_rank1_truncation_retention_loss',retention_loss(compressed),500)

def fp32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
base,da,db = fp32(100000000),fp32(-100000000),fp32(1)
ab,ba = fp32(fp32(base+da)+db),fp32(fp32(base+db)+da)
check('sequential_fp32_order_AB',ab,1.0)
check('sequential_fp32_order_BA',ba,0.0)

source_paths = []
for folder,head in [('ChatBubble-post-training-Mimo','19551ddebc87c93161c303865ce6d0f130de0093'),('ChatBubble-rineya-centralized-reasoning-system-v1','04fe137574cc38c04824b65beea9374f60c9042c')]:
    for relative in ['short-response-model-phase-0-1-2/scripts/phase_9_student_distillation/compose_lora_adapters_v22.py','short-response-model-phase-0-1-2/scripts/phase_4_model_soups/merge_delta_soup.py']:
        path = Path(r'C:\Users\clclc\BusinessProject')/folder/relative
        raw = path.read_bytes()
        source_paths.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'observed_branch_head':head})
result = {
 'schema':'RINEYA-LORA-GAUGE-COMPOSITION-CONDITIONAL-ANALYSIS-1',
 'checks':checks,'native_qualification_evidence':False,'model_loads':0,'optimizer_updates':0,'scientific_rows_read':0,
 'sources':source_paths,
 'primary_papers':[{'title':'LoRA: Low-Rank Adaptation of Large Language Models','url':'https://arxiv.org/abs/2106.09685v2','role':'Low-rank factorization context; algebra below independently derived.'},{'title':'TIES-Merging: Resolving Interference When Merging Models','url':'https://arxiv.org/abs/2306.01708v2','role':'Parameter-space interference context; no observed skill benefit claimed here.'}],
 'derived_results':[
  {'name':'Gauge invariance','statement':'D_i=s_i B_i A_i is invariant under B_i -> B_i R_i, A_i -> R_i^-1 A_i for any invertible R_i. Factor inner products and factor averages generally are not.'},
  {'name':'Factor averaging cross terms','statement':'For barB=sum_i p_i B_i and barA=sum_j q_j A_j, barB barA=sum_ij p_i q_j B_i A_j. This differs from sum_i c_i B_i A_i without further constraints. Even one adapter averaged with itself has squared coefficients unless normalized appropriately.'},
  {'name':'Exact signed concatenation','statement':'D=sum_i c_i s_i B_i A_i = [c_1 s_1 B_1 | ... | c_k s_k B_k] vertical_stack(A_1,...,A_k). Rank(D)<=sum_i rank_i; equality is possible for independent row and column subspaces. Exact representation may require a larger rank.'},
  {'name':'Reconstructed Gram without dense deltas','statement':'G_ij=s_i s_j tr((B_i^T B_j)(A_j A_i^T)); for real coefficients ||sum_i c_i D_i||F^2=c^T G c and G is PSD. Zero deltas have undefined cosine and must be flagged. This measures parameter geometry, not retained skill.'},
  {'name':'Conditional compression loss bound','statement':'For E=Dhat-D, if L has Hessian operator norm <=K on the entire segment from D to Dhat under Frobenius parameter norm, |L(Dhat)-L(D)|<=||gradient L(D)||F ||E||F + K ||E||F^2/2. Without those measured bounds, small factor/delta reconstruction error gives no universal retention guarantee.'},
  {'name':'Floating point permutations','statement':'Real fixed-delta sums commute; sequentially rounded merges need not be bitwise invariant. The FP32 toy base1e8, da=-1e8, db=1 yields AB=1 and BA=0. This does not measure the branch composer on any model.'}
 ],
 'branch_inspection':{'observed_implementation':'Both branches have matching source hashes for B-only weighted sequential merging and FP32 reconstructed-delta averaging. The inspected code avoids direct A/B averaging.','limits':'Static source inspection only; no PEFT/Transformers runtime, anchor provenance, tensor layout, module variant, scale convention, dtype round-trip or model behavior was validated.'},
 'examples':{'reconstructed_mean':encode(mean_delta),'factor_mean':encode(factor_mean),'equivalent_gauge_factor_mean':encode(gauge_mean),'factor_error_squared':'1/4','rank1_compression_loss_increase':'500','FP32_AB':ab,'FP32_BA':ba},
 'prospective_study_requirements':['Keep each delta bound to the same exact anchor, tensor names/shapes and layer convention.','Record LoRA scale, rank, variant and any non-LoRA saved modules; reconstructed weight formula is not sufficient for arbitrary adapter variants.','Measure reconstructed-delta Gram plus functional retention; factor cosine cannot substitute for either.','If rank compression is proposed, register its rank/error policy and separate retention measurements before reading held-out results.','Record a canonical accumulation order/dtype and verify actual tensor canaries before comparison.'],
 'authorization':'Conditional algebra and static source inspection only; no acquisition, DEV, confirmation, promotion or model execution.'
}
raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
with (ROOT/'CONDITIONAL_ANALYSIS.json').open('xb') as stream: stream.write(raw)
print(json.dumps({'checks':len(checks),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'model_loads':0,'optimizer_updates':0}))
