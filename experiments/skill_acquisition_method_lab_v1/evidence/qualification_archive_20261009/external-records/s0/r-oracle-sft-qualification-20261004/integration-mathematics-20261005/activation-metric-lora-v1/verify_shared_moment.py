"""Exact synthetic shared-moment identities; no frameworks or scientific inputs."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def transpose(a):return list(map(list,zip(*a)))
def multiply(a,b):return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]
def trace(a):return sum((a[i][i] for i in range(len(a))),F(0))
def weighted_error(d,t,x):
 e=[[a-b for a,b in zip(r,s)] for r,s in zip(d,t)]
 return trace(multiply(multiply(transpose(e),e),x))
def ref(name):
 path=ROOT/name;raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
checks=[]
def check(name,value):
 assert value,name
 checks.append({'name':name,'passed':True,'evidence':'EXACT_FRACTION_SYNTHETIC_ALGEBRA_ONLY'})
A=[[F(1),F(2)],[F(2),F(-1)]];B=[[F(1),F(3)],[F(2),F(1)]]
Q=[[F(2),F(1)],[F(1),F(3)]];X=[[F(4),F(1)],[F(1),F(2)]]
C=[[F(2),F(0)],[F(0),F(-3,2)]]
H=multiply(multiply(A,X),transpose(A));BG=multiply(multiply(transpose(B),Q),B)
D=multiply(multiply(B,C),A)
thin=trace(multiply(multiply(multiply(C,BG),C),H))
dense=trace(multiply(multiply(multiply(transpose(D),Q),D),X))
check('shared_response_moment_equals_dense_quadratic',thin==dense)
G=[[F(2),F(0)],[F(0),F(-3)]];Gi=[[F(1,2),F(0)],[F(0),F(-1,3)]]
Hg=multiply(multiply(Gi,H),transpose(Gi));FG=multiply(multiply(transpose(G),BG),G)
check('block_gauge_congruences_preserve_coefficient_energy',trace(multiply(multiply(multiply(C,FG),C),Hg))==thin)
I=[[F(1),F(0)],[F(0),F(1)]];rare=[[F(99,100),F(0)],[F(0),F(1,100)]]
rank1=[[F(1),F(0)],[F(0),F(0)]]
check('weighted_rank1_rare_direction_error_exact',weighted_error(rank1,I,rare)==F(1,100))
check('rare_group_squared_error_one',weighted_error(rank1,I,[[F(0),F(0)],[F(0),F(1)]])==1)
check('two_exact_basis_responses_require_rank2',I[0][0]*I[1][1]-I[0][1]*I[1][0]==1)
result={'schema':'CE8-SHARED-MOMENT-EXACT-EXAMPLES-1','checks':checks,'check_count':len(checks),
 'shared_energy':str(thin),'rare_group_average_error':'1/100','rare_group_conditional_error':'1',
 'source_refs':[ref('EFFICIENCY_AND_GROUP_LIMITS.json'),ref('verify_shared_moment.py')],
 'models_loaded':0,'optimizer_updates':0,'scientific_rows_read':0,'frameworks_loaded':0,'native_API_calls':0,
 'dispatch_eligibility_changed':False}
with (ROOT/'SHARED_MOMENT_EXACT_RECEIPT.json').open('xb') as stream:stream.write(json.dumps(result,sort_keys=True,indent=2).encode()+b'\n')
print(json.dumps({'checks':len(checks),'classification':'EXACT_SYNTHETIC_ALGEBRA_ONLY','energy':str(thin)}))
