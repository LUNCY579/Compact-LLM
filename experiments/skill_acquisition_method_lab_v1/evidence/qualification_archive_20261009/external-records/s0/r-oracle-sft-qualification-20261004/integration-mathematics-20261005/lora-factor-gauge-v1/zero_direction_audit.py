"""Inspect actual result method with synthetic scalars, without importing torch."""
import ast,hashlib,json,math,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=Path(r'C:\Users\clclc\BusinessProject\ChatBubble-post-training-Mimo\Rineya_Cross_Scale_Capability_Transfer_Lab_v1\src\rineya_cross_scale\delta_analysis.py')
raw=source.read_bytes();tree=ast.parse(raw)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Accumulator')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='result')
method.returns=None
module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
scope={'math':math};exec(compile(module,str(source),'exec'),scope)
def run(x2,y2,dot):
    record=types.SimpleNamespace(count=1,x2=x2,y2=y2,dot=dot,base2=1.0,abs_x=0.0,abs_y=0.0,abs_x2=0.0,abs_y2=0.0,abs_xy=0.0,sign_agree=0)
    return scope['result'](record)
zero=run(0.0,1.0,0.0);tiny=run(1e-40,1e-40,1e-40)
assert zero['cosine']==0.0 and zero['angular_degrees']==90.0
assert abs(tiny['cosine']-1e-10)<1e-24 and tiny['angular_degrees']>89.99
result={
 'schema':'RINEYA-ACTUAL-RESULT-METHOD-ZERO-DIRECTION-AUDIT-1',
 'source_ref':{'path':str(source),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
 'evidence':'Actual result method compiled from inspected AST; synthetic scalar inputs only; parent module and torch not imported.',
 'cases':[{'name':'one_zero_delta','x2':0.0,'y2':1.0,'dot':0.0,'observed_cosine':zero['cosine'],'observed_angle':zero['angular_degrees'],'mathematical_cosine':None,'reason':'A zero vector has no defined cosine or angle.'},
          {'name':'equal_tiny_nonzero_directions','x2':1e-40,'y2':1e-40,'dot':1e-40,'observed_cosine':tiny['cosine'],'observed_angle':tiny['angular_degrees'],'mathematical_cosine':1.0,'reason':'The absolute1e-30 denominator clamp changes direction geometry for small nonzero norms.'}],
 'conclusion':'The diagnostic must distinguish undefined zero directions from orthogonality. Clamp-induced near90degree angles cannot be treated as evidence of independent skills.',
 'concrete_correction':'In a separately reviewed diagnostic change, flag zero norms and emit null cosine/angle; compute nonzero dot/norm ratios with an explicit stable accumulation policy, and preserve uncertainty when norms underflow. Update downstream formatting to support null values. Do not silently alter previous reports.',
 'model_quality_claim':False,'model_loads':0,'optimizer_updates':0,'scientific_rows_read':0,'branch_sources_modified':False}
out=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
with (ROOT/'ZERO_DIRECTION_AUDIT.json').open('xb') as stream:stream.write(out)
print(json.dumps({'bytes':len(out),'sha256':hashlib.sha256(out).hexdigest(),'synthetic_cases':2,'torch_imported':False}))
