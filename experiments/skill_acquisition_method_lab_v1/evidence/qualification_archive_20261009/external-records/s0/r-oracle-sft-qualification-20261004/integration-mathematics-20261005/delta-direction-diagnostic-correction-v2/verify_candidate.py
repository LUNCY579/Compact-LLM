"""Execute corrected scalar/result/report nodes only; no tensor runtime imports."""
import ast,hashlib,json,math,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
path=ROOT/'delta_analysis.py';raw=path.read_bytes();tree=ast.parse(raw)
geometry=ast.parse((ROOT/'direction_geometry.py').read_bytes())
functions=[node for node in geometry.body if isinstance(node,ast.FunctionDef) and node.name in ('direction_metrics','optional_number')]
cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Accumulator')
result=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='result')
nodes=[*functions,result]
for node in nodes:
    node.returns=None
    for arg in node.args.args:arg.annotation=None
scope={'math':math};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
checks=[]
def yes(name,value):
    assert value,name;checks.append({'name':name,'passed':True,'evidence':'SYNTHETIC_SCALAR_ACTUAL_CANDIDATE_FUNCTIONS'})
def bad(name,work):
    try:work()
    except ValueError:yes(name,True);return
    raise AssertionError(name)
metric=scope['direction_metrics']
for norm in (1.0,1e-20,1e154,1e-150):
    value=metric(norm*norm,norm,norm)
    yes('same_direction_scale_'+str(norm),math.isclose(value['cosine'],1,rel_tol=1e-15) and value['angular_degrees']<1e-5)
yes('opposite_direction',metric(-1,1,1)['cosine']==-1)
yes('orthogonal_nonzero',metric(0,1,1)['angular_degrees']==90)
yes('one_zero_is_undefined',metric(0,0,1)['cosine'] is None and metric(0,0,1)['angular_degrees'] is None)
yes('both_zero_are_undefined',metric(0,0,0)['direction_status']=='undefined_zero_norm')
yes('rounding_uncertainty_keeps_raw_value',metric(1+1e-7,1,1)['raw_cosine_ratio']>1 and metric(1+1e-7,1,1)['cosine'] is None)
bad('zero_norm_inconsistent_dot_refused',lambda:metric(1,0,1))
bad('nonfinite_summary_refused',lambda:metric(float('inf'),1,1))
bad('negative_norm_refused',lambda:metric(0,-1,1))
def state(x2,y2,dot,base2=1):
    return types.SimpleNamespace(count=2,x2=x2,y2=y2,dot=dot,base2=base2,abs_x=0,abs_y=0,abs_x2=0,abs_y2=0,abs_xy=0,sign_agree=0)
zero=scope['result'](state(0,1,0,0))
yes('actual_result_zero_direction',zero['cosine'] is None and zero['direction_status']=='undefined_zero_norm')
yes('zero_anchor_relative_norm_is_undefined',zero['delta4_relative_norm'] is None and zero['delta8_relative_norm'] is None)
tiny=scope['result'](state(1e-40,1e-40,1e-40))
yes('actual_result_tiny_nonzero_geometry',math.isclose(tiny['cosine'],1,rel_tol=1e-15))
yes('undefined_correlation_safe',zero['magnitude_correlation'] is None)
analysis=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='analyze_delta_pair')
start=next(i for i,node in enumerate(analysis.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='global_stats' for t in node.targets))
renderer=ast.FunctionDef(name='render_actual_tail',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='report'),ast.arg(arg='output_markdown')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=analysis.body[start:],decorator_list=[])
capture={};scope.update(groups={'module:zero':zero},atomic_write_text=lambda p,t:capture.update(text=t))
exec(compile(ast.fix_missing_locations(ast.Module(body=[renderer],type_ignores=[])),str(path),'exec'),scope)
scope['render_actual_tail']({'global':zero},'MOCK_OUTPUT')
yes('actual_report_tail_formats_null_metrics','undefined' in capture['text'] and 'Direction status' in capture['text'])
yes('module_nulls_render_without_typeerror','| zero | undefined |' in capture['text'])
head_calls=[node for node in ast.walk(analysis) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='direction_metrics']
yes('head_path_uses_same_metric_policy',len(head_calls)==1)
yes('schema_version_1_1_is_explicit',any(isinstance(node,ast.Constant) and node.value=='1.1' for node in ast.walk(analysis)))
diagnostic=ast.parse((ROOT/'diagnostics.py').read_bytes())
tensor_method=next(node for node in diagnostic.body if isinstance(node,ast.FunctionDef) and node.name=='tensor_delta_stats')
yes('tensor_path_calls_same_geometry',sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='direction_metrics' for node in ast.walk(tensor_method))==1)
yes('legacy_absolute_clamp_removed_in_both_modules',b'1e-30' not in raw and b'1e-30' not in (ROOT/'diagnostics.py').read_bytes())
output={'schema':'RINEYA-DIRECTION-DIAGNOSTIC-CANDIDATE-MECHANICS-1','checks':checks,
 'candidate_refs':[{'path':str(ROOT/name),'bytes':len((ROOT/name).read_bytes()),'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in ('delta_analysis.py','diagnostics.py','direction_geometry.py')],
 'models_loaded':0,'optimizer_updates':0,'tensor_framework_imports':[],'scientific_rows_read':0,
 'limits':'Scalar/result/report source nodes only. No tensor accumulation, checkpoint reading, model behavior, or branch adoption validated.'}
payload=(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
with (ROOT/'MECHANICAL_CHECKS.json').open('xb') as stream:stream.write(payload)
print(json.dumps({'checks':len(checks),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}))
