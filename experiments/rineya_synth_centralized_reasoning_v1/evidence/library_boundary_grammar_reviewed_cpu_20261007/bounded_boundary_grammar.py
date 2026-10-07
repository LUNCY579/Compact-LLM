"""Optional CPU-only grammar adapter; exact library replay remains unchanged."""
from pathlib import Path
from types import ModuleType
import hashlib,json
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
SOURCE=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/prefix_library_evidence.py'
def require(value,message):
    if not value:raise ValueError(message)
def engine():
    raw=SOURCE.read_bytes();require(hashlib.sha256(raw).hexdigest()==ENGINE_PIN,'exact original ordered-plan replay source')
    m=ModuleType('CPU_original_library_replay');m.__file__=str(SOURCE);exec(compile(raw,str(SOURCE),'exec'),m.__dict__);return m
def validate_grammar(grammar,policy):
    require(type(grammar) is dict and set(grammar)=={'schema_version','kind','required_extensions','optional_extensions','maximum_boundaries','maximum_depth','maximum_observations','model'} and grammar['schema_version']==2 and grammar['kind']=='SOURCE_BOUND_FINITE_LIBRARY_BOUNDARY_GRAMMAR_V2' and grammar['model'] is False,'exact bounded grammar version and scope')
    required,optional=grammar['required_extensions'],grammar['optional_extensions']
    require(type(required) is list and type(optional) is list and len(required)==len(set(required)) and len(optional)==len(set(optional)) and not set(required).intersection(optional) and set(required)|set(optional)==set(policy['extension_transitions']),'exact finite required/optional extension identities')
    maximum=1+2*len(policy['extension_transitions'])
    require(type(grammar['maximum_boundaries']) is int and grammar['maximum_boundaries']==maximum and 0<maximum<=2048 and grammar['maximum_depth']==64 and type(grammar['maximum_depth']) is int and grammar['maximum_observations']==4096 and type(grammar['maximum_observations']) is int,'original finite event/depth bounds retained')
    return grammar
def derive_plan(raw,*,policy,grammar,expected_file_sha256):
    E=engine();validate_grammar(grammar,policy)
    require(type(raw) is bytes and 0<len(raw)<=E.MAX_PACKET_BYTES and hashlib.sha256(raw).hexdigest()==expected_file_sha256,'exact bounded externally selected raw packet')
    packet=E.parse(raw);rows=packet['observations'];require(type(rows) is list and len(rows)<=4096,'bounded grammar observations')
    plan=[];stack=[];created=set();executed=set();entered=set();framework=False;terminal=False
    for row in rows:
        event=row['event']
        if event=='enter':
            require(not terminal and len(plan)<grammar['maximum_boundaries'] and len(stack)<64,'bounded open grammar entry')
            boundary=row.get('boundary');require(type(boundary) is dict and set(boundary)=={'id','parent_id','name'},'actual emitted boundary name/parent required by grammar')
            bid=boundary['id'];parent=stack[-1]['id'] if stack else None;name=boundary['name']
            require(type(bid) is int and bid==len(plan)+1 and row['boundary_id']==bid and boundary['parent_id']==parent,'sequential source boundary IDs and exact LIFO parent')
            if name=='framework_import':require(not framework and not stack and not plan,'one first framework root');framework=True
            else:
                require(framework and stack and type(name) is str and name.startswith('extension:'),'extensions only inside original framework root')
                path,sep,operation=name[len('extension:'):].rpartition(':');require(sep and path in policy['extension_transitions'] and operation in ('create','exec') and (path,operation) not in entered,'finite source-bound single extension operation')
                require(operation!='exec' or path in created,'exec follows actual returned create')
                entered.add((path,operation))
            plan.append(dict(boundary));stack.append(dict(boundary))
        elif event in ('exit','boundary_error'):
            boundary=row.get('boundary');require(stack and boundary==stack[-1] and row['boundary_id']==stack[-1]['id'],'exact LIFO grammar return/failure')
            top=stack.pop();name=top['name']
            if event=='exit' and name!='framework_import':
                path,_,operation=name[len('extension:'):].rpartition(':');(created if operation=='create' else executed).add(path)
        elif event in ('precleanup','cleanup'):
            require(not stack,'terminal only after every boundary closes');terminal=True
        else:require(event=='initial','known bounded grammar checkpoint')
    success=packet.get('terminal',{}).get('status')=='complete'
    require(not stack,'no unclosed boundary grammar state')
    if success:require(framework and terminal and created==executed and set(grammar['required_extensions'])<=created&executed,'all entered and required extension pairs completed')
    return plan
def validate_observed(raw,*,policy,grammar,expected_identity,selected_file_map,selected_source_map,expected_file_sha256):
    E=engine();plan=derive_plan(raw,policy=policy,grammar=grammar,expected_file_sha256=expected_file_sha256)
    result=E.validate_observed_packet(raw,policy=policy,expected_identity=expected_identity,selected_file_map=selected_file_map,selected_source_map=selected_source_map,expected_boundaries=plan,expected_file_sha256=expected_file_sha256)
    result.update(boundary_registration='CPU_OPTIONAL_GRAMMAR_V2_CANDIDATE',actual_topology_checked_against_fixed_grammar=True,predeclared_order_required=False,exact_library_effect_replay_unchanged=True,native_authority_issued=False,native_PASS=False)
    return result
ENGINE_PIN='aebab23452fa53ad71edbd4762620d929b4b8a936d41eb278f3b1f15ff1a57f2'
