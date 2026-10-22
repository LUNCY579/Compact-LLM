"""Pure path binding requirements, not a permit or Native readiness flag."""
from pathlib import Path
REVIEW=Path(__file__).resolve().parent.parent
ACTOR_ROOT=REVIEW/'wrapper-cpu-timeout-dispatch-correction-source-v2'
DOTNET_ROOT=REVIEW/'isolated-dotnet-runtime-v2'
PYTHON_ROOT=REVIEW/'isolated-bare-python-runtime-v2'
PACKAGE_ROOT=REVIEW/'wrapper-private-runtime-preparation-empty-stdlib-correction-v1'
def require(v,m):
    if v is not True:raise RuntimeError(m)
def validate_v2_target_bindings(target):
    require(type(target) is dict and target['schema']=='CE8-EXACT-ACTOR-TARGET-2','closed target schema2 remains unchanged')
    require(target['actor_source_ref']['path']==str(ACTOR_ROOT/'CpuOuterLauncher.cs')
      and target['actor_source_freeze_ref']['path']==str(ACTOR_ROOT/'SOURCE_FREEZE.json'),'exact corrected actor source point')
    require(type(target.get('artifact_ref')) is dict and target['artifact_ref']['bytes']>0
      and target['artifact_ref']['path']==str(REVIEW/'cpu-outer-actor-compiled-net8-v4'/'Ce8CpuOuterActor.dll'),'future emitted current artifact required; absence is not readiness')
    require(target['runtime_ref']['path']==str(DOTNET_ROOT/'dotnet.exe'),'absolute private v2 dotnet process image')
    require(target['runtime_package_manifest_ref']['path']==str(PACKAGE_ROOT/'PRIVATE_RUNTIME_PACKAGE.json'),'exact v2 metadata producer')
    require([entry['root'] for entry in target['runtime_namespace_roots']]==[str(DOTNET_ROOT),str(PYTHON_ROOT)],'both closed loader namespaces must rebind together')
    deps=target['runtime_dependency_refs']
    require(type(deps) is dict and len(deps)==193,'same single authoritative193-file census; no v1 aliases')
    for role,base in (('dotnet',DOTNET_ROOT),('python',PYTHON_ROOT)):
        for key,item in deps.items():
            if key.startswith(role+'/'):
                relative=Path(key[len(role)+1:]);require(not relative.is_absolute() and '..' not in relative.parts,'closed relative dependency role')
                require(item['path']==str(base/relative),'runtime dependency byte identity does not authorize stale paths')
    require(deps['dotnet/dotnet.exe']==target['runtime_ref']
      and deps['python/python.exe']['path']==str(PYTHON_ROOT/'python.exe'),'parent runtime and fixed child image share same v2 package')
    return None
