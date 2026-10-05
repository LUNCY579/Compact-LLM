"""Actual outer-source fixture composition; CPU-only, all hardware trapped."""
from pathlib import Path
import importlib.util
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def source(name):
    path=HERE/name;spec=importlib.util.spec_from_file_location('outer_CPU_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
def selected_documents(package=None):
    helper=source('test_contract_loader.py')
    return helper.C,helper.metadata_fixture(package)
def actual_context_fixture(contract,config_path,anchors,*,read_document,roster,file_check,git):
    """Trap only host seams; execute the actual metadata/factory/readset body."""
    capability=contract._mint_CPU_native_context_fixture()
    with patch.object(contract,'sealed_json',side_effect=read_document),patch.object(contract,'physical_roster',side_effect=roster),patch.object(contract,'file_descriptor',side_effect=file_check),patch('subprocess.run',side_effect=git):
        context=contract._CPU_native_context_fixture(config_path,_capability=capability,**anchors)
    value=contract.context_value(context)
    contract.require(value['fixture_only'] is True and value['native_PASS'] is value['native_authority_issued'] is False,'outer fixture never creates native authority')
    return context
