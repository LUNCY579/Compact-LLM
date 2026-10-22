"""Run three unchanged original ARMR assertion functions without optional imports.

This is an explicitly isolated subset, NOT a substituted or passing full ARMR
suite. Reject any source bytes other than the audited integrated-base blobs.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch

EXPECTED={'src/armr/metrics.py':'0f09c20869372272fa78ef4193cf949a32af33db',
          'tests/test_dynamics_and_gates.py':'ce2ded11bf0e659ff4e4461fa08a70b8e37afe07'}
NAMES=('test_known_contracting_trajectory','test_known_oscillating_trajectory',
       'test_gold_does_not_misuse_keyword_as_exact_answer')


def run(root):
    for rel,sha in EXPECTED.items():
        b=(root/rel).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=sha:
            raise ValueError('inherited bytes changed: '+rel)
    spec=importlib.util.spec_from_file_location('audited_armr_metrics',root/'src/armr/metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse((root/'tests/test_dynamics_and_gates.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
    namespace={'torch':torch,'pytest':pytest,'hidden_metrics':module.hidden_metrics,'canonical_answer':module.canonical_answer}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<unchanged ARMR functions>','exec'),namespace)
    passed=[]
    for name in NAMES:namespace[name]();passed.append(name)
    return {'status':'PASS_ISOLATED_SUBSET','passed':len(passed),'failed':0,'functions':passed,
            'audited_git_blob_sha1':EXPECTED,'full_suite_executed':False,
            'limit':'Original function bodies and metric module; unrelated module imports and gate tests not executed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=run(a.root)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'],result['passed'])
