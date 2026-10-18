"""Verified original research package for resuming a study across controller fixes.

The outer controller owns resources and the live lease. The original native
research files remain byte-exact; no patched algorithm is injected into the study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def prepare(output):
    from .contracts import load_json, digest, require
    from .fusion_study import _checked
    root,protocol=_checked(output)
    require(protocol['schema_version']=='fuxi.fusion_study.1','unknown isolated study schema')
    require(protocol['protocol_hash']==digest({k:v for k,v in protocol.items() if k!='protocol_hash'}),
            'frozen study protocol changed')
    controller=load_json(root/'CONTROLLER_REQUEST.json')
    require(controller['code']['native']==protocol['code_hashes'],'controller and original research code differ')
    package=root/'frozen_runtime/fuxi_v1'
    for name,expected in protocol['code_hashes'].items():
        require(Path(name).name==name and name.endswith('.py'),'unsafe frozen module name')
        original=(root/'code'/name).read_bytes()
        require(hashlib.sha256(original).hexdigest()==expected,'frozen study module changed: '+name)
        target=package/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(target.read_bytes()==original,'isolated native module changed: '+name)
        else:
            with target.open('xb') as stream:stream.write(original)
    actual={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in package.glob('*.py')}
    require(actual==protocol['code_hashes'],'isolated research package has extra/changed modules')
    return package.parent

def child(workspace, request_path, runtime):
    # Verify with the current launcher before importing any old writer code.
    from .contracts import load_json,require
    request=load_json(request_path);expected=prepare(request['output'])
    require(Path(runtime).resolve()==expected,'frozen child runtime path mismatch')
    for name in list(sys.modules):
        if name=='fuxi_v1' or name.startswith('fuxi_v1.'):
            del sys.modules[name]
    sys.path.insert(0,str(expected))
    from fuxi_v1.unified_runner import _study
    _study(workspace,request)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--request',required=True)
    parser.add_argument('--runtime',required=True)
    args=parser.parse_args()
    child(args.workspace,args.request,args.runtime)
