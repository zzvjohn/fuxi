"""Consume an existing blueprint with its verified frozen module identity."""
from pathlib import Path
import argparse, hashlib, importlib.abc, importlib.util, json, runpy, sys

parser=argparse.ArgumentParser()
parser.add_argument('--action',choices=('status','run'),required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
workspace=Path(__file__).resolve().parents[1]
root=Path(args.output).resolve()
if not root.is_relative_to(workspace):
    raise SystemExit('Frozen blueprint must belong to this workspace')
protocol=json.loads((root/'RUN_PROTOCOL.json').read_text('utf8'))
body={k:v for k,v in protocol.items() if k!='protocol_hash'}
raw=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
if hashlib.sha256(raw).hexdigest()!=protocol['protocol_hash']:
    raise SystemExit('Frozen protocol content identity changed')
source=root/'code'
actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
if source.resolve()!=source or actual!=protocol['code_hashes'] or any(p.is_symlink() or p.is_junction() for p in source.iterdir()):
    raise SystemExit('Frozen module set/content changed; refuse invocation')

class VerifiedSourceLoader(importlib.abc.Loader):
    """Compile verified bytes directly; a .pyc is never an execution source."""
    def __init__(self,path):self.path=path
    def create_module(self,spec):return None
    def get_code(self,fullname):
        raw=self.path.read_bytes()
        if self.path.is_symlink() or self.path.is_junction() or hashlib.sha256(raw).hexdigest()!=actual[self.path.name]:
            raise ImportError('Frozen source drift at consumption: '+self.path.name)
        return compile(raw,str(self.path),'exec',dont_inherit=True)
    def get_filename(self,fullname):return str(self.path)
    def is_package(self,fullname):return self.path.name=='__init__.py'
    def exec_module(self,module):exec(self.get_code(module.__name__),module.__dict__)

class FrozenFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='fuxi_v1':name='__init__.py'
        elif fullname.startswith('fuxi_v1.') and fullname.count('.')==1:name=fullname.split('.')[-1]+'.py'
        elif fullname.startswith('fuxi_v1.'):raise ImportError('Unregistered frozen submodule: '+fullname)
        else:return None
        if name not in actual:raise ImportError('Unregistered frozen module: '+fullname)
        return importlib.util.spec_from_file_location(fullname,source/name,loader=VerifiedSourceLoader(source/name),
            submodule_search_locations=[str(source)] if name=='__init__.py' else None)

finder=FrozenFinder();sys.meta_path.insert(0,finder)
spec=finder.find_spec('fuxi_v1')
module=importlib.util.module_from_spec(spec)
sys.modules['fuxi_v1']=module
spec.loader.exec_module(module)
sys.argv=['fuxi_v1','blueprint',args.action,'--output',str(root)]
runpy.run_module('fuxi_v1',run_name='__main__')
