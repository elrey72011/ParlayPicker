"""Authenticated whole-module retained profile input;never a current approval."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import base64,hashlib,io,json,re,sys,types,zlib
import pytest
from tests.nfl_capacity_import_owner import predecessor_imports

ROOT=Path(__file__).resolve().parents[1]
CONSUMER='tests/test_ncaaf_sidebar_startup_scope.py'
DECLARATION='docs/paid-launch/ci-fifteen-retained-profile-input-v3.json'
OLD_HEAD='cd181088d1c685f377b0b2f3227881131d0f3ca5'
OLD_BINDING='d84983e285490da3e38128ceeec1b9c907632eeea62596e8e3fe3628d29499a1'
OLD_PROFILE='5513e877c0e467bf8217dce1b041453e942d4610af1c5734e72519cb197ea347'
CURRENT_PROFILE='c627bacb49c69ab13cbb309bcee1872e2bb22f41eab3248c367ff31f099ea5a2'
LIMIT=8*1024*1024
def require(ok):
    if not ok:raise ValueError('NATIVE_LPT_HISTORICAL_INPUT_CHANGED')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(x):return(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()

def restore_binding(binding,declaration):
    """Exact immutable target identity,not a cached acceptance or guessed layer."""
    value=deepcopy(binding);operations=declaration['binding_delta']
    require(isinstance(operations,list)and len(operations)<=2000)
    for op in operations:
        require(set(op)in({'kind','path'},{'kind','path','value'}))
        path=op['path'];require(isinstance(path,list)and 1<=len(path)<=32 and all(isinstance(p,str)for p in path))
        parent=value
        for key in path[:-1]:require(type(parent)is dict and key in parent);parent=parent[key]
        require(type(parent)is dict)
        if op['kind']=='remove':require(path[-1]in parent and set(op)=={'kind','path'});del parent[path[-1]]
        else:require(op['kind']=='set'and set(op)=={'kind','path','value'});parent[path[-1]]=deepcopy(op['value'])
    raw=canonical(value)
    require(declaration['old_head']==OLD_HEAD and declaration['old_binding_sha256']==OLD_BINDING
        and len(raw)==declaration['old_binding_bytes']<=LIMIT and sha(raw)==OLD_BINDING)
    return raw

def unpack(record):
    require(set(record)=={'bytes','sha256','compressed_b85'})
    z=zlib.decompressobj();raw=z.decompress(base64.b85decode(record['compressed_b85']),LIMIT+1)
    require(z.eof and not z.unused_data and not z.unconsumed_tail and len(raw)==record['bytes']<=LIMIT and sha(raw)==record['sha256'])
    return raw

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m

def build_physical_view(root,destination):
    root=Path(root).resolve();destination=Path(destination);require(root==ROOT)
    # Import aliases can belong to a preceding isolated fixture. Bootstrap from
    # exact physical source,not an ambient package or mutable loaded helper.
    source_path=root/'scripts/ncaaf_filtered_history_scope.py'
    with io.FileIO(source_path,'r')as stream:source=stream.read(LIMIT+1).replace(b'\r\n',b'\n')
    require(len(source)<=LIMIT)
    matches=re.findall(rb"^_BINDING_SHA256='([0-9a-f]{64})'$",source,re.M)
    require(len(matches)==1 and source.count(matches[0])==1)
    require(sha(source.replace(matches[0],b'0'*64))=='6d75005330e4b87b00e7c8535b92f25ba622c2d78c7a98c84932c9075ccafb86')
    current=types.ModuleType('exact_current_profile_history');current.__file__=str(source_path)
    exec(compile(source,str(source_path),'exec'),current.__dict__)
    require(current._BINDING_SHA256.encode()==matches[0]);binding=current.authenticate(root)
    declaration_raw=current.physical(root/DECLARATION)
    require(sha(declaration_raw)==binding['components'][DECLARATION])
    declaration=json.loads(declaration_raw)
    require(declaration['consumer']==CONSUMER and declaration['current_profile_sha256']==CURRENT_PROFILE
        and declaration['old_profile_sha256']==OLD_PROFILE
        and sha(current.physical(root/CONSUMER))==declaration['consumer_sha256']
        and sha(current.physical(root/'scripts/ci_test_file_costs_v1.json'))==CURRENT_PROFILE)
    old_raw=restore_binding(binding,declaration);old=json.loads(old_raw)
    before=declaration['before_sources'];require(set(before)<=set(old['components']))
    for path,expected in old['components'].items():
        if path in before:raw=unpack(before[path])
        else:
            raw=current.physical(root/path)
            require(sha(current.normalized(path,raw))==binding['components'][path])
            if path in current.NORMALIZED:
                require(raw.count(current._BINDING_SHA256.encode())==1)
                raw=raw.replace(current._BINDING_SHA256.encode(),OLD_BINDING.encode())
        normalized=raw.replace(OLD_BINDING.encode(),b'0'*64)if path in current.NORMALIZED else raw
        require(sha(normalized)==expected)
        target=destination/path;require(target.resolve().is_relative_to(destination.resolve()))
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    (destination/current.BINDING).write_bytes(old_raw)
    require(sha((destination/'scripts/ci_test_file_costs_v1.json').read_bytes())==OLD_PROFILE)
    h=load(destination/'scripts/ncaaf_filtered_history_scope.py','exact_pre_lpt_history');h.authenticate(destination)
    boundary=load(destination/'scripts/ncaaf_private_refresh_scope.py','exact_pre_lpt_private_boundary')
    successor=load(destination/'scripts/ncaaf_sidebar_startup_scope.py','exact_pre_lpt_sidebar')
    return dict(ROOT=destination,history=h,boundary=boundary,successor=successor)

@contextmanager
def owned_view(module,root,destination):
    values=build_physical_view(root,destination)
    import scripts
    with predecessor_imports(destination),pytest.MonkeyPatch.context()as patch:
        for leaf,key in [('ncaaf_filtered_history_scope','history'),('ncaaf_private_refresh_scope','boundary'),('ncaaf_sidebar_startup_scope','successor')]:
            patch.setitem(sys.modules,'scripts.'+leaf,values[key]);patch.setattr(scripts,leaf,values[key],raising=False)
        for name in ('ROOT','boundary','successor'):patch.setattr(module,name,values[name])
        yield values

@pytest.fixture(autouse=True)
def retained_native_lpt_profile_generation(request):
    # Module-wide logical generation;never dispatch on individual assertions.
    if Path(request.module.__file__).resolve()!=ROOT/CONSUMER:
        yield;return
    destination=request.getfixturevalue('tmp_path')/'SYNTHETIC-authenticated-pre-LPT-profile'
    with owned_view(request.module,ROOT,destination):yield
