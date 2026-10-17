"""Real synthetic Git seals; no full-history/account/provider dependency."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from scripts import ci_scheduling_fixture_scope as scope


def sha(b):return hashlib.sha256(b).hexdigest()
def require(ok,reason):
    if not ok:raise ValueError(reason)


@pytest.fixture
def sealed(tmp_path):
    root=tmp_path/'SYNTHETIC-repository';root.mkdir()
    def git(*a):return subprocess.check_output(['git',*a],cwd=root).decode().strip()
    def raw(*a):return subprocess.check_output(['git',*a],cwd=root)
    def write(p,b):
        path=root/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    def commit(text):
        git('add','--all');git('commit','-qm',text);return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC scope test');git('config','user.email','synthetic@example.invalid')
    git('config','gc.auto','0');git('config','maintenance.auto','false');git('config','core.autocrlf','false');git('config','core.longpaths','true')
    G='scripts/check_launch_change_scope.py';M='docs/paid-launch/launch-baseline-manifest.json'
    F='scripts/ncaaf_future_spread_scope.py';APP='app_core/SYNTHETIC.py'
    old_binding=dict(reviewed_blobs={F:'a'*40},base='SYNTHETIC inherited contract')
    prior=json.dumps(old_binding).encode();old_identity='7'*64
    guard=b"NCAAF_FUTURE_BINDING_SHA256='"+sha(prior).encode()+b"'\nNCAAF_FUTURE_REVIEWED_GUARD_SHA256='"+old_identity.encode()+b"'\n# mirror "+old_identity.encode()+b'\n'
    guard+=b' '*(8388490-len(guard))
    manifest=json.dumps(dict(base_sha='5'*40,protected_files=['core/frozen.py'])).encode()
    write(G,guard);write(scope.BINDING,prior);write(F,b'# SYNTHETIC predecessor\n');write(APP,b'SYNTHETIC before\n')
    write(M,manifest);write('core/frozen.py',b'SYNTHETIC protected bytes\n')
    base=commit('SYNTHETIC reviewed baseline')
    write(F,b'# SYNTHETIC successor loader\n');write(APP,b'SYNTHETIC after\n')
    paths=[G,scope.BINDING,F,APP]
    d=dict(proposal_sha256=scope.PROPOSAL,base=base,base_tree=git('rev-parse',base+'^{tree}'),
        ordered_base_parents=[],manifest_sha256=sha(manifest),previous_binding_sha256=sha(prior),
        previous_binding_b64=base64.b64encode(prior).decode(),previous_guard_sha256=sha(guard),previous_guard_identity=old_identity,
        paths=paths,reviewed_blobs={F:git('hash-object',F),APP:git('hash-object',APP)},
        source_reconstructions={APP:dict(before_sha256=sha(b'SYNTHETIC before\n'),after_sha256=sha(b'SYNTHETIC after\n'),
            reverse_patches=[dict(start=0,end=len(b'SYNTHETIC after\n'),text='SYNTHETIC before\n')])})
    binding=deepcopy(old_binding);binding['ci_fixture_copy']=d;binding['reviewed_blobs'][F]=d['reviewed_blobs'][F]
    body=(json.dumps(binding,sort_keys=True,indent=2)+'\n').encode();write(scope.BINDING,body)
    successor=guard.replace(sha(prior).encode(),sha(body).encode()).replace(old_identity.encode(),b'0'*64)
    identity=sha(successor);successor=successor.replace(b'0'*64,identity.encode());write(G,successor)
    impl=commit('SYNTHETIC implementation')
    def exists(rev,path):return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=root,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=root,GUARD_PATH=G,MANIFEST_PATH=M,git=git,git_bytes=raw,exists_at=exists,_require=require,
        NCAAF_FUTURE_BINDING_SHA256=sha(body),NCAAF_FUTURE_REVIEWED_GUARD_SHA256=identity,
        _ncaaf_future_guard_matches=lambda b:b==successor,run=lambda *a:(1,dict(status='SYNTHETIC original report unchanged')))
    policy=scope.make_policy(g,binding,impl);write(scope.POLICY,json.dumps(policy).encode());seal=commit('SYNTHETIC policy-only seal')
    return SimpleNamespace(g=g,binding=binding,write=write,commit=commit,impl=impl,seal=seal,prior_guard=guard,successor=successor)


def test_exact_successor_and_ordered_tested_merge(sealed):
    f=sealed;g=f.g
    result=scope.run(g,g.ROOT/g.MANIFEST_PATH,None,f.binding)
    assert result[0]==0,result[1]
    merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',f.binding['ci_fixture_copy']['base'],'-p',f.seal,'-m','SYNTHETIC ordered CI merge')
    g.git('checkout','-q',merge)
    result=scope.run(g,g.ROOT/g.MANIFEST_PATH,None,f.binding)
    assert result[0]==0,result[1]


@pytest.mark.parametrize('attack',['dirty','staged','committed','resealed','changed-binding','changed-source','rehashed-before-source'])
def test_mutation_and_rehashed_provenance_still_reject(sealed,attack):
    f=sealed;g=f.g;b=deepcopy(f.binding)
    if attack=='changed-binding':b['base']='UNREVIEWED'
    elif attack=='rehashed-before-source':b['ci_fixture_copy']['source_reconstructions']['app_core/SYNTHETIC.py']['before_sha256']='e'*64
    elif attack=='changed-source':f.write('app_core/SYNTHETIC.py',b'UNREVIEWED')
    else:
        if attack=='resealed':g.git('checkout','-q',f.impl)
        f.write('core/frozen.py',b'SYNTHETIC unauthorized mutation\n')
        if attack=='staged':g.git('add','core/frozen.py')
        if attack in {'committed','resealed'}:f.commit('SYNTHETIC unauthorized bytes')
        if attack=='resealed':
            impl=g.git('rev-parse','HEAD');f.write(scope.POLICY,json.dumps(scope.make_policy(g,b,impl)).encode());f.commit('SYNTHETIC reseal')
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)[0]!=0


@pytest.mark.parametrize('base',['0'*40,'HEAD','main','--help','5fb8e135',''])
def test_arbitrary_comparison_base_rejects(sealed,base):
    assert scope.run(sealed.g,sealed.g.ROOT/sealed.g.MANIFEST_PATH,base,sealed.binding)[0]!=0


def test_exact_predecessor_source_and_altered_source(sealed):
    f=sealed;g=f.g
    assert scope.predecessor_guard(f.successor,g,f.binding)==f.prior_guard
    with pytest.raises(ValueError):scope.predecessor_guard(f.successor+b'UNREVIEWED',g,f.binding)
    assert scope.previous_source('app_core/SYNTHETIC.py',b'SYNTHETIC after\n',g,f.binding)==b'SYNTHETIC before\n'
    assert scope.previous_source('app_core/SYNTHETIC.py',b'SYNTHETIC altered\n',g,f.binding)==b'SYNTHETIC altered\n'


def test_live_guard_bound_and_all_source_reconstructions_match_current_bytes():
    from scripts import check_launch_change_scope as g
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/scope.BINDING).read_bytes())
    current=(root/g.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    assert len(current)==8388490<=8*1024*1024 and g._ncaaf_future_guard_matches(current)
    previous=scope.predecessor_guard(current,g,binding)
    assert sha(previous)==binding['ci_fixture_copy']['previous_guard_sha256']
    for path,r in binding['ci_fixture_copy']['source_reconstructions'].items():
        value=(root/path).read_bytes().replace(b'\r\n',b'\n')
        assert sha(value)==r['after_sha256']
        assert sha(scope.previous_source(path,value,g,binding))==r['before_sha256']
    from scripts import scope_reconstruction
    assert scope_reconstruction.MAX_CONTEXT_FIELDS==512 and scope_reconstruction.MAX_SOURCE_BYTES==8*1024*1024



def test_new_current_scope_and_external_children_are_not_projected(tmp_path):
    from ci_fixture_scope_inputs import install
    child=SimpleNamespace(__name__='test_ncaaf_transport_scope',__file__=str(tmp_path/'test_ncaaf_transport_scope.py'))
    class NoMutation:
        def __getattr__(self,key):raise AssertionError('External synthetic child must remain live')
    assert install(SimpleNamespace(module=child),NoMutation()) is False
    current=SimpleNamespace(__name__='test_ci_scheduling_fixture_scope',__file__=__file__)
    assert install(SimpleNamespace(module=current),NoMutation()) is True

def test_cold_historical_loader_uses_exact_bytes_under_a_predecessor_module_context(monkeypatch):
    import sys,scripts
    from ci_fixture_scope_inputs import frozen_predecessor as previous
    from ncaaf_transport_scope_inputs import frozen_predecessor as older
    root=Path(__file__).resolve().parents[1]
    old,binding=previous(str(root))
    older.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(scripts,'check_launch_change_scope',old)
        patch.setitem(sys.modules,'scripts.check_launch_change_scope',old)
        ancestor,_=older(str(root))
        assert sha(Path(root/'scripts/check_launch_change_scope.py').read_bytes().replace(b'\r\n',b'\n'))!=binding['ci_fixture_copy']['previous_guard_sha256']
        assert ancestor.main.__globals__ is vars(ancestor)
        assert sys.modules[ancestor.__name__] is ancestor
    older.cache_clear()

def test_missing_original_validation_context_cannot_be_reported_as_passing(sealed):
    f=sealed
    def unavailable(*args):raise RuntimeError('SYNTHETIC missing original validation context')
    f.g.run=unavailable
    with pytest.raises(RuntimeError,match='missing original validation context'):
        scope.run(f.g,f.g.ROOT/f.g.MANIFEST_PATH,None,f.binding)


def test_unchanged_native_context_assertion_receives_no_projection():
    import test_ncaaf_transport_scope as inherited
    from ci_fixture_scope_inputs import install
    original=Path.read_bytes
    class NoMutation:
        def __getattr__(self,key):raise AssertionError('Native isolation assertion must remain unwrapped')
    request=SimpleNamespace(module=inherited,function=inherited.test_external_fixture_children_keep_their_actual_live_context)
    assert install(request,NoMutation()) is True
    assert Path.read_bytes is original and Path.read_bytes.__module__=='pathlib'
