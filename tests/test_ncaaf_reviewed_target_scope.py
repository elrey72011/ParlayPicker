"""Exact source/preimage and live scope decisions; synthetic Git only."""
import hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import ncaaf_reviewed_target_scope as scope


def require(ok,code):
    if not ok:raise ValueError(code)


@pytest.fixture
def sealed(tmp_path):
    repo=tmp_path/'SYNTHETIC-repository';repo.mkdir()
    def raw(*args):return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.DEVNULL)
    def git(*args):return raw(*args).decode().strip()
    def write(path,content):
        p=repo/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    def commit():git('add','--all');git('commit','-qm','SYNTHETIC exact source');return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC');git('config','user.email','synthetic@example.invalid')
    git('config','gc.auto','0');git('config','maintenance.auto','false');git('config','core.autocrlf','false')
    git('config','core.longpaths','true')
    G='scripts/check_launch_change_scope.py';M='docs/paid-launch/launch-baseline-manifest.json'
    before=b'SYNTHETIC original guard\n';prefix=("_RT_BINDING_SHA256='"+'a'*64+"'\nNCAAF_FUTURE_REVIEWED_GUARD_SHA256='"+'b'*64+"'\n").encode()
    after=prefix+before
    write(G,before);write('core/frozen.py',b'SYNTHETIC protected source\n')
    manifest=b'{"base_sha":"SYNTHETIC_EXACT_RECORDED_BASE","protected_files":["core/frozen.py"]}'
    write(M,manifest);base=commit();paths=[G,scope.BINDING,'app_core/new.py']
    write(G,after);write('app_core/new.py',b'SYNTHETIC selected successor\n')
    b=dict(proposal_sha256=scope.PROPOSAL,base=base,base_tree=git('rev-parse',base+'^{tree}'),manifest_sha256=scope.sha(manifest),paths=paths,
        reviewed_blobs={'app_core/new.py':git('hash-object','app_core/new.py')},source_reconstructions={G:dict(before_sha256=scope.sha(before),
            after_sha256=scope.sha(after.replace(b'a'*64,b'0'*64).replace(b'b'*64,b'0'*64)),reverse_patches=[dict(start=0,end=len(prefix),text='')])})
    write(scope.BINDING,json.dumps(b).encode());impl=commit()
    g=SimpleNamespace(ROOT=repo,GUARD_PATH=G,MANIFEST_PATH=M,git=git,git_bytes=raw,_require=require,
        exists_at=lambda rev,path:subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=repo,capture_output=True).returncode==0,
        run=lambda *a:(1,{'status':'SYNTHETIC unchanged legacy baseline report'}))
    policy=scope.make_policy(g,b,impl);write(scope.POLICY,json.dumps(policy).encode());seal=commit()
    return g,b,write,commit,impl,seal


def test_exact_pr_push_and_ordered_tested_merge_contexts(sealed):
    g,b,write,commit,impl,seal=sealed
    for base in (None,b['base'],'SYNTHETIC_EXACT_RECORDED_BASE'):
        assert scope.run(g,g.ROOT/g.MANIFEST_PATH,base,b)[0]==0
    tree=g.git('rev-parse',seal+'^{tree}')
    merge=g.git('commit-tree',tree,'-p',b['base'],'-p',seal,'-m','SYNTHETIC CI merge');g.git('checkout','-q',merge)
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,b['base'],b)[0]==0


@pytest.mark.parametrize('attack',['arbitrary_base','dirty','staged','committed','resealed','manifest','binding','source','extra_file','wrong_merge_order','wrong_merge_tree'])
def test_mutable_checkout_and_exact_bindings_never_cached(sealed,attack):
    g,b,write,commit,impl,seal=sealed
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)[0]==0
    base=None
    if attack=='arbitrary_base':base='UNREVIEWED_BASE'
    elif attack=='resealed':
        approved={p:g.git_bytes('show',impl+':'+p) for p in b['paths']}
        g.git('checkout','-q',b['base'])
        for path,raw in approved.items():write(path,raw)
        write('app_core/new.py',b'UNREVIEWED resealed implementation\n')
        altered=commit();write(scope.POLICY,json.dumps(scope.make_policy(g,b,altered)).encode());commit()
    elif attack in ('dirty','staged','committed'):
        write('core/frozen.py',b'UNREVIEWED required source\n')
        if attack=='staged':g.git('add','core/frozen.py')
        if attack=='committed':commit()
    elif attack=='manifest':write(g.MANIFEST_PATH,b'{"base_sha":"UNREVIEWED","protected_files":[]}');commit()
    elif attack=='binding':write(scope.BINDING,json.dumps(dict(b,proposal_sha256='0'*64)).encode());commit()
    elif attack=='source':write('app_core/new.py',b'UNREVIEWED selected source\n');commit()
    elif attack=='extra_file':write('UNREVIEWED.py',b'UNREVIEWED\n');commit()
    else:
        parents=[seal,b['base']] if attack=='wrong_merge_order' else [b['base'],seal]
        tree=g.git('rev-parse',(b['base'] if attack=='wrong_merge_tree' else seal)+'^{tree}')
        merge=g.git('commit-tree',tree,'-p',parents[0],'-p',parents[1],'-m','SYNTHETIC wrong merge');g.git('checkout','-q',merge)
    assert scope.run(g,g.ROOT/g.MANIFEST_PATH,base,b)[0]!=0


def test_current_sources_have_exact_preimages_and_unknown_bytes_stay_unknown():
    root=Path(__file__).resolve().parents[1];b=json.loads((root/scope.BINDING).read_bytes())
    for path,r in b['source_reconstructions'].items():
        # Current scope checks actual bytes, outside the inherited tests' narrow
        # Path.read_bytes predecessor-input projection.
        with (root/path).open('rb') as f:raw=f.read().replace(b'\r\n',b'\n')
        prior=scope.previous_source(path,raw,b)
        assert scope.sha(prior)==r['before_sha256']
        assert scope.previous_source(path,raw+b'UNREVIEWED',b)==raw+b'UNREVIEWED'
    assert (root/'scripts/check_launch_change_scope.py').stat().st_size<=8*1024*1024
    from scripts import scope_reconstruction
    assert scope_reconstruction.MAX_CONTEXT_FIELDS==512
