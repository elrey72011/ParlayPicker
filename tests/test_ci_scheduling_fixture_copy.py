"""SYNTHETIC actual CI-scheduling template copy/guard preservation."""
import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import pytest
import test_ci_scheduling_scope as historical
from test_ci_scheduling_scope import prepared as historical_template

def git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.STDOUT,timeout=30).decode().strip()

def actual_assessment(request,monkeypatch,tmp_path,prepared):
    frozen,_=request.getfixturevalue('ncaaf_released_scope_inputs')
    with monkeypatch.context() as patch:
        patch.setattr(historical,'guard',frozen)
        generator=historical.fx.__wrapped__(tmp_path,patch,prepared)
        with contextlib.closing(generator):
            value=next(generator)
            return historical.assess(value)

def test_actual_ci_template_copy_survives_controlled_real_repack(
    historical_template,request,monkeypatch,tmp_path):
    source,*_=historical_template
    assert git(source,'config','--local','--get','gc.auto')=='0'
    assert git(source,'config','--local','--get','maintenance.auto')=='false'
    assert list((source/'.git/objects/pack').glob('*.pack'))
    assert not list((source/'.git/objects').glob('[0-9a-f][0-9a-f]/*'))
    before=(git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))
    original_scan=os.scandir;repacked=[]
    def scan(path):
        # Only the intended copy boundary is intercepted. Other consumers retain
        # the native iterator and context-manager protocol, including cold guards.
        if Path(path).resolve()!=(source/'.git/objects').resolve() or repacked:
            return original_scan(path)
        with original_scan(path) as rows:entries=list(rows)
        repacked.append(True)
        git(source,'gc','--prune=never')
        return contextlib.nullcontext(iter(entries))
    with monkeypatch.context() as patch:
        patch.setattr(shutil.os,'scandir',scan)
        assert actual_assessment(request,patch,tmp_path/'valid',historical_template)[0]==0
    assert repacked==[True]
    assert (git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))==before
    git(source,'fsck','--full','--strict')

@pytest.mark.parametrize('attack',['missing_required_pack','corrupt_required_pack'])
def test_actual_ci_template_copy_rejects_missing_or_corrupt_required_objects(
    historical_template,request,monkeypatch,tmp_path,attack):
    source,*rest=historical_template
    packs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source/'.git/objects/pack').glob('*.pack')}
    assert packs
    git(source,'fsck','--full','--strict')
    damaged=tmp_path/'SYNTHETIC-damaged-template';shutil.copytree(source,damaged)
    head=git(damaged,'rev-parse','HEAD')
    required=[p.with_suffix('.pack') for p in (damaged/'.git/objects/pack').glob('*.idx')
        if any(line.startswith(head+' ') for line in git(damaged,'verify-pack','-v',str(p)).splitlines())]
    assert len(required)==1
    target=required[0];target.resolve().relative_to(damaged.resolve())
    target.chmod(target.stat().st_mode|stat.S_IWUSR)
    if attack=='missing_required_pack':target.unlink()
    else:target.write_bytes(b'SYNTHETIC_CORRUPTED_REQUIRED_GIT_PACK\n')
    with pytest.raises(subprocess.CalledProcessError):git(damaged,'fsck','--full','--strict')
    try:
        code,_=actual_assessment(request,monkeypatch,tmp_path/'rejected',(damaged,*rest))
    except subprocess.CalledProcessError:
        pass  # Required history cannot be read: this negative test requires rejection.
    else:
        assert code!=0,'Missing or corrupt required Git history was accepted'
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source/'.git/objects/pack').glob('*.pack')}==packs
    git(source,'fsck','--full','--strict')

