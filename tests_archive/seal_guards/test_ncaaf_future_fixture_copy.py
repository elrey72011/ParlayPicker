"""Synthetic real-Git template races and actual historical-copy rejection.

Controlled repacking explains a possible missing-object condition. It does not
identify the process that removed objects in the retained historical CI run.
"""
import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest
import test_readiness_dashboard_scope as historical
from test_readiness_dashboard_scope import prepared as historical_template


def git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,
        stderr=subprocess.STDOUT,timeout=30).decode().strip()


def create(repo,*,stable):
    repo.mkdir();git(repo,'init','-q')
    git(repo,'config','user.name','SYNTHETIC offline')
    git(repo,'config','user.email','offline@example.invalid')
    if stable:
        git(repo,'config','gc.auto','0')
        git(repo,'config','maintenance.auto','false')
    for i in range(40):
        (repo/f'SYNTHETIC-{i}.txt').write_text(('SYNTHETIC '+str(i)+'\n')*200)
    git(repo,'add','--all');git(repo,'commit','-qm','SYNTHETIC first revision')
    (repo/'SYNTHETIC-0.txt').write_text('SYNTHETIC second revision\n')
    git(repo,'add','--all');git(repo,'commit','-qm','SYNTHETIC second revision')
    if stable:git(repo,'gc','--prune=never')


def copy_after_actual_repack(repo,destination,monkeypatch):
    scan=os.scandir;repacked=[]
    @contextlib.contextmanager
    def enumerate_then_repack(path):
        with scan(path) as values:entries=list(values)
        if Path(path).resolve()==(repo/'.git/objects').resolve() and not repacked:
            repacked.append(True)
            # Real Git changes its own storage after enumeration. No invented
            # exception, manual deletion or ignored copy error supplies activity.
            git(repo,'gc','--prune=never')
        yield iter(entries)
    with monkeypatch.context() as patch:
        patch.setattr(shutil.os,'scandir',enumerate_then_repack)
        shutil.copytree(repo,destination)
    assert repacked==[True]


def test_real_repack_rejects_stale_copy_and_prepacked_template_preserves_history(tmp_path,monkeypatch):
    for stable in (False,True):
        source=tmp_path/('stable' if stable else 'loose')
        create(source,stable=stable)
        head,tree=git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}')
        destination=tmp_path/(source.name+'-copy')
        if stable:
            copy_after_actual_repack(source,destination,monkeypatch)
            assert git(destination,'rev-parse','HEAD')==head
            assert git(destination,'rev-parse','HEAD^{tree}')==tree
            git(destination,'fsck','--full','--strict')
        else:
            with pytest.raises(shutil.Error):
                copy_after_actual_repack(source,destination,monkeypatch)
        assert git(source,'rev-parse','HEAD')==head
        assert git(source,'rev-parse','HEAD^{tree}')==tree
        git(source,'fsck','--full','--strict')


def actual_historical_assessment(request,monkeypatch,tmp_path,prepared):
    # Use the original fixture, original integrated guard and actual repository
    # copying. Its exact predecessor guard is a test input, never a decision.
    frozen,_=request.getfixturevalue('ncaaf_released_scope_inputs')
    with monkeypatch.context() as patch:
        patch.setattr(historical,'guard',frozen)
        generator=historical.fx.__wrapped__(tmp_path,patch,prepared)
        with contextlib.closing(generator):
            value=next(generator)
            return historical.assess(value)


@pytest.mark.parametrize('attack',['missing_required_pack','corrupt_required_pack'])
def test_actual_historical_copy_rejects_missing_or_corrupt_required_objects(
    historical_template,request,monkeypatch,tmp_path,attack):
    source,*rest=historical_template
    packs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source/'.git/objects/pack').glob('*.pack')}
    assert packs and git(source,'config','gc.auto')=='0'
    assert git(source,'config','maintenance.auto')=='false'
    git(source,'fsck','--full','--strict')
    assert actual_historical_assessment(request,monkeypatch,tmp_path/'valid',historical_template)[0]==0

    damaged=tmp_path/'SYNTHETIC-damaged-template';shutil.copytree(source,damaged)
    head=git(damaged,'rev-parse','HEAD')
    required=[p.with_suffix('.pack') for p in (damaged/'.git/objects/pack').glob('*.idx')
        if any(line.startswith(head+' ') for line in git(damaged,'verify-pack','-v',str(p)).splitlines())]
    assert len(required)==1
    target=required[0];target.resolve().relative_to(damaged.resolve())
    # Git creates read-only packs on Windows. Change only the disposable damaged
    # copy's file attribute so this test can introduce its specified corruption.
    target.chmod(target.stat().st_mode | stat.S_IWUSR)
    if attack=='missing_required_pack':target.unlink()
    else:target.write_bytes(b'SYNTHETIC_CORRUPTED_REQUIRED_GIT_PACK\n')
    with pytest.raises(subprocess.CalledProcessError):git(damaged,'fsck','--full','--strict')
    try:
        code,_=actual_historical_assessment(request,monkeypatch,tmp_path/'rejected',(damaged,*rest))
    except subprocess.CalledProcessError:
        pass  # Expected negative case: required history cannot be read.
    else:
        assert code!=0,'Incomplete or corrupt required Git history was accepted'
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source/'.git/objects/pack').glob('*.pack')}==packs
    git(source,'fsck','--full','--strict')
