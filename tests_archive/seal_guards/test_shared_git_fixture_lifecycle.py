"""SYNTHETIC actual admission fixture: real repacking, required objects and cleanup."""
import contextlib
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import pytest
import test_nfl_admission_scope as admission
from test_nfl_admission_scope import prepared as admission_template
from scripts import check_launch_change_scope as guard

def git(repo,*args):
 return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.STDOUT,timeout=30).decode().strip()

def actual(prepared,monkeypatch,tmp_path):
 with monkeypatch.context() as patch:
  generator=admission.fx.__wrapped__(tmp_path,patch,prepared)
  with contextlib.closing(generator):
   value=next(generator)
   return admission.assess(value)

def test_actual_admission_template_copy_survives_controlled_repack(admission_template,monkeypatch,tmp_path):
 source,*_=admission_template
 assert git(source,'config','--get','gc.auto')=='0'
 assert git(source,'config','--get','maintenance.auto')=='false'
 assert list((source/'.git/objects/pack').glob('*.pack'))
 assert not list((source/'.git/objects').glob('[0-9a-f][0-9a-f]/*'))
 before=(git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))
 native=os.scandir;seen=[]
 def scan(path):
  try:chosen=Path(path).resolve()==(source/'.git/objects').resolve()
  except TypeError:chosen=False
  if not chosen or seen:return native(path)
  with native(path) as it:entries=list(it)
  seen.append(True);git(source,'gc','--prune=never')
  return contextlib.nullcontext(iter(entries))
 with monkeypatch.context() as patch:
  patch.setattr(os,'scandir',scan)
  assert actual(admission_template,patch,tmp_path/'valid')[0]==0
 assert seen==[True] and os.scandir is native
 assert (git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))==before
 git(source,'fsck','--full','--strict')

@pytest.mark.parametrize('attack',['missing_required_pack','corrupt_required_pack'])
def test_actual_admission_copy_rejects_missing_or_corrupt_required_objects(admission_template,monkeypatch,tmp_path,attack):
 source,*rest=admission_template
 git(source,'fsck','--full','--strict')
 original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'.git/objects/pack').glob('*.pack')}
 damaged=tmp_path/'SYNTHETIC-damaged';shutil.copytree(source,damaged)
 head=git(damaged,'rev-parse','HEAD')
 required=[p.with_suffix('.pack') for p in (damaged/'.git/objects/pack').glob('*.idx') if any(row.startswith(head+' ') for row in git(damaged,'verify-pack','-v',str(p)).splitlines())]
 assert len(required)==1
 pack=required[0];assert pack.resolve().is_relative_to(damaged.resolve())
 pack.chmod(pack.stat().st_mode|stat.S_IWUSR)
 if attack=='missing_required_pack':pack.unlink()
 else:pack.write_bytes(b'SYNTHETIC_CORRUPTED_REQUIRED_GIT_PACK\n')
 with pytest.raises(subprocess.CalledProcessError):git(damaged,'fsck','--full','--strict')
 try:code,_=actual((damaged,*rest),monkeypatch,tmp_path/'rejected')
 except subprocess.CalledProcessError:pass
 else:assert code!=0,'Damaged required Git history was accepted'
 assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'.git/objects/pack').glob('*.pack')}==original

def test_actual_admission_copy_failure_restores_root_and_hooks(admission_template,monkeypatch,tmp_path):
 source,*_=admission_template
 before=(git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))
 root=admission.guard.ROOT;native_copy=shutil.copytree;native_scan=os.scandir;seen=[]
 def fail(src,dst,*a,**kw):
  if Path(src).resolve()==source.resolve():
   seen.append(True);raise OSError('SYNTHETIC intended copy failure')
  return native_copy(src,dst,*a,**kw)
 with monkeypatch.context() as patch:
  patch.setattr(shutil,'copytree',fail)
  with pytest.raises(OSError,match='SYNTHETIC intended copy failure'):
   actual(admission_template,patch,tmp_path/'failed')
 assert seen==[True] and shutil.copytree is native_copy and os.scandir is native_scan
 assert admission.guard.ROOT==root
 assert (git(source,'rev-parse','HEAD'),git(source,'rev-parse','HEAD^{tree}'))==before
 git(source,'fsck','--full','--strict')

