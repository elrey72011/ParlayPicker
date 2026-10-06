"""Exact successor Git seals and hostile changes; all evidence is synthetic/offline."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import pytest
from scripts import check_launch_change_scope as guard
from scripts import nfl_calibration_scope as successor
from scripts.benchmark_drive_history_loading import blocked_network
SOURCE=Path(__file__).resolve().parents[1]


def git(repo,*a):return subprocess.check_output(["git",*a],cwd=repo).decode().strip()
def raw(repo,rev,path):return subprocess.check_output(["git","show",rev+":"+path],cwd=repo)
def write(repo,path,data):
 p=repo/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def commit(repo,message):
 git(repo,"add","--all");git(repo,"commit","--allow-empty","-qm",message);return git(repo,"rev-parse","HEAD")
def seal(repo,binding):
 implementation=commit(repo,"reviewed implementation")
 policy=successor.make_policy(guard,binding,implementation)
 write(repo,guard.NFL_CALIBRATION_POLICY_PATH,json.dumps(policy,indent=2).encode()+b"\n")
 candidate=commit(repo,"policy-only seal")
 return implementation,candidate,policy


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
 repo=tmp_path_factory.mktemp("source-intake-scope")/"r";repo.mkdir()
 original_root=guard.ROOT
 with blocked_network():
  git(repo,"init","-q");git(repo,"config","user.name","Offline Test");git(repo,"config","user.email","offline@example.invalid");git(repo,"config","core.autocrlf","false")
  current=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b"\r\n",b"\n")
  previous=guard._nfl_calibration_previous_guard_source(current)
  original=current.split(b"\nPOLICY_PATH =",1)[0]+b"\n"
  write(repo,guard.GUARD_PATH,original);write(repo,"core/protected.py",b"FROZEN = True\n")
  write(repo,"README.md",b"offline exact-offer intake fixture\n")
  write(repo,".github/workflows/paid-launch.yml",b"name: frozen offline fixture\n")
  write(repo,guard.CLOCK_TEST,b"assert future.expired == 1\n")
  recorded=commit(repo,"original frozen baseline")
  manifest=dict(base_sha=recorded,required_ancestry={"pr_2349_merge_commit":recorded},protected_files=["core/protected.py"],protected_git_blobs={"core/protected.py":git(repo,"rev-parse",recorded+":core/protected.py")},tooling_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in [guard.GUARD_PATH,".github/workflows/paid-launch.yml"]})
  manifest_raw=json.dumps(manifest).encode()+b"\n"
  old_policy=json.loads((SOURCE/guard.SOURCE_INTAKE_POLICY_PATH).read_text())
  retained=set(old_policy["unchanged_bindings"])|set(old_policy["implementation_changes"])|{guard.SOURCE_INTAKE_POLICY_PATH}|set(guard.NFL_CALIBRATION_FROZEN_PATHS)
  for path in retained|set(guard.NFL_CALIBRATION_PATHS):
   if path in {guard.GUARD_PATH,guard.MANIFEST_PATH,".github/workflows/paid-launch.yml"} or not (SOURCE/path).exists():continue
   data=(SOURCE/path).read_bytes().replace(b"\r\n",b"\n")
   if path in guard.NFL_CALIBRATION_PATHS:
    if path not in guard.NFL_CALIBRATION_PRIOR_SOURCE_RECONSTRUCTIONS:continue
    data=guard._nfl_calibration_previous_main_source(path,data)
   write(repo,path,data)
  # Predecessor's bindings are fixture-specific, its source remains an immutable input.
  fixture_policy=dict(old_policy,unchanged_bindings={p:git(repo,"hash-object",p) for p in retained if (repo/p).exists()})
  write(repo,guard.SOURCE_INTAKE_POLICY_PATH,json.dumps(fixture_policy).encode()+b"\n")
  write(repo,guard.MANIFEST_PATH,manifest_raw);write(repo,guard.GUARD_PATH,previous)
  base=commit(repo,"verified main fixture");guard.ROOT=repo
  binding=dict(guard.NFL_CALIBRATION_BINDINGS,base=base,base_tree=git(repo,"rev-parse","HEAD^{tree}"),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),previous_guard_sha256=hashlib.sha256(previous).hexdigest(),previous_policy_blob=guard.blob(base,guard.SOURCE_INTAKE_POLICY_PATH))
  for path in guard.NFL_CALIBRATION_PATHS:write(repo,path,(SOURCE/path).read_bytes().replace(b"\r\n",b"\n"))
  binding["reviewed_blobs"]={p:git(repo,"hash-object",p) for p in guard.NFL_CALIBRATION_PATHS if p!=guard.GUARD_PATH}
  implementation,candidate,policy=seal(repo,binding);guard.ROOT=original_root
 return repo,binding,implementation,candidate,policy


@pytest.fixture
def fx(tmp_path,monkeypatch,prepared):
 template,binding,implementation,candidate,policy=prepared
 repo=tmp_path/"r";shutil.copytree(template,repo);monkeypatch.setattr(guard,"ROOT",repo)
 with blocked_network():yield repo,binding,implementation,candidate,policy


def assess(fx,base=None):
 try:return guard._run_nfl_calibration_integrated(fx[0]/guard.MANIFEST_PATH,base,fx[1])
 except (OSError,ValueError) as e:return 1,{"reason_codes":[str(e)]}


def test_exact_seal_and_ci_merge_preserve_predecessor(fx):
 repo,binding,implementation,candidate,policy=fx
 code,report=assess(fx);assert code==0,report
 assert report["policy_valid"] and not report["new_existing_test_exceptions"]
 assert policy["schema_version"]==16
 assert guard._nfl_calibration_previous_guard_source(raw(repo,"HEAD",guard.GUARD_PATH),binding)==raw(repo,binding["base"],guard.GUARD_PATH)
 assert git(repo,"diff","--name-status",implementation,candidate).splitlines()==["A\t"+guard.NFL_CALIBRATION_POLICY_PATH]
 merge=git(repo,"commit-tree",git(repo,"rev-parse","HEAD^{tree}"),"-p",binding["base"],"-p",candidate,"-m","offline CI merge")
 git(repo,"checkout","-q",merge);assert assess(fx,binding["base"])[0]==0


def test_exact_predecessor_source_reconstruction(fx):
 repo,binding,_,_,_=fx
 for path in guard.NFL_CALIBRATION_PRIOR_SOURCE_RECONSTRUCTIONS:
  before=raw(repo,binding["base"],path);after=raw(repo,"HEAD",path)
  assert guard._nfl_calibration_previous_main_source(path,after)==before
  assert guard._nfl_calibration_previous_main_source(path,before)==before
 assert guard._dfs_guard_matches(raw(repo,"HEAD",guard.GUARD_PATH),guard.SOURCE_INTAKE_BINDINGS["successor_guard_sha256"])


@pytest.mark.parametrize("path",sorted(set(guard.NFL_CALIBRATION_PATHS)|{guard.MANIFEST_PATH,guard.SOURCE_INTAKE_POLICY_PATH,guard.NFL_INPUTS_POLICY_PATH,"core/protected.py","app_core/market_probability_model.py","app_core/weights_config.py","core/streamlit_pipeline.py","tests/test_nfl_native_provenance.py","tests/test_nfl_admission_bindings.py","app_core/nfl_native_provenance.py","app_core/nfl_inference_evidence.py",".github/workflows/paid-launch.yml","tests/paid_launch/case_isolation_and_scope.py"}))
def test_dirty_staged_committed_resealed_changes_reject(fx,path):
 repo,binding,_,_,_=fx
 write(repo,path,(repo/path).read_bytes()+b"\nFORCE_BYPASS = True\n")
 assert assess(fx)[0]!=0
 git(repo,"add",path);assert assess(fx)[0]!=0
 commit(repo,"unreviewed change");assert assess(fx)[0]!=0
 git(repo,"reset","--soft",binding["base"]);seal(repo,binding);assert assess(fx)[0]!=0


@pytest.mark.parametrize("field,value",[("implementation_changes",{}),("unchanged_bindings",{}),("tooling_sha256",{}),("base_tree","0"*40),("schema_version",15),("approval_reference","blanket approval"),("implementation_tree","0"*40),("extra_allowlist",["tests/"])])
def test_mutated_policy_rejects(fx,field,value):
 repo,_,implementation,_,policy=fx
 git(repo,"checkout","-q",implementation)
 write(repo,guard.NFL_CALIBRATION_POLICY_PATH,json.dumps(dict(policy,**{field:value})).encode()+b"\n")
 commit(repo,"mutated seal");assert assess(fx)[0]!=0


@pytest.mark.parametrize("attack",["extra_commit","extra_seal_path","wrong_parent_order","wrong_tree"])
def test_ancestry_and_policy_only_shape(fx,attack):
 repo,binding,implementation,candidate,_=fx
 if attack=="extra_commit":commit(repo,"extra")
 elif attack=="extra_seal_path":
  git(repo,"checkout","-q",implementation);write(repo,"extra.txt",b"extra");commit(repo,"bad seal")
 else:
  tree=git(repo,"rev-parse",("HEAD" if attack=="wrong_parent_order" else binding["base"])+"^{tree}")
  parents=[candidate,binding["base"]] if attack=="wrong_parent_order" else [binding["base"],candidate]
  merge=git(repo,"commit-tree",tree,"-p",parents[0],"-p",parents[1],"-m","hostile merge");git(repo,"checkout","-q",merge)
 assert assess(fx)[0]!=0


def test_predecessor_fixture_guard_reconstruction_peels_exact_successor():
 source=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
 predecessor=guard._nfl_calibration_previous_guard_source(source)
 assert guard._source_intake_previous_guard_source(source)==guard._source_intake_previous_guard_source(predecessor)
 with pytest.raises(ValueError,match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
  guard._source_intake_previous_guard_source(source+b'\nUNREVIEWED=True\n')
