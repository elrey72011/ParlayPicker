"""Real Git successor seals and resealing attacks; all fixtures remain offline."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

import pytest
from scripts import check_launch_change_scope as guard
from scripts import drive_history_scope as successor

SOURCE=Path(__file__).resolve().parents[1]


def git(repo,*args):
    return subprocess.check_output(["git",*args],cwd=repo).decode().strip()


def raw(repo,rev,path):
    return subprocess.check_output(["git","show",rev+":"+path],cwd=repo)


def write(repo,path,value):
    target=repo/path
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(value)


def commit(repo,message):
    git(repo,"add","--all")
    git(repo,"commit","-qm",message)
    return git(repo,"rev-parse","HEAD")


def seal(repo,binding):
    implementation=commit(repo,"bounded implementation")
    policy=successor.make_policy(guard,binding,implementation)
    write(repo,guard.DRIVE_POLICY_PATH,json.dumps(policy,indent=2).encode()+b"\n")
    candidate=commit(repo,"policy only seal")
    return implementation,candidate,policy


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    tmp_path=tmp_path_factory.mktemp("drive-scope-template")
    original_root=guard.ROOT
    from scripts.benchmark_drive_history_loading import blocked_network
    with blocked_network():
        repo=tmp_path/"repo"
        repo.mkdir()
        git(repo,"init","-q")
        git(repo,"config","user.name","Offline Test")
        git(repo,"config","user.email","offline@example.invalid")
        git(repo,"config","core.autocrlf","false")
        current_guard=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b"\r\n",b"\n")
        previous=guard._drive_previous_guard_source(current_guard)
        original=current_guard.split(b"\nPOLICY_PATH =",1)[0]+b"\n"
        write(repo,"README.md",b"offline fixture\n")
        write(repo,"core/protected.py",b"FROZEN = True\n")
        write(repo,guard.GUARD_PATH,original)
        write(repo,".github/workflows/paid-launch.yml",b"name: frozen offline workflow\n")
        write(repo,guard.CLOCK_TEST,b"assert future.expired == 1\n")
        recorded=commit(repo,"original baseline")
        manifest=dict(base_sha=recorded,required_ancestry={"pr_2349_merge_commit":recorded},
            protected_files=["core/protected.py"],
            protected_git_blobs={"core/protected.py":git(repo,"rev-parse",recorded+":core/protected.py")},
            tooling_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest()
                for p in (guard.GUARD_PATH,".github/workflows/paid-launch.yml")})
        manifest_raw=json.dumps(manifest).encode()+b"\n"
        retained=set(guard.DFS_UNCHANGED_PATHS)|set(guard.DFS_PATHS)|{guard.V4_POLICY_PATH}
        for path in retained | set(guard.DRIVE_PATHS):
            if path==guard.GUARD_PATH or path==guard.MANIFEST_PATH or path==".github/workflows/paid-launch.yml":
                continue
            source_path=SOURCE/path
            if not source_path.exists():
                continue
            value=source_path.read_bytes().replace(b"\r\n",b"\n")
            if path in guard.DRIVE_PATHS:
                if path in guard.DRIVE_PRIOR_SOURCE_RECONSTRUCTIONS:
                    value=guard._drive_previous_main_source(path,value)
                else:
                    # These application bytes are synthetic predecessor inputs;
                    # exact real-main reconstruction has its own frozen tests.
                    value=b"SYNTHETIC_PREDECESSOR = True\n"
            write(repo,path,value)
        write(repo,guard.MANIFEST_PATH,manifest_raw)
        write(repo,guard.GUARD_PATH,previous)
        base=commit(repo,"approved main")
        guard.ROOT=repo
        binding=dict(guard.DRIVE_BINDINGS,base=base,base_tree=git(repo,"rev-parse","HEAD^{tree}"),
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            previous_guard_sha256=hashlib.sha256(previous).hexdigest(),
            previous_policy_blob=guard.blob(base,guard.V4_POLICY_PATH))
        for path in guard.DRIVE_PATHS:
            write(repo,path,(SOURCE/path).read_bytes().replace(b"\r\n",b"\n"))
        binding["reviewed_blobs"]={p:git(repo,"hash-object","--",p) for p in guard.DRIVE_PATHS if p!=guard.GUARD_PATH}
        implementation,candidate,policy=seal(repo,binding)
        guard.ROOT=original_root
        return repo,binding,implementation,candidate,policy


@pytest.fixture
def fx(tmp_path,monkeypatch,prepared):
    from scripts.benchmark_drive_history_loading import blocked_network
    template,binding,implementation,candidate,policy=prepared
    repo=tmp_path/"repo"
    shutil.copytree(template,repo)
    monkeypatch.setattr(guard,"ROOT",repo)
    with blocked_network():
        yield repo,binding,implementation,candidate,policy


def assess(fx,base=None):
    try:
        return guard._run_drive_integrated(fx[0]/guard.MANIFEST_PATH,base,fx[1])
    except ValueError as exc:
        return 1,{"reason_codes":[str(exc)]}


def test_exact_candidate_ci_merge_and_prior_reconstruction(fx):
    repo,binding,_,candidate,_=fx
    code,report=assess(fx)
    assert code==0,report
    assert report["policy_valid"] and report["new_existing_test_exceptions"]==[]
    assert report["protected_changes"]==report["existing_test_changes"]==[]
    assert report["approved_exceptions"][0]["retained_unchanged"] is True
    assert guard.CLOCK_TEST in report["original_guard_report"]["existing_test_changes"]
    assert guard._drive_previous_guard_source(raw(repo,"HEAD",guard.GUARD_PATH),binding)==raw(repo,binding["base"],guard.GUARD_PATH)
    merge=git(repo,"commit-tree",git(repo,"rev-parse","HEAD^{tree}"),"-p",binding["base"],"-p",candidate,"-m","offline CI")
    git(repo,"checkout","-q",merge)
    assert assess(fx,binding["base"])[0]==0


@pytest.mark.parametrize("path", ["app_core/evidence_drive.py","app_core/evidence_remote.py",
    "app_core/public_history.py","streamlit_app.py","app/ui/public_results.py",
    "scripts/drive_history_scope.py","scripts/check_launch_change_scope.py",
    "tests/test_evidence_drive.py","tests/test_dfs_scope_policy.py",
    "app_core/draftkings_classic.py","app_core/ncaaf_schedule.py",
    "core/protected.py",guard.MANIFEST_PATH,guard.V4_POLICY_PATH,
    "tests/paid_launch/case_isolation_and_scope.py",".github/workflows/paid-launch.yml"])
def test_dirty_staged_committed_and_resealed_attacks_fail(fx,path):
    repo,binding,implementation,_,_=fx
    write(repo,path,(repo/path).read_bytes()+b"\nFORCE_BYPASS = True\n")
    assert assess(fx)[0]!=0
    git(repo,"add",path)
    assert assess(fx)[0]!=0
    commit(repo,"unauthorized followup")
    assert assess(fx)[0]!=0
    git(repo,"reset","--soft",binding["base"])
    seal(repo,binding)
    assert assess(fx)[0]!=0


@pytest.mark.parametrize("field,value", [("approval_reference","blanket approval"),("implementation_changes",{}),
    ("unchanged_bindings",{}),("tooling_sha256",{}),("base_tree","0"*40),
    ("extra_allowlist",["tests/"]),("schema_version",4),("implementation_tree","0"*40)])
def test_policy_mutations_fail(fx,field,value):
    repo,_,implementation,_,policy=fx
    git(repo,"checkout","-q",implementation)
    write(repo,guard.DRIVE_POLICY_PATH,json.dumps(dict(policy,**{field:value})).encode()+b"\n")
    commit(repo,"mutated seal")
    assert assess(fx)[0]!=0


@pytest.mark.parametrize("attack", ["extra_commit","extra_seal_path","wrong_parent_order","wrong_tree"])
def test_exact_ancestry_and_policy_only_shape(fx,attack):
    repo,binding,implementation,candidate,policy=fx
    if attack=="extra_commit":
        git(repo,"commit","--allow-empty","-qm","extra")
    elif attack=="extra_seal_path":
        git(repo,"checkout","-q",implementation)
        write(repo,guard.DRIVE_POLICY_PATH,json.dumps(policy).encode()+b"\n")
        write(repo,"README.md",b"unapproved seal content\n")
        commit(repo,"extra seal path")
    else:
        if attack=="wrong_tree":
            write(repo,"README.md",b"unapproved tree\n")
            git(repo,"add","README.md")
        parents=[candidate,binding["base"]] if attack=="wrong_parent_order" else [binding["base"],candidate]
        merge=git(repo,"commit-tree",git(repo,"write-tree"),"-p",parents[0],"-p",parents[1],"-m","invalid CI")
        git(repo,"checkout","-q",merge)
    assert assess(fx)[0]!=0


@pytest.mark.parametrize("path", ["shadow/app_core/evidence_drive.py","shadow/app_core/public_history.py",
    "shadow/streamlit_app.py","shadow/app_core/draftkings_classic.py",
    "shadow/app_core/ncaaf_schedule.py","sitecustomize.py","shadow/injected.pth"])
def test_untracked_shadows_and_hooks_fail(fx,path):
    write(fx[0],path,b"untracked bypass\n")
    assert assess(fx)[0]!=0


def test_crlf_and_cli_precedence_no_fallback(fx,monkeypatch,capsys):
    repo,binding,_,_,policy=fx
    git(repo,"config","core.autocrlf","true")
    for path in (guard.GUARD_PATH,guard.MANIFEST_PATH,guard.DRIVE_POLICY_PATH,
                 "scripts/drive_history_scope.py",".github/workflows/paid-launch.yml"):
        write(repo,path,(repo/path).read_bytes().replace(b"\n",b"\r\n"))
    assert assess(fx)[0]==0
    monkeypatch.setattr(guard,"DRIVE_BINDINGS",binding)
    monkeypatch.setattr(guard,"_run_dfs_integrated",lambda *args:pytest.fail("Invalid successor cannot fall back"))
    monkeypatch.setattr("sys.argv",["guard","--manifest",str(repo/guard.MANIFEST_PATH)])
    assert guard.main()==0
    write(repo,guard.DRIVE_POLICY_PATH,json.dumps(dict(policy,approval_reference="invalid")).encode()+b"\n")
    assert guard.main()!=0
