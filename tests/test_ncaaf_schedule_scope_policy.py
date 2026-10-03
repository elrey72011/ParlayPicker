"""Exact schedule successor seals, including malicious resealed candidates."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from scripts import check_launch_change_scope as g

SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode().strip()


def write(repo, path, value):
    p=repo/path
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(value)


def commit(repo):
    git(repo,"add","--all")
    git(repo,"commit","-qm","offline fixture")
    return git(repo,"rev-parse","HEAD")


def policy(repo,binding,implementation):
    return {"schema_version":1,"policy_version":g.SCHEDULE_POLICY_VERSION,"approval_reference":g.SCHEDULE_APPROVAL_REFERENCE,
            "base_sha":binding["base"],"base_tree":binding["base_tree"],"original_manifest_sha256":binding["manifest_sha256"],
            "implementation_commit":implementation,"implementation_tree":git(repo,"rev-parse",implementation+"^{tree}"),
            "implementation_changes":{p:{"before_blob":g.blob(binding["base"],p),"after_blob":g.blob(implementation,p)} for p in g.SCHEDULE_PATHS},
            "tooling_sha256":{p:hashlib.sha256(g.git_bytes("show",implementation+":"+p)).hexdigest() for p in (g.GUARD_PATH,".github/workflows/paid-launch.yml")},
            "unchanged_bindings":{p:g.blob(binding["base"],p) for p in g.SCHEDULE_IMMUTABLE}}


@pytest.fixture
def fx(tmp_path,monkeypatch):
    repo=tmp_path/"repo";repo.mkdir()
    git(repo,"init","-q");git(repo,"config","core.autocrlf","false")
    git(repo,"config","user.name","Offline");git(repo,"config","user.email","offline@example.invalid")
    write(repo,"README.md",b"offline\n")
    protected=["core/probability_calibration.py","models/model.txt","app_core/prediction_evidence.py"]
    for p in protected:write(repo,p,b"immutable science\n")
    write(repo,g.CLOCK_TEST,b"assert future.expired == 1\n")
    write(repo,"tests/test_unrelated.py",b"assert 1 == 1\n")
    write(repo,".github/workflows/paid-launch.yml",b"name: unchanged\n")
    recorded=commit(repo)
    manifest={"base_sha":recorded,"required_ancestry":{"pr_2349_merge_commit":recorded},"protected_files":protected,
              "protected_git_blobs":{p:git(repo,"rev-parse",recorded+":"+p) for p in protected},
              "tooling_sha256":{g.GUARD_PATH:"original", ".github/workflows/paid-launch.yml":hashlib.sha256((repo/".github/workflows/paid-launch.yml").read_bytes()).hexdigest()}}
    raw=json.dumps(manifest).encode();write(repo,g.MANIFEST_PATH,raw)
    for p in g.SCHEDULE_IMMUTABLE:
        if p in (g.MANIFEST_PATH,".github/workflows/paid-launch.yml"):continue
        write(repo,p,(SOURCE/p).read_bytes().replace(b"\r\n",b"\n"))
    previous=(SOURCE/g.GUARD_PATH).read_bytes().replace(b"\r\n",b"\n").split(b"\nSCHEDULE_POLICY_PATH =",1)[0]+b"\ndef main() -> int:\n    pass\n"
    write(repo,g.GUARD_PATH,previous)
    for p,edits in g.SCHEDULE_SHARED_EDITS.items():
        value=(SOURCE/p).read_bytes().replace(b"\r\n",b"\n").decode()
        for before,after in reversed(edits):
            assert value.count(after)==1
            value=value.replace(after,before,1)
        write(repo,p,value.encode())
    base=commit(repo)
    monkeypatch.setattr(g,"ROOT",repo)
    binding={"base":base,"base_tree":git(repo,"rev-parse",base+"^{tree}"),"manifest_sha256":hashlib.sha256(raw).hexdigest(),
             "previous_guard_sha256":hashlib.sha256(previous).hexdigest(),"clock_blob":g.blob(base,g.CLOCK_TEST)}
    for p in g.SCHEDULE_PATHS:write(repo,p,original_schedule_source(p))
    implementation=commit(repo)
    sealed=policy(repo,binding,implementation)
    write(repo,g.SCHEDULE_POLICY_PATH,json.dumps(sealed).encode());seal=commit(repo)
    return repo,binding,implementation,seal,sealed



def original_schedule_source(path):
    """Exercise the original v1 contract, never rebind it to the correction."""
    source = (SOURCE / path).read_bytes().replace(b"\r\n", b"\n")
    if path == g.GUARD_PATH:
        return source.split(b"\nCOVERAGE_POLICY_PATH =", 1)[0] + g.COVERAGE_PREVIOUS_CLI
    if path == "app_core/ncaaf_schedule.py":
        text = source.decode()
        for before, after in reversed(g.COVERAGE_MODULE_EDITS):
            assert text.count(after) == 1
            text = text.replace(after, before, 1)
        return text.encode()
    if path == "tests/test_ncaaf_schedule_scope_policy.py":
        text = source.decode()
        for before, after, count in reversed(g.COVERAGE_V1_TEST_EDITS):
            assert text.count(after) == count
            text = text.replace(after, before)
        return text.encode()
    return source

def assess(fx):
    repo,binding,*_=fx
    return g._run_schedule_integrated(repo/g.MANIFEST_PATH,binding["base"],binding)


def test_exact_schedule_seal_keeps_original_report_and_retained_clock(fx):
    code,report=assess(fx)
    assert code==0,report
    assert report["original_guard_report"]["status"]=="FAIL"
    assert report["approved_exceptions"][0]["retained_unchanged"]
    assert report["new_existing_test_exceptions"]==[]
    assert set(report["approved_integration_changes"])==set(g.SCHEDULE_PATHS)


def test_exact_ci_merge_parent_order_and_tree(fx):
    repo,binding,_,seal,_=fx
    git(repo,"checkout","-q",binding["base"])
    git(repo,"merge","--no-ff","-qm","CI merge",seal)
    assert assess(fx)[0]==0


@pytest.mark.parametrize("path",["tests/test_unrelated.py","core/probability_calibration.py","models/model.txt",g.MANIFEST_PATH,
                                g.CLOCK_TEST,g.POLICY_PATH,g.V3_POLICY_PATH,".github/workflows/paid-launch.yml",
                                "tests/paid_launch/case_isolation_and_scope.py"])
def test_unauthorized_extra_changes_rejected_even_with_rebound_policy(fx,path):
    repo,binding,_,_,_=fx
    git(repo,"checkout","-q",binding["base"])
    for p in g.SCHEDULE_PATHS:write(repo,p,original_schedule_source(p))
    write(repo,path,(repo/path).read_bytes()+b"\nUNAUTHORIZED\n")
    implementation=commit(repo)
    sealed=policy(repo,binding,implementation)
    write(repo,g.SCHEDULE_POLICY_PATH,json.dumps(sealed).encode());commit(repo)
    assert assess(fx)[0]==1


def test_existing_nested_entrypoint_is_immutable_not_a_path_exemption(fx):
    repo,*_=fx
    path="parlaypicker/app/streamlit_app.py"
    write(repo,path,(repo/path).read_bytes()+b"\nUNAUTHORIZED\n")
    commit(repo)
    assert assess(fx)[0]==1


@pytest.mark.parametrize("path",["core/streamlit_pipeline.py","streamlit_app.py","app_core/ncaaf_schedule.py",g.GUARD_PATH])
def test_shared_science_additive_module_and_tooling_reseal_cannot_expand_scope(fx,path):
    repo,binding,_,_,_=fx
    git(repo,"checkout","-q",binding["base"])
    for p in g.SCHEDULE_PATHS:write(repo,p,original_schedule_source(p))
    write(repo,path,(repo/path).read_bytes()+b"\nUNAUTHORIZED_AUTHORITY = True\n")
    implementation=commit(repo)
    sealed=policy(repo,binding,implementation)
    write(repo,g.SCHEDULE_POLICY_PATH,json.dumps(sealed).encode());commit(repo)
    code,report=assess(fx)
    assert code==1 and report["reason_codes"][0] in {"SCHEDULE_SCOPE_NOT_APPROVED","IMPLEMENTATION_BYTES_NOT_APPROVED","SUCCESSOR_GUARD_LOGIC_CHANGED"}


@pytest.mark.parametrize("path",["shadow/app_core/ncaaf_schedule.py","shadow/core/streamlit_pipeline.py","shadow/streamlit_app.py","sitecustomize.py","hook.pth"])
def test_untracked_runtime_shadows_and_hooks_are_rejected(fx,path):
    write(fx[0],path,b"unauthorized runtime shadow\n")
    code,report=assess(fx)
    assert code==1 and report["reason_codes"]==["PROTECTED_RUNTIME_SHADOWING_RISK"]


@pytest.mark.parametrize("field",["base_sha","base_tree","implementation_tree","policy_version","approval_reference"])
def test_incorrect_candidate_and_approval_identities_rejected(fx,field):
    repo,_,_,_,sealed=fx
    sealed[field]="wrong"
    write(repo,g.SCHEDULE_POLICY_PATH,json.dumps(sealed).encode());commit(repo)
    assert assess(fx)[0]==1
