"""Exact offline successor policy for bounded research estimate display/metadata work."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def make_policy(g, binding, implementation):
    manifest = json.loads(g.git_bytes("show", binding["base"]+":"+g.MANIFEST_PATH))
    retained = (set(manifest["protected_files"]) | set(g.DFS_UNCHANGED_PATHS) |
                set(g.DFS_PATHS) | set(g.DRIVE_PATHS) | {g.V4_POLICY_PATH,g.DRIVE_POLICY_PATH}) - set(g.ESTIMATE_PATHS)
    return dict(schema_version=6, policy_version=g.ESTIMATE_POLICY_VERSION,
        approval_reference=g.ESTIMATE_APPROVAL_REFERENCE, base_sha=binding["base"],
        base_tree=binding["base_tree"], original_manifest_sha256=binding["manifest_sha256"],
        previous_policy_blob=binding["previous_policy_blob"], implementation_commit=implementation,
        implementation_tree=g.git("rev-parse", implementation+"^{tree}"),
        implementation_changes={p:{"before_blob":g.blob(binding["base"],p),
                                   "after_blob":g.blob(implementation,p)} for p in g.ESTIMATE_PATHS},
        tooling_sha256={p:hashlib.sha256(g.git_bytes("show", implementation+":"+p)).hexdigest()
                        for p in manifest["tooling_sha256"]},
        unchanged_bindings={p:g.blob(binding["base"],p) for p in sorted(retained)})


def validate(g, manifest_path, base, binding):
    require = g._require
    require(manifest_path.resolve() == (g.ROOT/g.MANIFEST_PATH).resolve(), "MANIFEST_PATH_NOT_APPROVED")
    require(base in (None, binding["base"], "5fb8e13577c3092f1eda4ad9b787368d3f691c71"), "COMPARISON_BASE_NOT_APPROVED")
    require(g.git("rev-parse", binding["base"]+"^{tree}") == binding["base_tree"], "STARTING_MAIN_TREE_CHANGED")
    manifest_raw = g.git_bytes("show", binding["base"]+":"+g.MANIFEST_PATH)
    require(hashlib.sha256(manifest_raw).hexdigest() == binding["manifest_sha256"], "BASELINE_CHANGED")
    require(g.blob(binding["base"],g.DRIVE_POLICY_PATH) == binding["previous_policy_blob"], "PREVIOUS_POLICY_CHANGED")
    previous_guard = g.git_bytes("show", binding["base"]+":"+g.GUARD_PATH)
    require(hashlib.sha256(previous_guard).hexdigest() == binding["previous_guard_sha256"], "PREVIOUS_GUARD_CHANGED")
    policy_raw = g.git_bytes("show", "HEAD:"+g.ESTIMATE_POLICY_PATH)
    policy = json.loads(policy_raw)
    require((g.ROOT/g.ESTIMATE_POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == policy_raw, "POLICY_CHECKOUT_CHANGED")
    implementation = policy["implementation_commit"]
    require(g.git("show","-s","--format=%P",implementation).split() == [binding["base"]], "IMPLEMENTATION_PARENT_NOT_APPROVED")
    require(not g.exists_at(implementation,g.ESTIMATE_POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    require(set(g.git("diff","--name-only",binding["base"],implementation).splitlines()) == set(g.ESTIMATE_PATHS), "IMPLEMENTATION_CHANGE_SET_NOT_APPROVED")
    require(policy == make_policy(g,binding,implementation), "EXACT_POLICY_BINDINGS_CHANGED")
    for path, reviewed in binding["reviewed_blobs"].items():
        require(g.blob(implementation,path) == reviewed, "REVIEWED_IMPLEMENTATION_BYTES_CHANGED")
    require(set(binding["reviewed_blobs"]) == set(g.ESTIMATE_PATHS)-{g.GUARD_PATH}, "REVIEWED_BINDING_SET_CHANGED")
    successor = g.git_bytes("show",implementation+":"+g.GUARD_PATH)
    require(g._dfs_guard_matches(successor,binding["successor_guard_sha256"]), "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    require(g._estimate_previous_guard_source(successor, binding) == previous_guard, "PREVIOUS_GUARD_LOGIC_CHANGED")
    for path in g.ESTIMATE_PRIOR_SOURCE_RECONSTRUCTIONS:
        if not path.startswith("tests/"):
            continue
        fixture = g.git_bytes("show", implementation+":"+path)
        require(g._estimate_previous_main_source(path, fixture) ==
                g.git_bytes("show", binding["base"]+":"+path), "PRIOR_ASSERTIONS_CHANGED")
    parents = g.git("show","-s","--format=%P","HEAD").split()
    if len(parents)==2:
        require(parents[0]==binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate=parents[1]
        require(g.git("rev-parse","HEAD^{tree}")==g.git("rev-parse",candidate+"^{tree}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate=g.git("rev-parse","HEAD")
    require(g.git("show","-s","--format=%P",candidate).split()==[implementation], "CANDIDATE_NOT_POLICY_SEAL")
    require(g.git("diff","--name-status",implementation,candidate).splitlines()==["A\t"+g.ESTIMATE_POLICY_PATH], "SEAL_CHANGE_SET_NOT_APPROVED")
    require(not g.git("diff","--name-only") and not g.git("diff","--cached","--name-only"), "UNAUTHORIZED_CHECKOUT_CHANGE")
    require(all(g.blob("HEAD",p)==value for p,value in policy["unchanged_bindings"].items()), "IMMUTABLE_FILE_CHANGED")
    require(policy["tooling_sha256"][".github/workflows/paid-launch.yml"] ==
            json.loads(manifest_raw)["tooling_sha256"][".github/workflows/paid-launch.yml"], "WORKFLOW_CHANGED")
    conversions=[]
    checked=set(policy["unchanged_bindings"]) | set(g.ESTIMATE_PATHS) | {g.ESTIMATE_POLICY_PATH}
    for path in checked:
        committed=g.git_bytes("show","HEAD:"+path)
        checkout=(g.ROOT/path).read_bytes()
        require(checkout.replace(b"\r\n",b"\n")==committed, "UNAUTHORIZED_CHECKOUT_CHANGE")
        if path in policy["tooling_sha256"] and checkout != committed:
            conversions.append(path)
    for path in checked:
        suffix=Path(path).parts
        for other in g.ROOT.rglob(Path(path).name):
            relative=other.relative_to(g.ROOT)
            if relative.as_posix()=="parlaypicker/app/streamlit_app.py" and relative.as_posix() in policy["unchanged_bindings"]:
                continue
            require(tuple(relative.parts[-len(suffix):]) != tuple(suffix) or relative.as_posix()==path,
                    "PROTECTED_RUNTIME_SHADOWING_RISK")
    require(not any(p.is_file() and (p.name in {"sitecustomize.py","usercustomize.py"} or p.suffix==".pth")
                    for p in g.ROOT.rglob("*")), "PROTECTED_RUNTIME_SHADOWING_RISK")
    return policy,conversions


def run(g, manifest_path, base, binding):
    try:
        policy, conversions=validate(g,manifest_path,base,binding)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.CalledProcessError) as exc:
        return 1, dict(schema_version=6,status="FAIL",policy_valid=False,reason_codes=[str(exc)])
    _, original=g.run(manifest_path,base)
    report=copy.deepcopy(original)
    report.update(schema_version=6,policy_version=policy["policy_version"],policy_valid=True,
        original_guard_report=original,approved_integration_changes=policy["implementation_changes"],
        checkout_line_ending_conversions=conversions,new_existing_test_exceptions=[],
        approved_fixture_source_reconstructions={p:policy["implementation_changes"][p]
            for p in g.ESTIMATE_PRIOR_SOURCE_RECONSTRUCTIONS if p.startswith("tests/")})
    report["protected_changes"]=[row for row in original["protected_changes"] if row["path"] not in set(g.ESTIMATE_PATHS)|set(g.DRIVE_PATHS)]
    retained_clock = policy["unchanged_bindings"].get(g.CLOCK_TEST) == g.DFS_BINDINGS["clock_blob"]
    report["approved_exceptions"] = ([dict(path=g.CLOCK_TEST, before_blob=g.PRODUCTION_BINDINGS["before"],
        after_blob=g.DFS_BINDINGS["clock_blob"], retained_unchanged=True,
        approval_reference=g.APPROVAL_REFERENCE)] if retained_clock else [])
    report["existing_test_changes"]=[p for p in original["existing_test_changes"]
        if p not in set(g.ESTIMATE_PATHS)|set(g.DRIVE_PATHS) and not (p == g.CLOCK_TEST and retained_clock)]
    report["self_protected_changes"]=[p for p in original["self_protected_changes"] if p != g.GUARD_PATH]
    report["tooling_hash_mismatches"]=[p for p in original["tooling_hash_mismatches"] if p != g.GUARD_PATH and p not in conversions]
    report["runtime_shadowing"]=[p for p in original["runtime_shadowing"] if p != "parlaypicker/app/streamlit_app.py"]
    groups={"protected_changes":"PROTECTED_FILE_CHANGED","existing_test_changes":"EXISTING_TEST_EXPECTATION_CHANGED",
        "self_protected_changes":"SCOPE_GUARD_OR_BASELINE_CHANGED","tooling_hash_mismatches":"SCOPE_GUARD_TOOLING_HASH_MISMATCH",
        "runtime_shadowing":"PROTECTED_RUNTIME_SHADOWING_RISK"}
    removed={reason for group,reason in groups.items() if not report[group]}
    report["reason_codes"]=[r for r in original["reason_codes"] if r not in removed]
    report["status"]="FAIL" if report["reason_codes"] else "PASS"
    return (1 if report["reason_codes"] else 0),report
