"""Local-only authenticated fifteen-shard successor; no publication or CI grant."""
from pathlib import Path
from types import SimpleNamespace
import base64
import hashlib
import json
import subprocess
import zlib
import tempfile
import types

PARENT='a5430e5a93cece580d6645720b41f6a0d922d74a'
PARENT_TREE='7c5afb019fd81eacdff0794728ede6118a00cc49'
PARENT_BINDING='f7bbd9f837b6cb7369b5b2179e682e630ff3eec7519dcd2f6939756d233383ee'
PARENT_PROFILE='91379959e127cc38cc76e0845ea481701ff359a99f382eb0281cae5cf72009a0'
PROFILE_SHA256='5513e877c0e467bf8217dce1b041453e942d4610af1c5734e72519cb197ea347'
WORKFLOW_SHA256='8e44af98864d6bd86f883316b18be3a2a232f2c5223d6c4414ffda3bfa4487a1'
ASSIGNMENT_SHA256='59f968743e59ba1c0bcbc2292cd38bab6d7799aff0edcc9aab6c53ca6cf908f0'
OLD_POLICY='docs/paid-launch/launch-scope-policy-nfl-account-reset-v1.json'
POLICY='docs/paid-launch/launch-scope-policy-ci-fifteen-v1.json'
VALIDATION='docs/paid-launch/ci-fifteen-validation-v1.json'
INPUTS='docs/paid-launch/ci-fifteen-qualification-inputs-v1.json'
INVENTORY='docs/paid-launch/ci-fifteen-inventory-v1.json'
AUTHORIZATION='docs/paid-launch/ci-fifteen-local-qualification-v1.md'
HISTORICAL_CONSUMERS=frozenset({
    'tests/test_ci_test_shards.py','tests/test_ci_five_shards.py',
    'tests/test_ci_six_shards.py','tests/test_ncaaf_staged_ci_capacity.py',
    'tests/test_historical_scope_input_boundary.py'})


def require(value,reason):
    if not value:raise ValueError(reason)


def physical_bytes(path):
    import io
    with io.FileIO(path,'r')as f:return f.read()


def physical(path):
    raw=physical_bytes(path)
    require(len(raw)<=8*1024*1024,'CI_FIFTEEN_INPUT_LIMIT')
    return raw.replace(b'\r\n',b'\n')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def text(root,*args):return git(root,*args).decode().strip()


def _source_module(root,relative,expected):
    raw=physical(Path(root)/relative)
    require(sha(raw)==expected,'CI_FIFTEEN_HELPER_CHANGED')
    module=types.ModuleType('exact_ci_fifteen_'+Path(relative).stem);module.__file__=str(Path(root)/relative)
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    return module


def _scope(root):
    binding=json.loads(physical(Path(root)/'docs/research/ncaaf-filtered-history-bindings-v1.json'))
    relative='scripts/nfl_account_reset_scope.py'
    return _source_module(root,relative,binding['components'][relative])


def verify_profile(root):
    # Full profile identity includes every retained measured/estimated classification.
    require(sha(physical(Path(root)/'scripts/ci_test_file_costs_v1.json'))==PROFILE_SHA256,'CI_FIFTEEN_PROFILE_CHANGED')
    require(sha(git(root,'show',PARENT+':scripts/ci_test_file_costs_v1.json'))==PARENT_PROFILE,'CI_FIFTEEN_PARENT_PROFILE_CHANGED')
    inputs=json.loads(physical(Path(root)/INPUTS))
    require(inputs['profile_sha256']==PROFILE_SHA256 and inputs['predecessor_head']==PARENT
        and inputs['predecessor_tree']==PARENT_TREE and inputs['actual_ci_reserves']=='UNKNOWN'
        and inputs['planning_gate_seconds']==180 and inputs['required_actual_reserve_seconds']==120,
        'CI_FIFTEEN_QUALIFICATION_INPUT_CHANGED')


def make_contract(root,stamp,components):
    root=Path(root);binding='docs/research/ncaaf-filtered-history-bindings-v1.json'
    before=git(root,'show',PARENT+':.github/workflows/ci.yml')
    require(text(root,'rev-parse',PARENT+'^{tree}')==PARENT_TREE
        and sha(git(root,'show',PARENT+':'+binding))==PARENT_BINDING,'CI_FIFTEEN_PARENT_CHANGED')
    require(sha(physical(root/'.github/workflows/ci-fifteen.yml'))==WORKFLOW_SHA256,'CI_FIFTEEN_WORKFLOW_CHANGED')
    require(physical(root/'.github/workflows/ci.yml')==git(root,'show',PARENT+':.github/workflows/ci.yml'),'CI_THIRTEEN_HISTORICAL_WORKFLOW_CHANGED')
    consumers={p:sha(git(root,'show',PARENT+':'+p))for p in sorted(HISTORICAL_CONSUMERS)}
    require(all(sha(physical(root/p))==value for p,value in consumers.items()),'CI_FIFTEEN_HISTORICAL_CONSUMER_CHANGED')
    return dict(version=1,generation='authenticated-a5430e5-thirteen-workflow-v1',
        predecessor_head=PARENT,predecessor_tree=PARENT_TREE,predecessor_binding_sha256=PARENT_BINDING,
        predecessor_policy_sha256=sha(git(root,'show',PARENT+':'+OLD_POLICY)),
        historical_consumers=consumers,historical_workflow_path='.github/workflows/ci.yml',live_workflow_path='.github/workflows/ci-fifteen.yml',
        old_workflow_state_required_before_push='disabled',workflow_after_sha256=WORKFLOW_SHA256,
        workflow_before=dict(bytes=len(before),sha256=sha(before),compressed_b85=base64.b85encode(zlib.compress(before,9)).decode()),
        profile_sha256=PROFILE_SHA256,assignment_sha256=ASSIGNMENT_SHA256,shards=15,timeout_seconds=2700,
        inventory_sha256=sha(physical(root/INVENTORY)),qualification_inputs_sha256=sha(physical(root/INPUTS)),
        authorization_sha256=components[AUTHORIZATION],planning_gate_seconds=180,required_actual_reserve_seconds=120,
        hosted_authority=False,publication_authority=False)


def validate_topology(root,binding):
    runner=_source_module(root,'scripts/run_ci_tests.py','6be44f5bdf5934dea5784cd1819595e9c8386513fffa0769854e95d4095302fe')
    root=Path(root);metadata=binding['nfl_account_reset_successor']['ci_fifteen_contract']
    require(metadata['shards']==15 and metadata['timeout_seconds']==2700
        and metadata['planning_gate_seconds']==180 and metadata['required_actual_reserve_seconds']==120
        and metadata['hosted_authority']is False and metadata['publication_authority']is False,'CI_FIFTEEN_SCOPE_CHANGED')
    require(sha(physical(root/'.github/workflows/ci-fifteen.yml'))==WORKFLOW_SHA256,'CI_FIFTEEN_WORKFLOW_CHANGED')
    require(physical(root/'.github/workflows/ci.yml')==git(root,'show',PARENT+':.github/workflows/ci.yml'),'CI_THIRTEEN_HISTORICAL_WORKFLOW_CHANGED')
    previous=git(root,'show',PARENT+':.github/workflows/ci.yml').decode()
    expected=previous.replace('shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]',
        'shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]').replace('--shards 13','--shards 15').replace(
        'Reconcile complete test identities and outcomes across all nine shards',
        'Reconcile complete test identities and outcomes across all fifteen shards')
    # Exact authenticated transition preserves every timeout,runner and safety gate.
    require(physical(root/'.github/workflows/ci-fifteen.yml').decode()==expected,'CI_FIFTEEN_WORKFLOW_CONTRACT_CHANGED')
    values=json.loads(physical(root/INVENTORY));require(type(values)is list and len(values)==len(set(values))==9403,'CI_FIFTEEN_INVENTORY_CHANGED')
    items=[SimpleNamespace(nodeid=n,path=root/n.split('::')[0])for n in values]
    # Keep native parsing unchanged, independent of historical Path views and
    # platform checkout newlines. No process-global patch or cached verdict.
    with tempfile.TemporaryDirectory(prefix='ci15-profile-')as temporary:
        path=Path(temporary)/'physical-profile.json'
        path.write_bytes(physical(root/'scripts/ci_test_file_costs_v1.json'))
        profile=runner.load_profile(path)
    rows,files,loads=runner.assignment(items,15,profile,root)
    require(profile['sha256']==PROFILE_SHA256 and runner.digest(rows)==ASSIGNMENT_SHA256,'CI_FIFTEEN_ASSIGNMENT_CHANGED')
    inputs=json.loads(physical(root/INPUTS));reserves=[2700-151.10075-v for v in loads]
    require(reserves==inputs['projected_reserves']and min(reserves)>=180
        and len(inputs['stress_reserves'])==15 and min(inputs['stress_reserves'])>=180,'CI_FIFTEEN_PLANNING_GATE_FAILED')
    return dict(shards=15,identity_count=9403,whole_files=True,order_preserved=True,
        assignment_sha256=ASSIGNMENT_SHA256,profile_sha256=PROFILE_SHA256,
        minimum_projected_reserve=min(reserves),minimum_stress_reserve=min(inputs['stress_reserves']),
        planning_gate_seconds=180,required_actual_reserve_seconds=120,actual_ci_reserves='UNKNOWN')


def make_policy(root,binding,implementation,binding_sha):
    scope=_scope(root)
    evidence=json.loads(physical(Path(root)/VALIDATION))
    require(evidence['implementation_commit']==implementation
        and evidence['implementation_tree']==text(root,'rev-parse',implementation+'^{tree}')
        and evidence['binding_sha256']==binding_sha,'CI_FIFTEEN_VALIDATION_SUBJECT_CHANGED')
    return dict(version='ci-fifteen-local-only-seal-v1',base=scope.BASE,base_tree=scope.TREE,
        predecessor_head=PARENT,predecessor_tree=PARENT_TREE,predecessor_binding_sha256=PARENT_BINDING,
        retained_thirteen_policy_sha256=sha(physical(Path(root)/OLD_POLICY)),
        binding_sha256=binding_sha,implementation_commit=implementation,
        implementation_tree=evidence['implementation_tree'],implementation_parent=PARENT,implementation_paths=sorted(scope.PATHS),
        unchanged_blobs={p:v for p,v in scope.blobs(root,scope.BASE).items()if p not in scope.PATHS},
        topology=validate_topology(root,binding),validation_sha256=sha(physical(Path(root)/VALIDATION)),
        validation_status=evidence['status'],operational_authority=False,publication_authority=False,
        ci_authority=False,maximum_ci_cycles=0,
        workflow_state_prerequisite=dict(disable_before_push='.github/workflows/ci.yml',live_successor='.github/workflows/ci-fifteen.yml',external_authorization=False),
        condition='Local-only qualification. The unused thirteen-shard CI grant does not authorize fifteen. Exact owner amendment and publication hold lifting remain required; no retries, merge or deployment.')


def validate(root,binding,comparison_base,manifest,*,binding_sha):
    scope=_scope(root)
    root=Path(root)
    require(text(root,'rev-parse',scope.BASE+'^{tree}')==scope.TREE,'NFL_ACCOUNT_BASE_CHANGED')
    require(comparison_base in (None,scope.BASE),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'status','--porcelain'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==scope.binding_successor(root,binding_sha),'NFL_ACCOUNT_BINDING_CHANGED')
    require(all(sha(scope.normalized(p,physical(root/p),binding_sha))==value for p,value in scope.QUALIFIED_SOURCES.items()),'NFL_ACCOUNT_SOURCE_NOT_QUALIFIED')
    for p in scope.NORMALIZED:
        require(scope.reader_entry_parent(p,scope.normalized(p,physical(root/p),binding_sha))==scope.normalized(p,git(root,'show',scope.BASE+':'+p),scope.OLD_STAMP),'INHERITED_GUARD_LOGIC_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[PARENT]
        and text(root,'merge-base',scope.BASE,PARENT)==scope.BASE,'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',scope.BASE,implementation).splitlines())==scope.PATHS,'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');seal=head;parents=text(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents==[scope.BASE,parents[1]],'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
        and set(text(root,'diff','--name-status',implementation,seal).splitlines())=={'A\t'+POLICY,'A\t'+VALIDATION},'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=scope.blobs(root,'HEAD');require(actual.keys()==scope.blobs(root,scope.BASE).keys()|scope.PATHS|{POLICY,VALIDATION},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for p,oid in actual.items():
        raw=physical_bytes(root/p);blob=lambda v:hashlib.sha1(b'blob '+str(len(v)).encode()+b'\0'+v).hexdigest()
        require(oid in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=109,status='PASS',base=scope.BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(scope.PATHS|{POLICY,VALIDATION}),
        protected_changes=[],existing_test_changes=[],accepted=False,inference=False,
        ci_authority=False,publication_authority=False,qualification_status=policy['validation_status'],**policy['topology'])
