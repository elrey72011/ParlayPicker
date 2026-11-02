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
PROFILE_SHA256='b4d4bd2fc5c331e51843589733f36e229c0e46c3d261ac50594e282aa5e3e120'
WORKFLOW_SHA256='8e44af98864d6bd86f883316b18be3a2a232f2c5223d6c4414ffda3bfa4487a1'
ASSIGNMENT_SHA256='4d5bfeea76a87e266d1f0c2ab194d3197d53049a77505ee24a1b8ef0f4f05cbc'
OLD_POLICY='docs/paid-launch/launch-scope-policy-nfl-account-reset-v1.json'
POLICY='docs/paid-launch/launch-scope-policy-ci-fifteen-v6.json'
VALIDATION='docs/paid-launch/ci-fifteen-validation-v6.json'
IMPLEMENTATION_PARENT='5b9d4827fd2965996eca26533b8d568ae046888e'
IMPLEMENTATION_PARENT_TREE='e2f7e76a64237055b4e9d3b85b1f922f6283fdb5'
MAIN_PARENT='762cde0e88703a2fada5a2a8ef1bef2ca0a2df12'
MAIN_PARENT_TREE='ac781142ad1f18b8bfb2f4e5ee730bd3705f3416'
PAID_WORKFLOW='.github/workflows/paid-launch.yml'
CAPACITY_PARENT='cd181088d1c685f377b0b2f3227881131d0f3ca5'
CAPACITY_PARENT_TREE='17d3da40c09f4ae68c3aa60e74dfca2b2f5413db'
PROFILE_DESCRIPTOR='docs/paid-launch/ci-fifteen-native-lpt-profile-successor-v1.json'
PROFILE_DESCRIPTOR_SHA256='6d9958ab48ae8b53ca6705969becb4dea695b617953eeb8e5e03963c565f61c2'
LOCAL_AUTHORIZATION='docs/paid-launch/ci-fifteen-native-lpt-local-authorization-v1.md'
OLD_CONTEXT_RECORD='docs/paid-launch/nfl-capacity-context-generation-v1.json'
LOCAL_PARENT='839bf9a3b9f1addb3ac7cdfd656167177589be81'
LOCAL_PARENT_TREE='59c6b679b03959f1013e9263d29619460e4c7c8c'
PROFILE_CONSUMER='docs/paid-launch/ci-fifteen-retained-profile-input-v1.json'
CONTEXT_RECORD='docs/paid-launch/nfl-capacity-context-generation-v3.json'
INPUTS='docs/paid-launch/ci-fifteen-qualification-inputs-v2.json'
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
    require(inputs['profile_sha256']==PROFILE_SHA256 and inputs['predecessor_head']==CAPACITY_PARENT
        and inputs['predecessor_tree']==CAPACITY_PARENT_TREE and inputs['actual_ci_reserves']=='UNKNOWN'
        and inputs['planning_gate_seconds']==180 and inputs['required_actual_reserve_seconds']==120,
        'CI_FIFTEEN_QUALIFICATION_INPUT_CHANGED')
    require(text(root,'rev-parse',CAPACITY_PARENT+'^{tree}')==CAPACITY_PARENT_TREE,'CI_LPT_PARENT_CHANGED')
    declaration=json.loads(physical(Path(root)/PROFILE_DESCRIPTOR))
    require(sha(physical(Path(root)/PROFILE_DESCRIPTOR))==PROFILE_DESCRIPTOR_SHA256
        and declaration['predecessor_head']==CAPACITY_PARENT and declaration['predecessor_tree']==CAPACITY_PARENT_TREE
        and declaration['profile_sha256']==PROFILE_SHA256 and declaration['assignment_sha256']==ASSIGNMENT_SHA256,
        'CI_LPT_DESCRIPTOR_CHANGED')
    before_raw=git(root,'show',CAPACITY_PARENT+':scripts/ci_test_file_costs_v1.json')
    require(sha(before_raw)==declaration['predecessor_profile_sha256']=='5513e877c0e467bf8217dce1b041453e942d4610af1c5734e72519cb197ea347','CI_LPT_PROFILE_PARENT_CHANGED')
    before=json.loads(before_raw);after=json.loads(physical(Path(root)/'scripts/ci_test_file_costs_v1.json'))
    expected=json.loads(before_raw);rows={r['path']:r for r in expected['files']};seen=set()
    for row in declaration['changes']:
        path=row['path'];require(path not in seen and rows.get(path)==row['before'],'CI_LPT_COST_INPUT_CHANGED');seen.add(path)
        old=row['before'];new=row['after'];require(new['path']==path and new['kind']=='estimated'
            and(new['profiled_tests']==old['profiled_tests']if old is not None else new['profiled_tests']==0)
            and(old is None or old['seconds']is None or new['seconds']>=old['seconds']-1e-9),'CI_LPT_ESTIMATE_CHANGED')
        if old is None:expected['files'].append(new)
        else:expected['files'][expected['files'].index(old)]=new
        rows[path]=new
    source=declaration['source'];require(source['measured']is False and source['kind']=='estimated_retained_complete_context_envelope'
        and source['checkout_commit']==CAPACITY_PARENT and source['checkout_tree']==CAPACITY_PARENT_TREE
        and source['variability_multiplier']==1.15 and source['fixture_allowance_seconds']==5.0,'CI_LPT_ESTIMATE_CHANGED')
    expected['sources'][declaration['source_tag']]=source
    require(json.dumps(expected,sort_keys=True,separators=(',',':'))==json.dumps(after,sort_keys=True,separators=(',',':'))
        and expected==after,'CI_LPT_PROFILE_TRANSITION_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v3.json','docs/paid-launch/ci-fifteen-validation-v3.json',
                 'docs/paid-launch/ci-fifteen-qualification-inputs-v1.json',OLD_CONTEXT_RECORD):
        require(physical(Path(root)/path)==git(root,'show',CAPACITY_PARENT+':'+path),'CI_LPT_PRIOR_SEAL_CHANGED')
    require(text(root,'rev-parse',LOCAL_PARENT+'^{tree}')==LOCAL_PARENT_TREE,'CI_LPT_LOCAL_PARENT_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v4.json','docs/paid-launch/ci-fifteen-validation-v4.json','docs/paid-launch/nfl-capacity-context-generation-v2.json'):
        require(physical(Path(root)/path)==git(root,'show',LOCAL_PARENT+':'+path),'CI_LPT_FAILED_SEAL_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v5.json','docs/paid-launch/ci-fifteen-validation-v5.json'):
        require(physical(Path(root)/path)==git(root,'show',LOCAL_PARENT+':'+path),'CI_LPT_IMPORT_FAILURE_SEAL_CHANGED')
    consumer=json.loads(physical(Path(root)/PROFILE_CONSUMER))
    require(consumer['version']=='exact-pre-lpt-physical-input-v1'
        and consumer['consumer']=='tests/test_ncaaf_sidebar_startup_scope.py'
        and consumer['old_head']==CAPACITY_PARENT
        and consumer['old_binding_sha256']=='d84983e285490da3e38128ceeec1b9c907632eeea62596e8e3fe3628d29499a1'
        and consumer['current_profile_sha256']==PROFILE_SHA256,'CI_LPT_CONSUMER_GENERATION_CHANGED')





def main_caller_contract(root):
    """Exact event-to-base transition; native readers' admitted bases unchanged."""
    require(text(root,'rev-parse',MAIN_PARENT+'^{tree}')==MAIN_PARENT_TREE,'CI_MAIN_CALLER_PARENT_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v2.json',
                 'docs/paid-launch/ci-fifteen-validation-v2.json'):
        require(physical(Path(root)/path)==git(root,'show',MAIN_PARENT+':'+path),
                'CI_MAIN_CALLER_PRIOR_SEAL_CHANGED')
    before=git(root,'show',MAIN_PARENT+':'+PAID_WORKFLOW)
    old=b'        env:\n          BASE_SHA: ${{ github.event.pull_request.base.sha || \'5fb8e13577c3092f1eda4ad9b787368d3f691c71\' }}\n        run: python scripts/check_launch_change_scope.py --base "$BASE_SHA" --json-output protected-code-diff-report.json\n'
    new=b'        env:\n          SCOPE_EVENT: ${{ github.event_name }}\n          SCOPE_REF: ${{ github.ref }}\n          PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}\n        run: |\n          scope_args=()\n          case "$SCOPE_EVENT" in\n            pull_request)\n              test -n "$PR_BASE_SHA"\n              scope_args=(--base "$PR_BASE_SHA")\n              ;;\n            push)\n              test "$SCOPE_REF" = refs/heads/main\n              test -z "$PR_BASE_SHA"\n              ;;\n            *)\n              echo "Unsupported protected-scope event" >&2\n              exit 1\n              ;;\n          esac\n          python scripts/check_launch_change_scope.py "${scope_args[@]}" --json-output protected-code-diff-report.json\n'
    require(before.count(old)==1,'CI_MAIN_CALLER_PARENT_CHANGED')
    after=physical(Path(root)/PAID_WORKFLOW)
    require(after==before.replace(old,new,1),'CI_MAIN_CALLER_TRANSITION_CHANGED')
    return dict(version=1,predecessor_head=MAIN_PARENT,predecessor_tree=MAIN_PARENT_TREE,
        workflow=PAID_WORKFLOW,before_sha256=sha(before),after_sha256=sha(after),
        pr_base='actual_pull_request_base_sha',main_base='native_authenticated_baseline_no_override',
        push_ref='refs/heads/main',unexpected_event='REJECT',native_allowed_bases_unchanged=True)


def make_contract(root,stamp,components):
    root=Path(root);binding='docs/research/ncaaf-filtered-history-bindings-v1.json'
    before=git(root,'show',PARENT+':.github/workflows/ci.yml')
    require(text(root,'rev-parse',PARENT+'^{tree}')==PARENT_TREE
        and sha(git(root,'show',PARENT+':'+binding))==PARENT_BINDING,'CI_FIFTEEN_PARENT_CHANGED')
    require(sha(physical(root/'.github/workflows/ci-fifteen.yml'))==WORKFLOW_SHA256,'CI_FIFTEEN_WORKFLOW_CHANGED')
    require(physical(root/'.github/workflows/ci.yml')==git(root,'show',PARENT+':.github/workflows/ci.yml'),'CI_THIRTEEN_HISTORICAL_WORKFLOW_CHANGED')
    consumers={p:sha(git(root,'show',PARENT+':'+p))for p in sorted(HISTORICAL_CONSUMERS)}
    require(all(sha(physical(root/p))==value for p,value in consumers.items()),'CI_FIFTEEN_HISTORICAL_CONSUMER_CHANGED')
    require(text(root,'rev-parse',IMPLEMENTATION_PARENT+'^{tree}')==IMPLEMENTATION_PARENT_TREE,
        'CI_FIFTEEN_CORRECTION_PARENT_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v1.json',
                 'docs/paid-launch/ci-fifteen-validation-v1.json'):
        require(physical(root/path)==git(root,'show',IMPLEMENTATION_PARENT+':'+path),
            'CI_FIFTEEN_FAILED_SEAL_CHANGED')
    contract=dict(version=1,generation='authenticated-a5430e5-thirteen-workflow-v1',
        profile_generation_correction=dict(predecessor_head=IMPLEMENTATION_PARENT,
            predecessor_tree=IMPLEMENTATION_PARENT_TREE,context_record_sha256=components[OLD_CONTEXT_RECORD]),
        predecessor_head=PARENT,predecessor_tree=PARENT_TREE,predecessor_binding_sha256=PARENT_BINDING,
        predecessor_policy_sha256=sha(git(root,'show',PARENT+':'+OLD_POLICY)),
        historical_consumers=consumers,historical_workflow_path='.github/workflows/ci.yml',live_workflow_path='.github/workflows/ci-fifteen.yml',
        old_workflow_state_required_before_push='disabled',workflow_after_sha256=WORKFLOW_SHA256,
        workflow_before=dict(bytes=len(before),sha256=sha(before),compressed_b85=base64.b85encode(zlib.compress(before,9)).decode()),
        profile_sha256=PROFILE_SHA256,assignment_sha256=ASSIGNMENT_SHA256,shards=15,timeout_seconds=2700,
        inventory_sha256=sha(physical(root/INVENTORY)),qualification_inputs_sha256=sha(physical(root/INPUTS)),
        authorization_sha256=components[AUTHORIZATION],planning_gate_seconds=180,required_actual_reserve_seconds=120,
        hosted_authority=False,publication_authority=False)
    contract['main_caller_successor']=main_caller_contract(root)
    contract['native_lpt_profile_successor']=dict(predecessor_head=CAPACITY_PARENT,predecessor_tree=CAPACITY_PARENT_TREE,descriptor_sha256=PROFILE_DESCRIPTOR_SHA256,context_record_sha256=components[CONTEXT_RECORD],local_authorization_sha256=components[LOCAL_AUTHORIZATION],profile_sha256=PROFILE_SHA256,assignment_sha256=ASSIGNMENT_SHA256)
    contract['retained_profile_consumer_generation']=dict(predecessor_head=LOCAL_PARENT,predecessor_tree=LOCAL_PARENT_TREE,logical_head=CAPACITY_PARENT,logical_binding_sha256='d84983e285490da3e38128ceeec1b9c907632eeea62596e8e3fe3628d29499a1',descriptor_sha256=components[PROFILE_CONSUMER],helper_sha256=components['tests/native_lpt_profile_inputs.py'],conftest_sha256=components['tests/conftest.py'])
    return contract


def validate_topology(root,binding):
    runner=_source_module(root,'scripts/run_ci_tests.py','6be44f5bdf5934dea5784cd1819595e9c8386513fffa0769854e95d4095302fe')
    root=Path(root);metadata=binding['nfl_account_reset_successor']['ci_fifteen_contract']
    require(metadata.get('main_caller_successor')==main_caller_contract(root),'CI_MAIN_CALLER_CONTRACT_CHANGED')
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
    return dict(version='ci-fifteen-native-lpt-local-only-seal-v6',base=scope.BASE,base_tree=scope.TREE,
        predecessor_head=PARENT,predecessor_tree=PARENT_TREE,predecessor_binding_sha256=PARENT_BINDING,
        retained_thirteen_policy_sha256=sha(physical(Path(root)/OLD_POLICY)),
        binding_sha256=binding_sha,implementation_commit=implementation,
        implementation_tree=evidence['implementation_tree'],implementation_parent=LOCAL_PARENT,implementation_paths=sorted(scope.PATHS),
        unchanged_blobs={p:v for p,v in scope.blobs(root,scope.BASE).items()if p not in scope.PATHS},
        topology=validate_topology(root,binding),validation_sha256=sha(physical(Path(root)/VALIDATION)),
        validation_status=evidence['status'],operational_authority=False,publication_authority=False,
        ci_authority=False,maximum_ci_cycles=0,
        workflow_state_prerequisite=dict(disable_before_push='.github/workflows/ci.yml',live_successor='.github/workflows/ci-fifteen.yml',external_authorization=False),
        condition='Local qualification only. All previous hosted grants are consumed. A new exact-head publication/CI decision is required after local correctness and capacity pass;no push,retry,merge or deployment authority.')


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
    require(text(root,'show','-s','--format=%P',implementation).split()==[LOCAL_PARENT]
        and text(root,'merge-base',scope.BASE,LOCAL_PARENT)==scope.BASE,'IMPLEMENTATION_PARENT_NOT_APPROVED')
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
