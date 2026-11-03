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
PROFILE_SHA256='c627bacb49c69ab13cbb309bcee1872e2bb22f41eab3248c367ff31f099ea5a2'
WORKFLOW_SHA256='e9a83c926deec4f818b47f465468905c23873dfe288f7e2fa25a5bb4ce36168e'
ASSIGNMENT_SHA256='ae9043fb6cb21d41165d71dd6d2055be2242e053a9c2a739a9a11121df56dfe7'
OLD_POLICY='docs/paid-launch/launch-scope-policy-nfl-account-reset-v1.json'
POLICY='docs/paid-launch/launch-scope-policy-ci-fifteen-v13.json'
VALIDATION='docs/paid-launch/ci-fifteen-validation-v13.json'
IMPLEMENTATION_PARENT='5b9d4827fd2965996eca26533b8d568ae046888e'
IMPLEMENTATION_PARENT_TREE='e2f7e76a64237055b4e9d3b85b1f922f6283fdb5'
MAIN_PARENT='762cde0e88703a2fada5a2a8ef1bef2ca0a2df12'
MAIN_PARENT_TREE='ac781142ad1f18b8bfb2f4e5ee730bd3705f3416'
PAID_WORKFLOW='.github/workflows/paid-launch.yml'
CAPACITY_PARENT='cd181088d1c685f377b0b2f3227881131d0f3ca5'
CAPACITY_PARENT_TREE='17d3da40c09f4ae68c3aa60e74dfca2b2f5413db'
PROFILE_DESCRIPTOR='docs/paid-launch/ci-fifteen-native-lpt-profile-successor-v3.json'
PROFILE_DESCRIPTOR_SHA256='63232ad2a2c5a26466b1d56fdbdb0818f369f0da63a753ae65c3ca11dee12e44'
LOCAL_AUTHORIZATION='docs/paid-launch/ci-fifteen-native-lpt-local-authorization-v1.md'
OLD_CONTEXT_RECORD='docs/paid-launch/nfl-capacity-context-generation-v1.json'
LOCAL_PARENT='11a19e3f3a116e7c7bbdeea7f467b25d8240f905'
FIRST_REWEIGHT_PARENT='7c5ba8a7e506896c52b2242b8bfddb8430a68c03'
FIRST_REWEIGHT_PARENT_TREE='68688e22dede1cfcc22772f214092005caf48964'
LOCAL_PARENT_TREE='298c20f81da54c5ce0812e8a31bc1f07b42c48ca'
PROFILE_CONSUMER='docs/paid-launch/ci-fifteen-retained-profile-input-v6.json'
CONTEXT_RECORD='docs/paid-launch/nfl-capacity-context-generation-v9.json'
INPUTS='docs/paid-launch/ci-fifteen-qualification-inputs-v5.json'
SEAL_PARENT='733bcc826f2a0ebfd7526211b45ffcf6e7ab2091'
SEAL_PARENT_TREE='5972310679916c367a74338162399fd037f94184'
CORRECTION_PARENT='cc3db44cee57a2f042ce22d7539f1c1dce884f54'
CORRECTION_PARENT_TREE='656a060ef09d4f9e26c2ca6f626c0a27f8dd389a'
WORKFLOW_INPUT_DESCRIPTOR='docs/paid-launch/ci-fifteen-historical-workflow-input-v1.json'
WORKFLOW_INPUT_DESCRIPTOR_SHA256='b582df6a6f3839e121c16f91f7a29481a8c2b2046f2bd4c54aee6be8e8654dd8'
REFRESH_PARENT='dbc906a6359120ca3d72abe0dfb4bad7ac62cde6'
REFRESH_PARENT_TREE='c9da8e1c14a4ee5695674331308a3072cab1bb2f'
REFRESH_DESCRIPTOR='docs/paid-launch/ci-fifteen-refresh-click-deadline-v1.json'
REFRESH_DESCRIPTOR_SHA256='0b5175d09fcc05b03cb7d465c9ba8c1ba8f831ab3fdb88fb8632e8cf5c63c9de'
JOB_TIMEOUT_SECONDS=3000
TIMEOUT_RECORD='docs/paid-launch/ci-fifteen-fifty-timeout-v1.json'
TIMEOUT_RECORD_SHA256='48792108bcf6256a3e785c2e484fcce68eb32269b25d3139290898db6299c932'
JOB_OVERHEAD_SECONDS=157.13959995999107
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
        and inputs['planning_gate_seconds']==180 and inputs['required_actual_reserve_seconds']==120
        and inputs['job_overhead_seconds']==JOB_OVERHEAD_SECONDS>=151.10075,
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
    extra=declaration['extra_source']
    require(declaration['reweight_predecessor_head']==FIRST_REWEIGHT_PARENT
        and declaration['reweight_predecessor_tree']==FIRST_REWEIGHT_PARENT_TREE
        and sha(git(root,'show',FIRST_REWEIGHT_PARENT+':scripts/ci_test_file_costs_v1.json'))==declaration['reweight_predecessor_profile_sha256']=='b4d4bd2fc5c331e51843589733f36e229c0e46c3d261ac50594e282aa5e3e120'
        and extra['measured']is False and extra['kind']=='estimated_retained_complete_context_envelope'
        and extra['checkout_commit']==FIRST_REWEIGHT_PARENT and extra['checkout_tree']==FIRST_REWEIGHT_PARENT_TREE
        and extra['variability_multiplier']==1.15 and extra['file_allowance_seconds']==5.0
        and extra['all_prior_samples_retained']is True
        and extra['job_overhead_seconds']==JOB_OVERHEAD_SECONDS,'CI_LPT_REWEIGHT_PROVENANCE_CHANGED')
    expected['sources'][declaration['extra_source_tag']]=extra
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v8.json','docs/paid-launch/ci-fifteen-validation-v8.json',
                 'docs/paid-launch/ci-fifteen-native-lpt-profile-successor-v1.json','docs/paid-launch/ci-fifteen-retained-profile-input-v1.json',
                 'docs/paid-launch/nfl-capacity-context-generation-v4.json','docs/paid-launch/ci-fifteen-qualification-inputs-v2.json'):
        require(physical(Path(root)/path)==git(root,'show',LOCAL_PARENT+':'+path),'CI_LPT_REWEIGHT_PRIOR_CHANGED')
    qualified=declaration['qualification_source']
    require(declaration['qualification_reweight_predecessor_head']==LOCAL_PARENT
        and declaration['qualification_reweight_predecessor_tree']==LOCAL_PARENT_TREE
        and sha(git(root,'show',LOCAL_PARENT+':scripts/ci_test_file_costs_v1.json'))==declaration['qualification_reweight_predecessor_profile_sha256']=='c1a89b40def593953bada7aeac9127d0a82aa8ef4ec06469d7b2d0a352566e17'
        and qualified['measured']is False and qualified['kind']=='estimated_retained_complete_context_envelope'
        and qualified['checkout_commit']==LOCAL_PARENT and qualified['checkout_tree']==LOCAL_PARENT_TREE
        and qualified['variability_multiplier']==1.15 and qualified['file_allowance_seconds']==5.0
        and qualified['all_prior_samples_retained']is True
        and qualified['job_overhead_seconds']==JOB_OVERHEAD_SECONDS,'CI_LPT_QUALIFIED_CONTEXT_PROVENANCE_CHANGED')
    expected['sources'][declaration['qualification_source_tag']]=qualified
    require(declaration['qualification_prior_hashes']=={'docs/paid-launch/launch-scope-policy-ci-fifteen-v9.json': '03e2e92f605c6232d5924701b939f7281d4e9d4e878128e3b4fd6152b07589c0', 'docs/paid-launch/ci-fifteen-validation-v9.json': 'f877bb3bbb4fda55560f54f90034b30a0856b0e46962c6b9bdbf6102f9ad3117', 'docs/paid-launch/ci-fifteen-native-lpt-profile-successor-v2.json': '2fac3cd686129e9f2b3f5f70b35871c8e9b0e8a5e167659fc438539d9c92eff3', 'docs/paid-launch/ci-fifteen-retained-profile-input-v2.json': '71bfcc784ef60ee8777a2a7ab6f7ee30d8aeaa92b0c7e86862a835df9b1a72d2', 'docs/paid-launch/nfl-capacity-context-generation-v5.json': '560c8de2875fe2d634570b1787cbbe9d4b0dcaeceeee3c198fcf50fcde80db6c', 'docs/paid-launch/ci-fifteen-qualification-inputs-v3.json': '274426bf5120961beafe8b1ec642199b4215c3d6f9088df57e72c1c49d0b519c'},'CI_LPT_QUALIFICATION_PRIOR_DECLARATION_CHANGED')
    for path,digest in declaration['qualification_prior_hashes'].items():
        require(sha(physical(Path(root)/path))==digest,'CI_LPT_QUALIFICATION_PRIOR_CHANGED')
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
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v6.json','docs/paid-launch/ci-fifteen-validation-v6.json','docs/paid-launch/nfl-capacity-context-generation-v3.json'):
        require(physical(Path(root)/path)==git(root,'show',LOCAL_PARENT+':'+path),'CI_LPT_GENERATED_FAILURE_SEAL_CHANGED')
    require(physical(Path(root)/'tests/conftest.py')==git(root,'show',CAPACITY_PARENT+':tests/conftest.py'),'CI_LPT_ROOT_CONFTEST_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v7.json','docs/paid-launch/ci-fifteen-validation-v7.json'):
        require(physical(Path(root)/path)==git(root,'show',LOCAL_PARENT+':'+path),'CI_LPT_COLD_ALIAS_FAILURE_SEAL_CHANGED')
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


def verify_timeout(root):
    declaration=json.loads(physical(Path(root)/TIMEOUT_RECORD))
    require(sha(physical(Path(root)/TIMEOUT_RECORD))==TIMEOUT_RECORD_SHA256
        and declaration['predecessor_head']==SEAL_PARENT and declaration['predecessor_tree']==SEAL_PARENT_TREE
        and text(root,'rev-parse',SEAL_PARENT+'^{tree}')==SEAL_PARENT_TREE
        and declaration['timeout_seconds']==JOB_TIMEOUT_SECONDS==3000
        and declaration['old_timeout_seconds']==2700 and declaration['shards']==15
        and declaration['profile_sha256']==PROFILE_SHA256 and declaration['assignment_sha256']==ASSIGNMENT_SHA256
        and declaration['planning_gate_seconds']==180 and declaration['required_actual_reserve_seconds']==120
        and declaration['ci_authority']is False and declaration['publication_authority']is False,'CI_FIFTY_AUTHORITY_CHANGED')
    before=git(root,'show',SEAL_PARENT+':.github/workflows/ci-fifteen.yml')
    pattern=b'    runs-on: ubuntu-latest\n    timeout-minutes: 45\n'
    require(before.count(pattern)==1 and sha(before)==declaration['before_workflow_sha256']
        and physical(Path(root)/'.github/workflows/ci-fifteen.yml')==before.replace(pattern,pattern.replace(b'45',b'50'),1)
        and declaration['after_workflow_sha256']==WORKFLOW_SHA256,'CI_FIFTY_TRANSITION_CHANGED')
    require(declaration['prior_hashes']=={'docs/paid-launch/launch-scope-policy-ci-fifteen-v10.json': '4b9e03822186309c84d05de0de3a5386206aa3524d8f4a831cf0a62807d77438', 'docs/paid-launch/ci-fifteen-validation-v10.json': 'eedc5a0e4c03c7475f9ee5b71f38aa71a6d2cedbbc4201efba05348b56be4348', 'docs/paid-launch/ci-fifteen-qualification-inputs-v4.json': '1196cf2c10b2f0f3de4435c686da4059f657091646532c53df35e97e4e0c4f92', 'docs/paid-launch/ci-fifteen-retained-profile-input-v3.json': 'b4224f3c138986b7c18b8c44d95d3adc8ec199aba1c99cd764fc8701e7efe3dd', 'docs/paid-launch/nfl-capacity-context-generation-v6.json': 'ddadd1e33062d5a18dba3e6c4db74626f83a63d96139706b5edba675cba7690e'},'CI_FIFTY_PRIOR_DECLARATION_CHANGED')
    for path,digest in declaration['prior_hashes'].items():
        require(sha(physical(Path(root)/path))==digest,'CI_FIFTY_PRIOR_CHANGED')


def verify_historical_workflow_input(root):
    d=json.loads(physical(Path(root)/WORKFLOW_INPUT_DESCRIPTOR))
    require(sha(physical(Path(root)/WORKFLOW_INPUT_DESCRIPTOR))==WORKFLOW_INPUT_DESCRIPTOR_SHA256
        and d['predecessor_head']==CORRECTION_PARENT and d['predecessor_tree']==CORRECTION_PARENT_TREE
        and text(root,'rev-parse',CORRECTION_PARENT+'^{tree}')==CORRECTION_PARENT_TREE
        and d['publication_authority']is False and d['ci_authority']is False,'CI_HISTORICAL_WORKFLOW_DESCRIPTOR_CHANGED')
    require(sha(physical(Path(root)/d['consumer']))==d['consumer_sha256']
        and d['consumer']=='tests/test_ncaaf_postmerge_scope.py'
        and d['selected_call']=='test_workflow_fallback_and_command_are_the_real_unchanged_values'
        and d['generation']=='transport' and d['helper']=='tests/historical_scope_input_boundary.py',
        'CI_HISTORICAL_WORKFLOW_CONSUMER_CHANGED')
    old=git(root,'show',CORRECTION_PARENT+':'+d['helper'])
    anchor=b"    paths=set(binding['component_predecessors'])\n"
    addition=b"    # Only this frozen call needs the already-authenticated paid-workflow edge.\n    # Live/native callers continue reading the current explicit event routing.\n    if (plan.consumer,plan.node,plan.generation)==('tests/test_ncaaf_postmerge_scope.py',\n            'test_workflow_fallback_and_command_are_the_real_unchanged_values','transport'):\n        paths.add('.github/workflows/paid-launch.yml')\n"
    require(old.count(anchor)==1 and sha(old)==d['helper_before_sha256']
        and physical(Path(root)/d['helper'])==old.replace(anchor,anchor+addition,1)
        and sha(physical(Path(root)/d['helper']))==d['helper_after_sha256'],'CI_HISTORICAL_WORKFLOW_INPUT_CHANGED')
    require(physical(Path(root)/PAID_WORKFLOW)==git(root,'show',CORRECTION_PARENT+':'+PAID_WORKFLOW)
        and sha(physical(Path(root)/PAID_WORKFLOW))==d['physical_workflow_sha256'],
        'CI_MAIN_CALLER_TRANSITION_CHANGED')
    for path in ('docs/paid-launch/launch-scope-policy-ci-fifteen-v11.json','docs/paid-launch/ci-fifteen-validation-v11.json'):
        require(physical(Path(root)/path)==git(root,'show',CORRECTION_PARENT+':'+path),'CI_HISTORICAL_WORKFLOW_PRIOR_SEAL_CHANGED')
    return dict(descriptor_sha256=WORKFLOW_INPUT_DESCRIPTOR_SHA256,predecessor_head=CORRECTION_PARENT,
        helper_sha256=d['helper_after_sha256'],consumer_sha256=d['consumer_sha256'])


def verify_refresh_deadline(root):
    import ast
    d=json.loads(physical(Path(root)/REFRESH_DESCRIPTOR))
    require(sha(physical(Path(root)/REFRESH_DESCRIPTOR))==REFRESH_DESCRIPTOR_SHA256
        and d['predecessor_head']==REFRESH_PARENT and d['predecessor_tree']==REFRESH_PARENT_TREE
        and text(root,'rev-parse',REFRESH_PARENT+'^{tree}')==REFRESH_PARENT_TREE,
        'CI_REFRESH_DEADLINE_DESCRIPTOR_CHANGED')
    require(d['scope']=='TEST_REFRESH_CLICK_ONLY_300_SECONDS_LOCAL_QUALIFICATION'
        and (d['startup_seconds'],d['refresh_seconds'],d['static_rerender_seconds'])==(180,300,180)
        and (d['test_identities'],d['shards'],d['job_timeout_seconds'])==(9403,15,3000)
        and (d['planning_gate_seconds'],d['required_actual_reserve_seconds'])==(180,120)
        and all(d[k]is False for k in ('production_deadline_change','model_change','storage_change',
            'profile_change','assignment_change','publication_authority','ci_authority')),
        'CI_REFRESH_DEADLINE_SCOPE_CHANGED')
    expected={'tests/test_ncaaf_private_normal_refresh.py': (b"next(b for b in app.button if b.label=='Refresh picks').click().run()", b"next(b for b in app.button if b.label=='Refresh picks').click().run(timeout=300)"), 'tests/test_ncaaf_fresh_history.py': (b'assertions unchanged, with its approved 180-second harness budget.', b'assertions unchanged; only the Refresh click has the approved 300-second test budget.')}
    require(set(d['source_changes'])==set(expected) and d['consumers']==[{'nodeid': 'tests/test_ncaaf_private_normal_refresh.py::test_actual_normal_refresh_button_uses_private_caller', 'shard': 4}, {'nodeid': 'tests/test_ncaaf_history_continuation.py::test_continuation_actual_normal_refresh_button', 'shard': 7}, {'nodeid': 'tests/test_ncaaf_fresh_history.py::test_fresh_route_through_actual_normal_refresh_button', 'shard': 15}, {'nodeid': 'tests/test_ncaaf_retained_discovery.py::test_retained_history_prospective_discovery_actual_refresh_button', 'shard': 9}],
        'CI_REFRESH_DEADLINE_CONSUMERS_CHANGED')
    count=0
    for path,(before,after)in expected.items():
        prior=git(root,'show',REFRESH_PARENT+':'+path);current=physical(Path(root)/path)
        row=d['source_changes'][path]
        require(prior.count(before)==1 and current==prior.replace(before,after,1)
            and sha(prior)==row['before_sha256'] and sha(current)==row['after_sha256']
            and row['before_text']==before.decode() and row['after_text']==after.decode(),
            'CI_REFRESH_DEADLINE_PROGRAM_CHANGED')
        assertions=lambda raw:[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(raw))if isinstance(n,ast.Assert)]
        old=assertions(prior);new=assertions(current)
        require(old==new and sha((json.dumps(old,sort_keys=True,indent=2)+'\n').encode())==row['original_assertions_sha256'],
            'CI_REFRESH_DEADLINE_ASSERTIONS_CHANGED');count+=len(old)
    helper=physical(Path(root)/'tests/test_ncaaf_private_normal_refresh.py')
    require(helper.count(b'default_timeout=180')==1 and helper.count(b'.click().run(timeout=300)')==1,
        'CI_REFRESH_DEADLINE_ACTION_CHANGED')
    for path in ('tests/historical_scope_input_boundary.py','.github/workflows/paid-launch.yml',
            '.github/workflows/ci-fifteen.yml','scripts/run_ci_tests.py','scripts/ci_test_file_costs_v1.json',
            'tests/test_ncaaf_history_continuation.py','tests/test_ncaaf_retained_discovery.py'):
        require(physical(Path(root)/path)==git(root,'show',REFRESH_PARENT+':'+path),'CI_REFRESH_UNRELATED_SOURCE_CHANGED')
    return dict(descriptor_sha256=REFRESH_DESCRIPTOR_SHA256,predecessor_head=REFRESH_PARENT,
        paths=sorted(expected),refresh_seconds=300,startup_seconds=180,static_rerender_seconds=180,
        preserved_original_assertions=count,operational_authority=False)


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
        profile_sha256=PROFILE_SHA256,assignment_sha256=ASSIGNMENT_SHA256,shards=15,timeout_seconds=JOB_TIMEOUT_SECONDS,
        inventory_sha256=sha(physical(root/INVENTORY)),qualification_inputs_sha256=sha(physical(root/INPUTS)),
        authorization_sha256=components[AUTHORIZATION],planning_gate_seconds=180,required_actual_reserve_seconds=120,
        hosted_authority=False,publication_authority=False)
    contract['main_caller_successor']=main_caller_contract(root)
    contract['historical_workflow_input_successor']=verify_historical_workflow_input(root)
    contract['refresh_click_deadline_successor']=verify_refresh_deadline(root)
    contract['native_lpt_profile_successor']=dict(predecessor_head=CAPACITY_PARENT,predecessor_tree=CAPACITY_PARENT_TREE,descriptor_sha256=PROFILE_DESCRIPTOR_SHA256,context_record_sha256=components[CONTEXT_RECORD],local_authorization_sha256=components[LOCAL_AUTHORIZATION],profile_sha256=PROFILE_SHA256,assignment_sha256=ASSIGNMENT_SHA256)
    contract['retained_profile_consumer_generation']=dict(predecessor_head=LOCAL_PARENT,predecessor_tree=LOCAL_PARENT_TREE,logical_head=CAPACITY_PARENT,logical_binding_sha256='d84983e285490da3e38128ceeec1b9c907632eeea62596e8e3fe3628d29499a1',descriptor_sha256=components[PROFILE_CONSUMER],helper_sha256=components['tests/native_lpt_profile_inputs.py'],conftest_sha256=sha(physical(root/'tests/conftest.py')),consumer_sha256=components['tests/test_ncaaf_sidebar_startup_scope.py'])
    return contract


def validate_topology(root,binding):
    runner=_source_module(root,'scripts/run_ci_tests.py','6be44f5bdf5934dea5784cd1819595e9c8386513fffa0769854e95d4095302fe')
    root=Path(root);metadata=binding['nfl_account_reset_successor']['ci_fifteen_contract']
    require(metadata.get('main_caller_successor')==main_caller_contract(root),'CI_MAIN_CALLER_CONTRACT_CHANGED')
    require(metadata['shards']==15 and metadata['timeout_seconds']==JOB_TIMEOUT_SECONDS
        and metadata['planning_gate_seconds']==180 and metadata['required_actual_reserve_seconds']==120
        and metadata['hosted_authority']is False and metadata['publication_authority']is False,'CI_FIFTEEN_SCOPE_CHANGED')
    require(sha(physical(root/'.github/workflows/ci-fifteen.yml'))==WORKFLOW_SHA256,'CI_FIFTEEN_WORKFLOW_CHANGED')
    require(physical(root/'.github/workflows/ci.yml')==git(root,'show',PARENT+':.github/workflows/ci.yml'),'CI_THIRTEEN_HISTORICAL_WORKFLOW_CHANGED')
    previous=git(root,'show',PARENT+':.github/workflows/ci.yml').decode()
    expected=previous.replace('shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]',
        'shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]').replace('--shards 13','--shards 15').replace(
        'Reconcile complete test identities and outcomes across all nine shards',
        'Reconcile complete test identities and outcomes across all fifteen shards')
    require(expected.count('    runs-on: ubuntu-latest\n    timeout-minutes: 45\n')==1,'CI_FIFTY_PARENT_TIMEOUT_CHANGED')
    expected=expected.replace('    runs-on: ubuntu-latest\n    timeout-minutes: 45\n','    runs-on: ubuntu-latest\n    timeout-minutes: 50\n',1)
    verify_timeout(root)
    # Only the owner-approved full-suite timeout changes;other jobs and gates remain exact.
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
    inputs=json.loads(physical(root/INPUTS));reserves=[JOB_TIMEOUT_SECONDS-JOB_OVERHEAD_SECONDS-v for v in loads]
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
    return dict(version='ci-fifteen-refresh-deadline-local-only-seal-v13',base=scope.BASE,base_tree=scope.TREE,
        predecessor_head=PARENT,predecessor_tree=PARENT_TREE,predecessor_binding_sha256=PARENT_BINDING,
        retained_thirteen_policy_sha256=sha(physical(Path(root)/OLD_POLICY)),
        binding_sha256=binding_sha,implementation_commit=implementation,
        implementation_tree=evidence['implementation_tree'],implementation_parent=REFRESH_PARENT,implementation_paths=sorted(scope.PATHS),
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
    require(text(root,'show','-s','--format=%P',implementation).split()==[REFRESH_PARENT]
        and text(root,'merge-base',scope.BASE,REFRESH_PARENT)==scope.BASE,'IMPLEMENTATION_PARENT_NOT_APPROVED')
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
        protected_changes=[],existing_test_changes=[],approved_test_deadline_changes=verify_refresh_deadline(root),accepted=False,inference=False,
        ci_authority=False,publication_authority=False,qualification_status=policy['validation_status'],**policy['topology'])
