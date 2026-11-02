"""Exact date-only successor and authenticated timing qualification; no operations."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib
import io
import types

BASE='9fb69bf62c59f595a05478f2fc9cc2ed489840b9'
TREE='3edaa30fca0853d61d5da8879900b7724af93471'
OLD_STAMP='b5b240bfd806da5d5e786ecdd1ec6679b9d8677267e026d87516b7a373cf26ba'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION='docs/paid-launch/nfl-account-reset-precision-v1.md'
POLICY='docs/paid-launch/launch-scope-policy-ci-fifteen-v1.json'
PROFILE='scripts/ci_test_file_costs_v1.json'
TIMING_SOURCE='nfl-account-reset-local-estimate-20261101-v1'
TIMING_SOURCE_SHA256='fce60bfa3f62a61319a053afc3f4d336780b4c73ef2c3d063bf04766ea8e844b'
RETAINED_TIMING_SOURCE='nfl-account-reset-retained-phase-estimate-20261101-v2'
RETAINED_TIMING_SOURCE_SHA256='b4478fd689b2ecdd7e201e100af3b2cd4bb7d385ad9680dfa2fce1bee29aac90'
RETAINED_TESTS={'tests/test_ncaaf_schedule_fixture_clock.py': 11, 'tests/test_nfl_tls_trust.py': 26}
PRESERVED_HEAD='1d1742cb4e8ca7a7e5c3ef3e33a17874d3cf28ff'
PRESERVED_TREE='7849626e66b79cd581ac7581d8ecce837fea444c'
PRESERVED_PROFILE='0e9f70985b0d38d2ed31fe52e7563f3abc92f6a6577349b0beb2669aa23ae2dd'
FAILED_RELEASE_HEAD='b9a813824bec02d736cfb5fb5ca7131aa76b835e'
FAILED_RELEASE_TREE='3605333323b602317092bbafc5be94904f8e532b'
NEW_TESTS={'tests/test_nfl_account_reset.py':41,'tests/test_nfl_account_reset_scope.py':5,'tests/test_nfl_account_reset_ui.py':4}
NORMALIZED={'scripts/check_launch_change_scope.py','scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py'}
PATHS={BINDING,AUTHORIZATION,PROFILE,'app_core/nfl_private_workflow.py','app/ui/nfl_private_workflow.py',
       'tests/test_nfl_account_reset.py','tests/test_nfl_account_reset_ui.py',
       'tests/test_nfl_account_reset_scope.py',
       'scripts/nfl_account_reset_scope.py','scripts/ncaaf_sidebar_startup_scope.py',
       'tests/nfl_profile_generation_inputs.py','tests/historical_scope_input_boundary.py',
       'tests/historical_setup_compile.py','tests/test_historical_setup_compile.py',
       'tests/test_historical_program_compile.py','tests/test_authenticated_reader_entry.py',*NORMALIZED}
PATHS=PATHS|{'.github/workflows/ci-fifteen.yml','scripts/ci_fifteen_scope.py',
    'docs/paid-launch/launch-scope-policy-nfl-account-reset-v1.json',
    'docs/paid-launch/ci-fifteen-qualification-inputs-v1.json',
    'docs/paid-launch/ci-fifteen-inventory-v1.json',
    'docs/paid-launch/ci-fifteen-local-qualification-v1.md'}
LIMIT=8*1024*1024
QUALIFIED_SOURCES={'.github/workflows/ci-fifteen.yml': '8e44af98864d6bd86f883316b18be3a2a232f2c5223d6c4414ffda3bfa4487a1', 'app/ui/nfl_private_workflow.py': '6daf4985272a084e4fff8b3c9f91b1bd84551546ca116955f8d5b059a3622116', 'app_core/nfl_private_workflow.py': 'ca435b0739ae1b7a2e769144b5c1c3036187797c34f576095a670f65595d0163', 'docs/paid-launch/ci-fifteen-inventory-v1.json': '1956765dc42be7cc0d0f5e9ceeef379eec526e948971299deb09f0e7ac7227d5', 'docs/paid-launch/ci-fifteen-local-qualification-v1.md': '855dd2af4455cf02c435ed656c03c874f9f37e0da5b0f3d25064dd2e33d896eb', 'docs/paid-launch/ci-fifteen-qualification-inputs-v1.json': '63254807c43ddf869c6b35b636b52294c2a2c3e7f469dcab4edcdccde4451de1', 'docs/paid-launch/launch-scope-policy-nfl-account-reset-v1.json': '2604ac29099af88224dda7e205bbbc46e4cfe2483cf157e09f1c1df342650f89', 'docs/paid-launch/nfl-account-reset-precision-v1.md': '91a76963f5f7cdd2b499c15bf7a20bcff94cd222654ff2d6d3840e6d213d3331', 'scripts/check_launch_change_scope.py': '2fd81f35197ba453e8ed8aae2312ccdc88a5be28bfe786246ab22057422cb152', 'scripts/ci_fifteen_scope.py': '2e60ea5dbf4c495e497fbff3810be63f2429a9eb65ca994789326fad5c9bae2f', 'scripts/ci_test_file_costs_v1.json': '5513e877c0e467bf8217dce1b041453e942d4610af1c5734e72519cb197ea347', 'scripts/ncaaf_filtered_history_scope.py': '6d75005330e4b87b00e7c8535b92f25ba622c2d78c7a98c84932c9075ccafb86', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/ncaaf_sidebar_startup_scope.py': '9a75224dc65c29d8302c81c163af1dc4139a358053d6fe75ed38567f11f7f66a', 'tests/historical_scope_input_boundary.py': 'f91678b7d224ae1759ce619657ce07d26f4915813372b19062a0bccd2e2597dc', 'tests/historical_setup_compile.py': '106a68f23e593b40d5289f6ee1539c643e853ec7d3f548254c414e28504c8e00', 'tests/nfl_profile_generation_inputs.py': '2998f8088f4ca4abb98904e3f286ab53ebc05a8019408ffca9b4fa33843dce47', 'tests/test_authenticated_reader_entry.py': '77adc81cc24417a526203ecbd9d057a316380fdefdd1088da6e06f631875ed78', 'tests/test_historical_program_compile.py': '4d9e9373e10ddaf21f8f3094252bf08b628575061ef726f015e39a3254849a38', 'tests/test_historical_setup_compile.py': 'edc85a8eedf7c24b9908547f88ee96371e68f7695f6fb7efac8d07ea48d59493', 'tests/test_nfl_account_reset.py': 'a94fa1db078479db5c62ed8136ee46edab912d631b0a9a41babea4c86a77224a', 'tests/test_nfl_account_reset_scope.py': 'b1ec553aebdf1eb39432a752e72ff5b2e76da8a38d065e7aa7f5f2509f63c0e7', 'tests/test_nfl_account_reset_ui.py': 'bd33aae1e15dfecbf8ebd85aa696bd5f0fb32c271166f6127ce11f889e7cde17'}


READER_ENTRY_TRANSITIONS={'scripts/check_launch_change_scope.py': [('def _filtered_history_module():', 'def _filtered_history_module(*, read=None):'), ("exec(compile(source,module.__file__,'exec'),module.__dict__);module.authenticate(ROOT)", "exec(compile(source,module.__file__,'exec'),module.__dict__);current_binding=module.authenticate(ROOT)"), ('    module._NATIVE_DECOMPRESS=_FILTERED_HISTORY_NATIVE_DECOMPRESS\n    return module', "    module._NATIVE_DECOMPRESS=_FILTERED_HISTORY_NATIVE_DECOMPRESS\n    if read is not None:\n        if type(read) is not tuple or len(read)!=2:raise ValueError('NCAAF_FILTERED_READER_ARGUMENT_CHANGED')\n        return module._predecessor_source(current_binding,*read)\n    return module")], 'scripts/ncaaf_filtered_history_scope.py': [("    b=authenticate(root);raw=capacity_predecessor(b,relative,raw);raw=diagnostic_predecessor(b,relative,raw);record=b['component_predecessors'].get(relative)", '    return _predecessor_source(authenticate(root),relative,raw)\n\n\ndef _predecessor_source(b,relative,raw):\n    """Decode only within the entry that just authenticated this binding."""\n    raw=capacity_predecessor(b,relative,raw);raw=diagnostic_predecessor(b,relative,raw);record=b[\'component_predecessors\'].get(relative)')]}

READER_ENTRY_TRANSITIONS['scripts/check_launch_change_scope.py'].append((r"binding_raw=stream.read(8*1024*1024+1)",r"binding_raw=stream.read(8*1024*1024+1).replace(b'\r\n',b'\n')"))


def _fifteen_scope():
    """Fresh authenticated helper; standalone commands need no package import."""
    root=Path(__file__).resolve().parents[1];relative='scripts/ci_fifteen_scope.py'
    with io.FileIO(root/relative,'r')as f:raw=f.read(LIMIT+1).replace(b'\r\n',b'\n')
    require(len(raw)<=LIMIT and sha(raw)==QUALIFIED_SOURCES[relative],'CI_FIFTEEN_HELPER_CHANGED')
    module=types.ModuleType('exact_ci_fifteen_scope');module.__file__=str(root/relative)
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    return module


def reader_entry_parent(path,raw):
    for before,after in READER_ENTRY_TRANSITIONS.get(path,()):
        require(raw.count(after.encode())==1,'INHERITED_READER_ENTRY_CHANGED')
        raw=raw.replace(after.encode(),before.encode(),1)
    return raw


def require(value,code):
    if not value:raise ValueError(code)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def text(root,*args):return git(root,*args).decode().strip()


def record(raw):
    return dict(bytes=len(raw),sha256=sha(raw),compressed_b85=base64.b85encode(zlib.compress(raw,9)).decode())


def record_bytes(value):
    z=zlib.decompressobj();raw=z.decompress(base64.b85decode(value['compressed_b85']),LIMIT+1)
    require(z.eof and not z.unused_data and not z.unconsumed_tail and len(raw)==value['bytes']<=LIMIT
            and sha(raw)==value['sha256'],'NFL_ACCOUNT_PARENT_CHANGED')
    return raw


def normalized(path,raw,stamp):
    raw=raw.replace(b'\r\n',b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode())==1,'NFL_ACCOUNT_STAMP_CHANGED')
        raw=raw.replace(stamp.encode(),b'0'*64)
    return raw


def blobs(root,revision):
    return {p:m.split()[2] for line in text(root,'ls-tree','-r',revision).splitlines() for m,p in [line.split('\t',1)]}


def parent_binding(binding):
    m=binding['nfl_account_reset_successor'];parent=json.loads(json.dumps(binding))
    require(m['version']==1 and m['base']==BASE and m['tree']==TREE and m['paths']==sorted(PATHS),'NFL_ACCOUNT_BINDING_CHANGED')
    del parent['nfl_account_reset_successor']
    require(set(m['components'])==set(m['parent_components']) and set(m['capacity_edges'])==set(m['parent_capacity_after']), 'NFL_ACCOUNT_BINDING_CHANGED')
    for path,value in m['components'].items():
        require(parent['components'].get(path)==value,'NFL_ACCOUNT_BINDING_CHANGED')
        previous=m['parent_components'][path]
        if previous is None:parent['components'].pop(path)
        else:parent['components'][path]=previous
    for path,edge in m['capacity_edges'].items():
        require(parent['nfl_capacity_predecessors'].get(path)==edge,'NFL_ACCOUNT_BINDING_CHANGED')
        previous=m['parent_capacity_after'][path]
        if previous is None:parent['nfl_capacity_predecessors'].pop(path)
        else:parent['nfl_capacity_predecessors'][path]['after_sha256']=previous
    raw=canonical(parent)
    require(len(raw)==m['parent_binding']['bytes']<=LIMIT and sha(raw)==m['parent_binding']['sha256']==OLD_STAMP,'NFL_ACCOUNT_PARENT_CHANGED')
    return raw


def binding_successor(root,stamp):
    root=Path(root);before=git(root,'show',BASE+':'+BINDING)
    require(sha(before)==OLD_STAMP,'NFL_ACCOUNT_BASE_CHANGED')
    verify_profile(root)
    b=json.loads(before);prior_components=dict(b['components']);prior_edges=dict(b['nfl_capacity_predecessors'])
    components={};edges={};historical={};existing=blobs(root,BASE)
    for path in PATHS-{BINDING}:
        value=sha(normalized(path,(root/path).read_bytes(),stamp));components[path]=value
        if path in NORMALIZED and value!=prior_components[path]:
            # Only the exact approved reader entry differs; preserve the entire
            # old capacity preimage and its nested successor chain.
            original=normalized(path,git(root,'show',BASE+':'+path),OLD_STAMP)
            require(reader_entry_parent(path,normalized(path,(root/path).read_bytes(),stamp))==original,
                    'INHERITED_READER_ENTRY_CHANGED')
            edges[path]=dict(prior_edges[path],after_sha256=value)
        if path in existing and path not in NORMALIZED:
            old=git(root,'show',BASE+':'+path);historical[path]=record(old)
            prior=b['nfl_capacity_predecessors'].get(path)
            # Preserve nested predecessors; the new layer restores the exact
            # last released bytes before older consumers peel their own layer.
            if prior is None:
                edges[path]=dict(before_bytes=len(old),before_sha256=sha(old),before_b85=base64.b85encode(zlib.compress(old,9)).decode(),after_sha256=value)
            else:
                edges[path]=dict(prior,after_sha256=value)
        b['components'][path]=value
    b['nfl_capacity_predecessors'].update(edges)
    b['nfl_account_reset_successor']=dict(version=1,base=BASE,tree=TREE,parent_binding=dict(bytes=len(before),sha256=sha(before)),
        components=components,capacity_edges=edges,historical_sources=historical,paths=sorted(PATHS),
        parent_components={p:prior_components.get(p) for p in components},
        parent_capacity_after={p:prior_edges[p]['after_sha256'] if p in prior_edges else None for p in edges},
        authorization_sha256=components[AUTHORIZATION])
    b['nfl_account_reset_successor']['ci_fifteen_contract']=_fifteen_scope().make_contract(root,stamp,components)
    require(parent_binding(b)==before,'NFL_ACCOUNT_PARENT_CHANGED')
    return b


def verify_profile(root):
    return _fifteen_scope().verify_profile(root)


def verify_thirteen_profile(root):
    before=json.loads(git(root,'show',BASE+':'+PROFILE));after=json.loads((Path(root)/PROFILE).read_bytes())
    retained=after['sources'].get(RETAINED_TIMING_SOURCE)
    require(retained is not None and sha(canonical(retained))==RETAINED_TIMING_SOURCE_SHA256
            and retained['measured'] is False and retained['variability_multiplier']==1.15
            and retained['introduced_shared_fixture_phase_allowance_seconds']==5.0,
            'NFL_ACCOUNT_RETAINED_TIMING_SOURCE_CHANGED')
    require(len(after['files'])>=2 and {r['path'] for r in after['files'][-2:]}==RETAINED_TESTS.keys(),
            'NFL_ACCOUNT_RETAINED_TIMING_FILES_CHANGED')
    for row in after['files'][-2:]:
        observed=retained['files'][row['path']]
        maximum=max(observed['completed_linux_phase_seconds'],observed['completed_windows_phase_seconds'])
        require(observed['complete_file_phase_max_seconds']==maximum
                and row['profiled_tests']==RETAINED_TESTS[row['path']] and row['kind']=='estimated'
                and row['references']==[RETAINED_TIMING_SOURCE] and row['seconds']==round((maximum+5.0)*1.15,6)
                and row['retained_phase_estimate_basis']==dict(classification='estimate',
                    complete_file_phase_max_seconds=maximum,introduced_shared_fixture_phase_allowance=5.0,
                    variability_multiplier=1.15),'NFL_ACCOUNT_RETAINED_ESTIMATE_CHANGED')
    after=json.loads(json.dumps(after));after['files']=after['files'][:-2];after['sources'].pop(RETAINED_TIMING_SOURCE)
    require(sha((json.dumps(after,separators=(',',':'))+'\n').encode())==PRESERVED_PROFILE,
            'NFL_ACCOUNT_PRESERVED_PROFILE_CHANGED')
    source=after['sources'][TIMING_SOURCE]
    require(sha(canonical(source))==TIMING_SOURCE_SHA256 and source['measured'] is False
            and source['kind']=='estimated_new_file_phase_budget' and source['variability_multiplier']==1.15
            and source['introduced_shared_fixture_phase_allowance_seconds']==5.0,'NFL_ACCOUNT_TIMING_SOURCE_CHANGED')
    require(after['sources'].keys()==before['sources'].keys()|{TIMING_SOURCE}
            and after['files'][:len(before['files'])]==before['files'],'NFL_ACCOUNT_PRIOR_COST_CHANGED')
    new=after['files'][len(before['files']):]
    require(len(new)==len(NEW_TESTS) and {r['path'] for r in new}==NEW_TESTS.keys(),'NFL_ACCOUNT_TIMING_FILES_CHANGED')
    for row in new:
        seconds=source['supporting_completed_windows_phase_seconds'][row['path']]
        require(row['kind']=='estimated' and row['references']==[TIMING_SOURCE]
                and row['profiled_tests']==NEW_TESTS[row['path']] and row['seconds']==round((seconds+5.0)*1.15,6)
                and row['account_reset_estimate_basis']==dict(classification='estimate',completed_windows_phase_seconds=seconds,
                    introduced_shared_fixture_phase_allowance=5.0,variability_multiplier=1.15),'NFL_ACCOUNT_ESTIMATE_CHANGED')
    restored=json.loads(json.dumps(after));restored['files']=before['files'];restored['sources'].pop(TIMING_SOURCE)
    require(restored==before,'NFL_ACCOUNT_PRIOR_COST_CHANGED')


def make_policy(root,binding,implementation,binding_sha):
    if binding['nfl_account_reset_successor'].get('ci_fifteen_contract')is not None:
        return _fifteen_scope().make_policy(root,binding,implementation,binding_sha)
    return dict(version='nfl-account-reset-authenticated-reader-local-seal-v8',base=BASE,base_tree=TREE,binding_sha256=binding_sha,
        failed_release_head=FAILED_RELEASE_HEAD,failed_release_tree=FAILED_RELEASE_TREE,
        preserved_candidate=PRESERVED_HEAD,preserved_tree=PRESERVED_TREE,preserved_profile_sha256=PRESERVED_PROFILE,
        retained_timing_source_sha256=RETAINED_TIMING_SOURCE_SHA256,
        predecessor_policy_sha256=sha(git(root,'show',BASE+':docs/paid-launch/launch-scope-policy-sidebar-startup-v1.json')),
        authorization_sha256=binding['components'][AUTHORIZATION],implementation_commit=implementation,
        implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS},
        operational_authority=False,publication_authority=True,ci_authority=True,maximum_ci_cycles=1,
        release_condition="Owner-authorized one corrected-head cycle remains conditional on all local and capacity gates; no retries. Preserve all failed heads and measurements.")


def validate(root,binding,comparison_base,manifest,*,binding_sha):
    if binding['nfl_account_reset_successor'].get('ci_fifteen_contract')is not None:
        return _fifteen_scope().validate(root,binding,comparison_base,manifest,binding_sha=binding_sha)
    root=Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE,'NFL_ACCOUNT_BASE_CHANGED')
    require(comparison_base in (None,BASE),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'status','--porcelain'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==binding_successor(root,binding_sha),'NFL_ACCOUNT_BINDING_CHANGED')
    require(QUALIFIED_SOURCES and all(sha(normalized(p,(root/p).read_bytes(),binding_sha))==value
            for p,value in QUALIFIED_SOURCES.items()),'NFL_ACCOUNT_SOURCE_NOT_QUALIFIED')
    for path in NORMALIZED:
        require(normalized(path,git(root,'show',BASE+':'+path),OLD_STAMP)==reader_entry_parent(path,normalized(path,(root/path).read_bytes(),binding_sha)),'INHERITED_GUARD_LOGIC_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS,'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');seal=head;parents=text(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
            and text(root,'diff','--name-status',implementation,seal)=='A\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=blobs(root,'HEAD');require(actual.keys()==blobs(root,BASE).keys()|PATHS|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for path,oid in actual.items():
        raw=(root/path).read_bytes();blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(oid in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=108,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(PATHS|{POLICY}),protected_changes=[],
        existing_test_changes=[],accepted=False,inference=False,ci_authority=True,publication_authority=True)
