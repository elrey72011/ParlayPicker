"""Exact test-browser lifecycle successor preserving both failed capacity heads."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = 'f6a7506dc3300fbcf390dbcdd00f0688dba073a4'
TREE = '5bc678e613ec381c07f67110ee5f568f8961ff2f'
PARENTS = ['d55706bfdd9cb182f121bcb352fd9f7a2a30f436', '0a19b3bcd02b87c12adba0bf3681fbe8996fb640']
OLD_STAMP = '24d9bc417f9cb323f35395191d394fd6423b72ac924b7040b4bbf45f92a813db'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-capacity-profile-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-capacity-profile-v1.md'
LEDGER = 'docs/paid-launch/nfl-capacity-timing-ledger-v1.json'
CORRECTION = 'docs/paid-launch/nfl-capacity-fixture-correction-v1.md'
FAILED_HEAD = 'c15f4f4bf14be37ce62dbceebc92eb8dd197f373'
FAILED_TREE = '0c36b9cd64aadde41b31b309e23b7f06b723a769'
FAILED_IMPLEMENTATION = 'c82fbe614449a0a3195244eb83bcf99aff23352b'
PROFILE = 'scripts/ci_test_file_costs_v1.json'
BROWSER_AUTHORIZATION = 'docs/paid-launch/browser-harness-ownership-v1.md'
PREVIOUS_HEAD = '803d678bf69cbd7de1393080285ac4e5ac692ca8'
PREVIOUS_TREE = '2b8d788d36154dfceb231e105843b45f0dde0cce'
PREVIOUS_IMPLEMENTATION = 'c39b3ff6c5e9ddb4a03f7f3fecb4ebf4dfc2b409'
BROWSER_HELPER = 'tests/test_research_probability_browser.py'
BROWSER_PATHS = {BROWSER_HELPER, BROWSER_AUTHORIZATION, 'tests/browser_process_owner.py',
                 'tests/browser_process_windows.py', 'tests/test_browser_process_owner.py'}
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py', 'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, LEDGER, PROFILE, *NORMALIZED,
         'scripts/nfl_capacity_profile_scope.py', 'tests/nfl_capacity_profile_inputs.py',
         'tests/test_nfl_capacity_profile.py', 'tests/test_ncaaf_filtered_history_scope.py',
         CORRECTION, 'tests/nfl_capacity_import_owner.py', 'tests/test_nfl_capacity_fixture_isolation.py'} | BROWSER_PATHS
MAX_SOURCE = 8 * 1024 * 1024

HISTORY_ADDITION = '''_CAPACITY_IMAGES={} # Reauthenticated immutable bytes, never an acceptance cache.


def capacity_predecessor(binding,relative,raw):
    record=binding.get('nfl_capacity_predecessors',{}).get(relative)
    if record is None:return raw
    supplied=raw.replace(b'\\r\\n',b'\\n')
    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:return raw
    if sha(normalized(relative,supplied))!=record['after_sha256']:return raw
    key=(_BINDING_SHA256,relative,record['after_sha256'],record['before_sha256'],record['before_bytes'],record['before_b85'])
    old=_CAPACITY_IMAGES.get(key)
    if old is None:
        z=zlib.decompressobj();old=z.decompress(base64.b85decode(record['before_b85']),MAX_SOURCE+1)
        require(z.eof and not z.unused_data,'NFL_CAPACITY_PREDECESSOR_CHANGED')
        while _CAPACITY_IMAGES and (len(_CAPACITY_IMAGES)>=16 or sum(map(len,_CAPACITY_IMAGES.values()))+len(old)>_LEGACY_IMAGE_LIMIT):
            _CAPACITY_IMAGES.pop(next(iter(_CAPACITY_IMAGES)))
        if len(old)<=_LEGACY_IMAGE_LIMIT:_CAPACITY_IMAGES[key]=old
    require(type(old)is bytes and len(old)==record['before_bytes']<=MAX_SOURCE
        and sha(old)==record['before_sha256'],'NFL_CAPACITY_PREDECESSOR_CHANGED')
    return old


def capacity_parent_binding(binding):
    record=binding['nfl_capacity_parent_binding']
    components=(json.dumps(record['before_components'],sort_keys=True,separators=(',',':'))).encode()
    key=(_BINDING_SHA256,'parent-binding',record['before_sha256'],record['before_bytes'],sha(components))
    raw=_CAPACITY_IMAGES.get(key)
    if raw is None:
        prior=json.loads(json.dumps(binding))
        for field in ('nfl_capacity_profile','nfl_capacity_parent_binding','nfl_capacity_predecessors'):prior.pop(field,None)
        prior['components']=record['before_components']
        raw=(json.dumps(prior,sort_keys=True,indent=2,allow_nan=False)+'\\n').encode()
        while _CAPACITY_IMAGES and (len(_CAPACITY_IMAGES)>=16 or sum(map(len,_CAPACITY_IMAGES.values()))+len(raw)>_LEGACY_IMAGE_LIMIT):
            _CAPACITY_IMAGES.pop(next(iter(_CAPACITY_IMAGES)))
        if len(raw)<=_LEGACY_IMAGE_LIMIT:_CAPACITY_IMAGES[key]=raw
    require(type(raw)is bytes and len(raw)==record['before_bytes']<=MAX_SOURCE
        and sha(raw)==record['before_sha256']=='24d9bc417f9cb323f35395191d394fd6423b72ac924b7040b4bbf45f92a813db',
        'NFL_CAPACITY_PREDECESSOR_CHANGED')
    return raw


def _capacity_scope_module(root,binding):
    relative='scripts/nfl_capacity_profile_scope.py';source=physical(Path(root)/relative)
    require(sha(source)==binding['components'][relative],'NFL_CAPACITY_SCOPE_CHANGED')
    module=types.ModuleType('exact_nfl_capacity_profile_scope');module.__file__=str(Path(root)/relative)
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


'''.encode()


def require(ok, code):
    if not ok: raise ValueError(code)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def git(root, *args): return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args): return git(root, *args).decode().strip()


def blobs(root, revision):
    return {p:m.split()[2] for line in text(root,'ls-tree','-r',revision).splitlines() for m,p in [line.split('\t',1)]}


def canonical(value): return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def fixture_cost():
    return dict(path='tests/test_nfl_capacity_fixture_isolation.py',kind='estimated',
        profiled_tests=9,seconds=172.5,references=['2446-windows-owner-regressions'],completed_measurements=[],
        estimate_basis='New lifecycle regressions: Windows 9-case complete run 85.19s, not Linux execution evidence. Conservative 150s allowance times unchanged1.15. Existing rows and source records unchanged.')


def fixture_source():
    return dict(kind='completed_offline_junit',platform='Windows',tests=9,elapsed_seconds=85.19,
        checkout_commit=FAILED_HEAD,checkout_worktree_clean=False,checkout_tree_verified=False,
        junit_sha256='d8d15f349872a18a77240d22fea456be23a05dbf9021ccc20d13aa61e1d8be58',
        qualification='Modified isolated fixture draft at the recorded checkout HEAD; not a clean tested tree or Linux final-head timing. Scheduling cost is a conservative separate estimate.')


def browser_cost():
    return dict(path='tests/test_browser_process_owner.py',kind='estimated',profiled_tests=12,seconds=57.5,
        references=['2446-browser-owner-controls'],completed_measurements=[],
        estimate_basis='Test-only native lifecycle controls. Conservative 50s allowance times unchanged1.15; Windows nine-control draft run2.26s is not Linux timing or an exact final-file measurement.')


def browser_source():
    return dict(kind='completed_offline_junit',platform='Windows',tests=9,elapsed_seconds=2.26,
        checkout_commit=PREVIOUS_HEAD,checkout_worktree_clean=False,checkout_tree_verified=False,
        junit_sha256='d337ec3fa88c61d2edb3477a69c095209f2d2ce014ad262c7bac3641d7c5fe2d',
        qualification='Nine controls in modified isolated draft; not clean-tree/Linux final-head timing. Scheduling uses a separate conservative estimate.')


def normalized(path, raw, stamp):
    raw=raw.replace(b'\r\n',b'\n')
    if path in NORMALIZED:
        name='_BINDING_SHA256' if path.endswith('/ncaaf_filtered_history_scope.py') else '_FILTERED_HISTORY_BINDING_SHA256'
        marker=(name+"='"+stamp+"'").encode()
        require(raw.count(marker)==1,'NFL_CAPACITY_STAMP_CHANGED')
        raw=raw.replace(marker,marker.replace(stamp.encode(),b'0'*64),1)
    return raw


def history_successor(old):
    """Exact additions only; all older source generations and checks survive."""
    marker=b'def diagnostic_predecessor(binding,relative,raw):\n'
    require(old.count(marker)==1,'INHERITED_GUARD_LOGIC_CHANGED')
    new=old.replace(marker,HISTORY_ADDITION+marker,1)
    line=b"    record=binding.get('nfl_storage_diagnostic_predecessors',{}).get(relative)\n"
    new=new.replace(line,b"    raw=capacity_predecessor(binding,relative,raw)\n"+line,1)
    cond=b"    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:return raw\n"
    start=new.index(marker);head,tail=new[:start],new[start:]
    require(tail.count(cond)==1,'INHERITED_GUARD_LOGIC_CHANGED')
    tail=tail.replace(cond,b"    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:\n        prior=binding.get('nfl_capacity_profile',{}).get('prior_binding_sha256','').encode()\n        if not prior or supplied.count(prior)!=1:return raw\n        supplied=supplied.replace(prior,_BINDING_SHA256.encode())\n",1)
    new=head+tail
    # Physical graph authentication is unchanged. Decode/check a new immutable
    # predecessor only when its reader requests it; scope validation below
    # independently verifies every before-image against the exact base.
    marker=b"    b=authenticate(root);raw=diagnostic_predecessor(b,relative,raw);record=b['component_predecessors'].get(relative)\n"
    require(new.count(marker)==1,'INHERITED_GUARD_LOGIC_CHANGED')
    new=new.replace(marker,b"    b=authenticate(root);raw=capacity_predecessor(b,relative,raw);raw=diagnostic_predecessor(b,relative,raw);record=b['component_predecessors'].get(relative)\n",1)
    marker=b"    require(sha(current)==b['components'][path],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')\n"
    require(new.count(marker)==2,'INHERITED_GUARD_LOGIC_CHANGED')
    new=new.replace(marker,marker+b"    current=capacity_predecessor(b,path,current)\n")
    marker=b'def make_policy(root,b,implementation):\n'
    new=new.replace(marker,marker+b"    if b.get('nfl_capacity_profile') is not None:\n        return _capacity_scope_module(root,b).make_policy(root,b,implementation,_BINDING_SHA256)\n",1)
    marker=b"    root=Path(root);b=authenticate(root);repair=b.get('ui_display_repair')\n"
    new=new.replace(marker,marker+b"    if b.get('nfl_capacity_profile') is not None:\n        return _capacity_scope_module(root,b).validate(root,b,comparison_base,manifest,binding_sha=_BINDING_SHA256,preserved_test_report=_preserved_test_report)\n",1)
    return new


def make_policy(root,binding,implementation,binding_sha):
    c=binding['nfl_capacity_profile']
    return dict(version='nfl-browser-harness-seal-v3',base=BASE,base_tree=TREE,
        previous_head=PREVIOUS_HEAD,previous_tree=PREVIOUS_TREE,browser_authorization_sha256=c['browser_authorization_sha256'],
        failed_head=FAILED_HEAD,failed_tree=FAILED_TREE,correction_sha256=c['correction_sha256'],
        authorization_sha256=c['authorization_sha256'],binding_sha256=binding_sha,
        implementation_commit=implementation,implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),
        implementation_paths=sorted(PATHS),unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})


def validate(root,binding,comparison_base,manifest,*,binding_sha,preserved_test_report):
    root=Path(root);c=binding['nfl_capacity_profile']
    require(c==dict(base=BASE,base_tree=TREE,ordered_parents=PARENTS,prior_binding_sha256=OLD_STAMP,
        authorization_sha256=binding['components'][AUTHORIZATION],ledger_sha256=binding['components'][LEDGER],implementation_paths=sorted(PATHS),
        correction_sha256=binding['components'][CORRECTION],failed_head=FAILED_HEAD,failed_tree=FAILED_TREE,
        browser_authorization_sha256=binding['components'][BROWSER_AUTHORIZATION],previous_head=PREVIOUS_HEAD,previous_tree=PREVIOUS_TREE),'NFL_CAPACITY_SCOPE_CHANGED')
    require(text(root,'rev-parse',FAILED_HEAD+'^{tree}')==FAILED_TREE
        and text(root,'show','-s','--format=%P',FAILED_HEAD).split()==[FAILED_IMPLEMENTATION]
        and text(root,'show','-s','--format=%P',FAILED_IMPLEMENTATION).split()==[BASE], 'NFL_CAPACITY_FAILED_EVIDENCE_CHANGED')
    require(text(root,'rev-parse',PREVIOUS_HEAD+'^{tree}')==PREVIOUS_TREE
        and text(root,'show','-s','--format=%P',PREVIOUS_HEAD).split()==[PREVIOUS_IMPLEMENTATION]
        and text(root,'show','-s','--format=%P',PREVIOUS_IMPLEMENTATION).split()==[FAILED_HEAD], 'BROWSER_FAILED_EVIDENCE_CHANGED')
    for path in (LEDGER,AUTHORIZATION,CORRECTION,'tests/test_nfl_capacity_profile.py',
                 'tests/test_ncaaf_filtered_history_scope.py','tests/nfl_capacity_profile_inputs.py',
                 'tests/nfl_capacity_import_owner.py','tests/test_nfl_capacity_fixture_isolation.py'):
        require(git(root,'show',PREVIOUS_HEAD+':'+path)==(root/path).read_bytes(),'BROWSER_PRIOR_FIXTURE_CHANGED')
    for path in (LEDGER,AUTHORIZATION,'tests/test_nfl_capacity_profile.py'):
        require(git(root,'show',FAILED_HEAD+':'+path)==(root/path).read_bytes(),'NFL_CAPACITY_RETAINED_INPUT_CHANGED')
    old_profile=json.loads(git(root,'show',PREVIOUS_HEAD+':'+PROFILE));new_profile=json.loads((root/PROFILE).read_bytes())
    require({k:v for k,v in new_profile.items() if k not in ('files','sources')}=={k:v for k,v in old_profile.items() if k not in ('files','sources')}
        and new_profile['sources']==dict(old_profile['sources'],**{'2446-browser-owner-controls':browser_source()})
        and new_profile['files']==old_profile['files']+[browser_cost()], 'NFL_CAPACITY_RETAINED_INPUT_CHANGED')
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE and text(root,'show','-s','--format=%P',BASE).split()==PARENTS,'NFL_CAPACITY_BASE_CHANGED')
    inherited=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,inherited['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only') and not text(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    old_raw=git(root,'show',BASE+':'+BINDING);require(sha(old_raw)==OLD_STAMP,'NFL_CAPACITY_PREDECESSOR_CHANGED');prior=json.loads(old_raw)
    expected=copy.deepcopy(binding)
    for k in ('components','nfl_capacity_profile','nfl_capacity_parent_binding','nfl_capacity_predecessors'):expected.pop(k,None)
    unchanged=copy.deepcopy(prior);unchanged.pop('components');require(expected==unchanged,'NFL_CAPACITY_HISTORY_CHANGED')
    parent=dict(before_sha256=OLD_STAMP,before_bytes=len(old_raw),before_components=prior['components'])
    require(binding['nfl_capacity_parent_binding']==parent,'NFL_CAPACITY_PREDECESSOR_CHANGED')
    components=copy.deepcopy(prior['components']);records={};before=blobs(root,BASE)
    for path in PATHS-{BINDING}:
        raw=(root/path).read_bytes().replace(b'\r\n',b'\n');require(len(raw)<=MAX_SOURCE,'NFL_CAPACITY_SIZE_LIMIT')
        components[path]=sha(normalized(path,raw,binding_sha))
        if path in prior['components'] and path in before:
            old=git(root,'show',BASE+':'+path);records[path]=dict(before_sha256=sha(old),before_bytes=len(old),before_b85=base64.b85encode(zlib.compress(old,9)).decode(),after_sha256=components[path])
    require(binding['components']==components and binding['nfl_capacity_predecessors']==records,'NFL_CAPACITY_COMPONENTS_CHANGED')
    for path in NORMALIZED:
        old=git(root,'show',BASE+':'+path);new=(root/path).read_bytes().replace(b'\r\n',b'\n')
        wanted=history_successor(old) if path.endswith('/ncaaf_filtered_history_scope.py') else old
        require(normalized(path,new,binding_sha)==normalized(path,wanted,OLD_STAMP),'INHERITED_GUARD_LOGIC_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[PREVIOUS_HEAD],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(git(root,'show',implementation+':'+POLICY)==git(root,'show',PREVIOUS_HEAD+':'+POLICY),'NFL_CAPACITY_FAILED_EVIDENCE_CHANGED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split();seal=head
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation] and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=blobs(root,'HEAD');require(POLICY not in before and set(actual)==set(before)|(PATHS-set(before))|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for path,value in actual.items():
        raw=(root/path).read_bytes();blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    # Classification follows complete exact-policy/physical/ancestry validation.
    # Retain the legacy violation field's meaning; never hide an approved edit.
    path='tests/test_ncaaf_filtered_history_scope.py'
    original=git(root,'show',BASE+':'+path)
    anchor=b'ROOT=Path(__file__).resolve().parents[1]'
    wanted=original.replace(anchor,b'from tests.nfl_capacity_profile_inputs import historical_filtered_profile_generation\n\n'+anchor,1)
    require(original.count(anchor)==1 and git(root,'show',implementation+':'+path)==wanted,'CONTINUATION_PRIOR_ASSERTIONS_CHANGED')
    preserved=preserved_test_report(root,BASE,implementation,PATHS)
    adaptations={path:dict(before_sha256=sha(original),after_sha256=sha(wanted),
        before_blob=before[path],after_blob=actual[path],authorization=CORRECTION,authorization_sha256=c['correction_sha256'])}
    browser_before=git(root,'show',PREVIOUS_HEAD+':'+BROWSER_HELPER)
    browser_wanted=browser_before.replace(b'import pytest\n',b'import pytest\nfrom tests.browser_process_owner import retain_browser_failure, run_browser\n',1)
    browser_wanted=browser_wanted.replace(b'def inspect_browser(',b'@retain_browser_failure\ndef inspect_browser(',1)
    browser_wanted=browser_wanted.replace(b'    run=subprocess.run(command,capture_output=True,\n        timeout=40,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)\n',b'    run=run_browser(command,directory)\n',1)
    require(git(root,'show',implementation+':'+BROWSER_HELPER)==browser_wanted,'BROWSER_PRIOR_ASSERTIONS_CHANGED')
    adaptations[BROWSER_HELPER]=dict(before_sha256=sha(browser_before),after_sha256=sha(browser_wanted),
        before_blob=before[BROWSER_HELPER],after_blob=actual[BROWSER_HELPER],authorization=BROWSER_AUTHORIZATION,
        authorization_sha256=c['browser_authorization_sha256'])
    return dict(schema_version=97,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(PATHS|{POLICY}),protected_changes=[],
        **preserved,approved_fixture_test_adaptations=adaptations,new_existing_test_exceptions=[],
        accepted=False,inference=False,inherited_runtime_and_readers_unchanged=True)
