"""Exact passive-sidebar successor; immutable predecessor evidence stays frozen."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE='5dfbb74cd252140512acb0fd9f31d40306408888'
TREE='afca3f24e639ac841e847cd65394fc0dd2bd86e6'
PARENTS=['e5d3b4d5e79115d42093adcd2708c0413df75a66','e7f49864a9a01aaf46f6d50d81cc268b04f17057']
OLD_STAMP='1112e83c0603d8ff161d2430a9400f9b079fe2b1189e9d334fa6735965aeef9a'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
POLICY='docs/paid-launch/launch-scope-policy-sidebar-startup-v1.json'
AUTHORIZATION='docs/paid-launch/sidebar-startup-correction-v1.md'
PROFILE='scripts/ci_test_file_costs_v1.json'
NORMALIZED={'scripts/check_launch_change_scope.py','scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py'}
PATHS={BINDING,AUTHORIZATION,PROFILE,'app/ui/sidebar_controls.py','app/ui/results_overview.py','scripts/ncaaf_sidebar_startup_scope.py',
       'scripts/ncaaf_saved_refresh_scope.py','tests/nfl_profile_generation_inputs.py',
       'tests/test_legacy_sidebar_startup.py','tests/test_ncaaf_sidebar_startup_scope.py',*NORMALIZED}
SOURCE='sidebar-startup-local-estimate-20261030-v1'
NEW_TESTS={'tests/test_legacy_sidebar_startup.py','tests/test_ncaaf_sidebar_startup_scope.py'}
APPROVED={'app/ui/results_overview.py': 'a34a768cb47ee35afa5811785a133ac0193a4c61b59ae89c1ce85bbbc76f65a6', 'app/ui/sidebar_controls.py': '6fc0355631b9425385cf9411441f8b27977326b00addb8f5ec04312853a5b2e6', 'docs/paid-launch/sidebar-startup-correction-v1.md': '1935419ebf4cb78cf581d2b85088934f780212debf94de3d39f792e82db533ac', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ci_test_file_costs_v1.json': '4c99972662dcef210b140f9ef2584a49182f4c5a13faa6514fcbd044996717be', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/ncaaf_saved_refresh_scope.py': '03e97fcd755aae733541feb760a4d39a7300cc925a75587212e156f3bff2379b', 'tests/nfl_profile_generation_inputs.py': '7d40e38c134e49080e8e524e665a3a72fba44c02150b2981a10830be16359855', 'tests/test_legacy_sidebar_startup.py': '664317312e9f15fd3f8d68099892d7e9a19bae089c403270db80ea55fd9d26e2', 'tests/test_ncaaf_sidebar_startup_scope.py': '3c03959a1d66ca8acb5481a37079ae5b6f1260d15ad27ed02a6f7403e78d7a41'}
LIMIT=8*1024*1024

def require(value,code):
    if not value:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def text(root,*args):return git(root,*args).decode().strip()
def normalized(path,raw,stamp):
    raw=raw.replace(b'\r\n',b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode())==1,'SIDEBAR_BINDING_CHANGED')
        raw=raw.replace(stamp.encode(),b'0'*64)
    return raw
def blobs(root,revision):
    return {p:m.split()[2] for line in text(root,'ls-tree','-r',revision).splitlines() for m,p in [line.split('\t',1)]}
def saved(binding):
    return binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']['tls_trust_successor']['clock_fixture_successor']['ncaaf_saved_refresh_successor']
def metadata(binding):return saved(binding)['sidebar_startup_successor']

def record(raw):
    return dict(bytes=len(raw),sha256=sha(raw),compressed_b85=base64.b85encode(zlib.compress(raw,9)).decode())
def record_bytes(value):
    z=zlib.decompressobj();raw=z.decompress(base64.b85decode(value['compressed_b85']),LIMIT+1)
    require(z.eof and not z.unused_data and not z.unconsumed_tail and len(raw)==value['bytes']<=LIMIT and sha(raw)==value['sha256'],'SIDEBAR_PREDECESSOR_CHANGED')
    return raw

def parent_binding(binding):
    m=metadata(binding);prior=json.loads(json.dumps(binding))
    del saved(prior)['sidebar_startup_successor'];prior['components']=m['parent_components']
    prior['nfl_capacity_predecessors']={p:r for p,r in prior['nfl_capacity_predecessors'].items() if p in m['parent_predecessor_after_sha256']}
    for p,h in m['parent_predecessor_after_sha256'].items():prior['nfl_capacity_predecessors'][p]['after_sha256']=h
    raw=canonical(prior)
    require(len(raw)==m['parent_binding_bytes'] and sha(raw)==m['prior_binding_sha256']==OLD_STAMP,'SIDEBAR_PARENT_BINDING_CHANGED')
    return raw

def verify_profile(root):
    before=json.loads(git(root,'show',BASE+':'+PROFILE));after=json.loads((Path(root)/PROFILE).read_bytes())
    old_files={r['path']:r for r in before['files']};new_files={r['path']:r for r in after['files']}
    require(new_files.keys()==old_files.keys()|NEW_TESTS and all(new_files[p]==r for p,r in old_files.items()),'SIDEBAR_PROFILE_EXISTING_COST_CHANGED')
    require(SOURCE not in before['sources'] and after['sources'].keys()==before['sources'].keys()|{SOURCE},'SIDEBAR_PROFILE_SOURCE_CHANGED')
    for p in NEW_TESTS:
        row=new_files[p];require(row['kind']=='estimated' and row['profiled_tests']>0 and row['seconds']>0 and row['references']==[SOURCE],'SIDEBAR_PROFILE_NEW_COST_CHANGED')
    restored=json.loads(json.dumps(after));restored['files']=before['files'];restored['sources'].pop(SOURCE)
    require(restored==before,'SIDEBAR_PROFILE_UNRELATED_CHANGE')

def binding_successor(root,stamp):
    before=git(root,'show',BASE+':'+BINDING);require(sha(before)==OLD_STAMP,'SIDEBAR_BASE_BINDING_CHANGED')
    b=json.loads(before);prior=dict(b['components']);old_after={p:r['after_sha256'] for p,r in b['nfl_capacity_predecessors'].items()};existing=blobs(root,BASE);historical={}
    verify_profile(root)
    for p in PATHS-{BINDING}:
        raw=(Path(root)/p).read_bytes();digest=sha(normalized(p,raw,stamp))
        if p in existing and p not in NORMALIZED:historical[p]=record(git(root,'show',BASE+':'+p))
        if p in b['nfl_capacity_predecessors']:b['nfl_capacity_predecessors'][p]['after_sha256']=digest
        elif p in prior:
            old=git(root,'show',BASE+':'+p)
            b['nfl_capacity_predecessors'][p]=dict(before_bytes=len(old),before_sha256=sha(old),before_b85=base64.b85encode(zlib.compress(old,9)).decode(),after_sha256=digest)
        b['components'][p]=digest
    saved(b)['sidebar_startup_successor']=dict(version=1,base=BASE,tree=TREE,prior_binding_sha256=OLD_STAMP,parent_binding_bytes=len(before),
        parent_components=prior,parent_predecessor_after_sha256=old_after,paths=sorted(PATHS),historical_sources=historical,
        prior_component_sha256={p:prior[p] for p in historical if p in prior},authorization_sha256=b['components'][AUTHORIZATION])
    require(parent_binding(b)==before,'SIDEBAR_PARENT_BINDING_CHANGED')
    return b

def make_policy(root,binding,implementation,binding_sha):
    return dict(version='sidebar-startup-seal-v1',base=BASE,base_tree=TREE,ordered_parents=PARENTS,binding_sha256=binding_sha,
        predecessor_policy_sha256=sha(git(root,'show',BASE+':docs/paid-launch/launch-scope-policy-ncaaf-saved-refresh-v1.json')),
        authorization_sha256=binding['components'][AUTHORIZATION],implementation_commit=implementation,
        implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})

def validate(root,binding,comparison_base,manifest,*,binding_sha):
    root=Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE and text(root,'show','-s','--format=%P',BASE).split()==PARENTS,'SIDEBAR_BASE_CHANGED')
    inherited=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,inherited['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'status','--porcelain'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==binding_successor(root,binding_sha),'SIDEBAR_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p,(root/p).read_bytes(),binding_sha))==h for p,h in APPROVED.items()),'SIDEBAR_SOURCE_NOT_APPROVED')
    for p in NORMALIZED:require(normalized(p,git(root,'show',BASE+':'+p),OLD_STAMP)==normalized(p,(root/p).read_bytes(),binding_sha),'INHERITED_GUARD_LOGIC_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split();seal=head
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation] and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=blobs(root,'HEAD');require(actual.keys()==blobs(root,BASE).keys()|PATHS|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for p,oid in actual.items():
        raw=(root/p).read_bytes();blob=lambda x:hashlib.sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()
        require(oid in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=107,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,tree=text(root,'rev-parse',head+'^{tree}'),
        changed_files=sorted(text(root,'diff','--name-only',BASE,'HEAD').splitlines()),protected_changes=[],existing_test_changes=[],
        approved_test_helper_changes=['tests/nfl_profile_generation_inputs.py'],approved_fixture_test_adaptations={},accepted=False,inference=False)
