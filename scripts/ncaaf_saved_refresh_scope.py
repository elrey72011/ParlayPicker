"""Exact local saved NCAAF viewer successor; preserves frozen predecessor inputs."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE='e5d3b4d5e79115d42093adcd2708c0413df75a66'
TREE='b9b638cdb184782eb98d7c599dbce1dbbb067290'
PARENTS=['c548ddb2693b64c94942ff315eed93e3fcb2f56e','262f44cfe2323f5fde31a4497f94b2e80da1d46b']
OLD_STAMP='12050235edcb7e0bdb9101baaa59c1e16bfbd8c29322c8350ca7a8a5cd119d68'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION='docs/paid-launch/ncaaf-saved-refresh-viewer-v1.md'
POLICY='docs/paid-launch/launch-scope-policy-ncaaf-saved-refresh-v1.json'
NORMALIZED={'scripts/check_launch_change_scope.py','scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py'}
PATHS={BINDING,AUTHORIZATION,'scripts/ci_test_file_costs_v1.json','app/ui/ncaaf_saved_refresh.py','app/ui/publish_panel.py',
       'scripts/nfl_tls_clock_scope.py','scripts/ncaaf_saved_refresh_scope.py',
       'tests/nfl_profile_generation_inputs.py','tests/test_ncaaf_saved_refresh_preparation.py',
       'tests/test_ncaaf_saved_refresh_scope.py',*NORMALIZED}
APPROVED={'app/ui/ncaaf_saved_refresh.py': 'b39a4719b31292f58fa3b6547ab7ee3173901732b28d9010499d923b7f3ab7d0', 'app/ui/publish_panel.py': '1133728466654d8fb78f9b909df91b8220a8de1e528875e337a7ad5b021356e3', 'docs/paid-launch/ncaaf-saved-refresh-viewer-v1.md': '1b55298bc37b7783bdea78e126690f39ef93f1d4a106677d4ce7f2c4c5d74ec1', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ci_test_file_costs_v1.json': '1c6e2e31fce3da58bc7a6443d0a16a39bfd65d7b755588f22030b21bd43b21bb', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/nfl_tls_clock_scope.py': '39a2236bf473fb325c6ac32b57da79f6ea4a2431c0f525cffa412c01918d83db', 'tests/nfl_profile_generation_inputs.py': 'd7ffe52dddd73244083c393cb7d151b70f7e6242b3cb86fc9c1960edcee24ffd', 'tests/test_ncaaf_saved_refresh_preparation.py': '219cf7322f1581a717a8945264f81847a2501d10af1040ee3a91bf90e38912b5', 'tests/test_ncaaf_saved_refresh_scope.py': '1b0833da352cf9f41b72c97b9d8539f9f7203dd5c18fb1f5cfd81e9b722e0ad9'}

def require(ok,code):
    if not ok:raise ValueError(code)
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def text(root,*args):return git(root,*args).decode().strip()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return (json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
def normalized(path,raw,stamp):
    raw=raw.replace(b'\r\n',b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode())==1,'NCAAF_SAVED_BINDING_CHANGED')
        raw=raw.replace(stamp.encode(),b'0'*64)
    return raw
def blobs(root,revision):
    return {p:m.split()[2] for line in text(root,'ls-tree','-r',revision).splitlines()
        for m,p in [line.split('\t',1)]}
def metadata(b):
    return b['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']['tls_trust_successor']['clock_fixture_successor']
def binding_successor(root,stamp):
    raw=git(root,'show',BASE+':'+BINDING)
    require(sha(raw)==OLD_STAMP,'NCAAF_SAVED_PREDECESSOR_BINDING_CHANGED')
    b=json.loads(raw);prior=dict(b['components']);historical={};existing=blobs(root,BASE)
    for path in PATHS-{BINDING}:
        digest=sha(normalized(path,(Path(root)/path).read_bytes(),stamp))
        if path in existing and path not in NORMALIZED:
            before=git(root,'show',BASE+':'+path)
            historical[path]=dict(bytes=len(before),sha256=sha(before),compressed_b85=base64.b85encode(zlib.compress(before,9)).decode())
        if path in b['nfl_capacity_predecessors']:
            b['nfl_capacity_predecessors'][path]['after_sha256']=digest
        elif path in prior:
            before=git(root,'show',BASE+':'+path)
            b['nfl_capacity_predecessors'][path]=dict(before_bytes=len(before),before_sha256=sha(before),before_b85=base64.b85encode(zlib.compress(before,9)).decode(),after_sha256=digest)
        b['components'][path]=digest
    metadata(b)['ncaaf_saved_refresh_successor']=dict(version=1,base=BASE,tree=TREE,
        prior_binding_sha256=OLD_STAMP,authorization_sha256=b['components'][AUTHORIZATION],
        paths=sorted(PATHS),historical_sources=historical,
        prior_component_sha256={p:prior[p] for p in historical if p in prior})
    return b
def make_policy(root,binding,implementation,binding_sha):
    return dict(version='ncaaf-saved-refresh-seal-v1',base=BASE,base_tree=TREE,
        ordered_parents=PARENTS,binding_sha256=binding_sha,
        authorization_sha256=binding['components'][AUTHORIZATION],implementation_commit=implementation,
        implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})
def validate(root,binding,comparison_base,manifest,*,binding_sha):
    root=Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE and text(root,'show','-s','--format=%P',BASE).split()==PARENTS,'NCAAF_SAVED_BASE_CHANGED')
    inherited=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,inherited['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'status','--porcelain'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==binding_successor(root,binding_sha),'NCAAF_SAVED_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p,(root/p).read_bytes(),binding_sha))==digest for p,digest in APPROVED.items()),'NCAAF_SAVED_SOURCE_NOT_APPROVED')
    for path in NORMALIZED:
        require(normalized(path,git(root,'show',BASE+':'+path),OLD_STAMP)==normalized(path,(root/path).read_bytes(),binding_sha),'INHERITED_GUARD_LOGIC_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split();seal=head
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation] and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=blobs(root,'HEAD')
    require(actual.keys()==blobs(root,BASE).keys()|PATHS|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for path,value in actual.items():
        raw=(root/path).read_bytes()
        blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=106,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,tree=text(root,'rev-parse',head+'^{tree}'),
        changed_files=sorted(text(root,'diff','--name-only',BASE,'HEAD').splitlines()),protected_changes=[],existing_test_changes=[],
        approved_test_helper_changes=['tests/nfl_profile_generation_inputs.py'],
        approved_fixture_test_adaptations={},accepted=False,inference=False)
