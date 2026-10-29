"""Exact fixture-clock successor; original TLS approval expectations are frozen."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE='e4a490dd2830d1a293412f6a1a99cf529eb30084'
TREE='063c62d621dd5a9eb16f697f636c83db0fa79c2a'
MAIN='c548ddb2693b64c94942ff315eed93e3fcb2f56e'
PREDECESSOR_PARENTS=['6f5190d81e601aec6f6ac28a32f0583cae043676']
OLD_STAMP='14e603462380727923c58513842d02926e92dbc5692e994c90f00530e2eed342'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION='docs/paid-launch/ncaaf-schedule-fixture-clock-v1.md'
POLICY='docs/paid-launch/launch-scope-policy-nfl-tls-clock-v1.json'
FIXTURE='tests/test_ncaaf_normal_pipeline.py'
NORMALIZED={'scripts/check_launch_change_scope.py','scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py'}
PATHS={BINDING,AUTHORIZATION,FIXTURE,'scripts/nfl_tls_trust_scope.py','scripts/nfl_tls_clock_scope.py',
       'tests/nfl_profile_generation_inputs.py','tests/test_ncaaf_schedule_fixture_clock.py',*NORMALIZED}
APPROVED={'docs/paid-launch/ncaaf-schedule-fixture-clock-v1.md': '00b68898f31a281e4805b5f1db8b75fb5c0f06a8a3a08de3b89d919c5f8f721e', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/nfl_tls_trust_scope.py': 'ae56a3a44876595a329183a5dbd78f0adeebbb60249944dc15e944e6ae6f9afd', 'tests/nfl_profile_generation_inputs.py': '6cacf422682be99b8a53e8fc7ee5a19220c6d4631a086a2d6f6e832b844d9428', 'tests/test_ncaaf_normal_pipeline.py': 'c64830e1769af937709f760fc838d03fcff987b9e5b3f4235f549be78a6e5615', 'tests/test_ncaaf_schedule_fixture_clock.py': 'cd5bb13bafaf0acc9739721f12874af3bfd0cce2dd17463cafd66a1782ed9441'}


def require(ok, code):
    if not ok: raise ValueError(code)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def normalized(path, raw, stamp):
    raw = raw.replace(b'\r\n', b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode()) == 1, 'RECEIPT_BINDING_STAMP_CHANGED')
        raw = raw.replace(stamp.encode(), b'0' * 64)
    return raw


def blobs(root, revision):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', revision).splitlines()
            for m, p in [line.split('\t', 1)]}



def metadata(binding):
    return binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']['tls_trust_successor']


def binding_successor(root,stamp):
    raw=git(root,'show',BASE+':'+BINDING)
    require(sha(raw)==OLD_STAMP,'CLOCK_PREDECESSOR_BINDING_CHANGED')
    binding=json.loads(raw);prior_components=dict(binding['components'])
    historical={};existing=blobs(root,BASE)
    for path in PATHS-{BINDING}:
        digest=sha(normalized(path,(Path(root)/path).read_bytes(),stamp))
        if path in existing and path not in NORMALIZED:
            before=git(root,'show',BASE+':'+path)
            historical[path]=dict(bytes=len(before),sha256=sha(before),compressed_b85=base64.b85encode(zlib.compress(before,9)).decode())
        if path in binding['nfl_capacity_predecessors']:
            binding['nfl_capacity_predecessors'][path]['after_sha256']=digest
        elif path in prior_components:
            before=git(root,'show',BASE+':'+path)
            binding['nfl_capacity_predecessors'][path]=dict(before_bytes=len(before),before_sha256=sha(before),
                before_b85=base64.b85encode(zlib.compress(before,9)).decode(),after_sha256=digest)
        binding['components'][path]=digest
    metadata(binding)['clock_fixture_successor']=dict(version=1,base=BASE,tree=TREE,
        prior_binding_sha256=OLD_STAMP,authorization_sha256=binding['components'][AUTHORIZATION],
        paths=sorted(PATHS),historical_sources=historical,
        prior_component_sha256={p:prior_components[p] for p in historical if p in prior_components},
        fixture_change='explicit synthetic schedule observed_at; original assertions unchanged')
    return binding


def make_policy(root,binding,implementation,binding_sha):
    return dict(version='nfl-tls-clock-fixture-seal-v1',base=BASE,base_tree=TREE,main=MAIN,
        predecessor_parents=PREDECESSOR_PARENTS,binding_sha256=binding_sha,
        predecessor_policy_sha256=sha(git(root,'show',BASE+':docs/paid-launch/launch-scope-policy-nfl-tls-trust-v1.json')),
        authorization_sha256=binding['components'][AUTHORIZATION],implementation_commit=implementation,
        implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})


def validate(root,binding,comparison_base,manifest,*,binding_sha):
    root=Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE and
        text(root,'show','-s','--format=%P',BASE).split()==PREDECESSOR_PARENTS and
        text(root,'merge-base',MAIN,BASE)==MAIN,'CLOCK_BASE_CHANGED')
    inherited=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,MAIN,inherited['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only') and
        not text(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==binding_successor(root,binding_sha),'CLOCK_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p,(root/p).read_bytes(),binding_sha))==digest
        for p,digest in APPROVED.items()),'CLOCK_SOURCE_NOT_APPROVED')
    for path in NORMALIZED:
        require(normalized(path,git(root,'show',BASE+':'+path),OLD_STAMP)==
            normalized(path,(root/path).read_bytes(),binding_sha),'INHERITED_GUARD_LOGIC_CHANGED')
    before=git(root,'show',BASE+':'+FIXTURE)
    needle=b'"2026-10-30","2026-10-30",complete=True)'
    require(before.count(needle)==1 and (root/FIXTURE).read_bytes().replace(b'\r\n',b'\n')==
        before.replace(needle,b'"2026-10-30","2026-10-30",complete=True,observed_at=AT.isoformat())'),
        'ORIGINAL_FIXTURE_ASSERTIONS_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split();seal=head
    if len(parents)==2:
        require(parents[0]==MAIN,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation] and
        text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    actual=blobs(root,'HEAD')
    require(actual.keys()==blobs(root,BASE).keys()|PATHS|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for path,value in actual.items():
        raw=(root/path).read_bytes()
        blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    before_blobs=blobs(root,BASE)
    return dict(schema_version=105,status='PASS',base=MAIN,preserved_head=BASE,implementation=implementation,
        seal=seal,candidate=head,tree=text(root,'rev-parse',head+'^{tree}'),
        changed_files=sorted(text(root,'diff','--name-only',MAIN,'HEAD').splitlines()),
        protected_changes=[],existing_test_changes=[],approved_fixture_test_adaptations={FIXTURE:dict(
            before_sha256=sha(before),after_sha256=sha((root/FIXTURE).read_bytes()),
            before_blob=before_blobs[FIXTURE],after_blob=actual[FIXTURE],authorization=AUTHORIZATION,
            authorization_sha256=binding['components'][AUTHORIZATION],
            exact_change='explicit synthetic observed_at only; all original assertions preserved')},
        approved_test_helper_changes=['tests/nfl_profile_generation_inputs.py'],
        tls_preflight_and_production_chronology_unchanged=True,accepted=False,inference=False)
