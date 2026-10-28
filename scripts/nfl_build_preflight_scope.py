"""Exact owner-authorized build-inspection successor; historical seals stay intact."""
import base64
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = '3fb9302463f44589ea0d2c75b2ed100bf2f596d0'
TREE = 'cdc34e454f7b5312e2673ecfd39126956efe5d39'
PARENTS = ['2472bc05ee9f6c3ce30c387ee4695583a2a04281', '99d3265477cb15f146bbb142e8cbbc956a65cf88']
OLD_STAMP = 'a44f34e170c593ae5de31043896297074a16da686c6c476930060cdbb95d1bbc'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-build-preflight-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-build-preflight-v1.md'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py', 'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, 'app_core/nfl_build_preflight.py', 'app/ui/nfl_storage_diagnostic.py',
         'tests/test_nfl_build_preflight.py', 'scripts/nfl_build_preflight_scope.py',
         'scripts/qualification_repair_scope.py', 'tests/test_nfl_private_workflow_ui.py', *NORMALIZED}
APPROVED = {'app/ui/nfl_storage_diagnostic.py': 'eddc1139ae26bdd2dee3686e3f64ec56f411ed0c7ba9037d601ce69b2006873b', 'app_core/nfl_build_preflight.py': '14e777e849367eed7d542fe2c273de747b96784b159b7b24984fe00ca276b8f4', 'docs/paid-launch/nfl-build-preflight-v1.md': '780237bbf95812ae0393265ba6fe3fd4675659354b2105715312d05b28d1cc6e', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/qualification_repair_scope.py': 'b8adbbd2dbd4f527b002a54b6c98604cf230853253a1f7a0c98531a5cf5b773a', 'tests/test_nfl_build_preflight.py': '2853eb25d5cb3f54c299243e65cdeb2c954bcac848164f5dbbf55f78d86e323c', 'tests/test_nfl_private_workflow_ui.py': '59842a578688060bfad4bc25c56b5c6b1726e4d16e196f9834f2a7486b9c095e'}  # Frozen independently after local source review, before binding generation.


def require(ok, code):
    if not ok:
        raise ValueError(code)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def normalized(path, raw, stamp):
    raw = raw.replace(b'\r\n', b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode()) == 1, 'BUILD_PREFLIGHT_STAMP_CHANGED')
        raw = raw.replace(stamp.encode(), b'0'*64)
    return raw


def blobs(root, revision):
    return {p: meta.split()[2] for line in text(root,'ls-tree','-r',revision).splitlines()
            for meta,p in [line.split('\t',1)]}


def binding_successor(root, stamp):
    raw = git(root,'show',BASE+':'+BINDING)
    require(sha(raw)==OLD_STAMP, 'BUILD_PREFLIGHT_BASE_BINDING_CHANGED')
    b = json.loads(raw)
    for path in PATHS-{BINDING}:
        digest = sha(normalized(path,(Path(root)/path).read_bytes(),stamp))
        if path in {'scripts/qualification_repair_scope.py', 'app/ui/nfl_storage_diagnostic.py'}:
            old = git(root,'show',BASE+':'+path)
            b['nfl_capacity_predecessors'][path] = dict(before_sha256=sha(old),before_bytes=len(old),
                before_b85=base64.b85encode(zlib.compress(old,9)).decode(),after_sha256=digest)
        elif path in b['nfl_capacity_predecessors']:
            b['nfl_capacity_predecessors'][path]['after_sha256']=digest
        b['components'][path]=digest
    b['qualification_release_repair']['build_preflight'] = dict(version=1,base=BASE,tree=TREE,
        authorization_sha256=b['components'][AUTHORIZATION],prior_binding_sha256=OLD_STAMP,
        paths=sorted(PATHS))
    return b


def make_policy(root,binding,implementation,binding_sha):
    return dict(version='nfl-build-preflight-seal-v1',base=BASE,base_tree=TREE,ordered_parents=PARENTS,
        binding_sha256=binding_sha,authorization_sha256=binding['components'][AUTHORIZATION],
        implementation_commit=implementation,implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),
        implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})


def validate(root,binding,comparison_base,manifest,*,binding_sha):
    root=Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}')==TREE and text(root,'show','-s','--format=%P',BASE).split()==PARENTS,
            'BUILD_PREFLIGHT_BASE_CHANGED')
    baseline=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,baseline['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),
            'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only')
            and not text(root,'ls-files','--others','--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding==binding_successor(root,binding_sha), 'BUILD_PREFLIGHT_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p,(root/p).read_bytes(),binding_sha))==v for p,v in APPROVED.items()),
            'BUILD_PREFLIGHT_SOURCE_NOT_APPROVED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS|{POLICY},
            'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split();seal=head
    if len(parents)==2:
        require(parents[0]==BASE, 'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
        and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(blobs(root,'HEAD').keys()==blobs(root,BASE).keys()|PATHS|{POLICY},'UNREVIEWED_PATH')
    actual=blobs(root,'HEAD')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for p,v in actual.items():
        raw=(root/p).read_bytes()
        blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(v in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))),'UNAUTHORIZED_CHECKOUT_CHANGE')
    fixture='tests/test_nfl_private_workflow_ui.py'
    before=git(root,'show',BASE+':'+fixture);after=(root/fixture).read_bytes()
    tests=lambda raw:{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(raw))
                      if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    require(tests(before)==tests(after),'BUILD_PREFLIGHT_ORIGINAL_TEST_CHANGED')
    assertions=lambda raw:{ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Assert)}
    require(assertions(before)<=assertions(after),'BUILD_PREFLIGHT_ASSERTION_CHANGED')
    adaptations={fixture:dict(before_sha256=sha(before),after_sha256=sha(after),
        before_blob=blobs(root,BASE)[fixture],after_blob=actual[fixture],authorization=AUTHORIZATION,
        authorization_sha256=binding['components'][AUTHORIZATION])}
    return dict(schema_version=100,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(PATHS|{POLICY}),protected_changes=[],
        existing_test_changes=[],approved_fixture_test_adaptations=adaptations,approved_test_helper_changes=['tests/test_nfl_build_preflight.py'],
        production_store_model_and_transport_unchanged=True,accepted=False,inference=False)
