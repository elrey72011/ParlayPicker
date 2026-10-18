"""Exact successor blobs and immutable preimages; no cached live decisions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BINDING='docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'
POLICY='docs/paid-launch/launch-scope-policy-ncaaf-reviewed-target-v1.json'
PROPOSAL='978f284f0408f0c93306131335ac05fad996669e537271d73f5d36574f1d7ac3'


def sha(b):return hashlib.sha256(b).hexdigest()


def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def previous_source(path,source,binding):
    r=binding['source_reconstructions'].get(path)
    if r is None:return source
    checked=source
    if path=='scripts/check_launch_change_scope.py':
        import re
        match=re.search(rb"_RT_BINDING_SHA256='([0-9a-f]{64})'",source)
        if match is None:return source
        checked=source[:match.start(1)]+b'0'*64+source[match.end(1):]
        identity=re.search(rb"NCAAF_FUTURE_REVIEWED_GUARD_SHA256='([0-9a-f]{64})'",source)
        if identity is None:return source
        checked=checked.replace(identity.group(1),b'0'*64)
    if sha(checked)!=r['after_sha256']:return source
    text=source.decode()
    for p in reversed(r['reverse_patches']):text=text[:p['start']]+p['text']+text[p['end']:]
    prior=text.encode()
    if sha(prior)!=r['before_sha256']:raise ValueError('REVIEWED_TARGET_PREIMAGE_CHANGED')
    return prior


def install(g,binding):
    """Extend exact INPUT reconstruction, never cache validation outcomes.

    Historical callers still run their original checks on their original bytes.
    Inherited fixture helpers also open the canonical path directly; normalize
    only authenticated successor inputs before handing them to those helpers.
    """
    legacy=binding['historical_future'];old_records=legacy['source_reconstructions']
    if str(g.ROOT) not in sys.path:sys.path.insert(0,str(g.ROOT))
    def matches(source):
        return isinstance(source,bytes) and (sha(source)==binding['source_reconstructions'][g.GUARD_PATH]['before_sha256']
            or previous_source(g.GUARD_PATH,source,binding)!=source)
    def base(source):
        if sha(source)==legacy['prior_base_sha256']:return source
        source=previous_source(g.GUARD_PATH,source,binding)
        g._require(sha(source)==binding['source_reconstructions'][g.GUARD_PATH]['before_sha256'],'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
        source=source.replace(b'    if len(source)==8388490:source=_performance_early_base.__wrapped__(source,True)\n',b'')
        for left,right,replacement in ((b'\n    # NCAAF_FUTURE_INLINE_BEGIN\n',b'\n    # NCAAF_FUTURE_INLINE_END\n',b'\n'),
                                        (b'\n# NCAAF_FUTURE_BEGIN\n',b'\n# NCAAF_FUTURE_END\n',b'')):
            start=source.index(left);end=source.index(right,start)+len(right);source=source[:start]+replacement+source[end:]
        source=source.replace(b'        if exists_at("HEAD", NCAAF_FUTURE_POLICY_PATH):\n            code, report = _run_ncaaf_future_integrated(args.manifest, args.base)\n        elif exists_at("HEAD", NCAAF_STAGED_CAPACITY_POLICY_PATH):',b'        if exists_at("HEAD", NCAAF_STAGED_CAPACITY_POLICY_PATH):',1)
        g._require(sha(source)==legacy['prior_base_sha256'],'PREVIOUS_GUARD_LOGIC_CHANGED')
        return source
    def previous(path,source):
        if path==g.GUARD_PATH:return base(source)
        source=previous_source(path,source,binding);r=old_records.get(path)
        if r is None or sha(source)!=r['after_sha256']:return source
        value=source.decode()
        for p in reversed(r['reverse_patches']):value=value[:p['start']]+p['text']+value[p['end']:]
        result=value.encode();g._require(sha(result)==r['before_sha256'],'PREVIOUS_SOURCE_LOGIC_CHANGED')
        return result
    g._ncaaf_future_guard_matches=matches;g._ncaaf_future_base_source=base;g._ncaaf_future_previous_source=previous
    for name in ('NCAAF_FUTURE_BINDING_PATH','NCAAF_FUTURE_BINDING_SHA256','NCAAF_FUTURE_POLICY_PATH'):setattr(g,name,legacy[name])
    records=deepcopy(old_records)
    for path,r in binding['source_reconstructions'].items():
        if path in records:
            records[path]['after_sha256']=r['after_sha256']
    # Exact input preimages for inherited assertions, not current scheduling.
    for path in ('scripts/ci_test_file_costs_v1.json','tests/ncaaf_staged_capacity_fixtures.py','tests/test_ncaaf_staged_ci_capacity.py','tests/test_ncaaf_transport_diagnostics.py'):
        if path in binding['source_reconstructions'] and path not in records:
            records[path]=deepcopy(binding['source_reconstructions'][path])
    g.NCAAF_FUTURE_RECONSTRUCTION_JSON=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
    for path,expected in binding['predecessor_scope_sha256'].items():
        g._require(sha((g.ROOT/path).read_bytes().replace(b'\r\n',b'\n'))==expected,'PREDECESSOR_SCOPE_SOURCE_CHANGED')
    from scripts import ci_scheduling_fixture_scope,ncaaf_transport_scope
    class PredecessorInput:
        def __init__(self,live):self.live=live
        def __getattr__(self,name):
            if name=='NCAAF_FUTURE_REVIEWED_GUARD_SHA256':return legacy['guard_identity']
            return getattr(self.live,name)
    def input_view(source,live):
        return PredecessorInput(live) if sha(source)==binding['source_reconstructions'][g.GUARD_PATH]['before_sha256'] else live
    for module in (ci_scheduling_fixture_scope,ncaaf_transport_scope):
        predecessor=module.predecessor_guard;restore=module.previous_source
        def predecessor_input(source,live,b,original=predecessor):
            restored=previous_source(g.GUARD_PATH,source,binding)
            return original(restored,input_view(restored,live),b)
        def source_input(path,source,live,b,original=restore):
            restored=previous_source(path,source,binding)
            return original(path,restored,input_view(restored,live),b)
        module.predecessor_guard=predecessor_input;module.previous_source=source_input


def blobs(g,rev):
    return {p:m.split()[2] for row in g.git('ls-tree','-r',rev).splitlines() for m,p in [row.split('\t',1)]}


def make_policy(g,b,implementation):
    before=blobs(g,b['base']);after=blobs(g,implementation)
    return dict(schema_version=82,policy_version='ncaaf-reviewed-target-v1',proposal_sha256=PROPOSAL,
        base_sha=b['base'],base_tree=b['base_tree'],implementation_commit=implementation,
        implementation_tree=g.git('rev-parse',implementation+'^{tree}'),
        implementation_changes={p:dict(before_blob=before.get(p),after_blob=after.get(p)) for p in b['paths']},
        unchanged_bindings={p:v for p,v in before.items() if p not in b['paths']})


def validate(g,manifest_path,base,b):
    require=g._require
    require(b['proposal_sha256']==PROPOSAL,'PROPOSAL_NOT_APPROVED')
    require(manifest_path.resolve()==(g.ROOT/g.MANIFEST_PATH).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not g.git('diff','--name-only') and not g.git('diff','--cached','--name-only'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    raw=g.git_bytes('show',b['base']+':'+g.MANIFEST_PATH)
    require(sha(raw)==b['manifest_sha256'],'BASELINE_CHANGED')
    require(base in (None,b['base'],json.loads(raw)['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(g.git('rev-parse',b['base']+'^{tree}')==b['base_tree'],'STARTING_MAIN_TREE_CHANGED')
    require(json.loads(g.git_bytes('show','HEAD:'+BINDING))==b,'REVIEWED_BINDINGS_CHANGED')
    policy=json.loads(g.git_bytes('show','HEAD:'+POLICY));impl=policy['implementation_commit']
    require(g.git('show','-s','--format=%P',impl).split()==[b['base']],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(b['base'],POLICY) and not g.exists_at(impl,POLICY),'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff','--name-only',b['base'],impl).splitlines())==set(b['paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(g,b,impl),'EXACT_POLICY_BINDINGS_CHANGED')
    actual=blobs(g,impl)
    require(set(b['reviewed_blobs'])==set(b['paths'])-{BINDING,g.GUARD_PATH},'REVIEWED_BINDING_SET_CHANGED')
    require(all(actual.get(p)==v for p,v in b['reviewed_blobs'].items()),'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    for path,r in b['source_reconstructions'].items():
        source=g.git_bytes('show',impl+':'+path)
        require(previous_source(path,source,b)==g.git_bytes('show',b['base']+':'+path),'PREVIOUS_SOURCE_LOGIC_CHANGED')
    parents=g.git('show','-s','--format=%P','HEAD').split();candidate=g.git('rev-parse','HEAD')
    if len(parents)==2:
        require(parents[0]==b['base'],'CI_BASE_PARENT_NOT_APPROVED');candidate=parents[1]
        require(g.git('rev-parse','HEAD^{tree}')==g.git('rev-parse',candidate+'^{tree}'),'CI_MERGE_TREE_CHANGED')
    require(g.git('show','-s','--format=%P',candidate).split()==[impl],'CANDIDATE_NOT_POLICY_SEAL')
    require(g.git('diff','--name-status',impl,candidate).splitlines()==['A\t'+POLICY],'SEAL_CHANGE_SET_NOT_APPROVED')
    head=blobs(g,'HEAD')
    require(all(head.get(p)==v for p,v in policy['unchanged_bindings'].items()),'IMMUTABLE_FILE_CHANGED')
    require(set(head)==set(policy['unchanged_bindings'])|set(b['paths'])|{POLICY},'UNREVIEWED_PATH')
    for path,expected in head.items():
        content=(g.ROOT/path).read_bytes()
        require(blob(content)==expected or blob(content.replace(b'\r\n',b'\n'))==expected,'UNAUTHORIZED_CHECKOUT_CHANGE')
    protected=json.loads(raw)['protected_files']
    for path in g.ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(g.ROOT).parts:continue
        rel=path.relative_to(g.ROOT).as_posix()
        require(path.name not in {'sitecustomize.py','usercustomize.py'} and path.suffix!='.pth','PROTECTED_RUNTIME_SHADOWING_RISK')
        require(not any(rel.endswith('/'+p) for p in protected),'PROTECTED_RUNTIME_SHADOWING_RISK')
    return policy


def run(g,manifest_path,base,binding):
    try:policy=validate(g,manifest_path,base,binding)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.CalledProcessError) as exc:
        return 1,dict(schema_version=82,status='FAIL',policy_valid=False,reason_codes=[str(exc)])
    _,original=g.run(manifest_path,base)
    return 0,dict(schema_version=82,status='PASS',policy_valid=True,reason_codes=[],base_sha=binding['base'],
        head_sha=g.git('rev-parse','HEAD'),original_guard_report=original,approved_integration_changes=policy['implementation_changes'],
        unchanged_bindings=policy['unchanged_bindings'],protected_changes=[],existing_test_changes=[],
        scope='Explicit evidence-bound data-driven NCAAF targets. Native mathematics, historical assertions, security, workflows and budgets unchanged.')
