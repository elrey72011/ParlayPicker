"""Exact diagnostics successor; immutable predecessor and real checkout checks."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

POLICY='docs/paid-launch/launch-scope-policy-ncaaf-transport-diagnostics-v1.json'
BINDING='docs/paid-launch/ncaaf-future-spread-binding-v1.json'
PROPOSAL='10ae4725e6b3448a517112384e6917587cf1bebdeed17a44b6a2f1dd67097b13'


def blobs(g,rev):
    return {p:m.split()[2] for row in g.git('ls-tree','-r',rev).splitlines() for m,p in [row.split('\t',1)]}


def predecessor_guard(source, live, binding):
    require=live._require; d=binding['transport_diagnostics']
    old_hash=d['previous_guard_sha256']
    if hashlib.sha256(source).hexdigest()==old_hash:return source
    require(live._ncaaf_future_guard_matches(source),'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    value=source.replace(live.NCAAF_FUTURE_BINDING_SHA256.encode(),d['previous_binding_sha256'].encode())
    value=value.replace(live.NCAAF_FUTURE_REVIEWED_GUARD_SHA256.encode(),d['previous_guard_identity'].encode())
    require(hashlib.sha256(value).hexdigest()==old_hash,'PREVIOUS_GUARD_LOGIC_CHANGED')
    return value


def previous_source(path,source,live,binding):
    d=binding['transport_diagnostics']
    if path==live.GUARD_PATH:return predecessor_guard(source,live,binding)
    if path==BINDING:
        previous=base64.b64decode(d['previous_binding_b64'],validate=True)
        live._require(hashlib.sha256(previous).hexdigest()==d['previous_binding_sha256'],'PREVIOUS_BINDING_CHANGED')
        if hashlib.sha256(source).hexdigest()==live.NCAAF_FUTURE_BINDING_SHA256:return previous
        return source
    r=d['source_reconstructions'].get(path)
    if r is None or hashlib.sha256(source).hexdigest()!=r['after_sha256']:return source
    value=source.decode()
    for p in reversed(r['reverse_patches']):value=value[:p['start']]+p['text']+value[p['end']:]
    result=value.encode()
    live._require(hashlib.sha256(result).hexdigest()==r['before_sha256'],'PREVIOUS_SOURCE_LOGIC_CHANGED')
    return result


def make_policy(g,binding,implementation):
    d=binding['transport_diagnostics'];before=blobs(g,d['base']);after=blobs(g,implementation)
    return dict(schema_version=80,policy_version='ncaaf-transport-diagnostics-v1',proposal_sha256=PROPOSAL,
        base_sha=d['base'],base_tree=d['base_tree'],implementation_commit=implementation,
        implementation_tree=g.git('rev-parse',implementation+'^{tree}'),
        implementation_changes={p:dict(before_blob=before.get(p),after_blob=after.get(p)) for p in d['paths']},
        unchanged_bindings={p:v for p,v in before.items() if p not in d['paths']})


def validate(g,manifest_path,base,binding):
    require=g._require;d=binding['transport_diagnostics']
    require(d['proposal_sha256']==PROPOSAL,'PROPOSAL_NOT_APPROVED')
    require(manifest_path.resolve()==(g.ROOT/g.MANIFEST_PATH).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not g.git('diff','--name-only') and not g.git('diff','--cached','--name-only'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    raw=g.git_bytes('show',d['base']+':'+g.MANIFEST_PATH)
    require(hashlib.sha256(raw).hexdigest()==d['manifest_sha256'],'BASELINE_CHANGED')
    require(base in (None,d['base'],json.loads(raw)['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(g.git('rev-parse',d['base']+'^{tree}')==d['base_tree'],'STARTING_MAIN_TREE_CHANGED')
    parent_headers=[v.split()[1] for v in g.git('cat-file','-p',d['base']).splitlines() if v.startswith('parent ')]
    require(parent_headers==d['ordered_base_parents'],'PREDECESSOR_PARENTS_CHANGED')
    previous_raw=g.git_bytes('show',d['base']+':'+BINDING)
    require(hashlib.sha256(previous_raw).hexdigest()==d['previous_binding_sha256']
        and base64.b64decode(d['previous_binding_b64'],validate=True)==previous_raw,'PREVIOUS_BINDING_CHANGED')
    previous=json.loads(previous_raw)
    require(set(binding)==set(previous)|{'transport_diagnostics'}
        and all(binding[k]==v for k,v in previous.items() if k!='reviewed_blobs'),'INHERITED_BINDINGS_CHANGED')
    reviewed=dict(previous['reviewed_blobs']);reviewed['scripts/ncaaf_future_spread_scope.py']=d['reviewed_blobs']['scripts/ncaaf_future_spread_scope.py']
    require(binding['reviewed_blobs']==reviewed,'INHERITED_REVIEWED_BLOBS_CHANGED')
    binding_raw=g.git_bytes('show','HEAD:'+BINDING)
    require(json.loads(binding_raw)==binding and hashlib.sha256(binding_raw).hexdigest()==g.NCAAF_FUTURE_BINDING_SHA256,'REVIEWED_BINDINGS_CHANGED')
    prior_guard=g.git_bytes('show',d['base']+':'+g.GUARD_PATH)
    require(hashlib.sha256(prior_guard).hexdigest()==d['previous_guard_sha256'],'PREVIOUS_GUARD_CHANGED')
    raw_policy=g.git_bytes('show','HEAD:'+POLICY);policy=json.loads(raw_policy);impl=policy['implementation_commit']
    require(g.git('show','-s','--format=%P',impl).split()==[d['base']],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(d['base'],POLICY) and not g.exists_at(impl,POLICY),'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff','--name-only',d['base'],impl).splitlines())==set(d['paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(g,binding,impl),'EXACT_POLICY_BINDINGS_CHANGED')
    impl_blobs=blobs(g,impl)
    require(set(d['reviewed_blobs'])==set(d['paths'])-{g.GUARD_PATH,BINDING},'REVIEWED_BINDING_SET_CHANGED')
    require(all(impl_blobs.get(p)==v for p,v in d['reviewed_blobs'].items()),'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    successor=g.git_bytes('show',impl+':'+g.GUARD_PATH)
    require(predecessor_guard(successor,g,binding)==prior_guard,'PREVIOUS_GUARD_LOGIC_CHANGED')
    require(len(successor)==8388490<=8*1024*1024,'SUCCESSOR_GUARD_LIMIT_CHANGED')
    for path,r in d['source_reconstructions'].items():
        source=g.git_bytes('show',impl+':'+path)
        require(hashlib.sha256(source).hexdigest()==r['after_sha256'],'REVIEWED_SOURCE_CHANGED')
        require(previous_source(path,source,g,binding)==g.git_bytes('show',d['base']+':'+path),'PREVIOUS_SOURCE_LOGIC_CHANGED')
    parents=g.git('show','-s','--format=%P','HEAD').split();candidate=g.git('rev-parse','HEAD')
    if len(parents)==2:
        require(parents[0]==d['base'],'CI_BASE_PARENT_NOT_APPROVED');candidate=parents[1]
        require(g.git('rev-parse','HEAD^{tree}')==g.git('rev-parse',candidate+'^{tree}'),'CI_MERGE_TREE_CHANGED')
    require(g.git('show','-s','--format=%P',candidate).split()==[impl],'CANDIDATE_NOT_POLICY_SEAL')
    require(g.git('diff','--name-status',impl,candidate).splitlines()==['A\t'+POLICY],'SEAL_CHANGE_SET_NOT_APPROVED')
    head=blobs(g,'HEAD')
    require(all(head.get(p)==v for p,v in policy['unchanged_bindings'].items()),'IMMUTABLE_FILE_CHANGED')
    require(set(head)==set(policy['unchanged_bindings'])|set(d['paths'])|{POLICY},'UNREVIEWED_PATH')
    for path,expected in head.items():
        content=(g.ROOT/path).read_bytes()
        def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(blob(content)==expected or blob(content.replace(b'\r\n',b'\n'))==expected,'UNAUTHORIZED_CHECKOUT_CHANGE')
    for path in g.ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(g.ROOT).parts:continue
        rel=path.relative_to(g.ROOT).as_posix()
        require(path.name not in {'sitecustomize.py','usercustomize.py'} and path.suffix!='.pth','PROTECTED_RUNTIME_SHADOWING_RISK')
        require(not any(rel.endswith('/'+p) for p in json.loads(raw)['protected_files']),'PROTECTED_RUNTIME_SHADOWING_RISK')
    return policy


def run(g,manifest_path,base,binding):
    try:policy=validate(g,manifest_path,base,binding)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.CalledProcessError) as exc:
        return 1,dict(schema_version=80,status='FAIL',policy_valid=False,reason_codes=[str(exc)])
    _,original=g.run(manifest_path,base)
    return 0,dict(schema_version=80,status='PASS',policy_valid=True,reason_codes=[],
        base_sha=binding['transport_diagnostics']['base'],head_sha=g.git('rev-parse','HEAD'),
        original_guard_report=original,approved_integration_changes=policy['implementation_changes'],
        unchanged_bindings=policy['unchanged_bindings'],protected_changes=[],existing_test_changes=[],
        approved_fixture_inputs='Exact predecessor source inputs only; assertions and live decisions unchanged',
        scope='Opt-in sanitized NCAAF phase observations. No timeout, retry, model, acquisition-authority or workflow changes.')
