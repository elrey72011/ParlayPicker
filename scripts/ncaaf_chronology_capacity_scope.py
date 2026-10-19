"""Exact fixture-copy successor; immutable predecessor and live checkout checks."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

POLICY='docs/paid-launch/launch-scope-policy-ncaaf-chronology-capacity-v1.json'
BINDING='docs/paid-launch/ncaaf-reviewed-target-binding-v1.json'
PROPOSAL='0efcd102474e3afd8d6aee1a14293ce0e828eef958e2ac1683d7ea8673c8267a'


def blobs(g,rev):
    return {p:m.split()[2] for row in g.git('ls-tree','-r',rev).splitlines() for m,p in [row.split('\t',1)]}


def predecessor_guard(source, live, binding):
    require=live._require; d=binding['chronology_capacity']
    old_hash=d['previous_guard_sha256']
    if hashlib.sha256(source).hexdigest()==old_hash:return source
    import re
    identity=re.search(rb"NCAAF_FUTURE_REVIEWED_GUARD_SHA256='([0-9a-f]{64})'",source)
    current_binding=re.search(rb"_RT_BINDING_SHA256='([0-9a-f]{64})'",source)
    require(identity is not None and current_binding is not None,'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    identity=identity.group(1);current_binding=current_binding.group(1)
    require(len(source)==8388490 and source.count(identity)==2
        and hashlib.sha256(source.replace(identity,b'0'*64)).hexdigest()==identity.decode(),
        'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    expected=hashlib.sha256((json.dumps(binding,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    require(current_binding.decode()==expected,'REVIEWED_BINDINGS_CHANGED')
    value=source.replace(current_binding,d['previous_binding_sha256'].encode())
    value=value.replace(identity,d['previous_guard_identity'].encode())
    require(hashlib.sha256(value).hexdigest()==old_hash,'PREVIOUS_GUARD_LOGIC_CHANGED')
    return value


def previous_source(path,source,live,binding):
    d=binding['chronology_capacity']
    if path==live.GUARD_PATH:return predecessor_guard(source,live,binding)
    if path==BINDING:
        previous=base64.b64decode(d['previous_binding_b64'],validate=True)
        live._require(hashlib.sha256(previous).hexdigest()==d['previous_binding_sha256'],'PREVIOUS_BINDING_CHANGED')
        if hashlib.sha256(source).hexdigest()==hashlib.sha256((json.dumps(binding,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest():return previous
        return source
    r=d['source_reconstructions'].get(path)
    if r is None or hashlib.sha256(source).hexdigest()!=r['after_sha256']:return source
    value=source.decode()
    for p in reversed(r['reverse_patches']):value=value[:p['start']]+p['text']+value[p['end']:]
    result=value.encode()
    live._require(hashlib.sha256(result).hexdigest()==r['before_sha256'],'PREVIOUS_SOURCE_LOGIC_CHANGED')
    return result


def make_policy(g,binding,implementation):
    d=binding['chronology_capacity'];before=blobs(g,d['base']);after=blobs(g,implementation)
    return dict(schema_version=84,policy_version='ncaaf-chronology-capacity-v1',proposal_sha256=PROPOSAL,
        base_sha=d['base'],base_tree=d['base_tree'],implementation_commit=implementation,
        implementation_tree=g.git('rev-parse',implementation+'^{tree}'),
        implementation_changes={p:dict(before_blob=before.get(p),after_blob=after.get(p)) for p in d['paths']},
        unchanged_bindings={p:v for p,v in before.items() if p not in d['paths']})


def validate(g,manifest_path,base,binding):
    require=g._require;d=binding['chronology_capacity']
    require(d['proposal_sha256']==PROPOSAL,'PROPOSAL_NOT_APPROVED')
    require(manifest_path.resolve()==(g.ROOT/g.MANIFEST_PATH).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not g.git('diff','--name-only') and not g.git('diff','--cached','--name-only'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    raw=g.git_bytes('show',d['base']+':'+g.MANIFEST_PATH)
    require(hashlib.sha256(raw).hexdigest()==d['manifest_sha256'],'BASELINE_CHANGED')
    require(base in (None,d['base'],json.loads(raw)['base_sha']),'COMPARISON_BASE_NOT_APPROVED')
    require(g.git('rev-parse',d['base']+'^{tree}')==d['base_tree'],'STARTING_MAIN_TREE_CHANGED')
    parent_headers=[v.split()[1] for v in g.git('cat-file','-p',d['base']).splitlines() if v.startswith('parent ')]
    require(parent_headers==d['ordered_base_parents'],'PREDECESSOR_PARENTS_CHANGED')
    require(g.git('rev-parse',d['predecessor']+'^{tree}')==d['predecessor_tree'],'PREDECESSOR_TREE_CHANGED')
    prior_parents=[v.split()[1] for v in g.git('cat-file','-p',d['predecessor']).splitlines() if v.startswith('parent ')]
    require(prior_parents==d['predecessor_parents'],'PREDECESSOR_PARENTS_CHANGED')
    previous_raw=g.git_bytes('show',d['predecessor']+':'+BINDING)
    require(hashlib.sha256(previous_raw).hexdigest()==d['previous_binding_sha256']
        and base64.b64decode(d['previous_binding_b64'],validate=True)==previous_raw,'PREVIOUS_BINDING_CHANGED')
    previous=json.loads(previous_raw)
    require(set(binding)==set(previous)|{'chronology_capacity'}
        and all(binding[k]==v for k,v in previous.items()
                if k not in {'reviewed_blobs','source_reconstructions'}),'INHERITED_BINDINGS_CHANGED')
    reviewed=dict(previous['reviewed_blobs'])
    reviewed['scripts/ncaaf_reviewed_target_scope.py']=d['reviewed_blobs']['scripts/ncaaf_reviewed_target_scope.py']
    require(binding['reviewed_blobs']==reviewed,'INHERITED_REVIEWED_BLOBS_CHANGED')
    records=dict(previous['source_reconstructions'])
    for path,row in d['source_reconstructions'].items():
        record=binding['source_reconstructions'][path]
        require(record['after_sha256']==row['after_sha256'],'RECONSTRUCTION_SOURCE_CHANGED')
        old=g.git_bytes('show',d['predecessor']+':'+path)
        previous_row=previous['source_reconstructions'].get(path)
        if previous_row:
            value=old.decode()
            require(hashlib.sha256(old).hexdigest()==previous_row['after_sha256'],'PREVIOUS_SOURCE_CHANGED')
            for edit in reversed(previous_row['reverse_patches']):value=value[:edit['start']]+edit['text']+value[edit['end']:]
            old=value.encode()
            require(hashlib.sha256(old).hexdigest()==previous_row['before_sha256'],'PREVIOUS_SOURCE_LOGIC_CHANGED')
        require(record['before_sha256']==hashlib.sha256(old).hexdigest(),'INHERITED_RECONSTRUCTION_CHANGED')
        current=g.git_bytes('show','HEAD:'+path).decode()
        for edit in reversed(record['reverse_patches']):current=current[:edit['start']]+edit['text']+current[edit['end']:]
        require(current.encode()==old,'INHERITED_RECONSTRUCTION_CHANGED')
        records[path]=record
    require(binding['source_reconstructions']==records,'INHERITED_RECONSTRUCTIONS_CHANGED')
    binding_raw=g.git_bytes('show','HEAD:'+BINDING)
    expected=(json.dumps(binding,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(binding_raw==expected,'REVIEWED_BINDINGS_CHANGED')
    prior_guard=g.git_bytes('show',d['predecessor']+':'+g.GUARD_PATH)
    require(hashlib.sha256(prior_guard).hexdigest()==d['previous_guard_sha256'],'PREVIOUS_GUARD_CHANGED')
    raw_policy=g.git_bytes('show','HEAD:'+POLICY);policy=json.loads(raw_policy);impl=policy['implementation_commit']
    require(g.git('show','-s','--format=%P',impl).split()==[d['predecessor']],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(d['predecessor'],POLICY) and not g.exists_at(impl,POLICY),'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff','--name-only',d['base'],impl).splitlines())==set(d['paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(g,binding,impl),'EXACT_POLICY_BINDINGS_CHANGED')
    impl_blobs=blobs(g,impl)
    require(set(d['reviewed_blobs'])==set(d['paths'])-{g.GUARD_PATH,BINDING},'REVIEWED_BINDING_SET_CHANGED')
    require(all(impl_blobs.get(p)==v for p,v in d['reviewed_blobs'].items()),'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    predecessor_blobs=blobs(g,d['predecessor'])
    require(set(g.git('diff','--name-only',d['predecessor'],impl).splitlines())==set(d['new_paths']),'CAPACITY_CHANGE_SET_NOT_APPROVED')
    require(all(impl_blobs.get(path)==value for path,value in predecessor_blobs.items() if path not in d['new_paths']),
        'VALIDATED_FIXTURE_CORRECTION_CHANGED')
    successor=g.git_bytes('show',impl+':'+g.GUARD_PATH)
    require(predecessor_guard(successor,g,binding)==prior_guard,'PREVIOUS_GUARD_LOGIC_CHANGED')
    require(len(successor)==8388490<=8*1024*1024,'SUCCESSOR_GUARD_LIMIT_CHANGED')
    for path,r in d['source_reconstructions'].items():
        source=g.git_bytes('show',impl+':'+path)
        require(hashlib.sha256(source).hexdigest()==r['after_sha256'],'REVIEWED_SOURCE_CHANGED')
        require(previous_source(path,source,g,binding)==g.git_bytes('show',d['predecessor']+':'+path),'PREVIOUS_SOURCE_LOGIC_CHANGED')
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
        return 1,dict(schema_version=84,status='FAIL',policy_valid=False,reason_codes=[str(exc)])
    _,original=g.run(manifest_path,base)
    return 0,dict(schema_version=84,status='PASS',policy_valid=True,reason_codes=[],
        base_sha=binding['chronology_capacity']['base'],head_sha=g.git('rev-parse','HEAD'),
        original_guard_report=original,approved_integration_changes=policy['implementation_changes'],
        unchanged_bindings=policy['unchanged_bindings'],protected_changes=[],existing_test_changes=[],
        approved_fixture_inputs='Exact predecessor source inputs only; assertions and live decisions unchanged',
        scope='Exact eleven-shard current cardinality; historical assertions and validated fixture correction retained. Scheduler, profile, timeouts and production unchanged.',
        approved_capacity_changes=['Application matrix/reconciliation nine to eleven; exact current cardinality assertions; historical six/nine inputs unchanged'])
