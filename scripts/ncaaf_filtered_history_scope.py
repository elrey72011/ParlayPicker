"""Exact filtered-history successor and authenticated predecessor source views."""
import argparse
import ast
import base64
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import types
import zlib

PROPOSAL='4ba2ecdeaadc156070e3b402182a7eaa6cd4f6d4ea015dabb6aa2028a154f5a0'
AMENDMENT='162061ae1a7240680f915a9f16ea585cff70a619a69c49d99a0e46c14600a53b'
BASE='3e744bb3376c19688f38e1546c50443db7188ff8'
BASE_TREE='c3defb60b65607928d0527311747f15590eff854'
BINDING='docs/research/ncaaf-filtered-history-bindings-v1.json'
POLICY='docs/research/launch-scope-policy-ncaaf-filtered-history-v1.json'
_BINDING_SHA256='e03baea0d6b0ec9471c57310306041db5364ef768ac81d8f7e34220d9bd4cbd0'
NORMALIZED={'scripts/ncaaf_filtered_history_scope.py','scripts/ncaaf_private_refresh_scope.py','scripts/check_launch_change_scope.py'}
MAX_SOURCE=8*1024*1024
_PREDECESSOR_IMAGES={} # Bounded immutable bytes only; never validation decisions or modules.
_LEGACY_GUARD_IMAGES={}
_LEGACY_IMAGE_LIMIT=16*1024*1024
_NATIVE_B85DECODE=base64.b85decode
_NATIVE_B85CODE=getattr(base64.b85decode,'__code__',None)
_NATIVE_DECOMPRESS=zlib.decompressobj


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def physical(path):
    with io.FileIO(path,'r') as stream:
        size=os.fstat(stream.fileno()).st_size
        require(0<=size<=MAX_SOURCE,'NCAAF_FILTERED_SCOPE_LIMIT')
        raw=stream.read(size+1)
    require(len(raw)==size,'NCAAF_FILTERED_SOURCE_CHANGED')
    return raw.replace(b'\r\n',b'\n')


def normalized(relative,raw):
    if relative in NORMALIZED:
        require(raw.count(_BINDING_SHA256.encode())==1,'NCAAF_FILTERED_BINDING_CHANGED')
        return raw.replace(_BINDING_SHA256.encode(),b'0'*64)
    return raw


def authenticate(root):
    """Physical current bytes are checked on every call; no cached verdicts."""
    root=Path(root);raw=physical(root/BINDING)
    require(sha(raw)==_BINDING_SHA256,'NCAAF_FILTERED_BINDING_CHANGED')
    b=json.loads(raw)
    require(b['version']=='ncaaf-filtered-history-source-binding-v1' and b['proposal_sha256']==PROPOSAL
        and b.get('shared_boundary_amendment_sha256')==AMENDMENT
        and b['base']==BASE and b['base_tree']==BASE_TREE,'NCAAF_FILTERED_SCOPE_CHANGED')
    for relative,expected in b['components'].items():
        p=Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative,'NCAAF_FILTERED_SCOPE_CHANGED')
        require(sha(normalized(relative,physical(root/p)))==expected,'NCAAF_FILTERED_SOURCE_CHANGED')
    adapters=b.get('consumer_input_adapters',{})
    consumer='tests/test_ci_fixture_stability_scope.py'
    require(set(adapters)=={consumer},'NCAAF_FILTERED_CONSUMER_ADAPTER_CHANGED')
    adapter=adapters[consumer];edge=b['component_predecessors'][consumer]
    require(set(adapter)=={'generation','physical_sha256','reviewed_sha256','paths','calls'}
        and adapter['generation']=='reviewed-3e744bb-fixture-input-v1'
        and adapter['physical_sha256']==b['components'][consumer]==edge['after_sha256']
        and adapter['reviewed_sha256']==edge['before_sha256']
        and 1<=len(adapter['calls'])<=32 and 1<=len(adapter['paths'])<=256,
        'NCAAF_FILTERED_CONSUMER_ADAPTER_CHANGED')
    require(all(isinstance(p,str) and not Path(p).is_absolute() and '..' not in Path(p).parts and chr(92) not in p for p in adapter['paths'])
        and all(isinstance(c,list) and len(c)==2 and isinstance(c[0],str) and isinstance(c[1],int) and c[1]>0 for c in adapter['calls']),
        'NCAAF_FILTERED_CONSUMER_ADAPTER_CHANGED')
    return b



def predecessor_generation(root,binding,relative,expected):
    """One exact authenticated private-Refresh layer before this successor."""
    raw=physical(Path(root)/'docs/paid-launch/ncaaf-private-refresh-bindings-v1.json')
    require(sha(raw)==binding['predecessor_refresh_binding_sha256'],'NCAAF_FILTERED_PREDECESSOR_CHANGED')
    prior=json.loads(raw)['component_predecessors'].get(relative)
    if prior is None:return expected
    require(prior['before_sha256']==expected,'NCAAF_TARGET_RUNTIME_CONFLICT')
    return prior['after_sha256']


def predecessor_source(root,relative,raw):
    b=authenticate(root);record=b['component_predecessors'].get(relative)
    if record is None:return raw
    # An already older supplied view is left to its original strict reader.
    actual=sha(raw.replace(b'\r\n',b'\n'))
    if actual==record['before_sha256']:return raw
    # normalize() is strict for PHYSICAL current source. Supplied older views
    # need not carry this successor's stamp and belong to their strict reader.
    supplied=raw.replace(b'\r\n',b'\n')
    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:return raw
    if sha(normalized(relative,supplied))!=record['after_sha256']:return raw
    key=(_BINDING_SHA256,relative,record['after_sha256'],record['before_sha256'],record['before_bytes'],record['before_b85'])
    old=_PREDECESSOR_IMAGES.get(key)
    if old is None:
        z=zlib.decompressobj();old=z.decompress(base64.b85decode(record['before_b85']),MAX_SOURCE+1)
        require(z.eof and not z.unused_data,'NCAAF_FILTERED_PREDECESSOR_CHANGED')
        require(type(old) is bytes and len(old)==record['before_bytes']<=MAX_SOURCE
            and sha(old)==record['before_sha256'],'NCAAF_FILTERED_PREDECESSOR_CHANGED')
        if len(_PREDECESSOR_IMAGES)>=16:_PREDECESSOR_IMAGES.pop(next(iter(_PREDECESSOR_IMAGES),None),None)
        _PREDECESSOR_IMAGES[key]=old
    require(type(old) is bytes and len(old)==record['before_bytes']<=MAX_SOURCE
        and sha(old)==record['before_sha256'],'NCAAF_FILTERED_PREDECESSOR_CHANGED')
    return old


def advisory_profile_before(root,current):
    """Explicit authenticated510-row predecessor for current preservation checks."""
    b=authenticate(root);path='scripts/ci_test_file_costs_v1.json'
    require(sha(current)==b['components'][path],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    record=b['advisory_profile_before']
    require(set(record)=={'file_sha256','before_bytes','before_b85'}
        and record['file_sha256']=='3facc49a21124b179b5546d829389381122ef36e3cc70ee90650af5b658d112a'
        and type(record['before_bytes'])is int and 0<record['before_bytes']<=MAX_SOURCE
        and isinstance(record['before_b85'],str)and len(record['before_b85'])<=MAX_SOURCE*2,
        'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    decoder=zlib.decompressobj();prior=decoder.decompress(base64.b85decode(record['before_b85']),MAX_SOURCE+1)
    require(decoder.eof and not decoder.unused_data and len(prior)==record['before_bytes']
        and sha(prior)==record['file_sha256'],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    return prior



def fresh_advisory_profile_before(root,current):
    """Remove only the measured fresh-file row; authenticate fixed534-row bytes."""
    b=authenticate(root);path='scripts/ci_test_file_costs_v1.json'
    require(sha(current)==b['components'][path],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    record=b['fresh_advisory_addition']
    require(set(record)=={'before_profile_sha256','entry_sha256','source_key','source_sha256'}
        and record['before_profile_sha256']=='253a8551db604bd01c5304b7f74e61faaf9bb221ec5ab2b3ae6a2f05a1713bb4',
        'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    prior=json.loads(current)
    require(len(prior['files'])==535,'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    entry=prior['files'].pop();source=prior['sources'].pop(record['source_key'])
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    require(entry['path']=='tests/test_ncaaf_fresh_history.py'
        and entry['references']==[record['source_key']]
        and sha(canonical(entry))==record['entry_sha256']
        and sha(canonical(source))==record['source_sha256'],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    raw=(json.dumps(prior,sort_keys=True,indent=2)+'\n').encode()
    require(sha(raw)==record['before_profile_sha256'],'NCAAF_FILTERED_PROFILE_PARENT_CHANGED')
    return raw


def preserve_guard_reader(original,root):
    @functools.wraps(original)
    def checked(source,*args,**kwargs):
        # Authenticate the physical graph before considering an immutable image.
        selected=predecessor_source(root,'scripts/check_launch_change_scope.py',source)
        if (_NATIVE_B85CODE is None or base64.b85decode is not _NATIVE_B85DECODE or getattr(base64.b85decode,'__code__',None) is not _NATIVE_B85CODE
            or getattr(_NATIVE_B85DECODE,'__globals__',None) is not base64.__dict__
            or zlib.decompressobj is not _NATIVE_DECOMPRESS or not isinstance(zlib.decompressobj,types.BuiltinFunctionType)):
            return original(selected,*args,**kwargs)
        if len(args)!=1 or set(kwargs)-{'predecessor'} or kwargs.get('predecessor') is not None:
            return original(selected,*args,**kwargs)
        binding=args[0];record=binding.get('predecessor_guard') if type(binding)is dict else None
        if (type(record)is not dict or not {'sha256','bytes','source_b85'}<=set(record)
            or type(record['bytes'])is not int or not 0<record['bytes']<=MAX_SOURCE
            or not isinstance(record['sha256'],str) or not isinstance(record['source_b85'],str)
            or sha(selected)==record['sha256']):
            return original(selected,*args,**kwargs)
        identity=sha(json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        key=(_BINDING_SHA256,sha(selected),identity,record['sha256'],record['bytes'])
        options=dict(kwargs)
        if key in _LEGACY_GUARD_IMAGES:options['predecessor']=_LEGACY_GUARD_IMAGES[key]
        # The unchanged reader checks its current template, binding, length and
        # output hash even on a hit; malformed cached bytes must still reject.
        value=original(selected,*args,**options)
        if key not in _LEGACY_GUARD_IMAGES and type(value)is bytes and len(value)+4096<=_LEGACY_IMAGE_LIMIT:
            while _LEGACY_GUARD_IMAGES and (len(_LEGACY_GUARD_IMAGES)>=8
                or sum(len(v)+4096 for v in _LEGACY_GUARD_IMAGES.values())+len(value)+4096>_LEGACY_IMAGE_LIMIT):
                _LEGACY_GUARD_IMAGES.pop(next(iter(_LEGACY_GUARD_IMAGES)))
            _LEGACY_GUARD_IMAGES[key]=value
        return value
    return checked


def git(root,*args,raw=False):
    result=subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL)
    return result if raw else result.decode().strip()


def _historical_make_policy(root,b,implementation):
    return dict(version='ncaaf-filtered-history-seal-v1',proposal_sha256=PROPOSAL,shared_boundary_amendment_sha256=AMENDMENT,base=BASE,base_tree=BASE_TREE,
        implementation_commit=implementation,implementation_tree=git(root,'rev-parse',implementation+'^{tree}'),
        binding_sha256=_BINDING_SHA256,implementation_paths=b['implementation_paths'])


def _historical_validate(root,comparison_base=None,manifest=None):
    root=Path(root);b=authenticate(root)
    original=git(root,'show',BASE+':'+b['baseline_manifest'],raw=True)
    require(sha(original)==b['baseline_manifest_sha256'],'NCAAF_FILTERED_MANIFEST_CHANGED')
    require(manifest is None or Path(manifest).resolve()==(root/b['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    # The recorded old comparison base is permitted only after authentication.
    require(comparison_base in {None,BASE,json.loads(original)['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(git(root,'rev-parse',BASE+'^{tree}')==BASE_TREE,'NCAAF_FILTERED_BASE_CHANGED')
    require(not git(root,'diff','--name-only') and not git(root,'diff','--cached','--name-only')
        and not git(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy=json.loads(git(root,'show','HEAD:'+POLICY,raw=True));implementation=policy['implementation_commit']
    require(policy==make_policy(root,b,implementation),'EXACT_POLICY_BINDINGS_CHANGED')
    require(git(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    changed=set(git(root,'diff','--name-only',BASE,implementation).splitlines())
    require(changed==set(b['implementation_paths']) and len(changed)==len(b['implementation_paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    head=git(root,'rev-parse','HEAD');parents=git(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(git(root,'rev-parse',seal+'^{tree}')==git(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:seal=head
    require(git(root,'show','-s','--format=%P',seal).split()==[implementation],'SEAL_PARENT_CHANGED')
    require(git(root,'diff','--name-status',implementation,seal)=='A\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(git(root,'diff','--name-only',seal,head)=='','TESTED_TREE_CHANGED')
    return dict(schema_version=86,status='PASS',proposal_sha256=PROPOSAL,base=BASE,implementation=implementation,
        seal=seal,candidate=head,tree=git(root,'rev-parse',head+'^{tree}'),changed_files=sorted(changed|{POLICY}),
        protected_changes=[],existing_test_changes=[],accepted=False,inference=False)



def make_policy(root,b,implementation):
    continuation=b.get('history_continuation')
    if continuation is not None:
        return dict(version='ncaaf-history-continuation-seal-v6',authorization_record_sha256=continuation['authorization_record_sha256'],
            base=continuation['base'],base_tree=continuation['base_tree'],implementation_commit=implementation,
            implementation_tree=git(root,'rev-parse',implementation+'^{tree}'),binding_sha256=_BINDING_SHA256,
            implementation_paths=continuation['implementation_paths'])
    repair=b.get('ui_display_repair')
    if repair is None:return _historical_make_policy(root,b,implementation)
    return dict(version='ui-display-revalidation-seal-v1',amendment_sha256=repair['amendment_sha256'],
        base=repair['base'],base_tree=repair['base_tree'],implementation_commit=implementation,
        implementation_tree=git(root,'rev-parse',implementation+'^{tree}'),
        binding_sha256=_BINDING_SHA256,implementation_paths=repair['implementation_paths'])


def validate(root,comparison_base=None,manifest=None):
    root=Path(root);b=authenticate(root);repair=b.get('ui_display_repair')
    if b.get('history_continuation') is not None:return _validate_continuation(root,b,comparison_base,manifest)
    if repair is None:return _historical_validate(root,comparison_base,manifest)
    require(set(repair)=={'amendment_sha256','base','base_tree','reviewed_head','ordered_parents','implementation_paths','implementation_parent','predecessor_tree','inherited_policy_sha256'},'UI_SCOPE_CONTRACT_CHANGED')
    require(repair['base']=='795a2ea53a70ba8e96284a466404a6a42c848fa9'
        and repair['base_tree']=='85f25488a2330f8810ce3c320eba5a2d7d146802'
        and repair['reviewed_head']=='b711dca0febcb858233a7c72ea5c27685dc00716', 'UI_BASE_CHANGED')
    base=repair['base'];base_tree=repair['base_tree']
    require(git(root,'rev-parse',base+'^{tree}')==base_tree
        and git(root,'show','-s','--format=%P',base).split()==repair['ordered_parents']
        and git(root,'rev-parse',repair['reviewed_head']+'^{tree}')==base_tree,'UI_BASE_CHANGED')
    original=git(root,'show',BASE+':'+b['baseline_manifest'],raw=True)
    require(sha(original)==b['baseline_manifest_sha256'],'NCAAF_FILTERED_MANIFEST_CHANGED')
    require(manifest is None or Path(manifest).resolve()==(root/b['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(comparison_base in {None,base,BASE,json.loads(original)['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(not git(root,'diff','--name-only') and not git(root,'diff','--cached','--name-only')
        and not git(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy=json.loads(git(root,'show','HEAD:'+POLICY,raw=True));implementation=policy['implementation_commit']
    require(policy==make_policy(root,b,implementation),'EXACT_POLICY_BINDINGS_CHANGED')
    parent=repair['implementation_parent']
    require(parent=='13766dac2b02d99399dce37464e1e72e00784de7'
        and git(root,'rev-parse',parent+'^{tree}')==repair['predecessor_tree']
        and git(root,'merge-base',base,parent)==base
        and git(root,'show','-s','--format=%P',implementation).split()==[parent],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    inherited=git(root,'show',parent+':'+POLICY,raw=True)
    require(sha(inherited)==repair['inherited_policy_sha256']
        and git(root,'show',implementation+':'+POLICY,raw=True)==inherited,'INHERITED_POLICY_CHANGED')
    changed=set(git(root,'diff','--name-only',base,implementation).splitlines())
    # Only the exact authenticated prior policy is inherited by this linear continuation.
    changed.discard(POLICY)
    require(changed==set(repair['implementation_paths']) and len(changed)==len(repair['implementation_paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    head=git(root,'rev-parse','HEAD');parents=git(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents==[base,parents[1]],'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(git(root,'rev-parse',seal+'^{tree}')==git(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:seal=head
    require(git(root,'show','-s','--format=%P',seal).split()==[implementation],'SEAL_PARENT_CHANGED')
    require(git(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(git(root,'diff','--name-only',seal,head)=='','TESTED_TREE_CHANGED')
    return dict(schema_version=87,status='PASS',amendment_sha256=repair['amendment_sha256'],base=base,
        implementation=implementation,seal=seal,candidate=head,tree=git(root,'rev-parse',head+'^{tree}'),
        changed_files=sorted(changed|{POLICY}),protected_changes=[],existing_test_changes=[],accepted=False,inference=False)


def _validate_continuation(root,b,comparison_base,manifest):
    c=b['history_continuation'];base=c['base']
    require(set(c)=={'authorization_record_sha256','base','base_tree','ordered_parents','implementation_paths','inherited_policy_sha256'},'CONTINUATION_SCOPE_CHANGED')
    require(c['authorization_record_sha256']==b['components']['docs/research/ncaaf-history-continuation-v6.md'],'CONTINUATION_SCOPE_CHANGED')
    require(base=='91f1d0c4f82a568d4c0142e14f1ec8f2a32de64a'
        and c['base_tree']=='fbb36f8cce2e1bce227c672b34711aaa1fa31115'
        and git(root,'rev-parse',base+'^{tree}')==c['base_tree']
        and git(root,'show','-s','--format=%P',base).split()==c['ordered_parents'],'CONTINUATION_BASE_CHANGED')
    original=git(root,'show',BASE+':'+b['baseline_manifest'],raw=True)
    require(sha(original)==b['baseline_manifest_sha256'],'NCAAF_FILTERED_MANIFEST_CHANGED')
    require(manifest is None or Path(manifest).resolve()==(root/b['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(comparison_base in {None,base,BASE,json.loads(original)['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(not git(root,'diff','--name-only') and not git(root,'diff','--cached','--name-only')
        and not git(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy=json.loads(git(root,'show','HEAD:'+POLICY,raw=True));implementation=policy['implementation_commit']
    require(policy==make_policy(root,b,implementation),'EXACT_POLICY_BINDINGS_CHANGED')
    require(git(root,'merge-base',base,implementation)==base,'IMPLEMENTATION_PARENT_NOT_APPROVED')
    lineage=git(root,'rev-list','--first-parent',base+'..'+implementation).splitlines()
    require(1<=len(lineage)<=32,'IMPLEMENTATION_PARENT_NOT_APPROVED')
    for index,commit in enumerate(lineage):
        expected_parent=lineage[index+1] if index+1<len(lineage) else base
        require(git(root,'show','-s','--format=%P',commit).split()==[expected_parent], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
        # Local qualification may preserve an earlier policy-only seal as an
        # ancestor. The final implementation must still restore the exact base
        # policy, and the tested head must have the one exact final seal below.
        require(set(git(root,'diff','--name-only',expected_parent,commit).splitlines())<=set(c['implementation_paths'])|{POLICY}, 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    inherited=git(root,'show',base+':'+POLICY,raw=True)
    require(sha(inherited)==c['inherited_policy_sha256'] and git(root,'show',implementation+':'+POLICY,raw=True)==inherited,'INHERITED_POLICY_CHANGED')
    changed=set(git(root,'diff','--name-only',base,implementation).splitlines())
    require(changed==set(c['implementation_paths']) and len(changed)==len(c['implementation_paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    head=git(root,'rev-parse','HEAD');parents=git(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents==[base,parents[1]],'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(git(root,'rev-parse',seal+'^{tree}')==git(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:seal=head
    require(git(root,'show','-s','--format=%P',seal).split()==[implementation],'SEAL_PARENT_CHANGED')
    require(git(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(not git(root,'diff','--name-only',seal,head),'TESTED_TREE_CHANGED')
    test_report=_preserved_test_report(root,base,implementation,changed)
    return dict(schema_version=88,status='PASS',base=base,implementation=implementation,seal=seal,candidate=head,
        tree=git(root,'rev-parse',head+'^{tree}'),changed_files=sorted(changed|{POLICY}),protected_changes=[],
        **test_report,
        accepted=False,inference=False)


def _preserved_test_report(root,base,implementation,changed):
    """Legacy violation field stays empty only after exact approved validation.

    The caller already authenticates the full physical source graph, approved
    change set and seal. Expose helper edits separately, then also verify every
    original test function against the authenticated base, never candidate-
    derived expectations. No assertion can disappear behind an empty report.
    """
    existing=sorted(p for p in changed if p.startswith('tests/') and git(root,'ls-tree',base,'--',p))
    preserved=0
    def tests(raw):
        tree=ast.parse(raw)
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')}
    for path in existing:
        original=tests(git(root,'show',base+':'+path,raw=True))
        current=tests(git(root,'show',implementation+':'+path,raw=True))
        require(all(current.get(name)==body for name,body in original.items()),'CONTINUATION_PRIOR_ASSERTIONS_CHANGED')
        preserved+=len(original)
    return dict(existing_test_changes=[],approved_test_helper_changes=existing,preserved_original_test_bodies=preserved)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base');parser.add_argument('--manifest',type=Path);parser.add_argument('--json-output',type=Path)
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[1]
    try:result=validate(root,args.base,args.manifest);code=0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError):
        result=dict(schema_version=86,status='ERROR',reason_codes=['NCAAF_FILTERED_SCOPE_REJECTED']);code=2
    raw=json.dumps(result,sort_keys=True,indent=2)
    if args.json_output:args.json_output.write_text(raw+'\n',encoding='utf-8')
    print(raw);return code


if __name__=='__main__':raise SystemExit(main())
