"""Exact test-only public polling successor; no production or operational authority."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE='9c98c958725d06a6f9f9e0dbe90c0a76597cedb9'
TREE='fa949eeb4589638c5bd5a2429866816cbe6856f7'
POLICY='docs/research/launch-scope-policy-ncaaf-filtered-history-v1.json'
AUTHORIZATION='docs/research/public-refresh-harness-repair-v1.md'
PYTHON_TEST='tests/test_public_refresh.py'
JS_TEST='tests/public_refresh.cjs'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)

def text(root,*args):return git(root,*args).decode().strip()

def make_policy(root,b,implementation,binding_sha):
    c=b['public_refresh_harness']
    return dict(version='public-refresh-harness-seal-v1',base=c['base'],base_tree=c['base_tree'],
        authorization_record_sha256=c['authorization_record_sha256'],implementation_commit=implementation,
        implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),binding_sha256=binding_sha,
        implementation_paths=c['implementation_paths'])

def validate(root,b,comparison_base,manifest,*,binding_sha):
    c=b['public_refresh_harness'];require(set(c)==set('base base_tree ordered_parents authorization_record_sha256 implementation_paths inherited_policy_sha256 original_python_test_sha256 original_js_assertion_body_sha256'.split()),'PUBLIC_REFRESH_SCOPE_CHANGED')
    require(c['base']==BASE and c['base_tree']==TREE and text(root,'rev-parse',BASE+'^{tree}')==TREE
        and text(root,'show','-s','--format=%P',BASE).split()==c['ordered_parents']
        and c['ordered_parents']==['91f1d0c4f82a568d4c0142e14f1ec8f2a32de64a','31ea66dd35542e11f0841524f45b7c89e1fc519a'],'PUBLIC_REFRESH_BASE_CHANGED')
    require(c['authorization_record_sha256']==b['components'][AUTHORIZATION],'PUBLIC_REFRESH_SCOPE_CHANGED')
    baseline=git(root,'show',BASE+':'+b['baseline_manifest'])
    require(sha(baseline)==b['baseline_manifest_sha256'],'PUBLIC_REFRESH_MANIFEST_CHANGED')
    require(manifest is None or Path(manifest).resolve()==(Path(root)/b['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(comparison_base in {None,BASE,b['base'],json.loads(baseline)['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only')
        and not text(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(policy==make_policy(root,b,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    lineage=text(root,'rev-list','--first-parent',BASE+'..'+implementation).splitlines()
    require(1<=len(lineage)<=8 and text(root,'merge-base',BASE,implementation)==BASE,'IMPLEMENTATION_PARENT_NOT_APPROVED')
    for i,commit in enumerate(lineage):
        parent=lineage[i+1] if i+1<len(lineage) else BASE
        require(text(root,'show','-s','--format=%P',commit).split()==[parent]
            and set(text(root,'diff','--name-only',parent,commit).splitlines())<=set(c['implementation_paths'])|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    inherited=git(root,'show',BASE+':'+POLICY)
    require(sha(inherited)==c['inherited_policy_sha256'] and git(root,'show',implementation+':'+POLICY)==inherited,'INHERITED_POLICY_CHANGED')
    changed=set(text(root,'diff','--name-only',BASE,implementation).splitlines())
    require(changed==set(c['implementation_paths']) and len(changed)==len(c['implementation_paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    original=git(root,'show',BASE+':'+PYTHON_TEST)
    require(sha(original)==c['original_python_test_sha256']=='028f6b5930305fd5afa52dce1b7d90fc18c808df24409241df3398650ef8027c','PUBLIC_REFRESH_PRIOR_ASSERTIONS_CHANGED')
    before=git(root,'show',BASE+':'+JS_TEST).decode();body=before.split('(async()=>{',1)[1].split('})().catch(',1)[0]
    current=git(root,'show',implementation+':'+JS_TEST).decode()
    require(sha(body.encode())==c['original_js_assertion_body_sha256'] and current.count(body)==1,'PUBLIC_REFRESH_PRIOR_ASSERTIONS_CHANGED')
    functions=[n.name for n in ast.parse(git(root,'show',implementation+':'+PYTHON_TEST)).body if isinstance(n,ast.FunctionDef)]
    require('test_public_refresh_transport' in functions,'PUBLIC_REFRESH_PRIOR_ASSERTIONS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:seal=head
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
        and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(not text(root,'diff','--name-only',seal,head),'TESTED_TREE_CHANGED')
    return dict(schema_version=89,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(changed|{POLICY}),protected_changes=[],existing_test_changes=[],
        approved_test_helper_changes=[JS_TEST,PYTHON_TEST],approved_test_body_changes=[PYTHON_TEST+'::test_public_refresh_transport'],
        preserved_original_javascript_assertion_body_sha256=c['original_js_assertion_body_sha256'],accepted=False,inference=False)
