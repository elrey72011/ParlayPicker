"""Exact retained-history/prospective-discovery successor; no acquisition authority."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

BASE='311b924ffe4a4732738bafd2d24222c7a0b40459'
TREE='8a5b1066fcb81ce79da3791e982b9731a192c055'
POLICY='docs/research/launch-scope-policy-ncaaf-filtered-history-v1.json'
AUTHORIZATION='docs/research/ncaaf-retained-discovery-compatibility-v1.md'
PREPARED_PATCH='21220627c68436076d2dd1949836a786c463114a4253c1375241c87ac0010b58'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)
def text(root,*args):return git(root,*args).decode().strip()


def make_policy(root,b,implementation,binding_sha):
    c=b['retained_discovery_compatibility']
    return dict(version='ncaaf-retained-discovery-seal-v1',base=c['base'],base_tree=c['base_tree'],
        authorization_record_sha256=c['authorization_record_sha256'],prepared_patch_sha256=PREPARED_PATCH,
        implementation_commit=implementation,implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),
        binding_sha256=binding_sha,implementation_paths=c['implementation_paths'])


def validate(root,b,comparison_base,manifest,*,binding_sha):
    c=b['retained_discovery_compatibility']
    require(set(c)==set('base base_tree ordered_parents authorization_record_sha256 prepared_patch_sha256 implementation_paths inherited_policy_sha256'.split()),'RETAINED_DISCOVERY_SCOPE_CHANGED')
    require(c['base']==BASE and c['base_tree']==TREE and c['prepared_patch_sha256']==PREPARED_PATCH
        and text(root,'rev-parse',BASE+'^{tree}')==TREE
        and text(root,'show','-s','--format=%P',BASE).split()==c['ordered_parents']
        and c['ordered_parents']==['9c98c958725d06a6f9f9e0dbe90c0a76597cedb9','9ad14bed575a1713eced5d0d54774e69f898e11e'],'RETAINED_DISCOVERY_BASE_CHANGED')
    require(c['authorization_record_sha256']==b['components'][AUTHORIZATION],'RETAINED_DISCOVERY_SCOPE_CHANGED')
    original=git(root,'show',BASE+':'+b['baseline_manifest'])
    require(sha(original)==b['baseline_manifest_sha256'],'RETAINED_DISCOVERY_MANIFEST_CHANGED')
    require(manifest is None or Path(manifest).resolve()==(Path(root)/b['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(comparison_base in {None,BASE,b['base'],json.loads(original)['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only')
        and not text(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy=json.loads(git(root,'show','HEAD:'+POLICY));implementation=policy['implementation_commit']
    require(policy==make_policy(root,b,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    lineage=text(root,'rev-list','--first-parent',BASE+'..'+implementation).splitlines()
    require(1<=len(lineage)<=12 and text(root,'merge-base',BASE,implementation)==BASE,'IMPLEMENTATION_PARENT_NOT_APPROVED')
    for i,commit in enumerate(lineage):
        parent=lineage[i+1]if i+1<len(lineage)else BASE
        require(text(root,'show','-s','--format=%P',commit).split()==[parent]
            and set(text(root,'diff','--name-only',parent,commit).splitlines())<=set(c['implementation_paths'])|{POLICY},'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    inherited=git(root,'show',BASE+':'+POLICY)
    require(sha(inherited)==c['inherited_policy_sha256'] and git(root,'show',implementation+':'+POLICY)==inherited,'INHERITED_POLICY_CHANGED')
    changed=set(text(root,'diff','--name-only',BASE,implementation).splitlines())
    require(changed==set(c['implementation_paths']) and len(changed)==len(c['implementation_paths']),'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    preserved=0
    for path in sorted(p for p in changed if p.startswith('tests/') and text(root,'ls-tree',BASE,'--',p)):
        def bodies(raw):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name.startswith('test_')}
        old=bodies(git(root,'show',BASE+':'+path));current=bodies(git(root,'show',implementation+':'+path))
        require(all(current.get(k)==v for k,v in old.items()),'RETAINED_DISCOVERY_PRIOR_ASSERTIONS_CHANGED');preserved+=len(old)
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:seal=head
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
        and text(root,'diff','--name-status',implementation,seal)=='M\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    require(not text(root,'diff','--name-only',seal,head),'TESTED_TREE_CHANGED')
    return dict(schema_version=90,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(changed|{POLICY}),protected_changes=[],
        existing_test_changes=[],preserved_original_test_bodies=preserved,accepted=False,inference=False)
