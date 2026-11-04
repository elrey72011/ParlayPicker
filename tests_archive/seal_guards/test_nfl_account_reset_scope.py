"""Authenticate actual successor/parent bytes; no operational authority."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts import nfl_account_reset_scope as scope
from scripts import ncaaf_filtered_history_scope as history

ROOT=Path(__file__).resolve().parents[1]


def test_exact_parent_binding_and_all_prior_source_bytes_are_preserved():
    binding=history.authenticate(ROOT)
    prior=scope.parent_binding(binding)
    assert scope.sha(prior)==scope.OLD_STAMP
    parent=json.loads(prior)
    record=binding['nfl_account_reset_successor']
    for path,row in record['historical_sources'].items():
        assert scope.sha(scope.record_bytes(row))==parent['components'][path]


def test_nested_predecessors_reconstruct_the_same_authenticated_generation():
    binding=history.authenticate(ROOT);parent=json.loads(scope.parent_binding(binding))
    for path in binding['nfl_account_reset_successor']['historical_sources']:
        now=history.physical(ROOT/path)
        before=scope.record_bytes(binding['nfl_account_reset_successor']['historical_sources'][path])
        assert scope.sha(before)==parent['components'][path]
        assert history.capacity_predecessor(binding,path,now)==history.capacity_predecessor(parent,path,before)
        changed=now+b'\n# SYNTHETIC unapproved edit\n'
        assert history.capacity_predecessor(binding,path,changed)==changed


@pytest.mark.parametrize('attack',['parent-length','component'])
def test_relabelled_or_modified_parent_metadata_rejects(attack):
    binding=copy.deepcopy(history.authenticate(ROOT))
    if attack=='parent-length':binding['nfl_account_reset_successor']['parent_binding']['bytes']+=1
    else:binding['components']['app/ui/nfl_private_workflow.py']='0'*64
    with pytest.raises(ValueError):scope.parent_binding(binding)


def test_final_clean_seal_and_unapproved_checkout_edit_rejection(tmp_path):
    destination=tmp_path/'SYNTHETIC-account-scope'
    shallow=scope.text(ROOT,'rev-parse','--is-shallow-repository')=='true'
    transport=['--no-local','--depth=1'] if shallow else ['--shared']
    cloned=subprocess.run(['git','clone','-c','core.longpaths=true',*transport,'--quiet',str(ROOT),str(destination)],capture_output=True,text=True)
    assert cloned.returncode==0,cloned.stderr
    binding=history.authenticate(destination)
    if shallow:
        # Shallow source authentication is supported. It must not manufacture
        # the missing ancestry needed for the separate full-history seal audit.
        with pytest.raises(subprocess.CalledProcessError):
            scope.validate(destination,binding,None,None,binding_sha=history._BINDING_SHA256)
    else:
        result=scope.validate(destination,binding,None,None,binding_sha=history._BINDING_SHA256)
        assert result['status']=='PASS' and result['accepted'] is False and result['inference'] is False
    changed=destination/'app/ui/nfl_private_workflow.py'
    changed.write_bytes(changed.read_bytes()+b'\n# SYNTHETIC unapproved edit\n')
    with pytest.raises(ValueError,match='NCAAF_FILTERED_SOURCE_CHANGED'):history.authenticate(destination)
    if not shallow:
        with pytest.raises(ValueError,match='UNAUTHORIZED_CHECKOUT_CHANGE'):
            scope.validate(destination,binding,None,None,binding_sha=history._BINDING_SHA256)
