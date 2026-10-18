"""Explicit synthetic target subjects. Never actual owner review or transport."""
from copy import deepcopy
import hashlib,importlib.util,json
from pathlib import Path
import pytest
from app_core import ncaaf_reviewed_target as target,ncaaf_staged_acquisition as stage
from app_core import ncaaf_owner_spread as owner,ncaaf_future_target as historical
from test_ncaaf_model_compatibility import synthetic
import ncaaf_reviewed_target_fixtures as f


@pytest.mark.parametrize('case',[0,1])
def test_distinct_target_plan_capture_and_actual_reader(synthetic,monkeypatch,tmp_path,case):
    f.select(monkeypatch,case)
    packet,row=f.fixture(synthetic,monkeypatch,tmp_path)
    assert owner.load(packet)==packet
    assert packet['payload']['alias_review']['target_manifest']['canonical_target']==f.CANONICAL
    assert packet['payload']['staged_binding']['payload']['prequote']['payload']['history_row']['home_yards_games']==3
    p=f.plan();assert stage.planning(p)['network_requests']==0
    with pytest.raises(ValueError):historical.validate_plan(p)


@pytest.mark.parametrize('attack',['review','review_subject','orientation','neutral','kickoff','source','source_rehash','cutoff','weeks','inventory','future_observation','expired_review','season','oversize'])
def test_manifest_mutations_and_rehashed_subjects_reject(monkeypatch,attack):
    f.select(monkeypatch,0);p=f.plan();v=p['payload'];m=v['target_evidence']
    if attack=='review':m['owner_review']=None
    elif attack=='review_subject':m['owner_review']['subject_sha256']='0'*64
    elif attack=='orientation':v['target']['home_school'],v['target']['away_school']=v['target']['away_school'],v['target']['home_school']
    elif attack=='neutral':v['target']['neutral_site']=True
    elif attack=='kickoff':v['target']['start_utc']='2026-10-21T23:30:00Z'
    elif attack.startswith('source'):
        m['references'][0]['facts']['home_school']='SYNTHETIC different university'
        if attack=='source_rehash':m['references'][0]['sha256']=stage.model.digest({k:x for k,x in m['references'][0].items() if k!='sha256'})
    elif attack=='cutoff':m['history_inventory']['cutoff_utc']='2026-10-14T23:30:00Z'
    elif attack=='weeks':m['history_inventory']['yardage_weeks']=[1,1,2]
    elif attack=='inventory':m['history_inventory']['qualifying_games'].pop()
    elif attack=='future_observation':m['references'][0]['observed_at']='2026-10-18T16:00:00Z'
    elif attack=='expired_review':m['owner_review']['reviewed_at']='2026-10-21T00:00:00Z'
    elif attack=='season':m['season']=2025
    else:m['references'][0]['excerpt']='X'*65536
    with pytest.raises(ValueError):stage.validate_plan(stage.seal(v),executable=True)


def test_retired_target_review_cannot_transfer_even_after_envelope_rehash(monkeypatch):
    f.select(monkeypatch,0);a=f.plan();old=deepcopy(a['payload']['target_evidence'])
    f.select(monkeypatch,1);b=f.plan();b['payload']['target_evidence']=old
    with pytest.raises(ValueError):stage.planning(stage.seal(b['payload']))


@pytest.mark.parametrize('field',['homeClassification','awayClassification','venue'])
def test_original_cfbd_target_facts_cannot_be_invented(synthetic,monkeypatch,tmp_path,field):
    import json
    f.select(monkeypatch,0);seed=tmp_path/'seed';seed.mkdir();p,_=f.fixture(synthetic,monkeypatch,seed)
    root=tmp_path/'candidate';root.mkdir();raw=f.bodies();games=json.loads(raw[0]);games[0][field]='SYNTHETIC conflicting original';raw[0]=stage.model.encode(games)
    bundle,_,_=f.capture(monkeypatch,root,raw=raw)
    m=f.pair._observation(p)['model']
    with pytest.raises(ValueError):f.evidence.prepare(bundle,model_packet=m,prepared_at='2026-10-17T18:01:00Z')


@pytest.mark.parametrize('attack',[None,'manifest_hash','configuration_hash','runtime','configuration','window'])
def test_operator_actual_preflight_binds_controls_checkout_and_runtime(monkeypatch,tmp_path,attack):
    spec=importlib.util.spec_from_file_location('SYNTHETIC_preflight_operator',Path(__file__).resolve().parents[1]/'scripts/ncaaf_reviewed_target_operator.py')
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    f.select(monkeypatch,0);p=f.plan();a,raw=f.stage_configuration(p,monkeypatch,tmp_path)
    paths={name:str(tmp_path/(name+'.json')) for name in ('plan','authorization','configuration')}
    for name,value in [('plan',p),('authorization',a)]:Path(paths[name]).write_bytes(stage.model.encode(value))
    m=dict(version=op.VERSION,**paths,custody_root=str(tmp_path),setup_cutoff='2026-10-17T17:55:00Z',
        latest_start='2026-10-17T18:14:00Z',history=None,prequote=None)
    c=json.loads(raw)['payload'];m['runtime']=target.runtime_receipt(c['runtime']['code_head'],stage.model.digest(m))
    c['runtime']=deepcopy(m['runtime']);c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
    if attack=='configuration':c['plan_sha256']='0'*64
    raw=stage.model.encode(stage.seal(c));Path(paths['configuration']).write_bytes(raw)
    if attack=='runtime':m['runtime']['code_head']='0'*40
    sealed=stage.model.encode(stage.seal(m));path=tmp_path/'operation.json';path.write_bytes(sealed)
    mh=hashlib.sha256(sealed).hexdigest();ch=hashlib.sha256(raw).hexdigest()
    if attack=='manifest_hash':mh='0'*64
    if attack=='configuration_hash':ch='0'*64
    clock='2026-10-17T18:14:01Z' if attack=='window' else '2026-10-17T18:00:01Z'
    if attack is None:
        actual,plan,control,root=op.preflight(path,mh,ch,at=clock)
        assert actual==m and plan==p and control==raw and root==tmp_path
    else:
        with pytest.raises(ValueError):op.preflight(path,mh,ch,at=clock)
    assert not list(tmp_path.glob('*.attempts.jsonl')) and not list(tmp_path.glob('*.operator-claim.json'))
