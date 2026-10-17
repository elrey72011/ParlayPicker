"""Synthetic named-target successor; original stage contracts remain distinct."""
import base64
from copy import deepcopy
import json
import pytest
from app_core import ncaaf_future_target as target,ncaaf_future_staged as evidence,ncaaf_staged_acquisition as stage
from app_core import ncaaf_model_compatibility as model
from test_ncaaf_model_compatibility import synthetic
from ncaaf_future_pair_fixtures import plan,capture,fixture,Clock,stage_configuration


def test_named_history_planning_needs_no_future_provider_id_side_or_quote():
    p=plan();result=stage.planning(p)
    assert len(result['requests'])==4 and result['network_requests']==result['credential_loads']==result['storage_operations']==0
    assert p['payload']['target']['provider_event_id'] is None and p['payload']['identity_review'] is None
    assert p['payload']['parent_history_sha256'] is None and p['payload']['prequote_sha256'] is None
    assert all('side' not in r['params'] for r in result['requests'])


def test_named_history_and_later_quote_actual_original_readback(synthetic,monkeypatch,tmp_path):
    packet,row=fixture(synthetic,monkeypatch,tmp_path)
    b=packet['payload']['staged_binding']['payload'];h=b['history'];q=b['quote']
    assert stage.read_bundle(h)['attempts']==4 and stage.read_bundle(q)['attempts']==1
    assert stage.verify_journal(tmp_path,h) and stage.verify_journal(tmp_path,q)
    assert q['payload']['prior_original_bytes']==h['payload']['original_bytes']
    assert q['payload']['prior_active_seconds']==h['payload']['active_seconds']
    assert b['prequote']['payload']['event']['provider_event_id'] is None
    assert b['prequote']['payload']['event']['home_id']==111
    assert b['prequote']['payload']['publisher_availability']=='UNKNOWN'
    assert packet['payload']['alias_review']['discovery']['bundle']['payload']['identifiers']['state']=='NO_EXACT_DESCRIPTOR_MATCH'
    assert not h['payload']['accepted'] and not q['payload']['inference']


@pytest.mark.parametrize('attack',['target','orientation','kickoff','review_missing','review_subject','source','source_rehashed','future_id','side','limits'])
def test_fresh_history_bound_scope_rejects_before_dispatch(monkeypatch,tmp_path,attack):
    p=plan();v=p['payload']
    if attack=='target':v['target']['home_school']='Tulane'
    elif attack=='orientation':v['target']['home_school'],v['target']['away_school']=v['target']['away_school'],v['target']['home_school']
    elif attack=='kickoff':v['target']['start_utc']='2026-10-18T04:00:00Z'
    elif attack=='review_missing':v['target_evidence']['owner_review']=None
    elif attack=='review_subject':v['target_evidence']['owner_review']['subject_sha256']='0'*64
    elif attack.startswith('source'):
        r=v['target_evidence']['references'][0];r['excerpt']='Altered source excerpt'
        if attack=='source_rehashed':r['sha256']=model.digest({k:x for k,x in r.items() if k!='sha256'})
    elif attack=='future_id':v['target']['provider_event_id']='INVENTED'
    elif attack=='side':v['selected_side']='home'
    else:v['limits']['shared_active_seconds']=61
    p=stage.seal(v)
    with pytest.raises((ValueError,TypeError,KeyError)):
        stage.planning(p)
    assert not list(tmp_path.iterdir())


def test_old_staged_contract_still_rejects_new_target():
    from test_ncaaf_staged_acquisition import plan as old
    p=old();p['payload']['target'].update(home_team='Oregon',away_team='Nebraska',start_utc=target.CANONICAL['start_utc'])
    with pytest.raises(ValueError,match='NCAAF_STAGED_TARGET'):stage.planning(stage.seal(p['payload']))


def test_actual_cli_synthetic_history_execute_and_durable_readback(monkeypatch,tmp_path,capsys):
    from scripts import ncaaf_staged_acquisition as cli
    from ncaaf_future_pair_fixtures import bodies
    from test_ncaaf_pilot import Stream
    p=plan();a,config=stage_configuration(p,monkeypatch,tmp_path)
    sent=[];raws=bodies()
    def transport(req,limits,deadline,mono):
        sent.append(deepcopy(req));return Stream(raws[len(sent)-1])
    # Explicit synthetic injection at the same CLI boundary. The independent
    # test below proves the operational runtime rejects the blocked environment.
    monkeypatch.setattr(cli,'verify_runtime',lambda:None)
    monkeypatch.setattr(cli.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(cli.getpass,'getpass',lambda *a:'SYNTHETIC-NO-REAL-CREDENTIAL')
    class SyntheticTransport:
        def __call__(self,*a):return transport(*a)
    monkeypatch.setattr(stage.transport,'HttpsTransport',lambda selected:SyntheticTransport())
    monkeypatch.setattr(stage.transport,'utc',lambda:'2026-10-17T18:00:01Z')
    for name,value in [('plan.json',model.encode(p)),('authorization.json',model.encode(a)),('control.json',config)]:
        (tmp_path/name).write_bytes(value)
    assert cli.main(['plan',str(tmp_path/'plan.json')])==0
    assert json.loads(capsys.readouterr().out)['network_requests']==0
    assert cli.main(['execute',str(tmp_path/'plan.json'),'--authorization',str(tmp_path/'authorization.json'),
        '--controlled-bindings',str(tmp_path/'control.json'),'--bindings-sha256',__import__('hashlib').sha256(config).hexdigest(),
        '--custody-root',str(tmp_path)])==0
    assert json.loads(capsys.readouterr().out)['status']=='CAPTURED_UNREVIEWED' and sent==p['payload']['requests']
    b=json.loads((tmp_path/(p['sha256']+'.stage.json')).read_bytes())
    assert stage.read_bundle(b)['attempts']==4 and stage.verify_journal(tmp_path,b)
    assert [base64.b64decode(x['body_b64']) for x in b['payload']['objects']]==raws


def test_test_runtime_remains_network_blocked_and_operational_cli_rejects_it():
    from scripts import ncaaf_staged_acquisition as cli
    with pytest.raises(ValueError,match='NCAAF_STAGED_RUNTIME_UNSUPPORTED'):cli.verify_runtime()
    import requests
    with pytest.raises(requests.ConnectionError,match='External network blocked'):
        requests.get('https://api.collegefootballdata.com/games')


def test_fresh_quote_claim_cannot_be_reused_under_a_new_plan(synthetic,monkeypatch,tmp_path):
    packet,_=fixture(synthetic,monkeypatch,tmp_path);b=packet['payload']['staged_binding']['payload'];h=b['history'];pref=b['prequote']
    p=deepcopy(b['quote']['payload']['plan']);p['payload']['collection_authorization_ref']+='-new-plan';p=stage.seal(p['payload'])
    a,_=stage_configuration(p,monkeypatch,tmp_path);clock=Clock('2026-10-17T18:30:01Z');sent=[]
    result=stage.acquire(p,a,root=tmp_path,history_bundle=h,prequote=pref,
        transport=lambda *a:sent.append(a),wall=clock.wall,monotonic=clock.mono)
    assert result['payload']['status']=='INCOMPLETE' and result['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED'
    assert result['payload']['attempts']==0 and not sent and stage.verify_journal(tmp_path,result)


def test_interruption_is_spent_and_keeps_only_sanitized_failure(monkeypatch,tmp_path):
    p=plan();a,_=stage_configuration(p,monkeypatch,tmp_path);clock=Clock()
    def interrupted(*a):raise RuntimeError('SYNTHETIC PRIVATE EXCEPTION TEXT')
    b=stage.acquire(p,a,root=tmp_path,transport=interrupted,wall=clock.wall,monotonic=clock.mono)
    assert b['payload']['status']=='INCOMPLETE' and b['payload']['attempts']==1
    assert b'PRIVATE EXCEPTION' not in model.encode(b) and stage.verify_journal(tmp_path,b)
    with pytest.raises(ValueError):stage.acquire(p,a,root=tmp_path,transport=interrupted,wall=clock.wall,monotonic=clock.mono)
