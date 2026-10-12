"""Synthetic v2 responses/receipts only; external network remains blocked."""
from copy import deepcopy
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import urllib.parse
import pytest
from app_core import nfl_odds_acquisition as lane
from scripts import nfl_odds_acquisition as cli
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_odds_acquisition import (
    fixture as original_fixture, configuration, security, body, Transport,
    synthetic_socket_boundary, NOW, SECRET, CANARY,
)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network(): yield


def fixture(root, flags=None):
    p, a, raw=original_fixture(root)
    value=json.loads(raw)['payload']
    p['payload']['version']=lane.LISTING_VERSION
    p['payload']['request']['params'].update(flags if flags is not None else dict(includeSids='true',includeLinks='true'))
    p['payload']['permission']['request_sha256']=lane.digest(p['payload']['request'])
    p=lane.seal(p['payload'])
    a['payload'].update(version=lane.LISTING_AUTH_VERSION,plan_sha256=p['sha256'])
    a=lane.seal(a['payload'])
    value.update(version=lane.LISTING_CONFIG_VERSION,plan_sha256=p['sha256'],authorization=a,permission=p['payload']['permission'])
    return p,a,configuration(value)


def run(root,t,*,flags=None):
    p,a,raw=fixture(root,flags)
    with lane.controlled(p,raw,hashlib.sha256(raw).hexdigest(),root=root,wall=lambda:NOW,security=security) as approved:
        result=lane.acquire(p,approved,root=root,transport=t,wall=lambda:NOW)
    return p,result


def response():
    events=body();e=events[0];b=e['bookmakers'][0];m=b['markets'][0]
    e['sid']='SYNTHETIC-event';e['link']='https://example.invalid/event'
    b['link']='https://example.invalid/book';m['sid']='SYNTHETIC-market';m['link']='https://example.invalid/market'
    for i,o in enumerate(m['outcomes']):
        o['sid']='SYNTHETIC-outcome-'+str(i);o['link']='https://example.invalid/position/'+str(i)
    return events


FLAGS=[{k:v} for k in ('includeSids','includeLinks') for v in ('true','false')]
FLAGS += [dict(includeSids=a,includeLinks=b) for a in ('true','false') for b in ('true','false')]


@pytest.mark.parametrize('flags',FLAGS)
def test_real_cli_serializes_only_reviewed_flags_and_preserves_custody(monkeypatch,tmp_path,capsys,flags):
    from app_core import ncaaf_pilot_setup
    p,a,c=fixture(tmp_path,flags);raw=b'  '+json.dumps(response(),indent=2).encode()+b'\n'
    paths=[tmp_path/(k+'.json') for k in ('SYNTHETIC-plan','SYNTHETIC-auth','SYNTHETIC-config')]
    for path,data in zip(paths,(json.dumps(p).encode(),json.dumps(a).encode(),c)):path.write_bytes(data)
    calls=synthetic_socket_boundary(monkeypatch,raw)
    monkeypatch.setattr(cli,'verify_runtime',lambda:None) # Synthetic CLI only; unchanged production verifier separately tested.
    monkeypatch.setattr(cli,'private_credential',lambda:SECRET)
    monkeypatch.setattr(ncaaf_pilot_setup,'inspect_security',security)
    monkeypatch.setattr(lane,'controlled',partial(lane.controlled,wall=lambda:NOW))
    monkeypatch.setattr(lane,'acquire',partial(lane.acquire,wall=lambda:NOW))
    args=[str(paths[0]),'--execute','--authorization',str(paths[1]),'--custody-root',str(tmp_path),
          '--controlled-bindings',str(paths[2]),'--bindings-sha256',hashlib.sha256(c).hexdigest()]
    assert cli.main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert len(calls)==1
    target=calls[0].split(b'\r\n',1)[0].split(b' ')[1].decode()
    u=urllib.parse.urlsplit(target);query=urllib.parse.parse_qs(u.query)
    assert u.path==lane.SOURCE['endpoint']
    assert query=={k:[v] for k,v in dict(p['payload']['request']['params'],apiKey=SECRET).items()}
    journal=tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl')
    outcome=tmp_path/(p['sha256']+'.nfl-odds-unreviewed.json')
    retained=lane.read_back(outcome,journal);v=retained['payload']
    assert report==lane.execution_report(retained) and v['version']=='nfl-unaccepted-odds-body-v2'
    assert v['status']=='CAPTURED_UNREVIEWED' and v['attempted']==1
    assert lane.custody.read(v['object'])[0]==raw
    assert v['object']['body_sha256']==hashlib.sha256(raw).hexdigest()
    assert v['object']['receipt']['observed_at']==NOW and v['object']['receipt']['provider_clock_meaning']=='provider_market_last_update'
    rows=[json.loads(x) for x in journal.read_bytes().splitlines()]
    assert rows[1]['request']==p['payload']['request']
    refs=v['findings']['listing_references']
    assert len(refs)==5 and all(x['body_sha256']==v['object']['body_sha256'] and x['trusted'] is False for x in refs)
    assert refs[-1]['json_pointer']=='/0/bookmakers/0/markets/0/outcomes/1'
    assert {x['name'] for x in refs[-1]['fields']}=={'sid','link'}
    assert all(v[k] is False for k in ('source_acceptance','owner_reviewed','inference','public_export','qualified'))
    assert v['action']=='PASS' and v['stake']==0
    before=journal.read_bytes()
    assert cli.main(args)==1 and json.loads(capsys.readouterr().out)['reason']=='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'
    assert len(calls)==1 and journal.read_bytes()==before
    assert SECRET not in outcome.read_text() and CANARY not in outcome.read_text()


def test_original_contract_and_default_transport_still_reject_extra_flags(tmp_path):
    p,_,_=original_fixture(tmp_path)
    assert lane.validate(p)['version']==lane.VERSION
    for flags in FLAGS:
        changed=deepcopy(p['payload']);changed['request']['params'].update(flags)
        with pytest.raises(ValueError,match='NFL_ACQUIRE_ENDPOINT_NOT_APPROVED'):lane.validate(lane.seal(changed))
    p,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ENDPOINT_NOT_APPROVED'):
        lane.HttpsTransport(lambda:pytest.fail('No credential entry'))(p['payload']['request'],lane.LIMITS,60,lambda:0)


@pytest.mark.parametrize('case',['bool','uppercase','number','none','arbitrary','no_flags','endpoint','host'])
def test_rehashed_successor_cannot_expand_parameters_or_endpoint(tmp_path,case):
    p,_,_=fixture(tmp_path);v=p['payload'];params=v['request']['params']
    if case in {'bool','uppercase','number','none'}:params['includeLinks']={'bool':True,'uppercase':'TRUE','number':1,'none':None}[case]
    elif case=='arbitrary':params['includeBetLimits']='true'
    elif case=='no_flags':
        for k in lane.LISTING_FLAGS:params.pop(k)
    elif case=='endpoint':v['request']['endpoint']+='/other'
    elif case=='host':v['request']['host']='example.invalid'
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ENDPOINT_NOT_APPROVED'):lane.validate(lane.seal(v))


@pytest.mark.parametrize('case',['changed_flags','old_authorization','old_configuration','permission','review'])
def test_changed_request_cannot_borrow_approval_and_stops_before_spend(tmp_path,case):
    p,a,c=fixture(tmp_path);value=json.loads(c)['payload']
    if case=='changed_flags':
        p['payload']['request']['params']['includeLinks']='false';p=lane.seal(p['payload'])
    elif case=='old_authorization':
        value['authorization']['payload']['version']=lane.AUTH_VERSION
        value['authorization']=lane.seal(value['authorization']['payload'])
    elif case=='old_configuration':value['version']=lane.CONFIG_VERSION
    elif case=='permission':
        value['permission']['request_sha256']='0'*64;p['payload']['permission']=value['permission'];p=lane.seal(p['payload'])
        value['plan_sha256']=p['sha256'];value['authorization']['payload']['plan_sha256']=p['sha256']
        value['authorization']=lane.seal(value['authorization']['payload'])
    else:
        value['plan_sha256']='0'*64
    c=configuration(value)
    with pytest.raises(ValueError):
        with lane.controlled(p,c,hashlib.sha256(c).hexdigest(),root=tmp_path,wall=lambda:NOW,security=security):pytest.fail('Unreviewed request admitted')
    assert not list(tmp_path.glob('*.json*')) and lane.CONTROLLED.get() is None


@pytest.mark.parametrize('case',['missing','malformed','conflicting','malicious','credential'])
def test_optional_references_are_never_fetched_or_accepted(tmp_path,case):
    events=body() if case=='missing' else response();m=events[0]['bookmakers'][0]['markets'][0]
    if case=='malformed':m['sid']={'unrecognized':'structure'};m['link']=123
    elif case=='conflicting':m['outcomes'][0]['sid']=m['outcomes'][1]['sid']='SYNTHETIC-shared'
    elif case=='malicious':m['link']='javascript:fetch("https://evil.invalid")';m['outcomes'][0]['link']='https://evil.invalid/arbitrary'
    elif case=='credential':m['link']='https://evil.invalid/?apiKey='+SECRET
    raw=json.dumps(events).encode();t=Transport(raw);p,result=run(tmp_path,t);v=result['payload']
    assert len(t.calls)==v['attempted']==1
    assert not any(v[k] for k in ('inference','source_acceptance','owner_reviewed','public_export','qualified'))
    assert v['action']=='PASS' and v['stake']==0
    if case=='credential':assert v['status']=='INCOMPLETE' and v['object'] is None
    else:
        assert v['status']=='CAPTURED_UNREVIEWED' and lane.custody.read(v['object'])[0]==raw
        assert all(x['trusted'] is False for x in v['findings']['listing_references'])
        if case=='missing':assert all(x['optional_fields_state']=='UNKNOWN' for x in v['findings']['listing_references'])
    from app_core import nfl_scoring_research,nfl_scenario_research
    for reader in (nfl_scoring_research.checked,nfl_scenario_research.checked):
        with pytest.raises((ValueError,KeyError,TypeError)):reader(result,NOW)


@pytest.mark.parametrize('case',['over','gzip_over','redirect','pagination','timeout','http','incomplete'])
def test_v2_bounds_fail_and_spend_once(tmp_path,case):
    t=Transport()
    if case=='over':t.raw=b' '*(512*1024+1)
    elif case=='gzip_over':t.raw=gzip.compress(b' '*(512*1024+1));t.headers['content-encoding']='gzip'
    elif case=='redirect':t.status=302;t.headers['location']='https://example.invalid/'
    elif case=='pagination':t.headers['x-next-page']='2'
    elif case=='timeout':t.error=TimeoutError(CANARY)
    elif case=='http':t.status=429
    elif case=='incomplete':t.headers['content-length']='123456'
    p,result=run(tmp_path,t);v=result['payload']
    assert v['status']=='INCOMPLETE' and v['attempted']==len(t.calls)==1 and v['object'] is None
    saved=''.join(x.read_text() for x in tmp_path.glob('*.json*'))
    assert SECRET not in saved and CANARY not in saved
    before=(tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='NFL_ACQUIRE_ALREADY_ATTEMPTED_NO_RESUME'):run(tmp_path,t)
    assert len(t.calls)==1 and (tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl')).read_bytes()==before


def test_rehashed_derivative_pointer_conflict_rejects_static_readback(tmp_path):
    p,result=run(tmp_path,Transport(json.dumps(response()).encode()))
    result['payload']['findings']['listing_references'][0]['json_pointer']='/unrelated'
    changed=lane.seal(result['payload']);path=tmp_path/(p['sha256']+'.nfl-odds-unreviewed.json')
    path.write_text(json.dumps(changed)) # Labelled synthetic corruption only.
    with pytest.raises(ValueError,match='NFL_ACQUIRE_PRIVATE_READBACK'):
        lane.read_back(path,tmp_path/(p['sha256']+'.nfl-odds-attempts.jsonl'))


def test_successor_planning_never_loads_credentials_or_storage(monkeypatch,tmp_path,capsys):
    p,_,_=fixture(tmp_path);p['payload'].update(evidence_label='PROPOSAL',unresolved=['SYNTHETIC: no real authorization'])
    p=lane.seal(p['payload']);path=tmp_path/'proposal.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(cli,'private_credential',lambda:pytest.fail('credentials in planning'))
    monkeypatch.setattr(lane,'controlled',lambda *a,**k:pytest.fail('storage in planning'))
    monkeypatch.setattr(lane,'HttpsTransport',lambda *a,**k:pytest.fail('transport in planning'))
    assert cli.main([str(path)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['network_requests']==report['credentials']==report['storage_operations']==0
    assert report['odds_credits_max']==1 and report['maximum_additional_spend']==0
    assert list(tmp_path.iterdir())==[path]
