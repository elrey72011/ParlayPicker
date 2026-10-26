"""Synthetic only; the repository network blocker remains active."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import sqlite3

import pytest

from app_core.nfl_private_workflow import authenticate, Workflow
from app_core.nfl_workflow_inputs import FINDINGS, observed_pair
from app_core.nfl_workflow_store import Store, WorkflowError
from nfl_workflow_fixtures import KEY,TOKEN,SyntheticProvider,make_workflow,capture_source,capture_two


def test_two_uncatalogued_games_native_both_sides_persist_reopen_and_read_only_display(tmp_path,monkeypatch):
    path=tmp_path/'SYNTHETIC-workflow.sqlite3';service,provider=make_workflow(path)
    pairs=capture_two(service)
    assert len(pairs)==2 and all('UNCATALOGUED' in pair['payload']['prepared']['selected']['event']['canonical_event_id'] for pair in pairs)
    result=service.calculate([p['id'] for p in pairs],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert len(result)==4 and len(provider.calls)==3
    for pair in pairs:
        packets=pair['payload']['packets'];home=packets[0]['payload']['event']['home']
        assert packets[0]['payload']['scenario']['selected_team']==home
        assert packets[0]['payload']['features']==packets[1]['payload']['features']
        assert packets[0]['payload']['scenario']['signed_handicap']==-packets[1]['payload']['scenario']['signed_handicap']
    assert len(service.saved())==4
    def denied(*args,**kwargs):pytest.fail('saved display must not calculate or acquire')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',denied)
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities',denied)
    monkeypatch.setattr(Store,'put',denied);monkeypatch.setattr(Store,'claim',denied)
    # Fresh service with the same persistent fixture DB/key, not session state.
    def factory(readonly):
        assert readonly
        return sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    reopened=Workflow(Store(factory,KEY,synthetic=True),authenticate(TOKEN,TOKEN))
    for saved in reopened.saved():
        receipt=saved['payload']['receipt'];view=reopened.view(receipt)
        assert view['result_sha256']==receipt['result_sha256']
        assert view['payload']['inference_time']==saved['payload']['inference_time']
        assert reopened.store.download_native(receipt)


def test_durable_claim_is_atomic_across_connections_and_owners(tmp_path):
    path=tmp_path/'SYNTHETIC-atomic.sqlite3';store=Store.synthetic_fixture(path,KEY)
    def attempt(_):
        try:store.claim('same-intent',{'SYNTHETIC':'owner actual answer'});return True
        except WorkflowError as error:assert str(error)=='PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY';return False
    with ThreadPoolExecutor(max_workers=6) as pool:assert sum(pool.map(attempt,range(12)))==1
    other=Store.synthetic_fixture(path,KEY,owner='SYNTHETIC other owner')
    assert not other.spent('same-intent') and other.list('native-result')==[]


def test_wrong_owner_and_unsigned_record_rejected(tmp_path):
    with pytest.raises(ValueError):authenticate('wrong',TOKEN)
    service,_=make_workflow(tmp_path/'SYNTHETIC-auth.sqlite3')
    saved=service.store.put('SYNTHETIC',{'value':1})
    other=Store.synthetic_fixture(tmp_path/'SYNTHETIC-auth.sqlite3',KEY,owner='other')
    with pytest.raises(ValueError):other.get(saved['id'])
    wrong_key=Store.synthetic_fixture(tmp_path/'SYNTHETIC-auth.sqlite3',b'SYNTHETIC-wrong-key-at-least-32-bytes')
    with pytest.raises(ValueError,match='AUTHENTICATION'):wrong_key.get(saved['id'])


@pytest.mark.parametrize('attack,code',[('integer','INTEGER_LINE'),('identity','IDENTITY'),('stale','QUOTE_STALE')])
def test_original_captured_then_unsupported_market_stops_without_retry(tmp_path,attack,code):
    service,provider=make_workflow(tmp_path/('SYNTHETIC-'+attack+'.sqlite3'))
    capture_source(service)
    if attack=='integer':provider.line=3
    if attack=='identity':provider.wrong=True
    if attack=='stale':provider.stale=True
    selected=service.games()['games'][0]['event']['canonical_event_id'];prepared=service.prepare([selected]);op=prepared[0]['operation']
    service.capture_quotes([op['id']],remaining_credits=1,reset_at=(datetime.now(timezone.utc)+timedelta(days=5)).isoformat(),private_use=True,no_overage=True)
    with pytest.raises(ValueError,match=code):service.stage([op['id']])
    with pytest.raises(ValueError,match='ALREADY_SPENT'):service.capture_quotes([op['id']],remaining_credits=1,reset_at=(datetime.now(timezone.utc)+timedelta(days=5)).isoformat(),private_use=True,no_overage=True)
    assert len(provider.calls)==2 and service.store.get('market:'+op['id'])


def test_failed_capture_and_rerun_do_not_reopen_the_allowance(tmp_path):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-failed.sqlite3');provider.fail=True
    op=service.source_operation()
    with pytest.raises(ConnectionRefusedError):service.capture_source(op['id'],private_retention=True,private_research=True,permission_end=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat())
    provider.fail=False
    assert service.source_operation()['id']==op['id']
    with pytest.raises(ValueError,match='ALREADY_SPENT'):capture_source(service)
    assert len(provider.calls)==1
    outcome=service.store.get('outcome:'+op['id'])['payload']
    assert outcome['status']=='FAILED_NO_RETRY' and outcome['diagnostics']['exception_category']=='CONNECTION'


def test_missing_histories_changed_inputs_and_expiry_precede_prediction(tmp_path,monkeypatch):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-missing.sqlite3')
    provider.rows=[row for row in provider.rows if 'HISTORY-BUF-' not in row['game_id']]
    capture_source(service)
    with pytest.raises(ValueError,match='HISTORY_MISSING'):service.prepare([service.games()['games'][0]['event']['canonical_event_id']])


def test_checkbox_review_and_duplicate_calculation_are_not_synthesized(tmp_path):
    service,_=make_workflow(tmp_path/'SYNTHETIC-review.sqlite3');pairs=capture_two(service);ids=[p['id'] for p in pairs]
    with pytest.raises(ValueError,match='FINDINGS_REQUIRED'):service.calculate(ids,{},attestation=True,limitations=True)
    assert service.saved()==[]
    service.calculate(ids,{k:True for k in FINDINGS},attestation=True,limitations=True)
    with pytest.raises(ValueError,match='ALREADY_SPENT'):service.calculate(ids,{k:True for k in FINDINGS},attestation=True,limitations=True)


def test_interrupted_capture_is_durably_spent_and_requires_explicit_successor(tmp_path):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-interrupted.sqlite3')
    old=service.source_operation()
    def interrupted(*a,**k):raise KeyboardInterrupt()
    service.score_transport=interrupted
    end=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()
    with pytest.raises(KeyboardInterrupt):service.capture_source(old['id'],private_retention=True,private_research=True,permission_end=end)
    assert service.store.spent('operation:'+old['id'])
    service.score_transport=provider.scores
    with pytest.raises(ValueError,match='ALREADY_SPENT'):service.capture_source(old['id'],private_retention=True,private_research=True,permission_end=end)
    new=service.source_operation(predecessor=old['id'])
    assert new['id']!=old['id'] and new['payload']['predecessor']==old['id']
    assert provider.calls==[]  # Preparing a successor is not dispatch authority.
    service.capture_source(new['id'],private_retention=True,private_research=True,permission_end=end)
    assert len(provider.calls)==1 and service.store.spent('operation:'+old['id'])


def test_two_tabs_adopt_same_intent_and_only_one_dispatch(tmp_path):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-tabs.sqlite3');op=service.source_operation()
    def submit(_):
        try:
            assert service.source_operation()['id']==op['id']
            service.capture_source(op['id'],private_retention=True,private_research=True,permission_end=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat())
            return True
        except WorkflowError as exc:assert 'ALREADY_SPENT' in str(exc);return False
    with ThreadPoolExecutor(max_workers=2) as pool:assert sum(pool.map(submit,range(2)))==1
    assert len(provider.calls)==1


def test_permission_quote_and_kickoff_expiry_block_before_native_call(tmp_path,monkeypatch):
    service,_=make_workflow(tmp_path/'SYNTHETIC-clocks.sqlite3');pairs=capture_two(service)
    def denied(*a,**k):pytest.fail('Expired input must stop before native calculation')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',denied)
    first=pairs[0]['payload'];at=datetime.now(timezone.utc)+timedelta(minutes=16)
    service.clock=lambda:at.isoformat()
    with pytest.raises(ValueError,match='QUOTE_STALE'):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    service.clock=lambda:first['prepared']['selected']['event']['start']
    with pytest.raises(ValueError,match='KICKOFF'):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    source=first['prepared']['source'];from copy import deepcopy
    changed=deepcopy(source);changed['permission']['effective_until']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    from app_core.nfl_workflow_inputs import prepare_game
    with pytest.raises(ValueError,match='PERMISSION_EXPIRED'):prepare_game(changed,first['prepared']['selected'],datetime.now(timezone.utc).isoformat())


def test_changed_model_invalidates_approval_without_claiming_a_native_call(tmp_path,monkeypatch):
    service,_=make_workflow(tmp_path/'SYNTHETIC-changed.sqlite3');pairs=capture_two(service)
    monkeypatch.setattr('app_core.nfl_scenario_research.model_binding',lambda:{'SYNTHETIC':'changed model'})
    with pytest.raises(ValueError,match='INPUTS_CHANGED'):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert service.store.list('saved-index')==[]


def test_changed_acquisition_binding_stops_before_dispatch_or_calculation(tmp_path,monkeypatch):
    from app_core import nfl_workflow_acquisition as acquisition
    service,provider=make_workflow(tmp_path/'SYNTHETIC-method.sqlite3')
    op=service.source_operation();native=acquisition.method_binding
    with monkeypatch.context() as m:
        m.setattr(acquisition,'method_binding',lambda:dict(native(),version='SYNTHETIC unsupported successor'))
        with pytest.raises(ValueError,match='METHOD_CHANGED'):service.capture_source(op['id'],private_retention=True,private_research=True,permission_end=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat())
    assert provider.calls==[] and not service.store.spent('operation:'+op['id'])
    pairs=capture_two(service)
    monkeypatch.setattr(acquisition,'method_binding',lambda:dict(native(),version='SYNTHETIC unsupported successor'))
    with pytest.raises(ValueError,match='METHOD_CHANGED'):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert service.saved()==[]


def test_neutral_duplicate_schedule_and_broken_history_are_precise(tmp_path):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-schedule.sqlite3')
    provider.rows[-1]['location']='Neutral';capture_source(service)
    catalog=service.games();assert len(catalog['games'])==1
    assert catalog['rejected'][0]['reason']=='NFL_WORKFLOW_NEUTRAL_SITE_UNSUPPORTED'


def test_partial_native_failure_preserves_first_result_and_prohibits_repeat(tmp_path,monkeypatch):
    service,_=make_workflow(tmp_path/'SYNTHETIC-partial.sqlite3');pairs=capture_two(service)
    from app_core import nfl_scenario_research as scenario
    native=scenario.compute;count=[]
    def interrupted(packet,**kwargs):
        count.append(1)
        if len(count)==2:raise RuntimeError('SYNTHETIC interruption after first preserved result')
        return native(packet,**kwargs)
    monkeypatch.setattr(scenario,'compute',interrupted)
    with pytest.raises(RuntimeError):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert len(service.saved())==1 and len(count)==2
    with pytest.raises(ValueError,match='ALREADY_SPENT'):service.calculate([pairs[0]['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert len(service.saved())==1 and len(count)==2



def test_shared_kickoff_listing_maps_exact_target_without_discarding_originals(tmp_path):
    import base64,json
    from copy import deepcopy
    from app_core import nfl_observed_custody as custody
    service,_=make_workflow(tmp_path/'SYNTHETIC-shared-kickoff.sqlite3')
    p=capture_two(service)[0]['payload'];original=p['market']['object']
    events=json.loads(base64.b64decode(original['bytes_base64']));other=deepcopy(events[0])
    other.update(id='SYNTHETIC-other-event',home_team='Seattle Seahawks',away_team='Arizona Cardinals')
    other['bookmakers'][0]['markets'][0]['outcomes']=[dict(name='Seattle Seahawks',point=7.5,price=-110),dict(name='Arizona Cardinals',point=-7.5,price=-110)]
    raw=json.dumps([other,*events]).encode();obj=custody.retained(raw,receipt=deepcopy(original['receipt']))
    result=observed_pair(obj,p['prepared'],service.clock())
    assert result['provider_event_id']==events[0]['id']
    assert set(result['pair'])=={p['prepared']['selected']['event']['home'],p['prepared']['selected']['event']['away']}
    assert sorted(row['point'] for row in result['pair'].values())==[-2.5,2.5]
    assert custody.read(obj)[0]==raw  # Complete original, including the unrelated row.
    duplicate=deepcopy(events[0]);duplicate['id']='SYNTHETIC-ambiguous-target'
    bad=custody.retained(json.dumps([*events,duplicate]).encode(),receipt=deepcopy(original['receipt']))
    with pytest.raises(ValueError,match='TARGET_AMBIGUOUS'):observed_pair(bad,p['prepared'],service.clock())


def test_aggregate_record_fits_released_scoring_envelopes_without_raising_provider_limits(tmp_path):
    from app_core import nfl_scoring_profile as profile
    from app_core.nfl_workflow_store import encode,digest,MAX_RECORD
    assert profile.MAX_BODY==profile.MAX_WIRE==4*1024*1024
    assert profile.MAX_PACKET==12*1024*1024
    encoded='A'*(4*((profile.MAX_BODY+2)//3))
    payload={'SYNTHETIC-serialization-only':[encoded,encoded,encoded]}
    assert 16*1024*1024<len(encode(payload).encode())<MAX_RECORD
    service,_=make_workflow(tmp_path/'SYNTHETIC-aggregate-bound.sqlite3')
    saved=service.store.put('SYNTHETIC-aggregate-only',payload)
    assert digest(service.store.get(saved['id'])['payload'])==digest(payload)



def test_latest_launch_keeps_the_full_transport_budget_and_requires_new_successor_review(tmp_path):
    for remaining in (30,61):
        service,provider=make_workflow(tmp_path/('SYNTHETIC-latest-'+str(remaining)+'.sqlite3'))
        op=service.source_operation();end=datetime.fromisoformat(op['payload']['window']['not_after'])
        at=end-timedelta(seconds=remaining);service.clock=lambda:at.isoformat()
        permission=(end+timedelta(days=3)).isoformat()
        if remaining==30:
            with pytest.raises(ValueError,match='LATEST_LAUNCH_PASSED'):
                service.capture_source(op['id'],private_retention=True,private_research=True,permission_end=permission)
            assert provider.calls==[] and not service.store.spent('operation:'+op['id'])
            successor=service.source_operation(predecessor=op['id'])
            assert successor['id']!=op['id'] and provider.calls==[]
        else:
            service.capture_source(op['id'],private_retention=True,private_research=True,permission_end=permission)
            assert len(provider.calls)==1


def test_quote_expiring_during_native_claim_is_spent_without_a_predictor_call(tmp_path,monkeypatch):
    service,provider=make_workflow(tmp_path/'SYNTHETIC-post-claim-expiry.sqlite3')
    pair=capture_two(service)[0];original_claim=service.store.claim
    observed=observed_pair(pair['payload']['market']['object'],pair['payload']['prepared'],service.clock())
    late=datetime.fromisoformat(observed['updated_at'])+timedelta(seconds=901)
    def claim(identity,payload):
        result=original_claim(identity,payload)
        if identity.startswith('native:'):service.clock=lambda:late.isoformat()
        return result
    service.store.claim=claim
    def forbidden(*a,**k):pytest.fail('Expired quote must stop after the durable claim before native dispatch')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',forbidden)
    with pytest.raises(ValueError,match='QUOTE_STALE'):
        service.calculate([pair['id']],{k:True for k in FINDINGS},attestation=True,limitations=True)
    assert len(provider.calls)==3 and service.saved()==[]
    outcomes=service.store.list('native-outcome')
    assert len(outcomes)==1 and outcomes[0]['payload']['status']=='FAILED_OR_UNSAVED_NO_RETRY'
    assert outcomes[0]['payload']['reason']=='NFL_WORKFLOW_QUOTE_STALE'



def test_private_store_pages_are_disjoint_authenticated_and_read_only(tmp_path,monkeypatch):
    service,_=make_workflow(tmp_path/'SYNTHETIC-pages.sqlite3')
    expected={service.store.put('SYNTHETIC-page',{'slot':n})['id'] for n in range(7)}
    def denied(*a,**k):pytest.fail('Listing older records must never write or predict')
    monkeypatch.setattr(Store,'put',denied);monkeypatch.setattr(Store,'claim',denied)
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',denied)
    pages=[service.store.list('SYNTHETIC-page',limit=3,offset=x) for x in (0,3,6)]
    identities=[r['id'] for page in pages for r in page]
    assert len(identities)==7 and len(set(identities))==7 and set(identities)==expected
    assert service.saved(page=2)==[]
    for bad in (0,-1,True,1_000_001):
        with pytest.raises(ValueError,match='RECORD_LIMIT'):service.saved(page=bad)
