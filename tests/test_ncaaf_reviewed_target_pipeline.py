"""Synthetic complete caller/export/replay plus operation control boundaries."""
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import importlib.util
import pytest
from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_spread_execution as execution
from app_core import ncaaf_pipeline_evidence as adapter,ncaaf_reviewed_staged as evidence,ncaaf_reviewed_target as target
from app_core import ncaaf_staged_acquisition as stage
from test_ncaaf_model_compatibility import synthetic
import ncaaf_reviewed_target_fixtures as f


@pytest.mark.parametrize('case,line',[(0,-3.5),(1,-7)])
def test_actual_two_target_pair_capture_export_replay_private_display(synthetic,monkeypatch,tmp_path,case,line):
    from app_core import research_display,research_replay
    from app_core.per_game_boards import per_game_board
    import test_source_contract_pipeline as ef
    f.select(monkeypatch,case);p,rows=f.pair_fixture(synthetic,monkeypatch,tmp_path,line);before=deepcopy(p)
    a=f.inference_authorization(p,monkeypatch,paired=True)
    analysis=pair.admitted_analysis(p,rows,authorization=a,custody_root=tmp_path)
    assert len(analysis)==2
    assert [adapter.diagnose(r) for r in analysis.to_dict('records')]==[dict(status='COMPLETE',reason='AVAILABLE')]*2
    monkeypatch.setattr(ef,'NOW',f.NOW);monkeypatch.setattr(ef,'CAPTURE',(f.NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:f.NOW)
    monkeypatch.setattr('app_core.prediction_evidence.now_utc',lambda:(f.NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.public_board.datetime',ef.FrozenDateTime)
    exports=pair.capture_exports(analysis,custody_root=tmp_path,database=tmp_path/'private.sqlite3');saved=exports[0]
    assert not saved['captured'].production_eligible.fillna(False).any()
    assert all(v['production_bet_amount']==0 for v in saved['card'].wager_contract)
    frames=[per_game_board(saved['card'],saved['captured'],family=k,novig_only=True,college_fallback=True,private_research=True) for k in ('overall','sides','totals')]
    assert not frames[1].empty
    shown=json.loads(frames[1].iloc[0].ncaaf_spread_private_display)
    assert shown['wager_action']=='PASS' and shown['live_stake']==0 and shown['comparison']['shared_push']>=0
    assert (shown['comparison']['shared_push']>0)==float(line).is_integer()
    receipt=research_replay.retain_export(frames,saved['package'],saved['card'],saved['captured'],path=tmp_path/'private.sqlite3')
    _,sources=research_replay.read_export(receipt['export_id'],path=tmp_path/'private.sqlite3')
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('STATIC REPLAY RAN MATH'))
    for row in retained.to_dict('records'):
        assert adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE')
        display=pair.private_display(row)
        assert display['outcome_probabilities']['cover']==row['ml_probability'] and display['ev'] is None
        assert research_display.from_export(row)['probability'] is None
    assert 'body_b64' not in json.dumps(saved['package']) and p==before
    assert all(game['research_display']['probability'] is None for games in saved['package']['games'].values() for game in games)
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'SYNTHETIC-reviewed-target-display.py'
    script.write_text('import streamlit as st\nfrom app.ui.ncaaf_pipeline_research import render\nrender(st.session_state["games"],st.session_state["candidates"])',encoding='utf-8')
    app=AppTest.from_file(str(script),default_timeout=40);app.session_state['games']=saved['card'];app.session_state['candidates']=saved['captured'];app.session_state['ncaaf_private_research']=True;app.run()
    assert not app.exception and any(m.value==f'{analysis.iloc[0].ml_probability:.2%}' for m in app.metric)
    assert not adapter.ACCEPTED_PACKETS and execution.ACTIVE.get() is None


@pytest.mark.parametrize('attack',['body','stage','target','orientation','neutral','kickoff','alias','duplicate','review','review_after','config','config_rehashed','journal','features','model'])
def test_altered_bound_evidence_rejects_before_math(synthetic,monkeypatch,tmp_path,attack):
    f.select(monkeypatch,0);p,row=f.fixture(synthetic,monkeypatch,tmp_path);v=p['payload'];b=v['staged_binding']['payload'];o=pair._observation(p)
    if attack=='body':b['history']['payload']['objects'][0]['body_b64']='QQ=='
    elif attack=='stage':b['history']=deepcopy(b['quote'])
    elif attack in ('target','orientation','neutral','kickoff'):
        t=b['history']['payload']['plan']['payload']['target']
        if attack=='target':t['home_school']='SYNTHETIC substituted university'
        elif attack=='orientation':t['home_school'],t['away_school']=t['away_school'],t['home_school']
        elif attack=='neutral':t['neutral_site']=not t['neutral_site']
        else:t['start_utc']='2026-10-21T23:30:00Z'
        b['history']['payload']['plan']=stage.seal(b['history']['payload']['plan']['payload'])
    elif attack=='alias':v['alias_review']['crosswalk'][0]['provider_name']='SYNTHETIC other provider label'
    elif attack=='duplicate':v['alias_review']['schedule'].append(deepcopy(v['alias_review']['schedule'][0]))
    elif attack=='review':b['prequote']['payload']['history_review']=None
    elif attack=='review_after':b['prequote']['payload']['history_review']['reviewed_at']='2026-10-17T18:31:00Z'
    elif attack.startswith('config'):
        c=b['history']['payload']['control_receipt'];q=json.loads(base64.b64decode(c['file_b64']))
        q['payload']['plan_sha256']='0'*64
        raw=stage.model.encode(stage.seal(q['payload']));c.update(file_b64=base64.b64encode(raw).decode())
        if attack=='config_rehashed':c['file_sha256']=hashlib.sha256(raw).hexdigest()
    elif attack=='journal':b['history']['payload']['journal_b64']=base64.b64encode(b'SYNTHETIC substituted journal').decode()
    elif attack=='features':o['features']['values'][0]+=1
    else:o['model']['compatibility']['family']='total'
    for k in ('history','prequote','quote'):b[k]=stage.seal(b[k]['payload'])
    v['staged_binding']=stage.seal(b);p=stage.seal(v)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('INVALID INPUT REACHED MATH'))
    with pytest.raises((ValueError,KeyError,TypeError)):owner.load(p)
    assert not list(tmp_path.glob('*.inference-attempts.jsonl'))


@pytest.mark.parametrize('clock',['2026-10-17T18:30:03Z','2026-10-17T18:45:01Z','2026-10-20T23:30:00Z','2026-10-18T18:30:05Z'])
def test_actual_clock_preserves_age_freshness_review_and_pregame(synthetic,monkeypatch,tmp_path,clock):
    f.select(monkeypatch,0);p,row=f.fixture(synthetic,monkeypatch,tmp_path)
    monkeypatch.setattr(adapter,'generated_time',lambda:clock)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('INVALID CLOCK REACHED MATH'))
    with adapter.selected([p],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='unavailable'


@pytest.mark.parametrize('change',['scoring','yardage','cutoff','future','duplicate','week'])
def test_original_history_cutoff_minima_and_inventory(synthetic,monkeypatch,tmp_path,change):
    f.select(monkeypatch,0);seed=tmp_path/'seed';seed.mkdir();p,_=f.fixture(synthetic,monkeypatch,seed)
    root=tmp_path/'candidate';root.mkdir();raw=f.bodies();games=json.loads(raw[0])
    if change=='scoring':games=games[:5]
    elif change=='yardage':raw[-1]=stage.model.encode([])
    elif change in ('cutoff','future'):
        for g in games[1:]:g['startDate']='2026-10-13T23:30:00Z' if change=='cutoff' else f.CANONICAL['start_utc']
    elif change=='duplicate':games.append(deepcopy(games[1]))
    else:games[1]['week']=12
    raw[0]=stage.model.encode(games);h,_,_=f.capture(monkeypatch,root,raw=raw)
    with pytest.raises(ValueError):evidence.prepare(h,model_packet=pair._observation(p)['model'],prepared_at='2026-10-17T18:01:00Z')


def test_prequote_review_required_before_dispatch_and_native_attempt_is_spent(synthetic,monkeypatch,tmp_path):
    f.select(monkeypatch,0);packet,_=f.fixture(synthetic,monkeypatch,tmp_path);b=packet['payload']['staged_binding']['payload']
    h,pref,q=b['history'],deepcopy(b['prequote']),b['quote']['payload']['plan']
    pref['payload']['history_review']=None;pref=stage.seal(pref['payload']);p=deepcopy(q['payload']);p['prequote_sha256']=pref['sha256'];p=stage.seal(p)
    a,_=f.stage_configuration(p,monkeypatch,tmp_path);clock=f.Clock('2026-10-17T18:30:01Z')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):stage.acquire(p,a,root=tmp_path,transport=lambda *a:pytest.fail('UNREVIEWED DISPATCH'),history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    original=h['payload'];before_journal=(tmp_path/(original['plan']['sha256']+'.attempts.jsonl')).read_bytes()
    with pytest.raises(ValueError,match='ALREADY_SPENT'):f.capture(monkeypatch,tmp_path,original['plan'])
    assert (tmp_path/(original['plan']['sha256']+'.attempts.jsonl')).read_bytes()==before_journal


def test_operator_private_literal_rejects_without_retaining_entered_value(monkeypatch,tmp_path,capsys):
    # The input is a synthetic sentinel, never a real owner credential.
    spec=importlib.util.spec_from_file_location('SYNTHETIC_reviewed_operator',Path(__file__).resolve().parents[1]/'scripts/ncaaf_reviewed_target_operator.py')
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    from contextlib import contextmanager
    @contextmanager
    def controlled(*a,**k):yield {}
    monkeypatch.setattr(op.runpy,'run_path',lambda *a,**k:{'verify_runtime':lambda:None,'main':lambda *a:pytest.fail('WRONG INPUT REACHED NATIVE')})
    monkeypatch.setattr(op.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(op,'preflight',lambda *a,**k:({},dict(sha256='a'*64),b'{}',tmp_path))
    monkeypatch.setattr(op.stage,'controlled',controlled)
    monkeypatch.setattr(op.getpass,'getpass',lambda *a:'SYNTHETIC-SECRET-SENTINEL-NEVER-A-REAL-KEY')
    assert op.execute('SYNTHETIC','b'*64,'c'*64)==1
    output=capsys.readouterr().out
    assert 'SECRET-SENTINEL' not in output
    saved=json.loads((tmp_path/('a'*64+'.operator-report.json')).read_bytes())
    assert saved['reason']=='ENTITLEMENT_LITERAL_REQUIRED' and saved['provider_attempts']==0
    assert 'SECRET-SENTINEL' not in json.dumps(saved)
    with pytest.raises(FileExistsError):op.execute('SYNTHETIC','b'*64,'c'*64)


@pytest.mark.parametrize('case',[0,1])
def test_actual_cli_local_socket_originals_and_diagnostic_readback(monkeypatch,tmp_path,capsys,case):
    from test_ncaaf_transport_diagnostics import cli_module,local_wire
    from app_core import ncaaf_transport_diagnostics as diagnostic
    f.select(monkeypatch,case);p=f.plan();a,control=f.stage_configuration(p,monkeypatch,tmp_path)
    for name,raw in [('plan.json',stage.model.encode(p)),('auth.json',stage.model.encode(a)),('control.json',control)]:
        (tmp_path/name).write_bytes(raw)
    c=cli_module()
    with pytest.raises(ValueError,match='RUNTIME_UNSUPPORTED'):c.verify_runtime()
    monkeypatch.setattr(c,'verify_runtime',lambda:None) # explicit synthetic CLI/socket injection only
    monkeypatch.setattr(c.sys.stdin,'isatty',lambda:True)
    monkeypatch.setattr(c.getpass,'getpass',lambda *a:'SYNTHETIC-LOCAL-ONLY-KEY')
    if case:
        clock=f.Clock();acquire=stage.acquire
        def prompt(*a):clock.elapsed=600.;return 'SYNTHETIC-LOCAL-ONLY-KEY'
        def captured(*a,**kw):
            assert clock.elapsed==600.;kw['monotonic']=clock.mono
            return acquire(*a,**kw)
        monkeypatch.setattr(c.getpass,'getpass',prompt);monkeypatch.setattr(stage,'acquire',captured)
    monkeypatch.setattr(stage.transport,'utc',lambda:'2026-10-17T18:00:01Z')
    args=['execute',str(tmp_path/'plan.json'),'--authorization',str(tmp_path/'auth.json'),'--controlled-bindings',str(tmp_path/'control.json'),
        '--bindings-sha256',hashlib.sha256(control).hexdigest(),'--custody-root',str(tmp_path),'--transport-diagnostics']
    bodies=f.bodies()
    with local_wire(monkeypatch,bodies,closing=True) as wire:
        assert c.main(args)==0
        result=json.loads(capsys.readouterr().out)
        assert result['transport_diagnostics_readback']=='PASS' and result['attempts']==4
        bundle=json.loads((tmp_path/(p['sha256']+'.stage.json')).read_bytes())
        stage.read_bundle(bundle);stage.verify_journal(tmp_path,bundle)
        if case:assert bundle['payload']['active_seconds']==0 # Ten synthetic private-input minutes were excluded.
        raw=(tmp_path/(p['sha256']+'.transport-diagnostics.json')).read_bytes()
        diagnostic.read(tmp_path,bundle,raw,hashlib.sha256(raw).hexdigest())
        assert 'SYNTHETIC-LOCAL-ONLY-KEY' not in raw.decode()
        assert [o['body_sha256'] for o in bundle['payload']['objects']]==[hashlib.sha256(x).hexdigest() for x in bodies]
        assert c.main(args)==1 and len(wire['sent'])==4


def test_standalone_planning_and_operator_help_use_clean_runtime(monkeypatch,tmp_path):
    import subprocess,sys
    f.select(monkeypatch,1);p=f.plan();path=tmp_path/'SYNTHETIC-plan.json';path.write_bytes(stage.model.encode(p))
    root=Path(__file__).resolve().parents[1];libs=str(Path(pytest.__file__).resolve().parent.parent)
    for script,args in [('ncaaf_staged_acquisition.py',['plan',str(path)]),('ncaaf_reviewed_target_operator.py',['--help'])]:
        sp=root/'scripts'/script
        code='import sys,runpy;sys.path.insert(0,'+repr(libs)+');sys.argv='+repr([str(sp),*args])+';runpy.run_path('+repr(str(sp))+',run_name="__main__")'
        result=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=root,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
    assert not list(tmp_path.glob('*.attempts.jsonl'))


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt,'oversize'])
def test_interrupted_or_oversized_new_stage_keeps_spent_attempt_and_fixed_failure(monkeypatch,tmp_path,failure):
    from test_ncaaf_pilot import Stream
    f.select(monkeypatch,0);p=f.plan();a,_=f.stage_configuration(p,monkeypatch,tmp_path);clock=f.Clock();calls=[]
    def transport(*args):
        calls.append(1)
        if failure=='oversize':return Stream(b' '*(524288+1))
        raise failure('SYNTHETIC-SECRET-ERROR-NOT-RETAINED')
    b=stage.acquire(p,a,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.mono)
    assert len(calls)==1 and b['payload']['attempts']==1 and b['payload']['status']=='INCOMPLETE' and b['payload']['objects']==[]
    assert 'SYNTHETIC-SECRET-ERROR' not in json.dumps(b)
    stage.read_bundle(b,complete=False);stage.verify_journal(tmp_path,b)
    with pytest.raises(ValueError,match='ALREADY_SPENT'):stage.acquire(p,a,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.mono)
    assert len(calls)==1


def test_combined_original_bytes_leave_no_quote_budget(synthetic,monkeypatch,tmp_path):
    from test_ncaaf_owner_terms import fixture as base
    f.select(monkeypatch,0);packet,_=base(synthetic,monkeypatch);model=pair._observation(packet)['model']
    raw=[v+b' '*(524288-len(v)) for v in f.bodies()]
    h,_,_=f.capture(monkeypatch,tmp_path,raw=raw)
    assert h['payload']['original_bytes']==2097152
    pref=evidence.prepare(h,model_packet=model,prepared_at='2026-10-17T18:01:00Z')
    ident=f.identity(monkeypatch,tmp_path,pref);pref=f.review_prequote(pref,h);p=f.plan(target.QUOTE_PLAN,h,pref,ident)
    a,_=f.stage_configuration(p,monkeypatch,tmp_path);clock=f.Clock('2026-10-17T18:30:01Z')
    with pytest.raises(ValueError,match='COMBINED_LIMIT'):
        stage.acquire(p,a,root=tmp_path,history_bundle=h,prequote=pref,transport=lambda *a:pytest.fail('NO BYTE BUDGET'),wall=clock.wall,monotonic=clock.mono)
    assert not (tmp_path/(p['sha256']+'.attempts.jsonl')).exists()


def test_shared_active_budget_and_history_claim_survive_quote_failure(synthetic,monkeypatch,tmp_path):
    from test_ncaaf_owner_terms import fixture as base
    f.select(monkeypatch,0);packet,_=base(synthetic,monkeypatch);model=pair._observation(packet)['model']
    h,_,_=f.capture(monkeypatch,tmp_path)
    pref=evidence.prepare(h,model_packet=model,prepared_at='2026-10-17T18:01:00Z');ident=f.identity(monkeypatch,tmp_path,pref)
    pref=f.review_prequote(pref,h);p=f.plan(target.QUOTE_PLAN,h,pref,ident);a,_=f.stage_configuration(p,monkeypatch,tmp_path)
    clock=f.Clock('2026-10-17T18:30:01Z');calls=[]
    def exhausted(req,limits,deadline,mono):
        assert deadline-mono()==pytest.approx(60-h['payload']['active_seconds'])
        calls.append(1);clock.elapsed=60.;raise ValueError('NCAAF_PILOT_DEADLINE')
    q=stage.acquire(p,a,root=tmp_path,history_bundle=h,prequote=pref,transport=exhausted,wall=clock.wall,monotonic=clock.mono)
    assert len(calls)==1 and q['payload']['status']=='INCOMPLETE' and q['payload']['reason']=='NCAAF_STAGED_DEADLINE'
    assert q['payload']['prior_active_seconds']==h['payload']['active_seconds'] and q['payload']['prior_original_bytes']==h['payload']['original_bytes']
    stage.read_bundle(q,complete=False);stage.verify_journal(tmp_path,q)
    before=(tmp_path/(h['sha256']+'.quote-claim.json')).read_bytes()
    v=deepcopy(p['payload']);v['collection_authorization_ref']='SYNTHETIC-second-quote';p2=stage.seal(v)
    a2,_=f.stage_configuration(p2,monkeypatch,tmp_path);clock.elapsed=0.
    failed=stage.acquire(p2,a2,root=tmp_path,history_bundle=h,prequote=pref,transport=exhausted,wall=clock.wall,monotonic=clock.mono)
    assert failed['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED' and failed['payload']['attempts']==0
    assert len(calls)==1 and (tmp_path/(h['sha256']+'.quote-claim.json')).read_bytes()==before


def test_actual_manual_operator_two_prompts_native_capture_and_same_context_readback(monkeypatch,tmp_path,capsys):
    from test_ncaaf_transport_diagnostics import local_wire
    spec=importlib.util.spec_from_file_location('SYNTHETIC_complete_operator',Path(__file__).resolve().parents[1]/'scripts/ncaaf_reviewed_target_operator.py')
    op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
    f.select(monkeypatch,0);p=f.plan();a,raw=f.stage_configuration(p,monkeypatch,tmp_path)
    paths={name:str(tmp_path/(name+'.json')) for name in ('plan','authorization','configuration')}
    for name,value in [('plan',p),('authorization',a)]:Path(paths[name]).write_bytes(stage.model.encode(value))
    m=dict(version=op.VERSION,**paths,custody_root=str(tmp_path),setup_cutoff='2026-10-17T17:55:00Z',
        latest_start='2026-10-17T18:14:00Z',history=None,prequote=None)
    c=json.loads(raw)['payload'];m['runtime']=target.runtime_receipt(c['runtime']['code_head'],stage.model.digest(m))
    c['runtime']=deepcopy(m['runtime']);c['review']['subject_sha256']=stage.model.digest({k:v for k,v in c.items() if k!='review'})
    raw=stage.model.encode(stage.seal(c));Path(paths['configuration']).write_bytes(raw)
    sealed=stage.model.encode(stage.seal(m));path=tmp_path/'operation.json';path.write_bytes(sealed)
    native=op.runpy.run_path(str(op.ROOT/'scripts/ncaaf_staged_acquisition.py'),run_name='SYNTHETIC_NATIVE_OPERATOR_CLI')
    with pytest.raises(ValueError,match='RUNTIME_UNSUPPORTED'):native['verify_runtime']()
    # Explicit offline injection into the function's actual launch namespace;
    # production still rejects pytest/transport injection before any prompt.
    native['verify_runtime']=lambda:None
    native['main'].__globals__['verify_runtime']=native['verify_runtime']
    monkeypatch.setattr(op.runpy,'run_path',lambda *a,**k:native)
    monkeypatch.setattr(op.sys.stdin,'isatty',lambda:True)
    answers=iter(['INCLUDED','SYNTHETIC-OPERATOR-KEY'])
    monkeypatch.setattr(op.getpass,'getpass',lambda *a:next(answers))
    monkeypatch.setattr(stage.transport,'utc',lambda:'2026-10-17T18:00:01Z')
    mh=hashlib.sha256(sealed).hexdigest();ch=hashlib.sha256(raw).hexdigest()
    with local_wire(monkeypatch,f.bodies(),closing=True) as wire:
        assert op.execute(path,mh,ch)==0 and len(wire['sent'])==4
        report=tmp_path/(p['sha256']+'.operator-report.json');claim=tmp_path/(p['sha256']+'.operator-claim.json')
        saved=report.read_bytes();claimed=claim.read_bytes();result=json.loads(saved)
        assert result['status']=='CAPTURED_UNREVIEWED' and result['readback']=='PASS' and result['provider_attempts']==4
        assert result['configuration_file_sha256']==ch and result['manifest_file_sha256']==mh
        assert result['entitlement_attestation_at']=='2026-10-17T18:00:01Z' and stage.CONTROLLED.get() is None
        with pytest.raises(ValueError,match='ALREADY_SPENT'):op.execute(path,mh,ch)
        assert len(wire['sent'])==4 and report.read_bytes()==saved and claim.read_bytes()==claimed
    assert 'SYNTHETIC-OPERATOR-KEY' not in capsys.readouterr().out
    assert all(b'SYNTHETIC-OPERATOR-KEY' not in q.read_bytes() for q in tmp_path.iterdir() if q.is_file())
