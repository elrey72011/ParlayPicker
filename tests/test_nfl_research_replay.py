"""Synthetic NFL producer/capture/export replay; no provider, fitting or publication."""
from copy import deepcopy
from io import StringIO
import json
import sqlite3
import pandas as pd
import pytest
from app_core.research_replay import retain_export, read_export, frame_from_payload, frame_payload, encode, digest
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from scripts.benchmark_drive_history_loading import blocked_network
from test_estimate_availability import produced
from test_research_probability_producer import real_path
from test_research_probability_display import NOW, QUOTE


def nfl(**changes):
    return produced(league="NFL",market_type="total_under",best_pick="Under 51.5",total_line=51.5,
        live_total_line=51.5,odds_american=100,calibrated_probability=.54,model_probability=.54,
        expected_value=.08,feature_home_ppg=24.0,feature_away_ppg=22.0,
        feature_home_oppg=21.0,feature_away_oppg=23.0,
        feature_home_games_played=7,feature_away_games_played=7,**changes)


def route(monkeypatch,tmp_path,raw,*,captured_changes=None):
    with blocked_network():
        result=real_path(monkeypatch,tmp_path,raw)
        if captured_changes:
            # Corrupt the retained source consumed by the actual exporter. The
            # original immutable snapshot remains available for comparison.
            result["captured"]=result["captured"].copy()
            for field,value in captured_changes.items(): result["captured"][field]=value
            result["frames"]=[per_game_board(result["card"],result["captured"],family=f,novig_only=True)
                              for f in ("overall","sides","totals")]
            result["package"]=build_package(*result["frames"])
            from scripts.publish_board import render
            result["html"]=render(result["package"])
        receipt=retain_export(result["frames"],result["package"],result["card"],result["captured"],
                              path=tmp_path/"isolated-evidence.sqlite3")
        retained,sources=read_export(receipt["export_id"],path=tmp_path/"isolated-evidence.sqlite3")
    for family in ("overall","totals"):
        row=result["package"]["games"][family][0]
        assert row["status"]=="PASS"
        assert row.get("wager_contract",{}).get("production_bet_amount",0)==0
        assert row["win_estimate"] is None and row["ev"] is None
    return result,receipt,retained,sources


def test_actual_producer_and_private_replay_preserve_both_probabilities(monkeypatch,tmp_path):
    raw=nfl();original=deepcopy(raw)
    result,receipt,retained,sources=route(monkeypatch,tmp_path,raw)
    assert raw==original
    assert receipt["source_boundary"]=="RETAINED"
    assert retained["package_hash"]==digest(encode(result["package"]))
    saved=next(iter(sources.values()))
    before=frame_from_payload(saved["original"]["candidates"]).iloc[0]
    captured=frame_from_payload(saved["captured_candidates"])
    card=frame_from_payload(saved["captured_card"])
    assert before.ml_probability==raw["ml_probability"]
    assert before.calibrated_probability==.54
    assert before.ml_probability!=before.calibrated_probability
    assert before.ml_estimate_metadata==raw["ml_estimate_metadata"]
    assert "secret_unrelated_payload" not in before
    assert "MUST-NOT-LEAK" not in json.dumps(retained)
    assert "MUST-NOT-LEAK" not in json.dumps(saved)
    with blocked_network():
        replay=[per_game_board(card,captured,family=f,novig_only=True) for f in ("overall","sides","totals")]
        package=build_package(*replay)
    for family,frame in zip(("overall","sides","totals"),replay):
        assert frame_payload(frame)==retained["boards"][family]
    assert package==retained["package"]
    exported=pd.read_csv(StringIO(retained["per_game_csv"]["overall"])).iloc[0]
    trace=json.loads(exported.research_estimate_trace)
    assert trace["source"]["ml_probability"]["value"]==raw["ml_probability"]
    assert trace["source"]["best_available_probability"]["value"]==.54
    assert trace["missing_identity_fields"]==[]
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"]==pytest.approx(.54) and display["ev"]==pytest.approx(.08)
    assert display["inference_status"]=="UNKNOWN"  # Origin success belongs only to the raw model.
    assert "research_estimate_trace" not in json.dumps(result["package"])
    from test_research_probability_browser import inspect_browser
    browser=inspect_browser(result["package"],tmp_path/"browser",NOW,rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"]==pytest.approx(.54)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0


@pytest.mark.parametrize("field,value",[("quote_time","2026-10-01T19:00:00Z"),
    ("quote_timestamp","2026-10-01T19:00:00Z"),("selected_quote_recorded_at","invalid-clock"),
    ("quote_source","DraftKings"),("sportsbook","FanDuel")])
def test_conflicting_original_quote_facts_reject_at_actual_export(monkeypatch,tmp_path,field,value):
    result,_,retained,_=route(monkeypatch,tmp_path,nfl(),captured_changes={field:value})
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="ESTIMATE_IDENTITY_MISMATCH"
    assert display["probability"] is None and display["ev"] is None
    assert retained["package"]["games"]["overall"][0]["research_display"]==display


@pytest.mark.parametrize("mutation,reason",[("malformed","ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ("missing","ESTIMATE_PROVENANCE_NOT_RECORDED"),("contradiction","ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ("failed","INFERENCE_FAILED"),("target","ESTIMATE_IDENTITY_MISMATCH"),
    ("probability","ESTIMATE_IDENTITY_MISMATCH"),("push","UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ("event","ESTIMATE_IDENTITY_MISMATCH"),("candidate","ESTIMATE_IDENTITY_MISMATCH"),
    ("start","ESTIMATE_IDENTITY_MISMATCH"),("provider","ESTIMATE_IDENTITY_MISMATCH")])
def test_original_origin_corruption_never_becomes_a_blended_success(monkeypatch,tmp_path,mutation,reason):
    raw=nfl();metadata=json.loads(raw["ml_estimate_metadata"])
    if mutation=="malformed": raw["ml_estimate_metadata"]="{invalid"
    elif mutation=="missing": raw.pop("ml_estimate_metadata")
    else:
        if mutation in {"contradiction","failed"}: metadata["inference_status"]="failed"
        if mutation=="failed": raw["ml_inference_status"]="failed"
        if mutation=="target": metadata["target"]["value"]="total_over"
        if mutation=="probability": metadata["probability"]["value"]+=.1
        if mutation=="push": metadata["push_probability"]=.1
        if mutation in {"event","candidate","start","provider"}:
            key={"event":"matchup_id","candidate":"candidate_id","start":"game_start_utc","provider":"provider_event_id"}[mutation]
            metadata["identity"][key]={"state":"VALUE","value":"2026-10-01T23:00:00Z" if mutation=="start" else "other"}
        raw["ml_estimate_metadata"]=json.dumps(metadata)
    result,_,_,sources=route(monkeypatch,tmp_path,raw)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]==reason
    assert display["probability"] is None and display["ev"] is None
    saved=next(iter(sources.values()))
    before=frame_from_payload(saved["original"]["candidates"]).iloc[0]
    if mutation=="missing": assert "ml_estimate_metadata" not in before or pd.isna(before.ml_estimate_metadata)
    else: assert before.ml_estimate_metadata==raw["ml_estimate_metadata"]


@pytest.mark.parametrize("fields,missing",[(('quote_id',),['quote_id']),
    (('market_period','period'),['period']),(('settlement_rules',),['rules'])])
def test_missing_identity_is_explicit_and_never_reconstructed(monkeypatch,tmp_path,fields,missing):
    raw=nfl()
    for field in fields: raw.pop(field,None)
    result,_,retained,sources=route(monkeypatch,tmp_path,raw)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="ESTIMATE_PROVENANCE_NOT_RECORDED"
    trace=json.loads(result["frames"][0].iloc[0].research_estimate_trace)
    assert trace["missing_identity_fields"]==missing
    original=frame_from_payload(next(iter(sources.values()))["original"]["producer"]).iloc[0]
    for field in fields: assert field not in original
    assert json.loads(pd.read_csv(StringIO(retained["per_game_csv"]["overall"])).iloc[0].research_estimate_trace)==trace


def test_negative_ev_retains_and_replays_without_authority(monkeypatch,tmp_path):
    raw=nfl();raw.update(calibrated_probability=.4,expected_value=-.2)
    result,_,_,_=route(monkeypatch,tmp_path,raw)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"]==pytest.approx(.4) and display["ev"]==pytest.approx(-.2)


def test_private_tables_are_append_only_and_not_remote_tables(monkeypatch,tmp_path):
    _,receipt,_,_=route(monkeypatch,tmp_path,nfl())
    from app_core.evidence_remote import TABLES
    assert not {'research_replay_sources','research_replay_exports'} & set(TABLES)
    db=sqlite3.connect(tmp_path/"isolated-evidence.sqlite3")
    for table in ('research_replay_sources','research_replay_exports'):
        for statement in (f'DELETE FROM {table}',f'UPDATE {table} SET payload=payload'):
            with pytest.raises(sqlite3.IntegrityError,match='append-only'): db.execute(statement)
    db.execute('DROP TRIGGER immutable_research_replay_exports_UPDATE')
    db.execute('UPDATE research_replay_exports SET payload=?',('{}',));db.commit();db.close()
    with pytest.raises((ValueError,KeyError)):
        read_export(receipt['export_id'],path=tmp_path/"isolated-evidence.sqlite3")


def test_unknown_original_source_is_retained_as_unknown(monkeypatch,tmp_path):
    result,_,_,_=route(monkeypatch,tmp_path,nfl())
    with blocked_network():
        receipt=retain_export(result['frames'],result['package'],result['card'],result['captured'],path=tmp_path/'other.sqlite3')
        retained,sources=read_export(receipt['export_id'],path=tmp_path/'other.sqlite3')
    assert receipt['source_boundary']=='UNKNOWN' and sources=={}
    assert all(link['source_hash'] is None for link in retained['source_links'])


def preview_app():
    import streamlit as st
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(st.session_state['input_games'],st.session_state['input_candidates'])


@pytest.mark.parametrize('fail',[False,True])
def test_actual_preview_retains_evidence_and_reports_write_failure(monkeypatch,tmp_path,fail):
    result,_,_,_=route(monkeypatch,tmp_path,nfl())
    import shutil
    shutil.copyfile(tmp_path/'isolated-evidence.sqlite3',tmp_path/'evidence.sqlite3')
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR',str(tmp_path))
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','offline-private-preview-token')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    monkeypatch.setattr('app.ui.public_results.render_history',lambda *a,**k:[])
    if fail:
        def denied(*args,**kwargs): raise OSError('synthetic disk failure')
        monkeypatch.setattr('app_core.research_replay.retain_export',denied)
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(preview_app)
    at.session_state['input_games']=result['card']
    at.session_state['input_candidates']=result['captured']
    with blocked_network():
        at.run()
        at.text_input(key='publication_token').set_value('offline-private-preview-token').run()
    assert not at.exception
    saved=at.session_state['publication_preview']
    assert saved['package']['games']['overall'][0]['status']=='PASS'
    if fail:
        assert saved['research_replay_error']=='synthetic disk failure'
        assert any('Private research replay evidence was not saved' in w.value for w in at.warning)
    else:
        receipt=saved['research_replay_receipt']
        assert receipt['source_boundary']=='RETAINED'
        retained,sources=read_export(receipt['export_id'],path=tmp_path/'evidence.sqlite3')
        assert sources and retained['package']==saved['package']
        assert 'research_estimate_trace' in retained['per_game_csv']['overall']


def test_capture_replay_write_failure_rolls_back_both_records(monkeypatch,tmp_path):
    def fail(*args,**kwargs): raise OSError('synthetic disk failure')
    monkeypatch.setattr('app_core.research_replay.retain_source',fail)
    with blocked_network(),pytest.raises(OSError,match='synthetic disk failure'):
        real_path(monkeypatch,tmp_path,nfl())
    db=sqlite3.connect(tmp_path/'isolated-evidence.sqlite3')
    assert db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM research_replay_sources').fetchone()[0]==0
    db.close()



def unpack_bundle(data):
    from io import BytesIO
    from zipfile import ZipFile
    with ZipFile(BytesIO(data)) as archive:
        return {name:archive.read(name) for name in archive.namelist()}


def test_downloaded_replay_reproduces_rejection_and_verifies_receipts(monkeypatch,tmp_path):
    from app_core.research_replay import download_bundle
    import hashlib
    raw=nfl();raw['ml_estimate_metadata']='{invalid'
    with blocked_network():
        result,receipt,retained,sources=route(monkeypatch,tmp_path,raw)
        data,verified=download_bundle(receipt,expected_package_hash=digest(encode(result['package'])),
                                     path=tmp_path/'isolated-evidence.sqlite3')
        again,_=download_bundle(receipt,expected_package_hash=receipt['package_hash'],
                                path=tmp_path/'isolated-evidence.sqlite3')
    assert data==again
    files=unpack_bundle(data)
    assert json.loads(files['receipt.json'])==verified
    assert hashlib.sha256(files['export.json']).hexdigest()==receipt['export_id']
    assert hashlib.sha256(files['package.json']).hexdigest()==receipt['package_hash']
    assert all(hashlib.sha256(files[name]).hexdigest()==value for name,value in verified['files'].items())
    assert json.loads(files['export.json'])==retained
    assert json.loads(files['package.json'])==result['package']
    link=verified['source_links'][0]
    assert link['state']=='RETAINED' and link['snapshot_payload_hash']
    saved=json.loads(files[link['source_file']])
    assert saved==sources[link['snapshot_id']]
    assert saved['export_run_id']==link['export_run_id']
    original=frame_from_payload(saved['original']['candidates']).iloc[0]
    assert original.ml_estimate_metadata=='{invalid'
    captured=frame_from_payload(saved['captured_candidates']);card=frame_from_payload(saved['captured_card'])
    with blocked_network():
        replay=[per_game_board(card,captured,family=f,novig_only=True) for f in ('overall','sides','totals')]
        assert build_package(*replay)==json.loads(files['package.json'])
    for family,frame in zip(('overall','sides','totals'),replay):
        assert files['per-game/'+family+'.csv']==retained['per_game_csv'][family].encode()
        assert frame_payload(frame)==retained['boards'][family]
        if not frame.empty and family in ('overall','totals'):
            trace=json.loads(pd.read_csv(StringIO(files['per-game/'+family+'.csv'].decode())).iloc[0].research_estimate_trace)
            assert trace['display']['availability_reason']=='ESTIMATE_PROVENANCE_NOT_RECORDED'
    assert b'MUST-NOT-LEAK' not in b''.join(files.values())
    assert 'research_estimate_trace' not in files['package.json'].decode()
    with blocked_network(),pytest.raises(ValueError,match='receipt/package mismatch'):
        download_bundle(receipt,expected_package_hash='0'*64,path=tmp_path/'isolated-evidence.sqlite3')


def owner_download_app(monkeypatch,tmp_path,*,retained=True):
    result,_,_,_=route(monkeypatch,tmp_path,nfl())
    if retained:
        import shutil
        shutil.copyfile(tmp_path/'isolated-evidence.sqlite3',tmp_path/'evidence.sqlite3')
    token='offline-owner-download-token'
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR',str(tmp_path))
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN',token)
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    monkeypatch.setattr('app.ui.public_results.render_history',lambda *a,**k:[])
    import streamlit as st
    from app_core import research_replay
    downloads=[];reads=[]
    original_download=st.download_button;original_bundle=research_replay.download_bundle
    def record_download(*args,**kwargs):
        if kwargs.get('key')=='download_private_research_replay': downloads.append(args[1])
        return original_download(*args,**kwargs)
    def record_read(*args,**kwargs):
        reads.append(args[0])
        return original_bundle(*args,**kwargs)
    monkeypatch.setattr(st,'download_button',record_download)
    monkeypatch.setattr(research_replay,'download_bundle',record_read)
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(preview_app)
    at.session_state['input_games']=result['card'];at.session_state['input_candidates']=result['captured']
    return at,token,downloads,reads


def private_downloads(at):
    return [item for item in at.get('download_button')
            if item.proto.label=='Download private research replay bundle']


@pytest.mark.parametrize('retained',[True,False])
def test_owner_download_authorization_visibility_and_unknown_links(monkeypatch,tmp_path,retained):
    with blocked_network():
        at,token,downloads,reads=owner_download_app(monkeypatch,tmp_path,retained=retained)
        at.run()
        assert not at.exception and not private_downloads(at) and not reads and not downloads
        at.text_input(key='publication_token').set_value('wrong').run()
        assert not at.exception and not private_downloads(at) and not reads and not downloads
        at.text_input(key='publication_token').set_value(token).run()
        assert not at.exception and len(private_downloads(at))==1 and len(reads)==len(downloads)==1
        assert private_downloads(at)[0].proto.url
        saved=at.session_state['publication_preview']
        files=unpack_bundle(downloads[0]);receipt=json.loads(files['receipt.json'])
        assert json.loads(files['package.json'])==saved['package']
        assert receipt['export_id']==saved['research_replay_receipt']['export_id']
        assert receipt['package_hash']==digest(encode(saved['package']))
        assert receipt['source_boundary']==('RETAINED' if retained else 'UNKNOWN')
        assert all(l['state']==receipt['source_boundary'] for l in receipt['source_links'])
        assert saved['package']['games']['overall'][0]['status']=='PASS'
        assert 'research_estimate_trace' not in json.dumps(saved['package'])
        assert 'research_estimate_trace' in files['per-game/overall.csv'].decode()
        assert token.encode() not in b''.join(files.values())
        if not retained:
            assert not any(name.startswith('sources/') for name in files)
            assert all(l['source_hash'] is l['source_file'] is l['snapshot_payload_hash'] is None
                       for l in receipt['source_links'])
            assert any('bundle preserves UNKNOWN' in w.value for w in at.warning)
        at.text_input(key='publication_token').set_value('').run()
        assert not at.exception and not private_downloads(at) and len(reads)==len(downloads)==1
        monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','')
        at.run()
        assert not at.exception and not private_downloads(at) and len(reads)==len(downloads)==1


@pytest.mark.parametrize('tamper',['export','source','snapshot','receipt'])
def test_owner_download_integrity_failure_hides_archive_without_repair(monkeypatch,tmp_path,tamper):
    import hashlib
    with blocked_network():
        at,token,downloads,reads=owner_download_app(monkeypatch,tmp_path)
        at.run();at.text_input(key='publication_token').set_value(token).run()
        assert not at.exception and private_downloads(at)
        downloads.clear()
        saved=at.session_state['publication_preview']
        target=tmp_path/'evidence.sqlite3'
        if tamper=='receipt':
            saved['research_replay_receipt']['package_hash']='0'*64
            at.session_state['publication_preview']=saved
        else:
            db=sqlite3.connect(target)
            if tamper=='export':
                db.execute('DROP TRIGGER immutable_research_replay_exports_UPDATE')
                raw=db.execute('SELECT payload FROM research_replay_exports WHERE export_id=?',
                               (saved['research_replay_receipt']['export_id'],)).fetchone()[0]
                db.execute('UPDATE research_replay_exports SET payload=? WHERE export_id=?',
                           (raw+' ',saved['research_replay_receipt']['export_id']))
            elif tamper=='source':
                db.execute('DROP TRIGGER immutable_research_replay_sources_UPDATE')
                db.execute('UPDATE research_replay_sources SET payload=payload||? ',(' ',))
            else:
                triggers=db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='snapshots' AND upper(sql) LIKE '%BEFORE UPDATE%'").fetchall()
                assert triggers
                for (name,) in triggers: db.execute('DROP TRIGGER '+name)
                db.execute('UPDATE snapshots SET decisions=?',('[]',))
            db.commit();db.close()
        before=hashlib.sha256(target.read_bytes()).hexdigest()
        package_before=deepcopy(saved['package'])
        at.run()
        assert not at.exception and not private_downloads(at) and not downloads
        assert any('Private research replay download is unavailable' in w.value for w in at.warning)
        assert at.session_state['publication_preview']['package']==package_before
        assert hashlib.sha256(target.read_bytes()).hexdigest()==before
