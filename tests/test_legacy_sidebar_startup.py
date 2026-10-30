"""Actual application startup; synthetic settings, no live provider or inference."""
import builtins
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest
from scripts.benchmark_drive_history_loading import blocked_network

ROOT=Path(__file__).resolve().parents[1]
TOKEN='SYNTHETIC-owner-for-sidebar-regressions-only'
LABEL='Show legacy prospective and market tools'
LEGACY={'app.ui.ncaaf_prospective','app.ui.mlb_prospective','app.ui.nfl_market'}

@pytest.fixture(autouse=True)
def offline(monkeypatch,tmp_path):
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR',str(tmp_path/'SYNTHETIC-evidence'))
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN',TOKEN)
    monkeypatch.setenv('PARLAYPICKER_NFL_ACQUISITION_ENABLED','false')
    monkeypatch.setenv('PARLAYPICKER_NFL_INFERENCE_ENABLED','false')
    from app_core import market_probability_model,ncaaf_research,nfl_scenario_research
    from core import streamlit_pipeline
    for module,name in [(market_probability_model,'predict_market_probabilities'),(ncaaf_research,'centers'),
                        (nfl_scenario_research,'compute'),(streamlit_pipeline,'run_analysis_pipeline')]:
        monkeypatch.setattr(module,name,Mock(side_effect=AssertionError('SYNTHETIC_PREDICTION_FORBIDDEN')))
    with blocked_network():yield

def app(tmp_path,name='SYNTHETIC-main'):
    script=tmp_path/(name+'.py');script.write_text('import streamlit_app\nstreamlit_app.main()\n',encoding='utf-8')
    return AppTest.from_file(str(script),default_timeout=30).run()

def control(at,label):
    found=[x for x in at.checkbox if x.label==label]
    assert len(found)==1,dict(label=label,exceptions=len(at.exception))
    return found[0]

def no_database(*args,**kwargs):raise AssertionError('PASSIVE_DATABASE_ACCESS_FORBIDDEN')

def test_full_app_default_off_defers_legacy_imports_and_all_database_access(monkeypatch,tmp_path):
    native=builtins.__import__;calls=[]
    def guarded(name,*a,**k):
        if name in LEGACY:calls.append(name);raise AssertionError('PASSIVE_LEGACY_IMPORT_FORBIDDEN')
        return native(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded);monkeypatch.setattr(sqlite3,'connect',no_database)
    at=app(tmp_path)
    assert not at.exception and not control(at,LABEL).value and not calls
    assert any(x.label=='Use reviewed private NCAAF inputs for Refresh' for x in at.checkbox)
    assert any(x.label=='Refresh picks' for x in at.button)

def test_owner_navigation_and_saved_controls_do_not_initialize_storage(monkeypatch,tmp_path):
    monkeypatch.setattr(sqlite3,'connect',no_database);at=app(tmp_path)
    at.session_state['SYNTHETIC_unrelated']='keep'
    at.text_input(key='publication_token').input('wrong').run()
    assert not at.exception and not any(x.label=='Show saved NCAAF Refresh results' for x in at.checkbox)
    at.text_input(key='publication_token').input(TOKEN).run()
    assert not at.exception
    assert any(x.label=='Show saved NCAAF Refresh results' for x in at.checkbox)
    assert any(x.label=='Enable guided private NFL workflow' for x in at.checkbox)
    assert not control(at,LABEL).value and at.session_state['SYNTHETIC_unrelated']=='keep'
    at.text_input(key='publication_token').input('wrong').run()
    assert not at.exception and not any(x.label=='Show saved NCAAF Refresh results' for x in at.checkbox)
    assert at.session_state['SYNTHETIC_unrelated']=='keep'

def test_existing_legacy_files_do_not_cause_passive_or_swallowed_initialization(monkeypatch,tmp_path):
    from app_core import ncaaf_prospective_store,mlb_prospective_store,nfl_market_store
    for store in (ncaaf_prospective_store,mlb_prospective_store,nfl_market_store):
        store.connect().close()  # Explicitly owned empty SYNTHETIC fixture setup.
    calls=[]
    def forbidden(*a,**k):calls.append('INITIALIZATION_ATTEMPT');raise AssertionError('PASSIVE_DATABASE_ACCESS_FORBIDDEN')
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    at=app(tmp_path);at.run()
    assert not at.exception and not control(at,LABEL).value and not calls

@pytest.mark.parametrize('sport',['NCAAF','NFL'])
def test_sport_switch_rerender_and_fresh_session_remain_inactive(monkeypatch,tmp_path,sport):
    monkeypatch.setattr(sqlite3,'connect',no_database);at=app(tmp_path)
    next(x for x in at.selectbox if x.label=='Select Sport').set_value(sport).run()
    assert not at.exception and not control(at,LABEL).value
    at.run();assert not at.exception and not control(at,LABEL).value
    fresh=app(tmp_path,'SYNTHETIC-fresh-'+sport)
    assert not fresh.exception and not control(fresh,LABEL).value

def test_explicit_activation_uses_only_isolated_legacy_stores_then_disabling_stops_access(monkeypatch,tmp_path):
    native=sqlite3.connect;calls=[];allowed=(tmp_path/'SYNTHETIC-evidence').resolve()
    def isolated(database,*a,**k):
        path=Path(database).resolve()
        assert path.parent==allowed,'UNAUTHORIZED_DATABASE_WRITE'
        calls.append(path.name);return native(database,*a,**k)
    monkeypatch.setattr(sqlite3,'connect',isolated)
    at=app(tmp_path);assert not at.exception and not calls
    control(at,LABEL).check().run()
    assert not at.exception
    assert set(calls)=={'ncaaf-prospective.sqlite3','mlb-prospective.sqlite3','nfl-market.sqlite3'}
    assert any(x.label=='Refresh current NCAAF inputs' for x in at.button)
    assert any(x.label=='Refresh upcoming MLB games' for x in at.button)
    before=list(calls);control(at,LABEL).uncheck().run();at.run()
    assert not at.exception and calls==before
    from app_core import market_probability_model,ncaaf_research,nfl_scenario_research
    assert not market_probability_model.predict_market_probabilities.called
    assert not ncaaf_research.centers.called and not nfl_scenario_research.compute.called

def test_enabling_legacy_section_does_not_set_private_execution_flags(monkeypatch,tmp_path):
    from app.ui import ncaaf_prospective,mlb_prospective,nfl_market
    calls=[]
    for module,name in [(ncaaf_prospective,'render_ncaaf_prospective'),(mlb_prospective,'render_mlb_prospective'),(nfl_market,'render_nfl_market')]:
        monkeypatch.setattr(module,name,lambda name=name:calls.append(name))
    monkeypatch.setattr(sqlite3,'connect',no_database);at=app(tmp_path)
    control(at,LABEL).check().run()
    assert not at.exception and len(calls)==3
    assert not at.session_state.filtered_state.get('ncaaf_private_refresh',False)
    assert not at.session_state.filtered_state.get('nfl_workflow_enabled',False)
    assert at.session_state.filtered_state.get('pipeline_status')=='idle'
