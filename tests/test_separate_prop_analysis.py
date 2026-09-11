import ast
from pathlib import Path
import pandas as pd
import pytest


def setup_props(monkeypatch, *, fail=False):
    from app_core import weights_config, prop_runner, best_duos, gemini_bet_gate
    from core import streamlit_pipeline
    from app_core.odds_api import TheOddsAPIClient
    monkeypatch.setattr(weights_config,'ENABLE_STRIKEOUT_PROPS_PRODUCTION',True)
    monkeypatch.setattr(weights_config,'ENABLE_NFL_PLAYER_PROPS',False)
    monkeypatch.setattr(streamlit_pipeline,'_get_odds_api_key',lambda:'test-key')
    monkeypatch.setattr(TheOddsAPIClient,'__init__',lambda self,**kw:None)
    calls=[]
    def build(*args,**kw):
        calls.append('props')
        if fail:raise RuntimeError('provider failure')
        return pd.DataFrame([{'league':'MLB','best_pick':'Player Over 0.5 Hits',
            'Stake_Status':'Research / No Stake','prediction_generated_at':'2026-09-10T12:00:00Z'}])
    monkeypatch.setattr(prop_runner,'build_prop_card',build)
    monkeypatch.setattr(gemini_bet_gate,'apply_gemini_bet_gate',lambda card,**kw:card)
    monkeypatch.setattr(best_duos,'build_tiered_prop_parlays',lambda *a,**kw:pd.DataFrame())
    monkeypatch.setattr(streamlit_pipeline,'run_analysis_pipeline',lambda **kw:pytest.fail('prop run fetched games'))
    return calls


def test_prop_run_preserves_games_and_original_quote_time(monkeypatch):
    calls=setup_props(monkeypatch)
    from app_core.prop_analysis import run_prop_analysis
    old_games=pd.DataFrame([{'pick':'saved game'}])
    state={'best_picks_df':old_games,'diagnostics':{'game':'saved'}}
    updates=run_prop_analysis({'sports':['MLB'],'bankroll':1000,'use_gemini':False},state)
    state.update(updates)
    assert calls==['props'] and state['best_picks_df'] is old_games
    assert state['diagnostics']=={'game':'saved'}
    assert updates['strikeout_prop_card'].iloc[0]['prediction_generated_at']=='2026-09-10T12:00:00Z'
    assert updates['strikeout_prop_card'].iloc[0]['export_run_id']
    assert 'MLB player props' in updates['props_diagnostics']['stage_seconds']


def test_prop_failure_does_not_replace_previous_results(monkeypatch):
    setup_props(monkeypatch,fail=True)
    from app_core.prop_analysis import run_prop_analysis
    old=pd.DataFrame([{'pick':'old prop'}]);state={'strikeout_prop_card':old}
    with pytest.raises(RuntimeError):
        run_prop_analysis({'sports':['MLB'],'bankroll':1000,'use_gemini':False},state)
    assert state['strikeout_prop_card'] is old


def test_game_pipeline_has_no_prop_fetch_or_prop_state_update():
    tree=ast.parse(Path('streamlit_app.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_pipeline')
    names={n.id for n in ast.walk(function) if isinstance(n,ast.Name)}
    assert not {'build_prop_card','run_prop_analysis','run_gemini_prop_analysis','strikeout_prop_card'} & names
    keys={n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert 'strikeout_prop_card' not in keys
    assert 'Save prediction evidence' in keys


def test_gemini_gate_remains_enabled_for_explicit_prop_run(monkeypatch):
    setup_props(monkeypatch)
    from app_core import gemini_bet_gate, prop_runner
    from integrations import gemini_client
    from app_core.prop_analysis import run_prop_analysis
    calls=[]
    def review(card,state):calls.append('review');return card
    def gate(card,**kwargs):
        assert kwargs['enabled'] and kwargs['product']=='prop'
        calls.append('gate');return card
    monkeypatch.setattr(gemini_client,'run_gemini_prop_analysis',review)
    monkeypatch.setattr(gemini_bet_gate,'apply_gemini_bet_gate',gate)
    monkeypatch.setattr(prop_runner,'apply_prop_stake_status',lambda card:card)
    run_prop_analysis({'sports':['MLB'],'bankroll':1000,'use_gemini':True},{})
    assert calls==['review','gate']


def test_publishing_old_props_with_new_games_keeps_prop_age():
    from tests.test_public_board import boards
    from app_core.public_board import build_package
    props=pd.DataFrame([{'player':'Player','matchup':'A at B','best_pick':'Over 0.5',
        'export_run_id':'20260907T120000Z','prediction_generated_at':'2026-09-07T11:59:00Z'}])
    package=build_package(*boards(),props=props)
    assert package['props'][0]['as_of']=='2026-09-07T11:59:00+00:00'
    assert package['props'][0]['as_of']!=package['games']['overall'][0]['as_of']
