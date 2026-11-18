# Re-issued on 2026-11-18 from tests_archive/test_separate_prop_analysis.py without the tests of the removed app_core/prop_analysis.py; the tests kept are unchanged.
import ast
from pathlib import Path
import pandas as pd


def test_game_pipeline_has_no_prop_fetch_or_prop_state_update():
    tree=ast.parse(Path('streamlit_app.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_pipeline')
    names={n.id for n in ast.walk(function) if isinstance(n,ast.Name)}
    assert not {'build_prop_card','run_prop_analysis','run_gemini_prop_analysis','strikeout_prop_card'} & names
    keys={n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert 'strikeout_prop_card' not in keys
    assert 'Save prediction evidence' in keys


def test_publishing_old_props_with_new_games_keeps_prop_age():
    from tests.test_public_board import boards
    from app_core.public_board import build_package
    props=pd.DataFrame([{'player':'Player','matchup':'A at B','best_pick':'Over 0.5',
        'export_run_id':'20260907T120000Z','prediction_generated_at':'2026-09-07T11:59:00Z'}])
    package=build_package(*boards(),props=props)
    assert package['props'][0]['as_of']=='2026-09-07T11:59:00+00:00'
    assert package['props'][0]['as_of']!=package['games']['overall'][0]['as_of']
