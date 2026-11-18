import json
from datetime import date
import pandas as pd
import pytest
from app_core.pick_accuracy import build_accuracy_report, render_accuracy_markdown
from test_selector_validation import candidates


def fixture():
    f = candidates()
    f['blend_in_kalshi'] = .55
    f['blend_in_ml'] = .6
    f['blend_in_theover'] = .7
    f['ml_probability_source'] = 'score-distribution-v1:mlb'
    f['ml_target'] = f.market_type
    return f


def report(f=None):
    return build_accuracy_report(fixture() if f is None else f, evaluation_start='2026-09-03')


def test_paired_rankings_use_calibrated_probability_and_original_odds():
    f = fixture()
    f.loc[f.matchup_id.eq('game-3'), 'calibrated_probability'] = [.3, .7]
    f['selection_probability_used'] = [999, -999]*4
    f['best_available_score'] = f.selection_probability_used
    r = report(f)
    assert r['rankings']['Current ranking']['wins'] == 2
    assert r['rankings']['Probability first']['wins'] == 1
    assert r['rankings']['Sportsbook baseline']['wins'] == 0
    p = r['paired_ranking_changes']['Probability first']
    assert p['changed_picks'] == 1 and p['current_only_wins'] == 1 and p['both_decided'] == 2
    assert p['paired_win_rate_change'] == -.5
    assert r['rankings']['Probability first']['flat_roi'] == 0
    assert not r['live_changes']
    json.dumps(r, allow_nan=False)


def test_ties_do_not_consult_grades_or_input_order():
    f = fixture()
    f.calibrated_probability = .5
    first = report(f)
    f.candidate_outcome = f.candidate_outcome.map({'WIN':'LOSS','LOSS':'WIN'})
    second = report(f.sample(frac=1, random_state=12))
    get = lambda r: {c['matchup_id']: c['Probability first']['pick'] for c in r['choices']}
    assert get(first) == get(second)


def test_sources_use_same_original_tickets_and_never_impute_missing():
    f = fixture()
    f.loc[f.matchup_id.eq('game-3'), 'blend_in_theover'] = None
    f['theover_probability'] = .99  # ambiguous raw input cannot fill the oriented signal
    r = report(f)
    source = next(x for x in r['sources'] if x['league']=='All' and x['source']=='AI Analysis / TheOver')
    assert source['available_games'] == 1 and source['missing_or_unverified_games'] == 1
    assert source['source_metrics']['n'] == source['sportsbook_on_same_tickets']['n'] == 1
    assert source['source_metrics']['brier'] == pytest.approx(.09)
    assert source['sportsbook_on_same_tickets']['brier'] == pytest.approx(.36)
    assert r['all_sources_common_games'] == 1


def test_moneyline_model_cannot_be_scored_as_total_or_cover_model():
    f = fixture()
    f['ml_target'] = 'home_won'
    r = report(f)
    rows = [s for s in r['sources'] if s['source']=='Independent model']
    assert all(s['available_games']==0 for s in rows)


def test_postgame_or_incomplete_pools_are_excluded_not_replaced_by_other_rows():
    f = fixture()
    f.loc[f.matchup_id.eq('game-3'), 'prediction_generated_at'] = '2026-09-04T01:00:00Z'
    f.loc[f.matchup_id.eq('game-4') & f.market_type.eq('total_under'), 'candidate_outcome'] = 'PENDING'
    r = report(f)
    assert r['status']=='insufficient_verified_evidence'
    assert r['validation']['inventory']['eligible_events']==0
    assert r['rankings']['Probability first']['hit_rate'] is None
    assert r['validation']['exclusions']


def test_duplicate_downloads_do_not_inflate_and_pushes_are_not_losses():
    f = fixture()
    f.loc[f.matchup_id.eq('game-3'), 'candidate_outcome']='PUSH'
    r = report(pd.concat([f,f],ignore_index=True))
    assert r['rankings']['Current ranking']['games']==2
    assert r['rankings']['Current ranking']['pushes']==1
    assert r['rankings']['Current ranking']['hit_rate']==1
    assert r['rankings']['Current ranking']['flat_roi']==.5
    assert r['paired_ranking_changes']['Probability first']['both_decided']==1
    assert 'Live ranking and weights are unchanged' in render_accuracy_markdown(r)


def test_gemini_is_review_only_and_requires_pregame_evidence():
    f = fixture()
    f['gemini_reviewed_at'] = f.prediction_generated_at
    f['gemini_agreement'] = 'agree'
    f['gemini_review_model'] = 'fixture'
    f['gemini_review_input_hash'] = 'a'*64
    r = report(f)
    assert len(r['gemini_review'])==2
    assert all(row['Selections']==2 for row in r['gemini_review'])
    assert all('Gemini' not in s['source'] for s in r['sources'])


def test_cli_writes_both_reports_and_source_hashes(tmp_path):
    from scripts.compare_pick_accuracy import main
    f=fixture(); p=tmp_path/'ledger.csv'; f.to_csv(p,index=False)
    target=tmp_path/'accuracy.json'
    assert main(['--audits',str(p),'--evaluation-start','2026-09-03','--output',str(target)])==0
    assert len(json.loads(target.read_text())['input_files'][0]['sha256'])==64
    assert target.with_suffix('.md').exists()


def test_ui_is_on_demand_and_displays_comparison():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("""
from app.ui.pick_accuracy import render_pick_accuracy
from test_selector_validation import candidates
render_pick_accuracy(candidates())
""").run()
    assert not at.exception and len(at.dataframe)==0
    at.checkbox(key='show_pick_accuracy').check().run()
    assert not at.exception
    at.date_input(key='accuracy_evaluation_start').set_value(date(2026,9,3)).run()
    assert not at.exception and len(at.dataframe)==2
    assert at.dataframe[0].value.Games.tolist()==[2,2,2]
