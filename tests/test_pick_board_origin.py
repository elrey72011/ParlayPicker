"""Synthetic offline actual-pipeline regressions; no provider or wager authority."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from app_core import source_contract as adapter, odds_api, research_replay
from app_core.current_wagers_trace import _output_match, _selected_output_records
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from core import streamlit_pipeline as sp
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_contract_pipeline import fixture, pipeline, exported, NOW, INFERENCE, START
from test_research_probability_browser import inspect_browser


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


@pytest.fixture
def captured(monkeypatch, tmp_path):
    game, _ = fixture(monkeypatch)
    analysis = pipeline(monkeypatch, game)
    result = exported(monkeypatch, tmp_path, analysis)
    selected = result['package']['board_diagnostics']['traces'][0]['source_candidate_id']
    row = next(r for r in result['captured'].to_dict('records') if r['candidate_id'] == selected)
    return result, row, analysis


def test_actual_pipeline_originating_run_is_not_inference_clock(captured):
    result, row, analysis = captured
    package = result['package']; original = deepcopy(package)
    match = _output_match(row, package)
    assert match['status'] == 'MATCHED', match
    bound = match['selected_identity']
    assert bound['run_id'] == row['export_run_id']
    assert bound['inference_timestamp'] == INFERENCE != bound['run_id']
    assert bound['event_id'] == row['matchup_id'] and bound['quote_id'] == row['quote_id']
    assert row['ml_probability'] == next(x for x in analysis.to_dict('records')
        if x['market_type'] == row['market_type'])['ml_probability']
    assert row['ml_probability'] != row['best_available_probability']
    assert result['card'].iloc[0].wager_contract['production_bet_amount'] == 0
    assert not result['captured'].production_eligible.fillna(False).any()
    assert package == original


@pytest.mark.parametrize('field,value', [
    ('export_run_id', 'other-exact-run'), ('run_id', 'conflicting-run-alias'),
    ('candidate_id', 'other-candidate'), ('quote_id', 'other-exact-quote'),
    ('spread_line', 4.5), ('spread_line', 3.5000000001),
    ('odds_american', 105), ('odds_american', 104.0000000001),
    ('matchup_id', 'unrelated-event'), ('best_pick', 'Away +3.5'),
    ('opposing_odds_source', 'other-book'), ('quote_observed_at', 'not-a-clock'),
    ('quote_id', ' whitespace-quote'), ('export_run_id', ' padded-run ')])
def test_actual_pipeline_changed_candidate_identity_rejects(captured, field, value):
    result, row, _ = captured
    row = dict(row, **{field: value})
    assert _output_match(row, result['package'])['status'] != 'MATCHED'


@pytest.mark.parametrize('field', ['export_run_id', 'candidate_id', 'quote_id', 'matchup_id'])
def test_actual_pipeline_missing_candidate_identity_is_unresolved(captured, field):
    result, row, _ = captured
    row = dict(row); row.pop(field, None)
    if field == 'matchup_id':
        for alias in ('canonical_event_id', 'game_id'): row.pop(alias, None)
    assert _output_match(row, result['package'])['status'] == 'UNRESOLVED'


@pytest.mark.parametrize('attack', ['missing_display', 'missing_run', 'wrong_display_candidate',
    'wrong_diagnostic_candidate', 'wrong_diagnostic_quote', 'wrong_diagnostic_line',
    'wrong_diagnostic_price', 'wrong_diagnostic_event', 'padded_diagnostic_candidate',
    'padded_diagnostic_quote', 'invalid_schema', 'duplicate'])
def test_actual_pipeline_changed_output_binding_rejects(captured, attack):
    result, row, _ = captured; package = deepcopy(result['package'])
    output = package['games']['overall'][0]; display = output['research_display']
    trace = package['board_diagnostics']['traces'][0]
    if attack == 'missing_display': output.pop('research_display')
    elif attack == 'missing_run': display['identity']['export_run_id'] = ''
    elif attack == 'wrong_display_candidate': display['identity']['candidate_id'] = 'different-candidate'
    elif attack == 'wrong_diagnostic_candidate': trace['source_candidate_id'] = 'different-candidate'
    elif attack == 'wrong_diagnostic_quote': trace['quote_id'] = 'different-quote'
    elif attack == 'wrong_diagnostic_line': trace['line'] = 4.5
    elif attack == 'wrong_diagnostic_price': trace['odds'] = 105
    elif attack == 'wrong_diagnostic_event': trace['game_id'] = 'unrelated-event'
    elif attack == 'padded_diagnostic_candidate': trace['source_candidate_id'] = ' ' + trace['source_candidate_id']
    elif attack == 'padded_diagnostic_quote': trace['quote_id'] = ' ' + display['identity']['quote_id']
    elif attack == 'invalid_schema': display['version'] = 'unsupported-display-v9'
    elif attack == 'duplicate':
        package['games']['overall'].append(deepcopy(output))
        package['board_diagnostics']['traces'].append(deepcopy(trace))
    assert _output_match(row, package)['status'] == 'UNRESOLVED'


def test_runs_with_identical_offer_and_clocks_never_match(captured):
    result, row, _ = captured
    original = deepcopy(result['package']); other = dict(row, export_run_id='other-opaque-run')
    assert _output_match(other, original)['reason'] == 'EXPLICIT_IDENTITY_CONFLICT:run_id'
    # A timestamp representation that denotes the same instant is not a run alias.
    row = dict(row, export_run_id='2026-10-06T19:26:00Z')
    package = deepcopy(original)
    package['games']['overall'][0]['research_display']['identity']['export_run_id'] = '2026-10-06T19:26:00+00:00'
    assert _output_match(row, package)['status'] != 'MATCHED'
    assert original == result['package']


def test_actual_pipeline_complete_source_replay_browser_and_unknown_source(monkeypatch, tmp_path):
    game, _ = fixture(monkeypatch)
    analysis = pipeline(monkeypatch, game)
    complete = exported(monkeypatch, tmp_path/'complete', analysis)
    receipt = research_replay.retain_export(complete['frames'], complete['package'], complete['card'],
        complete['captured'], path=complete['db'])
    saved, sources = research_replay.read_export(receipt['export_id'], path=complete['db'])
    source = next(iter(sources.values()))
    frames = [per_game_board(research_replay.frame_from_payload(source['captured_card']),
        research_replay.frame_from_payload(source['captured_candidates']), family=f, novig_only=True)
        for f in ('overall', 'sides', 'totals')]
    assert build_package(*frames) == saved['package'] == complete['package']
    display = complete['package']['games']['overall'][0]['research_display']
    assert display['availability_reason'] == 'AVAILABLE'
    assert display['probability'] == complete['authority'].iloc[0].best_available_probability
    assert display['ev'] is None and display['break_even_probability'] is None
    assert display['value_reason'] == 'SETTLEMENT_VALUE_UNSUPPORTED'
    browser = inspect_browser(complete['package'], tmp_path/'complete-browser', NOW)
    assert browser['initial']['shown'][0]['probability'] == display['probability']
    assert browser['initial']['shown'][0]['ev'] is None
    assert browser['initial']['current'] == browser['initial']['top'] == 0
    monkeypatch.setattr(adapter, 'ACCEPTED_LISTINGS', {})
    missing_analysis = pipeline(monkeypatch, game)
    for field in ('ml_probability', 'calibrated_probability', 'expected_value'):
        assert analysis[field].tolist() == missing_analysis[field].tolist()
    missing = exported(monkeypatch, tmp_path/'missing', missing_analysis)
    output = missing['package']['games']['overall'][0]
    assert output['research_display']['availability_reason'] != 'AVAILABLE'
    # Deliberately populated legacy values cannot escape a rejecting object.
    from app_core.board_diagnostics import _trace, _digest
    output.update(win_estimate=.99, ev=.8, estimated_price_edge=.2, break_even_probability=.49)
    from app_core.price_value_display import display as legacy_value
    output.update(legacy_value(output['win_estimate'], output['odds'], output['ev'],
        push_probability=output.get('price_push_probability')))
    diagnostic = missing['package']['board_diagnostics']
    prior = diagnostic['traces'][0]
    diagnostic['traces'][0] = _trace({'matchup_id':prior['game_id'],
        'candidate_id':prior['source_candidate_id'], 'line':prior['line']}, output,
        pd.Timestamp(missing['package']['built_at']).to_pydatetime(), missing['package']['stale_after_minutes'])
    diagnostic['selected_rows_hash'] = _digest(missing['package']['games']['overall'])
    browser = inspect_browser(missing['package'], tmp_path/'missing-browser', NOW)
    shown = browser['initial']['shown'][0]
    assert all(shown[k] is None for k in ('probability', 'ev', 'edge', 'breakEven'))
    assert 'Market period not verified' in browser['initial']['cards'][0]
    assert 'Settlement rules not verified' in browser['initial']['cards'][0]
    assert browser['initial']['current'] == browser['initial']['top'] == 0


def mixed_pipeline(monkeypatch, *, declared=False):
    # Install the existing boundary mocks; the actual provider extraction,
    # predictor, blend, candidate authority and terminal gate are retained.
    game, _ = fixture(monkeypatch); pipeline(monkeypatch, game)
    games = []
    for key, home, away, kind, line in [
        ('baseball_mlb', 'Minnesota Twins', 'Pittsburgh Pirates', 'spreads', 1.5),
        ('icehockey_nhl', 'Boston Bruins', 'Ottawa Senators', 'spreads', 1.5),
        ('americanfootball_nfl', 'New Orleans Saints', 'Atlanta Falcons', 'totals', 47.5)]:
        outcomes = [dict(name=home if kind=='spreads' else 'Over', point=line, price=104),
                    dict(name=away if kind=='spreads' else 'Under', point=-line if kind=='spreads' else line, price=-104)]
        games.append(dict(id='synthetic-'+key, sport_key=key, home_team=home, away_team=away,
            matchup_id=f'{home}|{away}|2026-10-06',
            commence_time=START, bookmakers=[dict(key='novig', last_update='2026-10-06T19:00:00Z',
            markets=[dict(key=kind, last_update='2026-10-06T19:24:00Z', outcomes=outcomes)])]))
    if declared:
        for g in games:
            if g['sport_key']!='icehockey_nhl':
                g['bookmakers'][0]['markets'][0].update(period='full_game', settlement_rules='synthetic-unverified-rule-claim')
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, sport, date=None): return [deepcopy(g) for g in games if g['sport_key']==sport]
    monkeypatch.setattr(odds_api, 'TheOddsAPIClient', Client)
    def features(frame, *args):
        out=frame.copy(); league=out.league.str.upper(); out['League']=league
        for key,value in dict(feature_home_ppg=24., feature_away_ppg=22., feature_home_oppg=21.,
                feature_away_oppg=23., feature_home_games_played=7, feature_away_games_played=7,
                feature_home_win_pct=.55, feature_away_win_pct=.45, feature_diff_last5=.1,
                ml_feature_eligible=True, stats_resolution_status='resolved').items(): out[key]=value
        for key,value in dict(feature_home_ppg=4.5, feature_away_ppg=4., feature_home_oppg=3.8,
                feature_away_oppg=4.1).items(): out.loc[league=='MLB',key]=value
        return out
    monkeypatch.setattr('app_core.feature_processing.enrich_with_model_features', features)
    original=deepcopy(games)
    analysis,_,diagnostics=sp.run_analysis_pipeline(sports=['MLB','NHL','NFL'],use_ml=True,max_rows=30)
    assert analysis.groupby('league').size().to_dict()=={'MLB':2,'NFL':4,'NHL':2}
    assert diagnostics['market_specific_ml_predictions']==4
    # The two quote-less NFL spread placeholders add no independent events.
    assert games==original
    return analysis


def test_actual_mlb_nfl_total_missing_source_and_nhl_model_replay_browser(monkeypatch,tmp_path):
    analysis=mixed_pipeline(monkeypatch)
    result=exported(monkeypatch,tmp_path,analysis)
    assert len(result['package']['games']['overall'])==3
    for output in result['package']['games']['overall']:
        d=output['research_display']
        assert output['status']=='PASS' and d['probability'] is None and d['ev'] is None
        assert d['availability_reason']==('INFERENCE_UNAVAILABLE' if output['sport']=='NHL' else 'SOURCE_CONTRACT_NOT_VERIFIED')
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db']);source=next(iter(sources.values()))
    replay=research_replay.frame_from_payload(source['captured_candidates'])
    assert replay.ml_probability.fillna(-1).tolist()==result['captured'].ml_probability.fillna(-1).tolist()
    assert replay.best_available_probability.tolist()==result['captured'].best_available_probability.tolist()
    assert not replay.production_eligible.fillna(False).any()
    replay_card=research_replay.frame_from_payload(source['captured_card'])
    assert all(c['production_bet_amount']==0 for c in replay_card.wager_contract)
    for row in replay.to_dict('records'):
        if row['candidate_id'] in {t['source_candidate_id'] for t in saved['package']['board_diagnostics']['traces']}:
            assert _output_match(row,saved['package'])['status']=='MATCHED'
    browser=inspect_browser(saved['package'],tmp_path/'mixed-browser',NOW)
    cards=browser['initial']['cards']
    assert sum('No NHL spread/total model configured' in c for c in cards)==1
    assert sum('Market period not verified' in c and 'Settlement rules not verified' in c for c in cards)==2
    assert all('Price value: Research value unavailable: ' in c for c in cards)
    assert all('Price value: POSITIVE ESTIMATED VALUE' not in c and 'Price value: NEGATIVE ESTIMATED VALUE' not in c for c in cards)
    assert all(all(s[k] is None for k in ('probability','ev','edge','breakEven')) for s in browser['initial']['shown'])
    assert browser['initial']['current']==browser['initial']['top']==0


@pytest.mark.parametrize('sport,market,selection', [
    ('baseball_mlb','spreads','Home'),('americanfootball_nfl','totals','Under')])
def test_nfl_spread_receipt_cannot_be_rehashed_into_other_scope(monkeypatch,sport,market,selection):
    game,catalog=fixture(monkeypatch);game['sport_key']=sport
    m=game['bookmakers'][0]['markets'][0];m['key']=market;m['outcomes'][0]['name']=selection
    o=m['outcomes'][0];receipt=catalog[o['source_contract_ref']]['receipt']
    receipt['identity']=adapter.identity(game,game['bookmakers'][0],m,o)
    receipt['reference_team']=receipt['identity']['selection']
    catalog[o['source_contract_ref']]['sha256']=adapter.digest(receipt)
    result=adapter.adapt(game,game['bookmakers'][0],m,o)['source_contract']
    assert result['status']=='REJECTED' and 'SOURCE_SCOPE_UNSUPPORTED' in result['diagnostics']


def test_source_template_hashes_and_transport_claims_never_admit_exact_listing(monkeypatch,tmp_path):
    baseline=mixed_pipeline(monkeypatch)
    declared=mixed_pipeline(monkeypatch,declared=True)
    for field in ('ml_probability','calibrated_probability','expected_value'):
        assert baseline[field].fillna(-1).tolist()==declared[field].fillna(-1).tolist()
    result=exported(monkeypatch,tmp_path,declared)
    for output in result['package']['games']['overall']:
        if output['sport']=='NHL':continue
        d=output['research_display']
        assert d['availability_reason']=='SOURCE_CONTRACT_NOT_VERIFIED'
        assert all(d[k] is None for k in ('probability','ev','edge','break_even_probability'))
    for row in result['captured'].to_dict('records'):
        if row['league']=='NHL':continue
        origin=json.loads(row['ml_estimate_metadata']);bound=origin['producer_contract']['source_contract']
        assert bound['status']=='UNKNOWN' and bound['receipt'] is None
        assert 'SOURCE_MARKET_LISTING_BINDING_NOT_VERIFIED' in bound['diagnostics']
        assert adapter.replay(bound,origin['generated_at'])
        corrupted=deepcopy(bound);corrupted['status']='VERIFIED'
        assert 'SOURCE_CAPTURE_BINDING_CONFLICT' in adapter.replay(corrupted,origin['generated_at'])
    assert adapter.ACCEPTED_LISTINGS.keys()=={'synthetic-home','synthetic-away'}


def test_missing_optional_run_alias_is_not_a_contradiction(captured):
    result,row,_=captured
    assert _output_match(dict(row,run_id=pd.NA),result['package'])['status']=='MATCHED'
    assert _output_match(dict(row,run_id=float('nan')),result['package'])['status']=='MATCHED'


@pytest.mark.parametrize('padding',[' ','\t','\n'])
def test_opaque_candidate_id_is_never_whitespace_normalized(captured,padding):
    result,row,_=captured
    for changed in (padding+row['candidate_id'],row['candidate_id']+padding):
        match=_output_match(dict(row,candidate_id=changed),result['package'])
        assert match['status']=='UNRESOLVED'
        assert match['reason']=='EXPLICIT_IDENTITY_CONFLICT:source_candidate_id'
