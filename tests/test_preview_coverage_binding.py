"""Labelled SYNTHETIC preview fixtures; no acquisition or historical inference."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from app_core.game_coverage import publication_games
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, validate_package
from app_core.slate_coverage import build_coverage

RUN = '20261008T160000.000000Z'
START = '2026-10-08T23:00:00Z'
FAMILIES = ('overall', 'sides', 'totals')


def fixture(*, league='NFL', home='Atlanta Falcons', away='Carolina Panthers', events=None):
    event = dict(canonical_event_id='synthetic:one', home_team=home, away_team=away,
        home_team_id='synthetic:h', away_team_id='synthetic:a', original_start=START,
        schedule_status='SCHEDULED')
    inventory = dict(version='slate-inventory-v1', league=league, source='synthetic_schedule',
        evidence_label='SYNTHETIC', selected_date='2026-10-08', timezone='America/New_York',
        observed_at='2026-10-08T16:00:00Z', status='COMPLETE',
        completeness_basis='SYNTHETIC complete fixture', reasons=[], events=events or [event])
    report = build_coverage([inventory], selected_date='2026-10-08',
        as_of=inventory['observed_at'], run_id=RUN)
    final = dict(league=league, canonical_event_id=event['canonical_event_id'],
        matchup_id=event['canonical_event_id'], export_run_id=RUN,
        home_team=home, away_team=away, Home=home, Away=away,
        game_start_utc=START, **{'Commence (Local)': START, 'Local Date': '2026-10-08'},
        market_type='spread_home', best_pick=home+' -2.5', spread_line=-2.5,
        odds_american=-110, Bettable=False, Play_Stake=0.0)
    return report, final


def preview(report, finals, candidates=None):
    frame, _ = publication_games(pd.DataFrame(finals), pd.DataFrame(candidates or []), slate_report=report)
    boards = {f: per_game_board(frame, pd.DataFrame(candidates or []), f) for f in FAMILIES}
    package = build_package(**boards)
    validate_package(package)
    return frame, boards, package


@pytest.mark.parametrize('variant', ['canonical', 'case', 'league_case', 'alias', 'duplicate_equivalent', 'timezone', 'eastern_label'])
def test_equivalent_descriptors_build_actual_preview(variant):
    report, final = fixture()
    if variant == 'case': final.update(home_team='ATLANTA FALCONS', Home='atlanta falcons')
    if variant == 'league_case': final['league'] = 'nfl'
    if variant == 'alias': final.update(home_team='Atlanta', Home='Falcons', away_team='Carolina', Away='Panthers')
    if variant == 'duplicate_equivalent': final.update(League='nfl', sport='NFL', home_team='Atlanta', Home='Atlanta Falcons')
    if variant == 'timezone': final['Commence (Local)'] = '2026-10-08T19:00:00-04:00'
    if variant == 'eastern_label': final['Commence (Local)'] = '2026-10-08 7:00 PM ET'
    original = deepcopy(final)
    frame, boards, package = preview(report, [final])
    for family in FAMILIES:
        rows = package['games'][family]
        assert len(rows) == 1
        assert rows[0]['game'] == 'Carolina Panthers at Atlanta Falcons'
        assert rows[0]['sport'] == 'NFL'
        assert pd.Timestamp(rows[0]['start']) == pd.Timestamp(START)
        assert rows[0]['status'] == 'PASS' and boards[family].iloc[0]['Play_Stake'] == 0
        assert 'coverage_origin_descriptors' not in rows[0]
        assert 'coverage_candidate_descriptors' not in rows[0]
    # Presentation does not rewrite the input or original producer descriptors.
    assert final == original
    for field in ('Home', 'Away', 'home_team', 'away_team', 'game_start_utc', 'Commence (Local)'):
        assert frame.iloc[0][field] == original[field]


@pytest.mark.parametrize('home,away', [('LSU', 'McNeese'), ('Louisiana State Tigers', 'McNeese State Cowboys')])
def test_reviewed_ncaaf_aliases_use_canonical_presentation(home, away):
    report, final = fixture(league='NCAAF', home='LSU Tigers', away='McNeese Cowboys')
    final.update(home_team=home, Home=home, away_team=away, Away=away)
    _, _, package = preview(report, [final])
    assert all(g[0]['game'] == 'McNeese Cowboys at LSU Tigers' for g in package['games'].values())


@pytest.mark.parametrize('changes,field', [
    ({'Home': 'Carolina Panthers'}, 'Home'),
    ({'Away': 'Atlanta Falcons'}, 'Away'),
    ({'League': 'NCAAF'}, 'League'),
    ({'home_team': 'Carolina Panthers', 'Home': 'Carolina Panthers',
      'away_team': 'Atlanta Falcons', 'Away': 'Atlanta Falcons'}, 'home_team'),
    ({'Commence (Local)': '2026-10-09T00:00:00Z'}, 'Commence (Local)'),
    ({'game_start_utc': '2026-10-09T00:00:00Z', 'Commence (Local)': '2026-10-09T00:00:00Z'}, 'game_start_utc'),
    ({'schedule_event_id': 'synthetic:alien'}, 'schedule_event_id'),
])
def test_conflicting_descriptors_reject_with_private_first_field(changes, field):
    report, final = fixture(); final.update(changes)
    with pytest.raises(ValueError, match='^COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT$') as error:
        preview(report, [final])
    diagnostic = error.value.diagnostic
    assert diagnostic['canonical_event_id'] == 'synthetic:one'
    assert diagnostic['field'] == field
    assert diagnostic['expected'] != diagnostic['actual']
    assert diagnostic['board_category'] in {'publication_rows', 'overall'}


def test_selected_candidate_start_conflict_cannot_replace_schedule():
    report, final = fixture()
    candidate = dict(final, candidate_id='SYNTHETIC-candidate', best_available_family_rank=1,
        best_available_rank=1, game_start_utc='2026-10-09T00:00:00Z')
    with pytest.raises(ValueError, match='COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT') as error:
        preview(report, [final], [candidate])
    assert error.value.diagnostic['field'] == 'game_start_utc'
    assert error.value.diagnostic['board_category'] == 'sides'


@pytest.mark.parametrize('bad_clock', ['2026-11-01 1:30 AM ET', '2026-03-08 2:30 AM ET', 'not a clock', '2026-10-08T23:00:00'])
def test_ambiguous_nonexistent_malformed_or_naive_start_rejects_precisely(bad_clock):
    report, final = fixture(); final['Commence (Local)'] = bad_clock
    with pytest.raises(ValueError, match='COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT') as error:
        preview(report, [final])
    assert error.value.diagnostic['field'] == 'Commence (Local)'
    assert error.value.diagnostic['actual'] == bad_clock


@pytest.mark.parametrize('candidate_id', ['synthetic:one', ''])
def test_selected_candidate_equivalent_labels_and_clock_keep_exact_quote(candidate_id):
    report, final = fixture()
    candidate = dict(final, matchup_id=candidate_id, candidate_id='SYNTHETIC-exact-candidate',
        quote_id='SYNTHETIC-exact-quote', home_team='Atlanta', Home='Falcons',
        away_team='Carolina', Away='Panthers', game_start_utc='2026-10-08T19:00:00-04:00',
        market_type='spread_away', best_pick='Carolina Panthers +2.5', spread_line=2.5, odds_american=-115,
        best_available_family_rank=1, best_available_rank=1)
    _, boards, package = preview(report, [final], [candidate])
    assert boards['sides'].iloc[0]['candidate_id'] == 'SYNTHETIC-exact-candidate'
    assert boards['sides'].iloc[0]['quote_id'] == 'SYNTHETIC-exact-quote'
    assert boards['sides'].iloc[0]['coverage_candidate_descriptors']['game_start_utc'] == candidate['game_start_utc']
    assert all(r[0]['game'] == 'Carolina Panthers at Atlanta Falcons' for r in package['games'].values())


def test_foreign_candidate_never_lends_metrics_to_scheduled_event():
    report, final = fixture()
    candidate = dict(final, matchup_id='synthetic:alien', home_team='Other', Home='Other',
        best_pick='Other -2.5', candidate_id='SYNTHETIC-rejected-foreign', odds_american=123,
        calibrated_probability=.99, production_win_probability=.99, best_available_family_rank=1)
    _, boards, _ = preview(report, [final], [candidate])
    assert boards['sides'].iloc[0]['candidate_id'] != candidate['candidate_id']
    assert boards['sides'].iloc[0]['odds'] != candidate['odds_american']
    assert boards['sides'].iloc[0]['win_probability'] != .99
    assert boards['sides'].iloc[0]['Play_Stake'] == 0


def test_unknown_start_placeholder_remains_visible_without_inference():
    event = dict(canonical_event_id='synthetic:unknown-start', home_team='Atlanta Falcons',
        away_team='Carolina Panthers', home_team_id='h', away_team_id='a',
        original_start=None, schedule_status='SCHEDULED')
    report, _ = fixture(events=[event])
    _, _, package = preview(report, [])
    for group in package['games'].values():
        assert len(group) == 1 and group[0]['start'] is None
        assert group[0]['win_estimate'] is None and group[0]['status'] == 'PASS'


@pytest.mark.parametrize('missing_clock', [None, pd.NaT, pd.NA])
def test_missing_original_final_clock_retains_placeholder_not_candidate_metrics(missing_clock):
    report, final = fixture()
    final.pop('Commence (Local)')
    final.update(Home='ATLANTA FALCONS', game_start_utc=missing_clock,
        production_win_probability=.99, calibrated_probability=.99, odds_american=185, Play_Stake=100)
    _, boards, package = preview(report, [final])
    assert final['Home'] == 'ATLANTA FALCONS' and final['game_start_utc'] is missing_clock
    for family in FAMILIES:
        row = package['games'][family][0]
        assert row['record_kind'] == 'coverage_only_no_inference'
        assert row['win_estimate'] is None and row['odds'] is None and row['ev'] is None
        assert row['as_of'] is None and row['status'] == 'PASS'
        assert row['coverage_decision']['coverage_decision_state'] == 'UNVERIFIED'
        assert boards[family].iloc[0]['Play_Stake'] == 0


@pytest.mark.parametrize('changes', [{'Home': 'Carolina Panthers'}, {'football_identity_status': 'CONFLICT'}])
def test_missing_clock_does_not_hide_a_separate_identity_contradiction(changes):
    report, final = fixture(); final.pop('Commence (Local)'); final.pop('game_start_utc')
    final.update(changes)
    with pytest.raises(ValueError, match='COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT'):
        preview(report, [final])


def test_repeated_matchups_and_placeholders_keep_every_scheduled_event_once():
    report, final = fixture()
    first = dict(canonical_event_id='synthetic:one', home_team='Atlanta Falcons', away_team='Carolina Panthers',
        home_team_id='synthetic:h', away_team_id='synthetic:a', original_start=START, schedule_status='SCHEDULED')
    second = dict(first, canonical_event_id='synthetic:two', original_start='2026-10-09T01:00:00Z')
    report, _ = fixture(events=[first, second])
    _, _, package = preview(report, [final])
    for group in package['games'].values():
        assert {r['coverage_decision']['canonical_event_id'] for r in group} == {'synthetic:one', 'synthetic:two'}
        assert len(group) == 2
        placeholder = group[1]
        assert placeholder['record_kind'] == 'coverage_only_no_inference'
        assert placeholder['win_estimate'] is None and placeholder['odds'] is None and placeholder['ev'] is None
        assert placeholder['status'] == 'PASS' and placeholder['pick'] == ''


def test_ambiguous_same_instant_matchup_rejects_without_borrowing_candidate():
    report, final = fixture()
    first = dict(canonical_event_id='synthetic:one', home_team='Atlanta Falcons', away_team='Carolina Panthers',
        home_team_id='synthetic:h', away_team_id='synthetic:a', original_start=START, schedule_status='SCHEDULED')
    report, _ = fixture(events=[first, dict(first, canonical_event_id='synthetic:two')])
    final.pop('canonical_event_id'); final['matchup_id'] = 'unknown-provider-id'
    with pytest.raises(ValueError, match='COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT') as error:
        preview(report, [final])
    assert error.value.diagnostic['canonical_event_id'] == ''
    assert error.value.diagnostic['field'] == 'event_match'


def test_public_validator_still_rejects_tampered_canonical_descriptor():
    report, final = fixture(); _, _, package = preview(report, [final])
    package['games']['totals'][0]['game'] = 'Other at Atlanta Falcons'
    # Other frozen package contracts can reject tampering even earlier. Exercise
    # the unchanged coverage boundary against these actual package rows too.
    from app_core.slate_coverage import validate_public_coverage
    with pytest.raises(ValueError): validate_package(package)
    with pytest.raises(ValueError, match='COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT') as error:
        validate_public_coverage(package['slate_coverage'], package['games'])
    assert error.value.diagnostic['board_category'] == 'totals'
    assert error.value.diagnostic['field'] == 'game'


def test_diagnostic_excludes_private_inputs_and_credentials():
    report, final = fixture()
    final.update(Home='Carolina Panthers', credentials='SECRET', raw_dependencies={'private': 'SECRET'},
        probability=.987, provider_quotes='SECRET', token='SECRET')
    with pytest.raises(ValueError) as error: preview(report, [final])
    serialized = json.dumps(error.value.diagnostic)
    assert 'SECRET' not in serialized and '.987' not in serialized
    assert 'raw_dependencies' not in serialized and 'provider_quotes' not in serialized


def failure_app():
    from test_preview_coverage_binding import fixture
    from app_core.game_coverage import publication_games
    from app.ui.publish_panel import render_publish_panel
    import pandas as pd
    report, final = fixture()
    final.update(Home='Carolina Panthers', token='PRIVATE-SYNTHETIC-INPUT',
        raw_dependencies={'private': 'PRIVATE-SYNTHETIC-INPUT'})
    frame, _ = publication_games(pd.DataFrame([final]), pd.DataFrame(), slate_report=report)
    render_publish_panel(frame, pd.DataFrame(), lazy_history=True)


def test_failure_diagnostic_exists_before_package_and_only_after_owner_gate(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import (canonical_evidence_download, remote_canonical_download,
        source_evidence_panel, ncaaf_pipeline_research, activation_panel, public_results)
    for panel in (canonical_evidence_download, remote_canonical_download,
                  source_evidence_panel, ncaaf_pipeline_research, activation_panel):
        monkeypatch.setattr(panel, 'render', lambda *args: None)
    monkeypatch.setattr(public_results, 'render_history', lambda *args, **kwargs: [])
    token = 'SYNTHETIC-owner-token-12345'
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', token)
    app = AppTest.from_function(failure_app).run()
    assert not app.exception and not app.json and not app.get('download_button')
    app.text_input(key='publication_token').set_value('wrong').run()
    assert not app.json and not app.get('download_button')
    app.text_input(key='publication_token').set_value(token).run()
    assert not app.exception
    assert any('COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT' in e.value for e in app.error)
    value = json.loads(app.json[-1].value)
    assert value['field'] == 'Home' and value['canonical_event_id'] == 'synthetic:one'
    assert token not in app.json[-1].value and 'PRIVATE-SYNTHETIC-INPUT' not in app.json[-1].value
    assert app.get('download_button')
    assert 'publication_preview' not in app.session_state


def test_actual_browser_retains_canonical_alias_event_and_no_stakes(tmp_path):
    from datetime import datetime
    from test_research_probability_browser import inspect_browser
    report, final = fixture()
    final.update(home_team='Atlanta', Home='FALCONS', away_team='Carolina', Away='PANTHERS')
    _, _, package = preview(report, [final])
    result = inspect_browser(package, tmp_path/'synthetic-alias-preview', datetime.fromisoformat('2026-10-08T16:00:00+00:00'))
    assert result['overall'] == result['sides'] == result['totals'] == 1
    assert result['initial']['current'] == 0
    assert all(r['status'] == 'PASS' and r['stake'] == 0 for r in result['initial']['saved'])
    assert any('Carolina Panthers at Atlanta Falcons' in card for card in result['initial']['cards'])


def test_daily_consumers_keep_schedule_visible_and_publish_panel_reachable(monkeypatch):
    from app.ui.daily_dashboard import _render_game_board
    import streamlit as st
    report, final = fixture(); final['Home'] = 'Carolina Panthers'
    frame, _ = publication_games(pd.DataFrame([final]), pd.DataFrame(), slate_report=report)
    shown, errors = [], []
    monkeypatch.setattr(st, 'dataframe', lambda data, **kwargs: shown.append(data))
    monkeypatch.setattr(st, 'error', lambda error: errors.append(error))
    for family in FAMILIES: _render_game_board(frame, pd.DataFrame(), family)
    assert len(shown) == len(errors) == 3
    assert all(view.Event.tolist() == ['synthetic:one'] for view in shown)
    assert all(view['Wager status'].tolist() == ['PASS'] for view in shown)
    assert all(error.endswith('COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT') for error in errors)
    assert all('Carolina Panthers' not in error for error in errors)
