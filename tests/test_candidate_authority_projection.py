"""Real expanded production shape, with test-only policy and exposure."""
import pandas as pd
import pytest
from dataclasses import replace
from activation_fixture import setup, NOW
from core.streamlit_pipeline import build_best_picks_df


def source(tmp_path):
    row, policy, config = setup(tmp_path / 'ledger.db')
    row.update(home_team='Indianapolis Colts', away_team='Baltimore Ravens', selection='Indianapolis Colts -2.5', best_pick='Indianapolis Colts -2.5', game_date='2026-09-14', game_start_utc=row['start'],
               calibrated_probability=.62, model_probability=.62, expected_value=.1,
               edge=.1, market_probability=.5, odds_source='DraftKings',
               line_source='live', market_line_source='live', candidate_id='original')
    return row, policy, config


def build(rows, monkeypatch):
    monkeypatch.setattr('core.empirical_tiers.load_bucket_stats', lambda: {})
    monkeypatch.setattr('core.probability_calibration.load_calibration', lambda: None)
    diagnostics = {}
    best = build_best_picks_df(pd.DataFrame(rows), diagnostics_out=diagnostics)
    return best, diagnostics


def test_real_projection_retains_authority(tmp_path, monkeypatch):
    row, policy, config = source(tmp_path)
    best, diagnostics = build([row], monkeypatch)
    frame = diagnostics.get('candidate_authority_df', diagnostics['candidate_audit_df'])
    assert frame.iloc[0].get('team_ids') == row['team_ids']

    from app_core.candidate_evidence_schema import PRIVATE_AUTHORITY_FIELDS
    for field in set(PRIVATE_AUTHORITY_FIELDS) & set(row):
        if field not in {'matchup_id', 'best_pick'}:
            assert frame.iloc[0][field] == row[field], field
    from core.prospective_uncertainty import prepare_live
    from core.live_wager_contract import finalize_live_wagers
    prepared = prepare_live(frame, database=tmp_path/'evidence.db', plan_dir=tmp_path/'plans', now=NOW)
    out, audit = finalize_live_wagers(prepared, best, 1000, now=NOW,
        policies={'NFL':policy}, config=config, reviews=prepared)
    assert out.iloc[0]['production_bet_amount'] > 0, (out.to_dict('records'), audit)
    assert bool(out.iloc[0]['production_eligible'])


@pytest.mark.parametrize('change', [
    {'team_ids': None}, {'model_validated': None}, {'calibration_validated': False},
    {'quote_time': '2026-09-14T13:00:00+00:00'}, {'unvalidated': True},
])
def test_real_projection_still_fails_closed(tmp_path, monkeypatch, change):
    from core.prospective_uncertainty import prepare_live
    from core.live_wager_contract import finalize_live_wagers
    row, policy, config = source(tmp_path)
    if change.get('unvalidated'):
        policy = replace(policy, deployment_state='UNVALIDATED')
    else:
        row.update(change)
        if change.get('team_ids', 'present') is None:
            del row['team_ids']
    best, diag = build([row], monkeypatch)
    frame = diag['candidate_authority_df']
    if 'team_ids' not in row:
        assert 'team_ids' not in frame
    prepared = prepare_live(frame, database=tmp_path/'evidence.db', plan_dir=tmp_path/'plans', now=NOW)
    out, audit = finalize_live_wagers(prepared, best, 1000, now=NOW,
        policies={'NFL':policy}, config=config, reviews=prepared)
    assert out.iloc[0]['production_bet_amount'] == 0
    if 'team_ids' not in row:
        assert 'missing_stable_team_ids' in out.iloc[0]['production_gate_reason']


def test_real_runner_up_capture_and_boundary(tmp_path, monkeypatch):
    import sqlite3
    from io import StringIO
    from core.prospective_uncertainty import prepare_live
    from core.live_wager_contract import finalize_live_wagers, PUBLIC_FIELDS
    from app_core.prediction_evidence import begin_run, capture_run
    row, policy, config = source(tmp_path)
    runner = dict(row, candidate_id='runner', market_type='spread_away', line=2.5,
        spread_line=2.5, selection='Baltimore Ravens +2.5', best_pick='Baltimore Ravens +2.5',
        calibrated_probability=.60, model_probability=.60)
    row['identity_verified'] = False
    moneyline = dict(runner, candidate_id='moneyline', market_type='moneyline_away')
    row['api_secret_not_for_export'] = 'SENTINEL'
    best, diag = build([row, runner, moneyline], monkeypatch)
    private = diag['candidate_authority_df']
    assert set(private.candidate_id) == {'original', 'runner'}
    assert set(private.candidate_id) == set(diag['candidate_audit_df'].candidate_id)
    assert 'team_ids' not in diag['candidate_audit_df']
    assert 'api_secret_not_for_export' not in private
    assert 'team_ids' not in PUBLIC_FIELDS
    prepared = prepare_live(private, database=tmp_path/'evidence.db', plan_dir=tmp_path/'plans', now=NOW)
    out, audit = finalize_live_wagers(prepared, best, 1000, now=NOW,
        policies={'NFL':policy}, config=config, reviews=prepared)
    assert out.iloc[0]['candidate_id'] == 'runner'
    assert 0 < out.iloc[0]['production_bet_amount'] <= 2.5
    prepared['best_available_selected'] = prepared.candidate_id.eq('runner')
    context = begin_run({}, path=tmp_path/'saved.db')
    saved, card = capture_run(context, prepared, out, pd.DataFrame([row, runner]), path=tmp_path/'saved.db', authoritative_candidates=True)
    assert set(saved.candidate_id) == {'original', 'runner'}
    with sqlite3.connect(tmp_path/'saved.db') as db:
        stored = pd.read_csv(StringIO(db.execute('SELECT candidates FROM snapshots').fetchone()[0]))
    assert set(stored.candidate_id) == set(prepared.candidate_id)
    for field in ('evidence_snapshot_id', 'evidence_frozen_at',
                  'model_version', 'calibration_version', 'conservative_probability'):
        assert stored.set_index('candidate_id')[field].to_dict() == prepared.set_index('candidate_id')[field].to_dict()
    for field in ('team_ids', 'exact_quote_verified', 'quote_time', 'identity_verified'):
        assert saved.set_index('candidate_id')[field].to_dict() == prepared.set_index('candidate_id')[field].to_dict()


def test_private_contract_covers_adapter_and_gate_inputs():
    import ast
    import inspect
    from app_core.candidate_evidence_schema import PRIVATE_AUTHORITY_FIELDS
    from core import live_wager_contract, wager_decisions
    from core.candidate_maturity import INPUTS
    required = set(INPUTS)
    # Direct producer inputs read by the adapter and deterministic decision gate.
    for function in (live_wager_contract.adapt_candidate, wager_decisions.candidate_decision):
        for node in ast.walk(ast.parse(inspect.getsource(function))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'row'
                    and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant)):
                required.add(node.args[0].value)
    assert required <= set(PRIVATE_AUTHORITY_FIELDS)


def test_real_builder_through_live_pipeline(tmp_path, monkeypatch):
    import streamlit_app as app
    from app_core import prediction_evidence as evidence
    from core import live_wager_contract as authority, prospective_uncertainty as uncertainty
    row, policy, config = source(tmp_path)
    monkeypatch.setattr('core.empirical_tiers.load_bucket_stats', lambda: {})
    monkeypatch.setattr('core.probability_calibration.load_calibration', lambda: None)
    monkeypatch.setattr(app, 'run_analysis_pipeline', lambda **kwargs: (pd.DataFrame([row]), pd.DataFrame(), {}))
    monkeypatch.setattr(app, 'optimize_portfolio_allocation', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(app, 'generate_parlays', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(app, 'run_bankroll_simulation', lambda *a, **k: {})
    monkeypatch.setattr(app, '_enrich_with_kalshi_safe', lambda df: (df, None))
    monkeypatch.setattr(app, '_recompute_consensus_from_kalshi', lambda df, require_ml=False: df)
    real_prepare = uncertainty.prepare_live
    def prepare(frame):
        assert frame.iloc[0]['team_ids'] == row['team_ids']
        return real_prepare(frame, database=tmp_path/'prior.db', plan_dir=tmp_path/'plans', now=NOW)
    monkeypatch.setattr(uncertainty, 'prepare_live', prepare)
    real_finalize = authority.finalize_live_wagers
    monkeypatch.setattr(authority, 'finalize_live_wagers', lambda frame, best, bankroll:
        real_finalize(frame, best, bankroll, now=NOW, policies={'NFL': policy}, config=config, reviews=frame))
    real_begin, real_capture = evidence.begin_run, evidence.capture_run
    monkeypatch.setattr(evidence, 'begin_run', lambda controls: real_begin(controls, path=tmp_path/'saved.db'))
    monkeypatch.setattr(evidence, 'capture_run', lambda *a, **k: real_capture(*a, **k, path=tmp_path/'saved.db'))
    state, _, _ = app._run_pipeline({'sports':['NFL'], 'use_ml':False, 'theover_spreads':None,
        'theover_totals':None, 'bankroll':1000., 'use_gemini':False})
    diag = state['diagnostics']
    assert diag['prediction_snapshot_saved'], diag.get('prediction_snapshot_error')
    assert state['best_picks_df'].iloc[0]['production_bet_amount'] == 2.5
    assert diag['candidate_authority_df'].iloc[0]['team_ids'] == row['team_ids']
    assert set(diag['candidate_audit_df'].candidate_id) == set(diag['candidate_authority_df'].candidate_id)


def test_generated_identity_includes_exact_book_and_quote():
    from app_core.candidate_evidence_schema import authority_projection
    rows = [dict(game_id='event', sport='NFL', market_type='spread_home', line=-2.5,
        selection='Home -2.5', odds_american=-110, book=book, quote_time=time)
        for book, time in [('DraftKings', '2026-09-14T14:55:00Z'),
                           ('FanDuel', '2026-09-14T14:55:00Z'),
                           ('DraftKings', '2026-09-14T14:56:00Z')]]
    frame = authority_projection(pd.DataFrame(rows), [])
    assert frame.candidate_id.nunique() == 3
    reverse = authority_projection(pd.DataFrame(rows[::-1]), [])
    assert list(frame.candidate_id) == list(reverse.candidate_id)[::-1]
