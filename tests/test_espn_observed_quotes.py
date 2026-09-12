"""Research-only observation evidence never becomes provider update evidence."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from app_core import espn_ncaaf_odds as espn
from app_core.prediction_evidence import provider_quotes, bind_quote
from app_core.per_game_boards import espn_observed_quote, per_game_board, public_quote
from app_core.public_board import build_package, validate_package
from app_core.public_history import report
from app_core.locked_picks import lock_candidates, locked_selections
from test_ncaaf_selection import _espn_fcs_payload
from test_per_game_boards import final, quoted_candidate

OBSERVED = '2026-09-11T19:59:00+00:00'
RUN = '20260911T200000.000000Z'
AT = '2026-09-11T20:01:00+00:00'


def observed_candidate():
    row = quoted_candidate('draftkings', league='NCAAF', odds_feed_source=espn.ESPN_FALLBACK_SOURCE)
    quotes = json.loads(row['provider_quotes'])
    for quote in quotes:
        quote.update(recorded_at=None, observed_at=OBSERVED, observation_source=espn.ESPN_FALLBACK_SOURCE)
    row['provider_quotes'] = json.dumps(quotes)
    return row


def test_fetch_capture_survives_serialization_without_production_binding(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(OBSERVED).astimezone(tz or timezone.utc)
    monkeypatch.setattr(espn, 'datetime', Clock)
    class Response:
        def raise_for_status(self): pass
        def json(self): return _espn_fcs_payload()
    calls = []
    def get(*args, **kwargs):
        calls.append(args[0])
        return Response()
    monkeypatch.setattr(espn.requests, 'get', get)
    game = espn.fetch_espn_ncaaf_fcs_odds('2026-08-27')[0]
    raw = provider_quotes(game)
    assert len(calls) == 1  # No per-game timestamp requests or retries.
    assert all(q['observed_at'] == OBSERVED and q['recorded_at'] is None for q in json.loads(raw))
    row = dict(league='NCAAF', odds_feed_source=espn.ESPN_FALLBACK_SOURCE,
               provider_quotes=raw, export_run_id=RUN, market_type='spread_home',
               spread_line=4.5, odds_american=-110, opposing_odds_source='draftkings')
    assert espn_observed_quote(row) == OBSERVED
    assert not bind_quote(row)['quote_binding_verified']
    assert bind_quote(row)['odds_recorded_at'] == ''
    assert row['provider_quotes'] == raw
    def fail(*args, **kwargs): raise RuntimeError('offline')
    monkeypatch.setattr(espn.requests, 'get', fail)
    assert espn.fetch_espn_ncaaf_fcs_odds('2026-08-27') == []


@pytest.mark.parametrize('changes', [
    {'odds_american':-199}, {'total_line':10}, {'market_type':'total_over'},
    {'league':'MLB'}, {'odds_feed_source':'the_odds_api'},
    {'export_run_id':'20260911T203000.000000Z'},
    {'export_run_id':'20260911T195800.000000Z'},
])
def test_observation_requires_exact_quote_scope_and_freshness(changes):
    row = dict(observed_candidate(), **changes)
    assert espn_observed_quote(row) is None


@pytest.mark.parametrize('changes', [
    {'observed_at':None}, {'observed_at':'2026-09-11T19:59:00'},
    {'observation_source':'other'}, {'book':'fanduel'}, {'recorded_at':'2026-09-11T18:00:00Z'},
])
def test_missing_unaware_or_mislabeled_evidence_is_not_accepted(changes):
    row = observed_candidate()
    quotes = json.loads(row['provider_quotes'])
    for q in quotes: q.update(changes)
    row['provider_quotes'] = json.dumps(quotes)
    assert espn_observed_quote(row) is None


def test_observed_research_lock_preserves_price_provenance_and_expiry():
    board = pd.DataFrame([final(league='NCAAF',export_run_id=RUN,game_time_est='2026-09-11 7:00 PM ET')])
    row = observed_candidate()
    audit = pd.DataFrame([row])
    frames = [per_game_board(board,audit,f,novig_only=True,college_fallback=True) for f in ('overall','sides','totals')]
    package = build_package(*frames)
    validate_package(package)
    leg = package['games']['overall'][0]
    assert leg['quote_time_basis'] == 'espn_observed' and leg['quote_time'] == OBSERVED
    assert leg['status'] == 'PASS' and not package['parlays']
    locks = lock_candidates(package, AT)
    assert len(locks) == 1 and locks[0]['legs'][0]['odds'] == -105
    original = deepcopy(locks)
    assert len(lock_candidates(package,'2026-09-11T20:29:00+00:00')) == 1
    assert lock_candidates(package,'2026-09-11T20:29:01+00:00') == []
    assert lock_candidates(package,'2026-09-11T23:00:00+00:00') == []
    results = report([],[],locks=locks)
    assert results[0]['quote_time_basis'] == 'espn_observed'
    assert results[0]['quote_time'] == OBSERVED
    package.update(schema_version=5, results=results)
    validate_package(package)
    leg.update(odds=-120,quote_time='2026-09-11T20:00:00+00:00')
    assert locked_selections(locks) == original
    invalid = deepcopy(package)
    invalid['games']['overall'][0]['status'] = 'APPROVED'
    with pytest.raises(ValueError): validate_package(invalid)
    # A real provider timestamp takes precedence over observation evidence.
    actual = quoted_candidate('draftkings',league='NCAAF')
    assert public_quote(actual,True) == ('DraftKings', OBSERVED)
    assert espn_observed_quote(actual) is None


def test_duplicate_observations_fail_closed_and_novig_still_wins():
    row = observed_candidate()
    quotes = json.loads(row['provider_quotes'])
    row['provider_quotes'] = json.dumps(quotes + quotes)
    assert espn_observed_quote(row) is None
    board = pd.DataFrame([final(league='NCAAF',export_run_id=RUN)])
    audit = pd.DataFrame([observed_candidate(), quoted_candidate('novig',league='NCAAF',best_available_rank=99)])
    selected = per_game_board(board,audit,novig_only=True,college_fallback=True).iloc[0]
    assert selected.quote_source == 'Novig'
    assert 'quote_time_basis' not in selected
