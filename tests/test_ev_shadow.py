"""SYNTHETIC EV selector shadow run and recording (roadmap P1.2)."""
from datetime import datetime, timezone

import pandas as pd

from app_core.pipeline.store import Store
from app_core.selection import shadow

KEY = 'SYNTHETIC-ev-shadow-test-key-not-production'
NOW = datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)
FRAME = pd.DataFrame([
    dict(matchup_id='SYNTHETIC-G1', league='NFL', home_team='H', away_team='A', market_type='spread_home',
         odds_american=-110, opposing_odds_american=-110, ml_probability=0.7, spread_line=-2.5),
    dict(matchup_id='SYNTHETIC-G2', league='MLB', home_team='H2', away_team='A2', market_type='total_over',
         odds_american=-105, opposing_odds_american=float('nan'), ml_probability=float('nan'), total_line=8.5),
])
CURRENT = pd.DataFrame([dict(matchup_id='SYNTHETIC-G1', market_type='spread_home')])


def test_disabled_by_default():
    assert shadow.run(FRAME, CURRENT, setting=lambda k, d='': d) == (None, dict(games=0), 'EV_SHADOW_DISABLED')


def test_enabled_without_a_store_shows_decisions_but_records_nothing():
    on = {'PARLAYPICKER_EV_SHADOW': 'true'}
    decisions, summary, note = shadow.run(FRAME, CURRENT, setting=lambda k, d='': on.get(k, d))
    assert note == 'PIPELINE_STORE_NOT_CONFIGURED' and summary['games'] == 2 and 'recorded' not in summary
    by_game = decisions.set_index('matchup_id')
    assert by_game.loc['SYNTHETIC-G1', 'reason'] == 'MARKET_NOT_QUALIFIED'
    assert by_game.loc['SYNTHETIC-G2', 'reason'] == 'NO_USABLE_CANDIDATE:MARKET_PRICE_UNPAIRED'


def test_enabled_with_a_store_records_signed_decisions(tmp_path):
    path = tmp_path / 'SYNTHETIC-shadow.sqlite'
    settings = {'PARLAYPICKER_EV_SHADOW': 'true', 'PARLAYPICKER_PIPELINE_STORE_KEY': KEY,
                'PARLAYPICKER_PIPELINE_SQLITE_PATH': str(path)}
    decisions, summary, note = shadow.run(FRAME, CURRENT, setting=lambda k, d='': settings.get(k, d), clock=lambda: NOW)
    assert note is None and summary['recorded'] == 2
    store = Store.sqlite(path, KEY.encode(), owner='owner')
    nfl = store.list('NFL', 'ev-shadow-decision')
    assert len(nfl) == 1 and nfl[0]['payload']['decision'] == 'NO_BET'
    assert nfl[0]['payload']['decided_at'] == NOW.isoformat() and nfl[0]['payload']['agrees_with_current'] is True
    mlb = store.list('MLB', 'ev-shadow-decision')[0]['payload']
    assert mlb['reason'] == 'NO_USABLE_CANDIDATE:MARKET_PRICE_UNPAIRED' and mlb['current_market_type'] is None
