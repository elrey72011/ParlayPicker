import pandas as pd
import pytest
from core.streamlit_pipeline import (
    _expand_live_odds_to_bet_rows, _trusted_live_line_source_mask, build_best_picks_df,
)
from app_core.per_game_boards import per_game_board


@pytest.mark.parametrize('book', ['novig', 'fanduel', 'draftkings', 'betmgm'])
def test_priced_total_provenance_is_live_but_rejected_sources_are_not(book):
    sources = pd.Series([f'{book}_priced_total_consensus',
                         'rejected_live_total_price', 'upload',
                         'unknown_priced_total_consensus'])
    assert _trusted_live_line_source_mask(sources).tolist() == [True, False, False, False]


def test_ncaaf_consensus_totals_survive_ranking_and_public_board():
    # Quote values from the Sep 10 Florida A&M-Miami export. Novig's pair is
    # unusable; three standard books actually quote 65.5. Future date isolates
    # the ranking regression from the real game's changing pregame status.
    row = {
        'league': 'NCAAF', 'home_team': 'Miami', 'away_team': 'Florida Am',
        'game_date': '2099-09-10', 'matchup_id': 'miami-florida-am',
        'commence_time_raw': '2099-09-11T00:00:00Z',
        'game_time_est': '2099-09-10 8:00 PM ET',
        'novig_over_point': 36.5, 'novig_under_point': 36.5,
        'novig_over_price': -49900, 'novig_under_price': -100000,
        'novig_home_point': -30.5, 'novig_away_point': 30.5,
        'novig_home_price': -19900, 'novig_away_price': -100000,
        'draftkings_home_point': -59.5, 'draftkings_away_point': 59.5,
        'draftkings_home_price': -115, 'draftkings_away_price': -105,
        'betmgm_home_point': -58.5, 'betmgm_away_point': 58.5,
        'betmgm_home_price': -115, 'betmgm_away_price': -105,
        'fanduel_home_point': -59.5, 'fanduel_away_point': 59.5,
        'fanduel_home_price': -105, 'fanduel_away_price': -115,
        **{f'{book}_{side}_point': 65.5
           for book in ('fanduel', 'draftkings', 'betmgm') for side in ('over', 'under')},
        'fanduel_over_price': -115, 'fanduel_under_price': -105,
        'draftkings_over_price': -108, 'draftkings_under_price': -112,
        'betmgm_over_price': -108, 'betmgm_under_price': -110,
    }
    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    totals = expanded[expanded.market_type.str.startswith('total')]
    assert set(totals.total_line) == {65.5}
    assert set(totals.line_source) == {'fanduel_priced_total_consensus'}
    assert set(totals.odds_american) == {-115, -105}
    # Synthetic probabilities exercise ranking, not a real-game forecast.
    expanded['calibrated_probability'] = expanded.market_type.map(
        {'spread_home': .65, 'spread_away': .35, 'total_over': .48, 'total_under': .52})
    expanded['model_probability'] = expanded.calibrated_probability
    expanded['expected_value'] = -.05
    expanded['edge'] = -.01
    expanded['market_probability'] = .5
    diagnostics = {}
    final = build_best_picks_df(expanded, diagnostics_out=diagnostics)
    audit = diagnostics['candidate_audit_df']
    ranked_totals = audit[audit.market_type.str.startswith('total')]
    assert len(ranked_totals) == 2
    assert set(ranked_totals.best_available_family_rank) == {1, 2}
    assert diagnostics['preselection_dropped_total_candidate_count'] == 0
    assert set(ranked_totals.total_line) == {65.5}
    final['Bettable'] = False
    final['Play_Stake'] = 0
    board = per_game_board(final, audit, 'totals').iloc[0]
    assert board['pick'] in {'Over 65.5', 'Under 65.5'}
    assert board.odds in {-115, -105}
    assert board.status == 'PASS' and board.Play_Stake == 0
