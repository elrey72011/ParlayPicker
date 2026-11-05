"""SYNTHETIC market-anchored EV selector (roadmap P1). Pure; no network."""
import math

import pandas as pd
import pytest

from app_core.selection import ev_selector as ev
from app_core.selection.ev_selector import Policy


def row(game='SYNTHETIC-G1', market='spread_home', odds=-110, opposing=-110, model=0.60, league='NFL', line=-2.5):
    return dict(matchup_id=game, league=league, home_team='H', away_team='A', market_type=market,
                odds_american=odds, opposing_odds_american=opposing, ml_probability=model,
                spread_line=line if market.startswith('spread') else None,
                total_line=line if market.startswith('total') else None)


def test_fair_probability_is_a_two_way_multiplicative_devig():
    assert ev.fair_probability(-110, -110) == pytest.approx(0.5)
    p = ev.fair_probability(150, -180)
    assert p == pytest.approx((1 / 2.5) / (1 / 2.5 + 1 / (1 + 100 / 180)))
    assert ev.fair_probability(-110, None) is None and ev.fair_probability(-50, -110) is None


def test_anchor_equals_market_without_a_model_and_is_capped():
    policy = Policy()
    assert ev.anchored_probability(0.5, None, policy) == 0.5
    nudged = ev.anchored_probability(0.5, 0.6, policy)
    expected = 1 / (1 + math.exp(-(0.25 * math.log(0.6 / 0.4))))
    assert nudged == pytest.approx(expected) and 0.5 < nudged < 0.55
    assert ev.anchored_probability(0.5, 0.99, policy) == pytest.approx(0.55)   # capped at +5 points
    assert ev.anchored_probability(0.5, 0.01, policy) == pytest.approx(0.45)   # capped at -5 points


def test_every_game_gets_one_decision_with_a_specific_reason():
    frame = pd.DataFrame([
        row('G-unpaired', opposing=None),
        row('G-nomodel', model=None),
        row('G-small', model=0.52),
        row('G-strong', model=0.75),
    ])
    decisions = ev.select(frame).set_index('matchup_id')
    assert decisions.loc['G-unpaired', 'reason'] == 'NO_USABLE_CANDIDATE:MARKET_PRICE_UNPAIRED'
    assert decisions.loc['G-nomodel', 'reason'] == 'NO_INDEPENDENT_MODEL'
    assert decisions.loc['G-small', 'reason'] == 'EDGE_BELOW_THRESHOLD'
    assert decisions.loc['G-strong', 'reason'] == 'MARKET_NOT_QUALIFIED'  # nothing is qualified yet
    assert set(decisions['decision']) == {'NO_BET'}


def test_a_qualified_market_with_enough_edge_is_a_bet():
    decisions = ev.select(pd.DataFrame([row(model=0.75)]), qualified={('NFL', 'SPREAD')})
    d = decisions.iloc[0]
    assert (d['decision'], d['reason']) == ('BET', 'QUALIFIED_EDGE')
    assert d['market_probability'] == 0.5 and d['edge'] >= 0.02 and d['ev'] > 0
    assert ev.select(pd.DataFrame([row(model=0.75)]), qualified={('NFL', 'TOTAL')}).iloc[0]['decision'] == 'NO_BET'


def test_the_highest_ev_candidate_is_chosen_per_game():
    frame = pd.DataFrame([
        row(market='spread_home', model=0.55),
        row(market='spread_away', model=0.48, line=2.5),
        row(market='total_over', model=0.70, line=44.5),
        row(market='total_under', model=0.30, line=44.5),
    ])
    d = ev.select(frame).iloc[0]
    assert d['market_type'] == 'total_over' and d['family'] == 'TOTAL' and d['line'] == 44.5
    assert d['candidates'] == 4 and d['usable_candidates'] == 4


def test_agreement_with_the_current_selector_is_recorded():
    frame = pd.DataFrame([row('G1', model=0.7), row('G2', market='total_over', model=0.7, line=44.5)])
    current = pd.DataFrame([dict(matchup_id='G1', market_type='spread_home'),
                            dict(matchup_id='G2', market_type='total_under')])
    d = ev.select(frame, current=current).set_index('matchup_id')
    assert d.loc['G1', 'agrees_with_current'] is True or d.loc['G1', 'agrees_with_current'] == True  # noqa: E712
    assert d.loc['G2', 'agrees_with_current'] == False  # noqa: E712
    s = ev.summary(ev.select(frame, current=current))
    assert s['games'] == 2 and s['bet'] == 0 and s['compared_with_current'] == 2 and s['agrees_with_current'] == 1
    assert s['policy']['version'] == 'ev-selector-v1'


def test_empty_or_keyless_input_returns_no_decisions():
    assert ev.select(pd.DataFrame()).empty
    assert ev.select(pd.DataFrame([{'league': 'NFL'}])).empty
    assert ev.summary(pd.DataFrame())['games'] == 0


def test_families_and_market_scope_follow_core_market_policy():
    run_line = ev.select(pd.DataFrame([row(league='MLB', line=-1.5, model=0.75)]), qualified={('MLB', 'RUN_LINE')})
    assert run_line.iloc[0]['family'] == 'RUN_LINE' and run_line.iloc[0]['decision'] == 'BET'
    moneyline = ev.select(pd.DataFrame([row(market='h2h_home', model=0.75)]), qualified={('NFL', 'SPREAD')})
    assert moneyline.iloc[0]['reason'] == 'NOT_A_PRODUCTION_MARKET'  # moneylines are context only
