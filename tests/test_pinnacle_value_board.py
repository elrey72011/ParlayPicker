import pandas as pd

from app.ui.daily_dashboard import pinnacle_value_board


def _row(**kw):
    base = dict(matchup_id="g1", league="NCAAF", home_team="Home", away_team="Away", market_type="spread_home",
                best_pick="Home -3.5", odds_american=110, odds_source="odds_api", candidate_context="PREGAME",
                pinnacle_fair_probability=0.52)
    base.update(kw)
    return base


def test_keeps_prices_beating_pinnacle_and_best_per_game_market():
    frame = pd.DataFrame([
        _row(),                                                      # 0.52 * 2.10 - 1 = 9.2%
        _row(best_pick="Away +3.5", market_type="spread_away", pinnacle_fair_probability=0.48, odds_american=-110),
        _row(best_pick="Home -3", odds_american=105),                 # same game/market, lower EV
        _row(best_pick="Over 50.5", market_type="total_over", odds_american=-105, pinnacle_fair_probability=0.53),
    ])
    board = pinnacle_value_board(frame)
    assert board["Pick"].tolist() == ["Home -3.5", "Over 50.5"]
    assert abs(board["EV vs Pinnacle"][0] - (0.52 * 2.10 - 1)) < 1e-9
    assert board["Game"][0] == "Away at Home"


def test_excludes_missing_reference_low_ev_post_start_and_pinnacle_entry():
    frame = pd.DataFrame([
        _row(pinnacle_fair_probability=float("nan")),
        _row(matchup_id="g2", odds_american=-110),                    # 0.52 * 1.909 - 1 < 2%
        _row(matchup_id="g3", candidate_context="POST_START_DIAGNOSTIC"),
        _row(matchup_id="g4", odds_source="pinnacle"),
        _row(matchup_id="g5", odds_american=50),
    ])
    assert pinnacle_value_board(frame).empty


def test_empty_without_column_or_candidates():
    assert pinnacle_value_board(None).empty
    assert pinnacle_value_board(pd.DataFrame([{"best_pick": "x"}])).empty
