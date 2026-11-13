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


def test_board_pinnacle_columns_match_by_game_market_and_pick():
    from app.ui.daily_dashboard import pinnacle_shadow
    result = pd.DataFrame({"candidate_id": ["", ""], "matchup_id": ["g1", "g2"], "market_type": ["total_under", "spread_home"],
                           "pick": ["Under 46.5", "Home -3.5"], "odds": [104, -110]})
    candidates = pd.DataFrame({"candidate_id": ["a", "b"], "matchup_id": ["g1", "g2"],
                               "market_type": ["total_under", "spread_home"], "best_pick": ["Under 46.5", "Home -3.5"],
                               "pinnacle_fair_probability": [0.502226, float("nan")]})
    probs, ev = pinnacle_shadow(result, candidates)
    assert abs(probs[0] - 0.502226) < 1e-9 and abs(ev[0] - (0.502226 * 2.04 - 1)) < 1e-9
    assert pd.isna(probs[1]) and pd.isna(ev[1])
    assert pinnacle_shadow(result, candidates.assign(pinnacle_fair_probability=float("nan"))) is None


def test_best_pick_export_carries_pinnacle_and_kalshi_gate():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    for column in ("pinnacle_fair_probability", "pinnacle_ev", "kalshi_blend_gate"):
        assert column in BEST_PICK_COLUMNS and column in REQUIRED_BEST_PICK_EXPORT_COLUMNS


def _quotes(*items):
    import json
    return json.dumps([dict(market_type=kind, book=book, price=price, point=point) for kind, book, price, point in items])


def test_book_column_names_the_books_offering_the_exact_price_never_the_feed():
    """Owner 2026-11-13: the Book column read ``the_odds_api``, which is the feed, not a book."""
    from app.ui.daily_dashboard import BOOK_UNKNOWN
    frame = pd.DataFrame([
        _row(odds_feed_source="the_odds_api", spread_line=-3.5, provider_quotes=_quotes(
            ("spread_home", "novig", 110, -3.5), ("spread_home", "fanduel", 110, -3.5),
            ("spread_home", "draftkings", 105, -3.5), ("spread_home", "betmgm", 110, -4.0),
            ("spread_away", "novig", 110, 3.5), ("spread_home", "pinnacle", 110, -3.5))),
        _row(matchup_id="g2", odds_feed_source="the_odds_api"),                     # no quotes saved
        _row(matchup_id="g3", quote_bookmaker="draftkings", odds_feed_source="the_odds_api"),
    ])
    board = pinnacle_value_board(frame)
    assert board["Book"].tolist() == ["fanduel, novig", BOOK_UNKNOWN, "draftkings"]
    assert not board["Book"].str.contains("odds_api").any()


def test_a_price_only_pinnacle_offers_is_not_an_entry():
    frame = pd.DataFrame([_row(spread_line=-3.5, provider_quotes=_quotes(("spread_home", "pinnacle", 110, -3.5)))])
    assert pinnacle_value_board(frame).empty
