"""Kalshi enters the blend only at the exact book line and the pick's own team (owner 2026-11-06)."""
import pandas as pd

from app_core.kalshi_exact import gate


def _frame(rows):
    return pd.DataFrame(rows)


def test_2026_11_06_false_edges_are_dropped_and_exact_matches_kept():
    frame = _frame([
        # Oklahoma St -18.5 priced from "Oklahoma wins by over 20.5" (other school, 2 points away)
        dict(market_type="spread_home", spread_line=-18.5, home_team="Oklahoma St", away_team="Queens University",
             kalshi_line_diff=2.0, kalshi_matched_line=20.5, kalshi_market_title="Oklahoma wins by over 20.5 points"),
        # Over 143.5 priced from an Over 144.5 contract
        dict(market_type="total_over", total_line=143.5, kalshi_line_diff=1.0, kalshi_matched_line=144.5,
             kalshi_market_title="Full Game: Over 144.5 points scored"),
        # exact spread, team spelled "Kennesaw St." on Kalshi
        dict(market_type="spread_home", spread_line=-5.5, home_team="Kennesaw State", away_team="Austin Peay",
             kalshi_line_diff=0.0, kalshi_matched_line=5.5, kalshi_market_title="Kennesaw St. wins by over 5.5 points"),
        # exact line, but the contract names neither team
        dict(market_type="spread_home", spread_line=-20.5, home_team="Oklahoma St", away_team="Queens University",
             kalshi_line_diff=0.0, kalshi_matched_line=20.5, kalshi_market_title="Oklahoma wins by over 20.5 points"),
        # exact total
        dict(market_type="total_under", total_line=150.5, kalshi_line_diff=0.0, kalshi_matched_line=150.5,
             kalshi_market_title="Full Game: Over 150.5 points scored"),
        # moneyline: unchanged
        dict(market_type="moneyline_home", home_team="A", away_team="B"),
        # spread with Kalshi but no recorded contract line: dropped
        dict(market_type="spread_away", spread_line=3.5, home_team="A", away_team="B"),
    ])
    kalshi = pd.Series([0.894, 0.673, 0.55, 0.9, 0.48, 0.6, 0.7])
    used, reason = gate(frame, kalshi)
    assert reason.tolist() == ["KALSHI_LINE_NOT_EXACT", "KALSHI_LINE_NOT_EXACT", "", "KALSHI_TEAM_NOT_EXACT", "",
                               "", "KALSHI_LINE_NOT_EXACT"]
    assert used.isna().tolist() == [True, True, False, True, False, False, True]
    assert used[2] == 0.55 and used[4] == 0.48 and used[5] == 0.6


def test_missing_kalshi_and_matched_line_fallback():
    frame = _frame([dict(market_type="spread_away", spread_line=3.5, home_team="A", away_team="B",
                         kalshi_matched_line=3.5, kalshi_market_title="B wins by over 3.5 points"),
                    dict(market_type="total_over", total_line=140.5)])
    used, reason = gate(frame, pd.Series([0.4, float("nan")]))
    assert used[0] == 0.4 and reason.tolist() == ["", ""] and pd.isna(used[1])
    empty, why = gate(pd.DataFrame(), pd.Series(dtype="float64"))
    assert empty.empty and why.empty


def test_blend_without_dropped_kalshi_falls_to_tier_two():
    from core.streamlit_pipeline import compute_blended_probability
    s = lambda v: pd.Series([v], dtype="float64")
    with_kalshi = compute_blended_probability(s(0.5), s(0.894), s(float("nan")), s(float("nan")), s(0.5),
                                              league=pd.Series(["NCAAB"]), market_type=pd.Series(["spread_home"]))[0]
    without = compute_blended_probability(s(0.5), s(float("nan")), s(float("nan")), s(float("nan")), s(0.5),
                                          league=pd.Series(["NCAAB"]), market_type=pd.Series(["spread_home"]))[0]
    assert with_kalshi > 0.75 and abs(without - 0.5) < 1e-9


def test_app_post_kalshi_reblend_uses_the_gate():
    """The final blend is streamlit_app._recompute_consensus_from_kalshi (2026-11-06 exports:
    UConn/Michigan Under 143.5 at 67% from an Over 144.5 contract; Oklahoma St -17.5 at 76%
    from 'Oklahoma wins by over 16.5')."""
    import streamlit_app as app
    rows = pd.DataFrame({
        "home_team": ["Michigan", "Oklahoma St", "Kennesaw State"],
        "away_team": ["UConn", "Queens University", "Austin Peay"],
        "best_pick": ["Under 143.5", "Oklahoma St -17.5", "Kennesaw State -5.5"],
        "market_type": ["total_under", "spread_home", "spread_home"],
        "total_line": [143.5, float("nan"), float("nan")],
        "spread_line": [float("nan"), -17.5, -5.5],
        "odds_american": [-106, -102, -110],
        "opposing_odds_american": [-114, -118, -110],
        "market_probability": [0.5097, 0.5, 0.5],
        "kalshi_probability": [0.753, 0.896, 0.60],
        "kalshi_line_diff": [1.0, 1.0, 0.0],
        "kalshi_matched_line": [144.5, 16.5, 5.5],
        "kalshi_market_title": ["Full Game: Over 144.5 points scored", "Oklahoma wins by over 16.5 points",
                                "Kennesaw St. wins by over 5.5 points"],
        "ml_probability": [float("nan")] * 3, "theover_probability": [float("nan")] * 3,
        "league": ["NCAAB"] * 3,                  # exports carry no sentiment_diff (neutral default)
    })
    out = app._recompute_consensus_from_kalshi(rows)
    assert out["kalshi_blend_gate"].tolist() == ["KALSHI_LINE_NOT_EXACT", "KALSHI_LINE_NOT_EXACT", ""]
    assert out["blend_in_kalshi"].isna().tolist() == [True, True, False]
    assert out["blend_tier"].tolist() == [2, 2, 1]
    assert abs(out["calibrated_probability"][0] - out["market_probability"][0]) < 0.02
    assert abs(out["calibrated_probability"][1] - out["market_probability"][1]) < 0.02
    assert out["calibrated_probability"][2] > 0.52                     # exact contract still used
    assert out["kalshi_probability"][0] == 0.753                        # research value kept
