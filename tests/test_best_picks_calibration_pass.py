import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def _row(
    *,
    idx: int,
    league: str,
    market_type: str,
    win_prob: float,
    ev: float,
    edge: float,
    kalshi_probability=None,
):
    home = f"Home{idx}"
    away = f"Away{idx}"
    return {
        "league": league,
        "home_team": home,
        "away_team": away,
        "game_date": "2026-04-24",
        "matchup_id": f"2026-04-24|{home}|{away}",
        "market_type": market_type,
        "expected_value": ev,
        "edge": edge,
        "calibrated_probability": win_prob,
        "ml_probability": win_prob,
        "model_probability": win_prob,
        "odds_american": -110,
        "spread_line": -3.5 if "spread" in market_type else pd.NA,
        "total_line": 220.5 if "total" in market_type else pd.NA,
        "is_live_data": True,
        "used_stale_features": False,
        "odds_source": "odds_api",
        "kalshi_probability": kalshi_probability,
    }


def test_total_under_requires_stronger_bar_than_generic_totals():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.54),
            _row(idx=2, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.54),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["market_type"] == "total_over", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_under", "Pick_Status"].iloc[0] == "Below Threshold"


def test_nba_totals_are_no_longer_overpenalized_vs_weak_mlb_spreads():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.58, ev=0.05, edge=0.06, kalshi_probability=0.53),
            _row(idx=2, league="MLB", market_type="spread_home", win_prob=0.52, ev=0.02, edge=0.03, kalshi_probability=0.48),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["league"] == "NBA", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["league"] == "MLB", "Pick_Status"].iloc[0] == "Below Threshold"


def test_no_kalshi_totals_are_harder_than_kalshi_backed_totals():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            _row(idx=2, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    no_kalshi_row = out[out["kalshi_probability"].isna()].iloc[0]
    kalshi_row = out[out["kalshi_probability"].notna()].iloc[0]
    assert no_kalshi_row["consensus_agreement"] == "No Kalshi"
    assert no_kalshi_row["Pick_Status"] == "Below Threshold"
    assert kalshi_row["Pick_Status"] == "Actionable"


def test_agrees_does_not_auto_promote_in_standard_mode():
    df = pd.DataFrame(
        [
            # Agrees (gap +0.04) but still weak totals profile
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.57, ev=0.02, edge=0.03, kalshi_probability=0.53),
            # Neutral (gap ~0) same weak thresholds
            _row(idx=2, league="NFL", market_type="total_over", win_prob=0.57, ev=0.02, edge=0.03, kalshi_probability=0.57),
        ]
    )
    out = build_best_picks_df(df)
    statuses = out.sort_values("home_team")["Pick_Status"].astype(str).tolist()
    assert statuses == ["Below Threshold", "Below Threshold"]


def test_overs_and_sides_not_penalized_like_unders():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.03, edge=0.03, kalshi_probability=0.48),
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            _row(idx=3, league="MLB", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["market_type"] == "spread_home", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_over", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_under", "Pick_Status"].iloc[0] == "Below Threshold"


def test_diagnostics_blocked_rows_and_shadow_cards_populate():
    df = pd.DataFrame(
        [
            # Under-specific block: base would pass, stricter under bar blocks it.
            _row(idx=1, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            # NBA total penalty block.
            _row(idx=2, league="NBA", market_type="total_over", win_prob=0.58, ev=0.035, edge=0.045, kalshi_probability=0.54),
            # No Kalshi total penalty block.
            _row(idx=3, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            # Actionable side to keep card non-empty with non-total representation.
            _row(idx=4, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.03, edge=0.03, kalshi_probability=0.48),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["blocked_by_under_specific_thresholds"] >= 1
    assert diagnostics["blocked_by_nba_total_penalty"] >= 1
    assert diagnostics["blocked_by_no_kalshi_total_penalty"] >= 1
    assert "shadow_card_counts" in diagnostics
    shadow = diagnostics["shadow_card_counts"]
    for key in [
        "current_card",
        "overs_only_plus_sides_card",
        "no_unders_card",
        "no_nba_totals_card",
        "no_kalshi_totals_card",
    ]:
        assert key in shadow
    assert "actionable_counts_by_league_family" in diagnostics
