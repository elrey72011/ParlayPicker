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
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.07, edge=0.07, kalshi_probability=0.48),
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
            _row(idx=2, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.039, kalshi_probability=0.54),
            # No Kalshi total penalty block.
            _row(idx=3, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            # Actionable side to keep card non-empty with non-total representation.
            _row(idx=4, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.07, edge=0.07, kalshi_probability=0.48),
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


def test_mlb_spread_finalist_penalty_can_demote_weak_spread_winner():
    matchup_id = "2026-04-24|home1|away1"
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.60, ev=0.20, edge=0.20, kalshi_probability=0.50),
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.60, ev=0.19, edge=0.19, kalshi_probability=0.50),
        ]
    )
    df["matchup_id"] = matchup_id
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert len(out) == 1
    assert out.iloc[0]["market_type"] == "total_over"
    assert diagnostics["demoted_by_mlb_spread_finalist_score_penalty"] >= 1


def test_nba_side_bonus_can_promote_borderline_side():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="spread_home", win_prob=0.52, ev=0.01, edge=0.015, kalshi_probability=0.48),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert diagnostics["promoted_by_nba_side_bonus"] >= 1


def test_nba_over_bonus_can_promote_borderline_over():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.04, kalshi_probability=0.55),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert diagnostics["promoted_by_nba_over_bonus"] >= 1


def test_mlb_over_explicit_actionable_gate_blocks_weak_over():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.56, ev=0.05, edge=0.05, kalshi_probability=0.53),
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.58, ev=0.05, edge=0.05, kalshi_probability=0.53),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics).sort_values("home_team")
    assert out.iloc[0]["Pick_Status"] == "Below Threshold"
    assert "MLB over actionable gate" in out.iloc[0]["Status_Reason"]
    assert out.iloc[1]["Pick_Status"] == "Actionable"
    assert diagnostics["blocked_by_mlb_over_promotion_gate"] >= 1


def test_new_diagnostics_populate_without_regressing_existing_total_protections():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.53, ev=0.03, edge=0.03, kalshi_probability=0.48),
            _row(idx=2, league="NBA", market_type="spread_home", win_prob=0.52, ev=0.01, edge=0.015, kalshi_probability=0.48),
            _row(idx=3, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.04, kalshi_probability=0.55),
            _row(idx=4, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            _row(idx=5, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["blocked_by_mlb_spread_penalty"] >= 1
    assert diagnostics["promoted_by_nba_side_bonus"] >= 1
    assert diagnostics["promoted_by_nba_over_bonus"] >= 1
    assert diagnostics["blocked_by_no_kalshi_total_penalty"] >= 1
    assert diagnostics["blocked_by_under_specific_thresholds"] >= 1


def test_high_ev_alone_is_not_auto_blocked_as_suspicious_data():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="spread_home", win_prob=0.61, ev=0.41, edge=0.06, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert row["status_blocker_stage"] == "variance_guardrail"
    assert row["suspicious_data_flag"] is False or row["suspicious_data_flag"] == False


def test_suspicious_data_rows_still_blocked_with_explicit_reason_and_diagnostics():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.62, ev=0.45, edge=0.07, kalshi_probability=0.55),
        ]
    )
    df.loc[0, "line_source"] = "synthetic"
    df.loc[0, "line_delta"] = 12.0
    df.loc[0, "market_probability"] = 0.20
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert row["suspicious_data_flag"] is True or row["suspicious_data_flag"] == True
    assert "Blocked: suspicious_data_flag=true" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "suspicious_data_guardrail"
    assert diagnostics["blocked_by_suspicious_data"] >= 1
