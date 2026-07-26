import pandas as pd
import numpy as np

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
    # League-appropriate placeholder total so the implausible-live-total guard (a live total
    # outside the league's range is a bad read) doesn't reject fixtures using a default line.
    _lt = (
        {"MLB": 8.5, "NHL": 6.5, "NBA": 220.5, "NCAAB": 145.0, "NFL": 45.0, "NCAAF": 55.0}.get(str(league).upper(), 220.5)
        if "total" in market_type else pd.NA
    )
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
        "total_line": _lt,
        # Live-line provenance. build_best_picks_df now requires a trusted live line
        # (line_source containing "live" AND a numeric live_*_line) or it rejects the
        # row as "suspicious live line" -> No Play. Supply that here so these calibration
        # tests exercise the status logic, not the line-provenance reject path. Tests
        # that specifically exercise provenance override line_source (e.g. "synthetic").
        "line_source": "live",
        "live_spread_line": -3.5 if "spread" in market_type else pd.NA,
        "live_total_line": _lt,
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
            # Bumped 0.60 -> 0.64 (ev/edge raised) so the Kalshi-backed row clears the
            # current gates; the contrast (no-Kalshi cold-market penalty blocks the twin)
            # is what this test verifies and still holds.
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=None),
            _row(idx=2, league="NFL", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
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
            # Kalshi 0.48 -> 0.55 so it agrees with the home spread; a disagreeing Kalshi
            # now caps spreads at High Variance (deliberate guard the old fixture predated).
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.07, edge=0.07, kalshi_probability=0.55),
            # Over bumped 0.60 -> 0.68 to clear the raised MLB over gate; the point of the
            # test (Overs/sides are not penalized the way Unders are) is unchanged.
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.68, ev=0.12, edge=0.10, kalshi_probability=0.60),
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
    diagnostics = {"is_fallback_heavy": True}
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


def test_mlb_spread_finalist_penalty_can_demote_weak_spread_winner(monkeypatch):
    # Isolate the spread handicap from the separate empirical-direction selector.
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: None)
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


def test_nba_over_bonus_is_retired_and_no_longer_promotes():
    # NBA_OVER_ACTIONABLE_BONUS was retired to 0.0 (Overs no longer get a side-style
    # promotion bonus). A borderline NBA over that the bonus used to lift to Actionable
    # must now stay blocked, and the promotion must not fire.
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.04, kalshi_probability=0.55),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert out.iloc[0]["Pick_Status"] != "Actionable"
    assert diagnostics.get("promoted_by_nba_over_bonus", 0) == 0


def test_mlb_over_explicit_actionable_gate_blocks_weak_over():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.56, ev=0.05, edge=0.05, kalshi_probability=0.53),
            # Strong row bumped 0.58 -> 0.68 (k 0.60) to clear the raised MLB over gate;
            # the weak 0.56 row is still blocked, which is what this test verifies.
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.68, ev=0.12, edge=0.10, kalshi_probability=0.60),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics).sort_values("home_team")
    assert out.iloc[0]["Pick_Status"] == "Below Threshold"
    assert "MLB over actionable gate" in out.iloc[0]["Status_Reason"]
    assert out.iloc[1]["Pick_Status"] == "Actionable"
    assert diagnostics["blocked_by_mlb_over_promotion_gate"] >= 1


def test_low_line_over_guardrail_is_consensus_aware():
    # Sub-8.0 MLB overs are not a uniformly weak bucket, and consensus drives the call.
    # The 20 Jun calibration refresh (graded through 18 Jun) puts Neutral (~50%) and
    # Disagrees (~49%) low-line overs in the weak zone — both held at Below Threshold by
    # the guardrail — while Agrees (~59%), the one over bucket with realized edge, is not
    # blocked and promotes on its merits. (Earlier data read Disagrees ~60%, hence the old
    # High Variance expectation; the refit corrected that bucket down to ~49%.)
    def _low_over(kalshi):
        df = pd.DataFrame([
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.60, ev=0.10, edge=0.10, kalshi_probability=kalshi)
        ])
        df.loc[0, "best_pick"] = "Over 7.5"
        df.loc[0, "total_line"] = 7.5
        df.loc[0, "live_total_line"] = 7.5
        return build_best_picks_df(df).iloc[0]

    # Neutral case uses a Kalshi pick'em (~0.50): under directional consensus a clear
    # Kalshi lean (0.60) is Agrees, so the Neutral guardrail path is exercised at 0.50.
    neutral = _low_over(0.50)
    assert neutral["consensus_agreement"] == "Neutral"
    assert neutral["Pick_Status"] == "Below Threshold"
    assert neutral["status_blocker_stage"] == "low_line_over_guardrail"

    disagrees = _low_over(0.45)
    assert disagrees["consensus_agreement"] == "Disagrees"
    assert disagrees["Pick_Status"] == "Below Threshold"
    # Refreshed Disagrees-over bucket (~46%, n=56) is now a PROVEN loser, so the empirical
    # overlay's proven-losing-bucket guard benches it a step before the low-line-over
    # guardrail would; either way it is held Below Threshold.
    assert disagrees["status_blocker_stage"] in {
        "low_line_over_guardrail",
        "empirical_tier_overlay",
        "empirical_proven_losing_bucket",
    }

    # 1-Jul recency refit: over:Agrees realized ~47% recency-weighted (44% over the last
    # 21 days) — a proven-losing bucket — so even Agrees low-line overs are benched by the
    # proven-losing suppression (or the overlay/guardrail, depending on the current fitted
    # table). The guardrail stays consensus-aware in code; whichever stage catches it first,
    # a sub-8.0 over is held Below Threshold while the over buckets are cold.
    agrees = _low_over(0.60)
    assert agrees["consensus_agreement"] == "Agrees"
    assert agrees["Pick_Status"] == "Below Threshold"
    assert agrees["status_blocker_stage"] in {
        "low_line_over_guardrail",
        "empirical_tier_overlay",
        "empirical_proven_losing_bucket",
    }


def test_consensus_is_directional_same_side_is_agrees():
    # 16 Jun: consensus is DIRECTIONAL — Kalshi backing the same side as our pick is
    # "Agrees" regardless of which side is more confident. (The old rule required the
    # model to LEAD Kalshi, so after the market-trust reweight pulled the model below
    # Kalshi every same-side pick mislabeled as "Disagrees".)
    def _consensus(win_prob, kalshi):
        df = pd.DataFrame([
            _row(idx=1, league="MLB", market_type="total_under", win_prob=win_prob,
                 ev=0.04, edge=0.04, kalshi_probability=kalshi)
        ])
        return build_best_picks_df(df).iloc[0]["consensus_agreement"]

    # Kalshi backs our side but is MORE confident than the model -> Agrees (was Disagrees).
    assert _consensus(0.52, 0.60) == "Agrees"
    # Model leads the market on the same side -> still Agrees.
    assert _consensus(0.62, 0.55) == "Agrees"
    # Kalshi favors the OTHER side -> Disagrees.
    assert _consensus(0.60, 0.45) == "Disagrees"
    # Kalshi pick'em (within the neutral band) -> Neutral.
    assert _consensus(0.60, 0.50) == "Neutral"


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
    assert row["Pick_Status"] == "Actionable"
    assert "strong EV/edge" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "none"
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
    assert "No Play: blocked due to suspicious data" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "suspicious_data_guardrail"
    assert diagnostics["blocked_by_suspicious_data"] >= 1


def test_divergence_cannot_preserve_negative_ev_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=-0.01, edge=0.02, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_metric_basis"] == "raw"
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_cannot_preserve_negative_edge_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=0.03, edge=-0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_can_preserve_minimally_viable_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert "capped due to divergence" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_guardrail"


def test_divergence_high_ev_override_preserves_sub_floor_win_prob_row():
    # A strongly +EV, +edge divergent pick on a +money line with win prob just below the
    # 0.53 viability floor is preserved as High Variance instead of dropped to No Play.
    # Uses a non-MLB-total market (NBA total_under): the override is scoped OFF MLB totals
    # (divergence is anti-predictive there), so the mechanism is validated on a market that
    # still keeps it.
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_under", win_prob=0.515, ev=0.10, edge=0.06, kalshi_probability=0.25),
        ]
    )
    df.loc[0, "odds_american"] = 120  # +money: positive EV/edge despite sub-0.53 win prob
    df.loc[0, "ml_probability"] = 0.515
    out = build_best_picks_df(df)
    row = out.iloc[0]
    # The divergence floor no longer drops it to No Play. Final tier is then decided by the
    # downstream empirical overlay, but the divergence-stage decision is retained in
    # status_blocker_reason and must show the high-EV override.
    assert row["Pick_Status"] != "No Play"
    assert "high-EV override" in row["status_blocker_reason"]


def test_divergence_high_ev_override_excluded_for_mlb_totals():
    # The override is scoped OFF MLB totals: the 13-slate recap study showed divergent
    # MLB-total picks are negatively predictive (staked overs 33-39% vs near-market 54%),
    # so a divergent +EV MLB total below the 0.53 floor must revert to No Play rather than
    # be preserved into the staked tier. Same inputs as the NBA case above, only the
    # league/market differ.
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_under", win_prob=0.515, ev=0.10, edge=0.06, kalshi_probability=0.27),
        ]
    )
    df.loc[0, "odds_american"] = 120
    df.loc[0, "ml_probability"] = 0.515
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_high_ev_override_does_not_rescue_marginal_ev_row():
    # A divergent pick that clears the base viability floor (EV >= 0.03) but not the
    # high-EV override bar (EV >= 0.05) and is below the 0.53 win-prob floor must still
    # be denied: the override is a high-conviction exception, not a blanket waiver.
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_under", win_prob=0.515, ev=0.03, edge=0.02, kalshi_probability=0.27),
        ]
    )
    df.loc[0, "odds_american"] = -110  # near coin-flip price -> marginal EV at win prob 0.515
    df.loc[0, "ml_probability"] = 0.515
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_viability_diagnostics_populate():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=-0.01, edge=0.02, kalshi_probability=0.80),
            _row(idx=2, league="NBA", market_type="total_over", win_prob=0.56, ev=0.03, edge=-0.01, kalshi_probability=0.80),
            _row(idx=3, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[:, "ml_probability"] = 0.50
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["divergence_rows_preserved"] >= 1
    assert diagnostics["divergence_rows_blocked_by_viability_floor"] >= 2
    assert diagnostics["divergence_rows_negative_ev"] >= 1
    assert diagnostics["divergence_rows_negative_edge"] >= 1
    assert "No Play" in diagnostics["divergence_rows_by_pick_status"]


def test_effective_metric_transparency_when_blocked_by_effective_thresholds():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.036, edge=0.05, kalshi_probability=0.55),
        ]
    )
    diagnostics = {"is_fallback_heavy": True}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "Below Threshold"
    assert row["status_metric_basis"] == "effective"
    assert float(row["effective_expected_value"]) < float(row["expected_value"])
    assert "Effective EV" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "fallback_heavy_guardrail"
    assert row["status_blocker_reason"] == row["Status_Reason"]


def test_run_health_fields_are_export_visible_when_present():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    df["nba_stats_fetch_status"] = "cached"
    df["nba_stats_fetch_source"] = "cached"
    df["nba_stats_fetch_retries_used"] = 3
    df["stats_source_counts"] = "{'live': 0, 'cached': 1, 'fallback': 0, 'failed': 0}"
    df["fallback_summary_by_league"] = "{'NBA': 12}"
    df["fallback_heavy_slate_flag"] = True
    df["run_health_warning"] = "Run health warning: fallback usage is elevated"
    df["degraded_feature_subset_flag"] = True
    df["degraded_feature_subset_reason"] = "degraded_subset:all_constant_feature_count=10"
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["nba_stats_fetch_status"] == "cached"
    assert row["nba_stats_fetch_source"] == "cached"
    assert bool(row["fallback_heavy_slate_flag"]) is True
    assert "Run health warning" in row["run_health_warning"]
    assert bool(row["degraded_feature_subset_flag"]) is True
    assert "degraded_subset" in str(row["degraded_feature_subset_reason"])


def test_high_ev_clean_row_promotes_to_actionable_not_high_variance():
    df = pd.DataFrame(
        [
            _row(idx=10, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=0.59),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "Actionable"
    assert "strong EV/edge" in row["Status_Reason"]
    assert diagnostics["promoted_high_ev_to_actionable_no_uncertainty"] >= 1
    assert diagnostics["high_variance_due_only_high_ev"] == 0


def test_high_ev_can_be_capped_for_real_uncertainty_reason():
    df = pd.DataFrame(
        [
            _row(idx=11, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
        ]
    )
    df["degraded_feature_subset_flag"] = True
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert "capped due to" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "variance_uncertainty_guardrail"


def test_high_variance_inflation_diagnostics_populate():
    df = pd.DataFrame(
        [
            _row(idx=21, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=0.59),
            _row(idx=22, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
            _row(idx=23, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[df["home_team"] == "Home22", "degraded_feature_subset_flag"] = True
    df.loc[df["home_team"] == "Home23", "ml_probability"] = 0.50
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["high_variance_due_only_high_ev"] == 0
    assert diagnostics["promoted_high_ev_to_actionable_no_uncertainty"] >= 1
    assert diagnostics["high_variance_capped_due_to_degraded_subset"] >= 1
    assert diagnostics["high_variance_capped_due_to_no_kalshi"] >= 1
    assert diagnostics["high_variance_capped_due_to_divergence"] >= 1
    assert "High Variance/Speculative" in diagnostics["final_pick_status_counts"]


def test_side_balance_does_not_promote_side_blocked_by_degraded_subset_reason():
    df = pd.DataFrame(
        [
            _row(idx=30, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
            _row(idx=31, league="NBA", market_type="spread_home", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
        ]
    )
    # Side remains high variance due to degraded subset uncertainty and should not be balance-promoted.
    df.loc[df["market_type"] == "spread_home", "degraded_feature_subset_flag"] = True
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"] == "Actionable"]
    assert not actionable.empty
    assert actionable["market_type"].str.contains("spread|h2h", case=False, regex=True, na=False).sum() == 0
    assert diagnostics["side_balance_promotions"] == 0


def test_transparency_fields_are_populated_for_every_export_row():
    df = pd.DataFrame(
        [
            _row(idx=40, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            _row(idx=41, league="NFL", market_type="total_under", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[df["home_team"] == "Home41", "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    required = [
        "status_metric_basis",
        "effective_expected_value",
        "effective_edge",
        "effective_win_probability",
        "status_blocker_reason",
        "status_blocker_stage",
    ]
    for col in required:
        assert col in out.columns
    assert out["status_metric_basis"].notna().all()
    assert out["effective_expected_value"].notna().all()
    assert out["effective_edge"].notna().all()
    assert out["effective_win_probability"].notna().all()
    assert out["status_blocker_stage"].notna().all()

from core.streamlit_pipeline import ensure_best_pick_export_columns, REQUIRED_BEST_PICK_EXPORT_COLUMNS


def test_best_pick_export_guarantees_required_transparency_columns():
    df = pd.DataFrame([_row(idx=501, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55)])
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    for col in REQUIRED_BEST_PICK_EXPORT_COLUMNS:
        assert col in out.columns
    assert diagnostics["best_pick_export_required_columns_ok"] is True
    assert diagnostics["best_pick_export_missing_columns"] == []


def test_missing_export_columns_are_backfilled_and_logged(caplog):
    partial = pd.DataFrame({"best_pick": ["Over 220.5"], "expected_value": [0.02]})
    with caplog.at_level("WARNING"):
        out = ensure_best_pick_export_columns(partial, diagnostics_out={})
    for col in REQUIRED_BEST_PICK_EXPORT_COLUMNS:
        assert col in out.columns
    assert "best_pick_export_missing_columns" in caplog.text
    assert "best_pick_export_missing_line_columns" in caplog.text


def test_line_provenance_columns_are_required_for_export():
    required_line_cols = {
        "market_line_used",
        "market_line_source",
        "market_line_source_detail",
        "matched_live_spread_line",
        "matched_live_total_line",
        "upload_spread_line",
        "upload_total_line",
        "base_spread_line",
        "base_total_line",
        "line_consistency_flag",
        "line_consistency_reason",
        "line_provenance_warning",
    }
    assert required_line_cols.issubset(set(REQUIRED_BEST_PICK_EXPORT_COLUMNS))


def test_line_event_identity_columns_are_required_for_export():
    required_identity_cols = {
        "line_event_identity_match_flag",
        "line_event_identity_reason",
        "live_event_match_key",
        "line_candidate_count",
        "selected_live_event_source",
    }
    assert required_identity_cols.issubset(set(REQUIRED_BEST_PICK_EXPORT_COLUMNS))


def test_line_provenance_backfilled_and_diagnostics_set():
    partial = pd.DataFrame({"best_pick": ["Over 220.5"], "expected_value": [0.02]})
    diagnostics = {}
    out = ensure_best_pick_export_columns(partial, diagnostics_out=diagnostics)
    assert "market_line_used" in out.columns
    assert "line_consistency_flag" in out.columns
    assert "line_provenance_warning" in out.columns
    assert "line_event_identity_match_flag" in out.columns
    assert "line_event_identity_reason" in out.columns
    assert diagnostics["best_pick_export_line_columns_ok"] is False
    assert "market_line_used" in diagnostics["best_pick_export_missing_line_columns"]


def test_side_balance_guard_promotes_viable_side_when_actionable_is_totals_only():
    df = pd.DataFrame(
        [
            _row(idx=510, league="NBA", market_type="total_over", win_prob=0.58, ev=0.04, edge=0.04, kalshi_probability=0.56),
            _row(idx=511, league="NBA", market_type="spread_home", win_prob=0.54, ev=0.03, edge=0.03, kalshi_probability=0.30),
        ]
    )
    # Keep side out of initial Actionable via divergence cap, then allow balance guard promotion.
    df.loc[df["market_type"] == "spread_home", "ml_probability"] = 0.64
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert actionable["market_type"].astype(str).str.contains("spread|h2h", case=False, regex=True, na=False).any()
    assert int(diagnostics["side_promoted_by_balance_guard_count"]) >= 1
    assert "Promoted strongest viable side" in str(diagnostics["side_balance_guard_reason"])
    assert str(out.iloc[0]["actionable_family_counts"]).strip() not in {"", "{}", "MISSING_COMPUTATION"}


def test_side_balance_guard_does_not_promote_weak_sides():
    df = pd.DataFrame(
        [
            _row(idx=520, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
            _row(idx=521, league="NBA", market_type="spread_home", win_prob=0.51, ev=0.005, edge=0.005, kalshi_probability=0.58),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert actionable["market_type"].astype(str).str.contains("spread|h2h", case=False, regex=True, na=False).sum() == 0
    assert diagnostics["side_promoted_by_balance_guard_count"] == 0
    assert "No viable side candidates within margin" in str(diagnostics["side_balance_guard_reason"])


def test_totals_only_actionable_allowed_when_no_viable_sides_exist():
    df = pd.DataFrame([
        _row(idx=530, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
        _row(idx=531, league="NBA", market_type="total_under", win_prob=0.62, ev=0.06, edge=0.06, kalshi_probability=0.57),
    ])
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert not actionable.empty
    assert actionable["market_type"].astype(str).str.contains("total", case=False, regex=True, na=False).all()
    assert diagnostics["viable_side_candidates_count"] == 0
    assert bool(diagnostics["totals_only_actionable_flag"]) is True
    assert bool(out.iloc[0]["totals_only_actionable_flag"]) is True


def test_degraded_nba_rows_keep_run_health_fields_in_final_export():
    df = pd.DataFrame(
        [
            _row(idx=560, league="NBA", market_type="total_over", win_prob=0.57, ev=0.04, edge=0.04, kalshi_probability=0.55),
            _row(idx=561, league="NBA", market_type="spread_home", win_prob=0.54, ev=0.02, edge=0.02, kalshi_probability=0.40),
        ]
    )
    df["nba_stats_fetch_status"] = "failed"
    df["nba_stats_fetch_source"] = "failed"
    df["nba_stats_fetch_retries_used"] = 3
    df["fallback_summary_by_league"] = "{'NBA': 2}"
    df["fallback_heavy_slate_flag"] = True
    df["run_health_warning"] = "Run health warning: fallback usage is elevated."
    out = build_best_picks_df(df)
    assert (out["nba_stats_fetch_status"].astype(str) == "failed").all()
    assert (out["fallback_summary_by_league"].astype(str).str.contains("NBA", na=False)).all()
    assert (out["run_health_warning"].astype(str).str.len() > 0).all()


def test_no_regression_mlb_spread_suspicious_and_divergence_guardrails():
    df = pd.DataFrame(
        [
            _row(idx=540, league="MLB", market_type="spread_home", win_prob=0.52, ev=0.02, edge=0.03, kalshi_probability=0.48),
            _row(idx=541, league="NBA", market_type="total_over", win_prob=0.64, ev=0.50, edge=0.20, kalshi_probability=0.59),
            _row(idx=542, league="NBA", market_type="spread_home", win_prob=0.50, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    # suspicious high EV total with bad market status should remain blocked.
    df.loc[df["home_team"] == "Home541", "market_status"] = "suspended"
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)

    mlb_spread_row = out[out["home_team"] == "Home540"].iloc[0]
    suspicious_row = out[out["home_team"] == "Home541"].iloc[0]
    divergence_row = out[out["home_team"] == "Home542"].iloc[0]

    assert mlb_spread_row["Pick_Status"] != "Actionable"
    assert suspicious_row["status_blocker_stage"] == "suspicious_data_guardrail"
    assert divergence_row["status_blocker_stage"] in {"divergence_guardrail", "divergence_viability_floor"}


def test_line_fidelity_spread_orientation_uses_live_matched_line():
    df = pd.DataFrame([
        _row(idx=600, league="MLB", market_type="spread_away", win_prob=0.56, ev=0.05, edge=0.04, kalshi_probability=0.52)
    ])
    df.loc[0, "home_team"] = "Los Angeles"
    df.loc[0, "away_team"] = "Chicago"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_spread_line"] = 1.5
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["best_pick"] == "Chicago +1.5"
    assert float(row["market_line_used"]) == 1.5
    assert row["market_line_source"] == "live"


def test_line_fidelity_totals_use_live_total_not_upload_or_base():
    df = pd.DataFrame([
        _row(idx=601, league="MLB", market_type="total_over", win_prob=0.57, ev=0.05, edge=0.04, kalshi_probability=0.52)
    ])
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_total_line"] = 6.5
    # Upload within the MLB suspicious-delta tolerance (>2.0 runs trips the guard). The
    # live total stays authoritative over upload/base; an extreme delta (the old 12.5)
    # now correctly trips the guard and is covered by the suspicious-delta tests instead.
    df.loc[0, "uploaded_total_line"] = 7.0
    df.loc[0, "total_line"] = 7.0
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["best_pick"] == "Over 6.5"
    assert float(row["market_line_used"]) == 6.5
    assert bool(row["line_consistency_flag"]) is True

def test_suspicious_live_total_reresolves_and_uses_corrected_line():
    df = pd.DataFrame([
        _row(idx=700, league="NHL", market_type="total_over", win_prob=0.60, ev=0.06, edge=0.05, kalshi_probability=0.55)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Dallas"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_total_line"] = 5.5
    df.loc[0, "uploaded_total_line"] = 8.5
    df.loc[0, "total_line"] = 8.5
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert float(row["market_line_used"]) == 5.5
    assert row["best_pick"] == "Over 5.5"


def test_suspicious_unresolved_lines_become_no_play_and_not_viable():
    df = pd.DataFrame([
        _row(idx=701, league="NBA", market_type="spread_away", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.58),
        _row(idx=701, league="NBA", market_type="spread_away", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.58),
    ])
    df["home_team"] = "Minnesota"
    df["away_team"] = "Denver"
    df["line_source"] = "live_matched"
    df["live_spread_line"] = [15.5, 15.5]
    df["uploaded_spread_line"] = [-6.5, -6.5]
    out = build_best_picks_df(df)
    assert (out["Pick_Status"].astype(str) == "No Play").all()
    assert (out["market_line_source"].astype(str) == "rejected_live").all()
    assert out["market_line_used"].isna().all()
    assert out["matched_live_spread_line"].isna().all()
    assert out["best_pick"].astype(str).str.contains("line unresolved", case=False, na=False).all()
    assert not out["Pick_Status"].astype(str).isin(["Actionable", "High Variance/Speculative"]).any()


def test_spread_away_uses_away_signed_live_line_for_denver_minnesota_shape():
    df = pd.DataFrame([
        _row(idx=702, league="NBA", market_type="spread_away", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.56)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Denver"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_spread_line"] = -6.5
    df.loc[0, "uploaded_spread_line"] = -6.5
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["best_pick"] == "Denver -6.5"
    assert float(row["market_line_used"]) == -6.5
    assert row["market_line_source"] == "live"


def test_suspicious_total_after_validation_can_recover_with_upload_fallback():
    df = pd.DataFrame([
        _row(idx=708, league="NHL", market_type="total_over", win_prob=0.61, ev=0.06, edge=0.05, kalshi_probability=0.56)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Dallas"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_total_line"] = 16.5  # implausible NHL total -> rejected, recover via upload
    df.loc[0, "uploaded_total_line"] = 5.5
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] in {"High Variance/Speculative", "No Play"}
    assert row["market_line_source"] == "upload"
    assert float(row["market_line_used"]) == 5.5
    assert pd.isna(row["matched_live_total_line"])
    assert row["best_pick"] == "Over 5.5"


def test_nhl_implausible_live_total_with_no_plausible_reference_is_no_play():
    # Under plausibility-gated live totals, an implausible live total (a bad read or a
    # wrong-game cross-match) is rejected; with no plausible uploaded reference to recover
    # from, the row becomes No Play. (A plausible live total is trusted; see line_fidelity.)
    df = pd.DataFrame([
        _row(idx=703, league="NHL", market_type="total_over", win_prob=0.61, ev=0.06, edge=0.05, kalshi_probability=0.56)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Dallas"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_total_line"] = 16.5   # implausible NHL total
    df.loc[0, "uploaded_total_line"] = 2.5  # also implausible -> no recovery
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert row["best_pick"] == "Total line unresolved"
    assert pd.isna(row["market_line_used"])
    assert pd.isna(row["matched_live_total_line"])


def test_mlb_implausible_live_total_with_no_plausible_reference_is_no_play():
    # MLB counterpart: an implausible live total with no plausible reference -> No Play.
    df = pd.DataFrame([
        _row(idx=706, league="MLB", market_type="total_over", win_prob=0.61, ev=0.06, edge=0.05, kalshi_probability=0.56)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Toronto"
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_total_line"] = 22.5   # implausible MLB total
    df.loc[0, "uploaded_total_line"] = 3.5  # also implausible -> no recovery
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert row["best_pick"] == "Total line unresolved"
    assert pd.isna(row["market_line_used"])
    assert pd.isna(row["matched_live_total_line"])


def test_export_transparency_fields_still_present_after_reresolution_logic():
    df = pd.DataFrame([
        _row(idx=704, league="NBA", market_type="spread_away", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.58)
    ])
    df.loc[0, "line_source"] = "live_matched"
    df.loc[0, "live_spread_line"] = 15.5
    df.loc[0, "uploaded_spread_line"] = -6.5
    out = build_best_picks_df(df)
    required = [
        "market_line_used", "market_line_source", "market_line_source_detail",
        "matched_live_spread_line", "matched_live_total_line", "line_consistency_flag",
        "line_consistency_reason", "line_provenance_warning", "line_event_identity_match_flag",
        "line_event_identity_reason", "live_event_match_key", "line_candidate_count",
        "selected_live_event_source",
    ]
    for col in required:
        assert col in out.columns
    row = out.iloc[0]
    assert isinstance(row["line_event_identity_reason"], str)
    assert row["line_event_identity_reason"] != ""
    assert isinstance(row["live_event_match_key"], str)
    assert row["live_event_match_key"] != ""
    assert isinstance(row["selected_live_event_source"], str)
    assert row["selected_live_event_source"] != ""

def test_non_live_source_cannot_backfill_matched_live_spread_line_from_base():
    df = pd.DataFrame([
        _row(idx=705, league="NBA", market_type="spread_away", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.56)
    ])
    df.loc[0, "home_team"] = "Minnesota"
    df.loc[0, "away_team"] = "Denver"
    df.loc[0, "line_source"] = "uploaded_theover"
    df.loc[0, "spread_line"] = 15.5
    df.loc[0, "live_spread_line"] = 15.5
    df.loc[0, "uploaded_spread_line"] = -5.5
    out = build_best_picks_df(df)
    row = out.iloc[0]
    # Core point preserved: a non-live source must not backfill the matched-live slot.
    assert pd.isna(row["matched_live_spread_line"])
    # Current behavior: with no live event identity the spread is rejected outright
    # (No Play / rejected_live) rather than falling back to the uploaded spread line.
    assert row["Pick_Status"] == "No Play"
    assert row["market_line_source"] == "rejected_live"
    assert pd.isna(row["market_line_used"])


def test_total_upload_fallback_recovers_plausible_rejected_live_total_conservatively():
    df = pd.DataFrame([
        _row(idx=810, league="NHL", market_type="total_over", win_prob=0.62, ev=0.07, edge=0.06, kalshi_probability=0.56),
    ])
    df["home_team"] = ["A"]
    df["away_team"] = ["E"]
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5]  # implausible NHL total -> rejected (triggers upload fallback)
    df["uploaded_total_line"] = [5.5]
    df["upload_total_line"] = [5.5]
    out = build_best_picks_df(df)

    recovered = out.iloc[0]
    assert recovered["market_line_source"] == "upload"
    assert recovered["market_line_source_detail"] == "upload_total_fallback_after_rejected_live"
    assert float(recovered["market_line_used"]) == 5.5
    assert recovered["best_pick"] == "Over 5.5"
    assert pd.isna(recovered["matched_live_total_line"])
    assert recovered["line_consistency_reason"] == "recovered_with_upload_total_after_rejected_live"
    assert recovered["line_provenance_warning"] == "Live total rejected; using uploaded/reference total"
    # Subject is the conservative line recovery, not the exact tier: refreshed calibration
    # (20 Jun) lands this Below Threshold rather than High Variance.
    assert recovered["Pick_Status"] in {"High Variance/Speculative", "Below Threshold", "No Play"}
    assert recovered["Pick_Status"] != "Actionable"
    assert float(recovered["Kelly_Bet_Size"]) == 0.0



def test_total_upload_fallback_rejects_implausible_upload_total():
    df = pd.DataFrame([
        _row(idx=811, league="MLB", market_type="total_over", win_prob=0.62, ev=0.07, edge=0.06, kalshi_probability=0.56),
    ])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5]
    df["uploaded_total_line"] = [3.5]
    df["upload_total_line"] = [3.5]
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["market_line_source"] == "rejected_live"
    assert row["best_pick"] == "Total line unresolved"


def test_total_upload_fallback_does_not_affect_clean_live_total_or_spread():
    total_df = pd.DataFrame([
        _row(idx=812, league="NBA", market_type="total_over", win_prob=0.62, ev=0.07, edge=0.06, kalshi_probability=0.56),
    ])
    total_df["line_source"] = "live_matched"
    total_df["live_total_line"] = [235.5]
    total_df["uploaded_total_line"] = [236.0]
    total_df["upload_total_line"] = [236.0]
    total_out = build_best_picks_df(total_df).iloc[0]
    assert total_out["market_line_source"] == "live"
    assert float(total_out["market_line_used"]) == 235.5

    spread_df = pd.DataFrame([
        _row(idx=813, league="NBA", market_type="spread_away", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.58),
        _row(idx=813, league="NBA", market_type="spread_away", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.58),
    ])
    spread_df["line_source"] = "live_matched"
    spread_df["live_spread_line"] = [15.5, 15.5]
    spread_df["uploaded_spread_line"] = [-6.5, -6.5]
    spread_out = build_best_picks_df(spread_df)
    assert (spread_out["market_line_source"] == "rejected_live").all()


def test_negative_ev_row_cannot_remain_high_variance():
    df = pd.DataFrame([_row(idx=900, league="MLB", market_type="total_over", win_prob=0.56, ev=-0.02, edge=0.01, kalshi_probability=0.52)])
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] != "High Variance/Speculative"
    assert row["Pick_Status"] != "Actionable"
    assert row["Pick_Status"] == "No Play"


def test_negative_edge_row_cannot_remain_high_variance():
    df = pd.DataFrame([_row(idx=901, league="MLB", market_type="total_over", win_prob=0.56, ev=0.03, edge=-0.01, kalshi_probability=0.52)])
    out = build_best_picks_df(df)
    row = out.iloc[0]
    # Intent: a negative-edge row must not ride as High Variance. With a clean live
    # line it lands at "Below Threshold" (56% win prob fails the 65% MLB totals gate);
    # both that and "No Play" are non-actionable terminal states. (It previously read
    # "No Play" only because the line was rejected for provenance, masking this.)
    assert row["Pick_Status"] != "High Variance/Speculative"
    assert row["Pick_Status"] in {"No Play", "Below Threshold"}


def test_negative_ev_upload_fallback_row_becomes_no_play_and_zero_kelly():
    df = pd.DataFrame([_row(idx=902, league="NHL", market_type="total_over", win_prob=0.62, ev=-0.01, edge=0.03, kalshi_probability=0.56)])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5]  # implausible NHL total -> rejected (triggers upload fallback)
    df["uploaded_total_line"] = [5.5]
    df["upload_total_line"] = [5.5]
    row = build_best_picks_df(df).iloc[0]
    assert row["market_line_source"] == "upload"
    assert row["Pick_Status"] == "No Play"
    assert float(row["Kelly_Bet_Size"]) == 0.0


def test_positive_ev_upload_fallback_row_can_remain_high_variance_with_zero_kelly():
    df = pd.DataFrame([_row(idx=903, league="NHL", market_type="total_over", win_prob=0.62, ev=0.07, edge=0.06, kalshi_probability=0.56)])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5]  # implausible NHL total -> rejected (triggers upload fallback)
    df["uploaded_total_line"] = [5.5]
    df["upload_total_line"] = [5.5]
    row = build_best_picks_df(df).iloc[0]
    assert row["market_line_source"] == "upload"
    # Subject is the line recovery + zero Kelly, not the exact non-actionable tier: the
    # 20 Jun calibration refresh made this row Below Threshold rather than High Variance.
    assert row["Pick_Status"] in {"High Variance/Speculative", "Below Threshold", "No Play"}
    assert float(row["Kelly_Bet_Size"]) == 0.0

def test_negative_ev_row_is_downgraded_by_value_guardrail_with_reason_and_zero_kelly():
    diagnostics = {}
    df = pd.DataFrame([_row(idx=904, league="NHL", market_type="total_over", win_prob=0.62, ev=-0.01, edge=0.06, kalshi_probability=0.56)])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5]  # implausible NHL total -> rejected (triggers upload fallback)
    df["uploaded_total_line"] = [5.5]
    df["upload_total_line"] = [5.5]
    row = build_best_picks_df(df, diagnostics_out=diagnostics).iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert row["status_blocker_stage"] == "value_guardrail"
    assert row["status_blocker_reason"] == "Negative EV or edge after final validation"
    assert row["Status_Reason"] == "No Play: negative EV or negative edge after final validation"
    assert float(row["Kelly_Bet_Size"]) == 0.0
    assert diagnostics["negative_ev_high_variance_downgraded_count"] >= 1


def test_negative_value_guardrail_diagnostics_and_positive_upload_fallback_regression():
    diagnostics = {}
    df = pd.DataFrame([
        _row(idx=905, league="NHL", market_type="total_over", win_prob=0.62, ev=-0.02, edge=0.03, kalshi_probability=0.56),
        _row(idx=906, league="NHL", market_type="total_over", win_prob=0.62, ev=0.07, edge=0.06, kalshi_probability=0.56),
    ])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [16.5, 16.5]  # implausible NHL totals -> rejected (upload fallback)
    df["uploaded_total_line"] = [5.5, 5.5]
    df["upload_total_line"] = [5.5, 5.5]

    # Recovered rows now carry zeroed EV (their odds are unreliable), so the negative vs
    # positive rows can no longer be told apart by expected_value -- select by team instead.
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    neg_row = out[out["home_team"] == "Home905"].iloc[0]
    pos_row = out[out["home_team"] == "Home906"].iloc[0]

    assert neg_row["Pick_Status"] == "No Play"
    assert float(neg_row["Kelly_Bet_Size"]) == 0.0
    assert pos_row["market_line_source"] == "upload"
    # Non-actionable + zero Kelly is the contract; refreshed calibration (20 Jun) lands
    # this Below Threshold rather than High Variance.
    assert pos_row["Pick_Status"] in {"High Variance/Speculative", "Below Threshold", "No Play"}
    assert float(pos_row["Kelly_Bet_Size"]) == 0.0

    assert diagnostics["negative_ev_final_guardrail_count"] >= 1
    assert diagnostics["negative_edge_final_guardrail_count"] >= 0
    assert diagnostics["negative_ev_high_variance_downgraded_count"] >= 1


def test_total_over_production_shrink_is_single_not_compounded():
    # The production over-shrink must start from calibrated_probability, not the already-
    # shrunk effective_win_probability, or it compounds shrink^2. The reset originally
    # covered only MLB overs, so non-MLB (NBA/NHL) overs were silently double-shrunk to
    # 0.60^2 = 0.36 instead of the intended single 0.60.
    nhl = build_best_picks_df(pd.DataFrame([
        _row(idx=950, league="NHL", market_type="total_over", win_prob=0.62, ev=0.08, edge=0.06, kalshi_probability=0.58)
    ])).iloc[0]
    calib = float(nhl["calibrated_probability"])
    assert abs(float(nhl["production_win_probability"]) - (0.5 + 0.60 * (calib - 0.5))) < 1e-6
    assert abs(float(nhl["production_win_probability"]) - (0.5 + 0.36 * (calib - 0.5))) > 1e-3  # not double-shrunk

    # MLB overs were already correctly single-shrunk (0.85 from calibrated); unchanged.
    mlb = build_best_picks_df(pd.DataFrame([
        _row(idx=951, league="MLB", market_type="total_over", win_prob=0.62, ev=0.08, edge=0.06, kalshi_probability=0.58)
    ])).iloc[0]
    calib_m = float(mlb["calibrated_probability"])
    assert abs(float(mlb["production_win_probability"]) - (0.5 + 0.85 * (calib_m - 0.5))) < 1e-6


def test_alt_priced_live_total_rejected_to_uploaded_reference():
    # 20 Jun NYY case: the live matcher latched onto an ALT total (Over 12.5 at +285,
    # de-vig ~0.26) instead of the ~9.5 main line. Its raw value (12.5) is "plausible"
    # for MLB, so the value-only guard trusted it. The price shape (far from pick'em)
    # now marks it as an alt/mis-scrape -> rejected to the uploaded reference, never
    # staked off the garbage number.
    df = pd.DataFrame([_row(idx=970, league="MLB", market_type="total_over",
                            win_prob=0.68, ev=0.30, edge=0.09)])
    df["odds_american"] = [285]
    df["market_probability"] = [0.26]
    df["best_pick"] = ["Over 12.5"]
    df["total_line"] = [12.5]
    df["live_total_line"] = [12.5]
    df["uploaded_total_line"] = [9.5]
    df["upload_total_line"] = [9.5]
    row = build_best_picks_df(df).iloc[0]
    assert row["Pick_Status"] != "Actionable"
    assert float(row["Kelly_Bet_Size"]) == 0.0
    # The garbage 12.5 alt line is dropped; the uploaded reference (9.5) is used instead.
    assert float(row["market_line_used"]) == 9.5
    assert row["best_pick"] == "Over 9.5"
    assert row["market_line_source_detail"] == "upload_total_fallback_after_rejected_live"


def test_juiced_but_real_main_total_not_alt_rejected():
    # A real main total can be moderately juiced (de-vig ~0.57) and must NOT be mistaken
    # for an alt line: the band is generous enough to keep trusting it.
    df = pd.DataFrame([_row(idx=971, league="MLB", market_type="total_under",
                            win_prob=0.60, ev=0.05, edge=0.04, kalshi_probability=0.58)])
    df["odds_american"] = [-135]
    df["market_probability"] = [0.574]
    df["best_pick"] = ["Under 8.5"]
    df["total_line"] = [8.5]
    df["live_total_line"] = [8.5]
    df["upload_total_line"] = [8.5]
    row = build_best_picks_df(df).iloc[0]
    assert str(row["market_line_source_detail"]) != "upload_total_fallback_after_rejected_live"
    assert str(row["status_blocker_stage"]) != "line_provenance"


def test_extreme_live_total_divergence_rejected_even_when_in_range():
    # 20 Jun Cubs landmine: live total 13.5 vs uploaded 9.0 (4.5-run gap), force-staked
    # $750 on Over 13.5. At -160 (de-vig 0.60) the alt-price guard missed it; the fix is
    # the tightened MLB live-plausibility ceiling (13.5 > 13.0 = outlier) so the material
    # divergence is no longer exempted -> rejected to the uploaded reference, losing
    # live-source status so the force-deploy data-safety gate skips it ($0).
    df = pd.DataFrame([_row(idx=981, league="MLB", market_type="total_over",
                            win_prob=0.716, ev=0.16, edge=0.12, kalshi_probability=0.797)])
    df["odds_american"] = [-160]
    df["market_probability"] = [0.596]
    df["best_pick"] = ["Over 13.5"]
    df["total_line"] = [13.5]
    df["live_total_line"] = [13.5]
    df["uploaded_total_line"] = [9.0]
    df["upload_total_line"] = [9.0]
    row = build_best_picks_df(df).iloc[0]
    assert float(row["market_line_used"]) == 9.0
    assert row["best_pick"] == "Over 9.0"
    assert str(row["market_line_source"]) != "live"          # force-deploy requires live source
    assert str(row["line_provenance_warning"]) != ""         # ...and an empty provenance warning
    assert float(row["Kelly_Bet_Size"]) == 0.0


def test_implausible_live_total_without_upload_is_rejected():
    # 21 Jun Arizona "Over 3.5": a 3.5 MLB total with NO uploaded reference to diverge from
    # is a bad live read on its own value -> rejected (No Play), not headlining the card.
    df = pd.DataFrame([_row(idx=991, league="MLB", market_type="total_over",
                            win_prob=0.61, ev=0.11, edge=0.08, kalshi_probability=0.68)])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [3.5]
    df["total_line"] = [3.5]
    df["best_pick"] = ["Over 3.5"]
    row = build_best_picks_df(df).iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert str(row["market_line_source"]) == "rejected_live"


def test_plausible_live_total_without_upload_is_kept():
    # Control: a plausible MLB total (8.5) with no upload is NOT flagged by the value check.
    df = pd.DataFrame([_row(idx=992, league="MLB", market_type="total_over",
                            win_prob=0.61, ev=0.08, edge=0.06, kalshi_probability=0.56)])
    df["line_source"] = "live_matched"
    df["live_total_line"] = [8.5]
    df["total_line"] = [8.5]
    df["best_pick"] = ["Over 8.5"]
    row = build_best_picks_df(df).iloc[0]
    assert str(row["market_line_source"]) == "live"
