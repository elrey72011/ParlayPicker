import pandas as pd

from app_core.strategy_lab_realized import (
    REALIZED_MODE_ACTIONABLE_ONLY,
    REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
    REALIZED_MODE_ALL_GRADED,
    REALIZED_MODE_PRODUCTION_CARD,
    build_realized_strategy_lab,
)
from app.ui import strategy_lab_dashboard as dash


def _graded_with_statuses():
    return pd.DataFrame(
        [
            {"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 1.91, "Stake": 100, "actual_home_score": 5, "actual_away_score": 4, "Status": "Actionable"},
            {"league": "MLB", "home_team": "C", "away_team": "D", "Pick Taken": "C -1.5", "Pick_Outcome": "LOSS", "decimal_odds": 2.00, "Stake": 50, "actual_home_score": 2, "actual_away_score": 5, "Status": "High Variance/Speculative"},
            {"league": "MLB", "home_team": "E", "away_team": "F", "Pick Taken": "Under 7.5", "Pick_Outcome": "WIN", "decimal_odds": 2.00, "Stake": 25, "actual_home_score": 6, "actual_away_score": 4, "Status": "Below Threshold"},
            {"league": "MLB", "home_team": "G", "away_team": "H", "Pick Taken": "G ML", "Pick_Outcome": "LOSS", "decimal_odds": 2.50, "Stake": 10, "actual_home_score": 1, "actual_away_score": 3, "Status": "No Play"},
        ]
    )


def test_realized_pnl_computation_win_loss_push():
    graded = pd.DataFrame(
        [
            {"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 1.91, "Stake": 100, "actual_home_score": 5, "actual_away_score": 4},
            {"league": "MLB", "home_team": "C", "away_team": "D", "Pick Taken": "C -1.5", "Pick_Outcome": "LOSS", "decimal_odds": 1.80, "Stake": 50, "actual_home_score": 2, "actual_away_score": 5},
            {"league": "MLB", "home_team": "E", "away_team": "F", "Pick Taken": "Under 7.5", "Pick_Outcome": "PUSH", "decimal_odds": 2.00, "Stake": 20, "actual_home_score": 4, "actual_away_score": 4},
        ]
    )

    realized, summary, _ = build_realized_strategy_lab(graded)

    assert realized.loc[0, "Gross Return"] == 191
    assert realized.loc[0, "Net Profit"] == 91
    assert realized.loc[1, "Gross Return"] == 0
    assert realized.loc[1, "Net Profit"] == -50
    assert realized.loc[2, "Gross Return"] == 20
    assert realized.loc[2, "Net Profit"] == 0

    assert summary["Total Staked"] == 170
    assert summary["Gross Returned"] == 211
    assert summary["Net P/L"] == 41


def test_default_realized_mode_is_actionable_only():
    realized, summary, diagnostics = build_realized_strategy_lab(_graded_with_statuses())

    assert summary["Strategy Mode"] == REALIZED_MODE_ACTIONABLE_ONLY
    assert set(realized.loc[realized["Include In Totals"], "Status Bucket"]) == {"Actionable"}
    assert diagnostics["outside_mode_count"] == 3


def test_default_excludes_no_play_from_bankroll_and_summary():
    realized, summary, _ = build_realized_strategy_lab(_graded_with_statuses())

    no_play_row = realized[realized["Status Bucket"] == "No Play"].iloc[0]
    assert not bool(no_play_row["Include In Totals"])
    assert summary["Bet Count"] == 1
    assert summary["Total Staked"] == 100


def test_actionable_plus_high_variance_mode_selection():
    realized, summary, _ = build_realized_strategy_lab(
        _graded_with_statuses(),
        strategy_mode=REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
    )

    included_statuses = set(realized.loc[realized["Include In Totals"], "Status Bucket"])
    assert included_statuses == {"Actionable", "High Variance/Speculative"}
    assert summary["Bet Count"] == 2


def test_all_graded_mode_includes_all_status_buckets():
    realized, summary, _ = build_realized_strategy_lab(
        _graded_with_statuses(),
        strategy_mode=REALIZED_MODE_ALL_GRADED,
    )

    assert summary["Bet Count"] == 4
    assert realized["Include In Totals"].sum() == 4


def test_status_bucket_summary_aggregates_each_bucket():
    _, _, diagnostics = build_realized_strategy_lab(
        _graded_with_statuses(),
        strategy_mode=REALIZED_MODE_ACTIONABLE_ONLY,
    )
    summary_df = diagnostics["status_bucket_summary"].set_index("Status Bucket")

    assert int(summary_df.loc["Actionable", "Bet Count"]) == 1
    assert int(summary_df.loc["High Variance/Speculative", "Bet Count"]) == 1
    assert int(summary_df.loc["Below Threshold", "Bet Count"]) == 1
    assert int(summary_df.loc["No Play", "Bet Count"]) == 1
    assert float(summary_df.loc["Actionable", "Net P/L"]) == 91
    assert float(summary_df.loc["No Play", "Net P/L"]) == -10


def test_roi_and_net_pl_change_by_mode():
    _, actionable_summary, _ = build_realized_strategy_lab(
        _graded_with_statuses(),
        strategy_mode=REALIZED_MODE_ACTIONABLE_ONLY,
    )
    _, all_summary, _ = build_realized_strategy_lab(
        _graded_with_statuses(),
        strategy_mode=REALIZED_MODE_ALL_GRADED,
    )

    assert actionable_summary["Net P/L"] != all_summary["Net P/L"]
    assert actionable_summary["ROI"] != all_summary["ROI"]


def test_mismatched_picks_are_detected_and_excluded():
    graded = pd.DataFrame(
        [
            {"league": "MLB", "home_team": "Baltimore", "away_team": "Boston", "Pick Taken": "Under 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "Stake": 10, "actual_home_score": 2, "actual_away_score": 1},
        ]
    )
    strategy = pd.DataFrame(
        [
            {"league": "MLB", "home_team": "Baltimore", "away_team": "Boston", "best_pick": "Boston -5.5"},
        ]
    )

    realized, summary, diagnostics = build_realized_strategy_lab(graded, strategy)

    assert realized.loc[0, "Result"] == "MISMATCH"
    assert not realized.loc[0, "Include In Totals"]
    assert summary["Total Staked"] == 0
    assert summary["Mismatch Count"] == 1
    assert len(diagnostics["pick_mismatches"]) == 1


def test_realized_outcome_comes_from_recap_source_not_strategy_override():
    graded = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Saint Louis",
                "away_team": "Seattle",
                "Pick Taken": "Under 7.5",
                "Pick_Outcome": "LOSS",
                "decimal_odds": 1.91,
                "Stake": 10,
                "actual_home_score": 5,
                "actual_away_score": 4,
            }
        ]
    )
    strategy = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Saint Louis",
                "away_team": "Seattle",
                "Pick Taken": "Under 7.5",
                "Result": "WIN",  # ignored, canonical recap source controls outcome
            }
        ]
    )

    realized, summary, _ = build_realized_strategy_lab(graded, strategy)

    assert realized.loc[0, "Result"] == "LOSS"
    assert summary["Loss Count"] == 1
    assert summary["Win Count"] == 0


def test_same_matchup_different_total_line_is_not_auto_graded():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "Texas", "away_team": "New York", "Pick Taken": "Over 4.5", "Pick_Outcome": "WIN", "decimal_odds": 1.9, "actual_home_score": 5, "actual_away_score": 2}])
    strategy = pd.DataFrame([{"league": "MLB", "home_team": "Texas", "away_team": "New York", "Pick Taken": "Over 9.5"}])
    realized, summary, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "Result"] == "MISMATCH"
    assert "Line mismatch" in realized.loc[0, "Excluded Reason"]
    assert summary["Bet Count"] == 0


def test_same_matchup_different_spread_line_is_not_auto_graded():
    graded = pd.DataFrame([{"league": "NBA", "home_team": "LAL", "away_team": "BOS", "Pick Taken": "LAL -2.5", "Pick_Outcome": "WIN", "decimal_odds": 1.9, "actual_home_score": 110, "actual_away_score": 101}])
    strategy = pd.DataFrame([{"league": "NBA", "home_team": "LAL", "away_team": "BOS", "Pick Taken": "LAL -6.5"}])
    realized, summary, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "Result"] == "MISMATCH"
    assert summary["ROI"] == 0


def test_same_matchup_opposite_direction_is_not_auto_graded():
    graded = pd.DataFrame([{"league": "NBA", "home_team": "Los Angeles", "away_team": "Houston", "Pick Taken": "Over 207.5", "Pick_Outcome": "LOSS", "decimal_odds": 1.9, "actual_home_score": 105, "actual_away_score": 98}])
    strategy = pd.DataFrame([{"league": "NBA", "home_team": "Los Angeles", "away_team": "Houston", "Pick Taken": "Under 207.5"}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "Result"] == "MISMATCH"
    assert "Direction mismatch" in realized.loc[0, "Excluded Reason"]


def test_exact_pick_identity_match_grades_normally():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "Toronto", "away_team": "Boston", "Pick Taken": "Over 7.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "Stake": 10, "actual_home_score": 8, "actual_away_score": 4}])
    strategy = pd.DataFrame([{"league": "MLB", "home_team": "Toronto", "away_team": "Boston", "Pick Taken": "Over 7.5"}])
    realized, summary, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "pick_identity_match"]
    assert realized.loc[0, "Result"] == "WIN"
    assert summary["Bet Count"] == 1


def test_export_run_id_mismatch_excludes_row_from_realized():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "Toronto", "away_team": "Boston", "Pick Taken": "Over 7.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "Stake": 10, "actual_home_score": 8, "actual_away_score": 4, "export_run_id": "run_a"}])
    strategy = pd.DataFrame([{"league": "MLB", "home_team": "Toronto", "away_team": "Boston", "Pick Taken": "Over 7.5", "export_run_id": "run_b"}])
    realized, summary, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "Excluded Reason"] == "Export run mismatch"
    assert summary["Bet Count"] == 0


def test_no_play_defaults_to_zero_bet_amount():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4, "Status": "No Play"}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_ALL_GRADED, default_stake=10)
    assert realized.loc[0, "Stake"] == 0


def test_high_variance_defaults_to_zero_in_production_card_mode():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4, "Status": "High Variance/Speculative"}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_PRODUCTION_CARD, default_stake=10)
    assert realized.loc[0, "Stake"] == 0


def test_below_threshold_defaults_to_zero_in_production_card_mode():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4, "Status": "Below Threshold"}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_PRODUCTION_CARD, default_stake=10)
    assert realized.loc[0, "Stake"] == 0


def test_manual_non_production_stake_is_flagged():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4, "Status": "No Play", "Stake": 15}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_PRODUCTION_CARD, default_stake=10)
    assert bool(realized.loc[0, "manual_non_production_stake_flag"]) is True
    assert realized.loc[0, "stake_warning"] == "Non-production row has manual stake"


def test_strategy_lab_preserves_pick_identity_fields():
    graded = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4}])
    strategy = pd.DataFrame([{"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "pick_id": "pick-1", "canonical_pick_key": "k1", "market_line_used": 8.5, "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True}])
    realized, _, _ = build_realized_strategy_lab(graded, strategy)
    assert realized.loc[0, "pick_id"] == "pick-1"
    assert realized.loc[0, "canonical_pick_key"] == "k1"
    assert float(realized.loc[0, "market_line_used"]) == 8.5


def test_total_slate_exposure_cap_works():
    graded = pd.DataFrame([{"league": "MLB", "home_team": f"H{i}", "away_team": f"A{i}", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "actual_home_score": 6, "actual_away_score": 4, "Status": "Actionable"} for i in range(10)])
    realized, _, _ = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_ALL_GRADED, default_stake=100, starting_bankroll=1000)
    assert realized["Stake"].max() <= 40.0 + 1e-9
    assert realized["Stake"].sum() <= 250.0 + 1e-9


def test_realized_summary_by_league_market_family():
    graded = pd.DataFrame([
        {"league": "MLB", "home_team": "A", "away_team": "B", "Pick Taken": "Over 8.5", "Pick_Outcome": "WIN", "decimal_odds": 2.0, "Stake": 10, "actual_home_score": 6, "actual_away_score": 4, "Status": "Actionable"},
        {"league": "MLB", "home_team": "C", "away_team": "D", "Pick Taken": "C ML", "Pick_Outcome": "LOSS", "decimal_odds": 2.0, "Stake": 10, "actual_home_score": 1, "actual_away_score": 3, "Status": "Actionable"},
    ])
    _, _, diagnostics = build_realized_strategy_lab(graded, strategy_mode=REALIZED_MODE_ALL_GRADED)
    summary_df = diagnostics["league_market_family_summary"]
    assert set(summary_df["market_family"]) >= {"total", "moneyline"}


def test_strategy_lab_theoretical_render_still_works(monkeypatch):
    calls = []

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummySt:
        def subheader(self, *args, **kwargs):
            calls.append("subheader")

        def info(self, *args, **kwargs):
            calls.append("info")

        def tabs(self, labels):
            calls.append(("tabs", tuple(labels)))
            return DummyContext(), DummyContext()

        def columns(self, n):
            class DummyCol(DummyContext):
                def metric(self, *args, **kwargs):
                    calls.append("metric")

            return tuple(DummyCol() for _ in range(n))

        def selectbox(self, label, options, index=0, help=None):
            calls.append(("selectbox", label, options[index]))
            return options[index]

        def multiselect(self, *args, **kwargs):
            calls.append("multiselect")
            return []

        def markdown(self, *args, **kwargs):
            calls.append("markdown")

        def bar_chart(self, *args, **kwargs):
            calls.append("bar_chart")

        def line_chart(self, *args, **kwargs):
            calls.append("line_chart")

        def write(self, *args, **kwargs):
            calls.append("write")

        def dataframe(self, *args, **kwargs):
            calls.append("dataframe")

        def caption(self, *args, **kwargs):
            calls.append("caption")

        def metric(self, *args, **kwargs):
            calls.append("metric")

        def expander(self, *args, **kwargs):
            return DummyContext()

        def warning(self, *args, **kwargs):
            calls.append("warning")

    monkeypatch.setattr(dash, "st", DummySt())
    monkeypatch.setattr(
        dash,
        "run_performance_pipeline",
        lambda: pd.DataFrame(
            [
                {
                    "league": "MLB",
                    "home_team": "A",
                    "away_team": "B",
                    "Pick Taken": "Over 8.5",
                    "Pick_Outcome": "WIN",
                    "decimal_odds": 1.91,
                    "Stake": 100,
                    "actual_home_score": 5,
                    "actual_away_score": 4,
                    "Status": "Actionable",
                }
            ]
        ),
    )

    analysis_df = pd.DataFrame({"edge": [0.01, 0.02], "expected_value": [0.02, 0.03]})
    portfolio_df = pd.DataFrame({"recommended_bet": [10, 20]})
    parlays_df = pd.DataFrame({"parlay": ["A+B"]})

    dash.render_strategy_lab(analysis_df, portfolio_df, parlays_df, {"bankroll_curves": [[100, 101, 102]]})

    assert ("tabs", ("Theoretical", "Realized")) in calls
    assert "bar_chart" in calls
    assert ("selectbox", "Realized strategy mode", "Actionable only") in calls


def test_warning_for_broad_mode(monkeypatch):
    warnings = []

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummySt:
        def caption(self, *args, **kwargs):
            return None

        def selectbox(self, *args, **kwargs):
            return "All graded bets"

        def multiselect(self, *args, **kwargs):
            return []

        def columns(self, n):
            class DummyCol(DummyContext):
                def metric(self, *args, **kwargs):
                    return None

            return tuple(DummyCol() for _ in range(n))

        def metric(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

        def expander(self, *args, **kwargs):
            return DummyContext()

        def write(self, *args, **kwargs):
            return None

        def warning(self, msg, *args, **kwargs):
            warnings.append(msg)

        def info(self, *args, **kwargs):
            return None

    monkeypatch.setattr(dash, "st", DummySt())
    monkeypatch.setattr(dash, "run_performance_pipeline", _graded_with_statuses)
    dash._render_realized_strategy_lab(pd.DataFrame())

    assert any("broad exploratory strategy" in msg for msg in warnings)
