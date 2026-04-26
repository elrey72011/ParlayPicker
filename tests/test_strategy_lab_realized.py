import pandas as pd

from app_core.strategy_lab_realized import build_realized_strategy_lab
from app.ui import strategy_lab_dashboard as dash


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
            return tuple(DummyContext() for _ in range(n))

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

    monkeypatch.setattr(dash, "st", DummySt())
    monkeypatch.setattr(dash, "run_performance_pipeline", lambda: pd.DataFrame())

    analysis_df = pd.DataFrame({"edge": [0.01, 0.02], "expected_value": [0.02, 0.03]})
    portfolio_df = pd.DataFrame({"recommended_bet": [10, 20]})
    parlays_df = pd.DataFrame({"parlay": ["A+B"]})

    dash.render_strategy_lab(analysis_df, portfolio_df, parlays_df, {"bankroll_curves": [[100, 101, 102]]})

    assert ("tabs", ("Theoretical", "Realized")) in calls
    assert "bar_chart" in calls
