import pandas as pd

from app.ui.results_dashboard import _performance_recap_table


def test_performance_recap_separates_wagers_precision_and_coverage():
    source = pd.DataFrame(
        {
            "league": ["MLB", "MLB", "NFL"],
            "home_team": ["Home A", "Home B", "Home C"],
            "away_team": ["Away A", "Away B", "Away C"],
            "best_pick": ["Home A +1.5", "Away B +1.5", "Over 40.5"],
            "Outcome": ["WIN", "WIN", "LOSS"],
            "Bettable": [True, False, False],
            "Play_Stake": [5.0, 0.0, 0.0],
            "Wager_Instruction": [
                "BET - APP APPROVED",
                "DO NOT BET - $0 PASS / RESEARCH",
                "DO NOT BET - $0 PASS / RESEARCH",
            ],
            "Export_Scope": [
                "PRODUCTION BET", "BEST AVAILABLE PICK / RESEARCH", "COVERAGE / RESEARCH"
            ],
            "Precision_Card": [False, True, False],
            "Precision_Rank": [pd.NA, 1, pd.NA],
            "Precision_Wager_Approved": [False, False, False],
            "Pick_Status": ["Actionable", "Best Available / Pass", "Best Available / Pass"],
            "export_run_id": ["run-1", "run-1", "run-1"],
        }
    )

    recap = _performance_recap_table(source)

    assert recap["Evaluation Scope"].tolist() == [
        "PRODUCTION-APPROVED WAGER",
        "PRECISION SHORTLIST / RESEARCH",
        "COVERAGE BOARD / DIAGNOSTIC",
    ]
    assert recap["Production Record"].tolist() == [True, False, False]
    assert recap["App-Approved Wager"].tolist() == [True, False, False]
    assert recap["Approved Stake"].tolist() == [5.0, 0.0, 0.0]
    assert recap["Wager Instruction"].tolist()[1].startswith("DO NOT BET")
