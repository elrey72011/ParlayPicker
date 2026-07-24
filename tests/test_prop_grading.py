import pandas as pd

from app_core.prop_grading import (
    grade_prop_export,
    grading_summary,
    ledger_coverage_summary,
    merge_prop_ledgers,
    validate_prop_export,
)


def _card():
    return pd.DataFrame([
        {
            "player": "Juan Soto",
            "participant_type": "batter",
            "market_type": "batter_hits_over",
            "best_pick": "Juan Soto Over 0.5 Hits",
            "line": 0.5,
            "odds_american": -150,
            "RawWinProbability": 0.72,
            "WinProbability": 0.65,
            "Kelly_Bet_Size": 1.0,
            "Stake_Status": "Funded",
            "CalibrationSource": "directional",
            "CalibrationSampleSize": 24,
            "CalibrationProfileSampleSize": 24,
            "production_eligible": True,
            "production_gate_reason": "Core production qualified",
        },
        {
            "player": "Juan Soto",
            "participant_type": "batter",
            "market_type": "batter_total_bases_over",
            "best_pick": "Juan Soto Over 1.5 TB",
            "line": 1.5,
            "odds_american": 110,
            "RawWinProbability": 0.68,
            "WinProbability": 0.62,
            "Kelly_Bet_Size": 0.0,
            "Stake_Status": "Qualified / No Stake",
        },
    ])


def test_normal_export_grades_funded_and_research_rows_and_preserves_raw_probability():
    calls = []

    def fetch(player_id, participant_type):
        calls.append((player_id, participant_type))
        return {"hits": 1, "total_bases": 2}

    graded = grade_prop_export(
        _card(),
        "2026-07-21",
        name_resolver=lambda card, date: {"juan soto": 1},
        actual_fetcher=fetch,
    )
    assert len(graded) == 2
    assert graded["result"].tolist() == ["WIN", "WIN"]
    assert graded["RawWinProbability"].tolist() == [0.72, 0.68]
    assert graded.loc[0, "CalibrationSource"] == "directional"
    assert graded.loc[0, "CalibrationSampleSize"] == 24
    assert bool(graded.loc[0, "production_eligible"])
    assert graded.loc[0, "production_gate_reason"] == "Core production qualified"
    assert graded["game_date"].eq("2026-07-21").all()
    assert calls == [(1, "batter")]


def test_ledger_merge_replaces_repeat_upload_without_duplication():
    original = pd.DataFrame([{
        "game_date": "2026-07-21", "player": "Juan Soto",
        "pick": "Juan Soto Over 0.5 Hits", "result": None,
    }])
    corrected = pd.DataFrame([{
        "game_date": "2026-07-21", "player": "Juan Soto",
        "pick": "Juan Soto Over 0.5 Hits", "result": "WIN",
    }])
    merged = merge_prop_ledgers(original, corrected)
    assert len(merged) == 1
    assert merged.iloc[0]["result"] == "WIN"


def test_summary_separates_pushes_and_unresolved_rows():
    ledger = pd.DataFrame({"result": ["WIN", "LOSS", "PUSH", None]})
    summary = grading_summary(ledger)
    assert summary == {
        "graded": 2, "wins": 1, "losses": 1,
        "pushes": 1, "unresolved": 1, "win_rate": 0.5,
    }


def test_ledger_coverage_summary_exposes_single_day_history():
    ledger = pd.DataFrame({
        "game_date": ["2026-07-23"] * 3,
        "result": ["WIN", "LOSS", None],
    })
    assert ledger_coverage_summary(ledger) == {
        "rows": 3,
        "settled": 2,
        "date_count": 1,
        "start_date": "2026-07-23",
        "end_date": "2026-07-23",
    }


def test_ledger_coverage_summary_reports_cumulative_date_range():
    ledger = pd.DataFrame({
        "game_date": ["2026-07-21", "2026-07-22", "2026-07-23"],
        "result": ["WIN", "LOSS", "WIN"],
    })
    summary = ledger_coverage_summary(ledger)
    assert summary["settled"] == 3
    assert summary["date_count"] == 3
    assert summary["start_date"] == "2026-07-21"
    assert summary["end_date"] == "2026-07-23"


def test_invalid_export_reports_missing_columns():
    valid, message = validate_prop_export(pd.DataFrame({"player": ["A"]}))
    assert not valid
    assert "market_type" in message
    assert "best_pick" in message

