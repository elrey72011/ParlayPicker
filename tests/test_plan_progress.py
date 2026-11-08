"""Daily progress of a frozen bound plan against its windows (app_core.plan_progress)."""
from datetime import datetime, timezone
import importlib

from app_core import market_skill_validation as ms
from app_core import plan_progress as pp

PLAN = {"validation_plan_id": "prospective-ncaab-spread-bound-v2", "sport": "NCAAB", "market_family": "SPREAD",
        "model_id": "pinnacle-model:NCAAB:SPREAD:x", "minimum_independent_sample": 200,
        "validation_start": "2030-01-01T00:00:00+00:00", "validation_end": "2030-01-11T00:00:00+00:00",
        "holdout_start": "2030-01-11T00:00:00+00:00", "holdout_end": "2030-01-21T00:00:00+00:00",
        "market_skill_methodology": ms.methodology(0.02)}


def _evaluation(games, graded, pending, bets, with_close):
    return {"validation": {"unique_events": games, "scored_observations": graded,
                           "outcomes": {"WIN": graded // 2, "LOSS": graded - graded // 2, "PENDING": pending}},
            "holdout": {"unique_events": 0, "scored_observations": 0, "outcomes": {}},
            "market_skill": {"validation": {"bets": bets, "bet_close_coverage": with_close / bets if bets else 0.0,
                                            "bet_roi": 0.04, "bet_mean_clv": 0.011},
                             "holdout": {"bets": 0, "bet_close_coverage": 0.0, "bet_roi": None, "bet_mean_clv": None}}}


def test_on_pace_window_projects_games_and_bets_to_its_end():
    now = datetime(2030, 1, 5, 12, tzinfo=timezone.utc)                     # 45% through validation
    rows = pp.progress(PLAN, _evaluation(games=110, graded=95, pending=15, bets=27, with_close=24), now=now)
    assert len(rows) == 1                                                   # holdout not started
    row = rows[0]
    assert (row["window"], row["state"], row["elapsed"], row["days_left"]) == ("validation", "IN_PROGRESS", 0.45, 5.5)
    assert (row["games"], row["graded"], row["ungraded"]) == (110, 95, 15)
    assert row["projected_games"] == 244 and row["projected_bets"] == 60 and row["min_bets"] == 50
    assert row["bet_close_coverage"] == round(24 / 27, 3) and row["on_pace"] is True


def test_short_on_bets_or_closes_is_flagged_early():
    now = datetime(2030, 1, 5, 12, tzinfo=timezone.utc)
    few_bets = pp.progress(PLAN, _evaluation(games=110, graded=95, pending=15, bets=12, with_close=12), now=now)[0]
    assert few_bets["projected_bets"] == 27 and few_bets["on_pace"] is False
    few_closes = pp.progress(PLAN, _evaluation(games=110, graded=95, pending=15, bets=27, with_close=15), now=now)[0]
    assert few_closes["projected_bets"] == 60 and few_closes["on_pace"] is False


def test_windows_not_started_are_omitted_and_an_ended_window_reads_ended():
    assert pp.progress(PLAN, _evaluation(0, 0, 0, 0, 0), now=datetime(2029, 12, 31, tzinfo=timezone.utc)) == []
    rows = pp.progress(PLAN, _evaluation(230, 230, 0, 55, 50), now=datetime(2030, 1, 12, tzinfo=timezone.utc))
    assert [(r["window"], r["state"]) for r in rows] == [("validation", "ENDED"), ("holdout", "IN_PROGRESS")]
    assert rows[0]["elapsed"] == 1.0 and rows[0]["on_pace"] is True


def test_table_in_the_readiness_report():
    rows = pp.progress(PLAN, _evaluation(110, 95, 15, 12, 12), now=datetime(2030, 1, 5, 12, tzinfo=timezone.utc))
    text = "\n".join(pp.markdown(rows))
    assert "### Plan progress" in text and "| validation (in_progress) | 45% | 5.5 | 110 (95 / 15) | 244 / 200 |" in text
    assert "**NO**" in text and "+4.0%" in text
    assert "No frozen bound plan has a window running yet." in "\n".join(pp.markdown([]))
    report = importlib.import_module("scripts.market_readiness_report")
    full = report.markdown({"markets": [], "plan_progress": rows})
    assert "### Plan progress" in full
    assert pp.from_store("/nonexistent/prospective-evidence.sqlite3") == []
