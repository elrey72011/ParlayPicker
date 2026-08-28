from __future__ import annotations

import pandas as pd

from app_core.performance_pipeline import grade_picks_with_live_results


def test_mixed_slate_backfills_wnba_and_grades_every_row(monkeypatch):
    picks = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Boston Red Sox",
                "away_team": "Oakland Athletics",
                "best_pick": "Over 4.5",
            },
            {
                "league": "WNBA",
                "home_team": "Las Vegas Aces",
                "away_team": "New York Liberty",
                "best_pick": "New York Liberty +4.5",
            },
        ]
    )
    calls: list[list[str]] = []

    def fake_fetch(leagues, target_date=None, attempts=2):
        normalized = [str(league).upper() for league in leagues]
        calls.append(normalized)
        if len(calls) == 1:
            out = pd.DataFrame(
                [
                    {
                        "league": "MLB",
                        "home_team": "Boston Red Sox",
                        "away_team": "Oakland Athletics",
                        "home_score": 5,
                        "away_score": 0,
                        "date": "2026-07-30",
                    }
                ]
            )
        else:
            out = pd.DataFrame(
                [
                    {
                        "league": "WNBA",
                        "home_team": "Las Vegas Aces",
                        "away_team": "New York Liberty",
                        "home_score": 104,
                        "away_score": 99,
                        "date": "2026-07-30",
                    }
                ]
            )
        out.attrs["unsupported_leagues"] = []
        return out

    monkeypatch.setattr("app_core.performance_pipeline.fetch_yesterdays_results", fake_fetch)
    graded = grade_picks_with_live_results(picks)

    assert calls == [["MLB", "WNBA"], ["WNBA"]]
    assert graded["Pick_Outcome"].tolist() == ["WIN", "LOSS"]
    assert graded["grading_status"].tolist() == ["GRADED", "GRADED"]
    assert graded.attrs["settled_count"] == 2
    assert graded.attrs["unresolved_count"] == 0


def test_unsupported_rows_are_explicit_not_silently_pending(monkeypatch):
    picks = pd.DataFrame(
        [{"league": "XYZ", "home_team": "Home", "away_team": "Away", "best_pick": "Over 1.5"}]
    )

    def fake_fetch(leagues, target_date=None, attempts=2):
        out = pd.DataFrame()
        out.attrs["unsupported_leagues"] = ["XYZ"]
        return out

    monkeypatch.setattr("app_core.performance_pipeline.fetch_yesterdays_results", fake_fetch)
    graded = grade_picks_with_live_results(picks)

    assert graded.loc[0, "grading_status"] == "UNSUPPORTED LEAGUE"
    assert graded.loc[0, "grading_issue"] == "no_results_provider_configured"
    assert graded.attrs["unsupported_leagues"] == ["XYZ"]


def test_ncaaf_fcs_final_with_mascot_variant_grades_pick(monkeypatch):
    picks = pd.DataFrame(
        [
            {
                "league": "NCAAF",
                "home_team": "West Florida Argonauts",
                "away_team": "Southern Illinois",
                "best_pick": "Under 55.5",
            }
        ]
    )

    def fake_fetch(leagues, target_date=None, attempts=2):
        out = pd.DataFrame(
            [
                {
                    "league": "NCAAF",
                    "home_team": "WEST FLORIDA ARGONAUTS",
                    "away_team": "SOUTHERN ILLINOIS SALUKIS",
                    "home_score": 34,
                    "away_score": 31,
                    "date": "2026-08-27",
                }
            ]
        )
        out.attrs["unsupported_leagues"] = []
        return out

    monkeypatch.setattr(
        "app_core.performance_pipeline.fetch_yesterdays_results", fake_fetch
    )
    graded = grade_picks_with_live_results(picks)

    assert graded.loc[0, "actual_home_score"] == 34
    assert graded.loc[0, "actual_away_score"] == 31
    assert graded.loc[0, "Pick_Outcome"] == "LOSS"
    assert graded.loc[0, "grading_status"] == "GRADED"
    assert graded.attrs["settled_count"] == 1
    assert graded.attrs["unresolved_count"] == 0
