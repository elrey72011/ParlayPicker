import pandas as pd
import pytest

from app_core import external_data_fetcher as external


def test_nfl_injury_context_retains_questionable_players_and_weighted_impact(monkeypatch):
    records = [
        {"team": "Los Angeles Rams", "player": "Puka Nacua", "position": "WR", "status": "questionable"},
        {"team": "Los Angeles Rams", "player": "Jordan Whittington", "position": "WR", "status": "doubtful"},
        {"team": "New York Giants", "player": "Starting Corner", "position": "CB", "status": "questionable"},
    ]
    monkeypatch.setattr(external, "_fetch_espn_injuries", lambda league: records)
    external._injury_fetch_status["NFL"] = "available"

    context = external.fetch_injury_context(
        "NFL", "Los Angeles Rams", "New York Giants", "2026-09-21"
    )

    assert context["home"] == 1
    assert context["away"] == 0
    assert context["home_impact"] == pytest.approx(1.10)
    assert context["away_impact"] == pytest.approx(0.28)
    assert "Puka Nacua" in context["home_summary"]
    assert "Jordan Whittington" in context["home_summary"]
    assert context["status"] == "available"


def test_external_enrichment_keeps_nfl_injury_evidence(monkeypatch):
    context = {
        "home": 1,
        "away": 0,
        "home_impact": 1.10,
        "away_impact": 0.28,
        "home_summary": "Puka Nacua (WR) Questionable",
        "away_summary": "Starting Corner (CB) Questionable",
        "source": "espn_injuries",
        "status": "available",
    }
    monkeypatch.setattr(external, "fetch_injury_context", lambda *args: context)
    frame = pd.DataFrame([{
        "league": "NFL",
        "home_team": "Los Angeles Rams",
        "away_team": "New York Giants",
        "game_date": "2026-09-21",
    }])

    enriched = external.enrich_with_external_data(frame)

    assert enriched.loc[0, "injury_home_impact"] == pytest.approx(1.10)
    assert enriched.loc[0, "injury_away_impact"] == pytest.approx(0.28)
    assert enriched.loc[0, "injury_context_source"] == "espn_injuries"
    assert enriched.loc[0, "injury_context_status"] == "available"
    assert "Puka Nacua" in enriched.loc[0, "injury_home_summary"]
