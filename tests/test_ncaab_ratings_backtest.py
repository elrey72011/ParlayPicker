"""NCAAB ratings backtest: ratings recover true strengths, never see later games, and score honestly. SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import importlib
import random

import pytest

from app_core import ncaab_ratings_backtest as bt


def _season(year, strengths, *, seed, games_per_day=6, days=120, noise=8.0):
    """Synthetic season: margin = strength difference + 3 at home + noise; totals around 140."""
    rng = random.Random(seed)
    teams = sorted(strengths)
    start = datetime(year - 1, 11, 4, 19, tzinfo=timezone.utc)
    games = []
    for day in range(days):
        for k in range(games_per_day):
            home, away = rng.sample(teams, 2)
            margin = strengths[home] - strengths[away] + 3 + rng.gauss(0, noise)
            total = 140 + rng.gauss(0, 12)
            games.append({"id": f"{year}-{day}-{k}", "start": start + timedelta(days=day, minutes=k),
                          "home": home, "away": away, "neutral": False,
                          "home_points": round((total + margin) / 2), "away_points": round((total - margin) / 2)})
    return games


STRENGTHS = {f"T{i}": (i - 10) * 1.5 for i in range(21)}


def test_ratings_recover_strengths_and_never_see_the_same_day_or_later():
    games = _season(2024, STRENGTHS, seed=1)
    predictions, finals = bt.season_predictions(games)
    fitted = [finals[t][0] for t in sorted(STRENGTHS, key=lambda t: STRENGTHS[t])]
    true = sorted(STRENGTHS.values())
    centred = [f - sum(fitted) / len(fitted) for f in fitted]
    import numpy as np
    assert np.corrcoef(centred, true)[0, 1] > 0.98                           # the order and spacing recovered
    assert sum(abs(c - t) for c, t in zip(centred, true)) / len(true) < 1.5  # within 1.5 points on average
    # The first day's games see no results at all: only the priors (zero) and home court.
    first_day = [g for g in games if g["start"].date() == games[0]["start"].date()]
    assert all(predictions[g["id"]][0] == pytest.approx(predictions[first_day[0]["id"]][0]) for g in first_day)
    # Changing a later game never changes an earlier prediction.
    altered = [dict(g) for g in games]
    altered[-1]["home_points"] += 50
    again, _ = bt.season_predictions(altered)
    assert all(again[g["id"]] == predictions[g["id"]] for g in games[:-1])


def test_opening_lines_are_read_when_present():
    rows = [{"gameId": 1, "lines": [{"provider": "DraftKings", "spread": -4.5, "overUnder": 141.5,
                                     "spreadOpen": -3.5, "overUnderOpen": 139.5}]},
            {"gameId": 2, "lines": [{"provider": "consensus", "spread": 2.0, "overUnder": 150.0}]}]
    lines = bt.select_lines(rows)
    assert lines[1] == {"spread": -4.5, "total": 141.5, "spread_open": -3.5, "total_open": 139.5,
                        "provider": "DraftKings"}
    assert lines[2]["spread_open"] is None and lines[2]["spread"] == 2.0


def _lines_from_truth(games, *, off_by=0.0):
    """Lines set at the true expected margin (an efficient market), optionally off by ``off_by``."""
    out = {}
    for g in games:
        expected = STRENGTHS[g["home"]] - STRENGTHS[g["away"]] + 3
        out[g["id"]] = {"spread": -expected + off_by, "total": 140.0, "spread_open": -expected, "total_open": 140.0,
                        "provider": "x"}
    return out


def test_an_efficient_market_shows_no_edge_and_a_soft_one_does():
    prior, calibration, scored = (_season(y, STRENGTHS, seed=y) for y in (2023, 2024, 2025))
    efficient = bt.run(prior, calibration, scored, _lines_from_truth(calibration), _lines_from_truth(scored))
    close = efficient["SPREAD"]["lines"]["closing"]["Jan-Apr"]["by_threshold"]["0.55"]
    assert close["picks"] < 0.2 * efficient["SPREAD"]["lines"]["closing"]["Jan-Apr"]["games"]
    # Lines that are 4 points off from the truth: the ratings find the difference and win more than lose.
    soft = bt.run(prior, calibration, scored, _lines_from_truth(calibration, off_by=4.0),
                  _lines_from_truth(scored, off_by=4.0))
    row = soft["SPREAD"]["lines"]["closing"]["Jan-Apr"]["by_threshold"]["0.55"]
    assert row["picks"] > 0 and row["hit_rate"] > 0.6
    text = bt.markdown({"prior_season": 2023, "calibration_season": 2024, "scored_season": 2025, "games": {},
                        "closing_lined": 1, "opening_lined": 1, "report": soft})
    assert "| SPREAD | closing | Jan-Apr |" in text and "k = " in text


def test_script_fetches_three_seasons_and_lines_for_two(monkeypatch):
    script = importlib.import_module("scripts.ncaab_ratings_backtest")
    calls = []
    monkeypatch.setattr(script.data, "fetch", lambda endpoint, season, token, get: calls.append((endpoint, season)) or [])
    monkeypatch.setattr(script.bt, "run", lambda *a: {"SPREAD": {"blocker": "TOO_FEW_CALIBRATION_GAMES"}})
    result = script.run("token", 2025, get=object())
    assert sorted(calls) == [("games", 2023), ("games", 2024), ("games", 2025), ("lines", 2024), ("lines", 2025)]
    assert result["prior_season"] == 2023 and result["opening_lined"] == 0
