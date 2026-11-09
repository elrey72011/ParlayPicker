"""NCAAF ratings and Elo vs opening and closing lines: honest fits, no look-ahead, real parsing. SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import importlib
import random

from app_core import ncaaf_ratings_backtest as bt

STRENGTHS = {f"T{i}": (i - 10) * 2.0 for i in range(21)}


def _season(year, *, seed, weeks=15, per_week=10, noise=13.0):
    """Saturdays from late August; margin = strength difference + 2.5 at home + noise; Elo tracks strength."""
    rng = random.Random(seed)
    teams = sorted(STRENGTHS)
    start = datetime(year, 8, 30, 19, tzinfo=timezone.utc)
    rows = []
    for week in range(weeks):
        for k in range(per_week):
            home, away = rng.sample(teams, 2)
            margin = STRENGTHS[home] - STRENGTHS[away] + 2.5 + rng.gauss(0, noise)
            total = 55 + rng.gauss(0, 14)
            rows.append({"id": year * 1000 + week * per_week + k, "season": year, "completed": True,
                         "startDate": (start + timedelta(weeks=week, minutes=k)).isoformat().replace("+00:00", "Z"),
                         "homeTeam": home, "awayTeam": away, "neutralSite": False,
                         "homePoints": round((total + margin) / 2), "awayPoints": round((total - margin) / 2),
                         "homePregameElo": 1500 + 25 * STRENGTHS[home], "awayPregameElo": 1500 + 25 * STRENGTHS[away]})
    return bt.parse_games(rows)


def _lines(games, *, open_off=0.0):
    """Closing lines at the true expected margin; opening lines ``open_off`` points off."""
    out = {}
    for g in games:
        spread = -(STRENGTHS[g["home"]] - STRENGTHS[g["away"]] + 2.5)
        out[g["id"]] = {"spread": spread, "total": 55.0, "spread_open": spread + open_off, "total_open": 55.0,
                        "provider": "consensus"}
    return out


YEARS = range(2021, 2027)


def _data(open_off):
    seasons = {y: _season(y, seed=y) for y in YEARS}
    return seasons, {y: _lines(seasons[y], open_off=open_off) for y in YEARS}


def test_parse_keeps_finished_games_with_elo_and_skips_unfinished():
    rows = [{"id": 1, "startDate": "2025-09-06T16:00:00.000Z", "completed": True, "homeTeam": "A", "awayTeam": "B",
             "homePoints": 31, "awayPoints": 17, "neutralSite": True, "homePregameElo": 1700, "awayPregameElo": 1600},
            {"id": 2, "startDate": "2025-09-06T16:00:00.000Z", "completed": False, "homeTeam": "C", "awayTeam": "D",
             "homePoints": None, "awayPoints": None},
            {"id": 3, "startDate": "2025-09-06T20:00:00.000Z", "completed": True, "homeTeam": "E", "awayTeam": "F",
             "homePoints": 10, "awayPoints": 7}]
    games = bt.parse_games(rows)
    assert [g["id"] for g in games] == [1, 3]
    assert bt.elo_margin(games[0]) == 4.0            # neutral: 100 Elo points / 25, no home points
    assert bt.elo_margin(games[1]) is None           # no Elo, no Elo prediction
    lines = bt.select_lines([{"id": 1, "lines": [
        {"provider": "Bovada", "spread": -3.0, "overUnder": 50.0},
        {"provider": "consensus", "spread": -3.5, "spreadOpen": -2.5, "overUnder": 51.5, "overUnderOpen": 49.0}]}])
    assert lines[1] == {"spread": -3.5, "total": 51.5, "spread_open": -2.5, "total_open": 49.0, "provider": "consensus"}


def test_ratings_carry_across_seasons_and_never_see_later_games():
    seasons = {y: _season(y, seed=y) for y in (2021, 2022)}
    before = bt.predictions(seasons)["ratings"]
    altered = {2021: seasons[2021], 2022: [dict(g) for g in seasons[2022]]}
    altered[2022][-1]["home_points"] += 60
    after = bt.predictions(altered)["ratings"]
    assert all(after[g["id"]] == before[g["id"]] for y in (2021, 2022) for g in seasons[y][:-1] if g["id"] in before)
    # The second season's opener already knows the first season (the carried prior), so it is not a flat guess.
    first = seasons[2022][:10]
    assert len({round(before[g["id"]][0], 6) for g in first}) > 1


def test_efficient_lines_show_no_edge_and_soft_openers_do():
    seasons, lines = _data(open_off=6.0)
    report = bt.run(seasons, lines, [2022, 2023, 2024], [2025, 2026])
    for model in ("SPREAD ratings", "SPREAD elo"):
        fits = report[model]["fits"]
        assert fits["opening"]["k"] > 0.25 and fits["closing"]["k"] < 0.1   # ratings are noisier, so shrunk more
        for year in ("2025", "2026"):
            opening = report[model]["lines"]["opening"][year]["Oct-Jan"]["by_threshold"]["0.55"]
            assert opening["picks"] > 0 and opening["hit_rate"] > 0.6
    assert "TOTAL elo" not in report and "TOTAL ratings" in report


def test_markdown_lists_every_model_line_and_season():
    seasons, lines = _data(open_off=6.0)
    result = {"seasons": list(YEARS), "calibration_years": [2022, 2023, 2024], "scored_years": [2025, 2026],
              "games": {}, "lined": {}, "report": bt.run(seasons, lines, [2022, 2023, 2024], [2025, 2026])}
    text = bt.markdown(result)
    for row in ("| SPREAD elo | opening | 2026 | Oct-Jan |", "| SPREAD ratings | closing | fit |",
                "| TOTAL ratings | opening | 2025 | Aug-Sep |"):
        assert row in text


def test_script_fetches_each_season_once_and_lines_after_the_first():
    script = importlib.import_module("scripts.ncaaf_ratings_backtest")
    calls = []

    class Response:
        status_code = 200

        def __init__(self, url, params):
            calls.append((url.rsplit("/", 1)[1], params["year"], params["seasonType"]))

        def json(self):
            return []

    result = script.run("token", 2021, 2026, get=lambda url, params, **kw: Response(url, params))
    assert sorted(c for c in calls if c[0] == "games") == [("games", y, "both") for y in range(2021, 2027)]
    assert len([c for c in calls if c[0] == "lines"]) == 10 and not any(c[1] == 2021 for c in calls if c[0] == "lines")
    assert result["calibration_years"] == [2022, 2023, 2024] and result["scored_years"] == [2025, 2026]
