"""NCAAB possession model: recovers tempo and efficiency, never sees later games, and is scored honestly. SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pytest

from app_core import ncaab_possession_model as pm

MU, EDGE, TAU = 1.04, 0.022, 68.0


def _teams(count=24, seed=7):
    """{team: (offence, defence, tempo, conference)}: the first half is the stronger conference."""
    rng = random.Random(seed)
    out = {}
    for i in range(count):
        strong = i < count // 2
        out[f"T{i:02d}"] = (rng.gauss(0.04 if strong else -0.04, 0.05), rng.gauss(0.04 if strong else -0.04, 0.05),
                           rng.gauss(0, 2.5), "Strong" if strong else "Weak")
    return out


TEAMS = _teams()


def _expected(home, away, neutral=False):
    """(home margin, total) the true ratings give."""
    site = 0.0 if neutral else 1.0
    pace = TAU + TEAMS[home][2] + TEAMS[away][2]
    home_ppp = MU + site * EDGE + TEAMS[home][0] - TEAMS[away][1]
    away_ppp = MU - site * EDGE + TEAMS[away][0] - TEAMS[home][1]
    return pace * (home_ppp - away_ppp), pace * (home_ppp + away_ppp)


def _season(year, *, seed, days=90, games_per_day=8, outsiders=0):
    """(API-shaped game rows, API-shaped team box-score rows) for one synthetic season."""
    rng = random.Random(seed)
    names = sorted(TEAMS)
    start = datetime(year - 1, 11, 4, 23, tzinfo=timezone.utc)
    games, box = [], []

    def play(game_id, when, home, away, home_points, away_points, possessions, conferences):
        games.append({"id": game_id, "startDate": when.isoformat().replace("+00:00", "Z"), "homeTeam": home,
                      "awayTeam": away, "homePoints": home_points, "awayPoints": away_points, "neutralSite": False,
                      "homeConference": conferences[0], "awayConference": conferences[1],
                      "homePeriodPoints": [home_points // 2, home_points - home_points // 2]})
        # Two rows a game, as the API returns them; the model needs only the first.
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            box.append({"gameId": game_id, "team": team, "opponent": opponent, "isHome": is_home, "gameMinutes": 40,
                        "pace": possessions, "teamStats": {"possessions": possessions},
                        "opponentStats": {"possessions": possessions}})

    for day in range(days):
        for k in range(games_per_day):
            home, away = rng.sample(names, 2)
            possessions = TAU + TEAMS[home][2] + TEAMS[away][2] + rng.gauss(0, 3)
            home_ppp = MU + EDGE + TEAMS[home][0] - TEAMS[away][1] + rng.gauss(0, 0.11)
            away_ppp = MU - EDGE + TEAMS[away][0] - TEAMS[home][1] + rng.gauss(0, 0.11)
            home_points, away_points = round(possessions * home_ppp), round(possessions * away_ppp)
            if home_points == away_points:
                home_points += 1
            play(f"{year}-{day}-{k}", start + timedelta(days=day, minutes=k), home, away, home_points, away_points,
                 possessions, (TEAMS[home][3], TEAMS[away][3]))
    for n in range(outsiders):                      # an outside opponent: a few road games, each lost by about 40
        for k in range(3):
            host = names[(n * 3 + k) % len(names)]
            play(f"{year}-out-{n}-{k}", start + timedelta(days=5 + k, minutes=30 + n), host, f"Outside {n}", 95, 55, 70.0,
                 (TEAMS[host][3], None))
    return games, box


def _parsed(year, **kwargs):
    games, box = _season(year, **kwargs)
    return pm.parse_games(games), pm.parse_box(box)


def test_the_book_recovers_efficiency_and_tempo_and_never_sees_the_same_day_or_later():
    games, box = _parsed(2025, seed=1)
    predictions, finals, cover = pm.season_predictions(games, box)
    names = sorted(TEAMS)
    assert cover == {"games": len(games), "with_box_score": len(games), "rated_teams": len(names)}
    net = [finals[t][0] + finals[t][1] for t in names]
    assert np.corrcoef(net, [TEAMS[t][0] + TEAMS[t][1] for t in names])[0, 1] > 0.97
    assert np.corrcoef([finals[t][2] for t in names], [TEAMS[t][2] for t in names])[0, 1] > 0.97
    # The league level takes up whatever the teams' offences and defences average to, and their tempos.
    mean = [sum(TEAMS[t][i] for t in names) / len(names) for i in range(3)]
    assert finals["__league__"]["efficiency"] == pytest.approx(MU + mean[0] - mean[1], abs=0.01)
    assert finals["__league__"]["tempo"] == pytest.approx(TAU + 2 * mean[2], abs=0.5)
    assert finals["__league__"]["overtime"] == pytest.approx(1.0, abs=0.01)       # no overtime in this season
    # Late in the season the predictions are close to what the true ratings give, and closer than early on.
    def miss(part):
        return sum(abs(predictions[g["id"]][0] - _expected(g["home"], g["away"])[0]) for g in part) / len(part)
    assert miss(games[-200:]) < 3.0 and miss(games[-200:]) < miss(games[:200]) - 1.0
    # The first day sees no results: every game gets the same margin and total (the priors and the home edge).
    first_day = [g for g in games if pm._day(g) == pm._day(games[0])]
    assert len(first_day) == 8
    assert all(predictions[g["id"]] == pytest.approx(predictions[first_day[0]["id"]]) for g in first_day)
    assert predictions[first_day[0]["id"]] == pytest.approx((2 * pm.PARAMS["home_edge"] * 68.0 * 1.0075,
                                                             2 * 1.04 * 68.0 * 1.0075))
    # Changing the last game changes no earlier prediction; changing a game changes no other game of its day.
    altered = [dict(g) for g in games]
    altered[-1]["home_points"] += 50
    altered[8]["home_points"] += 50                 # the first game of the second day
    again, _, _ = pm.season_predictions(altered, box)
    second_day = [g for g in games if pm._day(g) == pm._day(games[8])]
    assert all(again[g["id"]] == predictions[g["id"]] for g in games[:8] + second_day)
    assert again[games[16]["id"]] != predictions[games[16]["id"]]                 # the third day does see it


def test_a_late_game_shares_its_day_with_the_evening_before():
    def game(game_id, when, home, away, home_points=80, away_points=60):
        return {"id": game_id, "start": when, "home": home, "away": away, "home_points": home_points,
                "away_points": away_points, "neutral": False, "periods": 2}
    evening = datetime(2026, 1, 10, 23, 30, tzinfo=timezone.utc)
    schedule = [game(1, evening, "A", "B"), game(2, evening + timedelta(hours=3, minutes=30), "A", "C"),   # 10 PM Eastern
                game(3, evening + timedelta(hours=20), "B", "C")]                                        # the next evening
    assert pm._day(schedule[0]) == pm._day(schedule[1]) != pm._day(schedule[2])
    filler = [game(100 + i, evening - timedelta(days=40 - i), ("A", "B", "C")[i % 3], ("B", "C", "A")[i % 3], 70, 69)
              for i in range(30)]                   # enough earlier games for all three teams to be rated
    base, _, _ = pm.season_predictions(filler + schedule, {})
    changed = [dict(g) for g in filler + schedule]
    changed[30]["home_points"] = 120                # the 6:30 PM game, still being played when the late one tips off
    again, _, _ = pm.season_predictions(changed, {})
    assert again[2] == base[2] and again[3] != base[3]


def test_outside_opponents_are_pooled_into_one_team():
    games, box = _parsed(2025, seed=2, outsiders=3)
    assert pm.rated_teams(games) == set(TEAMS)
    predictions, finals, cover = pm.season_predictions(games, box)
    outside = [g for g in games if g["away"].startswith("Outside")]
    assert len(outside) == 9 and not any(g["id"] in predictions for g in outside)       # used, never predicted
    assert cover["games"] == len(games) and cover["rated_teams"] == len(TEAMS)
    assert pm.POOLED in finals and not any(name.startswith("Outside") for name in finals)
    assert finals[pm.POOLED][0] + finals[pm.POOLED][1] < -0.15       # far below every rated team
    # A game between two unrated teams teaches nothing about a rated one and is left out.
    extra = dict(outside[0], id="both-out", home="Outside 9", away="Outside 8")
    _, _, with_extra = pm.season_predictions(games + [extra], box)
    assert with_extra["games"] == cover["games"]


def test_a_blowout_counts_for_the_cap_and_keeps_the_total():
    def book_after(home_points, away_points):
        book = pm.PossessionBook({"A", "B"})
        book.add({"home": "A", "away": "B", "neutral": True, "home_points": home_points, "away_points": away_points},
                 (70.0, 40.0))
        return book
    capped, by_cap = book_after(100, 40), book_after(82.5, 57.5)          # 60 apart and 25 apart, the same 140 points
    assert np.allclose(capped.e_vec, by_cap.e_vec) and np.allclose(capped.e_mat, by_cap.e_mat)
    assert np.allclose(book_after(90, 50).e_vec, capped.e_vec)            # 40 apart is capped to the same 25
    free = pm.PossessionBook({"A", "B"}, params={"margin_cap": 0})
    free.add({"home": "A", "away": "B", "neutral": True, "home_points": 100, "away_points": 40}, (70.0, 40.0))
    assert not np.allclose(free.e_vec, capped.e_vec)                      # no cap: the 60 points count in full


def test_possessions_and_minutes_come_from_the_box_score_with_fallbacks():
    estimate = {"fieldGoals": {"attempted": 60}, "rebounds": {"offensive": 10}, "turnovers": {"total": 12},
                "freeThrows": {"attempted": 20}}                          # 60 - 10 + 12 + 0.475 * 20 = 71.5
    rows = [
        {"gameId": 1, "gameMinutes": 40, "pace": 70.0, "teamStats": {"possessions": 71}, "opponentStats": {"possessions": 69}},
        {"gameId": 1, "gameMinutes": 40, "teamStats": {"possessions": 10}, "opponentStats": {"possessions": 10}},   # 2nd row
        {"gameId": 2, "gameMinutes": 225, "teamStats": estimate, "opponentStats": {"possessions": None}},  # player minutes
        {"gameId": 3, "pace": 64.0, "teamStats": {"possessions": 72}, "opponentStats": {"possessions": 72}},
        {"gameId": 4, "teamStats": {"possessions": 300}, "opponentStats": {"possessions": 300}},           # not a real game
        {"gameId": 5, "teamStats": {}, "opponentStats": None},                                             # nothing to read
        {"teamStats": {"possessions": 70}, "opponentStats": {"possessions": 70}},                          # no game id
    ]
    box = pm.parse_box(rows)
    assert box == {1: {"possessions": 70.0, "minutes": 40.0, "pace": 70.0},
                   2: {"possessions": 71.5, "minutes": 45.0, "pace": None},
                   3: {"possessions": 72.0, "minutes": None, "pace": 64.0}}
    assert pm.game_tempo({"id": 1, "periods": 2}, box) == (70.0, 40.0)
    assert pm.game_tempo({"id": 2, "periods": 2}, box) == (pytest.approx(71.5 * 40 / 45), 45.0)     # the box score wins
    assert pm.game_tempo({"id": 3, "periods": 4}, box) == (pytest.approx(72.0 * 40 / 50), 50.0)     # two overtimes
    assert pm.game_tempo({"id": 3, "periods": 0}, box) == (pytest.approx(64.0), 45.0)               # from the pace
    assert pm.game_tempo({"id": 9, "periods": 2}, box) is None
    assert pm.game_minutes({"periods": 0}) == 40.0 and pm.game_minutes({"periods": 3}) == 45.0


def test_a_game_with_no_box_score_is_given_the_average_pace_and_overtime_lifts_the_forecast():
    book = pm.PossessionBook({"A", "B"})
    assert book.average_tempo() == 68.0
    book.add({"home": "A", "away": "B", "neutral": False, "home_points": 75, "away_points": 70, "periods": 2}, (75.0, 40.0))
    tempo_before = book.t_vec.copy()
    book.add({"home": "B", "away": "A", "neutral": False, "home_points": 70, "away_points": 75, "periods": 2}, None)
    assert book.games == 2 and book.boxed == 1 and np.array_equal(book.t_vec, tempo_before)
    assert book.average_tempo() == 75.0
    # Overtime: a season of 45-minute games moves the factor toward 45 / 40, held back by the prior.
    for _ in range(200):
        book.add({"home": "A", "away": "B", "neutral": True, "home_points": 80, "away_points": 78, "periods": 3},
                 (70.0, 45.0))
    overtime = book.solve()[2]
    assert overtime == pytest.approx((40.0 / 40 + 200 * 45.0 / 40 + 200 * 1.0075) / 401)
    efficiency, tempo, _ = book.solve()
    margin, total = book.predict({"home": "A", "away": "B", "neutral": True}, efficiency, tempo, overtime)
    regulation = book.predict({"home": "A", "away": "B", "neutral": True}, efficiency, tempo, 1.0)
    assert total == pytest.approx(regulation[1] * overtime) and margin == pytest.approx(regulation[0] * overtime)


def test_tune_picks_the_grid_entry_with_the_smallest_error(monkeypatch):
    first, second = _parsed(2024, seed=3, days=40), _parsed(2025, seed=4, days=40)
    grid = ({"carry": 0.0, "prior_weight": 4.0}, {"carry": 0.9, "prior_weight": 4.0}, {"carry": 0.9, "prior_weight": 50.0})
    solved = []
    real = pm.season_predictions

    def counting(games, box, prior=None, params=None):
        solved.append((prior is None, dict(params or {})))
        return real(games, box, prior, params)
    monkeypatch.setattr(pm, "season_predictions", counting)
    chosen, rows = pm.tune(first[0], first[1], second[0], second[1], grid)
    monkeypatch.undo()
    # The first season has nothing to carry, so it is run once for each setting of the rest: twice, not three times.
    assert [params for no_prior, params in solved if no_prior] == [{"prior_weight": 4.0}, {"prior_weight": 50.0}]
    assert len(solved) == 5
    assert [(row["carry"], row["prior_weight"]) for row in rows] == [(0.0, 4.0), (0.9, 4.0), (0.9, 50.0)]
    best = min(rows, key=lambda row: row["margin_rmse"] + row["total_rmse"])
    assert chosen == {"carry": best["carry"], "prior_weight": best["prior_weight"]}
    assert len(pm.GRID) == 18 and set(pm.GRID[0]) == {"carry", "prior_weight", "margin_cap"}
    # The same teams are as good both seasons, so carrying last season over beats starting from nothing.
    assert chosen["carry"] == 0.9 and rows[0]["margin_rmse"] > rows[1]["margin_rmse"]
    assert all(row["games"] == len(second[0]) for row in rows)
    assert pm.prediction_error(second[0], {}) == {"games": 0, "margin_rmse": None, "total_rmse": None}


def test_accuracy_reads_models_and_lines_on_the_same_games():
    def game(game_id, month, home_points, away_points):
        return {"id": game_id, "start": datetime(2026, month, 5, 23, tzinfo=timezone.utc), "home_points": home_points,
                "away_points": away_points}
    games = [game(1, 1, 80, 70), game(2, 12, 60, 66), game(3, 1, 75, 70), game(4, 1, 70, 60)]
    lines = {1: {"spread": -8.0, "spread_open": -6.0, "total": 152.0, "total_open": 149.0},
             2: {"spread": 3.0, "spread_open": 1.0, "total": 130.0, "total_open": 128.0},
             3: {"spread": -5.0, "spread_open": None, "total": 140.0, "total_open": 139.0},       # no opening spread
             4: {"spread": -9.0, "spread_open": -9.0, "total": 131.0, "total_open": 131.0}}
    models = {"a": {1: (12.0, 151.0), 2: (-2.0, 124.0), 3: (5.0, 146.0), 4: (0.0, 0.0)},
              "b": {1: (9.0, 147.0), 2: (-8.0, 130.0), 3: (4.0, 144.0)}}                           # b never saw game 4
    out = pm.accuracy(games, models, lines)
    assert out["SPREAD"]["all"] == {"games": 2, "a": 3.0, "b": 1.5, "opening_line": 4.5, "closing_line": 2.5}
    assert out["SPREAD"]["Jan-Apr"] == {"games": 1, "a": 2.0, "b": 1.0, "opening_line": 4.0, "closing_line": 2.0}
    assert out["SPREAD"]["Nov-Dec"]["games"] == 1
    assert out["TOTAL"]["all"] == {"games": 3, "a": pytest.approx(4 / 3), "b": pytest.approx(8 / 3),
                                   "opening_line": pytest.approx(3.0), "closing_line": pytest.approx(11 / 3)}
    assert pm.accuracy([], models, lines)["SPREAD"]["all"] == {"games": 0}


def test_conference_tiers_rank_last_seasons_ratings():
    games, box = _parsed(2025, seed=5, days=40, outsiders=2)
    _, finals, _ = pm.season_predictions(games, box)
    tiers = pm.conference_tiers(games, finals, top=1)
    assert tiers == {"Strong": "high", "Weak": "other"}
    assert pm.conference_tiers(games, finals) == {"Strong": "high", "Weak": "high"}        # fewer than seven exist

    def tier(home, away):
        return pm.game_tier({"home_conference": home, "away_conference": away}, tiers)
    assert (tier("Strong", "Strong"), tier("Strong", "Weak"), tier("Weak", "Weak"), tier(None, "Weak")) == (
        "high", "mixed", "low", "low")
    assert pm.conference_tiers(games, {}, top=1) == {"Strong": "other", "Weak": "other"}     # no ratings: none high


def _lines(games, *, open_off=0.0):
    """Closing lines at what the true ratings give; opening spreads ``open_off`` points too cheap for the home side."""
    out = {}
    for game in games:
        if game["home"] not in TEAMS or game["away"] not in TEAMS:
            continue
        margin, total = _expected(game["home"], game["away"])
        out[game["id"]] = {"spread": -margin, "total": total, "spread_open": -margin + open_off, "total_open": total,
                           "provider": "x"}
    return out


def test_the_run_scores_both_models_on_the_same_seasons():
    seasons = [_parsed(year, seed=year, outsiders=2) for year in (2024, 2025, 2026)]
    grid = ({"carry": 0.5, "prior_weight": 4.0}, {"carry": 0.9, "prior_weight": 4.0})
    lines = [_lines(games, open_off=4.0) for games, _ in seasons]
    report = pm.run(seasons[0], seasons[1], seasons[2], lines[1], lines[2], grid=grid)
    assert report["tuning"]["chosen"] in grid and len(report["tuning"]["grid"]) == 2
    assert set(report["coverage"]) == {"prior", "calibration", "scored"}
    assert report["coverage"]["scored"]["with_box_score"] == len(seasons[2][0])
    lined = len(lines[2])
    row = report["accuracy"]["SPREAD"]["all"]
    assert row["games"] == lined and set(row) == {"games", "possession", "ratings", "opening_line", "closing_line"}
    assert row["closing_line"] < row["opening_line"] and row["possession"] < row["opening_line"]
    assert report["accuracy"]["TOTAL"]["Jan-Apr"]["games"] > 0 and report["accuracy"]["TOTAL"]["all"]["ratings"] > 0
    assert set(report["models"]) == {"possession", "ratings"}
    for name in ("possession", "ratings"):          # an opening line 4 points off: both find it, in every reading
        spread = report["models"][name]["SPREAD"]
        assert spread["fits"]["opening"]["k"] > 0.5 > spread["fits"]["closing"]["k"]
        assert spread["line_move_slope"]["all"]["slope"] > 0.5 and spread["line_move_slope"]["all"]["t"] > 10
        assert spread["disagreement"]["opening"]["Jan-Apr"]["2"]["hit_rate"] > 0.58
    assert report["high_conferences"] == ["Strong", "Weak"]
    tiers = report["by_tier"]["SPREAD"]
    assert set(tiers) == {"high", "mixed", "low"} and tiers["high"]["line_move_slope"]["games"] == lined
    assert tiers["low"]["line_move_slope"] == {"games": 0, "slope": None, "se": None, "t": None}
    # Which side: the opening spread is too cheap for the home side, so that is the side the model takes and wins on.
    sides = report["by_side"]["SPREAD"]
    assert sides["over_or_home"]["2"]["picks"] > 5 * sides["under_or_away"]["2"]["picks"]
    assert sides["over_or_home"]["2"]["hit_rate"] > 0.58
    assert sides["over_or_home"]["2"]["picks"] + sides["under_or_away"]["2"]["picks"] == \
        report["models"]["possession"]["SPREAD"]["disagreement"]["opening"]["all"]["2"]["picks"]
    # With no model: the home side beat that cheap opening spread far more often than the fair closing one.
    lean = report["market_lean"]["SPREAD"]
    assert lean["opening"]["share"] > 0.6 > lean["closing"]["share"] > 0.4
    assert lean["opening"]["over_or_home"] + lean["opening"]["under_or_away"] + lean["opening"]["push"] == lined
    text = pm.markdown({"prior_season": 2024, "calibration_season": 2025, "scored_season": 2026, "games": {},
                        "closing_lined": lined, "opening_lined": lined, "report": report})
    for expected in ("## NCAAB possession model vs the points ratings and the lines (CBBD)",
                     "### Tuning grid (calibration season; smaller is better)",
                     "### How far each misses the final result",
                     "| SPREAD | all | ", "top conferences: Strong, Weak", "| SPREAD | high | ",
                     "## Possession model vs archived lines", "## Points ratings vs archived lines",
                     "### Did the closing line move toward the ratings?",
                     "### Which side the possession model took at the opening line (no fit)", "| TOTAL | Under | 4+ |",
                     "### With no model: how often the Over and the home side beat each line", "| SPREAD | opening | ",
                     "about 0.7 points better than the close for a spread and 1.0 for a total"):
        assert expected in text, expected
    from app_core import ncaab_point_value, ncaab_ratings_backtest
    assert ncaab_ratings_backtest.POINT_VALUE == ncaab_point_value.ASSUMED          # one working value, not two


def test_the_script_reads_three_seasons_of_games_and_box_scores_and_two_of_lines(monkeypatch, tmp_path):
    script = importlib.import_module("scripts.ncaab_possession_backtest")
    seasons = {year: _season(year, seed=year, days=50, outsiders=1) for year in (2024, 2025, 2026)}
    calls = []

    def fetch(endpoint, season, token, get):
        calls.append((endpoint, season))
        games, box = seasons[season]
        if endpoint == "games":
            return games
        if endpoint == "games/teams":
            return box
        return [{"gameId": game["id"], "lines": [{"provider": "consensus", "spread": -2.5, "spreadOpen": -1.5,
                                                  "overUnder": 141.5, "overUnderOpen": 140.5}]} for game in games]
    monkeypatch.setattr(script.data, "fetch", fetch)
    result = script.run("token", 2026, get=object())
    assert sorted(calls) == [("games", 2024), ("games", 2025), ("games", 2026), ("games/teams", 2024),
                             ("games/teams", 2025), ("games/teams", 2026), ("lines", 2025), ("lines", 2026)]
    played = len(seasons[2026][0])
    assert result["games"]["2026"] == played and result["box_scores"]["2026"] == played     # one entry a game, not two
    assert result["closing_lined"] == played and result["opening_lined"] == played and result["providers"] == ["consensus"]
    assert (result["prior_season"], result["calibration_season"], result["scored_season"]) == (2024, 2025, 2026)
    cover = result["report"]["coverage"]["scored"]
    assert cover["games"] == cover["with_box_score"] == played and cover["rated_teams"] == len(TEAMS)
    assert len(result["report"]["tuning"]["grid"]) == len(pm.GRID)
    json.dumps(result, default=str)                                                          # the report can be saved
    # main: needs a key, writes the report, and prints every section.
    monkeypatch.delenv("CBBD_API_KEY", raising=False)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    with pytest.raises(SystemExit, match="MISSING_CREDENTIALS"):
        script.main(["--output", str(tmp_path / "out.json")])
    monkeypatch.setenv("CFBD_API_KEY", "token")
    monkeypatch.setattr(script, "run", lambda token, season: dict(result, scored_season=season))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert script.main(["--output", str(tmp_path / "deep" / "out.json")]) == 0
    saved = json.loads((tmp_path / "deep" / "out.json").read_text(encoding="utf-8"))
    assert saved["scored_season"] == 2026 and "## Points ratings vs archived lines" in summary.read_text(encoding="utf-8")


def test_the_workflow_offers_both_reports_and_runs_by_hand_only():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/ncaab-ratings-backtest.yml").read_text(encoding="utf-8")
    assert "options: [ratings, possession]" in text and "default: ratings" in text
    assert "if: ${{ inputs.report != 'possession' }}" in text       # the ratings screen still runs by default
    assert "if: ${{ inputs.report == 'possession' }}" in text and "python -u scripts/ncaab_possession_backtest.py" in text
    assert "workflow_dispatch:" in text
    assert not re.search(r"^\s*(schedule|push|pull_request|workflow_run):", text, flags=re.M)
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"CBBD_API_KEY", "CFBD_API_KEY"}


def test_by_side_and_market_lean_count_each_game_once():
    def game(game_id, home_points, away_points):
        return {"id": game_id, "start": datetime(2026, 1, 5, 23, tzinfo=timezone.utc), "home_points": home_points,
                "away_points": away_points}
    games = [game(1, 80, 70), game(2, 60, 66), game(3, 70, 70), game(4, 75, 60)]
    lines = {1: {"spread": -8.0, "spread_open": -6.0, "total": 152.0, "total_open": 145.0},
             2: {"spread": 3.0, "spread_open": 1.0, "total": 130.0, "total_open": 131.0},
             3: {"spread": 0.0, "spread_open": -2.0, "total": 140.0, "total_open": 140.0},
             4: {"spread": -9.0, "spread_open": None, "total": 131.0, "total_open": None}}
    lean = pm.market_lean(games, lines)
    # Closing spreads: game 1 wins by 10 laying 8 (home), game 2 loses by 6 getting 3 (away), game 3 pushes, game 4 home.
    assert lean["SPREAD"]["closing"] == {"over_or_home": 2, "under_or_away": 1, "push": 1, "share": pytest.approx(2 / 3)}
    assert lean["SPREAD"]["opening"] == {"over_or_home": 1, "under_or_away": 2, "push": 0, "share": pytest.approx(1 / 3)}
    assert lean["TOTAL"]["opening"] == {"over_or_home": 1, "under_or_away": 1, "push": 1, "share": 0.5}
    assert pm.market_lean([], lines)["TOTAL"]["closing"]["share"] is None
    predictions = {1: (10.0, 151.0), 2: (-9.0, 125.0), 3: (2.0, 150.0)}       # game 4 has no opening line
    sides = pm.by_side(games, predictions, lines)
    # Totals at the open: game 1 Over by 6 (150 > 145: won), game 2 Under by 6 (126 < 131: won), game 3 Over by 10 (push).
    assert sides["TOTAL"]["over_or_home"]["6"] == {"picks": 2, "wins": 1, "losses": 0, "hit_rate": 1.0,
                                                   "roi_per_pick": pytest.approx(100 / 110 / 2)}
    assert sides["TOTAL"]["under_or_away"]["6"]["picks"] == 1 and sides["TOTAL"]["under_or_away"]["6"]["wins"] == 1
    assert sides["TOTAL"]["over_or_home"]["8"]["picks"] == 1                   # only game 3 is 8 or more apart
