"""The NCAAB availability test: who was missing is read from earlier games only, and the verdict rule holds.

Every season here is synthetic. In one world the closing line knows who is out; in the other it does not.
The test must call the first "priced in" and the second "worth a forward test".
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

import pytest

from app_core import ncaab_availability as model
from app_core import ncaab_possession_model as possession
from app_core import ncaab_ratings_backtest as ratings

ROOT = Path(__file__).resolve().parents[1]
STARS, BENCH = ("s1", "s2", "s3"), ("b1", "b2", "b3", "b4", "b5")
STAR_COST = 5.0            # points a team loses with a star out


def _when(year, day):
    return (datetime(year - 1, 11, 4, 23, tzinfo=timezone.utc) + timedelta(days=day)).isoformat().replace("+00:00", "Z")


def _box_row(game_id, team, is_home, out=()):
    players = [{"athleteId": f"{team}-{name}", "name": name, "minutes": 0 if name in out else minutes}
               for names, minutes in ((STARS, 32), (BENCH, 18)) for name in names]
    return {"gameId": game_id, "team": team, "isHome": is_home, "players": players}


def _season(year, *, priced, seed, teams=48, days=110):
    """(game rows, line rows, player rows): a star's absence costs ``STAR_COST``; ``priced`` puts it in the line."""
    rng = random.Random(seed)
    names = [f"T{i:02d}" for i in range(teams)]
    strength = {name: rng.gauss(0, 6) for name in names}
    out_until = {}                                      # (team, star) -> last day of the absence
    games, lines, box, game_id = [], [], [], year * 100000
    for day in range(days):
        order = names[:]
        rng.shuffle(order)
        for home, away in zip(order[0::2], order[1::2]):
            game_id += 1
            missing = {}
            for team in (home, away):
                for star in STARS:
                    if out_until.get((team, star), -1) < day and rng.random() < 0.012:
                        out_until[(team, star)] = day + rng.choice((1, 8, 20))
                missing[team] = [star for star in STARS if out_until.get((team, star), -1) >= day]
            known = strength[home] - strength[away] + 3.0
            truth = known - STAR_COST * (len(missing[home]) - len(missing[away]))
            margin = round(truth + rng.gauss(0, 10)) or 1
            total = round(140 + rng.gauss(0, 12))
            home_points = (total + margin) // 2
            games.append({"id": game_id, "startDate": _when(year, day), "homeTeam": home, "awayTeam": away,
                          "homePoints": home_points, "awayPoints": home_points - margin, "neutralSite": False,
                          "homeConference": "Strong" if home < "T24" else "Weak",
                          "awayConference": "Strong" if away < "T24" else "Weak"})
            spread = -round((truth if priced else known) * 2) / 2
            lines.append({"gameId": game_id, "lines": [{"provider": "consensus", "spread": spread,
                                                         "spreadOpen": -round(known * 2) / 2, "overUnder": 140.5}]})
            box += [_box_row(game_id, home, True, missing[home]), _box_row(game_id, away, False, missing[away])]
    return games, lines, box


def _report(priced):
    seasons, prior = [], possession.parse_games(_season(2024, priced=priced, seed=1)[0])
    for year, seed in ((2025, 2), (2026, 3)):
        games, lines, box = _season(year, priced=priced, seed=seed)
        parsed = possession.parse_games(games)
        seasons.append((year, parsed, prior, model.parse_players(box), ratings.select_lines(lines)))
        prior = parsed
    return model.run(seasons)


# ---- who was missing ----

def _one_team(star_out_in):
    """Team A's games 1..N at home to a different opponent each time; s1 sits in the listed games."""
    games = [{"id": n, "start": datetime(2025, 11, n, tzinfo=timezone.utc), "home": "A", "away": f"O{n}",
              "home_points": 70.0, "away_points": 60.0, "neutral": False} for n in range(1, 11)]
    box = model.parse_players([_box_row(n, "A", True, ("s1",) if n in star_out_in else ()) for n in range(1, 11)])
    return games, box


def test_a_team_is_judged_only_after_five_games_and_the_streak_counts_games_already_missed():
    games, box = _one_team({7, 8})
    status = model.availability(games, box)
    assert sorted(game for game, _ in status) == [6, 7, 8, 9, 10]
    assert status[(6, True)] == {"key": 3, "absent": 0, "streak": None}
    assert status[(7, True)] == {"key": 3, "absent": 1, "streak": 0}
    assert status[(8, True)] == {"key": 3, "absent": 1, "streak": 1}
    assert status[(9, True)]["absent"] == 0


def test_nothing_about_a_game_depends_on_games_after_it():
    games, box = _one_team({7, 8})
    whole = model.availability(games, box)
    for upto in range(6, 11):
        early = model.availability(games[:upto], {key: value for key, value in box.items() if key[0] <= upto})
        assert all(early[key] == whole[key] for key in early) and (upto, True) in early


def test_a_player_listed_with_no_minutes_did_not_play_and_a_bench_player_is_never_key():
    rows = [_box_row(1, "A", True, ("s1",))]
    assert "A-s1" not in model.parse_players(rows)[(1, True)]
    games, box = _one_team(set())
    box[(7, True)].pop("A-b1")                                   # an 18-minute player sits
    assert model.availability(games, box)[(7, True)]["absent"] == 0


def test_a_star_out_since_before_the_team_could_be_judged_stops_counting_as_key():
    games, box = _one_team(set(range(3, 11)))                    # 2 games played, then gone
    status = model.availability(games, box)
    assert all(entry["key"] == 2 and entry["absent"] == 0 for entry in status.values())


# ---- the case from the short side's view ----

def test_the_cover_and_the_line_move_are_read_from_the_short_sides_view():
    games, box_rows = [], []
    for n in range(1, 8):
        games += [{"id": n, "start": datetime(2025, 11, n, tzinfo=timezone.utc), "home": "H", "away": "A" if n == 7 else f"X{n}",
                   "home_points": 80.0, "away_points": 70.0, "neutral": False},
                  {"id": 100 + n, "start": datetime(2025, 11, n, 1, tzinfo=timezone.utc), "home": f"Y{n}", "away": "A",
                   "home_points": 60.0, "away_points": 70.0, "neutral": False}]
        box_rows += [_box_row(n, "H", True), _box_row(100 + n, "A", False)]
    games = [game for game in games if game["id"] != 107]
    box_rows = [row for row in box_rows if row["gameId"] != 107] + [_box_row(7, "A", False, ("s2",))]
    lines = {7: {"spread": -4.0, "spread_open": -2.0, "total": 145.5, "total_open": None, "provider": "consensus"}}
    short, full = model.cases(sorted(games, key=lambda g: g["start"]), model.parse_players(box_rows), lines, {})
    assert full == [] and len(short) == 1
    case = short[0]
    assert (case["site"], case["absent"], case["streak"], case["streak_label"]) == ("away", 1, 0, "first game missed")
    assert case["cover"] == -6.0          # home won by 10 laying 4: the short visitors missed by 6
    assert case["move"] == 2.0            # and the line had moved 2 points against them
    assert case["total_error"] == 4.5


def test_the_record_counts_the_fade_and_pushes_stay_out_of_the_rate():
    r = model.record([{"cover": -3.0, "move": 1.0}, {"cover": -1.0, "move": None}, {"cover": 2.0, "move": 0.0},
                      {"cover": 0.0, "move": None}, {"cover": None, "move": None}])
    assert (r["games"], r["covered"], r["failed"], r["pushes"]) == (4, 1, 2, 1)
    assert r["fade_rate"] == pytest.approx(2 / 3) and r["fade_units"] == pytest.approx(2 * 100 / 110 - 1)
    assert r["mean_move"] == 0.5 and r["moves"] == 2 and r["fade_low"] < 2 / 3 < r["fade_high"]


# ---- the verdict ----

def test_a_line_that_knows_who_is_out_is_called_priced_in():
    report = _report(priced=True)
    headline = report["verdict"]["headline"]
    assert headline["games"] >= model.MIN_CASES and report["verdict"]["answer"] == "priced in"
    assert headline["mean_move"] > 3                      # the line moved by about the star's worth


def test_a_line_that_does_not_know_is_called_worth_a_forward_test():
    report = _report(priced=False)
    headline = report["verdict"]["headline"]
    assert report["verdict"]["answer"] == "worth a forward test"
    assert headline["fade_rate"] > 0.58 and headline["mean_cover"] < -3 and headline["mean_move"] == 0
    assert set(report["verdict"]["by_season"]) == {"2025", "2026"}
    assert report["coverage"]["2026"]["lined_with_one_box_score"] == 0


def test_too_few_games_is_said_plainly_and_a_single_losing_season_blocks_the_verdict():
    few = [{"season": 2026, "streak": 0, "cover": -5.0, "move": None}] * (model.MIN_CASES - 1)
    assert model.verdict(few)["answer"] == "too few games to say"
    strong = [{"season": 2025, "streak": 0, "cover": -5.0, "move": None}] * 400
    weak = ([{"season": 2026, "streak": 0, "cover": 5.0, "move": None}] * 21
            + [{"season": 2026, "streak": 0, "cover": -5.0, "move": None}] * 19)
    assert model.verdict(strong)["answer"] == "worth a forward test"
    assert model.verdict(strong + weak)["answer"] == "priced in"
    later = [{"season": 2026, "streak": 3, "cover": -5.0, "move": None}] * 5000
    assert model.verdict(later)["answer"] == "too few games to say"      # only first games missed are the headline


# ---- reading the provider ----

class _Response:
    status_code = 200

    def __init__(self, rows):
        self._rows = rows

    def json(self):
        return self._rows


def test_both_teams_rows_are_kept_and_a_full_window_is_halved_until_it_fits(monkeypatch):
    monkeypatch.setattr(model, "SPLIT_AT", 6)
    calls = []

    def get(url, params, **_):
        calls.append((params["startDateRange"], params["endDateRange"]))
        assert url.endswith("/games/players") and params["season"] == 2026
        first, last = (datetime.fromisoformat(params[key]).date() for key in ("startDateRange", "endDateRange"))
        rows = []
        for day in range((last - first).days + 1):               # one game a day in the first week of December
            when = first + timedelta(days=day)
            if (when.year, when.month) == (2025, 12) and when.day <= 7:
                rows += [_box_row(when.day, "H", True), _box_row(when.day, "A", False)]
        return _Response(rows)

    box = model.fetch_players(2026, "token", get)
    assert sorted(box) == sorted((day, side) for day in range(1, 8) for side in (True, False))
    assert len(calls) > 25 and len(set(calls)) == len(calls)


def test_a_day_that_alone_fills_a_window_stops_the_run_rather_than_lose_games(monkeypatch):
    monkeypatch.setattr(model, "SPLIT_AT", 2)
    with pytest.raises(ValueError, match="single day"):
        model.fetch_players(2026, "token", lambda *a, **k: _Response([_box_row(1, "H", True), _box_row(1, "A", False)]))


def test_the_script_runs_end_to_end_and_writes_only_its_report(tmp_path, monkeypatch):
    import scripts.ncaab_availability_backtest as script
    worlds = {year: _season(year, priced=True, seed=year, teams=24, days=40) for year in (2024, 2025, 2026)}

    def get(url, params, **_):
        games, lines, box = worlds[params["season"]]
        first, last = params["startDateRange"], params["endDateRange"] + "T99"
        ids = {row["id"] for row in games if first <= row["startDate"] <= last}
        source = {"games": games, "lines": lines, "games/players": box}[url.split(".com/")[1]]
        return _Response([row for row in source if row.get("id", row.get("gameId")) in ids])

    result = script.run("token", 2026, get)
    assert result["seasons"] == [2025, 2026] and result["tier_season"] == 2024
    coverage = result["report"]["coverage"]["2026"]
    assert coverage["finished_games"] == 480 == coverage["lined_with_both_box_scores"]
    json.dumps(result, default=str)
    text = model.markdown(result)
    assert "**Answer: " in text and "Research only" in text and "Not a betting record" in text


def test_it_is_research_only_started_by_hand_and_read_by_nothing_else():
    workflow = (ROOT / ".github/workflows/ncaab-availability-backtest.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in workflow and "schedule" not in workflow and "contents: read" in workflow
    secrets = {line.split("secrets.")[1].split()[0] for line in workflow.splitlines() if "secrets." in line}
    assert secrets == {"CBBD_API_KEY", "CFBD_API_KEY"}
    readers = sorted(str(path.relative_to(ROOT)) for folder in ("app", "app_core", "core", "scripts")
                     for path in (ROOT / folder).rglob("*.py") if "ncaab_availability" in path.read_text(encoding="utf-8")
                     and path.name != "ncaab_availability.py")
    assert readers == ["scripts/ncaab_availability_backtest.py"]
    assert "ncaab_availability" not in (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    source = (ROOT / "app_core/ncaab_availability.py").read_text(encoding="utf-8")
    assert "open(" not in source and "write" not in source.replace("write-up", "").replace("written", "")
