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


def test_opening_lines_get_their_own_fit():
    prior, calibration, scored = (_season(y, STRENGTHS, seed=y) for y in (2023, 2024, 2025))

    def lines(games):
        # Closing lines are efficient; opening lines are 4 points off: only the opening fit finds an edge.
        out = _lines_from_truth(games)
        for g in games:
            out[g["id"]]["spread_open"] = out[g["id"]]["spread"] + 4.0
        return out
    report = bt.run(prior, calibration, scored, lines(calibration), lines(scored))["SPREAD"]
    assert report["fits"]["opening"]["k"] > 0.5 > report["fits"]["closing"]["k"]
    assert report["lines"]["opening"]["Jan-Apr"]["by_threshold"]["0.55"]["hit_rate"] > 0.6


# --- 2026-11-17: readings that need no fitted shrink, the line-move check, and the field-name record ----------


def _game(game_id, month=1):
    return {"id": game_id, "start": datetime(2026, month, 10, 19, tzinfo=timezone.utc), "home": "H", "away": "A",
            "neutral": False, "home_points": 70, "away_points": 68}


def test_by_disagreement_grades_the_ratings_side_with_no_fit():
    g = _game(1)
    cases = [(3.0, 5.0, g), (3.0, -1.0, g), (-4.5, -2.0, g), (-6.0, 1.0, g), (1.0, 9.0, g), (2.0, 0.0, g)]
    out = bt.by_disagreement(cases)
    # 2 points or more: five games (the 1-point disagreement is left out), two won, two lost, one push.
    assert out["2"] == {"picks": 5, "wins": 2, "losses": 2, "hit_rate": 0.5,
                        "roi_per_pick": pytest.approx((2 * 100 / 110 - 2) / 5)}
    assert (out["4"]["picks"], out["4"]["wins"], out["4"]["losses"]) == (2, 1, 1)
    assert out["6"]["picks"] == 1 and out["6"]["hit_rate"] == 0.0 and out["6"]["roi_per_pick"] == -1.0
    assert bt.by_disagreement([])["2"] == {"picks": 0, "wins": 0, "losses": 0, "hit_rate": None, "roi_per_pick": None}


def test_select_pairs_takes_both_numbers_from_one_provider():
    rows = [
        # consensus has no opening number here: the pair comes from the next provider that has both.
        {"gameId": 1, "lines": [{"provider": "DraftKings", "spread": -5.0, "spreadOpen": -3.5, "overUnder": 141.5,
                                 "overUnderOpen": 139.5},
                                {"provider": "consensus", "spread": -4.5, "overUnder": 142.0}]},
        {"gameId": 2, "lines": [{"provider": "consensus", "spread": 2.0, "overUnder": 150.0}]},      # no opening number
        {"gameId": 3, "lines": [{"provider": "Bovada", "spread": 1.5, "spreadOpen": 2.5}]},          # a spread only
        {"lines": [{"provider": "consensus", "spread": 1.0, "spreadOpen": 1.0}]},                    # no game id
    ]
    assert bt.select_pairs(rows) == {1: {"SPREAD": (-3.5, -5.0), "TOTAL": (139.5, 141.5)}, 3: {"SPREAD": (2.5, 1.5)}}
    # select_lines mixes two books for game 1 (consensus close, DraftKings open): right for grading, wrong for a move.
    mixed = bt.select_lines(rows)[1]
    assert (mixed["spread_open"], mixed["spread"]) == (-3.5, -4.5)


def test_line_moves_follow_the_side_the_ratings_took_at_the_open():
    games = [_game(i) for i in range(1, 8)]
    predictions = {1: (6.0, 150.0), 2: (1.0, 140.0), 3: (6.0, 150.0), 4: (3.5, 145.0), 5: (6.0, 150.0), 6: (6.0, 150.0)}
    pairs = {1: {"SPREAD": (-3.5, -5.0), "TOTAL": (145.0, 147.5)},     # home and Over taken: both moved their way
             2: {"SPREAD": (-3.5, -5.0), "TOTAL": (145.0, 147.5)},     # away and Under taken: both moved against
             3: {"SPREAD": (-3.5, -3.5)},                              # unchanged, and no pair for the total
             4: {"SPREAD": (-3.5, -5.0), "TOTAL": (145.0, 147.5)},     # the ratings sit exactly on both lines: no side
             5: {},                                                    # no pair (game 6 has no entry at all)
             7: {"SPREAD": (-3.5, -5.0)}}                              # no prediction
    spread = bt.line_moves(games, predictions, pairs, "SPREAD")
    assert [(edge, moved, game["id"]) for edge, moved, game in spread] == [(2.5, 1.5, 1), (2.5, -1.5, 2), (2.5, 0.0, 3)]
    total = bt.line_moves(games, predictions, pairs, "TOTAL")
    assert [(edge, moved, game["id"]) for edge, moved, game in total] == [(5.0, 2.5, 1), (5.0, -2.5, 2)]


def test_move_summary_and_slope():
    import statistics
    g = _game(1)
    moves = [(2.5, 1.5, g), (2.5, -1.5, g), (5.0, 0.0, g), (7.0, 2.0, g), (0.5, 1.0, g)]
    out = bt.move_summary(moves)
    assert out["0"]["games"] == 5 and out["0"]["mean_points"] == pytest.approx(0.6)
    assert out["0"]["se"] == pytest.approx(statistics.stdev([1.5, -1.5, 0.0, 2.0, 1.0]) / 5 ** 0.5)
    assert (out["0"]["toward"], out["0"]["against"], out["0"]["unchanged"]) == (3, 1, 1)
    assert out["2"]["games"] == 4 and out["2"]["mean_points"] == pytest.approx(0.5)
    assert out["4"]["games"] == 2 and out["4"]["mean_points"] == pytest.approx(1.0)
    assert out["6"] == {"games": 1, "mean_points": 2.0, "se": None, "toward": 1, "against": 0, "unchanged": 0}
    assert bt.move_summary([])["0"] == {"games": 0, "mean_points": None, "se": None, "toward": 0, "against": 0,
                                        "unchanged": 0}
    # The slope: a close that gives back exactly a quarter of every disagreement reads 0.25 with no error.
    quarter = bt.move_slope([(edge, edge / 4, g) for edge in (1.0, 2.0, 3.0, 4.0, 8.0)])
    assert quarter["slope"] == pytest.approx(0.25) and quarter["se"] == pytest.approx(0.0, abs=1e-12)
    mixed = bt.move_slope(moves)
    assert mixed["games"] == 5 and mixed["slope"] == pytest.approx(14.5 / 86.75)
    assert mixed["t"] == pytest.approx(mixed["slope"] / mixed["se"])
    assert bt.move_slope(moves[:2]) == {"games": 2, "slope": None, "se": None, "t": None}


def test_an_opening_line_the_close_corrects_shows_in_both_readings_and_an_efficient_one_in_neither():
    prior, calibration, scored = (_season(y, STRENGTHS, seed=y) for y in (2023, 2024, 2025))

    def soft_open(games):
        out = _lines_from_truth(games)
        for g in games:
            out[g["id"]]["spread_open"] = out[g["id"]]["spread"] + 4.0      # the home side opens 4 points too cheap
        return out
    report = bt.run(prior, calibration, scored, soft_open(calibration), soft_open(scored))["SPREAD"]
    opening, closing = report["disagreement"]["opening"], report["disagreement"]["closing"]
    assert opening["Jan-Apr"]["2"]["picks"] > 200 and opening["Jan-Apr"]["2"]["hit_rate"] > 0.58
    assert opening["all"]["2"]["picks"] == opening["Nov-Dec"]["2"]["picks"] + opening["Jan-Apr"]["2"]["picks"]
    assert closing["Jan-Apr"]["2"]["picks"] < opening["Jan-Apr"]["2"]["picks"]      # the close is right: less to disagree with
    moves, slope = report["line_moves"], report["line_move_slope"]
    assert moves["all"]["0"]["games"] == len(scored) and moves["all"]["0"]["toward"] > 0.9 * len(scored)
    assert moves["Jan-Apr"]["2"]["mean_points"] > 3.5                                # of the 4 points the close took back
    assert slope["all"]["slope"] > 0.5 and slope["all"]["t"] > 10
    assert slope["Jan-Apr"]["slope"] > slope["Nov-Dec"]["slope"]                     # the ratings sharpen as games come in
    # An efficient market, the same at the open and the close: nothing moves, so nothing moves toward the ratings.
    efficient = bt.run(prior, calibration, scored, _lines_from_truth(calibration), _lines_from_truth(scored))["SPREAD"]
    assert efficient["line_moves"]["all"]["0"] == {"games": len(scored), "mean_points": 0.0, "se": 0.0, "toward": 0,
                                                   "against": 0, "unchanged": len(scored)}
    assert efficient["line_move_slope"]["all"]["slope"] == 0.0 and efficient["line_move_slope"]["all"]["t"] is None
    # Too few calibration games for any fit: both readings are still reported, because neither needs one.
    few = bt.run(prior, calibration[:100], scored, soft_open(calibration[:100]), soft_open(scored))
    assert few["SPREAD"]["blocker"] == "TOO_FEW_CALIBRATION_GAMES" and few["TOTAL"]["blocker"] == "TOO_FEW_CALIBRATION_GAMES"
    assert few["SPREAD"]["line_move_slope"]["all"]["slope"] > 0.5
    assert few["SPREAD"]["disagreement"]["opening"]["all"]["2"]["picks"] > 0
    text = bt.markdown({"prior_season": 2023, "calibration_season": 2024, "scored_season": 2025, "games": {},
                        "closing_lined": 1, "opening_lined": 1, "report": {"SPREAD": report, "TOTAL": few["TOTAL"]}})
    assert "### Where the ratings disagree with the line (no fitted shrink)" in text
    assert "| SPREAD | opening | Jan-Apr | 2+ |" in text and "| SPREAD | Jan-Apr | 2+ | " in text
    assert "### Did the closing line move toward the ratings?" in text
    assert "share of the disagreement the close gave back" in text
    assert "| TOTAL | | | | TOO_FEW_CALIBRATION_GAMES |" in text and "| TOTAL | opening | all | 2+ |" in text


def test_the_scored_seasons_pairs_come_from_one_provider_even_without_passing_them():
    games = [_game(1), _game(2)]
    lines = {1: {"spread": -5.0, "total": 147.5, "spread_open": -3.5, "total_open": 145.0, "provider": "x"},
             2: {"spread": -5.0, "total": None, "spread_open": None, "total_open": 145.0, "provider": "x"}}
    assert bt._pairs_from_lines(lines) == {1: {"SPREAD": (-3.5, -5.0), "TOTAL": (145.0, 147.5)}}
    assert bt.line_moves(games, {1: (6.0, 150.0), 2: (6.0, 150.0)}, bt._pairs_from_lines(lines), "SPREAD")[0][:2] == (2.5, 1.5)


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


def test_script_records_field_names_only_and_passes_one_provider_pairs(monkeypatch):
    import json
    script = importlib.import_module("scripts.ncaab_ratings_backtest")

    def fetch(endpoint, season, token, get):
        if endpoint == "games":
            return [{"id": season, "homeTeam": "Secret State", "pace": 68.2}]
        return [{"gameId": season, "homeTeam": "Secret State",
                 "lines": [{"provider": "consensus", "spread": -2.5, "spreadOpen": -1.5}]}]
    seen = []

    def run(*args):
        seen.append(args)
        return {"SPREAD": {"blocker": "TOO_FEW_CALIBRATION_GAMES"}}
    monkeypatch.setattr(script.data, "fetch", fetch)
    monkeypatch.setattr(script.bt, "run", run)
    result = script.run("token", 2026, get=object())
    assert result["schema"] == {"games": ["homeTeam", "id", "pace"], "lines": ["gameId", "homeTeam", "lines"],
                                "line": ["provider", "spread", "spreadOpen"]}
    assert "Secret" not in json.dumps(result["schema"]) and "68.2" not in json.dumps(result["schema"])
    assert len(seen) == 1 and seen[0][5] == {2026: {"SPREAD": (-1.5, -2.5)}}      # the scored season's pairs
    assert result["opening_lined"] == 1 and result["providers"] == ["consensus"]


def test_field_names_keeps_names_and_types_and_no_values():
    import json
    script = importlib.import_module("scripts.ncaab_ratings_backtest")
    row = {"team": "Secret State", "pace": 68.2,
           "teamStats": {"possessions": 70, "points": {"total": 71, "byPeriod": [30, 41]},
                         "fourFactors": {"effectiveFieldGoalPct": 51.2}}}
    names = script.field_names([row], depth=3)
    assert names == [{"pace": "float", "team": "str",
                      "teamStats": {"fourFactors": {"effectiveFieldGoalPct": "float"},
                                    "points": {"byPeriod": ["int"], "total": "int"}, "possessions": "int"}}]
    assert "Secret" not in json.dumps(names) and "71" not in json.dumps(names)
    assert script.field_names([]) == [] and script.field_names({"a": {"b": {"c": {"d": 1}}}}, depth=1) == {"a": {"b": "dict"}}


def test_probe_asks_for_one_day_of_team_box_scores_and_never_raises():
    script = importlib.import_module("scripts.ncaab_ratings_backtest")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs["params"], kwargs["headers"]))
        return _Response(200, [{"gameId": 9, "team": "Secret State", "teamStats": {"possessions": 70}}, {"gameId": 9}])
    probe = script.probe_team_stats("token", 2026, get=get)
    assert calls == [("https://api.collegebasketballdata.com/games/teams",
                      {"season": 2026, "startDateRange": "2026-01-15", "endDateRange": "2026-01-15"},
                      {"Authorization": "Bearer token"})]
    assert probe == {"endpoint": "games/teams", "day": "2026-01-15", "rows": 2,
                     "fields": [{"gameId": "int", "team": "str", "teamStats": {"possessions": "int"}}]}
    refused = script.probe_team_stats("token", 2026, get=lambda url, **kwargs: _Response(403, []))
    assert refused == {"endpoint": "games/teams", "day": "2026-01-15", "error": "CBBD games/teams HTTP 403"}

    def broken(url, **kwargs):
        raise RuntimeError("the message of an unexpected error is not kept")
    assert script.probe_team_stats("token", 2026, get=broken)["error"] == "RuntimeError"
    text = script.schema_markdown({"schema": {"games": ["id"], "lines": ["gameId"], "line": ["spread"]},
                                   "team_stats_probe": probe})
    assert "- games: id" in text and "games/teams, 2026-01-15, 2 rows" in text and "possessions" in text
    assert "Secret" not in text
    assert "HTTP 403" in script.schema_markdown({"schema": {}, "team_stats_probe": refused})


def test_main_adds_the_probe_only_when_asked(monkeypatch, tmp_path):
    import json
    script = importlib.import_module("scripts.ncaab_ratings_backtest")
    monkeypatch.setenv("CBBD_API_KEY", "token")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    seasons, probes = [], []

    def run(token, scored_season):
        seasons.append(scored_season)
        return {"prior_season": scored_season - 2, "calibration_season": scored_season - 1,
                "scored_season": scored_season, "schema": {"games": [], "lines": [], "line": []}, "games": {},
                "closing_lined": 0, "opening_lined": 0, "providers": [], "report": {}}

    def probe(token, season):
        probes.append(season)
        return {"endpoint": "games/teams", "day": f"{season}-01-15", "rows": 0, "fields": []}
    monkeypatch.setattr(script, "run", run)
    monkeypatch.setattr(script, "probe_team_stats", probe)
    assert script.main(["--output", str(tmp_path / "a.json")]) == 0
    assert seasons == [2025] and probes == []
    assert "team_stats_probe" not in json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert script.main(["--output", str(tmp_path / "b.json"), "--scored-season", "2026", "--probe"]) == 0
    assert seasons == [2025, 2026] and probes == [2026]
    assert json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))["team_stats_probe"]["day"] == "2026-01-15"


def test_the_workflow_runs_the_script_by_hand_only():
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/ncaab-ratings-backtest.yml").read_text(encoding="utf-8")
    assert "python -u scripts/ncaab_ratings_backtest.py" in text and "--probe" in text
    assert "workflow_dispatch:" in text
    assert not re.search(r"^\s*(schedule|push|pull_request|workflow_run):", text, flags=re.M)     # never on its own
    assert "default: '2026'" in text                      # 2025-26: the first season with opening lines all year
    assert "permissions:\n  contents: read\njobs:" in text and ": write" not in text      # reads the repository, no more
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"CBBD_API_KEY", "CFBD_API_KEY"}   # no Odds API, no Drive
