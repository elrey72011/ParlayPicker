"""NCAAB market-anchored screen on archived lines. SYNTHETIC games only."""
import importlib.util
from pathlib import Path
import random

from app_core import ncaab_line_backtest as bt


def _season(seed, year, n_teams=24, rounds=14, edge=0.0):
    """Synthetic season; with ``edge`` the line under-rates team strength, so form predicts covers."""
    rng = random.Random(seed)
    strength = {f"T{i}": rng.gauss(0, 8) for i in range(n_teams)}
    games, lines, gid = [], [], 0
    for r in range(rounds):
        teams = list(strength)
        rng.shuffle(teams)
        for h, a in zip(teams[::2], teams[1::2]):
            gid += 1
            true_margin = strength[h] - strength[a] + 3
            margin = round(true_margin + rng.gauss(0, 10))
            total = round(140 + rng.gauss(0, 15))
            hp = (total + margin) // 2
            games.append({"id": year * 10000 + gid, "startDate": f"{year - 1}-11-{1 + r:02d}T23:00:00Z",
                          "homeTeam": h, "awayTeam": a, "homePoints": hp, "awayPoints": total - hp,
                          "neutralSite": False})
            spread = -round((1 - edge) * true_margin * 2) / 2 + 0.5
            lines.append({"gameId": year * 10000 + gid, "lines": [
                {"provider": "Bovada", "spread": spread, "overUnder": 140.5},
                {"provider": "consensus", "spread": spread, "overUnder": None}]})
    return games, lines


def test_parsers_take_the_preferred_provider_and_skip_unfinished_games():
    games = bt.parse_games([{"id": 1, "startDate": "2025-01-01T00:00:00Z", "homeTeam": "A", "awayTeam": "B",
                             "homePoints": 70, "awayPoints": 60},
                            {"id": 2, "startDate": "2025-01-02T00:00:00Z", "homeTeam": "A", "awayTeam": "B",
                             "homePoints": None, "awayPoints": None}])
    assert [g["id"] for g in games] == [1]
    lines = bt.select_lines([{"gameId": 1, "lines": [{"provider": "Bovada", "spread": -4, "overUnder": 141},
                                                    {"provider": "consensus", "spread": -3.5, "overUnder": None}]}])
    assert lines[1] == {"spread": -3.5, "total": 141.0, "provider": "consensus"}


def test_features_use_only_earlier_games():
    games, _ = _season(1, 2024, rounds=8)
    parsed = bt.parse_games(games)
    feats = bt.features(parsed)
    first = parsed[0]
    assert first["id"] not in feats                      # no prior games yet
    assert feats and all(len(f) == len(bt.FEATURES) for f in feats.values())


def test_no_edge_market_gives_no_reliable_edge_and_an_edge_market_is_found():
    flat = bt.run(*[bt.parse_games(_season(s, y)[0]) for s, y in ((2, 2024), (3, 2025))],
                  bt.select_lines(_season(2, 2024)[1]), bt.select_lines(_season(3, 2025)[1]))
    assert flat["SPREAD"]["games"] > 0 and flat["SPREAD"]["mean_abs_centre"] < 3
    tr, tl = _season(4, 2024, rounds=20, edge=0.5)
    te, el = _season(5, 2025, rounds=20, edge=0.5)
    found = bt.run(bt.parse_games(tr), bt.parse_games(te), bt.select_lines(tl), bt.select_lines(el))
    assert found["SPREAD"]["by_threshold"]["0.55"]["hit_rate"] > 0.55


def test_script_runs_end_to_end_with_a_fake_client(tmp_path):
    spec = importlib.util.spec_from_file_location("ncaab_bt_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "ncaab_line_backtest.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    data = {2024: _season(6, 2024), 2025: _season(7, 2025)}

    class Response:
        def __init__(self, value):
            self.status_code, self._value = 200, value

        def json(self):
            return self._value

    def get(url, params, **kwargs):
        games, lines = data[params["season"]]
        return Response(games if url.endswith("/games") else lines)

    result = script.run("token", 2024, 2025, get=get)
    assert result["test_lined"] > 0 and set(result["report"]) == {"SPREAD", "TOTAL"}
    text = bt.markdown(result)
    assert "NCAAB market-anchored model" in text and "| SPREAD |" in text
