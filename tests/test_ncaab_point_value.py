"""The measured value of a point of line (owner 2026-11-12). SYNTHETIC games; no network."""
import importlib.util
from pathlib import Path

import pytest

from app_core import ncaab_point_value as pv

ROOT = Path(__file__).resolve().parents[1]


def _games(differences, spread=-4.5, total=140.5):
    """One game per wanted home cover margin ``d`` (home margin + spread)."""
    games, lines = [], {}
    for index, d in enumerate(differences):
        margin = d - spread
        home = (total + margin) / 2
        games.append({"id": index, "home_points": home, "away_points": home - margin})
        lines[index] = {"spread": spread, "total": total, "provider": "consensus"}
    return games, lines


def test_the_value_of_a_point_is_the_share_of_games_inside_the_band_over_its_width():
    # 1,000 games: cover margins spread evenly from -49.75 to +49.75 in half-point steps (5 games each).
    values = [(-49.75 + 0.5 * (i % 200)) for i in range(1000)]
    value, se, n = pv.point_value(values, 1.0)
    assert n == 1000 and value == pytest.approx(0.01)            # 2% of games within a point, over 2 points
    assert se == pytest.approx((0.02 * 0.98 / 1000) ** 0.5 / 2)
    assert pv.point_value(values, 3.0)[0] == pytest.approx(0.01)   # a flat distribution: the same at any move
    assert pv.point_value(values[:100], 1.0) == (None, None, 100)   # too few games to say
    # The same number as the definition: (W(k) - W(-k)) / 2k.
    assert (pv.win_share(values, 1.0) - pv.win_share(values, -1.0)) / 2 == pytest.approx(value)
    assert pv.win_share([], 1.0) is None


def test_a_push_counts_as_half_and_the_band_edge_as_half():
    values = [0.0] * 100 + [1.0] * 100 + [5.0] * 100            # whole-number lines: pushes and edges
    assert pv.win_share(values, 0.0) == pytest.approx((50 + 100 + 100) / 300)
    assert pv.win_share(values, -1.0) == pytest.approx((0 + 50 + 100) / 300)
    value, _, _ = pv.point_value(values, 1.0)
    assert value == pytest.approx((100 + 50) / 300 / 2)          # the pushes, and half of the games on the edge
    assert (pv.win_share(values, 1.0) - pv.win_share(values, -1.0)) / 2 == pytest.approx(value)


def test_differences_follow_the_home_spread_and_the_total():
    games, lines = _games([2.0, -3.0])
    found = pv.differences(games, lines)
    assert [d for d, _ in found["SPREAD"]] == pytest.approx([2.0, -3.0]) and found["SPREAD"][0][1] == 4.5
    assert [d for d, _ in found["TOTAL"]] == pytest.approx([0.0, 0.0]) and found["TOTAL"][0][1] == 140.5
    lines[1] = {"spread": None, "total": 150.0, "provider": "consensus"}
    assert len(pv.differences(games, lines)["SPREAD"]) == 1       # a game without a spread is left out of it
    assert pv.differences(games, {}) == {"SPREAD": [], "TOTAL": []}


def test_the_report_groups_by_line_size_and_names_the_assumption():
    close, lines = _games([(-9.75 + 0.5 * (i % 40)) for i in range(400)], spread=-3.5)
    wide, more = _games([(-9.75 + 0.5 * (i % 40)) for i in range(400)], spread=-22.5)
    for index, game in enumerate(wide):
        game["id"] = 1000 + index
        lines[1000 + index] = more[index]
    result = {"seasons": [2025, 2026], "games": 800, "lined": 800, "providers": ["consensus"],
              "report": pv.measure(close + wide, lines)}
    rows = {row["group"]: row for row in result["report"]["SPREAD"]["rows"]}
    assert rows["all games"]["games"] == 800 and rows["under 5"]["games"] == 400 and rows["20 or more"]["games"] == 400
    assert rows["5 to 10"]["games"] == 0 and rows["5 to 10"]["by_move"]["1"]["value_per_point"] is None
    assert rows["under 5"]["by_move"]["1"]["value_per_point"] == pytest.approx(0.05)   # 10% within a point
    text = pv.markdown(result)
    assert "### Spread (home cover margin): assumed 3.6% a point" in text and "assumed 2.4% a point" in text
    assert "| under 5 | 400 |" in text and "5.00% (" in text and "not any book's price" in text
    assert pv.markdown({"seasons": [], "report": {}}).startswith("## NCAAB value of a point")


def test_the_script_reads_seasons_and_the_workflow_can_run_it(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("ncaab_point_value_script", ROOT / "scripts" / "ncaab_point_value.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []

    def fetch(endpoint, season, token, get):
        calls.append((endpoint, season))
        if endpoint == "games":
            return [{"id": f"{season}-{i}", "startDate": "2025-01-01T00:00:00Z", "homeTeam": "H", "awayTeam": "A",
                     "homePoints": 70 + i % 5, "awayPoints": 68} for i in range(3)]
        return [{"gameId": f"{season}-{i}", "lines": [{"provider": "consensus", "spread": -2.5, "overUnder": 139.5}]}
                for i in range(3)]
    monkeypatch.setattr(script.bt, "fetch", fetch)
    result = script.run("token", [2025, 2026], get=object())
    assert calls == [("games", 2025), ("lines", 2025), ("games", 2026), ("lines", 2026)]
    assert result["games"] == 6 and result["lined"] == 6 and result["providers"] == ["consensus"]
    monkeypatch.delenv("CBBD_API_KEY", raising=False)
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="MISSING_CREDENTIALS"):
        script.main(["--output", str(tmp_path / "out.json")])
    text = (ROOT / ".github/workflows/ncaab-line-backtest.yml").read_text(encoding="utf-8")
    assert "options: [backtest, point_value]" in text and "scripts/ncaab_point_value.py" in text
    assert "if: ${{ inputs.report != 'point_value' }}" in text      # the line screen still runs by default
