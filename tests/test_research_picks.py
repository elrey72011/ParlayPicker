"""Research picks board and replay grading. SYNTHETIC events from the research-model builder."""
import pytest

from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core import research_picks as rp
from tests.test_prospective_research_models import _at, _event


@pytest.fixture
def clock(monkeypatch):
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    return state


def test_board_picks_upcoming_games_and_replay_grades_finished_ones(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    assert rp.board(path, "NBA", "SPREAD", now=_at(0))["blocker"] == "NO_MODEL_YET"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    models.train_scope(path, "NBA", "SPREAD")
    for index in range(72, 76):
        _event(path, clock, "NBA", index)      # quotes 1h before start, then final score
    current = rp.board(path, "NBA", "SPREAD", now=_at(76, 1))
    assert current["model_id"] and current["blocker"] is None and current["training_games"] >= 60
    graded = rp.replay(path, "NBA", "SPREAD", now=_at(80))
    assert graded and all(g["outcome"] in ("WIN", "LOSS", "PUSH") for g in graded)
    assert all(0 <= g["win_probability"] <= 1 and g["american_odds"] == -110 for g in graded)
    summary = rp.record(graded)
    assert summary["picks"] == len(graded) and summary["wins"] + summary["losses"] + summary["pushes"] == len(graded)
    assert rp.replay(path, "NHL", "PUCK_LINE", now=_at(80)) == []   # NHL needs the official check


def test_pick_takes_the_best_ev_side_at_the_best_price():
    event = {"event_id": "e", "home_team": "H", "away_team": "A"}
    def q(book, sel, line, american):
        dec = 1 + (american / 100 if american > 0 else 100 / abs(american))
        return {"event_id": "e", "market_family": "SPREAD", "selection": sel, "line": line, "american_odds": american,
                "decimal_odds": dec, "sportsbook": book, "quote_id": book + sel, "quote_verified": True}
    offers = {}
    for quote in (q("a", "H", -3.5, -110), q("a", "A", 3.5, -110), q("b", "H", -3.5, -105), q("b", "A", 3.5, -115)):
        offers.setdefault(rp._home_line(event, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    fit = {"coefficients": [10.0], "means": [], "scales": [], "residual_sd": 10.0}   # home by 10
    pick = rp._pick(event, "SPREAD", offers, fit, [1.0])
    assert pick["selection"] == "H" and pick["sportsbook"] == "b" and pick["american_odds"] == -105
    assert pick["ev"] > 0 and pick["consensus_probability"] is None   # fewer than 3 books


def test_script_markdown_shows_board_blockers_and_record():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("research_picks_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "research_picks.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    result = {"boards": [
        {"sport": "NCAAB", "market_family": "SPREAD", "blocker": None, "training_games": 80, "picks": [
            {"scheduled_start": "2026-11-06T00:00:00+00:00", "away_team": "A", "home_team": "H", "selection": "H",
             "line": -3.5, "american_odds": -105, "sportsbook": "book", "win_probability": 0.56, "ev": 0.09,
             "consensus_probability": 0.51}]},
        {"sport": "NBA", "market_family": "SPREAD", "blocker": "NO_MODEL_YET", "training_games": None, "picks": []}],
        "records": {"NCAAB SPREAD": dict(rp.record([]), positive_ev=rp.record([]))}}
    text = script.markdown(result)
    assert "| A @ H | H | -3.5 | -105 | book | 56.0% | +9.0% | 51.0% |" in text
    assert "NO_MODEL_YET" in text and "| NCAAB SPREAD | 0 | 0-0-0 |" in text and "NO_BET" in text
