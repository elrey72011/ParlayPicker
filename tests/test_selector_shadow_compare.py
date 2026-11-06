"""EV selector vs the current research rule on the same replayed games (D7). SYNTHETIC data."""
from app_core import prospective_research_models as models
from app_core import research_picks as rp
from app_core import selector_shadow_compare as cmp
from tests.test_prospective_research_models import _at, _event
from tests.test_research_picks import clock  # noqa: F401  (fixture)


def _q(book, sel, line, american, stamp="2026-01-01T00:00:00+00:00"):
    dec = 1 + (american / 100 if american > 0 else 100 / abs(american))
    return {"event_id": "e", "market_family": "SPREAD", "selection": sel, "line": line, "american_odds": american,
            "decimal_odds": dec, "sportsbook": book, "quote_id": book + sel + stamp, "quote_verified": True,
            "quote_timestamp": stamp}


EVENT = {"event_id": "e", "home_team": "H", "away_team": "A", "scheduled_start": "2026-01-02T00:00:00+00:00"}


def _offers(quotes):
    out = {}
    for quote in quotes:
        out.setdefault(rp._home_line(EVENT, quote), {}).setdefault(quote["sportsbook"], {})[quote["selection"]] = quote
    return out


def test_ev_selector_caps_the_model_and_needs_a_two_point_edge():
    offers = _offers([_q("a", "H", -3.5, -110), _q("a", "A", 3.5, -110)])
    strong = cmp.ev_selector_pick(EVENT, "SPREAD", offers, 10.0, 10.0)      # model ~74% on H -3.5
    assert strong["selection"] == "H" and abs(strong["market_probability"] - 0.5) < 1e-9
    assert abs(strong["anchored_probability"] - strong["market_probability"]) <= 0.05 + 1e-9
    assert strong["bet"] and strong["edge"] >= 0.02
    weak = cmp.ev_selector_pick(EVENT, "SPREAD", offers, 4.0, 10.0)        # model barely above the line
    assert not weak["bet"]                                                  # edge < 2 pts or EV <= 0 at -110


def test_ev_selector_needs_a_paired_price():
    assert cmp.ev_selector_pick(EVENT, "SPREAD", _offers([_q("a", "H", -3.5, -110)]), 10.0, 10.0) is None


def test_close_fair_uses_each_books_last_pregame_quote():
    quotes = [_q("a", "H", -3.5, -110), _q("a", "A", 3.5, -110),
              _q("a", "H", -3.5, -130, "2026-01-01T23:00:00+00:00"), _q("a", "A", 3.5, 110, "2026-01-01T23:00:00+00:00"),
              _q("a", "H", -3.5, -200, "2026-01-02T01:00:00+00:00")]                        # after start: ignored
    home = cmp.close_fair(EVENT, quotes, "SPREAD", "H", -3.5)
    away = cmp.close_fair(EVENT, quotes, "SPREAD", "A", 3.5)
    assert 0.54 < home < 0.545 and abs(home + away - 1) < 1e-9


def test_compare_grades_both_rules_on_the_same_games(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    models.train_scope(path, "NBA", "SPREAD")
    for index in range(72, 76):
        _event(path, clock, "NBA", index)
    games = cmp.compare(path, "NBA", "SPREAD", now=_at(80))
    assert games and len(games) == len(rp.replay(path, "NBA", "SPREAD", now=_at(80), min_ev=float("-inf")))
    for game in games:
        for rule in cmp.RULES:
            if game[rule]:
                assert game[rule]["outcome"] in ("WIN", "LOSS", "PUSH")
    summary = cmp.summarize(games)
    assert summary["games"] == len(games)
    for rule in cmp.RULES:
        assert summary[rule]["picks"] + summary[rule]["no_bet"] == len(games)
    assert summary["same_side"] <= summary["both_bet"]


def test_script_markdown_shows_the_shadow_table():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("research_picks_script_shadow",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "research_picks.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    summary = cmp.summarize([
        {"current": {"selection": "H", "line": -3.5, "decimal_odds": 1.91, "ev": 0.05, "outcome": "WIN", "clv": 0.02},
         "ev_selector": {"selection": "H", "line": -3.5, "decimal_odds": 1.91, "ev": 0.03, "outcome": "WIN", "clv": 0.02}},
        {"current": {"selection": "A", "line": 3.5, "decimal_odds": 1.91, "ev": 0.01, "outcome": "LOSS", "clv": -0.01},
         "ev_selector": None}])
    text = script.markdown({"boards": [], "records": {}, "selector_comparison": {"NBA SPREAD (score)": summary}})
    assert "| NBA SPREAD (score) | 2 | current | 2 | 1-1-0 |" in text
    assert "| NBA SPREAD (score) | 2 | ev_selector | 1 | 1-0-0 |" in text and "| 1/1 |" in text
