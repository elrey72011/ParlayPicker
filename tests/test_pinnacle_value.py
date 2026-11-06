"""Pinnacle value picks (D4 option A). SYNTHETIC quotes; no network."""
from datetime import datetime, timedelta, timezone

from app_core import pinnacle_value as pv

EVENT = {"event_id": "e", "sport": "NBA", "home_team": "H", "away_team": "A",
         "scheduled_start": "2026-11-06T23:00:00+00:00"}
T0 = datetime(2026, 11, 6, 18, tzinfo=timezone.utc)


def q(book, sel, line, american, minutes=0):
    dec = 1 + (american / 100 if american > 0 else 100 / abs(american))
    stamp = (T0 + timedelta(minutes=minutes)).isoformat()
    return {"event_id": "e", "market_family": "SPREAD", "selection": sel, "line": line, "american_odds": american,
            "decimal_odds": dec, "sportsbook": book, "quote_id": f"{book}{sel}{minutes}", "quote_verified": True,
            "quote_timestamp": stamp}


def pair(book, home, away, minutes=0, line=-3.5):
    return [q(book, "H", line, home, minutes), q(book, "A", -line, away, minutes)]


def test_pick_beats_pinnacle_fair_at_a_us_book_and_never_at_pinnacle():
    quotes = pair("pinnacle", -125, 105) + pair("a", 105, -125) + pair("b", -110, -110)
    by_line = pv._book_quotes(EVENT, quotes, T0 + timedelta(hours=1))
    chosen = pv.pick(EVENT, "SPREAD", by_line)
    assert chosen["selection"] == "H" and chosen["sportsbook"] == "a" and chosen["american_odds"] == 105
    assert 0.53 < chosen["win_probability"] < 0.535 and chosen["ev"] >= pv.MIN_EV


def test_no_pick_without_pinnacle_or_without_enough_value():
    assert pv.pick(EVENT, "SPREAD", pv._book_quotes(EVENT, pair("a", 105, -125), T0 + timedelta(hours=1))) is None
    flat = pair("pinnacle", -110, -110) + pair("a", -108, -112)
    assert pv.pick(EVENT, "SPREAD", pv._book_quotes(EVENT, flat, T0 + timedelta(hours=1))) is None


def test_book_quotes_use_only_the_newest_quote_seen_before_the_cutoff():
    quotes = pair("pinnacle", -110, -110, 0) + pair("pinnacle", -140, 120, 60)
    early = pv._book_quotes(EVENT, quotes, T0 + timedelta(minutes=30))[-3.5]["pinnacle"]["H"]
    late = pv._book_quotes(EVENT, quotes, T0 + timedelta(minutes=90))[-3.5]["pinnacle"]["H"]
    assert early["american_odds"] == -110 and late["american_odds"] == -140


def test_replay_decides_at_the_first_qualifying_moment_and_grades_clv_against_pinnacles_close(monkeypatch):
    quotes = (pair("pinnacle", -110, -110, 0) + pair("a", -110, -110, 0)          # no value yet
              + pair("pinnacle", -125, 105, 30) + pair("a", 105, -125, 30)       # value appears
              + pair("pinnacle", -140, 120, 240))                                 # Pinnacle closes further
    score = {"event_id": "e", "grading_version": 1, "home_score": 110, "away_score": 100}
    monkeypatch.setattr(pv.research, "_read_scope", lambda path, sport, market: ({"e": EVENT}, quotes, [score]))
    monkeypatch.setattr(pv.evidence, "verify_provider_score", lambda event, s: True)
    graded = pv.replay("unused", "NBA", "SPREAD", now=T0 + timedelta(days=1))
    assert len(graded) == 1
    g = graded[0]
    assert g["quote_timestamp"] == (T0 + timedelta(minutes=30)).isoformat() and g["outcome"] == "WIN"
    assert g["clv"] > 0.05                                    # +105 entry vs a close near 56% fair
    summary = pv.record(graded)
    assert summary["picks"] == 1 and summary["clv_measured"] == 1 and summary["beat_close_share"] == 1.0


def test_board_reports_blockers_outside_scope_and_before_pinnacle_data(monkeypatch):
    assert pv.board("unused", "NHL", "PUCK_LINE")["blocker"] == "PINNACLE_NOT_IN_SCOPE"
    monkeypatch.setattr(pv.research, "_read_scope", lambda path, sport, market: ({"e": EVENT}, pair("a", -110, -110), []))
    assert pv.board("unused", "NBA", "SPREAD", now=T0)["blocker"] == "NO_PINNACLE_QUOTES_YET"


def test_research_report_markdown_shows_the_pinnacle_value_board_and_clv():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rp_script_pv", Path(__file__).resolve().parents[1] / "scripts" / "research_picks.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    from app_core import research_picks as rp
    board = {"sport": "NBA", "market_family": "SPREAD", "method": pv.METHOD, "blocker": None, "training_games": None,
             "picks": [{"scheduled_start": "2026-11-06T23:00:00+00:00", "away_team": "A", "home_team": "H",
                        "selection": "H", "line": -3.5, "american_odds": 105, "sportsbook": "a",
                        "win_probability": 0.545, "ev": 0.117, "consensus_probability": 0.545}]}
    rec = dict(pv.record([]), positive_ev=rp.record([]))
    text = script.markdown({"boards": [board], "records": {f"NBA SPREAD ({pv.METHOD})": rec}})
    assert "### NBA SPREAD (Pinnacle value)" in text and "beat Pinnacle's no-vig line" in text
    assert "### Pinnacle value: closing-line value" in text
