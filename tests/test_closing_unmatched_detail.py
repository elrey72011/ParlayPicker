"""Why a saved pick found no single live closing quote (diagnostic only). SYNTHETIC rows."""
import json

from app_core import activation_closing as ac

PICK = {"matchup_id": "m1", "provider_namespace": "odds_api", "provider_event_id": "e1",
        "quote_bookmaker": "DraftKings", "market_type": "spread_home"}


def _row(*quotes, matchup="m1"):
    return {"matchup_id": matchup, "provider_quotes": json.dumps(list(quotes))}


def _q(book="DraftKings", market="spread_home", event="e1", ns="odds_api"):
    return {"provider_namespace": ns, "provider_event_id": event, "book": book, "market_type": market}


def test_each_reason_names_the_first_failed_check():
    why = ac._why_unmatched
    assert why(dict(PICK, provider_event_id=None), [_row(_q())], 0) == "saved_pick_has_no_provider_identity"
    assert why(PICK, [_row(_q(), matchup="other")], 0) == "game_not_in_live_odds"
    assert why(PICK, [_row()], 0) == "live_game_has_no_provider_quotes"
    assert why(PICK, [_row(_q(event="e9"))], 0) == "live_quotes_have_another_provider_event"
    assert why(PICK, [_row(_q(book="FanDuel"))], 0) == "book_not_quoting_this_game"
    assert why(PICK, [_row(_q(market="total_over"))], 0) == "book_not_quoting_this_market"
    assert why(PICK, [_row(_q(), _q())], 2) == "several_quotes_same_book_and_market"


def test_capture_live_reports_the_detail_and_keeps_the_existing_reason(tmp_path, monkeypatch):
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 6, 23, 0, tzinfo=timezone.utc)
    audit = pd.DataFrame([dict(PICK, game_start_utc=(now + timedelta(minutes=20)).isoformat(), sport="NCAAF")])
    monkeypatch.setattr("app_core.prediction_evidence.load_snapshots", lambda db: [(None, audit, None)])
    live = pd.DataFrame([_row(_q(book="FanDuel"))])
    detail = {}
    out = ac.capture_live(tmp_path / "e.sqlite3", fetch=lambda sports: live, now=now, unmatched=detail)
    assert out == {"verified": 0, "unavailable": 1, "reasons": {"missing_or_ambiguous_exact_quote": 1}}  # unchanged
    assert detail == {"book_not_quoting_this_game": 1}


def test_the_slate_loop_prints_the_detail(tmp_path, capsys):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from closing_capture_loop import run

    def capture(db, unmatched=None):
        unmatched["game_not_in_live_odds"] = 2
        return {"verified": 0, "unavailable": 2, "reasons": {"missing_or_ambiguous_exact_quote": 2}}
    ticks = iter([0, 0])
    run(tmp_path / "e.sqlite3", 1, 5, capture=capture, restore=lambda db: None, backup=lambda db: None,
        clock=lambda: next(ticks), sleep=lambda s: None)
    assert '"unmatched_detail": {"game_not_in_live_odds": 2}' in capsys.readouterr().out
