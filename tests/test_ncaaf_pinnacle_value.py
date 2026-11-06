"""NCAAF Pinnacle value picks from NCAAF capture records. SYNTHETIC records; no network."""
from datetime import datetime, timedelta, timezone
import json

from app_core import ncaaf_pinnacle_value as npv
from app_core import ncaaf_prospective as ncaaf

KICK = datetime(2026, 11, 7, 20, tzinfo=timezone.utc)


def q(book, market_type, point, price, at):
    return {"book": book, "market_type": market_type, "point": point, "price": price, "recorded_at": at.isoformat()}


def capture(at, us, reference):
    event = {"event_id": "g1", "cfbd_id": 77, "home": "Home U", "away": "Away St", "home_id": 1, "away_id": 2,
             "start": KICK.isoformat(), "models": {"ridge": {"candidates": us}}, "reference_quotes": reference}
    return {"kind": "capture", "created_at": at.isoformat(), "data": {"captured_at": at.isoformat(), "events": [event]}}


def test_pick_beats_pinnacle_at_a_us_book_on_spreads_and_totals():
    at = KICK - timedelta(hours=6)
    us = [q("draftkings", "spread_home", -3.5, 105, at), q("draftkings", "spread_away", 3.5, -125, at),
          q("fanduel", "total_over", 52.5, -110, at), q("fanduel", "total_under", 52.5, -110, at)]
    ref = [q("pinnacle", "spread_home", -3.5, -125, at), q("pinnacle", "spread_away", 3.5, 105, at),
           q("pinnacle", "total_over", 52.5, -105, at), q("pinnacle", "total_under", 52.5, -115, at)]
    chosen = npv.pick(capture(at, us, ref)["data"]["events"][0])
    assert chosen["market_type"] == "spread_home" and chosen["sportsbook"] == "draftkings"
    assert 0.53 < chosen["win_probability"] < 0.535 and chosen["ev"] >= npv.MIN_EV


def test_no_pick_without_pinnacle_and_pinnacle_is_never_the_pick():
    at = KICK - timedelta(hours=6)
    us = [q("draftkings", "spread_home", -3.5, 105, at), q("pinnacle", "spread_home", -3.5, 150, at)]
    assert npv.pick(capture(at, us, [])["data"]["events"][0]) is None


def test_board_and_replay_with_clv_against_the_last_pinnacle_capture():
    early, late = KICK - timedelta(hours=8), KICK - timedelta(hours=1)
    first = capture(early, [q("draftkings", "spread_home", -3.5, 105, early), q("draftkings", "spread_away", 3.5, -125, early)],
                    [q("pinnacle", "spread_home", -3.5, -125, early), q("pinnacle", "spread_away", 3.5, 105, early)])
    last = capture(late, [q("draftkings", "spread_home", -3.5, -140, late), q("draftkings", "spread_away", 3.5, 120, late)],
                   [q("pinnacle", "spread_home", -3.5, -140, late), q("pinnacle", "spread_away", 3.5, 120, late)])
    scores = {"kind": "scores", "created_at": (KICK + timedelta(hours=4)).isoformat(),
              "data": {"scores": [{"cfbd_id": 77, "home_id": 1, "away_id": 2, "home_score": 31, "away_score": 21}]}}
    records = [first, last, scores]
    board = npv.board(records, now=KICK - timedelta(hours=2))
    assert board["picks"] and board["picks"][0]["selection"] == "Home U"
    graded = npv.replay(records, now=KICK + timedelta(days=1))
    assert len(graded) == 1 and graded[0]["outcome"] == "WIN"            # 31-21 covers -3.5
    assert graded[0]["american_odds"] == 105 and graded[0]["clv"] > 0.05  # Pinnacle closed near -140
    assert npv.board(records, now=KICK + timedelta(hours=1))["blocker"] == "NO_PINNACLE_QUOTES_YET"


def test_settle_handles_each_side_and_push():
    assert npv.settle("spread_home", -3.5, 24, 21) == "LOSS"
    assert npv.settle("spread_away", 3.5, 24, 21) == "WIN"
    assert npv.settle("total_over", 45.0, 24, 21) == "PUSH"
    assert npv.settle("total_under", 44.5, 24, 21) == "LOSS"


def test_capture_keeps_fresh_pinnacle_quotes_as_reference_only(monkeypatch):
    from app_core import prediction_evidence
    now = KICK - timedelta(hours=3)
    quotes = [q("pinnacle", "spread_home", -3.5, -125, now), q("pinnacle", "spread_away", 3.5, 105, now),
              q("pinnacle", "spread_home", -2.5, -110, now - timedelta(hours=2)),        # stale: dropped
              q("draftkings", "spread_home", -3.5, 105, now)]                          # not reference
    monkeypatch.setattr(prediction_evidence, "provider_quotes", lambda event: json.dumps(quotes))
    kept = ncaaf.reference_quotes({}, now)
    assert [(r["book"], r["point"]) for r in kept] == [("pinnacle", 3.5), ("pinnacle", -3.5)]
