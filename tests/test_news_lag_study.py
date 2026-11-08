"""News lag study: news selection, price views, the credit caps and one run per label. SYNTHETIC."""
from datetime import datetime, timezone

import pytest

from app_core import news_lag_study as nls

ROWS = [{"game_date": "11/07/2026", "game_time": "07:30 (ET)", "matchup": "BKN@ATL", "team": "Atlanta Hawks",
         "player": "A", "status": "Out"}]


def _change(player, previous, status, material=True):
    return {"game_date": "11/07/2026", "matchup": "BKN@ATL", "team": "Atlanta Hawks", "player": player,
            "previous": previous, "status": status, "material": material}


def _manifest(*entries):
    return {"reports": [{"slot": slot, "rows": ROWS, "changes": changes} for slot, changes in entries]}


def test_news_is_pregame_flips_first_then_late_first_listings():
    found = nls.candidates([_manifest(
        ("2026-11-07T13:00:00-05:00", [_change("A", None, "Out")]),                 # first listing 6.5 h out: no
        ("2026-11-07T17:30:00-05:00", [_change("B", None, "Out")]),                 # first listing 2 h out: yes
        ("2026-11-07T18:30:00-05:00", [_change("C", "Questionable", "Out")]),       # flip: yes, first
        ("2026-11-07T18:45:00-05:00", [_change("D", "Probable", "Questionable", material=False)]),
        ("2026-11-07T19:25:00-05:00", [_change("E", "Questionable", "Out")]))])     # 5 min before tip: no
    assert [(g["slot"][11:16], g["flip"]) for g in found] == [("18:30", True), ("17:30", False)]
    assert found[0]["approx_start"] == "2026-11-07T19:30:00-05:00"


def test_event_match_and_price_view_at_pinnacles_line():
    events = [{"id": "x", "home_team": "Atlanta Hawks", "away_team": "Brooklyn Nets"},
              {"id": "y", "home_team": "Dallas Mavericks", "away_team": "Brooklyn Nets"}]
    event = nls.match_event(events, "BKN@ATL")
    assert event["id"] == "x" and nls.match_event(events, "BOS@MIL") is None

    def book(key, point, home, away):
        return {"key": key, "markets": [{"key": "spreads", "outcomes": [
            {"name": "Atlanta Hawks", "point": point, "price": home},
            {"name": "Brooklyn Nets", "point": -point, "price": away}]}]}
    response = {"timestamp": "t", "data": {"bookmakers": [
        book("pinnacle", -4.5, -130, 110), book("draftkings", -4.5, -110, -110), book("fanduel", -3.5, -110, -110)]}}
    view = nls.snapshot_view(event, response)
    assert view["pinnacle_line"] == -4.5 and view["pinnacle_home"] == pytest.approx(0.5427, abs=1e-4)
    assert view["books"]["draftkings"]["ev"] == pytest.approx(0.036, abs=1e-3)     # lags Pinnacle's move
    assert view["books"]["fanduel"]["ev"] is None                                  # another line: not compared
    assert view["best_ev"] == view["books"]["draftkings"]["ev"]


def _odds(pin_home, pin_away, dk_home):
    return {"timestamp": "t", "data": {"bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [
            {"name": "Atlanta Hawks", "point": -4.5, "price": pin_home},
            {"name": "Brooklyn Nets", "point": 4.5, "price": pin_away}]}]},
        {"key": "draftkings", "markets": [{"key": "spreads", "outcomes": [
            {"name": "Atlanta Hawks", "point": -4.5, "price": dk_home},
            {"name": "Brooklyn Nets", "point": 4.5, "price": -110}]}]}]}}


def _get_factory(remaining=15000, cost=10):
    calls = []

    def get(url, params):
        calls.append((url, params))
        headers = {"x-requests-last": "1" if url == nls.EVENTS_URL else str(cost),
                   "x-requests-remaining": str(remaining - 10 * len(calls))}
        if url == nls.EVENTS_URL:
            return 200, headers, {"data": [{"id": "x", "home_team": "Atlanta Hawks", "away_team": "Brooklyn Nets",
                                            "commence_time": "2026-11-08T00:30:00Z"}]}
        before = params["date"] < "2026-11-07T23:30:00Z"
        return 200, headers, _odds(-110, -110, -110) if before else _odds(-130, 110, -110)
    return get, calls


def _groups(n=3):
    return [{"slot": "2026-11-07T18:30:00-05:00", "game_date": "11/07/2026", "matchup": "BKN@ATL",
             "approx_start": "2026-11-07T19:30:00-05:00", "flip": True, "changes": []} for _ in range(n)]


def test_run_measures_the_lag_and_stays_inside_the_credit_cap():
    get, calls = _get_factory()
    result = nls.run(_groups(3), get, api_key="k", max_credits=90, min_remaining=4000)
    # One group costs 1 + 4 x 10 = 41: two fit in 90, the third would pass the cap.
    assert result["stopped"] == "BUDGET_CAP" and len(result["groups"]) == 2 and result["spent"] == 82
    assert all(p["apiKey"] == "k" for _, p in calls) and len(calls) == 10
    group = result["groups"][0]
    assert [s["offset"] for s in group["snapshots"]] == [-30, 0, 10, 30]
    assert group["analysis"]["best_ev_before"] == pytest.approx(-0.045, abs=1e-3)
    assert group["analysis"]["best_ev_after"] == pytest.approx(0.036, abs=1e-3)
    assert group["analysis"]["pinnacle_prob_move"] > 0.02
    assert result["summary"]["pinnacle_moved"] == 2 and result["summary"]["after_news_2pct"]["at_least"] == 2
    assert result["summary"]["control_before_1pct"]["at_least"] == 0


def test_run_stops_at_the_monthly_floor():
    get, _ = _get_factory(remaining=4030)
    result = nls.run(_groups(3), get, api_key="k", max_credits=5000, min_remaining=4000)
    assert result["stopped"] == "MONTHLY_FLOOR" and result["spent"] < 50


class Client:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": self.objects[Key]}

    def put_object(self, *, Key, Body, IfNoneMatch):
        assert Key not in self.objects
        self.objects[Key] = Body


def test_a_label_runs_once_so_the_budget_cannot_be_spent_twice():
    client, now = Client(), datetime(2026, 11, 8, tzinfo=timezone.utc)
    assert nls.claim(client, "v1", max_credits=5000, now=now) is True
    assert nls.claim(client, "v1", max_credits=5000, now=now) is False
    assert nls.claim(client, "v2", max_credits=5000, now=now) is True


def test_script_refuses_more_than_the_owner_budget():
    import importlib
    script = importlib.import_module("scripts.news_lag_study")
    with pytest.raises(SystemExit, match="owner's budget"):
        script.main(["--start", "2026-10-21", "--end", "2026-11-07", "--max-credits", "6000", "--output", "x.json"])
    text = script.markdown({"label": "v1", "spent": 4980, "max_credits": 5000, "stopped": "BUDGET_CAP",
                            "missing_days": [], "summary": {"news_studied": 121, "pinnacle_moved": 40,
                                                            "after_news_1pct": {"at_least": 9, "with_price": 38}}})
    assert "credits spent 4980 of 5000" in text and "Pinnacle moved: a stake book >= 1% better: 9 of 38" in text


def test_an_unknown_bookmaker_key_is_dropped_once_and_the_snapshot_retried():
    get, calls = _get_factory()

    def strict(url, params):
        if url != nls.EVENTS_URL and "novig" in params["bookmakers"]:
            return 422, {"x-requests-remaining": "15000"}, {"message": "invalid bookmakers"}
        return get(url, params)
    result = nls.run(_groups(1), strict, api_key="k", max_credits=5000, min_remaining=4000)
    assert result["books"] == ["pinnacle", "draftkings", "fanduel", "betmgm"]
    assert all(s["view"] for s in result["groups"][0]["snapshots"]) and result["spent"] == 41
