import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.streamlit_pipeline as sp


def _sample_game():
    return {
        "id": "g1",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "commence_time": "2026-03-14T23:00:00Z",
        "bookmakers": [
            {
                "key": "novig",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Boston Celtics", "point": -3.5, "price": -108},
                            {"name": "Miami Heat", "point": 3.5, "price": -102},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 220.5, "price": -105},
                            {"name": "Under", "point": 220.5, "price": -115},
                        ],
                    },
                ],
            }
        ],
    }


def test_fetch_live_odds_dataframe_uses_passed_date_and_skips_today_filter(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            calls.append((sport_key, date))
            return [_sample_game()]

    monkeypatch.setattr(sp, "ODDS_API_AVAILABLE", True)
    monkeypatch.setattr(sp, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")

    # If old behavior is still present, this would be called and fail the test.
    monkeypatch.setattr(sp, "filter_games_today_only", lambda games: (_ for _ in ()).throw(AssertionError("today filter should not run")))

    df = sp.fetch_live_odds_dataframe(sports=["NBA"], date="2026-03-14T16:00:00Z")

    assert not df.empty
    assert calls == [("basketball_nba", "2026-03-14T16:00:00Z")]
    assert "novig_home_price" in df.columns
