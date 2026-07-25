import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import odds_api
import core.streamlit_pipeline as sp


def _sample_game():
    return {
        "id": "g1",
        "matchup_id": "basketball_nba:boston celtics:miami heat:2026-03-14",
        "sport_key": "basketball_nba",
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

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")

    # If old behavior is still present, this would be called and fail the test.
    monkeypatch.setattr(odds_api, "filter_games_today_only", lambda games: (_ for _ in ()).throw(AssertionError("today filter should not run")))

    df = sp.fetch_live_odds_dataframe(sports=["NBA"], date="2026-03-14T16:00:00Z")

    assert not df.empty
    assert calls == [("basketball_nba", "2026-03-14T16:00:00Z")]
    assert "novig_home_price" in df.columns


def test_fetch_live_odds_dataframe_accepts_novig_key_variants(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            return [
                {
                    "id": "g2",
                    "matchup_id": "basketball_ncaab:southern:prairie view am:2026-03-14",
                    "sport_key": "basketball_ncaab",
                    "home_team": "Southern",
                    "away_team": "Prairie View A&M",
                    "commence_time": "2026-03-14T23:00:00Z",
                    "bookmakers": [
                        {
                            "key": "novig_us",
                            "markets": [
                                {
                                    "key": "totals",
                                    "outcomes": [
                                        {"name": "Over", "point": 145.5, "price": -114},
                                        {"name": "Under", "point": 145.5, "price": 101},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")

    df = sp.fetch_live_odds_dataframe(sports=["NCAAB"], date="2026-03-14T16:00:00Z")

    assert not df.empty
    assert float(df.iloc[0]["novig_under_point"]) == 145.5
    assert float(df.iloc[0]["novig_under_price"]) == 101


def test_ncaab_postseason_cutoff_uses_season_ending_year():
    games = pd.DataFrame(
        {
            "league": ["NCAAB", "NCAAB", "NCAAB", "NCAAB", "NBA"],
            "game_date": [
                "2026-11-15T23:00:00Z",
                "2027-03-16T23:00:00Z",
                "2027-03-17T23:00:00Z",
                "2028-03-18T23:00:00Z",
                "2028-03-18T23:00:00Z",
            ],
        }
    )
    assert sp.is_postseason_ncaab(games).tolist() == [False, False, True, True, False]
