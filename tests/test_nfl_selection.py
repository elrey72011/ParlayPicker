from app.ui.sidebar_controls import FALLBACK_SPORTS, _resolve_sports_options
from app_core import odds_api
import core.streamlit_pipeline as sp


def _nfl_game(*, sport_key, game_id, home_team, away_team):
    return {
        "id": game_id,
        "matchup_id": f"{sport_key}:{away_team}:{home_team}:2026-08-15",
        "sport_key": sport_key,
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": "2026-08-15T23:00:00Z",
        "bookmakers": [],
    }


def test_nfl_is_a_default_selectable_sport():
    assert "NFL" in FALLBACK_SPORTS
    assert "NFL" in _resolve_sports_options()


def test_nfl_selection_fetches_preseason_and_regular_season_games(monkeypatch):
    calls = []
    games = {
        "americanfootball_nfl_preseason": [
            _nfl_game(
                sport_key="americanfootball_nfl_preseason",
                game_id="nfl-preseason-1",
                home_team="New England Patriots",
                away_team="Washington Commanders",
            )
        ],
        "americanfootball_nfl": [
            _nfl_game(
                sport_key="americanfootball_nfl",
                game_id="nfl-regular-1",
                home_team="Philadelphia Eagles",
                away_team="Dallas Cowboys",
            )
        ],
    }

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            calls.append((sport_key, date))
            return games[sport_key]

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")
    monkeypatch.setattr(odds_api, "filter_games_today_only", lambda rows: rows)

    frame = sp.fetch_live_odds_dataframe(sports=["NFL"])

    assert calls == [
        ("americanfootball_nfl_preseason", None),
        ("americanfootball_nfl", None),
    ]
    assert set(frame["game_id"]) == {"nfl-preseason-1", "nfl-regular-1"}
    assert frame["league"].eq("NFL").all()
