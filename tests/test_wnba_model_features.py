import pytest

import app_core.feature_processing as feature_processing


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "children": [
                {
                    "standings": {
                        "entries": [
                            {
                                "team": {"displayName": "New York Liberty"},
                                "stats": [
                                    {"name": "wins", "value": 20},
                                    {"name": "losses", "value": 10},
                                    {"name": "gamesPlayed", "value": 30},
                                    {"name": "avgPointsFor", "value": 86.2},
                                    {"name": "avgPointsAgainst", "value": 79.4},
                                ],
                            }
                        ]
                    }
                }
            ]
        }


def test_espn_wnba_standings_supply_resolved_scoring_features(monkeypatch):
    monkeypatch.setattr(feature_processing.requests, "get", lambda *args, **kwargs: _Response())
    clear = getattr(feature_processing.fetch_from_espn_wnba, "clear", None)
    if callable(clear):
        clear()

    rows = feature_processing.fetch_from_espn_wnba(2099)

    assert len(rows) == 1
    assert rows[0]["league_key"] == "WNBA"
    assert rows[0]["team_norm"] == "NEW YORK LIBERTY"
    assert rows[0]["win_pct"] == pytest.approx(2 / 3)
    assert rows[0]["points_per_game"] == pytest.approx(86.2)
    assert rows[0]["points_allowed_per_game"] == pytest.approx(79.4)

