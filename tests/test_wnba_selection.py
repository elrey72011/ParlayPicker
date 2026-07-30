import pandas as pd
import pytest

from app.ui.sidebar_controls import FALLBACK_SPORTS, _resolve_sports_options
from app_core import odds_api
from collect_historical_data import ODDS_API_SPORTS
from core.team_mapper import normalize_team_name
import core.streamlit_pipeline as sp


def _wnba_live_game():
    return {
        "id": "wnba-1",
        "matchup_id": "basketball_wnba:indiana fever:las vegas aces:2026-07-29",
        "sport_key": "basketball_wnba",
        "home_team": "Las Vegas Aces",
        "away_team": "Indiana Fever",
        "commence_time": "2026-07-29T23:00:00Z",
        "bookmakers": [
            {
                "key": "novig",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Las Vegas Aces", "point": -4.5, "price": -108},
                            {"name": "Indiana Fever", "point": 4.5, "price": -102},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 177.5, "price": -105},
                            {"name": "Under", "point": 177.5, "price": -115},
                        ],
                    },
                ],
            }
        ],
    }


def test_wnba_is_a_default_selectable_sport():
    assert "WNBA" in FALLBACK_SPORTS
    assert "WNBA" in _resolve_sports_options()


def test_wnba_uses_official_odds_api_sport_key(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            calls.append((sport_key, date))
            return [_wnba_live_game()]

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")

    frame = sp.fetch_live_odds_dataframe(
        sports=["WNBA"], date="2026-07-29T16:00:00Z"
    )

    assert calls == [("basketball_wnba", "2026-07-29T16:00:00Z")]
    assert not frame.empty
    assert frame.iloc[0]["league"] == "WNBA"
    assert float(frame.iloc[0]["novig_home_point"]) == -4.5
    assert float(frame.iloc[0]["novig_over_point"]) == 177.5
    assert ODDS_API_SPORTS["WNBA"] == "basketball_wnba"


def test_wnba_full_names_normalize_to_theover_city_names():
    assert normalize_team_name("Las Vegas Aces") == normalize_team_name("Las Vegas")
    assert normalize_team_name("Indiana Fever") == normalize_team_name("Indiana")
    assert normalize_team_name("Portland Fire") == normalize_team_name("Portland")
    assert normalize_team_name("Toronto Tempo") == normalize_team_name("Toronto")


def test_wnba_connecticut_sun_does_not_inherit_college_uconn_alias():
    raw = pd.DataFrame(
        [
            {
                "League": "WNBA",
                "HomeTeam": "Chicago Sky",
                "AwayTeam": "Connecticut Sun",
                "PickTeam": "Chicago Sky",
                "Line": -4.5,
                "WinProbability": 0.58,
                "Market": "Spread",
            }
        ]
    )

    normalized = sp._normalize_upload(raw)

    assert normalized.iloc[0]["home_team"] == "Chicago"
    assert normalized.iloc[0]["away_team"] == "Connecticut"


def test_best_pick_export_restores_connecticut_after_generic_uconn_alias(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame(
        [
            {
                "game_id": "wnba-connecticut-export",
                "league": "WNBA",
                "home_team": "Chicago",
                # Reproduce the post-ingestion alias observed in the live export.
                "away_team": "UConn",
                "game_date": pd.Timestamp("2026-07-30", tz="UTC"),
                "market_type": "spread_away",
                "spread_line": 4.5,
                "live_spread_line": 4.5,
                "total_line": pd.NA,
                "live_total_line": pd.NA,
                "calibrated_probability": 0.62,
                "model_probability": 0.62,
                "ml_probability": pd.NA,
                "expected_value": 0.08,
                "edge": 0.06,
                "market_probability": 0.50,
                "kalshi_probability": 0.55,
                "consensus_agreement": "Agrees",
                "odds_american": -110,
                "odds_source": "test",
                "line_source": "live",
                "market_line_source": "live",
                "line_consistency_flag": True,
                "line_event_identity_match_flag": True,
                "is_live_data": True,
                "used_stale_features": False,
            }
        ]
    )
    diagnostics = {}
    best = sp.build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert len(best) == 1
    winner = best.iloc[0]
    assert winner["Home"] == "Chicago"
    assert winner["Away"] == "Connecticut"
    assert winner["best_pick"] == "Connecticut +4.5"
    assert "uconn" not in str(winner["matchup_id"]).lower()
    assert "uconn" not in str(winner.get("canonical_pick_key", "")).lower()

    audit = diagnostics["candidate_audit_df"]
    assert audit["home_team"].eq("Chicago").all()
    assert audit["away_team"].eq("Connecticut").all()
    assert audit["best_pick"].eq("Connecticut +4.5").all()
    assert not audit["matchup_id"].astype(str).str.contains("uconn", case=False).any()


def test_theover_wnba_pick_code_binds_line_and_probability_to_home_team():
    raw = pd.DataFrame(
        [
            {
                "League": "WNBA",
                "HomeTeam": "Las Vegas",
                "AwayTeam": "Indiana",
                "HomeKalshi": "LV",
                "AwayKalshi": "IND",
                "Pick": "LV",
                "Line": -4.5,
                "WinProbability": 0.62,
                "Market": "Spread",
            }
        ]
    )

    normalized = sp._normalize_upload(raw)
    home, away = sp._build_spread_rows(normalized)

    assert home.iloc[0]["league"] == "WNBA"
    assert float(home.iloc[0]["spread_line"]) == -4.5
    assert float(home.iloc[0]["theover_probability"]) == 0.62
    assert float(away.iloc[0]["spread_line"]) == 4.5
    assert float(away.iloc[0]["theover_probability"]) == 0.38



def test_theover_full_pick_team_survives_team_name_normalization():
    raw = pd.DataFrame(
        [
            {
                "League": "NBA",
                "HomeTeam": "Boston Celtics",
                "AwayTeam": "Miami Heat",
                "PickTeam": "Boston Celtics",
                "Line": -3.5,
                "WinProbability": 0.57,
                "Market": "Spread",
            }
        ]
    )

    normalized = sp._normalize_upload(raw)
    home, away = sp._build_spread_rows(normalized)

    assert float(home.iloc[0]["spread_line"]) == -3.5
    assert float(home.iloc[0]["theover_probability"]) == 0.57
    assert float(away.iloc[0]["spread_line"]) == 3.5
    assert float(away.iloc[0]["theover_probability"]) == pytest.approx(0.43)

def _line_candidate(market_type, **overrides):
    row = {
        "league": "WNBA",
        "home_team": "Las Vegas",
        "away_team": "Indiana",
        "game_date": "2026-07-29",
        "matchup_id": "wnba-las-ind-20260729",
        "market_type": market_type,
        "line_source": "live_odds",
        "orientation_source": "exact_match",
        "odds_american": -110,
        "live_total_line": pd.NA,
        "total_line": pd.NA,
        "live_spread_line": pd.NA,
        "spread_line": pd.NA,
        "market_probability": 0.5,
    }
    row.update(overrides)
    return row


def test_wnba_total_range_accepts_standard_line_and_rejects_corrupt_line():
    valid_spread = _line_candidate(
        "spread_home", live_spread_line=-4.5, spread_line=-4.5
    )
    normal_total = _line_candidate(
        "total_over", live_total_line=177.5, total_line=177.5
    )
    corrupt_total = _line_candidate(
        "total_under", live_total_line=17.5, total_line=17.5
    )

    diagnostics = {}
    out = sp._filter_preselection_line_integrity(
        pd.DataFrame([valid_spread, normal_total, corrupt_total]),
        diagnostics_out=diagnostics,
    )

    assert set(out["market_type"]) == {"spread_home", "total_over"}
    assert diagnostics["preselection_invalid_total_candidate_count"] == 1
    assert diagnostics["preselection_dropped_total_candidate_count"] == 1
