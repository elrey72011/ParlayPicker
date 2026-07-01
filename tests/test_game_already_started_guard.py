"""In-progress game guard: rows whose odds-API commence time predates the run
carry IN-GAME odds, not pre-game lines, and must be hard-benched to No Play.

Motivating slate (1 Jul, 4:40 PM ET run): four games 1-3 hours underway rode
through with in-game prices — one with a 19.5 MLB "total" and -100000
moneylines. The line-provenance guard only caught them because those lines
looked corrupt; a plausible-looking in-game line would have been staked as if
it were a pre-game price.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.streamlit_pipeline as sp


def _analysis_row(**overrides) -> dict:
    row = {
        "league": "MLB",
        "home_team": "Philadelphia",
        "away_team": "Pittsburgh",
        "game_date": pd.Timestamp("2026-07-01", tz="UTC"),
        "market_type": "total_under",
        "total_line": 8.5,
        "spread_line": pd.NA,
        "expected_value": 0.05,
        "edge": 0.04,
        "calibrated_probability": 0.58,
        "effective_win_probability": 0.58,
        "market_probability": 0.50,
        "ml_probability": 0.58,
        "odds_american": -110,
        "odds_source": "odds_api",
        "consensus_agreement": "Agrees",
        "is_live_data": True,
        "candidate_source": "upload",
        "orientation_source": "model",
        "upload_match_reason": "matched",
    }
    row.update(overrides)
    return row


def test_started_game_is_benched_to_no_play():
    df = pd.DataFrame(
        [
            _analysis_row(game_already_started_flag=True),
            _analysis_row(
                home_team="Atlanta", away_team="Saint Louis", game_already_started_flag=False
            ),
        ]
    )
    best = sp.build_best_picks_df(df)
    started = best[best["home_team"] == "Philadelphia"].iloc[0]
    upcoming = best[best["home_team"] == "Atlanta"].iloc[0]

    assert started["Pick_Status"] == "No Play"
    assert started["status_blocker_stage"] == "game_already_started"
    assert "already started" in str(started["Status_Reason"])
    assert float(pd.to_numeric(started["Kelly_Bet_Size"], errors="coerce") or 0) == 0.0
    # The untouched game must not be collateral damage.
    assert upcoming["Pick_Status"] != "No Play" or "already started" not in str(upcoming["Status_Reason"])


def test_missing_flag_column_changes_nothing():
    df = pd.DataFrame([_analysis_row()])
    best = sp.build_best_picks_df(df)
    assert "already started" not in str(best.iloc[0]["Status_Reason"])


def test_pipeline_flags_past_commence_time(monkeypatch):
    base_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Philadelphia"],
            "away_team": ["Pittsburgh"],
            "game_date": ["2026-07-01T00:00:00Z"],
            "odds_american": [-110],
        }
    )
    totals_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Philadelphia"],
            "away_team": ["Pittsburgh"],
            "line": [8.5],
            "pick": ["Under 8.5"],
            "winprobability": [0.40],
            "odds_american": [-110],
        }
    )
    past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)).isoformat()
    live_odds_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Philadelphia"],
            "away_team": ["Pittsburgh"],
            "game_date": [past],
            "commence_time_raw": [past],
            "total_line": [8.5],
            "total_over_odds": [-110],
            "total_under_odds": [-110],
        }
    )
    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: live_odds_df)

    analysis_df, _best, _diags = sp.run_analysis_pipeline(
        sports=["MLB"], max_rows=20, use_ml=False, spreads_df=None, totals_df=totals_df
    )

    assert "game_already_started_flag" in analysis_df.columns
    flagged = analysis_df["game_already_started_flag"].fillna(False).astype(bool)
    assert flagged.any(), "past commence_time_raw must set the started flag"


def test_pipeline_does_not_flag_rows_without_commence_time(monkeypatch):
    # Upload-only fallback slate: no commence_time_raw at all -> nothing flagged
    # (date-only game_date midnights must never trip the guard).
    base_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Philadelphia"],
            "away_team": ["Pittsburgh"],
            "game_date": ["2026-07-01T00:00:00Z"],
            "odds_american": [-110],
        }
    )
    totals_df = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Philadelphia"],
            "away_team": ["Pittsburgh"],
            "line": [8.5],
            "pick": ["Under 8.5"],
            "winprobability": [0.60],
            "odds_american": [-110],
        }
    )
    monkeypatch.setattr(sp, "load_base_data", lambda: base_df)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: pd.DataFrame())

    analysis_df, _best, _diags = sp.run_analysis_pipeline(
        sports=["MLB"], max_rows=20, use_ml=False, spreads_df=None, totals_df=totals_df
    )

    assert not analysis_df["game_already_started_flag"].fillna(False).astype(bool).any()
