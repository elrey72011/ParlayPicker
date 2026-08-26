import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.streamlit_pipeline import _theover_upload_coverage, build_theover_bet_rows


EXPECTED_COLUMNS = [
    "league",
    "home_team",
    "away_team",
    "game_date",
    "market_type",
    "spread_line",
    "total_line",
    "theover_probability",
    "odds_american",
    "market_probability",
    "expected_value",
    "edge",
    "best_pick",
]


def test_build_theover_bet_rows_missing_odds_defaults_to_pd_na():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick": ["Over"],
            "line": [221.5],
            "winprobability": [0.57],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert not out.empty
    assert pd.isna(out.loc[0, "odds_american"])


def test_build_theover_bet_rows_missing_probability_column_does_not_crash():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick": ["Under"],
            "line": [219.5],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert not out.empty
    assert "theover_probability" in out.columns
    assert pd.isna(out.loc[0, "theover_probability"])


def test_build_theover_bet_rows_only_totals_upload():
    totals_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "selection": ["Over"],
            "points": [220.5],
            "probability": [58],
            "american_odds": [-105],
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["NBA"])

    assert len(out) == 2
    assert set(out["market_type"].tolist()) == {"total_over", "total_under"}
    assert out["total_line"].iloc[0] == 220.5
    assert out["odds_american"].iloc[0] == -105


def test_build_theover_bet_rows_only_spreads_upload():
    spreads_df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "pick_team": ["Boston Celtics"],
            "spread_line": [-3.5],
            "win_probability": [0.54],
        }
    )

    out = build_theover_bet_rows(spreads_df, None, ["NBA"])

    assert len(out) == 2
    assert set(out["market_type"].tolist()) == {"spread_home", "spread_away"}
    assert sorted(out["spread_line"].tolist()) == [-3.5, 3.5]
    assert out["matchup_id"].notna().all()
    assert out["matchup_id"].astype(str).str.len().gt(0).all()


def test_theover_coverage_separates_file_rows_from_usable_side_signals():
    sides_df = pd.DataFrame(
        {
            "League": ["MLB", "WNBA"],
            "HomeTeam": ["Atlanta", "Seattle"],
            "AwayTeam": ["Boston", "Toronto"],
            "Pick": ["ATL", "TOR"],
            "Line": [-130, 6.5],
            "WinProbability": [pd.NA, pd.NA],
            "Market": ["Moneyline", "Spread"],
        }
    )

    coverage = _theover_upload_coverage(sides_df, "spreads")

    assert coverage == {
        "file_game_count": 2,
        "market_game_count": 1,
        "probability_game_count": 0,
    }


def test_theover_totals_coverage_counts_normalized_loader_headers():
    totals_df = pd.DataFrame(
        {
            "league": ["MLB", "WNBA"],
            "home_team": ["Atlanta", "Seattle"],
            "away_team": ["Boston", "Toronto"],
            "line": [8.5, 177.5],
            "winprobability": [0.61, 0.55],
            "market": ["Total", "Total"],
        }
    )

    coverage = _theover_upload_coverage(totals_df, "totals")

    assert coverage == {
        "file_game_count": 2,
        "market_game_count": 2,
        "probability_game_count": 2,
    }


def test_build_theover_bet_rows_empty_uploads_returns_stable_schema():
    out = build_theover_bet_rows(pd.DataFrame(), pd.DataFrame(), ["NBA"])

    assert out.empty
    for col in EXPECTED_COLUMNS:
        assert col in out.columns


def test_degraded_theover_feed_is_down_weighted_not_nulled():
    # 15 Jun MLB totals: TheOver legitimately rated six of eight games at hit-rate
    # 0.75, which the degradation guard folds (over picks 0.75 and under picks
    # P(over)=0.25 both map to magnitude 0.25) into one cluster and flags as
    # "degraded." Previously this nulled TheOver for the whole slate, flipping
    # picks. The guard now shrinks each totals read toward 0.50 by
    # THEOVER_DEGRADED_FADE_SHRINK (0.50) instead — so the signal survives, damped.
    totals_df = pd.DataFrame(
        {
            "League": ["MLB"] * 8,
            "HomeTeam": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "AwayTeam": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "Pick": ["Over"] * 6 + ["Under"] * 2,
            "Line": [8.5] * 8,
            # P(Over): six clustered at 0.75 (-> magnitude 0.25), plus two genuine
            # high-confidence reads so the slate is plausibly real, not constant.
            "WinProbability": [0.75, 0.75, 0.75, 0.75, 0.95, 0.80, 0.25, 0.25],
            "Market": ["Total"] * 8,
        }
    )

    out = build_theover_bet_rows(None, totals_df, ["MLB"])
    overs = out[out["market_type"] == "total_over"].reset_index(drop=True)

    # Not nulled: every totals read retains a (damped) value.
    assert overs["theover_probability"].notna().all()
    # Shrunk toward 0.50 by 50%: a 0.95 read becomes 0.5 + 0.5*(0.95-0.5) = 0.725.
    assert abs(float(overs.loc[4, "theover_probability"]) - 0.725) < 1e-6
    # The 0.75 reads become 0.625, still a directional Over lean (> 0.5).
    assert (overs["theover_probability"].dropna() != 0.5).any()
    assert float(overs.loc[0, "theover_probability"]) > 0.5

    # The degradation warning must survive the canonical reindex so the downstream
    # production degraded-run guard can halve Kelly: since we now retain (down-weight)
    # the signal, the de-staking safety has to travel with it. The reason string
    # contains "degraded", which is what the Kelly-reduction guard keys on.
    warn = out["run_health_warning"].astype("string").fillna("")
    assert (warn.str.len() > 0).any()
    assert warn.str.contains("degraded", case=False).any()
