import pandas as pd

from app.ui.kalshi_diagnostics import build_kalshi_diagnostic_frames
from streamlit_app import _kalshi_coverage_metrics


def _diagnostic_frame() -> pd.DataFrame:
    rows = []
    for game_number, league in [(1, "MLB"), (2, "NCAAF"), (3, "NFL")]:
        for market_type in ("spread_home", "spread_away", "total_over", "total_under"):
            matched = game_number == 1 or (game_number == 2 and market_type.startswith("total"))
            rows.append(
                {
                    "league": league,
                    "away_team": f"Away {game_number}",
                    "home_team": f"Home {game_number}",
                    "game_date": "2026-08-29",
                    "market_type": market_type,
                    "best_pick": market_type,
                    "kalshi_match_status": "matched" if matched else "miss",
                    "kalshi_match_reason": "total_match" if matched else "no_series_events",
                    "kalshi_candidate_event_count": 1 if matched else 0,
                }
            )
    return pd.DataFrame(rows)


def test_kalshi_diagnostics_separate_row_and_game_coverage() -> None:
    frame = _diagnostic_frame()

    diagnostics = build_kalshi_diagnostic_frames(frame)
    metrics = _kalshi_coverage_metrics(frame)

    assert diagnostics["overall"] == {
        "attempted_rows": 12,
        "matched_rows": 6,
        "row_coverage": 0.5,
        "attempted_games": 3,
        "matched_games": 2,
        "game_coverage": 2 / 3,
    }
    assert metrics["kalshi_match_rate"] == 0.5
    assert metrics["kalshi_game_match_rate"] == 2 / 3
    assert diagnostics["market_summary"].set_index("market_family").loc["total", "matched_rows"] == 4


def test_kalshi_diagnostics_preserve_every_export_row() -> None:
    frame = pd.DataFrame(
        {
            "league": ["NCAAF"] * 130,
            "away_team": [f"Away {index}" for index in range(130)],
            "home_team": [f"Home {index}" for index in range(130)],
            "game_date": ["2026-08-29"] * 130,
            "market_type": ["spread_home"] * 130,
            "kalshi_match_status": ["miss"] * 124 + ["matched"] * 6,
            "kalshi_match_reason": ["no_series_events"] * 124 + ["spread_match"] * 6,
        }
    )

    diagnostics = build_kalshi_diagnostic_frames(frame)

    assert len(diagnostics["misses"]) == 124
    assert len(diagnostics["matched"]) == 6
    assert int(diagnostics["reason_summary"].iloc[0]["rows"]) == 124
