"""Synthetic fallback datasets for ML training when live history is sparse."""

from __future__ import annotations

from typing import Dict, List

ADJUSTABLE_KEYS = [
    "home_avg_for",
    "home_avg_against",
    "home_form_pct",
    "home_trend_score",
    "home_record_pct",
    "away_avg_for",
    "away_avg_against",
    "away_form_pct",
    "away_trend_score",
    "away_record_pct",
    "sentiment_home",
    "sentiment_away",
    "sentiment_diff",
    "home_ml_implied",
    "away_ml_implied",
]

_TEMPLATES: Dict[str, List[Dict[str, float]]] = {
    "americanfootball_nfl": [
        {
            "home_avg_for": 27.5,
            "home_avg_against": 21.2,
            "home_form_pct": 0.66,
            "home_trend_score": 0.5,
            "home_record_pct": 0.68,
            "away_avg_for": 23.1,
            "away_avg_against": 25.6,
            "away_form_pct": 0.48,
            "away_trend_score": -0.3,
            "away_record_pct": 0.45,
            "sentiment_home": 0.15,
            "sentiment_away": -0.05,
            "sentiment_diff": 0.20,
            "home_ml_implied": 0.58,
            "away_ml_implied": 0.42,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 24.1,
            "home_avg_against": 27.4,
            "home_form_pct": 0.44,
            "home_trend_score": -0.6,
            "home_record_pct": 0.42,
            "away_avg_for": 28.2,
            "away_avg_against": 20.8,
            "away_form_pct": 0.72,
            "away_trend_score": 0.7,
            "away_record_pct": 0.70,
            "sentiment_home": -0.10,
            "sentiment_away": 0.18,
            "sentiment_diff": -0.28,
            "home_ml_implied": 0.41,
            "away_ml_implied": 0.59,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
        {
            "home_avg_for": 30.4,
            "home_avg_against": 19.5,
            "home_form_pct": 0.80,
            "home_trend_score": 0.8,
            "home_record_pct": 0.75,
            "away_avg_for": 21.9,
            "away_avg_against": 28.7,
            "away_form_pct": 0.35,
            "away_trend_score": -0.8,
            "away_record_pct": 0.40,
            "sentiment_home": 0.22,
            "sentiment_away": -0.18,
            "sentiment_diff": 0.40,
            "home_ml_implied": 0.66,
            "away_ml_implied": 0.34,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 18.7,
            "home_avg_against": 26.1,
            "home_form_pct": 0.32,
            "home_trend_score": -0.9,
            "home_record_pct": 0.30,
            "away_avg_for": 22.9,
            "away_avg_against": 19.4,
            "away_form_pct": 0.63,
            "away_trend_score": 0.6,
            "away_record_pct": 0.62,
            "sentiment_home": -0.22,
            "sentiment_away": 0.14,
            "sentiment_diff": -0.36,
            "home_ml_implied": 0.33,
            "away_ml_implied": 0.67,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
    ],
    "icehockey_nhl": [
        {
            "home_avg_for": 3.4,
            "home_avg_against": 2.6,
            "home_form_pct": 0.70,
            "home_trend_score": 0.6,
            "home_record_pct": 0.68,
            "away_avg_for": 2.7,
            "away_avg_against": 3.1,
            "away_form_pct": 0.46,
            "away_trend_score": -0.4,
            "away_record_pct": 0.48,
            "sentiment_home": 0.10,
            "sentiment_away": -0.04,
            "sentiment_diff": 0.14,
            "home_ml_implied": 0.61,
            "away_ml_implied": 0.39,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 2.5,
            "home_avg_against": 3.4,
            "home_form_pct": 0.38,
            "home_trend_score": -0.7,
            "home_record_pct": 0.40,
            "away_avg_for": 3.3,
            "away_avg_against": 2.8,
            "away_form_pct": 0.66,
            "away_trend_score": 0.7,
            "away_record_pct": 0.63,
            "sentiment_home": -0.08,
            "sentiment_away": 0.15,
            "sentiment_diff": -0.23,
            "home_ml_implied": 0.37,
            "away_ml_implied": 0.63,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
        {
            "home_avg_for": 3.1,
            "home_avg_against": 3.0,
            "home_form_pct": 0.52,
            "home_trend_score": 0.1,
            "home_record_pct": 0.51,
            "away_avg_for": 3.0,
            "away_avg_against": 3.2,
            "away_form_pct": 0.49,
            "away_trend_score": -0.1,
            "away_record_pct": 0.48,
            "sentiment_home": 0.04,
            "sentiment_away": -0.01,
            "sentiment_diff": 0.05,
            "home_ml_implied": 0.51,
            "away_ml_implied": 0.49,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 2.8,
            "home_avg_against": 3.6,
            "home_form_pct": 0.35,
            "home_trend_score": -0.8,
            "home_record_pct": 0.38,
            "away_avg_for": 3.5,
            "away_avg_against": 2.9,
            "away_form_pct": 0.62,
            "away_trend_score": 0.8,
            "away_record_pct": 0.60,
            "sentiment_home": -0.14,
            "sentiment_away": 0.12,
            "sentiment_diff": -0.26,
            "home_ml_implied": 0.36,
            "away_ml_implied": 0.64,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
    ],
    "basketball_nba": [
        {
            "home_avg_for": 118.4,
            "home_avg_against": 112.6,
            "home_form_pct": 0.68,
            "home_trend_score": 0.6,
            "home_record_pct": 0.66,
            "away_avg_for": 111.2,
            "away_avg_against": 114.5,
            "away_form_pct": 0.44,
            "away_trend_score": -0.5,
            "away_record_pct": 0.44,
            "sentiment_home": 0.12,
            "sentiment_away": -0.03,
            "sentiment_diff": 0.15,
            "home_ml_implied": 0.64,
            "away_ml_implied": 0.36,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 112.1,
            "home_avg_against": 116.9,
            "home_form_pct": 0.41,
            "home_trend_score": -0.6,
            "home_record_pct": 0.42,
            "away_avg_for": 119.3,
            "away_avg_against": 109.7,
            "away_form_pct": 0.71,
            "away_trend_score": 0.7,
            "away_record_pct": 0.69,
            "sentiment_home": -0.09,
            "sentiment_away": 0.16,
            "sentiment_diff": -0.25,
            "home_ml_implied": 0.38,
            "away_ml_implied": 0.62,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
        {
            "home_avg_for": 115.0,
            "home_avg_against": 113.8,
            "home_form_pct": 0.54,
            "home_trend_score": 0.2,
            "home_record_pct": 0.53,
            "away_avg_for": 114.4,
            "away_avg_against": 115.1,
            "away_form_pct": 0.49,
            "away_trend_score": -0.1,
            "away_record_pct": 0.50,
            "sentiment_home": 0.05,
            "sentiment_away": 0.01,
            "sentiment_diff": 0.04,
            "home_ml_implied": 0.53,
            "away_ml_implied": 0.47,
            "home_field_advantage": 1.0,
            "home_win": 1,
        },
        {
            "home_avg_for": 108.6,
            "home_avg_against": 118.3,
            "home_form_pct": 0.33,
            "home_trend_score": -0.7,
            "home_record_pct": 0.35,
            "away_avg_for": 120.7,
            "away_avg_against": 111.9,
            "away_form_pct": 0.67,
            "away_trend_score": 0.8,
            "away_record_pct": 0.64,
            "sentiment_home": -0.13,
            "sentiment_away": 0.14,
            "sentiment_diff": -0.27,
            "home_ml_implied": 0.35,
            "away_ml_implied": 0.65,
            "home_field_advantage": 1.0,
            "home_win": 0,
        },
    ],
}

_GENERIC_TEMPLATE = {
    "home_avg_for": 5.0,
    "home_avg_against": 4.5,
    "home_form_pct": 0.55,
    "home_trend_score": 0.1,
    "home_record_pct": 0.55,
    "away_avg_for": 4.7,
    "away_avg_against": 5.1,
    "away_form_pct": 0.48,
    "away_trend_score": -0.1,
    "away_record_pct": 0.49,
    "sentiment_home": 0.05,
    "sentiment_away": -0.02,
    "sentiment_diff": 0.07,
    "home_ml_implied": 0.56,
    "away_ml_implied": 0.44,
    "home_field_advantage": 1.0,
    "home_win": 1,
}

_TEMPLATES.setdefault(
    "generic",
    [
        dict(_GENERIC_TEMPLATE),
        {
            **_GENERIC_TEMPLATE,
            "home_win": 0,
            "sentiment_diff": -0.07,
            "home_ml_implied": 0.44,
            "away_ml_implied": 0.56,
        },
    ],
)


def build_sample_rows(sport_key: str, rows_needed: int) -> List[Dict[str, float]]:
    if rows_needed <= 0:
        return []

    templates = _TEMPLATES.get(sport_key) or _TEMPLATES["generic"]
    generated: List[Dict[str, float]] = []

    for idx in range(rows_needed):
        template = templates[idx % len(templates)]
        row = dict(template)

        for sd_key in [
            "sportsdata_home_power_index",
            "sportsdata_home_turnover_margin",
            "sportsdata_home_net",
            "sportsdata_home_strength_delta",
            "sportsdata_away_power_index",
            "sportsdata_away_turnover_margin",
            "sportsdata_away_net",
            "sportsdata_away_strength_delta",
            "sportsdata_strength_delta_diff",
        ]:
            row.setdefault(sd_key, 0.0)

        # Create small deterministic adjustments so each row remains unique.
        scale = (idx // len(templates)) * 0.02
        swing = ((idx % 5) - 2) * 0.01
        for key in ADJUSTABLE_KEYS:
            value = row.get(key)
            if value is None:
                continue
            if key.startswith("home_"):
                row[key] = float(max(0.0, value + scale))
            elif key.startswith("away_"):
                row[key] = float(max(0.0, value - scale / 2))
            elif key == "sentiment_diff":
                row[key] = float(value + swing)
            elif key == "sentiment_home":
                row[key] = float(value + swing / 2)
            elif key == "sentiment_away":
                row[key] = float(value - swing / 2)
            elif key in {"home_ml_implied", "away_ml_implied"}:
                row[key] = float(min(0.96, max(0.04, value + swing / 3)))
        generated.append(row)

    return generated


__all__ = ["build_sample_rows"]
