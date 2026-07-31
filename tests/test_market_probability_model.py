import pandas as pd
import pytest

from app_core.market_probability_model import predict_market_probabilities


def _resolved_rows(market_types, spread_lines=None, total_lines=None, league="MLB"):
    count = len(market_types)
    return pd.DataFrame(
        {
            "League": [league] * count,
            "market_type": market_types,
            "spread_line": spread_lines or [pd.NA] * count,
            "total_line": total_lines or [pd.NA] * count,
            "feature_home_win_pct": [0.58] * count,
            "feature_away_win_pct": [0.46] * count,
            "feature_home_ppg": [4.9 if league == "MLB" else 86.0] * count,
            "feature_away_ppg": [4.2 if league == "MLB" else 80.0] * count,
            "feature_home_oppg": [4.0 if league == "MLB" else 79.0] * count,
            "feature_away_oppg": [4.8 if league == "MLB" else 84.0] * count,
            "feature_diff_last5": [0.10] * count,
            "ml_feature_eligible": [True] * count,
            "stats_resolution_status": ["resolved"] * count,
            "odds_american": [-110, -110][:count],
            "market_probability": [0.50, 0.50][:count],
            "theover_probability": [0.70, 0.30][:count],
        }
    )


def test_spread_probabilities_match_exact_side_and_are_complements():
    frame = _resolved_rows(
        ["spread_home", "spread_away"],
        spread_lines=[1.5, -1.5],
    )

    out = predict_market_probabilities(frame)

    assert out["ml_probability"].notna().all()
    assert out["ml_target"].eq("spread_cover").all()
    assert out["ml_probability"].sum() == pytest.approx(1.0)
    assert out["ml_probability_source"].str.startswith("score-distribution-v1").all()


def test_total_probabilities_are_complements_and_do_not_reuse_theover_or_price():
    frame = _resolved_rows(
        ["total_over", "total_under"],
        total_lines=[8.5, 8.5],
    )
    changed_external_signals = frame.copy()
    changed_external_signals["odds_american"] = [250, -300]
    changed_external_signals["market_probability"] = [0.28, 0.72]
    changed_external_signals["theover_probability"] = [0.05, 0.95]

    out = predict_market_probabilities(frame)
    changed = predict_market_probabilities(changed_external_signals)

    assert out["ml_probability"].sum() == pytest.approx(1.0)
    assert out["ml_target"].tolist() == ["total_over", "total_under"]
    assert changed["ml_probability"].tolist() == pytest.approx(out["ml_probability"].tolist())


def test_unresolved_stats_are_not_disguised_as_model_probability():
    frame = _resolved_rows(["total_over"], total_lines=[8.5])
    frame.loc[0, "ml_feature_eligible"] = False
    frame.loc[0, "stats_resolution_status"] = "unresolved"

    out = predict_market_probabilities(frame)

    assert pd.isna(out.loc[0, "ml_probability"])
    assert out.loc[0, "ml_feature_quality"] == "unavailable"


def test_wnba_spread_receives_target_specific_probability():
    frame = _resolved_rows(
        ["spread_home", "spread_away"],
        spread_lines=[-4.5, 4.5],
        league="WNBA",
    )

    out = predict_market_probabilities(frame)

    assert out["ml_probability"].notna().all()
    assert out["ml_probability"].sum() == pytest.approx(1.0)
    assert out["ml_probability_source"].eq("score-distribution-v1:wnba").all()


def test_legacy_scaled_mlb_scoring_features_are_unscaled():
    raw = _resolved_rows(["total_over"], total_lines=[8.5])
    scaled = raw.copy()
    for column in (
        "feature_home_ppg",
        "feature_away_ppg",
        "feature_home_oppg",
        "feature_away_oppg",
    ):
        scaled[column] = scaled[column] * 25.0

    raw_out = predict_market_probabilities(raw)
    scaled_out = predict_market_probabilities(scaled)

    assert scaled_out.loc[0, "ml_probability"] == pytest.approx(raw_out.loc[0, "ml_probability"])

