import numpy as np
import pandas as pd
import pytest

from scripts.validate_nfl_market_model import (
    FEATURES, PROTOCOL, evaluate, fit_candidate, implied, metrics,
    pregame_features, probability_above, scoring_projection,
)


def schedule():
    rows = []
    for number, date in enumerate(pd.date_range("2022-09-01", periods=8, freq="7D")):
        rows.append(dict(game_id=str(number), game_type="REG", season=2022,
                         gameday=date, home_team="KC", away_team="LV", location="Home",
                         home_score=20+number, away_score=17+number%3,
                         spread_line=3.5, total_line=44.5,
                         home_spread_odds=-110, away_spread_odds=-110,
                         over_odds=-110, under_odds=-110))
    return pd.DataFrame(rows)


def test_current_and_future_scores_cannot_change_pregame_features():
    source = schedule()
    baseline = pregame_features(source).set_index("game_id")
    changed = source.copy()
    changed.loc[changed.index >= 4, ["home_score", "away_score"]] = [99, 0]
    mutated = pregame_features(changed).set_index("game_id")
    pd.testing.assert_series_equal(baseline.loc["4", FEATURES], mutated.loc["4", FEATURES])
    assert mutated.loc["5", "home_ppg"] != baseline.loc["5", "home_ppg"]


def test_same_day_scores_and_input_order_do_not_leak():
    source = schedule()
    extra = source.iloc[[4]].copy()
    extra["game_id"] = "4-second"
    extra[["home_score", "away_score"]] = [99, 0]
    rows = pregame_features(pd.concat([source, extra]).sample(frac=1, random_state=7)).set_index("game_id")
    assert rows.loc["4", FEATURES].tolist() == rows.loc["4-second", FEATURES].tolist()


def test_relocations_retain_history_but_unplayed_games_do_not_add_history():
    source = schedule()
    source.loc[:3, "away_team"] = "OAK"
    assert "4" in pregame_features(source).game_id.tolist()
    source.loc[0, ["home_score", "away_score"]] = np.nan
    assert "4" not in pregame_features(source).game_id.tolist()
    assert "5" in pregame_features(source).game_id.tolist()


def test_duplicate_schedule_rejected_instead_of_double_counted():
    source = schedule()
    with pytest.raises(ValueError, match="unique"):
        pregame_features(pd.concat([source, source.iloc[[0]]]))


def test_probability_conditions_on_push_and_complements_opposite_side():
    from math import erf, sqrt
    cdf = lambda x: .5*(1+erf((x-5)/(10*sqrt(2))))
    expected = (1-cdf(3.5))/(1-cdf(3.5)+cdf(2.5))
    assert probability_above(5, 3, 10) == pytest.approx(expected)
    assert probability_above(5, 3, 10)+probability_above(-5, -3, 10) == pytest.approx(1)
    assert probability_above(5, 3.5, 10) == pytest.approx(1-cdf(3.5))
    assert probability_above(5, 3.5, 10)+probability_above(-5, -3.5, 10) == pytest.approx(1)
    with pytest.raises(ValueError):
        probability_above(5, np.nan, 10)


def fitted():
    rows = pregame_features(schedule())
    train = rows.iloc[:2].copy()
    calibration = rows.iloc[2:].copy()
    return fit_candidate(train, calibration), rows


def test_fit_and_projections_do_not_use_lines_prices_or_external_predictions():
    model, rows = fitted()
    mutated = rows.copy()
    for column in ["spread_line", "total_line", "home_spread_odds", "over_odds",
                   "kalshi_probability", "theover_probability", "gemini_confidence"]:
        mutated[column] = 999
    other = fit_candidate(mutated.iloc[:2], mutated.iloc[2:])
    assert other == model
    assert scoring_projection(model, rows, "margin") == pytest.approx(scoring_projection(other, mutated, "margin"))


def test_calibration_cannot_overlap_training():
    rows = pregame_features(schedule())
    with pytest.raises(ValueError, match="precede"):
        fit_candidate(rows, rows)


def test_historical_spread_sign_push_exclusion_and_missing_price_counts():
    model, rows = fitted()
    model["targets"]["margin"] = {"coefficients": [0]+[0]*len(FEATURES), "sigma": 14}
    rows = rows.copy()
    rows["home_score"] = [24, 20, 23, 30]
    rows["away_score"] = [20, 24, 20, 20]
    rows["spread_line"] = 3
    rows.loc[rows.index[-1], "away_spread_odds"] = np.nan
    report = evaluate(model, rows, "margin")
    assert report["scored"] == 2
    assert report["pushes_excluded"] == 1
    assert report["missing_prices_or_line"] == 1
    assert report["candidate"]["brier"] == pytest.approx(metrics(np.array([1, 0]),
        np.repeat(probability_above(0, 3, 14), 2))["brier"])
    assert report["decision"]["promotable"] is False


def test_invalid_prices_excluded_and_fixed_protocol_is_research_only():
    values = implied(pd.Series([-110, 100, 0, 50, np.nan, np.inf]))
    assert values.iloc[0] == pytest.approx(110/210)
    assert values.iloc[1] == .5
    assert values.iloc[2:].isna().all()
    assert PROTOCOL["automatic_installation"] is False
    assert PROTOCOL["evaluation_seasons"] == [2024, 2025]
