import pandas as pd

from app_core.prop_calibration import (
    DEFAULT_PROP_RESULTS_PATH,
    DEFAULT_PROP_RESULTS_RUNTIME_PATH,
    apply_prop_calibration,
    directional_market_key,
    fit_prop_calibration,
    load_prop_results_log,
    normalize_prop_results,
    persist_prop_results_log,
)
from app_core.prop_runner import (
    apply_market_probation,
    apply_production_prop_gate,
    market_records_from_log,
)


def _card(market_type="pitcher_strikeouts_over", probability=0.72):
    return pd.DataFrame([{
        "market_type": market_type,
        "best_pick": "Pitcher Over 5.5 Ks",
        "WinProbability": probability,
        "edge": probability - 0.50,
        "odds_american": -110,
        "Kelly_Bet_Size": 5.0,
    }])


def test_repo_bundled_prop_ledger_loads_independent_of_working_directory():
    assert DEFAULT_PROP_RESULTS_PATH.is_absolute()
    assert DEFAULT_PROP_RESULTS_RUNTIME_PATH.is_absolute()
    ledger = load_prop_results_log()
    assert ledger is not None
    assert len(ledger) >= 572
    assert ledger["result"].isin(["WIN", "LOSS"]).sum() >= 539
    assert pd.to_datetime(ledger["game_date"]).max() >= pd.Timestamp("2026-08-02")


def test_runtime_prop_ledger_is_persisted_and_restored_over_baseline(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    runtime_path = tmp_path / "runtime.csv"
    baseline = pd.DataFrame({
        "game_date": ["2026-08-01"],
        "player": ["Player One"],
        "pick": ["Player One Over 0.5 Hits"],
        "market_type": ["batter_hits_over"],
        "result": ["LOSS"],
        "raw_probability": [0.65],
    })
    cumulative = pd.concat([
        baseline.assign(result="WIN"),
        pd.DataFrame({
            "game_date": ["2026-08-02"],
            "player": ["Player Two"],
            "pick": ["Player Two Under 1.5 Hits"],
            "market_type": ["batter_hits_under"],
            "result": ["LOSS"],
            "raw_probability": [0.62],
        }),
    ], ignore_index=True)
    baseline.to_csv(baseline_path, index=False)

    assert persist_prop_results_log(cumulative, runtime_path)
    restored = load_prop_results_log(
        baseline_path,
        runtime_path=runtime_path,
    )

    assert restored is not None
    assert len(restored) == 2
    assert restored.loc[restored["player"].eq("Player One"), "result"].item() == "WIN"
    assert restored["game_date"].max() == "2026-08-02"


def test_directional_market_keys_do_not_pool_over_and_under():
    assert directional_market_key("pitcher_strikeouts_over") == "pitcher_strikeouts_over"
    assert directional_market_key("pitcher_strikeouts_under") == "pitcher_strikeouts_under"
    assert directional_market_key("", "Player Under 1.5 Hits") == "batter_hits_under"


def test_missing_ledger_shrinks_model_toward_market_and_haircuts_confidence():
    out = apply_prop_calibration(_card(), None)
    row = out.iloc[0]
    assert row["MarketProbability"] == 0.50
    assert row["CalibrationSource"] == "market_blend_fallback"
    assert row["MarketProbability"] < row["WinProbability"] < row["RawWinProbability"]
    assert row["ConservativeWinProbability"] < row["CalibratedProbability"]


def test_calibration_excludes_same_day_results_to_prevent_leakage():
    ledger = pd.DataFrame({
        "market_type": ["pitcher_strikeouts_over"] * 25,
        "result": ["WIN"] * 20 + ["LOSS"] * 5,
        "raw_probability": [0.65] * 25,
        "game_date": ["2026-07-22"] * 25,
    })
    clean = normalize_prop_results(ledger, as_of_date="2026-07-22")
    assert clean.empty
    assert fit_prop_calibration(ledger, as_of_date="2026-07-22") == {}


def test_directional_probation_requires_that_side_to_be_profitable():
    ledger = pd.DataFrame({
        "market_type": (
            ["pitcher_strikeouts_over"] * 20
            + ["pitcher_strikeouts_under"] * 20
        ),
        "result": (
            ["WIN"] * 14 + ["LOSS"] * 6
            + ["WIN"] * 8 + ["LOSS"] * 12
        ),
        "odds_american": [-110] * 40,
    })
    records = market_records_from_log(ledger, detailed=True)
    card = pd.concat([
        _card("pitcher_strikeouts_over"),
        _card("pitcher_strikeouts_under"),
    ], ignore_index=True)
    out = apply_market_probation(card, records)
    assert not bool(out.loc[0, "Market_Probation"])
    assert bool(out.loc[1, "Market_Probation"])
    assert out.loc[1, "Kelly_Bet_Size"] == 1.0


def test_pooled_calibration_can_rank_but_cannot_fund_an_unproven_direction():
    card = pd.DataFrame({
        "player": ["A", "B"],
        "matchup": ["X @ Y", "Z @ Q"],
        "market_type": ["pitcher_strikeouts_over", "pitcher_strikeouts_over"],
        "best_pick": ["A Over 5.5 Ks", "B Over 5.5 Ks"],
        "line": [5.5, 5.5],
        "expected_count": [6.5, 6.5],
        "odds_american": [-110, -110],
        "WinProbability": [0.66, 0.66],
        "expected_value": [0.08, 0.08],
        "Market_Probation": [False, False],
        "Pick_Status": ["Actionable", "Actionable"],
        "Kelly_Bet_Size": [1.0, 1.0],
        "CalibrationSource": ["pooled", "directional"],
        "CalibrationSampleSize": [100, 20],
    })
    out = apply_production_prop_gate(card)
    assert out["production_eligible"].tolist() == [False, True]
    assert "calibration" in out.loc[0, "production_gate_reason"].lower()


def test_pooled_calibration_is_shrink_only_and_uses_directional_uncertainty():
    ledger = pd.DataFrame({
        "market_type": ["batter_total_bases_over"] * 100,
        "result": ["WIN"] * 90 + ["LOSS"] * 10,
        "raw_probability": [0.65] * 100,
        "odds_american": [-110] * 100,
        "game_date": ["2026-07-22"] * 100,
    })
    out = apply_prop_calibration(
        _card("pitcher_strikeouts_over", probability=0.72),
        ledger,
        as_of_date="2026-07-23",
    )
    row = out.iloc[0]
    assert row["CalibrationSource"] == "pooled"
    assert row["CalibrationSampleSize"] == 0
    assert row["DirectionalCalibrationSampleSize"] == 0
    assert row["CalibrationProfileSampleSize"] == 100
    assert row["CalibratedProbability"] <= row["RawWinProbability"]
    assert row["ConservativeWinProbability"] < row["CalibratedProbability"]
