import os
import sys
import warnings

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.prediction_engine import PredictionEngine


def test_predict_batch_no_settingwithcopy_warning_for_league_assignment():
    engine = PredictionEngine(model_path="models/does_not_exist.json")
    source = pd.DataFrame(
        [
            {
                "league": "ncaam",
                "home_team": "Team A",
                "away_team": "Team B",
                "market_type": "h2h_home",
                "decimal_odds": 1.91,
                "odds_american": -110,
            }
        ]
    )
    sliced = source[source["league"].notna()]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.predict_batch(sliced)

    setting_with_copy = [
        w for w in caught if isinstance(w.message, pd.errors.SettingWithCopyWarning)
    ]
    assert not setting_with_copy
