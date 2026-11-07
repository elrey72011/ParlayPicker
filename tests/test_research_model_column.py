"""Research model probability column (app_core.research_model_column), on synthetic research models."""
import math

import pandas as pd

from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core import research_model_column as column
from tests.test_prospective_research_models import _at, _event, clock  # noqa: F401 (fixture)


def _store_with_prediction(tmp_path, clock):  # noqa: F811
    path = tmp_path / "evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    for market in ("SPREAD", "TOTAL"):
        assert models.train_scope(path, "NBA", market)["status"] == "FITTED_RESEARCH_ONLY"
    future = _event(path, clock, "NBA", 73)
    clock[0] = _at(73, 1)
    for market in ("SPREAD", "TOTAL"):
        assert models.predict_scope(path, "NBA", market)["predictions"] == 1
    event = pe.load_record(path, "prospective_event", future)
    stored = {m: next(p for p in pe.read_records(path, "prospective_prediction", sport="NBA", market_family=m)
                      if p["event_id"] == future) for m in ("SPREAD", "TOTAL")}
    return path, event, stored


def test_model_probability_for_each_side_and_line(tmp_path, clock):  # noqa: F811
    path, event, stored = _store_with_prediction(tmp_path, clock)
    predictions = column.load(path, now=_at(73, 1))
    assert len(predictions) == 2
    home, away = event["home_team"], event["away_team"]
    board = pd.DataFrame([
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "spread_home",
         "best_pick": f"{home} -2.5", "odds_american": -110, "game_start_utc": event["scheduled_start"]},
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "spread_away",
         "best_pick": f"{away} +2.5", "odds_american": -110, "game_start_utc": event["scheduled_start"]},
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "total_over",
         "best_pick": "Over 202.5", "odds_american": -110, "game_start_utc": event["scheduled_start"]},
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "total_under",
         "best_pick": "Under 202.5", "odds_american": -110, "game_start_utc": event["scheduled_start"]},
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "spread_home",
         "best_pick": f"{home} -4", "odds_american": 120, "game_start_utc": event["scheduled_start"]},
        {"league": "NCAAF", "home_team": home, "away_team": away, "market_type": "spread_home",
         "best_pick": f"{home} -2.5", "odds_american": -110, "game_start_utc": event["scheduled_start"]},
    ])
    out = column.attach(board, predictions)
    p = out["research_model_probability"]
    # The frozen predictions are home -2.5 and Over 202.5: the column reproduces them exactly.
    assert math.isclose(p[0], stored["SPREAD"]["mean_probability"], abs_tol=1e-9)
    assert math.isclose(p[2], stored["TOTAL"]["mean_probability"], abs_tol=1e-9)
    # The other side at the same half-point line is the complement.
    assert math.isclose(p[0] + p[1], 1, abs_tol=1e-9) and math.isclose(p[2] + p[3], 1, abs_tol=1e-9)
    # A whole-number line has a push; EV uses the candidate's own price.
    assert out["research_model_push_probability"][4] > 0
    loss = 1 - p[4] - out["research_model_push_probability"][4]
    assert math.isclose(out["research_model_ev"][4], p[4] * 1.2 - loss, abs_tol=1e-9)
    assert math.isclose(out["research_model_ev"][0], p[0] * (100 / 110) - (1 - p[0]), abs_tol=1e-9)
    # No research model for the sport: empty. No other column changes.
    assert pd.isna(p[5])
    pd.testing.assert_frame_equal(out[board.columns], board)
    assert out["research_model_predicted_at"][0] == stored["SPREAD"]["prediction_timestamp"]


def test_short_board_names_match_and_ambiguity_stays_empty():
    start = pd.Timestamp("2030-01-01T20:00:00Z").to_pydatetime()
    prediction = {"sport": "NBA", "family": "SPREAD", "home_team": "Oklahoma City Thunder",
                  "away_team": "Washington Wizards", "start": start, "mean": 9.0, "sd": 12.0,
                  "predicted_at": "2030-01-01T18:00:00+00:00"}
    board = pd.DataFrame([{"league": "NBA", "home_team": "Oklahoma City", "away_team": "Washington",
                           "market_type": "spread_home", "best_pick": "Oklahoma City -11.5",
                           "odds_american": -108, "game_start_utc": "2030-01-01T20:10:00Z"}])
    assert column.attach(board, [prediction])["research_model_probability"].notna().all()
    assert column.attach(board, [prediction, dict(prediction)])["research_model_probability"].isna().all()
    later = dict(prediction, start=start + pd.Timedelta(hours=6).to_pytimedelta())
    assert column.attach(board, [later])["research_model_probability"].isna().all()


def test_apply_is_off_by_setting_and_never_raises():
    board = pd.DataFrame([{"league": "NBA", "market_type": "spread_home", "best_pick": "A -2.5"}])
    out, diagnostic = column.apply(board, setting=lambda k, d=None: "false")
    assert diagnostic["status"] == "OFF" and out["research_model_probability"].isna().all()

    def broken(*args, **kwargs):
        raise FileNotFoundError("no store yet")
    out, diagnostic = column.apply(board, setting=lambda k, d=None: d, refresh=broken, read=broken)
    assert diagnostic["status"] == "NO_MATCHING_PREDICTION"
    assert diagnostic["store"]["status"] == "ERROR" and diagnostic["error"] == "FileNotFoundError"
    pd.testing.assert_frame_equal(out[board.columns], board)


def test_game_board_shows_model_probability_and_ev_with_push():
    from app.ui.daily_dashboard import research_model_shadow
    result = pd.DataFrame({"matchup_id": ["g1", "g2", "g3"], "market_type": ["spread_home", "total_over", "spread_home"],
                           "pick": ["A -4", "Over 210.5", "C -1.5"], "odds": [120, -110, -110]})
    candidates = pd.DataFrame({"matchup_id": ["g1", "g2"], "market_type": ["spread_home", "total_over"],
                               "best_pick": ["A -4", "Over 210.5"],
                               "research_model_probability": [0.45, 0.40],
                               "research_model_push_probability": [0.05, 0.0]})
    probs, ev = research_model_shadow(result, candidates)
    assert math.isclose(probs[0], 0.45) and math.isclose(ev[0], 0.45 * 1.2 - 0.50)
    assert math.isclose(ev[1], 0.40 * (100 / 110) - 0.60)  # negative EV is kept
    assert pd.isna(probs[2]) and pd.isna(ev[2])
    assert research_model_shadow(result, candidates.drop(columns="research_model_probability")) is None


def test_best_pick_export_carries_research_model_columns():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    assert set(column.COLUMNS) <= set(BEST_PICK_COLUMNS) & set(REQUIRED_BEST_PICK_EXPORT_COLUMNS)


def test_status_says_why_each_row_has_no_probability(tmp_path, clock):  # noqa: F811
    path, event, _ = _store_with_prediction(tmp_path, clock)
    home, away, start = event["home_team"], event["away_team"], event["scheduled_start"]
    board = pd.DataFrame([
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "spread_home",
         "best_pick": f"{home} -2.5", "odds_american": -110, "game_start_utc": start},
        {"league": "NCAAF", "home_team": home, "away_team": away, "market_type": "spread_home",
         "best_pick": f"{home} -2.5", "odds_american": -110, "game_start_utc": start},
        {"league": "NBA", "home_team": home, "away_team": away, "market_type": "moneyline_home",
         "best_pick": home, "odds_american": -110, "game_start_utc": start},
        {"league": "NBA", "home_team": "Elsewhere", "away_team": away, "market_type": "spread_home",
         "best_pick": "Elsewhere -2.5", "odds_american": -110, "game_start_utc": start},
    ])
    predictions = column.load(path, now=_at(73, 1))
    status = column.attach(board, predictions)["research_model_status"].tolist()
    assert status == ["ATTACHED", "NO_RESEARCH_MODEL_FOR_SPORT", "UNSUPPORTED_MARKET", "NO_PREDICTION_FOR_GAME"]
    assert column.attach(board, predictions + predictions)["research_model_status"][0] == "AMBIGUOUS_GAME_MATCH"


def test_store_wide_status_on_every_row():
    board = pd.DataFrame([{"league": "NBA", "market_type": "spread_home", "best_pick": "A -2.5"}] * 2)
    on = lambda k, d=None: d  # noqa: E731

    def missing(*args, **kwargs):
        raise FileNotFoundError("store")
    out, _ = column.apply(board, setting=on, refresh=lambda s: {"status": "MISSING_CREDENTIALS"}, read=missing)
    assert set(out["research_model_status"]) == {"STORE_MISSING_CREDENTIALS"}
    out, _ = column.apply(board, setting=on, read=missing,
                          refresh=lambda s: {"status": "IN_PROGRESS_BACKGROUND", "last": None})
    assert set(out["research_model_status"]) == {"STORE_NOT_DOWNLOADED_YET (IN_PROGRESS_BACKGROUND)"}
    failed = {"status": "STARTED_IN_BACKGROUND", "last": {"status": "REFRESH_FAILED", "error": "HttpError"}}
    out, _ = column.apply(board, setting=on, refresh=lambda s: failed, read=lambda *a, **k: [])
    assert set(out["research_model_status"]) == {
        "NO_CURRENT_PREDICTIONS_IN_STORE (STARTED_IN_BACKGROUND; last restore REFRESH_FAILED HttpError)"}
    out, _ = column.apply(board, setting=lambda k, d=None: "false")
    assert set(out["research_model_status"]) == {"OFF"}
