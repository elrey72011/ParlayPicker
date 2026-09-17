import json
from copy import deepcopy
import pandas as pd
import pytest
from app_core import mlb_spread_total_model as m
from test_mlb_spread_total_training import trained
from test_mlb_spread_total_model import future_receipt


def candidate(trained):
    receipt, now = future_receipt(m.load_mlb_spread_total_model(trained))
    p = receipt["payload"]
    p["quote"].update(provider_namespace="odds_api", provider_event_id="odds-1", sportsbook="Novig", decimal_odds=2.0)
    receipt["sha256"] = m.digest(p)
    row = {k: p[k] for k in ("home_team_id", "away_team_id", "game_start_utc")}
    row.update(league="MLB", provider_namespace="odds_api", provider_event_id="odds-1",
               provider_ids={"mlb": p["provider_event_id"], "odds_api": "odds-1"},
               market_type="spread_home", spread_line=-1.5, quote_bookmaker="Novig", odds_american=100,
               mlb_pregame_receipts=json.dumps({"spread_home": receipt}),
               model_probability=.51, production_bet_amount=0)
    return row, now


def test_live_provider_bridge_supplies_model_prediction_without_promoting(trained):
    row, now = candidate(trained)
    result = m.attach_challenger(pd.DataFrame([row]), model_path=trained, now=now).iloc[0]
    assert result.mlb_challenger_status == "RESEARCH"
    predicted = json.loads(result.mlb_challenger_result)
    assert 0 < predicted["probability"] < 1
    assert predicted["input_receipt"]["sha256"] == predicted["receipt_hash"]
    assert predicted["production_bet_amount"] == 0 and not predicted["production_eligible"]
    assert result.model_probability == .51 and result.production_bet_amount == 0
    assert "model_version" not in result


@pytest.mark.parametrize("field,value", [
    ("provider_ids", {"mlb": "wrong", "odds_api": "odds-1"}),
    ("provider_event_id", "other"), ("home_team_id", "wrong"),
    ("quote_bookmaker", "FanDuel"), ("odds_american", -110),
    ("spread_line", -2.5), ("market_type", "moneyline_home"),
    ("game_start_utc", "2000-01-01T00:00:00Z")])
def test_mismatched_live_candidate_fails_closed(trained, field, value):
    row, now = candidate(trained); row[field] = value
    result = m.attach_challenger(pd.DataFrame([row]), model_path=trained, now=now).iloc[0]
    assert result.mlb_challenger_status == "PREGAME_EVIDENCE_UNAVAILABLE"
    assert result.mlb_challenger_result is None
    assert result.model_probability == .51


def test_stale_receipt_fails_closed(trained):
    from datetime import timedelta
    row, now = candidate(trained)
    result = m.attach_challenger(pd.DataFrame([row]), model_path=trained, now=now+timedelta(minutes=31)).iloc[0]
    assert result.mlb_challenger_status == "PREGAME_EVIDENCE_UNAVAILABLE"
