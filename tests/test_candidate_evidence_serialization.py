from datetime import date, datetime, timezone
import json
import numpy as np
import pandas as pd
import pytest
from app_core.candidate_evidence_schema import project, evidence_value, FIELDS
from core.exposure_ledger import digest


def test_live_timestamp_projection_has_deterministic_json_hash():
    row = dict(league='MLB', matchup_id='event-1', market_type='total_over',
               best_pick='Over 8.5', total_line=8.5, odds_american=-110,
               game_start_utc=pd.Timestamp('2026-09-15T19:00:00-04:00'),
               odds_recorded_at=pd.Timestamp('2026-09-15T18:45:00-04:00'),
               prediction_generated_at=datetime(2026,9,15,22,46,tzinfo=timezone.utc),
               game_date=date(2026,9,15), model_available_at=pd.NaT,
               model_trained_through=pd.NA, quote_binding_verified=np.bool_(True))
    source = pd.DataFrame([row])
    result = project(source).iloc[0].to_dict()
    assert result['game_start_utc'] == '2026-09-15T19:00:00-04:00'
    assert result['model_trained_through'] is None
    assert result['model_available_at'] is None
    assert result['event_date'] == '2026-09-15'
    assert result['payload_hash'] == digest({k: result[k] for k in FIELDS if k != 'payload_hash'})
    assert project(project(source)).iloc[0]['payload_hash'] == result['payload_hash']
    assert isinstance(source.iloc[0]['game_start_utc'], pd.Timestamp)


def test_naive_timestamp_is_not_given_a_timezone():
    from core.wager_decisions import aware
    value = evidence_value(pd.Timestamp('2026-09-15 12:00:00'))
    assert value == '2026-09-15T12:00:00'
    assert aware(value) is None


def test_nested_numpy_and_missing_scalars_are_json_safe():
    value = evidence_value({'items': [np.int64(2), np.float64(.5), np.bool_(False), pd.NA,
                                     np.datetime64('NaT'), np.float64('inf')]})
    assert json.loads(json.dumps(value, allow_nan=False)) == {'items': [2,.5,False,None,None,None]}
    with pytest.raises(TypeError):
        digest(evidence_value(object()))


def test_prepare_live_accepts_real_dataframe_timestamp(tmp_path):
    from core.prospective_uncertainty import prepare_live
    result = prepare_live(pd.DataFrame([{'league':'MLB','matchup_id':'one',
        'game_start_utc':pd.Timestamp('2026-09-15T23:00:00Z')}]),
        database=tmp_path/'evidence.sqlite3', plan_dir=tmp_path/'plans')
    assert result.iloc[0]['game_start_utc'] == '2026-09-15T23:00:00+00:00'
    assert pd.isna(result.iloc[0]['conservative_probability'])
