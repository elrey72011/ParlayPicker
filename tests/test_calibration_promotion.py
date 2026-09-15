import pandas as pd

from scripts.fit_calibration import validate_calibration_promotion


def test_calibration_promotion_uses_future_holdout_and_beats_baselines():
    dates = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "slate_date": dates,
            "prob": ([0.20, 0.80] * 100),
            "win": ([0, 1] * 100),
        }
    )

    result = validate_calibration_promotion(frame, min_train_rows=100)

    assert result["promotable"] is True
    assert result["train_rows"] == 160
    assert result["test_rows"] == 40
    assert result["train_end"] < result["test_start"]


def test_calibration_promotion_rejects_missing_dates():
    frame = pd.DataFrame({"prob": [0.6, 0.4], "win": [1, 0]})
    result = validate_calibration_promotion(frame, min_train_rows=1)
    assert result["promotable"] is False


def test_calibration_promotion_rejects_insufficient_training_history():
    frame = pd.DataFrame(
        {
            "slate_date": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
            "prob": [0.6, 0.4] * 5,
            "win": [1, 0] * 5,
        }
    )

    result = validate_calibration_promotion(frame, min_train_rows=100)

    assert result["promotable"] is False
    assert "training rows" in result["reason"]



def provenance_artifact(path):
    """Synthetic immutable artifact, never a deployed validation claim."""
    from core.probability_calibration import calibration_digest, save_calibration
    knots = [[.2,.25],[.8,.75]]
    meta = dict(source='test-only dated fixture', calibration_trained_through='2026-09-01T00:00:00+00:00',
        calibration_available_at='2026-09-02T00:00:00+00:00',
        validation=dict(promotable=True,train_end='2026-08-01T00:00:00+00:00',test_start='2026-08-02T00:00:00+00:00'))
    meta['calibration_version'] = calibration_digest(dict(knots=knots,meta=meta))
    save_calibration(knots,path,meta)
    return meta


def test_final_calibration_fit_records_actual_cutoff_and_identity(tmp_path):
    import json
    from scripts.fit_calibration import main
    from core.probability_calibration import calibration_digest, calibration_provenance, load_calibration
    dates=pd.date_range('2025-01-01',periods=200,tz='UTC')
    pd.DataFrame(dict(game_date=dates,effective_win_probability=[.2,.8]*100,
        **{'W/L':['LOSS','WIN']*100})).to_csv(tmp_path/'graded.csv',index=False)
    out=tmp_path/'artifact.json'
    before=pd.Timestamp.now(tz='UTC')
    assert main([str(tmp_path),str(out)]) == 0
    after=pd.Timestamp.now(tz='UTC')
    payload=json.loads(out.read_text());meta=payload['meta']
    assert meta['calibration_version'] == calibration_digest(payload)
    assert pd.Timestamp(meta['calibration_trained_through']) == dates.max()
    assert pd.Timestamp(meta['calibration_trained_through']) > pd.Timestamp(meta['validation']['train_end'])
    assert before <= pd.Timestamp(meta['calibration_available_at']) <= after
    assert meta['validation']['promotable'] is True and meta['source']
    assert calibration_provenance(load_calibration(out))['calibration_version'] == meta['calibration_version']


def test_provenance_requires_exact_approved_artifact(tmp_path,monkeypatch):
    import json
    from core import probability_calibration as pc
    out=tmp_path/'calibration.json';meta=provenance_artifact(out)
    monkeypatch.setattr(pc,'DEFAULT_CALIBRATION_PATH',out)
    table=pc.load_calibration()
    assert pc.calibration_provenance(table)['calibration_version'] == meta['calibration_version']
    assert pc.apply_calibration(pd.Series([.4,.6]),table).equals(pc.apply_calibration(pd.Series([.4,.6]),list(table)))
    table[0][1]=.9
    assert pc.calibration_provenance(table) == {}
    for field,value in [('calibration_available_at','2099-01-01T00:00:00Z'),
                        ('calibration_trained_through','invalid'),('calibration_version','corrupt')]:
        provenance_artifact(out);payload=json.loads(out.read_text());payload['meta'][field]=value
        if field!='calibration_version':payload['meta']['calibration_version']=pc.calibration_digest(payload)
        out.write_text(json.dumps(payload))
        assert pc.calibration_provenance(pc.load_calibration()) == {}
    provenance_artifact(out);payload=json.loads(out.read_text());payload['meta']['validation']['promotable']=False
    payload['meta']['calibration_version']=pc.calibration_digest(payload);out.write_text(json.dumps(payload))
    assert pc.load_calibration() is None
    assert pc.calibration_provenance(pc.load_calibration(out)) == {}
    assert pc.calibration_provenance([[.2,.3],[.8,.7]]) == {}


def test_missing_provenance_and_overlap_are_not_authority(tmp_path,monkeypatch):
    import json
    from core import probability_calibration as pc
    out=tmp_path/'calibration.json';monkeypatch.setattr(pc,'DEFAULT_CALIBRATION_PATH',out)
    provenance_artifact(out);payload=json.loads(out.read_text())
    payload['meta'].pop('calibration_available_at');payload['meta']['calibration_version']=pc.calibration_digest(payload)
    out.write_text(json.dumps(payload))
    assert pc.calibration_provenance(pc.load_calibration()) == {}
    payload['meta']['validation']['test_start']=payload['meta']['validation']['train_end']
    out.write_text(json.dumps(payload))
    assert pc.load_calibration() is None
