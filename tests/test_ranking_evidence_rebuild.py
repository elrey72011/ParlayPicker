from core.ranking_evidence_rebuild import build, rebuild


def row(**kw):
    return dict(dict(sport='MLB',game_id='a',selected_as_best_pick=True,
                     prediction_generated_at='2026-09-15T15:00:00+00:00',
                     game_start_utc='2026-09-15T23:00:00+00:00',market_type='total_over',
                     consensus_agreement='Neutral',candidate_outcome='WIN'),**kw)


def test_latest_selection_without_outcome_selection_or_duplicate_weight():
    earlier=row(prediction_generated_at='2026-09-15T14:00:00+00:00')
    later=row(candidate_outcome='LOSS')
    result=build([earlier,later,later])
    assert result['eligible_games']==1
    assert result['sports']['MLB']['overall']['win_rate']==0
    assert result['sports']['MLB']['meta']['recency_anchor']=='2026-09-15'


def test_ambiguous_selection_missing_consensus_and_unsettled_fail_closed():
    assert build([row(),row(market_type='total_under')])['eligible_games']==0
    assert build([row(consensus_agreement=None)])['exclusions']['missing_saved_consensus']==1
    assert build([row(candidate_outcome='PUSH')])['eligible_games']==0


def test_sport_isolation_and_no_activation():
    report=build([row(),row(sport='NFL',candidate_outcome='LOSS')])
    assert report['status']=='CANDIDATE_NOT_ACTIVATED'
    assert report['sports']['MLB']['overall']['win_rate']==1
    assert report['sports']['NFL']['overall']['win_rate']==0


def test_rebuild_failure_writes_error_not_healthy(monkeypatch,tmp_path):
    def fail(_):raise ValueError('private detail')
    monkeypatch.setattr('core.activation_validation.read_dataset',fail)
    path=tmp_path/'report.json'
    assert rebuild('unused',path)['status']=='ERROR'
    assert 'private detail' not in path.read_text()


def test_latest_producer_health_does_not_hide_legacy_exclusions_or_activate():
    old = row(prediction_generated_at='2026-09-14T15:00:00+00:00', exclusion_reasons=['missing_model_version'])
    new = row(exclusion_reasons=['missing_calibration_version'])
    report = build([], [old, new])
    assert report['exclusions']['missing_model_version'] == 1
    assert report['latest_producer_health']['MLB']['exclusions'] == {'missing_calibration_version': 1}
    assert report['eligible_games'] == 0
    assert report['status'] == 'BLOCKED_NO_ELIGIBLE_EVIDENCE'
