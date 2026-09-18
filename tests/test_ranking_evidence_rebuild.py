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

def test_producer_requirements_separates_configuration_from_settlement():
    original = row(mlb_challenger_status='NOT_CONFIGURED',
        exclusion_reasons=['missing_model_version','missing_calibration_version','missing_outcome'])
    report = build([], [original])
    needs = report['latest_producer_health']['MLB']['producer_requirements']
    assert needs['counts'] == {'model_artifact_not_configured':1,
                              'matching_calibration_provenance_required':1,
                              'verified_settlement_required':1}
    assert report['eligible_games'] == 0
    assert report['sports'] == {}


def test_challenger_provenance_is_never_used_as_baseline_authority():
    original = row(mlb_challenger_status='RESEARCH', mlb_challenger_result={'model_version':'challenger'},
                   exclusion_reasons=['missing_model_version'])
    report = build([], [original])
    assert report['latest_producer_health']['MLB']['producer_requirements']['counts'] == {'production_model_provenance_missing':1}
    assert report['eligible_games'] == 0


def test_projection_preserves_producer_facts_without_inventing_missing_facts():
    import pandas as pd
    from app_core.candidate_evidence_schema import project, FIELDS
    source = dict(league='MLB', market_type='total_under', model_version='original-model',
                  model_trained_through='2026-09-01T00:00:00Z', calibration_version='original-calibration',
                  selection_policy_version='original-policy', identity_verified=False,
                  selection_probability_source='baseline', mlb_challenger_status='NOT_CONFIGURED')
    saved = project(pd.DataFrame([source])).iloc[0]
    for key in source:
        assert saved[key] == source[key]
    assert 'selection_probability_source' in FIELDS
    assert pd.isna(saved['conservative_probability'])
    assert pd.isna(saved['evidence_version'])
