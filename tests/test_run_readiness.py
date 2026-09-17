import json

import pandas as pd
import pytest

from app_core import prediction_evidence as evidence
from core.run_readiness import build_readiness, render_readiness
from test_prediction_evidence import frozen, fixture_frames


def saved(frozen):
    context, db, _ = frozen
    audit, final = fixture_frames()
    return evidence.capture_run(context, audit, final, audit, path=db)


def test_complete_evidence_can_be_ready_without_a_wager(frozen):
    audit, final = saved(frozen)
    final['wager_approved'] = False
    final['Kelly_Bet_Size'] = 0
    final['production_win_probability'] = .596
    final['calibrated_probability'] = .66
    audit.loc[audit.best_available_selected, 'calibrated_probability'] = .66
    final['Status_Reason'] = 'Production probability below 60%'
    original = audit.copy(deep=True)
    report = build_readiness(audit, final)
    row = report['games'][0]
    assert row['readiness'] == 'ready_for_grading'
    assert row['wager_decision'] == 'pass'
    assert row['displayed_probability'] == .66
    assert row['production_probability'] == .596
    assert 'Production probability below 60%' in row['wager_reasons']
    pd.testing.assert_frame_equal(audit, original)
    assert report['production_changes'] is False
    json.dumps(report, allow_nan=False)


def test_bad_unselected_candidate_blocks_entire_evidence_pool(frozen):
    audit, final = saved(frozen)
    audit.loc[~audit.best_available_selected, 'quote_binding_verified'] = False
    report = build_readiness(audit, final)
    assert report['games'][0]['readiness'] == 'blocked'
    assert report['games'][0]['wager_decision'] == 'approved'
    assert report['candidates'][1]['issues'] == ['quote_binding_unverified']


def test_missing_card_or_conflicting_card_does_not_infer_approval(frozen):
    audit, final = saved(frozen)
    assert build_readiness(audit)['games'][0]['wager_decision'] == 'unknown'
    alternate = final.copy()
    alternate['wager_approved'] = False
    row = build_readiness(audit, pd.concat([final, alternate]))['games'][0]
    assert row['wager_decision'] == 'unknown'
    assert 'final_decision_missing_or_ambiguous' in row['evidence_blockers']


def test_missing_production_probability_is_not_replaced_by_display(frozen):
    audit, final = saved(frozen)
    row = build_readiness(audit, final)['games'][0]
    assert row['displayed_probability'] == .6
    assert row['production_probability'] is None
    assert row['feature_timestamp'] is None
    assert 'feature_freshness_unavailable' in row['data_warnings']


def test_incomplete_and_conflicting_pools_are_reported(frozen):
    audit, final = saved(frozen)
    assert 'candidate_pool_incomplete' in build_readiness(audit.iloc[:1], final)['games'][0]['evidence_blockers']
    alternate = audit.iloc[:1].copy()
    alternate['calibrated_probability'] = .9
    row = build_readiness(pd.concat([audit, alternate]), final)['games'][0]
    assert 'conflicting_candidate_records' in row['evidence_blockers']


def test_timing_and_quote_age_use_capture_not_wall_clock(frozen):
    audit, final = saved(frozen)
    row = build_readiness(audit, final)['games'][0]
    assert row['after_freeze_day'] is True
    assert row['earliest_evaluation_slate'] == '2026-09-02'
    assert 'quote_age_above_diagnostic_limit' in row['data_warnings']
    audit['game_start_utc'] = '2026-09-03T14:00:00Z'
    audit['odds_recorded_at'] = '2026-09-03T16:00:00Z'
    row = build_readiness(audit, final)['games'][0]
    assert 'not_pregame_at_capture' in row['evidence_blockers']
    assert 'quote_after_prediction' in row['evidence_blockers']


def test_scores_do_not_change_readiness(frozen):
    audit, final = saved(frozen)
    before = build_readiness(audit, final)
    audit['candidate_outcome'] = ['WIN', 'LOSS']
    audit['actual_home_score'] = 100
    assert build_readiness(audit, final) == before


def test_repeated_exports_are_deduplicated_and_runs_stay_separate(frozen):
    audit, final = saved(frozen)
    assert len(build_readiness(pd.concat([audit, audit]), final)['games']) == 1
    second = audit.copy()
    second['snapshot_id'] = 'another-snapshot'
    second['export_run_id'] = '20260903T160000Z'
    assert len(build_readiness(pd.concat([audit, second]), final)['games']) == 2


def test_empty_and_invalid_inputs_are_explicit():
    assert build_readiness(None)['status'] == 'no_candidate_evidence'
    with pytest.raises(ValueError):
        build_readiness(None, quote_warning_minutes=float('nan'))
    report = build_readiness(pd.DataFrame([{'best_pick': 'A'}, {'best_pick': 'B'}]))
    assert len(report['games']) == 2
    assert all(row['readiness'] == 'blocked' for row in report['games'])
    assert 'Read-only diagnostics' in render_readiness(report)


def test_mismatched_export_values_are_visible(frozen):
    audit, final = saved(frozen)
    final['calibrated_probability'] = .99
    final['odds_american'] = -120
    row = build_readiness(audit, final)['games'][0]
    assert 'final_probability_mismatch' in row['evidence_blockers']
    assert 'final_price_mismatch' in row['evidence_blockers']


def test_verified_quote_does_not_override_rejected_final_line(frozen):
    audit, final = saved(frozen)
    audit.loc[audit.best_available_selected, "best_pick"] = "Total line unresolved"
    final["best_pick"] = "Total line unresolved"
    report = build_readiness(audit, final)
    candidate = report["candidates"][0]
    assert candidate["quote_verified"] and not candidate["line_eligible"]
    assert "final_line_rejected" in report["games"][0]["evidence_blockers"]


def test_push_rows_require_explicit_push_mass(frozen):
    audit, final = saved(frozen)
    audit["total_line"] = 8
    audit["probability_semantics"] = "win_unconditional_with_push"
    audit["push_probability"] = .1
    audit["market_push_probability"] = .1
    assert build_readiness(audit, final)["games"][0]["readiness"] == "ready_for_grading"
    assert build_readiness(audit, final)["candidates"][0]["settlement_rule"] == "push_on_equal"
    audit.loc[~audit.best_available_selected, "market_push_probability"] = None
    assert "probability_semantics_unverified" in build_readiness(audit, final)["games"][0]["evidence_blockers"]


def test_unselected_missing_or_rejected_lines_are_ineligible(frozen):
    audit, final = saved(frozen)
    for changes in (
        {"total_line": None},
        {"total_line": float("inf")},
        {"odds_source": "rejected_live_orientation"},
        {"line_source": "rejected_live_spread_price"},
        {"best_pick": "Away (No Line)"},
        {"line_consistency_flag": False},
        {"line_event_identity_match_flag": False},
    ):
        case = audit.copy()
        for key, value in changes.items():
            case.loc[~case.best_available_selected, key] = value
        report = build_readiness(case, final)
        candidate = next(c for c in report["candidates"] if not c["selected"])
        assert not candidate["line_eligible"], changes
        assert "final_line_rejected" in candidate["issues"]
        assert report["games"][0]["readiness"] == "blocked"


def test_moneyline_without_point_is_not_a_missing_spread(frozen):
    audit, final = saved(frozen)
    audit.loc[~audit.best_available_selected, "market_type"] = "moneyline_away"
    audit.loc[~audit.best_available_selected, "total_line"] = None
    report = build_readiness(audit, final)
    candidate = next(c for c in report["candidates"] if not c["selected"])
    assert candidate["line_eligible"]


def test_structured_wager_metadata_dedup_preserves_conflicts(frozen):
    audit, final = saved(frozen)
    audit['metadata'] = [{'teams': ['away', 'home'], 'flags': {'fresh': True}} for _ in range(len(audit))]
    final['wager_contract'] = [{'production_eligible': False, 'reasons': ['UNVALIDATED']} for _ in range(len(final))]
    original = final.copy(deep=True)
    baseline = build_readiness(audit, final)
    repeated = build_readiness(pd.concat([audit, audit]), pd.concat([final, final]))
    assert repeated == baseline
    pd.testing.assert_frame_equal(final, original)
    changed = final.copy(deep=True)
    changed['wager_contract'] = [{'production_eligible': True, 'reasons': []} for _ in range(len(changed))]
    result = build_readiness(audit, pd.concat([final, changed]))
    assert 'final_decision_missing_or_ambiguous' in result['games'][0]['evidence_blockers']


@pytest.mark.parametrize('change,expected', [
    ({},None),
    ({'model_trained_through':'invalid'},'model_provenance_missing'),
    ({'model_available_at':'2026-09-04T00:00:00Z'},'model_provenance_timing_invalid'),
    ({'model_trained_through':'2026-09-04T00:00:00Z'},'model_provenance_timing_invalid'),
])
def test_authoritative_provenance_readiness_uses_only_supplied_facts(frozen,change,expected):
    context,db,_=frozen
    audit,final=fixture_frames()
    audit['candidate_id']=['over','under'];final['candidate_id']=['over']
    facts=dict(model_version='synthetic-trained-artifact-sha256',model_trained_through='2026-09-01T00:00:00Z',
        model_available_at='2026-09-02T00:00:00Z',training_cutoff_basis='max game_start_utc in test-only data')
    facts.update(change)
    for field,value in facts.items():audit[field]=value
    evidence.capture_run(context,audit,final,audit,path=db,authoritative_candidates=True)
    _,saved,card=evidence.load_snapshots(db)[0]
    blocks=build_readiness(saved,card)['games'][0]['evidence_blockers']
    if expected:assert expected in blocks
    else:assert 'model_provenance_missing' not in blocks
    for field,value in facts.items():assert saved[field].eq(value).all()
    assert saved.calibration_version.isna().all()
    assert saved.calibration_available_at.isna().all()
    from core.activation_validation import reasons
    assert 'missing_calibration_version' in reasons(saved.iloc[0].to_dict())
    assert 'calibration_not_available' in reasons(saved.iloc[0].to_dict())


def test_missing_authoritative_model_calibration_is_not_bundle_provenance(frozen):
    context,db,_=frozen
    audit,final=fixture_frames()
    audit['candidate_id']=['over','under'];final['candidate_id']=['over']
    evidence.capture_run(context,audit,final,audit,path=db,authoritative_candidates=True)
    _,saved,card=evidence.load_snapshots(db)[0]
    assert saved.decision_bundle_version.eq(context['model_version']).all()
    for field in ('model_version','model_trained_through','model_available_at','calibration_version','calibration_available_at'):
        assert saved[field].isna().all()
    assert 'model_provenance_missing' in build_readiness(saved,card)['games'][0]['evidence_blockers']


def test_challenger_diagnostics_do_not_supply_missing_authority(frozen):
    audit, final = saved(frozen)
    audit['league'] = 'MLB'
    audit['mlb_challenger_status'] = 'RESEARCH'
    audit['mlb_challenger_result'] = json.dumps({
        'probability': .61, 'model_version': 'challenger-v1', 'receipt_hash': 'receipt-v1'})
    audit['model_version'] = ''
    report = build_readiness(audit, final, diagnostics={'mlb_receipt_health': {'receipts_created': 2}})
    assert report['counts']['games'] == 1
    assert report['mlb_receipt_health'] == {'receipts_created': 2}
    assert report['mlb_challenger_status_counts'] == {'RESEARCH': len(audit)}
    assert report['candidates'][0]['mlb_challenger_probability'] == .61
    assert report['candidates'][0]['mlb_challenger_receipt_hash'] == 'receipt-v1'
    assert 'model_provenance_missing' in report['games'][0]['evidence_blockers']
    assert build_readiness(audit, final)['mlb_receipt_health'] == {}
    json.dumps(report, allow_nan=False)


def test_live_dashboard_uses_authoritative_evidence():
    import ast
    from pathlib import Path
    tree = ast.parse(Path('streamlit_app.py').read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'render_readiness_dashboard']
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[0].args[0]) == 'candidate_authority_df'
