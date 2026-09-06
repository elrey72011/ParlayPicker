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
