from copy import deepcopy
from app.ui.lock_picks import total_input_review, total_input_coverage


def test_saved_warning_labels_and_legacy_are_not_inferred():
    leg = dict(sport='MLB', market='total_over', game='A at B', pick='Over 8.5',
               total_input_version='mlb-total-inputs-v1', total_input_status='DEGRADED',
               total_input_reason_codes='missing_theover|stale_empirical_evidence')
    rows = [dict(legs=[leg]), dict(legs=[dict(sport='MLB', market='total_under', pick='Under 9')]),
            dict(legs=[{**leg, 'market': 'spread_home'}]),
            dict(legs=[{**leg, 'total_input_status': 'COMPLETE'}])]
    before = deepcopy(rows)
    result = total_input_review(rows)
    assert len(result) == 2
    assert result[0]['Recorded input warnings'] == 'Empirical evidence stale'
    assert result[1]['Saved total-input status'] == 'Not recorded'
    assert rows == before


def test_optional_coverage_does_not_hide_other_blockers_or_mutate_records():
    leg = dict(sport='MLB', market='total_under', game='A at B', pick='Under 8.5',
               total_input_version='mlb-total-inputs-v1', total_input_status='DEGRADED',
               total_input_reason_codes='missing_theover')
    rows = [dict(legs=[leg])]
    before = deepcopy(rows)
    assert total_input_review(rows) == []
    assert len(total_input_coverage(rows)) == 1
    assert rows == before
    for reason in ('stale_empirical_evidence', 'missing_target_model', 'degraded_feature_subset'):
        changed = [dict(legs=[dict(leg, total_input_reason_codes='missing_theover|' + reason)])]
        assert len(total_input_review(changed)) == 1
        assert len(total_input_coverage(changed)) == 1
    assert total_input_coverage([dict(legs=[dict(sport='MLB', market='total_under')])]) == []
