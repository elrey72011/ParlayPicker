from copy import deepcopy
from app.ui.lock_picks import total_input_review


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
    assert result[0]['Recorded input warnings'] == 'TheOver input missing; Empirical evidence stale'
    assert result[1]['Saved total-input status'] == 'Not recorded'
    assert rows == before
