"""Qualification follows the existing prospective-evidence authority chain. SYNTHETIC readiness rows."""
from app_core.selection import qualification as q
from core.market_policy import SPORT_MARKET_FAMILIES


def fake(states):
    def readiness(path, sport, family):
        state, blocker = states.get((sport, family), ('UNVALIDATED', 'NO_PROSPECTIVE_EVENTS'))
        return dict(deployment_state=state, next_blocker=blocker)
    return readiness


def test_only_a_validated_deployment_qualifies_and_every_other_market_names_its_blocker():
    a = q.assess(readiness=fake({('NFL', 'SPREAD'): ('PROVISIONAL_VALIDATED', None),
                                 ('NBA', 'TOTAL'): ('UNVALIDATED', 'VALIDATION_SAMPLE_INSUFFICIENT')}))
    assert set(a) == {f'{s} {f}' for s, fs in SPORT_MARKET_FAMILIES.items() for f in fs}  # all 12, MLB RUN_LINE etc.
    assert a['NFL SPREAD'] == dict(qualified=True, deployment_state='PROVISIONAL_VALIDATED', next_blocker=None)
    assert a['NBA TOTAL']['next_blocker'] == 'VALIDATION_SAMPLE_INSUFFICIENT'
    assert q.qualified_set(a) == {('NFL', 'SPREAD')}


def test_unreadable_evidence_qualifies_nothing():
    def broken(path, sport, family):
        raise ValueError('SYNTHETIC unreadable')
    a = q.assess(readiness=broken)
    assert q.qualified_set(a) == frozenset()
    assert {v['next_blocker'] for v in a.values()} == {'EVIDENCE_UNREADABLE:ValueError'}


def test_an_absent_evidence_store_qualifies_nothing(tmp_path):
    a = q.assess(tmp_path / 'SYNTHETIC-absent.sqlite3')
    assert q.qualified_set(a) == frozenset()
    assert {v['next_blocker'] for v in a.values()} == {'NO_PROSPECTIVE_EVENTS'}
