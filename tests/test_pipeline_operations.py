"""SYNTHETIC operation lifecycle on the pipeline store (roadmap M1.2)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from app_core.pipeline.operations import (CANCELLED, COMPLETED, EXPIRED, FAILED, READY, RUNNING,
                                          AWAITING_REVIEW, OperationError, Operations)
from app_core.pipeline.store import Store, StoreError, digest

KEY = b'SYNTHETIC-pipeline-operations-test-key-not-production'
OWNER = 'SYNTHETIC owner'
T0 = datetime(2026, 11, 8, 17, 0, tzinfo=timezone.utc)
SUBJECT = {'event': 'SYNTHETIC-NFL-1', 'market': 'spread', 'line': -2.5}


class Clock:
    def __init__(self):
        self.value = T0

    def __call__(self):
        return self.value


class Work:
    """Counts calls so every test can prove the work ran at most once."""
    def __init__(self, output=None, error=None):
        self.calls, self.output, self.error = 0, output or {'p': 0.53}, error

    def __call__(self, operation):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.output, subject=operation['subject'])


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    path = tmp_path / 'SYNTHETIC-ops.sqlite'

    def ops():  # a new Operations/Store pair, as after a restart
        return Operations(Store.sqlite(path, KEY, owner=OWNER, clock=lambda: clock().isoformat()), clock=clock)
    return ops, clock


def deadline(minutes=20):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def reviewed(ops, op_id, subject=SUBJECT):
    return ops.review('NFL', op_id, reviewer='SYNTHETIC reviewer', subject_sha256=digest(subject),
                      findings={'event_mapping': 'Verified against SYNTHETIC schedule'},
                      attestation='SYNTHETIC attestation for this exact subject')


def test_review_then_run_completes_once_and_result_reopens_after_restart(env):
    make, _ = env
    ops = make()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline())
    assert ops.status('NFL', op['id'])['state'] == AWAITING_REVIEW
    with pytest.raises(OperationError, match='OPERATION_NOT_READY:AWAITING_REVIEW'):
        ops.run('NFL', op['id'], Work())
    assert reviewed(ops, op['id'])['state'] == READY
    work = Work()
    stored = ops.run('NFL', op['id'], work)
    assert work.calls == 1 and stored['output']['p'] == 0.53
    after_restart = make()
    assert after_restart.status('NFL', op['id'])['state'] == COMPLETED
    assert after_restart.result('NFL', op['id']) == stored  # rerender: no recompute
    with pytest.raises(OperationError, match='OPERATION_NOT_READY:COMPLETED'):
        after_restart.run('NFL', op['id'], work)
    assert work.calls == 1
    assert [e['state'] for e in after_restart.status('NFL', op['id'])['events']] == [READY, RUNNING, COMPLETED]


def test_duplicate_clicks_run_the_work_exactly_once(env):
    make, _ = env
    op = make().create('NFL', 'acquisition', SUBJECT, not_after=deadline(), review_required=False)
    work = Work()

    def click(_):
        try:
            make().run('NFL', op['id'], work)
            return 'ran'
        except StoreError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(click, range(12)))
    assert outcomes.count('ran') == 1 and work.calls == 1
    assert set(outcomes) - {'ran'} <= {'PIPELINE_OPERATION_ALREADY_SPENT_NO_RETRY', 'OPERATION_NOT_READY:RUNNING',
                                       'OPERATION_NOT_READY:COMPLETED', 'OPERATION_CONCURRENT_TRANSITION'}
    assert make().status('NFL', op['id'])['state'] == COMPLETED


def test_duplicate_create_returns_the_same_operation(env):
    make, clock = env
    first = make().create('NFL', 'calculation', SUBJECT, not_after=deadline())
    clock.value = T0 + timedelta(minutes=1)
    assert make().create('NFL', 'calculation', SUBJECT, not_after=deadline(30)) == first


def test_failure_is_recorded_with_a_safe_code_and_needs_an_explicit_successor(env):
    make, _ = env
    ops = make()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(), review_required=False)
    with pytest.raises(RuntimeError):
        ops.run('NFL', op['id'], Work(error=RuntimeError('provider said: key=SECRET-VALUE')))
    status = ops.status('NFL', op['id'])
    assert status['state'] == FAILED and status['events'][-1]['reason'] == 'OPERATION_EXCEPTION:RuntimeError'
    assert 'SECRET' not in repr(status)
    with pytest.raises(OperationError, match='OPERATION_NOT_READY:FAILED'):
        ops.run('NFL', op['id'], Work())
    with pytest.raises(OperationError, match='SUBJECT_CONFLICT'):
        ops.create('NFL', 'calculation', dict(SUBJECT, line=-3.5), not_after=deadline(), predecessor=op['id'])
    successor = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(), predecessor=op['id'], review_required=False)
    assert successor['id'] != op['id'] and successor['payload']['predecessor'] == op['id']
    assert ops.run('NFL', successor['id'], Work())['output']['p'] == 0.53
    assert ops.status('NFL', op['id'])['state'] == FAILED  # predecessor preserved


def test_open_operation_cannot_have_a_successor(env):
    ops = env[0]()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline())
    with pytest.raises(OperationError, match='PREDECESSOR_STILL_OPEN'):
        ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(), predecessor=op['id'])


def test_expired_operation_is_never_run_and_records_expiry(env):
    make, clock = env
    ops = make()
    op = ops.create('NFL', 'acquisition', SUBJECT, not_after=deadline(5), review_required=False)
    clock.value = T0 + timedelta(minutes=5)
    work = Work()
    with pytest.raises(OperationError, match='OPERATION_EXPIRED'):
        ops.run('NFL', op['id'], work)
    assert work.calls == 0 and ops.status('NFL', op['id'])['state'] == EXPIRED
    assert not ops.store.spent('NFL', 'acquisition-claim:' + op['id'])
    with pytest.raises(OperationError, match='ALREADY_EXPIRED'):
        ops.create('NFL', 'acquisition', dict(SUBJECT, line=-1.5), not_after=deadline(5))


def test_review_after_expiry_is_refused(env):
    make, clock = env
    ops = make()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(5))
    clock.value = T0 + timedelta(minutes=6)
    with pytest.raises(OperationError, match='OPERATION_EXPIRED'):
        reviewed(ops, op['id'])
    assert ops.reconcile('NFL', op['id'])['state'] == EXPIRED


def test_crash_during_run_becomes_unknown_failure_and_is_never_retried(env):
    make, clock = env
    ops = make()
    op = ops.create('NFL', 'acquisition', SUBJECT, not_after=deadline(), review_required=False)

    class Crash(BaseException):
        """Simulates the process dying mid-request (not an ordinary exception)."""
    with pytest.raises(Crash):
        ops.run('NFL', op['id'], Work(error=Crash()))
    restarted = make()
    assert restarted.status('NFL', op['id'])['state'] == RUNNING
    assert restarted.reconcile('NFL', op['id'])['state'] == RUNNING  # still inside its window
    clock.value = T0 + timedelta(minutes=21)
    status = restarted.reconcile('NFL', op['id'])
    assert status['state'] == FAILED and status['events'][-1]['reason'] == 'INTERRUPTED_OUTCOME_UNKNOWN'
    with pytest.raises(OperationError, match='OPERATION_NOT_READY:FAILED'):
        restarted.run('NFL', op['id'], Work())


def test_acquisition_and_calculation_have_separate_claims(env):
    ops = env[0]()
    acquisition = ops.create('NFL', 'acquisition', SUBJECT, not_after=deadline(), review_required=False)
    calculation = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(), review_required=False)
    assert acquisition['id'] != calculation['id']
    ops.run('NFL', acquisition['id'], Work())
    assert ops.store.spent('NFL', 'acquisition-claim:' + acquisition['id'])
    assert not ops.store.spent('NFL', 'calculation-claim:' + calculation['id'])
    assert ops.run('NFL', calculation['id'], Work(output={'p': 0.6}))['output']['p'] == 0.6


def test_cancel_only_before_running(env):
    ops = env[0]()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline())
    with pytest.raises(OperationError, match='REASON_CODE_REQUIRED'):
        ops.cancel('NFL', op['id'], reason='owner changed mind')
    assert ops.cancel('NFL', op['id'], reason='OWNER_CANCELLED')['state'] == CANCELLED
    with pytest.raises(OperationError, match='TRANSITION_INVALID:CANCELLED->READY'):
        reviewed(ops, op['id'])


def test_review_must_bind_the_exact_subject_and_real_findings(env):
    ops = env[0]()
    op = ops.create('NFL', 'calculation', SUBJECT, not_after=deadline())
    with pytest.raises(OperationError, match='REVIEW_SUBJECT_CONFLICT'):
        reviewed(ops, op['id'], subject=dict(SUBJECT, line=-3.5))
    with pytest.raises(OperationError, match='FINDINGS_REQUIRED'):
        ops.review('NFL', op['id'], reviewer='SYNTHETIC reviewer', subject_sha256=digest(SUBJECT),
                   findings={}, attestation='x')
    assert ops.status('NFL', op['id'])['state'] == AWAITING_REVIEW


def test_mixed_batch_two_sports_one_failure_one_stale(env):
    make, clock = env
    ops = make()
    batch = {
        'nfl_ok': ops.create('NFL', 'calculation', SUBJECT, not_after=deadline(), review_required=False),
        'nba_ok': ops.create('NBA', 'calculation', {'event': 'SYNTHETIC-NBA-1', 'market': 'total', 'line': 221.5},
                             not_after=deadline(), review_required=False),
        'nba_fail': ops.create('NBA', 'calculation', {'event': 'SYNTHETIC-NBA-2', 'market': 'total', 'line': 219.5},
                               not_after=deadline(), review_required=False),
        'nfl_stale': ops.create('NFL', 'acquisition', dict(SUBJECT, event='SYNTHETIC-NFL-2'),
                                not_after=deadline(2), review_required=False),
    }
    sports = {'nfl_ok': 'NFL', 'nba_ok': 'NBA', 'nba_fail': 'NBA', 'nfl_stale': 'NFL'}
    clock.value = T0 + timedelta(minutes=3)
    outcome = {}
    for name, op in batch.items():
        work = Work(error=ValueError('MODEL_INPUT_MISSING')) if name == 'nba_fail' else Work()
        try:
            make().run(sports[name], op['id'], work)
            outcome[name] = 'ran'
        except (OperationError, ValueError) as exc:
            outcome[name] = str(exc)
    assert outcome == {'nfl_ok': 'ran', 'nba_ok': 'ran', 'nba_fail': 'MODEL_INPUT_MISSING', 'nfl_stale': 'OPERATION_EXPIRED'}
    states = {name: make().status(sports[name], op['id'])['state'] for name, op in batch.items()}
    assert states == {'nfl_ok': COMPLETED, 'nba_ok': COMPLETED, 'nba_fail': FAILED, 'nfl_stale': EXPIRED}
    assert make().status('NBA', batch['nba_fail']['id'])['events'][-1]['reason'] == 'MODEL_INPUT_MISSING'
