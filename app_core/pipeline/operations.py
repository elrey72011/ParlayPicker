"""Durable operation state machine on the signed pipeline store (roadmap M1.2).

An *operation* is one bounded unit of work for one sport: a provider request
(``acquisition``) or a model run (``calculation``). Its life is recorded as an
append-only chain of signed events:

    AWAITING_REVIEW ──review──► READY ──run──► RUNNING ──► COMPLETED
          │                       │                 └────► FAILED
          └──────cancel / expire──┴──► CANCELLED / EXPIRED

Rules:
- ``run`` spends a one-use claim, keyed by kind and operation, *before* calling
  the work function. A duplicate click, rerender, logout or restart therefore
  cannot repeat a provider request or a calculation. Nothing resets a claim.
- Reading status or a result never calls the work function.
- After ``not_after`` an open operation can only become EXPIRED. An operation
  left RUNNING by a crash becomes FAILED(``INTERRUPTED_OUTCOME_UNKNOWN``) through
  ``reconcile`` and is never retried in place.
- A retry is an explicit *successor*: a new operation for the same subject that
  names a FAILED, EXPIRED or CANCELLED predecessor. The predecessor's records
  stay unchanged.
- A review records exactly the reviewer's own findings, bound to the
  operation's subject digest. Nothing here invents a finding.
- Failure details are stored as developer-authored codes, never as exception
  messages, which could contain secrets.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re

from app_core.pipeline.store import Store, StoreError, digest

AWAITING_REVIEW, READY, RUNNING = 'AWAITING_REVIEW', 'READY', 'RUNNING'
COMPLETED, FAILED, EXPIRED, CANCELLED = 'COMPLETED', 'FAILED', 'EXPIRED', 'CANCELLED'
TERMINAL = frozenset({COMPLETED, FAILED, EXPIRED, CANCELLED})
RETRYABLE_PREDECESSOR = frozenset({FAILED, EXPIRED, CANCELLED})
KINDS = frozenset({'acquisition', 'calculation'})
TRANSITIONS = {
    AWAITING_REVIEW: {READY, CANCELLED, EXPIRED},
    READY: {RUNNING, CANCELLED, EXPIRED},
    RUNNING: {COMPLETED, FAILED},
}
MAX_EVENTS = 16
_CODE = re.compile(r'^[A-Z][A-Z0-9_:]{2,120}$')


class OperationError(StoreError):
    """Developer-authored operation error code."""


def require(ok, code):
    if not ok:
        raise OperationError(code)


def now():
    return datetime.now(timezone.utc)


def _instant(value):
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise OperationError('OPERATION_CLOCK_INVALID') from None
    require(moment.tzinfo is not None, 'OPERATION_CLOCK_INVALID')
    return moment


def failure_code(exc):
    """A safe code for a failure: the error's own code, or only its class name."""
    first = exc.args[0] if getattr(exc, 'args', None) else None
    if isinstance(exc, ValueError) and isinstance(first, str) and _CODE.match(first):
        return first
    return 'OPERATION_EXCEPTION:' + type(exc).__name__


class Operations:
    def __init__(self, store: Store, *, clock=now):
        require(isinstance(store, Store), 'OPERATION_STORE_REQUIRED')
        self.store, self.clock = store, clock

    # -- reading -----------------------------------------------------------
    def get(self, sport, operation_id):
        return self.store.get(sport, operation_id, kind='operation')

    def events(self, sport, operation_id):
        found = []
        for sequence in range(1, MAX_EVENTS + 1):
            try:
                event = self.store.get(sport, f'{operation_id}:event:{sequence}', kind='operation-event')
            except StoreError as exc:
                if str(exc) == 'PIPELINE_RECORD_MISSING':
                    return found
                raise
            payload = event['payload']
            require(payload['operation_id'] == operation_id and payload['sequence'] == sequence,
                    'OPERATION_EVENT_CONFLICT')
            found.append(payload)
        raise OperationError('OPERATION_EVENT_LIMIT')

    def status(self, sport, operation_id):
        """Current state, derived only from stored records."""
        operation = self.get(sport, operation_id)['payload']
        events = self.events(sport, operation_id)
        state = events[-1]['state'] if events else operation['initial_state']
        expired = state not in TERMINAL and state != RUNNING and self._past(operation)
        return dict(operation_id=operation_id, sport=sport, kind=operation['kind'], state=state,
                    expired_unrecorded=expired, events=events, not_after=operation['not_after'],
                    predecessor=operation['predecessor'])

    def result(self, sport, operation_id):
        """The stored output of a COMPLETED operation. Never recomputes."""
        status = self.status(sport, operation_id)
        require(status['state'] == COMPLETED, 'OPERATION_RESULT_UNAVAILABLE')
        claim = self.store.claim_record(sport, self._claim_id(status['kind'], operation_id))
        output_id = status['events'][-1]['output_id']
        require(claim['payload']['operation_id'] == operation_id, 'OPERATION_CLAIM_CONFLICT')
        return self.store.get(sport, output_id, kind='operation-output')['payload']

    # -- writing -----------------------------------------------------------
    def create(self, sport, kind, subject, *, not_after, review_required=True, predecessor=None):
        """Create an operation, or return the existing identical one (duplicate click)."""
        require(kind in KINDS, 'OPERATION_KIND_UNSUPPORTED')
        require(isinstance(subject, dict) and subject, 'OPERATION_SUBJECT_REQUIRED')
        require(_instant(not_after) > self.clock(), 'OPERATION_ALREADY_EXPIRED')
        if predecessor is not None:
            prior = self.status(sport, predecessor)
            prior_subject = self.get(sport, predecessor)['payload']['subject']
            require(prior['kind'] == kind and prior_subject == subject, 'OPERATION_SUCCESSOR_SUBJECT_CONFLICT')
            require(prior['state'] in RETRYABLE_PREDECESSOR, 'OPERATION_PREDECESSOR_STILL_OPEN')
        identity = 'operation:' + digest(dict(sport=sport, kind=kind, subject=subject, predecessor=predecessor))
        payload = dict(kind=kind, subject=subject, subject_sha256=digest(subject), not_after=not_after,
                       initial_state=AWAITING_REVIEW if review_required else READY, predecessor=predecessor)
        try:
            return self.store.get(sport, identity, kind='operation')
        except StoreError as exc:
            if str(exc) != 'PIPELINE_RECORD_MISSING':
                raise
        try:
            return self.store.put(sport, 'operation', payload, identity=identity)
        except StoreError as exc:
            if str(exc) != 'PIPELINE_RECORD_CONFLICT':
                raise
            return self.store.get(sport, identity, kind='operation')  # a concurrent click won

    def review(self, sport, operation_id, *, reviewer, subject_sha256, findings, attestation):
        """Record the reviewer's own findings. Moves AWAITING_REVIEW to READY."""
        operation = self.get(sport, operation_id)['payload']
        require(isinstance(reviewer, str) and reviewer.strip(), 'OPERATION_REVIEWER_REQUIRED')
        require(subject_sha256 == operation['subject_sha256'], 'OPERATION_REVIEW_SUBJECT_CONFLICT')
        require(isinstance(findings, dict) and findings and all(isinstance(k, str) and isinstance(v, str) and v.strip()
                for k, v in findings.items()), 'OPERATION_REVIEW_FINDINGS_REQUIRED')
        require(isinstance(attestation, str) and attestation.strip(), 'OPERATION_REVIEW_ATTESTATION_REQUIRED')
        return self._transition(sport, operation_id, READY, review=dict(
            reviewer=reviewer, subject_sha256=subject_sha256, findings=findings, attestation=attestation))

    def cancel(self, sport, operation_id, *, reason):
        require(isinstance(reason, str) and _CODE.match(reason), 'OPERATION_REASON_CODE_REQUIRED')
        return self._transition(sport, operation_id, CANCELLED, reason=reason)

    def run(self, sport, operation_id, work):
        """Spend the one-use claim, then call ``work(operation_payload)`` exactly once.

        ``work`` must return a JSON-serializable dict. Its result is stored before
        COMPLETED is recorded. Any exception records FAILED with a safe code.
        """
        status = self.status(sport, operation_id)
        if status['expired_unrecorded']:
            self._transition(sport, operation_id, EXPIRED, reason='OPERATION_EXPIRED')
            raise OperationError('OPERATION_EXPIRED')
        require(status['state'] == READY, 'OPERATION_NOT_READY:' + status['state'])
        operation = self.get(sport, operation_id)['payload']
        self.store.claim(sport, self._claim_id(operation['kind'], operation_id),
                         dict(operation_id=operation_id, kind=operation['kind'],
                              subject_sha256=operation['subject_sha256'], maximum_calls=1))
        self._transition(sport, operation_id, RUNNING)
        try:
            output = work(operation)
            require(isinstance(output, dict), 'OPERATION_OUTPUT_INVALID')
            stored = self.store.put(sport, 'operation-output', dict(operation_id=operation_id, output=output),
                                    identity=operation_id + ':output')
        except Exception as exc:  # recorded, never retried
            self._transition(sport, operation_id, FAILED, reason=failure_code(exc))
            raise
        self._transition(sport, operation_id, COMPLETED, output_id=stored['id'])
        return stored['payload']

    def reconcile(self, sport, operation_id):
        """Record what restart can prove: expiry, or an interrupted run's unknown outcome."""
        status = self.status(sport, operation_id)
        if status['expired_unrecorded']:
            return self._transition(sport, operation_id, EXPIRED, reason='OPERATION_EXPIRED')
        if status['state'] == RUNNING and self._past(self.get(sport, operation_id)['payload']):
            return self._transition(sport, operation_id, FAILED, reason='INTERRUPTED_OUTCOME_UNKNOWN')
        return status

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _claim_id(kind, operation_id):
        return f'{kind}-claim:{operation_id}'

    def _past(self, operation):
        return self.clock() >= _instant(operation['not_after'])

    def _transition(self, sport, operation_id, state, **details):
        status = self.status(sport, operation_id)
        current = status['state']
        require(state in TRANSITIONS.get(current, ()), f'OPERATION_TRANSITION_INVALID:{current}->{state}')
        if state in {READY, RUNNING}:
            require(not status['expired_unrecorded'], 'OPERATION_EXPIRED')
        sequence = len(status['events']) + 1
        event = dict(operation_id=operation_id, sequence=sequence, state=state, at=self.clock().isoformat(), **details)
        try:
            self.store.put(sport, 'operation-event', event, identity=f'{operation_id}:event:{sequence}')
        except StoreError as exc:
            if str(exc) == 'PIPELINE_RECORD_CONFLICT':
                raise OperationError('OPERATION_CONCURRENT_TRANSITION') from None
            raise
        return self.status(sport, operation_id)
