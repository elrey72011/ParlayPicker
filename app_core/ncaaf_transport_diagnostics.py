"""Opt-in private transport observations; never source admission or retry authority.

Only fixed enums, hashes, bounded counters and monotonic durations are retained.
The canonical capture/journal and historical readers keep their existing formats.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from app_core.nfl_acquisition_diagnostics import elapsed, status

VERSION = 'ncaaf-transport-diagnostics-v1'
UNKNOWN = 'UNKNOWN'
PHASES = frozenset(('credential_lookup', 'connect_dns_tcp_tls', 'request_write',
                    'response_headers', 'response_body_read', 'body_validation'))
HASH = re.compile(r'[0-9a-f]{64}\Z')
FIELDS = frozenset(('version', 'plan_sha256', 'authorization_sha256',
    'configuration_file_sha256', 'bundle_sha256', 'bundle_file_sha256',
    'journal_sha256', 'source_sha256', 'status', 'reason', 'active_seconds',
    'attempts', 'provider_receipt', 'credit_consumption', 'acceptance', 'inference'))
ATTEMPT_FIELDS = frozenset(('attempt', 'phases', 'failure_phase', 'http_status',
    'request_write_completed', 'complete_original', 'reason'))


def require(ok):
    if not ok: raise ValueError('NCAAF_TRANSPORT_DIAGNOSTICS_INVALID')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 86400


def sources():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
        'app_core/ncaaf_transport_diagnostics.py', 'app_core/ncaaf_pilot.py',
        'app_core/ncaaf_staged_acquisition.py', 'scripts/ncaaf_staged_acquisition.py')}


class Collector:
    """Main-thread observations only; timed/late workers never mutate diagnostics."""
    def __init__(self):
        self.attempts = []
        self.current = None
        self.phase = None
        self.started = None
        self.monotonic = None

    def begin(self, attempt, monotonic):
        require(self.current is None and type(attempt) is int and attempt == len(self.attempts)+1 and attempt <= 4)
        self.monotonic = monotonic
        self.current = dict(attempt=attempt, phases=[], failure_phase=UNKNOWN,
            http_status=UNKNOWN, request_write_completed=False, complete_original=False, reason=None)

    def _close_phase(self):
        if self.phase is None: return
        seconds = elapsed(self.started, self.monotonic())
        require(seconds != UNKNOWN)
        values = self.current['phases']
        found = next((v for v in values if v['phase'] == self.phase), None)
        if found is None: values.append(dict(phase=self.phase, seconds=seconds, visits=1))
        else:
            found['seconds'] = round(found['seconds']+seconds, 6)
            found['visits'] += 1

    def enter(self, phase):
        require(phase in PHASES and self.current is not None)
        if self.phase == phase: return
        self._close_phase()
        self.phase = phase
        self.started = self.monotonic()

    def observed_status(self, value):
        require(self.current is not None)
        self.current['http_status'] = status(value)

    def wrote_request(self):
        require(self.current is not None)
        self.current['request_write_completed'] = True

    def finish(self, reason=None, *, complete=False):
        if self.current is None: return
        from app_core.ncaaf_staged_acquisition import SAFE_CODES
        require(reason in SAFE_CODES | {None, 'NCAAF_STAGED_ACQUISITION_FAILURE'})
        self._close_phase()
        self.current.update(reason=reason, complete_original=complete,
            failure_phase=(self.phase or UNKNOWN) if reason else 'NONE')
        self.attempts.append(self.current)
        self.current = self.phase = self.started = None


def checked(value):
    return _checked(value, expected_sources=sources())


def _checked(value, *, expected_sources):
    from app_core.ncaaf_staged_acquisition import SAFE_CODES
    require(isinstance(value, dict) and set(value) == FIELDS and value['version'] == VERSION)
    for key in ('plan_sha256', 'authorization_sha256', 'configuration_file_sha256',
                'bundle_sha256', 'bundle_file_sha256', 'journal_sha256'):
        require(type(value[key]) is str and HASH.fullmatch(value[key]))
    require(value['source_sha256'] == expected_sources)
    require(value['status'] in {'INCOMPLETE', 'CAPTURED_UNREVIEWED'}
        and value['reason'] in SAFE_CODES | {None, 'NCAAF_STAGED_ACQUISITION_FAILURE'}
        and finite(value['active_seconds']) and value['provider_receipt'] == value['credit_consumption'] == UNKNOWN
        and value['acceptance'] is False and value['inference'] is False)
    require(type(value['attempts']) is list and len(value['attempts']) <= 4)
    for i, item in enumerate(value['attempts'], 1):
        require(type(item) is dict and set(item) == ATTEMPT_FIELDS and type(item['attempt']) is int and item['attempt'] == i)
        require(item['http_status'] == UNKNOWN or status(item['http_status']) != UNKNOWN)
        require(type(item['request_write_completed']) is bool and type(item['complete_original']) is bool
            and item['reason'] in SAFE_CODES | {None, 'NCAAF_STAGED_ACQUISITION_FAILURE'}
            and item['failure_phase'] in PHASES | {'NONE', UNKNOWN})
        require(type(item['phases']) is list and len(item['phases']) <= len(PHASES))
        seen = set()
        for phase in item['phases']:
            require(type(phase) is dict and set(phase) == {'phase', 'seconds', 'visits'}
                and phase['phase'] in PHASES and phase['phase'] not in seen
                and finite(phase['seconds']) and type(phase['visits']) is int and 1 <= phase['visits'] <= 100000)
            seen.add(phase['phase'])
        require(item['failure_phase'] in {'NONE', UNKNOWN} or item['failure_phase'] in seen)
        require((item['reason'] is None) == (item['failure_phase']=='NONE'))
        require(not item['request_write_completed'] or 'request_write' in seen)
        require(item['http_status']==UNKNOWN or ('response_headers' in seen and item['request_write_completed']))
        require(not item['complete_original'] or (item['http_status']==200 and item['request_write_completed']
            and 'body_validation' in seen and item['reason'] is None))
    require(sum(p['seconds'] for a in value['attempts'] for p in a['phases']) <= value['active_seconds']+.001)
    return deepcopy(value)


def retain(root, bundle, collector):
    from app_core import ncaaf_staged_acquisition as lane
    require(isinstance(collector, Collector) and collector.current is None)
    b = lane.read_bundle(bundle, complete=False)
    lane.verify_journal(root, bundle)
    require(len(collector.attempts) == b['attempts'])
    value = dict(version=VERSION, plan_sha256=b['plan']['sha256'],
        authorization_sha256=b['authorization']['sha256'],
        configuration_file_sha256=b['control_receipt']['file_sha256'],
        bundle_sha256=bundle['sha256'], bundle_file_sha256=hashlib.sha256(lane.model.encode(bundle)).hexdigest(),
        journal_sha256=b['journal_sha256'], source_sha256=sources(), status=b['status'], reason=b['reason'],
        active_seconds=b['active_seconds'], attempts=collector.attempts,
        provider_receipt=UNKNOWN, credit_consumption=UNKNOWN, acceptance=False, inference=False)
    sealed = lane.seal(checked(value))
    raw = lane.model.encode(sealed)
    require(len(raw) <= 32*1024)
    lane._exclusive(Path(root)/(b['plan']['sha256']+'.transport-diagnostics.json'), raw)
    read(root, bundle, raw, hashlib.sha256(raw).hexdigest())
    return sealed


def read(root, bundle, raw, expected_file_sha256):
    from app_core import ncaaf_staged_acquisition as lane
    require(type(raw) is bytes and 0 < len(raw) <= 32*1024
        and type(expected_file_sha256) is str and HASH.fullmatch(expected_file_sha256)
        and hashlib.sha256(raw).hexdigest() == expected_file_sha256)
    d = json.loads(raw, object_pairs_hook=lane.custody._unique, parse_constant=lane.custody._constant)
    require(type(d) is dict and set(d) == {'payload', 'sha256'} and lane.model.digest(d['payload']) == d['sha256'])
    value = checked(d['payload']); b = lane.read_bundle(bundle, complete=False)
    lane.verify_journal(root, bundle)
    require((Path(root)/(b['plan']['sha256']+'.stage.json')).read_bytes() == lane.model.encode(bundle))
    require(value['plan_sha256'] == b['plan']['sha256'] and value['authorization_sha256'] == b['authorization']['sha256']
        and value['configuration_file_sha256'] == b['control_receipt']['file_sha256']
        and value['bundle_sha256'] == bundle['sha256'] and value['journal_sha256'] == b['journal_sha256']
        and value['bundle_file_sha256'] == hashlib.sha256(lane.model.encode(bundle)).hexdigest()
        and value['active_seconds'] == b['active_seconds'] and value['reason'] == b['reason']
        and value['status'] == b['status'] and len(value['attempts']) == b['attempts'])
    for i, item in enumerate(value['attempts']):
        require(item['complete_original'] is (i < len(b['objects'])))
        if item['complete_original']: require(item['http_status'] == 200 and item['reason'] is None)
    return value
