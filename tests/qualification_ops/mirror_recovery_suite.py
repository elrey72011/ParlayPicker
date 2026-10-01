"""Offline transport publication and explicit chained recovery regressions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
import copy
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tempfile
from contextlib import redirect_stdout
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auth_recovery_suite as auth
from paths import DRIVER, SOURCE, LEGACY
prior = auth.prior
OLD = HERE / 'fixtures/previous_oauth_snapshot_acquire.py'
RUN = None
RESULTS = []


def denied(code=32):
    exc = PermissionError(errno.EACCES, 'PRIVATE_EXCEPTION_MUST_NOT_ESCAPE')
    exc.winerror = code
    return exc


def hold_handle(path, ready, release):
    """Separate Windows process: FILE_SHARE_READ|WRITE, deliberately no DELETE."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise RuntimeError('SYNTHETIC_HANDLE_OPEN_FAILED')
    try:
        Path(ready).write_text('READY', encoding='ascii')
        deadline = time.monotonic() + 10
        while not Path(release).exists() and time.monotonic() < deadline:
            time.sleep(.01)
    finally:
        close(handle)


class Tests(unittest.TestCase):
    def setUp(self):
        self.case=RUN/self._testMethodName;self.case.mkdir()
        self.temporary=tempfile.TemporaryDirectory(prefix='ParlayPicker-offline-mirror-')
        self.tmp=Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.fx=auth.fixture(self.tmp/'synthetic',40)
        self.m=prior.load_driver(DRIVER);self.m.source_check(self.fx.spec)
        self.backend=auth.AuthWire(self.fx.files,self.fx.wire)
        self.measure={};self.faults=[]
        self.addCleanup(lambda:auth.write(self.case/'measurements.json',
            dict(self.measure,faults=self.faults,synthetic_only=True,
                 real_socket_attempts=auth.REAL_SOCKET_ATTEMPTS+prior.REAL_REQUESTS)))
    fresh = auth.Tests.fresh
    actual_capture = auth.Tests.actual_capture
    fails = auth.Tests.fails
    meter = auth.Tests.meter
    failed_fixture = auth.Tests.failed_fixture
    history = auth.Tests.history
    hashes = auth.Tests.hashes

    def budget(self):
        self.fresh()
        meter = self.meter()
        for key, value in [('drive_get_attempts', 15204), ('oauth_attempts', 22),
                           ('observed_body_bytes', 140811320)]:
            meter.increment(key, value)
        return meter

    def test_M02_actual_old_writer_prefix_gap(self):
        self.fresh()
        old = prior.load_driver(OLD)
        meter = old.TransportBudget(self.fx.root, self.fx.spec, dict.fromkeys(old.TransportBudget.keys, 0))
        for key, value in [('drive_get_attempts', 15204), ('oauth_attempts', 22),
                           ('observed_body_bytes', 140811320)]:
            meter.increment(key, value)
        with patch.object(Path, 'replace', side_effect=denied()):
            with self.assertRaises(PermissionError):
                meter.increment('observed_body_bytes', 1)
        self.fails(lambda: old.TransportBudget(self.fx.root, self.fx.spec,
                   dict.fromkeys(old.TransportBudget.keys, 0)), 'TRANSPORT_MIRROR_CONFLICT')
        j = self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0),
                                       allow_one_event_prefix=True)
        self.assertEqual(j['sequence'], 4)
        self.assertEqual(j['publication_gap_events'], 1)
        self.assertEqual(j['counters']['observed_body_bytes'], 140811321)
        self.measure.update(old_loader='TRANSPORT_MIRROR_CONFLICT', events=4,
                            journal_body_bytes=140811321, historical_cause_proved=False)

    def test_M03_local_retry_no_charge_replay(self):
        meter = self.budget()
        before = meter.sequence
        original = Path.replace
        calls = []
        def replacement(path, target):
            calls.append(True)
            if len(calls) <= 2: raise denied()
            return original(path, target)
        with patch.object(Path, 'replace', replacement):
            meter.increment('observed_body_bytes', 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(meter.sequence, before + 1)
        self.assertEqual(meter['oauth_attempts'], 22)
        self.assertEqual(meter['drive_get_attempts'], 15204)
        self.assertEqual(self.backend.oauth_posts, 0)
        self.measure.update(local_replace_attempts=3, new_journal_events=1, network_calls=0)

    def holder(self):
        ready, release = self.tmp/'holder-ready', self.tmp/'holder-release'
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                   '--hold-no-delete', str(self.fx.root/'transport-current.json'), str(ready), str(release)]
        child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists(), 'WINDOWS_HOLDER_NOT_READY')
        def cleanup():
            release.write_text('RELEASE', encoding='ascii')
            child.wait(timeout=15)
        self.addCleanup(cleanup)
        return child, release

    @unittest.skipUnless(os.name == 'nt', 'Actual Windows delete-sharing handles only')
    def test_M03_windows_subprocess_delete_sharing_release(self):
        meter = self.budget()
        child, release = self.holder()
        original = Path.replace
        errors = []
        def replacement(path, target):
            try: return original(path, target)
            except OSError as exc:
                errors.append(getattr(exc, 'winerror', None))
                raise
        releaser = threading.Thread(target=lambda: (time.sleep(.2), release.write_text('RELEASE')))
        releaser.start()
        self.addCleanup(releaser.join)
        with patch.object(Path, 'replace', replacement):
            meter.increment('observed_body_bytes', 1)
        releaser.join()
        child.wait(timeout=15)
        self.assertTrue(errors)
        self.assertTrue(all(code in (5, 32, 33) for code in errors))
        self.assertEqual(meter.sequence, 4)
        self.assertEqual(meter['observed_body_bytes'], 140811321)
        self.measure.update(real_windows_share_codes=errors, holder_subprocess=True,
                            network_calls=0, new_journal_events=1, completed_after_release=True)

    @unittest.skipUnless(os.name == 'nt', 'Actual Windows delete-sharing handles only')
    def test_M04_windows_persistent_lock_bounded(self):
        meter = self.budget()
        self.holder()
        begun = time.monotonic()
        with self.assertRaises(self.m.PublicationError) as caught:
            meter.increment('observed_body_bytes', 1)
        elapsed = time.monotonic() - begun
        self.assertLess(elapsed, 3.5)
        fields = caught.exception.first_error
        self.assertIn(fields['winerror'], (5, 32, 33))
        self.assertIn(fields['sharing_probe_winerror'], (32, 33))
        self.assertLessEqual(fields['local_retry_count'], 7)
        self.assertEqual(meter.sequence, 4)
        self.measure.update(real_windows_share_code=fields['winerror'],
                            local_retries=fields['local_retry_count'], bounded_elapsed=elapsed)

    def test_M04_access_denial_and_nonsharing_errors_do_not_retry(self):
        for fault in [denied(5), FileNotFoundError(errno.ENOENT, 'PRIVATE'),
                      OSError(errno.ENOSPC, 'PRIVATE')]:
            with self.subTest(exception=type(fault).__name__):
                meter = self.budget() if not self.fx.root.exists() else self.meter()
                with patch.object(self.m, 'transient_sharing_code', return_value=None), \
                     patch.object(Path, 'replace', side_effect=fault) as replacement:
                    with self.assertRaises(self.m.PublicationError) as caught:
                        meter.increment('observed_body_bytes', 1)
                self.assertEqual(replacement.call_count, 1)
                self.assertEqual(caught.exception.first_error['local_retry_count'], 0)
                # Explicit synthetic successor projection; never used on an incident.
                j = self.m.read_transport_journal(self.fx.root, dict.fromkeys(meter.keys, 0))
                value = dict(counters=j['counters'], journal_tail_sha256=j['tail'])
                self.m.save(self.fx.root/'transport-current.json',
                            dict(value, canonical_sha256=hashlib.sha256(self.m.canonical(value)).hexdigest()))

    def test_M04_persistent_injected_sharing_bound(self):
        meter = self.budget()
        with patch.object(Path, 'replace', side_effect=denied()) as replacement:
            with self.assertRaises(self.m.PublicationError) as caught:
                meter.increment('observed_body_bytes', 1)
        self.assertEqual(replacement.call_count, 8)
        self.assertEqual(caught.exception.first_error['local_retry_count'], 7)
        self.assertLessEqual(caught.exception.first_error['retry_elapsed_seconds'], 1.6)
        self.assertEqual(meter.sequence, 4)

    def test_M05_temp_flush_failure_retains_one_durable_event(self):
        meter = self.budget()
        actual_fsync = self.m.os.fsync
        calls = []
        def fsync(fd):
            calls.append(fd)
            if len(calls) == 2: raise OSError(errno.ENOSPC, 'PRIVATE')
            return actual_fsync(fd)
        with patch.object(self.m.os, 'fsync', fsync):
            with self.assertRaises(self.m.PublicationError) as caught:
                meter.increment('observed_body_bytes', 1)
        self.assertEqual(caught.exception.first_error['attempted_operation'], 'flush_temp')
        j = self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0),
                                       allow_one_event_prefix=True)
        self.assertEqual(j['publication_gap_events'], 1)

    def test_M05_before_append_and_partial_append_fail_closed(self):
        meter = self.budget()
        original = Path.open
        def fail_open(path, *a, **kw):
            if path.name == 'transport-events.jsonl' and a and a[0] == 'ab':
                raise PermissionError(errno.EACCES, 'PRIVATE')
            return original(path, *a, **kw)
        with patch.object(Path, 'open', fail_open):
            with self.assertRaises(self.m.PublicationError): meter.increment('drive_get_attempts', 1)
        self.assertEqual(self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0))['sequence'], 3)
        # A genuinely interrupted event is not recovered as a valid prefix gap.
        with (self.fx.root/'transport-events.jsonl').open('ab') as stream: stream.write(b'{"sequence":')
        self.fails(lambda: self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0),
                   allow_one_event_prefix=True), 'TRANSPORT_JOURNAL_INVALID')

    def test_M05_after_append_fsync_failure_never_replays(self):
        meter = self.budget()
        with patch.object(self.m.os, 'fsync', side_effect=OSError(errno.EIO, 'PRIVATE')):
            with self.assertRaises(self.m.PublicationError): meter.increment('observed_body_bytes', 1)
        j = self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0),
                                       allow_one_event_prefix=True)
        self.assertEqual(j['sequence'], 4)
        self.assertEqual(j['counters']['observed_body_bytes'], 140811321)
        self.assertEqual(self.backend.oauth_posts, 0)

    def test_M05_replace_completed_then_failure_no_replay(self):
        meter=self.budget()
        original=Path.replace
        def replaced(path,target):
            original(path,target)
            raise OSError(errno.EIO,'PRIVATE')
        with patch.object(Path,'replace',replaced):
            with self.assertRaises(self.m.PublicationError):
                meter.increment('observed_body_bytes',1)
        journal=self.m.validated_transport(self.fx.root,dict.fromkeys(meter.keys,0))
        self.assertEqual(journal['sequence'],4)
        self.assertEqual(journal['publication_gap_events'],0)
        self.assertEqual(journal['counters']['observed_body_bytes'],140811321)

    def test_M05_temp_open_failure(self):
        meter = self.budget()
        original = Path.open
        def fail_temp(path, *a, **kw):
            if '.publish-' in path.name:
                raise PermissionError(errno.EACCES, 'PRIVATE')
            return original(path, *a, **kw)
        with patch.object(Path, 'open', fail_temp):
            with self.assertRaises(self.m.PublicationError) as caught:
                meter.increment('observed_body_bytes', 1)
        self.assertEqual(caught.exception.first_error['attempted_operation'], 'write_temp')
        self.assertEqual(meter.sequence, 4)

    def test_M06_multithreaded_writer_and_short_observers(self):
        meter = self.budget()
        observations, faults = [], []
        stop = threading.Event()
        def observe():
            while not stop.is_set():
                try:
                    raw = self.m.read_record_bytes(self.fx.root/'transport-current.json')
                    time.sleep(.001)  # Cooperative handle has already closed.
                    d = json.loads(raw)
                    self.assertEqual(d['canonical_sha256'], hashlib.sha256(self.m.canonical(
                        {k:v for k,v in d.items() if k!='canonical_sha256'})).hexdigest())
                    observations.append(d['counters']['observed_body_bytes'])
                except BaseException as exc:
                    faults.append(type(exc).__name__)
                    break
                time.sleep(.002)
        reader = threading.Thread(target=observe)
        reader.start()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: meter.increment('observed_body_bytes', 1), range(320)))
        except self.m.PublicationError as exc:
            self.measure['publication_first_error']=exc.first_error
            raise
        finally:
            stop.set()
            reader.join()
        self.assertFalse(faults)
        self.assertTrue(observations)
        j = self.m.validated_transport(self.fx.root, dict.fromkeys(meter.keys, 0))
        self.assertEqual(j['sequence'], 323)
        self.assertEqual(j['counters']['observed_body_bytes'], 140811640)
        self.measure.update(observations=len(observations), exact_updates=320,
                            journal_events=323, reader_faults=faults)

    def test_M08_mirror_and_journal_fault_rules(self):
        meter = self.budget()
        originals = {name:(self.fx.root/name).read_bytes()
                     for name in ('transport-events.jsonl','transport-current.json')}
        zero = dict.fromkeys(meter.keys, 0)
        for fault in ('missing','bad-digest','divergent','ahead','unknown-key','bad-sequence',
                      'bad-link','counter-rewind','truncated','two-event-gap'):
            with self.subTest(fault=fault):
                for name,raw in originals.items(): (self.fx.root/name).write_bytes(raw)
                path = self.fx.root/'transport-current.json'
                if fault == 'missing': path.unlink()
                elif fault == 'bad-digest': path.write_text('{}')
                elif fault in ('divergent','ahead'):
                    mirror=json.loads(path.read_bytes())
                    if fault=='divergent':mirror['counters']['observed_body_bytes']+=1
                    else:mirror['journal_tail_sha256']='a'*64
                    mirror['canonical_sha256']=hashlib.sha256(self.m.canonical({k:v for k,v in mirror.items() if k!='canonical_sha256'})).hexdigest()
                    prior.write_json(path,mirror)
                elif fault=='two-event-gap':
                    current=self.m.TransportBudget(self.fx.root,self.fx.spec,zero)
                    with patch.object(Path,'replace',side_effect=denied(5)):
                        for _ in range(2):
                            with self.assertRaises(self.m.PublicationError):current.increment('observed_body_bytes',1)
                else:
                    path=self.fx.root/'transport-events.jsonl'
                    events=[json.loads(line) for line in path.read_bytes().splitlines()]
                    if fault=='truncated':path.write_bytes(path.read_bytes()[:-1])
                    else:
                        event=events[-1]
                        if fault=='unknown-key':event['unapproved']='PRIVATE'
                        elif fault=='bad-sequence':event['sequence']+=1
                        elif fault=='bad-link':event['previous_sha256']='b'*64
                        elif fault=='counter-rewind':event['counters']['drive_get_attempts']-=1
                        event['canonical_sha256']=hashlib.sha256(self.m.canonical({k:v for k,v in event.items() if k!='canonical_sha256'})).hexdigest()
                        path.write_bytes(b''.join(self.m.canonical(e)+b'\n' for e in events))
                with self.assertRaises(RuntimeError):
                    self.m.validated_transport(self.fx.root,zero,allow_one_event_prefix=True)
        self.measure.update(rejected_faults=10, real_requests=0)

    def test_M11_first_error_survives_mirror_loading_and_diagnostic_write_failure(self):
        meter=self.budget()
        with patch.object(Path,'replace',side_effect=denied(5)):
            with self.assertRaises(self.m.PublicationError) as caught:meter.increment('observed_body_bytes',1)
        report=self.m.retain_worker_failure(self.fx.root,self.fx.spec,'capture-block',
                                           'LOCAL_PUBLICATION_FAILED',caught.exception)
        self.assertEqual(report['transport']['observed_body_bytes'],140811321)
        self.assertFalse(report['mirror_matches_journal'])
        self.assertEqual(report['first_error']['winerror'],5)
        self.assertEqual(report['first_error']['journal_sequence'],4)
        trace=io.StringIO()
        with redirect_stderr(trace),patch.object(self.m,'seal',side_effect=OSError(errno.ENOSPC,'PRIVATE_SECRET')):
            fallback=self.m.retain_worker_failure(self.fx.root,self.fx.spec,'capture-block',
                                                  'LOCAL_PUBLICATION_FAILED',caught.exception)
        text=trace.getvalue()
        self.assertIn('diagnostic_record_retained',text)
        self.assertNotIn('PRIVATE_EXCEPTION',text)
        self.assertNotIn('PRIVATE_SECRET',text)
        self.assertNotIn(str(self.fx.root),text)
        self.assertEqual(fallback['reason'],'LOCAL_PUBLICATION_FAILED')

    def test_M07_M09_M10_M12_full_generation_chain_and_supervised_continuation(self):
        # Actual canonical writer/codec: 15,032 accepted and 8 missing synthetic objects.
        self.fx=auth.fixture(self.tmp/'large',15040)
        self.backend=auth.AuthWire(self.fx.files,self.fx.wire)
        self.failed_fixture()
        # Restore the production-sized SYNTHETIC cap before binding original ancestry.
        self.fx.spec['max_new_objects_per_slice']=5000
        self.fx.save_spec()
        state=self.m.verified_record(self.fx.root/'state.json')
        state['spec_sha256']=auth.sha(self.fx.path)
        self.m.persist_state(self.fx.root,state)
        h1=self.history()
        v2root=self.fx.root.with_name('acquire-'+str(self.fx.spec['anchor_run_id'])+'-recovery-v2')
        expected1={k:h1[k] for k in ('original_operation_id','accepted_count','incurred_transport',
                                    'state_sha256','journal_sha256','journal_tail_sha256')}
        expected1['file_register_canonical_sha256']=hashlib.sha256(self.m.canonical(h1['files'])).hexdigest()
        v2={'original_driver_path':str(LEGACY),'original_driver_sha256':auth.sha(LEGACY),
            'replacement_driver_sha256':auth.sha(OLD),'original_spec_sha256':auth.sha(self.fx.path),
            'predecessor_destination':str(self.fx.root),'successor_destination':str(v2root),
            'expected_predecessor':expected1,'additional_oauth_allowance':20,
            'successor_max_slices':7,'historical_elapsed_seconds':54.171}
        v2path=self.tmp/'synthetic-v2-addendum.json';auth.write(v2path,v2)
        effective=self.m.recovery_policy(self.fx.spec,h1,20,7)
        effective.update(destination=str(v2root),recovery_mode=True,
                         working_disk_predecessor_bytes=sum(x['bytes'] for x in h1['files'].values()))
        for key in ('capture_total_wall_seconds','total_wall_seconds'):effective[key]-=54.171
        self.m.args.recovery_spec=v2path
        self.m.initialize_successor(effective,v2,h1,'SYNTHETIC-WORKER')
        # Baseline link must bind the historical v2, not today's replacement.
        link=self.m.verified_record(v2root/'recovery-link.json')
        link['replacement_driver_sha256']=auth.sha(OLD)
        link['canonical_sha256']=hashlib.sha256(self.m.canonical({k:v for k,v in link.items() if k!='canonical_sha256'})).hexdigest()
        auth.write(v2root/'recovery-link.json',link)
        self.m.save(v2root/'effective-spec.json',effective)
        state=self.m.verified_record(v2root/'state.json');state['spec_sha256']=auth.sha(v2root/'effective-spec.json')
        pinned=self.m.anchor(self.fx.spec)[2]
        completed=set(self.m.accepted_objects(v2root,state,effective,pinned))
        pending=sorted(set(pinned)-completed)
        # Real wrapper's seal/state writers retain real codec bytes. No fabricated scientific rows.
        for slice_number in range(1,4):
            baseline=len(completed)
            self.m.seal(v2root,f'slice-{slice_number:02d}-attempt.json',
                {'status':'ATTEMPT_STARTED_NO_AUTOMATIC_RETRY','accepted_reused_logical_objects':baseline,
                 'source_revision':effective['source_revision'],'storage_scope_hash':effective['storage_scope_hash']})
            for offset in range(0,5000,8):
                objects={}
                for name in pending[:8]:
                    body=self.fx.wire[name];item={'content_sha256':hashlib.sha256(body).hexdigest(),
                        'metadata_token':pinned[name]['metadata_token'],'bytes':len(body)}
                    (v2root/'raw'/item['content_sha256']).write_bytes(body);objects[name]=item
                batch=(offset//8)+1+(4 if slice_number==1 else 0)
                ref=self.m.seal(v2root,f'slice-{slice_number:02d}-batch-{batch:04d}.json',
                    {'source_revision':effective['source_revision'],'storage_scope_hash':effective['storage_scope_hash'],
                     'operation_id':'SYNTHETIC-SLICE-INVENTORY','objects':objects,'read_metrics':{'retries':0}})
                state['accepted_batches'].append(ref);completed.update(objects);pending=pending[8:]
            self.m.seal(v2root,f'slice-{slice_number:02d}-result.json',
                {'slice':slice_number,'status':'PARTIAL','newly_retained_logical_objects':5000,
                 'terminal_reason':'OBJECT_LIMIT','source_revision':effective['source_revision'],
                 'storage_scope_hash':effective['storage_scope_hash']})
        state['next_slice']=4;self.m.persist_state(v2root,state)
        self.m.seal(v2root,'slice-04-attempt.json',{'status':'ATTEMPT_STARTED_NO_AUTOMATIC_RETRY','source_revision':effective['source_revision'],
                     'storage_scope_hash':effective['storage_scope_hash']})
        meter=self.m.TransportBudget(v2root,effective,state['transport'])
        for key,value in [('drive_get_attempts',15204),('oauth_attempts',22),
                          ('observed_body_bytes',140811320)]:
            meter.increment(key,value-meter[key])
        with patch.object(Path,'replace',side_effect=denied(5)):
            with self.assertRaises(self.m.PublicationError):meter.increment('observed_body_bytes',1)
        self.m.seal(v2root,'operation-blocked.json',
                    {'status':'BLOCKED','reason':'WORKER_BLOCKED_NO_RETRY','elapsed_seconds':3641.547})
        recovery={'recovery_generation':3,'additional_oauth_allowance':0,
                  'original_driver_sha256':auth.sha(LEGACY),'original_spec_sha256':auth.sha(self.fx.path),
                  'v2_driver_path':str(OLD),'v2_recovery_spec_path':str(v2path),
                  'v2_recovery_spec_sha256':auth.sha(v2path),'predecessor_destination':str(v2root),
                  'successor_destination':str(v2root.with_name('acquire-'+str(self.fx.spec['anchor_run_id'])+'-recovery-v3')),
                  'historical_elapsed_seconds':3695.718,'successor_max_slices':3}
        self.m.args.spec=self.fx.path
        h2=self.m.inspect_linked_predecessor(self.fx.spec,recovery)
        self.assertEqual(h2['accepted_count'],15032)
        self.assertEqual(len(self.m.verified_record(v2root/'state.json')['accepted_batches']),1879)
        self.assertEqual(h2['attempts_used'],5)
        self.assertEqual(h2['incurred_transport'],{'drive_get_attempts':15204,'oauth_attempts':22,'observed_body_bytes':140811321})
        proposed=self.m.linked_recovery_policy(self.fx.spec,h2,3)
        self.assertEqual(proposed['max_oauth_token_attempts'],40)
        self.assertAlmostEqual(proposed['capture_total_wall_seconds'],20304.282)
        self.assertAlmostEqual(proposed['total_wall_seconds'],23304.282)
        self.fails(lambda:self.m.linked_recovery_policy(self.fx.spec,h2,4),'RECOVERY_SLICE_ALLOWANCE_INVALID')
        self.assertEqual(proposed['working_disk_predecessor_bytes'],h2['predecessor_disk_bytes'])
        before={label:{str(p.relative_to(r)):auth.sha(p) for p in r.rglob('*') if p.is_file()}
                for label,r in [('v1',self.fx.root),('v2',v2root)]}
        # M09 invalid accepted payload blocks admission; restore only this synthetic fixture.
        target=v2root/'raw'/next(iter(h2['accepted_objects'].values()))['content_sha256']
        raw=target.read_bytes();target.write_bytes(b'corrupt')
        self.fails(lambda:self.m.inspect_linked_predecessor(self.fx.spec,recovery),'RAW_CACHE_REJECTED')
        target.write_bytes(raw)
        # Effective-spec/addendum/ancestry hashes are enforced, not silently rebound.
        path=v2root/'effective-spec.json';raw=path.read_bytes();bad=json.loads(raw);bad['max_oauth_token_attempts']=60
        auth.write(path,bad)
        self.fails(lambda:self.m.inspect_linked_predecessor(self.fx.spec,recovery),'RECOVERY_EFFECTIVE_SPEC_CONFLICT')
        path.write_bytes(raw)
        for key,value,reason in [
            ('historical_elapsed_seconds',0,'RECOVERY_ELAPSED_CHAIN_CONFLICT'),
            ('v2_recovery_spec_sha256','0'*64,'RECOVERY_ADDENDUM_CHAIN_CONFLICT'),
            ('additional_oauth_allowance',20,'RECOVERY_ALLOWANCE_INVALID')]:
            altered=dict(recovery);altered[key]=value
            self.fails(lambda:self.m.inspect_linked_predecessor(self.fx.spec,altered),reason)
        recovery['expected_predecessor']={k:h2[k] for k in ('accepted_count','incurred_transport','state_sha256',
            'journal_sha256','journal_tail_sha256','effective_spec_sha256','recovery_link_sha256',
            'blocked_record_sha256','file_register_canonical_sha256','attempts_used')}
        recovery['replacement_driver_sha256']=auth.sha(DRIVER)
        recovery_path=self.tmp/'synthetic-v3-addendum.json';auth.write(recovery_path,recovery)
        self.m.args=SimpleNamespace(spec=self.fx.path,approved_spec_sha256=auth.sha(self.fx.path),
            recovery_spec=recovery_path,approved_recovery_sha256=auth.sha(recovery_path),
            approved_driver_sha256=auth.sha(DRIVER))
        self.fx.backend_file()
        actual_popen=subprocess.Popen
        def child(command,**kw):
            if '--worker' not in command:return actual_popen(command,**kw)
            return actual_popen([sys.executable,'-B','-X','utf8',str(HERE/'auth_recovery_suite.py'),
                '--offline-child',str(self.fx.base/'fake-wire.json'),*command[3:]],**kw)
        with patch.dict(os.environ,{'LOCALAPPDATA':str(self.fx.base/'local-app-data'),
             'PARLAYPICKER_DRIVE_FOLDER_ID':'fixture-folder','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT':'SYNTHETIC_ONLY'}),patch.object(subprocess,'Popen',child):
            code=self.m.supervise(self.fx.spec)
        self.assertEqual(code,0)
        successor=Path(recovery['successor_destination'])
        reconciliation=self.m.verified_record(successor/'transport-reconciliation.json')
        self.assertEqual(reconciliation['publication_gap_events'],1)
        self.assertEqual(reconciliation['charged_slice_attempts'],5)
        self.assertEqual(reconciliation['incurred_transport']['observed_body_bytes'],140811321)
        assessment=self.m.verified_record(successor/'local-assessment.json')
        self.assertEqual(assessment['snapshot_sha256_before'],assessment['snapshot_sha256_after'])
        for table in ('prospective_model','prospective_calibration','prospective_prediction',
                      'prospective_validation_artifact','prospective_deployment_review'):
            self.assertEqual(assessment['canonical_counts'][table],0)
        self.assertTrue(all(p['status']=='BLOCKED_MISSING_BINDING' for p in assessment['plan_assessments']))
        for label,r in [('v1',self.fx.root),('v2',v2root)]:
            self.assertEqual(before[label],{str(p.relative_to(r)):auth.sha(p) for p in r.rglob('*') if p.is_file()})
        result=self.m.verified_record(successor/'slice-01-result.json')
        self.assertEqual(result['accepted_reused_logical_objects'],15032)
        self.assertEqual(result['newly_retained_logical_objects'],8)
        self.assertEqual(result['transport']['oauth_attempts'],23)
        self.measure.update(accepted_reused=15032,ledger_references=1879,charged_slices=5,remaining_slices=3,
            historical_gets=15204,historical_oauth=22,historical_body=140811321,charged_elapsed=3695.718,
            remaining_oauth_before_new_work=18,supervisor_exit=code,complete_synthetic_snapshot=True,
            blocked_scientific_readiness=True,predecessors_unchanged=True,new_objects=8)

class Recorded(unittest.TextTestResult):
    def startTest(self,test):self.started=time.monotonic();super().startTest(test)
    def entry(self,test,status,detail=None):
        result={'test':test.id(),'status':status,'seconds':time.monotonic()-self.started}
        if detail:result['detail']=detail
        RESULTS.append(result)
    def addSuccess(self,test):self.entry(test,'PASS');super().addSuccess(test)
    def addSkip(self,test,reason):self.entry(test,'SKIP',reason);super().addSkip(test,reason)
    def addFailure(self,test,error):self.entry(test,'FAIL',self._exc_info_to_string(error,test));super().addFailure(test,error)
    def addError(self,test,error):self.entry(test,'ERROR',self._exc_info_to_string(error,test));super().addError(test,error)

if __name__=='__main__':
    if '--hold-no-delete' in sys.argv:
        index=sys.argv.index('--hold-no-delete')
        hold_handle(*sys.argv[index+1:index+4])
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument('--run-directory',type=Path,required=True)
        parser.add_argument('--select')
        options=parser.parse_args()
        RUN=options.run_directory.resolve();RUN.mkdir(parents=True,exist_ok=False);auth.RUN=RUN
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
        if options.select:suite=unittest.TestSuite(t for t in suite if options.select in t.id())
        result=unittest.TextTestRunner(verbosity=2,resultclass=Recorded).run(suite)
        auth.write(RUN/'test-results.json',{'driver_sha256':auth.sha(DRIVER),'tests_run':result.testsRun,
            'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
            'results':RESULTS,'real_socket_attempts':auth.REAL_SOCKET_ATTEMPTS+prior.REAL_REQUESTS,'synthetic_only':True})
        raise SystemExit(0 if result.wasSuccessful() else 1)
