"""Operations proposal, outside application source. Default is OFFLINE preflight.

An execution flag does not confer owner authorization. Capture, local import and
assessment require the separately approved exact spec. Never call remote sync.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid
import os
from pathlib import Path
import shutil
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


# Local retries are projections only: never repeat a journal/resource action.
PUBLICATION_ATTEMPTS = 8
PUBLICATION_SECONDS = 1.5
_publication_locks = {}
_publication_guard = threading.Lock()


class PublicationError(RuntimeError):
    def __init__(self, details):
        super().__init__('LOCAL_PUBLICATION_FAILED')
        self.first_error = details


def error_fields(exc):
    return {'exception_class': type(exc).__name__,
            'errno': getattr(exc, 'errno', None),
            'winerror': getattr(exc, 'winerror', None)}


def file_role(path):
    return {'transport-current.json': 'transport_mirror', 'state.json': 'accepted_state',
            'effective-spec.json': 'effective_spec'}.get(Path(path).name, 'mutable_record')


def transient_sharing_code(exc, target, temporary):
    """Code 5 alone is insufficient: prove sharing contention or delete/create access."""
    code = getattr(exc, 'winerror', None)
    if code in (32, 33):
        return code
    if code != 5 or os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    if not target.exists() or not temporary.exists():
        return None
    attributes = kernel.GetFileAttributesW
    attributes.argtypes = [wintypes.LPCWSTR]
    attributes.restype = wintypes.DWORD
    for candidate in (target, temporary):
        attr = attributes(str(candidate))
        if attr == 0xffffffff or attr & 1:  # Invalid/read-only: no transient classification.
            return None
        # Request DELETE access with all sharing enabled; never delete or alter.
        handle = create(str(candidate), 0x00010000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            probe = ctypes.get_last_error()
            return probe if probe in (32, 33) else None
        close(handle)
    # Prove parent create/rename capability too. No file/ACL is changed.
    handle = create(str(target.parent), 2, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        probe = ctypes.get_last_error()
        return probe if probe in (32, 33) else None
    close(handle)
    return 0  # Delete/create access is currently proven; bounded replacement retry.


def save(path, value):
    """Serialized fsynced same-directory publication, with bounded sharing retries.

    Windows sharing/lock violations (32/33), or replacement-only code 5 with
    proven delete/create access and non-read-only files, are retryable. ACL denial,
    ENOSPC, missing paths and unsupported failures stop. Directory fsync/power-loss
    persistence is not claimed; incomplete unique temp files remain evidence.
    """
    path = Path(path)
    with _publication_guard:
        lock = _publication_locks.setdefault(str(path.resolve()), threading.RLock())
    with lock:
        temporary = path.with_name(path.name + '.publish-' + uuid.uuid4().hex + '.tmp')
        operation = 'write_temp'
        try:
            with temporary.open('xb') as stream:
                stream.write(json.dumps(value, sort_keys=True, indent=2).encode() + b'\n')
                operation = 'flush_temp'
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise PublicationError(dict(error_fields(exc), attempted_operation=operation,
                file_role=file_role(path), local_retry_count=0, retry_elapsed_seconds=0.0)) from None
        begun = time.monotonic()
        first = None
        for attempt in range(1, PUBLICATION_ATTEMPTS + 1):
            if first is not None and time.monotonic() - begun >= PUBLICATION_SECONDS:
                raise PublicationError(dict(first, attempted_operation='replace',
                    file_role=file_role(path), local_retry_count=attempt - 1,
                    retry_elapsed_seconds=time.monotonic() - begun)) from None
            try:
                temporary.replace(path)
                return
            except OSError as exc:
                if first is None:
                    first = error_fields(exc)
                elapsed = time.monotonic() - begun
                sharing = transient_sharing_code(exc, path, temporary)
                if sharing is not None:
                    if sharing:
                        first['sharing_probe_winerror'] = sharing
                    else:
                        first['replacement_access_checks_passed'] = True
                retryable = sharing is not None
                if not retryable or attempt == PUBLICATION_ATTEMPTS or elapsed >= PUBLICATION_SECONDS:
                    raise PublicationError(dict(first, attempted_operation='replace',
                        file_role=file_role(path), local_retry_count=attempt - 1,
                        retry_elapsed_seconds=elapsed, final_exception=error_fields(exc))) from None
                wait = min(.02 * 2 ** (attempt - 1), .25, PUBLICATION_SECONDS - elapsed)
                time.sleep(max(0.0, wait))


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_check(spec):
    source = Path(spec['source_checkout'])
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == spec['source_revision'], 'SOURCE_SHA_CONFLICT')
    require(git('rev-parse', 'HEAD^{tree}') == spec['source_tree'], 'SOURCE_TREE_CONFLICT')
    require(not git('status', '--porcelain=v1', '--untracked-files=no'), 'SOURCE_MODIFIED')
    sys.path.insert(0, str(source))


def anchor(spec):
    from app_core.read_only_census import _checkpoint_digest
    retained = Path(spec['retained_directory'])
    cp_path, rp_path = retained / 'slice-01/checkpoint.json', retained / 'slice-01/report.json'
    require(digest(cp_path) == spec['anchor_checkpoint_file_sha256'], 'CHECKPOINT_FILE_CONFLICT')
    require(digest(rp_path) == spec['anchor_report_sha256'], 'REPORT_FILE_CONFLICT')
    register = json.loads((retained / 'artifact-register.json').read_bytes())
    archives = [x for x in register['files'] if x['sha256'] == spec['anchor_archive_sha256']]
    require(len(archives) == 1 and digest(archives[0]['path']) == spec['anchor_archive_sha256'], 'ARCHIVE_CONFLICT')
    cp, rp = json.loads(cp_path.read_bytes()), json.loads(rp_path.read_bytes())
    require(_checkpoint_digest(cp) == cp['checkpoint_sha256'] == spec['anchor_checkpoint_canonical_sha256'], 'CHECKPOINT_DIGEST_CONFLICT')
    require(cp['source_revision'] == rp['source_revision'] == spec['source_revision'], 'ANCHOR_SOURCE_CONFLICT')
    require(cp['storage_scope_hash'] == spec['storage_scope_hash'], 'ANCHOR_SCOPE_CONFLICT')
    require(cp['inventory_membership'][spec['namespace']] == spec['canonical_membership_sha256'], 'MEMBERSHIP_CONFLICT')
    require(rp['status'] == 'COMPLETE', 'ANCHOR_INCOMPLETE')
    pinned = {k: v for k, v in cp['processed'].items() if k.startswith(spec['namespace'])}
    require(len(pinned) == spec['canonical_objects'], 'PINNED_COUNT_CONFLICT')
    membership=hashlib.sha256(canonical(sorted((name,item['metadata_token']) for name,item in pinned.items()))).hexdigest()
    require(membership == spec['canonical_membership_sha256'], 'PROCESSED_MEMBERSHIP_CONFLICT')
    return cp, rp, pinned


def disk_bytes(root):
    size=0
    for path in root.rglob('*'):
        try:
            if path.is_file(): size+=path.stat().st_size
        except FileNotFoundError:
            pass # A concurrent cache tempfile rename does not invalidate data.
    return size


_last_disk_check = (None, 0.0)
def budget_disk(root, spec, *, force=False):
    global _last_disk_check
    # Monitored stop threshold, not a reservation or a filesystem quota. Avoid
    # rescanning the entire byte corpus on every SQL insertion.
    if force or _last_disk_check[0]!=str(root.resolve()) or time.monotonic() - _last_disk_check[1] >= 5:
        require(disk_bytes(root)+spec.get('working_disk_predecessor_bytes',0) <= spec['max_working_disk_bytes'], 'WORKING_DISK_LIMIT')
        _last_disk_check = (str(root.resolve()), time.monotonic())


def seal(root, name, value):
    # Accepted ledgers are immutable. The supervisor saves separately on failure.
    target = root / name
    require(not target.exists(), 'LEDGER_ALREADY_EXISTS')
    value = dict(value, sealed_at=now())
    value['canonical_sha256'] = hashlib.sha256(canonical(value)).hexdigest()
    with target.open('xb') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2).encode()+b'\n')
        stream.flush();os.fsync(stream.fileno())
    return {'file': name, 'sha256': digest(target)}


def check_seal(root, reference):
    require(Path(reference['file']).name==reference['file'], 'LEDGER_PATH_CONFLICT')
    target = root / reference['file']
    require(digest(target) == reference['sha256'], 'RAW_LEDGER_FILE_CONFLICT')
    record = json.loads(target.read_bytes())
    unsigned = {k: v for k, v in record.items() if k != 'canonical_sha256'}
    require(hashlib.sha256(canonical(unsigned)).hexdigest() == record['canonical_sha256'], 'RAW_LEDGER_DIGEST_CONFLICT')
    return record


def read_record_bytes(path):
    # In-process observers participate in the same short publication lock.
    path = Path(path)
    with _publication_guard:
        lock = _publication_locks.setdefault(str(path.resolve()), threading.RLock())
    with lock:
        return _read_record_bytes_unlocked(path)


def _read_record_bytes_unlocked(path):
    """Cooperative Windows observers explicitly allow read/write/delete sharing."""
    path = Path(path)
    if os.name != 'nt':
        return path.read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as stream:
        return stream.read()


def verified_record(path):
    try: record=json.loads(read_record_bytes(path))
    except (OSError,ValueError): raise RuntimeError('SEALED_RECORD_MISSING_OR_INVALID') from None
    unsigned={k:v for k,v in record.items() if k!='canonical_sha256'}
    require(record.get('canonical_sha256')==hashlib.sha256(canonical(unsigned)).hexdigest(),'SEALED_RECORD_DIGEST_CONFLICT')
    return record


def persist_state(root, state):
    commits=sorted(root.glob('state-commit-*.json'))
    previous=digest(commits[-1]) if commits else None
    unsigned={k:v for k,v in state.items() if k!='canonical_sha256'}
    unsigned['state_sequence']=len(commits)+1
    unsigned['canonical_sha256']=hashlib.sha256(canonical(unsigned)).hexdigest()
    raw=json.dumps(unsigned,sort_keys=True,indent=2).encode()+b'\n'
    seal(root,f"state-commit-{len(commits)+1:06d}.json",{'state_sha256':hashlib.sha256(raw).hexdigest(),'sequence':len(commits)+1,'previous_commit_sha256':previous})
    save(root/'state.json',unsigned)
    state.clear();state.update(unsigned)


def load_state(root,spec, approved_spec_path=None):
    state=verified_record(root/'state.json')
    commits=sorted(root.glob('state-commit-*.json'))
    require(commits and len(commits)==state.get('state_sequence'),'STATE_JOURNAL_CONFLICT')
    previous=None
    for sequence,path in enumerate(commits,1):
        commit=verified_record(path)
        require(path.name==f'state-commit-{sequence:06d}.json' and commit.get('sequence')==sequence and commit.get('previous_commit_sha256')==previous,'STATE_JOURNAL_CONFLICT')
        previous=digest(path)
    require(commit['state_sha256']==digest(root/'state.json'),'STATE_JOURNAL_CONFLICT')
    require(state['spec_sha256']==digest(approved_spec_path or args.spec),'RAW_STATE_SPEC_CONFLICT')
    require(state.get('source_revision')==spec['source_revision'] and state.get('storage_scope_hash')==spec['storage_scope_hash'] and state.get('canonical_membership_sha256')==spec['canonical_membership_sha256'],'RAW_STATE_BINDING_CONFLICT')
    names=[ref['file'] for ref in state['accepted_batches']]
    require(len(names)==len(set(names)) and set(names)=={p.name for p in root.glob('slice-*-batch-*.json')},'RAW_LEDGER_JOURNAL_CONFLICT')
    return state


def accepted_objects(root,state,spec,pinned):
    completed={}
    for ref in state['accepted_batches']:
        record=check_seal(root,ref)
        require(record['source_revision']==spec['source_revision'] and record['storage_scope_hash']==spec['storage_scope_hash'],'RAW_LEDGER_BINDING_CONFLICT')
        for name,item in record['objects'].items():
            require(name in pinned and name not in completed and item['content_sha256']==pinned[name]['content_sha256'] and item['metadata_token']==pinned[name]['metadata_token'],'RAW_LEDGER_CONTENT_CONFLICT')
            require(digest(root/'raw'/item['content_sha256'])==item['content_sha256'],'RAW_CACHE_REJECTED')
            completed[name]=item
    return completed


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'TRANSPORT_JOURNAL_INVALID')
        result[key] = value
    return result


def read_transport_journal(root, floor):
    """Read-only validation. Hash-chain consistency is not an authenticated signature."""
    keys = {'drive_get_attempts', 'oauth_attempts', 'observed_body_bytes'}
    require(set(floor) == keys and all(type(v) is int and v >= 0 for v in floor.values()),
            'TRANSPORT_COUNTER_REWIND')
    sequence, tail = 0, None
    counters = dict.fromkeys(keys, 0)
    events = {}
    path = Path(root) / 'transport-events.jsonl'
    if path.exists():
        with path.open('rb') as stream:
            for line in stream:
                require(line.endswith(b'\n'), 'TRANSPORT_JOURNAL_INVALID')
                try:
                    event = json.loads(line, object_pairs_hook=_unique_json_pairs)
                except (ValueError, TypeError):
                    raise RuntimeError('TRANSPORT_JOURNAL_INVALID') from None
                require(type(event) is dict and set(event) ==
                        {'sequence', 'previous_sha256', 'counters', 'canonical_sha256'},
                        'TRANSPORT_JOURNAL_INVALID')
                unsigned = {k: v for k, v in event.items() if k != 'canonical_sha256'}
                require(event['canonical_sha256'] == hashlib.sha256(canonical(unsigned)).hexdigest()
                        and type(event['sequence']) is int and event['sequence'] == sequence + 1
                        and event['previous_sha256'] == tail, 'TRANSPORT_JOURNAL_CONFLICT')
                current = event['counters']
                require(type(current) is dict and set(current) == keys and
                        all(type(current[k]) is int and current[k] >= counters[k] for k in keys),
                        'TRANSPORT_COUNTER_REWIND')
                sequence, tail, counters = event['sequence'], event['canonical_sha256'], current
                events[tail] = (sequence, dict(counters))
    require(all(counters[k] >= floor[k] for k in keys), 'TRANSPORT_COUNTER_REWIND')
    return {'sequence': sequence, 'tail': tail, 'counters': counters, 'events': events}


def validated_transport(root, floor, *, allow_one_event_prefix=False):
    journal = read_transport_journal(root, floor)
    if journal['sequence']:
        mirror = verified_record(Path(root) / 'transport-current.json')
        require(set(mirror) == {'counters', 'journal_tail_sha256', 'canonical_sha256'},
                'TRANSPORT_MIRROR_CONFLICT')
        prefix = journal['events'].get(mirror['journal_tail_sha256'])
        require(prefix is not None and mirror['counters'] == prefix[1], 'TRANSPORT_MIRROR_CONFLICT')
        gap = journal['sequence'] - prefix[0]
        require(gap == 0 or (allow_one_event_prefix and gap == 1), 'TRANSPORT_MIRROR_CONFLICT')
        journal.update(prefix_sequence=prefix[0], prefix_tail=mirror['journal_tail_sha256'],
                       published_counters=mirror['counters'], publication_gap_events=gap)
    else:
        require(not (Path(root) / 'transport-current.json').exists(), 'TRANSPORT_MIRROR_CONFLICT')
        journal.update(prefix_sequence=0, prefix_tail=None, publication_gap_events=0)
    return journal


class TransportBudget(dict):
    """Durable incurred actions; a strict loader never repairs predecessor files."""
    keys = ('drive_get_attempts', 'oauth_attempts', 'observed_body_bytes')
    def __init__(self, root, spec, floor):
        self.root, self.spec = Path(root), spec
        self.lock = threading.RLock()
        journal = validated_transport(root, floor)
        self.sequence, self.tail = journal['sequence'], journal['tail']
        self.last_published_tail = self.tail
        super().__init__(journal['counters'])
    def increment(self, key, amount):
        with self.lock:
            require(key in self.keys and type(amount) is int and amount >= 0,
                    'TRANSPORT_INCREMENT_INVALID')
            self[key] += amount
            unsigned = {'sequence': self.sequence + 1, 'previous_sha256': self.tail,
                        'counters': dict(self)}
            event = dict(unsigned, canonical_sha256=hashlib.sha256(canonical(unsigned)).hexdigest())
            operation = 'append_journal'
            try:
                with (self.root / 'transport-events.jsonl').open('ab') as stream:
                    stream.write(canonical(event) + b'\n')
                    operation = 'flush_journal'
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise PublicationError(dict(error_fields(exc), attempted_operation=operation,
                    file_role='transport_journal', local_retry_count=0, retry_elapsed_seconds=0.0,
                    attempted_journal_sequence=self.sequence + 1,
                    journal_sequence=self.sequence, journal_tail_sha256=self.tail,
                    last_published_mirror_tail=self.last_published_tail,
                    attempted_transport=dict(self),
                    journal_write_completion='UNVERIFIED_UNTIL_READ_ONLY_VALIDATION')) from None
            self.sequence += 1
            self.tail = event['canonical_sha256']
            try:
                self.mirror()
            except PublicationError as exc:
                exc.first_error.update(journal_sequence=self.sequence,
                    journal_tail_sha256=self.tail, last_published_mirror_tail=self.last_published_tail,
                    incurred_transport=dict(self))
                raise
    def mirror(self):
        unsigned = {'counters': dict(self), 'journal_tail_sha256': self.tail}
        save(self.root / 'transport-current.json',
             dict(unsigned, canonical_sha256=hashlib.sha256(canonical(unsigned)).hexdigest()))
        self.last_published_tail = self.tail


def guarded_session_factory(spec,meter,allowed_media,authorized_session,auth_state=None):
    from requests.adapters import HTTPAdapter
    folder=os.environ['PARLAYPICKER_DRIVE_FOLDER_ID']
    class ReadOnlyAdapter(HTTPAdapter):
        def __init__(self,token=False):
            super().__init__(max_retries=0);self.token=token
        def send(self,request,**kwargs):
            url=urlparse(request.url);query=parse_qs(url.query)
            media=query.get('alt')==['media']
            with meter.lock:
                if self.token:
                    require(request.method=='POST' and url.scheme=='https' and url.netloc=='oauth2.googleapis.com' and url.path=='/token' and not url.query,'UNAPPROVED_AUTH_ENDPOINT')
                    require(meter['oauth_attempts']<spec['max_oauth_token_attempts'],'OAUTH_REQUEST_LIMIT')
                    key='oauth_attempts'
                else:
                    require(request.method=='GET' and url.scheme=='https' and url.netloc=='www.googleapis.com','REMOTE_WRITE_OR_ENDPOINT_DENIED')
                    if media:
                        ident=url.path.rsplit('/',1)[-1]
                        require(ident in allowed_media and url.path=='/drive/v3/files/'+ident,'UNPINNED_MEDIA_DENIED')
                    elif url.path=='/drive/v3/files':
                        require(query.get('q')==[f"'{folder}' in parents and trashed = false"],'UNAPPROVED_INVENTORY_SCOPE')
                    else:
                        require(url.path=='/drive/v3/files/'+folder,'UNAPPROVED_METADATA_ENDPOINT')
                    require(meter['drive_get_attempts']<spec['max_drive_get_attempts'],'DRIVE_REQUEST_LIMIT')
                    key='drive_get_attempts'
                require(meter['observed_body_bytes']<spec['transport_observed_body_bytes_stop'],'BODY_BYTE_STOP')
                meter.increment(key,1)
            kwargs['stream']=True
            response=super().send(request,**kwargs)
            chunks=[];size=0
            primary_error=None
            try:
                for block in response.iter_content(65536):
                    size+=len(block)
                    with meter.lock:
                        meter.increment('observed_body_bytes',len(block))
                        require(meter['observed_body_bytes']<=spec['transport_observed_body_bytes_stop'],'BODY_BYTE_STOP')
                    require(not media or size<=spec['max_object_bytes'],'REMOTE_OBJECT_SIZE_LIMIT')
                    chunks.append(block)
                response._content=b''.join(chunks);response._content_consumed=True
                return response
            except BaseException as exc:
                primary_error=exc
                raise
            finally:
                # Every incurred increment already publishes its projection.
                # Cleanup must never restart an exhausted publication cycle.
                try:
                    response.close()
                except Exception as cleanup:
                    secondary=dict(error_fields(cleanup),
                        attempted_operation='close_response',file_role='http_response')
                    if primary_error is None:
                        cleanup.first_error=secondary
                        raise
                    details=getattr(primary_error,'first_error',None)
                    if details is None:
                        details=dict(error_fields(primary_error),
                            attempted_operation='consume_response_body',file_role='http_response')
                        primary_error.first_error=details
                    details['secondary_cleanup_error']=secondary
    def factory():
        if auth_state is None:
            session=authorized_session()
        else:
            session=auth_state.session(authorized_session)
        require(session._auth_request_session is not None,'AUTH_METER_UNAVAILABLE')
        for scheme in ('https://','http://'):
            session.mount(scheme,ReadOnlyAdapter())
            session._auth_request_session.mount(scheme,ReadOnlyAdapter(token=True))
        return session
    return factory


class CoordinatedCredentials:
    """In-memory delegate; library expiry and 401 rules remain authoritative.

    Separate request sessions share only this locked credential state. Each
    thread remembers the refresh generation used for its request: concurrent
    401s for the same old generation cause one refresh, while a current 401 still
    triggers the library's bounded refresh/retry route. No tokens are persisted.
    """
    def __init__(self, credentials):
        self.credentials=credentials
        self.lock=threading.RLock()
        self.local=threading.local()
        self.generation=0
    def __getattr__(self,name):
        return getattr(self.credentials,name)
    def before_request(self,request,method,url,headers):
        with self.lock:
            invalid_before=not self.credentials.valid
            before=(self.credentials.token,self.credentials.expiry)
            self.credentials.before_request(request,method,url,headers)
            if invalid_before or before!=(self.credentials.token,self.credentials.expiry):
                self.generation+=1
            self.local.used_generation=self.generation
    def refresh(self,request):
        with self.lock:
            used=getattr(self.local,'used_generation',None)
            if not self.credentials.valid or used is None or used==self.generation:
                self.credentials.refresh(request)
                self.generation+=1


class CaptureAuthState:
    """One capture-process lifetime, including all approved slice boundaries."""
    def __init__(self):
        self.credentials=None
        self.lock=threading.RLock()
    def session(self,authorized_session):
        from google.auth.transport.requests import AuthorizedSession
        with self.lock:
            if self.credentials is None:
                seed=authorized_session()
                # Injected plain sessions are supported by the pre-existing
                # transport regression harness. Real configuration constructs
                # AuthorizedSession through the unchanged application factory.
                if not isinstance(seed,AuthorizedSession):
                    return seed
                self.credentials=CoordinatedCredentials(seed.credentials)
                seed.close()
            session=AuthorizedSession(self.credentials,max_refresh_attempts=2)
            session.trust_env=False
            session._auth_request_session.trust_env=False
            return session


_capture_auth_states={}


def capture(spec, number):
    from app_core.evidence_drive import DriveStore, _authorized_session
    from app_core.read_only_census import _metadata_token
    cp, rp, pinned = anchor(spec)
    root = Path(spec['destination'])
    state = load_state(root,spec)
    require(state['next_slice'] == number, 'RAW_SLICE_ORDER_CONFLICT')
    require(not (root/f'slice-{number:02d}-attempt.json').exists(),'SLICE_RETRY_NOT_AUTHORIZED')
    if number>1:
        prior=verified_record(root/f'slice-{number-1:02d}-result.json')
        require(prior['status']=='PARTIAL' and prior['newly_retained_logical_objects']>0 and prior['terminal_reason'] in {'PROCESSING_DEADLINE','OBJECT_LIMIT','BYTE_LIMIT'},'PREVIOUS_SLICE_NOT_RESUMABLE')
    completed=accepted_objects(root,state,spec,pinned)
    baseline=len(completed)
    started=time.monotonic()
    meter=TransportBudget(root,spec,state['transport'])
    allowed_media=set()
    seal(root,f'slice-{number:02d}-attempt.json',{'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'accepted_reused_logical_objects':baseline,'transport_before':dict(meter),'status':'ATTEMPT_STARTED_NO_AUTOMATIC_RETRY'})
    auth_state=_capture_auth_states.setdefault(str(root.resolve()),CaptureAuthState())
    session_factory=guarded_session_factory(spec,meter,allowed_media,_authorized_session,auth_state)
    session=None
    try:
        session=session_factory()
        store=DriveStore(os.environ['PARLAYPICKER_DRIVE_FOLDER_ID'],session=session)
        store._session_factory=session_factory
        require(store.storage_scope_hash()==spec['storage_scope_hash'],'SECURE_STORAGE_BINDING_CONFLICT')
        inventory=store.discover_complete_inventory(namespace=spec['namespace'])
        groups=defaultdict(list)
        for item in inventory.files:
            if item['name'].startswith(spec['namespace']): groups[item['name']].append(item)
        for name,entry in pinned.items():
            require(name in groups,'PINNED_OBJECT_MISSING')
            require(_metadata_token(groups[name])==entry['metadata_token'],'PINNED_METADATA_CHANGED')
            require(all(x.get('sha256Checksum') in (None,'',entry['content_sha256']) for x in groups[name]),'PROVIDER_CHECKSUM_CONFLICT')
            allowed_media.update(x['id'] for x in groups[name])
        seal(root,f'slice-{number:02d}-inventory.json',{'source_revision':spec['source_revision'],'storage_scope_hash':store.storage_scope_hash(),'credential_principal_identifier':session.credentials.service_account_email,'operation_id':inventory.operation_id,'anchor_operation_id':cp['operation_id'],'anchor_as_of':spec['as_of'],'fresh_canonical_objects':len(groups),'added_names_excluded':sorted(set(groups)-set(pinned)),'pinned_missing':[],'pinned_metadata_changes':[],'listing_pages':inventory.listing_pages,'metadata_items_seen':inventory.metadata_items_seen})
        pending=sorted(set(pinned)-set(completed));metrics=Counter();batch_number=max([int(p.stem.rsplit('-',1)[1]) for p in root.glob(f'slice-{number:02d}-batch-*.json')],default=0);reason=None
        while pending:
            budget_disk(root,spec)
            if time.monotonic()-started>=spec['slice_processing_seconds']: reason='PROCESSING_DEADLINE';break
            if len(completed)-baseline>=spec['max_new_objects_per_slice']: reason='OBJECT_LIMIT';break
            if metrics['bytes_downloaded']>=spec['soft_bytes_per_slice']: reason='BYTE_LIMIT';break
            if meter['observed_body_bytes']>=spec['soft_bytes_total']: reason='BLOCK_BYTE_LIMIT';break
            selected=pending[:min(spec['batch_objects'],spec['max_new_objects_per_slice']-(len(completed)-baseline))]
            values=store.read_verified_prefixes(Prefixes=selected,inventory=inventory,cache_dir=root/'raw',full_verify=False)
            objects={}
            for name in selected:
                records=values[name]
                require(len(records)==1 and records[0][0]==name,'EXACT_NAME_READ_CONFLICT')
                raw=records[0][1]
                require(hashlib.sha256(raw).hexdigest()==pinned[name]['content_sha256'],'PINNED_CONTENT_CHANGED')
                target=root/'raw'/pinned[name]['content_sha256']
                if not target.exists():
                    temporary=target.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(target)
                require(digest(target)==pinned[name]['content_sha256'],'RAW_RETENTION_CONFLICT')
                objects[name]={'content_sha256':pinned[name]['content_sha256'],'metadata_token':pinned[name]['metadata_token'],'bytes':len(raw)}
            read_metrics=asdict(store.last_read_report)
            for key in ('bytes_downloaded','objects_downloaded','objects_reused','retries','retry_wait_ms'): metrics[key]+=read_metrics[key]
            batch_number+=1
            ref=seal(root,f'slice-{number:02d}-batch-{batch_number:04d}.json',{'source_revision':spec['source_revision'],'storage_scope_hash':store.storage_scope_hash(),'operation_id':inventory.operation_id,'objects':objects,'read_metrics':read_metrics,'transport':dict(meter)})
            state['accepted_batches'].append(ref);state['transport']=dict(meter)
            persist_state(root,state)
            completed.update(objects);pending=pending[len(selected):]
        require(len(completed)>baseline or not pending,'NO_FORWARD_PROGRESS')
        result={'status':'PARTIAL' if pending else 'RAW_COMPLETE','terminal_reason':reason,'source_revision':spec['source_revision'],'storage_scope_hash':store.storage_scope_hash(),'slice':number,'accepted_reused_logical_objects':baseline,'newly_retained_logical_objects':len(completed)-baseline,'cumulative_objects':len(completed),'remaining_objects':len(pending),'physical_metrics':dict(metrics),'transport':dict(meter),'elapsed_seconds':time.monotonic()-started,'soft_byte_overshoot':max(0,metrics['bytes_downloaded']-spec['soft_bytes_per_slice']),'processing_deadline_overshoot':max(0,time.monotonic()-started-spec['slice_processing_seconds'])}
        budget_disk(root,spec,force=True)
        seal(root,f'slice-{number:02d}-result.json',result)
        state['next_slice']=number+1;state['transport']=dict(meter);persist_state(root,state)
    finally:
        if session is not None: session.close()


def capture_block(spec):
    """Internal slices share auth state; supervisor still enforces worker wall.

    A thread watchdog exits this capture process at a slice wall boundary,
    including when an underlying HTTP call is blocked. It never resumes or
    starts another slice after failure. OS process exit ends worker threads.
    """
    begun=time.monotonic()
    root=Path(spec['destination'])
    complete=False
    for number in range(1,spec['max_slices']+1):
        remaining=spec['capture_total_wall_seconds']-(time.monotonic()-begun)
        if remaining<=0: break
        deadline=time.monotonic()+min(spec['slice_wall_seconds'],remaining)
        done=threading.Event()
        def watch():
            if not done.wait(max(0,deadline-time.monotonic())):
                print('STAGE_WALL_LIMIT',file=sys.stderr,flush=True)
                os._exit(4)
        watchdog=threading.Thread(target=watch,daemon=True)
        watchdog.start()
        try:
            capture(spec,number)
            require(time.monotonic()<=deadline,'STAGE_WALL_LIMIT')
        finally:
            done.set();watchdog.join()
        result=verified_record(root/f'slice-{number:02d}-result.json')
        if result['status']=='RAW_COMPLETE': complete=True;break
        require(result['terminal_reason'] in {'PROCESSING_DEADLINE','OBJECT_LIMIT','BYTE_LIMIT','BLOCK_BYTE_LIMIT'},'UNEXPECTED_PARTIAL_STOP')
        if result['terminal_reason']=='BLOCK_BYTE_LIMIT' or result['transport']['observed_body_bytes']>=spec['soft_bytes_total']: break
    seal(root,'capture-block-result.json',{'status':'RAW_COMPLETE' if complete else 'PARTIAL','slices_attempted':number,'transport':dict(TransportBudget(root,spec,load_state(root,spec)['transport'])),'reason':None if complete else 'APPROVED_CAPTURE_BLOCK_STOP','credentials_persisted':False})


def materialize(spec):
    from app_core import prospective_remote as codec, prospective_evidence as evidence
    from app_core.prospective_reconciliation import ensure_reconciliation_schema
    from app_core.canonical_schema import CANONICAL_PRIMARY_KEYS
    cp, rp, pinned = anchor(spec)
    root = Path(spec['destination'])
    state = load_state(root,spec)
    accepted = accepted_objects(root,state,spec,pinned)
    require(set(accepted) == set(pinned), 'FULL_RAW_CORPUS_NOT_ACCEPTED')
    latest=verified_record(root/f"slice-{state['next_slice']-1:02d}-result.json")
    require(latest['status']=='RAW_COMPLETE' and latest['cumulative_objects']==len(pinned),'RAW_COMPLETE_RESULT_REQUIRED')
    path = root/'canonical.unaccepted.sqlite3'
    require(not path.exists() and not (root/'canonical.sqlite3').exists(), 'SNAPSHOT_ALREADY_EXISTS')
    ensure_reconciliation_schema(path)
    expected = Counter(f['record_type'] for entry in pinned.values() for f in entry['facts'])
    groups = defaultdict(list)
    for key in pinned:
        groups[key[len(spec['namespace']):].split('/',1)[0]].append(key)
    with closing(evidence.connect(path)) as db:
        db.execute('PRAGMA journal_mode=DELETE')
        page_size=db.execute('PRAGMA page_size').fetchone()[0]
        db.execute('PRAGMA max_page_count='+str(spec['max_snapshot_input_bytes']//page_size))
        schema = codec._schema(db)
        require(set(schema) == set(CANONICAL_PRIMARY_KEYS), 'SCHEMA_CLOSURE_CONFLICT')
        db.execute('BEGIN IMMEDIATE')
        db.execute('PRAGMA defer_foreign_keys=ON')
        for table in codec._table_order(db,schema):
            columns, primary = schema[table]
            for key in sorted(groups[table]):
                budget_disk(root, spec)
                raw_path = root/'raw'/pinned[key]['content_sha256']
                require(digest(raw_path) == pinned[key]['content_sha256'], 'FULL_RAW_HASH_CONFLICT')
                decoded_table,row = codec._decode(key,raw_path.read_bytes(),schema)
                require(decoded_table == table,'DECODE_TABLE_CONFLICT')
                # Actual reconciled schema uses payload/payload_hash, already
                # checked by the existing codec. It has no evidence_json field.
                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",row)
        require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'DEPENDENCY_CLOSURE_FAILED')
        db.commit()
        require([tuple(row) for row in db.execute('PRAGMA integrity_check')]==[('ok',)],'SQLITE_INTEGRITY_FAILED')
        actual = {}
        replayed = set()
        for table,(columns,primary) in schema.items():
            actual[table] = db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            require(actual[table] == expected[table], 'RESTORED_COUNT_CONFLICT')
            for row in db.execute(f"SELECT {','.join(columns)} FROM {table}"):
                key, raw = codec._encode(table,columns,primary,tuple(row))
                require(key in pinned and hashlib.sha256(raw).hexdigest() == pinned[key]['content_sha256'], 'DATABASE_WIRE_REPLAY_FAILED')
                replayed.add(key)
        require(replayed == set(pinned),'DATABASE_MEMBERSHIP_CONFLICT')
    require(path.stat().st_size <= spec['max_snapshot_input_bytes'], 'SNAPSHOT_INPUT_SIZE_LIMIT')
    require(not any(Path(str(path)+suffix).exists() for suffix in ('-wal','-journal')), 'SNAPSHOT_SIDECAR_CONFLICT')
    budget_disk(root,spec,force=True)
    target=root/'canonical.sqlite3';path.replace(target)
    seal(root, 'snapshot-acceptance.json', {'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'anchor_run_id':spec['anchor_run_id'],'anchor_as_of':spec['as_of'],'canonical_membership_sha256':spec['canonical_membership_sha256'],'canonical_objects':len(pinned),'snapshot_sha256':digest(target),'snapshot_bytes':target.stat().st_size,'canonical_table_counts':actual,'foreign_key_violations':0,'sqlite_integrity':'ok','wire_replay':'ALL_PINNED_BYTES_MATCH','qualification':'NOT_RUN'})


def assess(spec):
    from app_core import prospective_evidence as evidence
    from app_core import football_stage2
    from app_core.canonical_schema import CANONICAL_PRIMARY_KEYS
    from app_core.census_eligibility import ACTIVE_SQL, ELIGIBLE_SQL
    cp,rp,pinned=anchor(spec)
    root=Path(spec['destination']);path=root/'canonical.sqlite3'
    accepted=verified_record(root/'snapshot-acceptance.json')
    expected=Counter(f['record_type'] for item in pinned.values() for f in item['facts'])
    require(accepted.get('source_revision')==spec['source_revision'] and accepted.get('storage_scope_hash')==spec['storage_scope_hash'] and accepted.get('canonical_membership_sha256')==spec['canonical_membership_sha256'] and accepted.get('anchor_as_of')==spec['as_of'] and accepted.get('anchor_run_id')==spec['anchor_run_id'],'SNAPSHOT_ACCEPTANCE_BINDING_CONFLICT')
    require(accepted.get('canonical_objects')==len(pinned) and accepted.get('wire_replay')=='ALL_PINNED_BYTES_MATCH' and accepted.get('sqlite_integrity')=='ok' and accepted.get('foreign_key_violations')==0,'SNAPSHOT_ACCEPTANCE_INCOMPLETE')
    require(not any(Path(str(path)+suffix).exists() for suffix in ('-wal','-journal')),'ASSESSMENT_SIDECAR_CONFLICT')
    before=digest(path)
    require(before==accepted['snapshot_sha256'],'ASSESSMENT_INPUT_REJECTED')
    require(path.stat().st_size<=spec['max_snapshot_input_bytes'],'ASSESSMENT_INPUT_LIMIT')
    uri=path.resolve().as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri,uri=True)) as db:
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        counts={table:db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in CANONICAL_PRIMARY_KEYS}
        require(counts==accepted['canonical_table_counts'] and counts=={table:expected[table] for table in CANONICAL_PRIMARY_KEYS},'ASSESSMENT_COUNT_CONFLICT')
        rows=football_stage2._selected_rows(db)
        issues=[]
        for row in rows:
            replayed,blockers=football_stage2._verify_selected(db,row)
            issues.append({'training_row_id':row['training_row_id'],'blockers':blockers,'replay_valid':replayed is not None})
        require(all(not item['blockers'] for item in issues),'FOOTBALL_SOURCE_REPLAY_FAILED')
        manifests={}
        for sport in ('NFL','NCAAF'):
            for market in ('SPREAD','TOTAL'):
                selected=[row for row in rows if (row['sport'],row['market_family'])==(sport,market)]
                parameters=(sport,market)
                raw=db.execute("SELECT COUNT(*) FROM prospective_football_training_row WHERE sport=? AND market_family=? AND training_row_status='TRAINING_READY'",parameters).fetchone()[0]
                active=db.execute('SELECT COUNT(*) FROM ('+ACTIVE_SQL+') WHERE sport=? AND market_family=?',parameters).fetchone()[0]
                eligible=db.execute('WITH active AS ('+ACTIVE_SQL+') SELECT COUNT(*) FROM ('+ELIGIBLE_SQL+') WHERE sport=? AND market_family=?',parameters).fetchone()[0]
                manifests[sport+'/'+market]={'stored_ready_rows':raw,'active_training_rows':active,'active_eligible_rows':eligible,'independent_games':len({row['game_id'] for row in selected}),'game_ids':sorted({row['game_id'] for row in selected}),'training_row_ids':sorted(row['training_row_id'] for row in selected)}
                reported=next(x for x in rp['scopes'] if x['scope']==sport+'/'+market)
                require(sorted('football-one-observation-v1:'+row['training_row_id'] for row in selected)==sorted(reported['eligibility']['manifest_ids']),'MANIFEST_REPLAY_CONFLICT')
    plans=evidence.read_records(path,'prospective_validation_plan')
    models=evidence.read_records(path,'prospective_model')
    calibration=evidence.read_records(path,'prospective_calibration')
    expected_plans={f['validation_plan_id']:f['artifact_hash'] for entry in pinned.values() for f in entry['facts'] if f['record_type']=='prospective_validation_plan'}
    require({p['validation_plan_id']:p['artifact_hash'] for p in plans}==expected_plans,'PLAN_IDENTITIES_CHANGED')
    reports=[]
    for plan in plans:
        record={'validation_plan_id':plan['validation_plan_id'],'model_id':plan['model_id'],'calibration_id':plan['calibration_id'],'as_of':spec['as_of'],'governing_release_route':'UNKNOWN'}
        if not plan['model_id'] or not plan['calibration_id']:
            record.update(status='BLOCKED_MISSING_BINDING',cohorts='UNKNOWN',metrics='NOT_COMPUTED')
        else:
            record['evaluation']=evidence.evaluate_validation_plan(path,plan['validation_plan_id'],as_of=spec['as_of'])
        reports.append(record)
    require(digest(path)==before,'READ_ONLY_SNAPSHOT_CHANGED')
    require(not any(Path(str(path)+suffix).exists() for suffix in ('-wal','-journal')),'ASSESSMENT_SIDECAR_CONFLICT')
    scope_reports=[]
    for original in rp['scopes']:
        scope=original['scope'];sport,market=scope.split('/')
        model_records=[r for r in models if (r['sport'],r['market_family'])==(sport,market)]
        calibration_records=[r for r in calibration if (r['sport'],r['market_family'])==(sport,market)]
        scope_reports.append({'scope':scope,'canonical_model_records':model_records,'canonical_calibration_records':calibration_records,'plan_ids':[p['validation_plan_id'] for p in plans if (p['sport'],p['market_family'])==(sport,market)],'football_manifest':manifests.get(scope,{'independent_games':None,'status':'UNKNOWN_READER_UNSUPPORTED'}),'bound_cohorts':'UNKNOWN_WITHOUT_EXACT_BINDING_AND_TRAINING_LINEAGE','governing_release_route':'UNKNOWN','qualification':'BLOCKED_MISSING_CANONICAL_MODEL_OR_CALIBRATION' if not model_records or not calibration_records else 'REQUIRES_BOUND_EVALUATION_REVIEW','production_eligible':False})
    seal(root,'local-assessment.json',{'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'as_of':spec['as_of'],'snapshot_sha256_before':before,'snapshot_sha256_after':digest(path),'canonical_counts':counts,'models':models,'calibrations':calibration,'plans':plans,'plan_assessments':reports,'scopes':scope_reports,'football_manifests':manifests,'football_source_replay_issues':issues,'eight_non_football_manifest_calculations':'UNKNOWN_READER_UNSUPPORTED','training_selection_calibration_partitions':'UNKNOWN_WITHOUT_EXACT_MODEL_TRAINING_LINEAGE','production_eligible':False,'recommended_stake':0,'owner_release_authority':False})


def supervise(spec):
    begun=time.monotonic()
    require(args.approved_spec_sha256==digest(args.spec),'EXACT_SPEC_APPROVAL_HASH_REQUIRED')
    require(all(os.environ.get(k) for k in ('PARLAYPICKER_DRIVE_FOLDER_ID','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')),'BLOCKED_SECURE_CONFIGURATION')
    root=Path(spec['destination']).resolve()
    if getattr(args,'recovery_spec',None):
        require(args.approved_recovery_sha256==digest(args.recovery_spec),'EXACT_RECOVERY_APPROVAL_HASH_REQUIRED')
        require(args.approved_driver_sha256==digest(__file__),'EXACT_DRIVER_APPROVAL_HASH_REQUIRED')
        recovery=json.loads(args.recovery_spec.read_bytes())
        require(recovery['replacement_driver_sha256']==digest(__file__),'RECOVERY_DRIVER_CONFLICT')
        return supervise_recovery(spec,recovery)
    expected=Path(os.environ['LOCALAPPDATA'])/'ParlayPicker'/'qualification-canonical-snapshot'/spec['source_revision']/f"acquire-{spec['anchor_run_id']}-v1"
    require(root==expected.resolve() and not root.exists(),'ISOLATED_DESTINATION_CONFLICT_OR_RESUME_NOT_AUTHORIZED')
    require(shutil.disk_usage(root.anchor).free>=spec['required_initial_free_disk_bytes'],'INITIAL_FREE_DISK_LIMIT')
    root.mkdir(parents=True);(root/'raw').mkdir()
    worker_token=secrets.token_hex(32)
    persist_state(root,{'spec_sha256':digest(args.spec),'worker_token_sha256':hashlib.sha256(worker_token.encode()).hexdigest(),'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'canonical_membership_sha256':spec['canonical_membership_sha256'],'next_slice':1,'accepted_batches':[],'transport':{'drive_get_attempts':0,'oauth_attempts':0,'observed_body_bytes':0}})
    return run_approved_workers(spec,root,worker_token,begun)


def run_approved_workers(spec,root,worker_token,begun):
    def run_worker(phase,limit,number=None):
        source_check(spec)
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--spec',str(Path(args.spec).resolve()),'--approved-spec-sha256',args.approved_spec_sha256,'--worker',phase]
        if number is not None: command+=['--slice-number',str(number)]
        stamp=f'{phase}-{number or 1}'
        with (root/(stamp+'.log')).open('wb') as log:
            worker_env=dict(os.environ,PARLAYPICKER_OPERATIONS_WORKER_TOKEN=worker_token)
            process=subprocess.Popen(command,stdout=log,stderr=log,env=worker_env)
            deadline=time.monotonic()+limit
            first_slice_deadline=time.monotonic()+min(limit,spec['slice_wall_seconds'])
            try:
                while process.poll() is None:
                    require(time.monotonic()<=deadline,'STAGE_WALL_LIMIT')
                    if phase=='capture-block' and not (root/'slice-01-result.json').exists():
                        require(time.monotonic()<=first_slice_deadline,'STAGE_WALL_LIMIT')
                    require(time.monotonic()-begun<=spec['total_wall_seconds'],'TOTAL_WALL_LIMIT')
                    budget_disk(root,spec)
                    time.sleep(1)
                require(time.monotonic()<=deadline,'STAGE_WALL_LIMIT')
                if phase=='capture-block' and not (root/'slice-01-result.json').exists():
                    require(time.monotonic()<=first_slice_deadline,'STAGE_WALL_LIMIT')
                require(time.monotonic()-begun<=spec['total_wall_seconds'],'TOTAL_WALL_LIMIT')
                budget_disk(root,spec,force=True)
                require(process.returncode!=4,'STAGE_WALL_LIMIT')
                require(process.returncode==0,'WORKER_BLOCKED_NO_RETRY')
            except BaseException:
                process.kill();process.wait()
                raise
    try:
        remaining_capture=spec['capture_total_wall_seconds']-(time.monotonic()-begun)
        if remaining_capture<=0:
            seal(root,'operation-result.json',{'status':'PARTIAL','reason':'APPROVED_CAPTURE_BLOCK_STOP','elapsed_seconds':time.monotonic()-begun,'snapshot_materialized':False,'qualification_executed':False,'further_execution_authorized':False})
            return 2
        run_worker('capture-block',remaining_capture)
        capture_outcome=verified_record(root/'capture-block-result.json')
        full=capture_outcome['status']=='RAW_COMPLETE'
        if not full:
            seal(root,'operation-result.json',{'status':'PARTIAL','reason':'APPROVED_CAPTURE_BLOCK_STOP','elapsed_seconds':time.monotonic()-begun,'snapshot_materialized':False,'qualification_executed':False,'further_execution_authorized':False})
            return 2
        run_worker('materialize',spec['materialization_wall_seconds'])
        run_worker('assess',spec['assessment_wall_seconds'])
        budget_disk(root,spec,force=True)
        seal(root,'operation-result.json',{'status':'SNAPSHOT_AND_ASSESSMENT_COMPLETE','elapsed_seconds':time.monotonic()-begun,'inventory_completion_is_model_qualification':False,'production_eligible':False,'recommended_stake':0})
        return 0
    except BaseException as exc:
        # Do not serialize credential-bearing exception details or HTTP headers.
        reason=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
        seal(root,'operation-blocked.json',{'status':'BLOCKED','reason':reason,'elapsed_seconds':time.monotonic()-begun,'in_flight_transfer_and_disk_overrun':'UNKNOWN if interrupted; inspect retained meter and files','no_retry_authorized':True,'no_cold_restart_authorized':True})
        return 3


def sanitized_reason(exc):
    if isinstance(exc,RuntimeError):
        message=str(exc)
        if message and len(message)<=100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in message): return message
    from google.auth.exceptions import RefreshError
    if isinstance(exc,RefreshError):return 'OAUTH_REFRESH_FAILED'
    return 'WORKER_'+type(exc).__name__.upper()+'_FAILED'


def retain_worker_failure(root, spec, stage, reason, exc=None):
    """Retain the first error without depending on successful mirror loading."""
    report = {'stage': stage, 'reason': reason, 'retry_authorized': False,
              'snapshot_accepted': (root / 'snapshot-acceptance.json').is_file(),
              'assessment_report_present': (root / 'local-assessment.json').is_file()}
    details = getattr(exc, 'first_error', None)
    if details:
        report['first_error'] = dict(details, stage=stage)
    elif exc is not None:
        report['first_error'] = dict(error_fields(exc), stage=stage,
            attempted_operation='UNKNOWN_NOT_RECORDED', file_role='UNKNOWN_NOT_RECORDED')
    try:
        state = verified_record(root / 'state.json')
        # Diagnostic ledger count, not authority for future cache reuse.
        report['accepted_objects'] = sum(len(check_seal(root, ref)['objects'])
                                        for ref in state['accepted_batches'])
        journal = read_transport_journal(root, state['transport'])
        report['transport'] = journal['counters']
        report['journal_sequence'] = journal['sequence']
        report['journal_tail_sha256'] = journal['tail']
        try:
            mirror = verified_record(root / 'transport-current.json')
            report['last_published_mirror_tail'] = mirror.get('journal_tail_sha256')
            report['mirror_matches_journal'] = (mirror.get('counters') == journal['counters']
                and mirror.get('journal_tail_sha256') == journal['tail'])
        except Exception:
            report['last_published_mirror_tail'] = 'UNKNOWN_INVALID_OR_MISSING'
    except Exception as diagnostic:
        report['diagnostic_incomplete_reason'] = sanitized_reason(diagnostic)
    try:
        seal(root, 'worker-blocked-' + stage + '.json', report)
    except Exception:
        # Only sanitized allowlisted fields reach stderr; preserve the first error.
        print(json.dumps(dict(report, diagnostic_record_retained=False)), file=sys.stderr, flush=True)
    return report


def inspect_predecessor(root, original_spec_path, spec, expected=None):
    """Read-only verification; no wire requests, migration or orphan admission."""
    root=Path(root);original_spec_path=Path(original_spec_path)
    original=json.loads(original_spec_path.read_bytes())
    require(digest(original_spec_path)==digest(args.spec),'RECOVERY_ORIGINAL_SPEC_CONFLICT')
    for key in ('source_revision','source_tree','storage_scope_hash','canonical_membership_sha256','as_of','anchor_run_id'):
        require(original[key]==spec[key],'RECOVERY_BINDING_CONFLICT')
    cp,rp,pinned=anchor(original)
    state=load_state(root,original,original_spec_path)
    accepted=accepted_objects(root,state,original,pinned)
    require((root/'transport-events.jsonl').is_file() and (root/'transport-current.json').is_file(),'RECOVERY_DURABLE_TRANSPORT_MISSING')
    meter=TransportBudget(root,original,state['transport'])
    blocked=verified_record(root/'operation-blocked.json')
    require(blocked['status']=='BLOCKED' and blocked['reason']=='WORKER_BLOCKED_NO_RETRY','RECOVERY_TERMINAL_CONFLICT')
    inventories=sorted(root.glob('slice-*-inventory.json'))
    require(inventories,'RECOVERY_INVENTORY_MISSING')
    inventory=verified_record(inventories[-1])
    require(inventory['source_revision']==spec['source_revision'] and inventory['storage_scope_hash']==spec['storage_scope_hash'] and inventory['anchor_as_of']==spec['as_of'],'RECOVERY_INVENTORY_BINDING_CONFLICT')
    cache={}
    for path in (root/'raw').iterdir():
        require(path.is_file() and len(path.name)==64 and digest(path)==path.name,'RECOVERY_CACHE_CONFLICT')
        cache[path.name]=path.stat().st_size
    accepted_hashes={item['content_sha256'] for item in accepted.values()}
    files={str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file()}
    result={'original_operation_id':inventory['operation_id'],'accepted_objects':accepted,'accepted_count':len(accepted),'unledgered_cache':{k:v for k,v in cache.items() if k not in accepted_hashes},'incurred_transport':dict(meter),'accepted_media_bytes':sum(item['bytes'] for item in accepted.values()),'retained_cache_bytes':sum(cache.values()),'total_transferred_media_bytes':'UNKNOWN_INTERRUPTED_BATCH','state_sha256':digest(root/'state.json'),'journal_sha256':digest(root/'transport-events.jsonl'),'journal_tail_sha256':meter.tail,'original_spec_sha256':digest(original_spec_path),'files':files,'attempts_used':len(list(root.glob('slice-*-attempt.json')))}
    if expected:
        for key in ('original_operation_id','accepted_count','incurred_transport','state_sha256','journal_sha256','journal_tail_sha256'):
            require(result[key]==expected[key],'RECOVERY_PREDECESSOR_IDENTITY_CONFLICT')
        require(hashlib.sha256(canonical(files)).hexdigest()==expected['file_register_canonical_sha256'],'RECOVERY_FILE_REGISTER_CONFLICT')
    return result


def file_register(root):
    return {str(p.relative_to(root)).replace('\\', '/'):
            {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()}


def initiated_slices(root, spec):
    markers = sorted(root.glob('slice-*-attempt.json'))
    require(markers, 'RECOVERY_ATTEMPTS_MISSING')
    for index, path in enumerate(markers, 1):
        record = verified_record(path)
        require(path.name == f'slice-{index:02d}-attempt.json'
                and record['status'] == 'ATTEMPT_STARTED_NO_AUTOMATIC_RETRY',
                'RECOVERY_ATTEMPT_CHAIN_CONFLICT')
        require(record.get('source_revision') == spec['source_revision']
                and record.get('storage_scope_hash') == spec['storage_scope_hash'],
                'RECOVERY_ATTEMPT_BINDING_CONFLICT')
    return len(markers)


def inspect_linked_predecessor(spec, recovery):
    """Offline v1 -> v2 verification. Never fix/copy a projection in either root."""
    require(recovery.get('recovery_generation') == 3, 'RECOVERY_GENERATION_INVALID')
    require(recovery.get('additional_oauth_allowance') == 0, 'RECOVERY_ALLOWANCE_INVALID')
    original_path = Path(args.spec)
    addendum_path = Path(recovery['v2_recovery_spec_path'])
    require(digest(original_path) == recovery['original_spec_sha256'],
            'RECOVERY_ORIGINAL_SPEC_CONFLICT')
    require(digest(addendum_path) == recovery['v2_recovery_spec_sha256'],
            'RECOVERY_ADDENDUM_CHAIN_CONFLICT')
    v2 = json.loads(addendum_path.read_bytes())
    require(v2['original_spec_sha256'] == digest(original_path)
            and digest(v2['original_driver_path']) == v2['original_driver_sha256']
            and digest(recovery['v2_driver_path']) == v2['replacement_driver_sha256'],
            'RECOVERY_DRIVER_CHAIN_CONFLICT')
    original_driver_identity = recovery.get('original_driver_sha256')
    require(type(original_driver_identity) is str and len(original_driver_identity) == 64
            and all(c in '0123456789abcdef' for c in original_driver_identity),
            'RECOVERY_ORIGINAL_DRIVER_IDENTITY_INVALID')
    require(original_driver_identity == v2['original_driver_sha256'],
            'RECOVERY_ORIGINAL_DRIVER_ANCESTRY_CONFLICT')
    original_root = Path(v2['predecessor_destination'])
    root = Path(recovery['predecessor_destination'])
    require(root.resolve() == Path(v2['successor_destination']).resolve()
            and original_root.resolve() != root.resolve(), 'RECOVERY_ANCESTRY_CONFLICT')
    history = inspect_predecessor(original_root, original_path, spec, v2['expected_predecessor'])
    old_block = verified_record(original_root / 'operation-blocked.json')
    require(math.isclose(v2['historical_elapsed_seconds'], old_block['elapsed_seconds'],
                         abs_tol=1e-6), 'RECOVERY_ELAPSED_CHAIN_CONFLICT')
    expected_effective = recovery_policy(spec, history, v2['additional_oauth_allowance'],
                                         v2['successor_max_slices'])
    expected_effective.update(destination=v2['successor_destination'], recovery_mode=True,
        working_disk_predecessor_bytes=sum(item['bytes'] for item in history['files'].values()))
    for key in ('total_wall_seconds', 'capture_total_wall_seconds'):
        expected_effective[key] -= v2['historical_elapsed_seconds']
    effective_path = root / 'effective-spec.json'
    effective = json.loads(effective_path.read_bytes())
    require(effective == expected_effective, 'RECOVERY_EFFECTIVE_SPEC_CONFLICT')
    link = verified_record(root / 'recovery-link.json')
    require(link['original_operation_id'] == history['original_operation_id']
            and link['original_state_sha256'] == history['state_sha256']
            and link['original_journal_sha256'] == history['journal_sha256']
            and link['original_journal_tail_sha256'] == history['journal_tail_sha256']
            and link['original_spec_sha256'] == digest(original_path)
            and link['recovery_spec_sha256'] == digest(addendum_path)
            and link['original_driver_sha256'] == v2['original_driver_sha256']
            and link['replacement_driver_sha256'] == v2['replacement_driver_sha256']
            and link['incurred_transport'] == history['incurred_transport']
            and link['accepted_reused_objects'] == history['accepted_count']
            and link['additional_oauth_allowance'] == v2['additional_oauth_allowance']
            and link['combined_oauth_limit'] == effective['max_oauth_token_attempts']
            and link['original_files'] == history['files'],
            'RECOVERY_LINK_BINDING_CONFLICT')
    # Verify the immutable copied v1 provenance separately from top-level v2 attempts.
    for name, item in history['files'].items():
        require(digest(root / 'predecessor-evidence' / name) == item['sha256'],
                'RECOVERY_PROVENANCE_CONFLICT')
    state = load_state(root, effective, effective_path)
    require(state['predecessor_operation_id'] == history['original_operation_id']
            and state['predecessor_state_sha256'] == history['state_sha256']
            and state['recovery_addendum_sha256'] == digest(addendum_path),
            'RECOVERY_STATE_CHAIN_CONFLICT')
    _, _, pinned = anchor(spec)
    accepted = accepted_objects(root, state, effective, pinned)  # Each payload hash, not size.
    journal = validated_transport(root, state['transport'], allow_one_event_prefix=True)
    # The complete inherited journal is an exact byte prefix, not just a numeric floor.
    inherited = (original_root / 'transport-events.jsonl').read_bytes()
    with (root / 'transport-events.jsonl').open('rb') as stream:
        require(stream.read(len(inherited)) == inherited, 'RECOVERY_JOURNAL_ANCESTRY_CONFLICT')
    block = verified_record(root / 'operation-blocked.json')
    require(block['status'] == 'BLOCKED' and block['reason'] == 'WORKER_BLOCKED_NO_RETRY',
            'RECOVERY_TERMINAL_CONFLICT')
    first_attempts = initiated_slices(original_root, spec)
    second_attempts = initiated_slices(root, effective)
    require(first_attempts == history['attempts_used'], 'RECOVERY_ATTEMPT_CHAIN_CONFLICT')
    require(second_attempts <= effective['max_slices']
            and state['next_slice'] == second_attempts, 'RECOVERY_ATTEMPT_CHAIN_CONFLICT')
    for index in range(1, second_attempts):
        result = verified_record(root / f'slice-{index:02d}-result.json')
        require(result['slice'] == index and result['status'] == 'PARTIAL'
                and result['newly_retained_logical_objects'] > 0
                and result['terminal_reason'] in {'OBJECT_LIMIT', 'BYTE_LIMIT', 'PROCESSING_DEADLINE'}
                and result['source_revision'] == spec['source_revision']
                and result['storage_scope_hash'] == spec['storage_scope_hash'],
                'RECOVERY_RESULT_CHAIN_CONFLICT')
    require(not (root / f'slice-{second_attempts:02d}-result.json').exists(),
            'RECOVERY_TERMINAL_CONFLICT')
    charged_elapsed = old_block['elapsed_seconds'] + block['elapsed_seconds']
    require(math.isfinite(charged_elapsed) and charged_elapsed >= 0
            and math.isclose(charged_elapsed, recovery['historical_elapsed_seconds'],
                             abs_tol=1e-6), 'RECOVERY_ELAPSED_CHAIN_CONFLICT')
    files = file_register(root)
    accepted_hashes = {item['content_sha256'] for item in accepted.values()}
    orphans = {p.name: p.stat().st_size for p in (root / 'raw').iterdir()
               if p.is_file() and p.name not in accepted_hashes}
    # No separate top-level v2 UUID exists: the sealed link digest is its identity.
    result = dict(original_operation_id=history['original_operation_id'],
        immediate_predecessor_identity={'kind': 'SEALED_RECOVERY_LINK_SHA256',
                                       'sha256': digest(root / 'recovery-link.json')},
        accepted_objects=accepted, accepted_count=len(accepted), unledgered_cache=orphans,
        incurred_transport=journal['counters'], state_sha256=digest(root / 'state.json'),
        journal_sha256=digest(root / 'transport-events.jsonl'), journal_tail_sha256=journal['tail'],
        effective_spec_sha256=digest(effective_path), recovery_link_sha256=digest(root / 'recovery-link.json'),
        blocked_record_sha256=digest(root / 'operation-blocked.json'),
        original_spec_sha256=digest(original_path), files=files,
        file_register_canonical_sha256=hashlib.sha256(canonical(files)).hexdigest(),
        attempts_used=first_attempts + second_attempts, historical_elapsed_seconds=charged_elapsed,
        predecessor_disk_bytes=sum(x['bytes'] for x in history['files'].values())
                               + sum(x['bytes'] for x in files.values()),
        combined_oauth_limit=effective['max_oauth_token_attempts'], journal_projection=journal,
        published_mirror_sha256=digest(root / 'transport-current.json'))
    temporary = root / 'transport-current.json.tmp'
    if temporary.is_file():
        result['corroborating_temp_sha256'] = digest(temporary)
        try:
            temp = verified_record(temporary)
            result['temp_matches_journal'] = (temp['counters'] == journal['counters']
                and temp['journal_tail_sha256'] == journal['tail'])
        except Exception:
            result['temp_matches_journal'] = False  # Never authority for reconstruction.
    expected = recovery.get('expected_predecessor')
    if expected is not None:
        for key in ('accepted_count', 'incurred_transport', 'state_sha256', 'journal_sha256',
                    'journal_tail_sha256', 'effective_spec_sha256', 'recovery_link_sha256',
                    'blocked_record_sha256', 'file_register_canonical_sha256', 'attempts_used'):
            require(result[key] == expected[key], 'RECOVERY_PREDECESSOR_IDENTITY_CONFLICT')
    return result


def linked_recovery_policy(original, history, remaining_slices):
    require(type(remaining_slices) is int
            and 0 < remaining_slices <= original['max_slices'] - history['attempts_used'],
            'RECOVERY_SLICE_ALLOWANCE_INVALID')
    effective = dict(original)
    effective.update(max_oauth_token_attempts=history['combined_oauth_limit'],
                     max_slices=remaining_slices, recovery_mode=True,
                     working_disk_predecessor_bytes=history['predecessor_disk_bytes'])
    used = history['incurred_transport']
    require(used['oauth_attempts'] < effective['max_oauth_token_attempts']
            and used['drive_get_attempts'] < effective['max_drive_get_attempts']
            and used['observed_body_bytes'] < effective['soft_bytes_total'],
            'BLOCKED_AUTHORIZATION_EXHAUSTED')
    for key in ('capture_total_wall_seconds', 'total_wall_seconds'):
        effective[key] -= history['historical_elapsed_seconds']
        require(effective[key] > 0, 'RECOVERY_WALL_ALLOWANCE_EXHAUSTED')
    return effective


def supervise_linked_recovery(spec, recovery):
    begun = time.monotonic()  # Verification/preparation is charged to this invocation.
    require(recovery.get('expected_predecessor') is not None, 'RECOVERY_EXPECTED_IDENTITY_REQUIRED')
    history = inspect_linked_predecessor(spec, recovery)
    effective = linked_recovery_policy(spec, history, recovery['successor_max_slices'])
    root = Path(recovery['successor_destination']).resolve()
    expected = Path(os.environ['LOCALAPPDATA']) / 'ParlayPicker' / 'qualification-canonical-snapshot' / spec['source_revision'] / f"acquire-{spec['anchor_run_id']}-recovery-v3"
    require(root == expected.resolve() and not root.exists(), 'ISOLATED_SUCCESSOR_DESTINATION_CONFLICT')
    require(shutil.disk_usage(root.anchor).free >= spec['required_initial_free_disk_bytes'],
            'INITIAL_FREE_DISK_LIMIT')
    effective['destination'] = str(root)
    worker_token = secrets.token_hex(32)
    initialize_successor(effective, recovery, history, worker_token)
    save(root / 'effective-spec.json', effective)
    old_spec, old_approved = args.spec, args.approved_spec_sha256
    try:
        args.spec = root / 'effective-spec.json'
        args.approved_spec_sha256 = digest(args.spec)
        state = verified_record(root / 'state.json')
        state['spec_sha256'] = args.approved_spec_sha256
        persist_state(root, state)
        budget_disk(root, effective, force=True)
        return run_approved_workers(effective, root, worker_token, begun)
    finally:
        args.spec, args.approved_spec_sha256 = old_spec, old_approved


def recovery_policy(spec, history, additional_oauth, remaining_slices):
    """Only an explicitly approved addendum can increase exhausted OAuth cap."""
    require(type(additional_oauth) is int and 0<=additional_oauth<=20,'RECOVERY_ALLOWANCE_INVALID')
    require(0<remaining_slices<=spec['max_slices']-history['attempts_used'],'RECOVERY_SLICE_ALLOWANCE_INVALID')
    effective=dict(spec)
    effective['max_oauth_token_attempts']=spec['max_oauth_token_attempts']+additional_oauth
    effective['max_slices']=remaining_slices
    used=history['incurred_transport']
    require(used['oauth_attempts']<effective['max_oauth_token_attempts'],'BLOCKED_AUTHORIZATION_EXHAUSTED')
    require(used['drive_get_attempts']<spec['max_drive_get_attempts'] and used['observed_body_bytes']<spec['soft_bytes_total'],'BLOCKED_AUTHORIZATION_EXHAUSTED')
    return effective


def initialize_successor(spec, recovery, history, worker_token):
    """Prospective linked copy; predecessor is never changed or relabeled."""
    root=Path(spec['destination']);prior=Path(recovery['predecessor_destination'])
    require(not root.exists(),'SUCCESSOR_ALREADY_EXISTS_NO_RETRY')
    root.mkdir(parents=True);(root/'raw').mkdir()
    # Full predecessor provenance is retained privately, including orphan bytes.
    provenance=root/'predecessor-evidence'
    shutil.copytree(prior,provenance)
    for name,item in history['files'].items():
        require(digest(provenance/name)==item['sha256'] and digest(prior/name)==item['sha256'],'RECOVERY_COPY_CHANGED')
    shutil.copyfile(prior/'transport-events.jsonl',root/'transport-events.jsonl')
    if 'journal_projection' in history:
        journal = history['journal_projection']
        unsigned = {'counters': journal['counters'], 'journal_tail_sha256': journal['tail']}
        save(root/'transport-current.json',
             dict(unsigned, canonical_sha256=hashlib.sha256(canonical(unsigned)).hexdigest()))
        seal(root, 'transport-reconciliation.json', {
            'immediate_predecessor_identity': history['immediate_predecessor_identity'],
            'predecessor_state_sha256': history['state_sha256'],
            'predecessor_effective_spec_sha256': history['effective_spec_sha256'],
            'predecessor_file_register_sha256': history['file_register_canonical_sha256'],
            'original_spec_sha256': history['original_spec_sha256'],
            'approved_addendum_sha256': digest(args.recovery_spec),
            'tested_driver_sha256': digest(__file__),
            'journal_sha256': history['journal_sha256'],
            'journal_tail_sha256': journal['tail'], 'journal_sequence': journal['sequence'],
            'published_mirror_sha256': history['published_mirror_sha256'],
            'prefix_tail_sha256': journal['prefix_tail'],
            'prefix_sequence': journal['prefix_sequence'],
            'publication_gap_events': journal['publication_gap_events'],
            'derived_successor_mirror_sha256': digest(root/'transport-current.json'),
            'corroborating_temp_sha256': history.get('corroborating_temp_sha256'),
            'temporary_is_authority': False, 'incurred_transport': history['incurred_transport'],
            'charged_slice_attempts': history['attempts_used'],
            'charged_elapsed_seconds': history['historical_elapsed_seconds'],
            'combined_oauth_limit': history['combined_oauth_limit'],
            'source_revision': spec['source_revision'],
            'storage_scope_hash': spec['storage_scope_hash'],
            'canonical_membership_sha256': spec['canonical_membership_sha256'],
            'hash_chain_is_external_signature': False})
    else:
        shutil.copyfile(prior/'transport-current.json',root/'transport-current.json')
    # Only accepted evidence is placed into the active cache. Orphans remain in
    # immutable predecessor provenance; the successor deliberately re-downloads
    # their pinned logical objects, then admits them through a new batch ledger.
    for item in history['accepted_objects'].values():
        target=root/'raw'/item['content_sha256']
        if not target.exists(): shutil.copyfile(prior/'raw'/item['content_sha256'],target)
    original_state=verified_record(prior/'state.json')
    for ref in original_state['accepted_batches']:
        shutil.copyfile(prior/ref['file'],root/ref['file'])
    # New slice-1 batches start at 0005, preserving the four predecessor ledgers.
    # Internal successor slice index is 1..7 and is explicitly linked above.
    persist_state(root,{'spec_sha256':digest(args.spec),'worker_token_sha256':hashlib.sha256(worker_token.encode()).hexdigest(),'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'canonical_membership_sha256':spec['canonical_membership_sha256'],'next_slice':1,'accepted_batches':original_state['accepted_batches'],'transport':history['incurred_transport'],'predecessor_operation_id':history['original_operation_id'],'predecessor_state_sha256':history['state_sha256'],'recovery_addendum_sha256':digest(args.recovery_spec)})
    link = {'original_operation_id':history['original_operation_id'],'original_driver_sha256':recovery['original_driver_sha256'],'replacement_driver_sha256':digest(__file__),'original_spec_sha256':history['original_spec_sha256'],'recovery_spec_sha256':digest(args.recovery_spec),'original_state_sha256':history['state_sha256'],'original_journal_sha256':history['journal_sha256'],'original_journal_tail_sha256':history['journal_tail_sha256'],'incurred_transport':history['incurred_transport'],'additional_oauth_allowance':recovery.get('additional_oauth_allowance',0),'combined_oauth_limit':spec['max_oauth_token_attempts'],'accepted_reused_objects':history['accepted_count'],'unledgered_files_not_admitted':len(history['unledgered_cache']),'original_files':history['files'],'new_source_storage_membership_unchanged':True,'orphan_policy':'PRESERVE_ORIGINAL_REDOWNLOAD_PINNED_OBJECT_THEN_LEDGER'}
    if 'journal_projection' in history:
        link.update(recovery_generation=3, recovery_id=uuid.uuid4().hex,
                    immediate_predecessor_identity=history['immediate_predecessor_identity'],
                    charged_slice_attempts=history['attempts_used'],
                    charged_elapsed_seconds=history['historical_elapsed_seconds'])
    seal(root, 'recovery-link.json', link)


def supervise_recovery(spec,recovery):
    """Only entered after exact driver/spec/addendum hashes and explicit flag."""
    if recovery.get("recovery_generation") == 3:
        return supervise_linked_recovery(spec, recovery)
    begun=time.monotonic()
    require(digest(recovery['original_driver_path'])==recovery['original_driver_sha256'],'RECOVERY_ORIGINAL_DRIVER_CONFLICT')
    require(digest(args.spec)==recovery['original_spec_sha256'],'RECOVERY_ORIGINAL_SPEC_CONFLICT')
    history=inspect_predecessor(recovery['predecessor_destination'],args.spec,spec,recovery['expected_predecessor'])
    effective=recovery_policy(spec,history,recovery['additional_oauth_allowance'],recovery['successor_max_slices'])
    effective['destination']=recovery['successor_destination']
    effective['recovery_mode']=True
    effective['working_disk_predecessor_bytes']=sum(item['bytes'] for item in history['files'].values())
    # The entire old elapsed wall allowance remains charged to the chain.
    effective['total_wall_seconds']-=recovery['historical_elapsed_seconds']
    effective['capture_total_wall_seconds']-=recovery['historical_elapsed_seconds']
    root=Path(effective['destination']).resolve()
    expected=Path(os.environ['LOCALAPPDATA'])/'ParlayPicker'/'qualification-canonical-snapshot'/spec['source_revision']/f"acquire-{spec['anchor_run_id']}-recovery-v2"
    require(root==expected.resolve() and not root.exists(),'ISOLATED_SUCCESSOR_DESTINATION_CONFLICT')
    require(shutil.disk_usage(root.anchor).free>=spec['required_initial_free_disk_bytes'],'INITIAL_FREE_DISK_LIMIT')
    worker_token=secrets.token_hex(32)
    initialize_successor(effective,recovery,history,worker_token)
    # Workers receive a non-secret effective specification retained alongside
    # the signed-by-owner hashes, and preserve the original spec in provenance.
    save(root/'effective-spec.json',effective)
    old_spec=args.spec;old_approved=args.approved_spec_sha256
    try:
        args.spec=root/'effective-spec.json';args.approved_spec_sha256=digest(args.spec)
        state=verified_record(root/'state.json');state['spec_sha256']=args.approved_spec_sha256;persist_state(root,state)
        budget_disk(root,effective,force=True)
        return run_approved_workers(effective,root,worker_token,begun)
    finally:
        args.spec=old_spec;args.approved_spec_sha256=old_approved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--check-plan',action='store_true')
    parser.add_argument('--execute-approved-operation',action='store_true')
    parser.add_argument('--approved-spec-sha256')
    parser.add_argument('--worker',choices=('capture','capture-block','materialize','assess'),help=argparse.SUPPRESS)
    parser.add_argument('--slice-number',type=int)
    parser.add_argument('--recovery-spec',type=Path)
    parser.add_argument('--approved-recovery-sha256')
    parser.add_argument('--approved-driver-sha256')
    args=parser.parse_args()
    spec=json.loads(args.spec.read_bytes())
    source_check(spec)
    cp,rp,pinned=anchor(spec)
    if args.worker:
        # Only the already-approved supervisor can create this state/destination.
        require((Path(spec['destination'])/'state.json').is_file(),'APPROVED_OPERATION_STATE_MISSING')
        state=load_state(Path(spec['destination']),spec)
        token=os.environ.get('PARLAYPICKER_OPERATIONS_WORKER_TOKEN','')
        require(token and hashlib.sha256(token.encode()).hexdigest()==state['worker_token_sha256'] and args.approved_spec_sha256==state['spec_sha256']==digest(args.spec),'APPROVED_SUPERVISOR_REQUIRED')
        try:
            if args.worker=='capture-block': capture_block(spec)
            elif args.worker=='capture': capture(spec,args.slice_number)
            elif args.worker=='materialize': materialize(spec)
            else: assess(spec)
        except BaseException as exc:
            reason=sanitized_reason(exc)
            try:
                retain_worker_failure(Path(spec['destination']),spec,args.worker,reason,exc)
            except BaseException:
                print('FIRST_ERROR_DIAGNOSTIC_INCOMPLETE',file=sys.stderr,flush=True)
            print(reason,file=sys.stderr,flush=True)
            sys.exit(1)
    elif args.execute_approved_operation:
        sys.exit(supervise(spec))
    else:
        print(json.dumps({'status':'OFFLINE_PLAN_CHECK_ONLY','spec_sha256':digest(args.spec),'source_revision':spec['source_revision'],'pinned_canonical_objects':len(pinned),'anchor_digest_verified':True,'lane_A_local_qualification':'BLOCKED_INPUT','secure_environment_presence':{k:bool(os.environ.get(k)) for k in ('PARLAYPICKER_DRIVE_FOLDER_ID','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')},'destination_created':False,'capture_executed':False,'materialization_executed':False,'qualification_executed':False},sort_keys=True,indent=2))
