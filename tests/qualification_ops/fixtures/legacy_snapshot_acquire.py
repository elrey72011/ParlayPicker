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


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode() + b'\n')
    temporary.replace(path)


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
        require(disk_bytes(root) <= spec['max_working_disk_bytes'], 'WORKING_DISK_LIMIT')
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


def verified_record(path):
    try: record=json.loads(Path(path).read_bytes())
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


def load_state(root,spec):
    state=verified_record(root/'state.json')
    commits=sorted(root.glob('state-commit-*.json'))
    require(commits and len(commits)==state.get('state_sequence'),'STATE_JOURNAL_CONFLICT')
    previous=None
    for sequence,path in enumerate(commits,1):
        commit=verified_record(path)
        require(path.name==f'state-commit-{sequence:06d}.json' and commit.get('sequence')==sequence and commit.get('previous_commit_sha256')==previous,'STATE_JOURNAL_CONFLICT')
        previous=digest(path)
    require(commit['state_sha256']==digest(root/'state.json'),'STATE_JOURNAL_CONFLICT')
    require(state['spec_sha256']==digest(args.spec),'RAW_STATE_SPEC_CONFLICT')
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


class TransportBudget(dict):
    """Durable monotonic attempt/body journal, including failed/in-flight reads."""
    keys=('drive_get_attempts','oauth_attempts','observed_body_bytes')
    def __init__(self,root,spec,floor):
        self.root,self.spec=root,spec
        self.lock=threading.RLock();self.sequence=0;self.tail=None
        counters={key:0 for key in self.keys}
        path=root/'transport-events.jsonl'
        if path.exists():
            with path.open('rb') as stream:
                for line in stream:
                    try: event=json.loads(line)
                    except ValueError: raise RuntimeError('TRANSPORT_JOURNAL_INVALID') from None
                    unsigned={k:v for k,v in event.items() if k!='canonical_sha256'}
                    require(event.get('canonical_sha256')==hashlib.sha256(canonical(unsigned)).hexdigest() and event.get('sequence')==self.sequence+1 and event.get('previous_sha256')==self.tail,'TRANSPORT_JOURNAL_CONFLICT')
                    current=event['counters']
                    require(set(current)==set(self.keys) and all(type(current[k]) is int and current[k]>=counters[k] for k in self.keys),'TRANSPORT_COUNTER_REWIND')
                    counters=current;self.sequence=event['sequence'];self.tail=event['canonical_sha256']
        require(all(counters[key]>=floor[key] for key in self.keys),'TRANSPORT_COUNTER_REWIND')
        super().__init__(counters)
        if self.sequence:
            mirror=verified_record(root/'transport-current.json')
            require(mirror['counters']==counters and mirror['journal_tail_sha256']==self.tail,'TRANSPORT_MIRROR_CONFLICT')
    def increment(self,key,amount):
        self[key]+=amount
        unsigned={'sequence':self.sequence+1,'previous_sha256':self.tail,'counters':dict(self)}
        event=dict(unsigned,canonical_sha256=hashlib.sha256(canonical(unsigned)).hexdigest())
        with (self.root/'transport-events.jsonl').open('ab') as stream:
            stream.write(canonical(event)+b'\n');stream.flush();os.fsync(stream.fileno())
        self.sequence+=1;self.tail=event['canonical_sha256']
        self.mirror()
    def mirror(self):
        unsigned={'counters':dict(self),'journal_tail_sha256':self.tail}
        save(self.root/'transport-current.json',dict(unsigned,canonical_sha256=hashlib.sha256(canonical(unsigned)).hexdigest()))


def guarded_session_factory(spec,meter,allowed_media,authorized_session):
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
            finally:
                response.close()
                with meter.lock: meter.mirror()
    def factory():
        session=authorized_session()
        require(session._auth_request_session is not None,'AUTH_METER_UNAVAILABLE')
        for scheme in ('https://','http://'):
            session.mount(scheme,ReadOnlyAdapter())
            session._auth_request_session.mount(scheme,ReadOnlyAdapter(token=True))
        return session
    return factory


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
    session_factory=guarded_session_factory(spec,meter,allowed_media,_authorized_session)
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
        pending=sorted(set(pinned)-set(completed));metrics=Counter();batch_number=0;reason=None
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
    expected=Path(os.environ['LOCALAPPDATA'])/'ParlayPicker'/'qualification-canonical-snapshot'/spec['source_revision']/f"acquire-{spec['anchor_run_id']}-v1"
    require(root==expected.resolve() and not root.exists(),'ISOLATED_DESTINATION_CONFLICT_OR_RESUME_NOT_AUTHORIZED')
    require(shutil.disk_usage(root.anchor).free>=spec['required_initial_free_disk_bytes'],'INITIAL_FREE_DISK_LIMIT')
    root.mkdir(parents=True);(root/'raw').mkdir()
    worker_token=secrets.token_hex(32)
    persist_state(root,{'spec_sha256':digest(args.spec),'worker_token_sha256':hashlib.sha256(worker_token.encode()).hexdigest(),'source_revision':spec['source_revision'],'storage_scope_hash':spec['storage_scope_hash'],'canonical_membership_sha256':spec['canonical_membership_sha256'],'next_slice':1,'accepted_batches':[],'transport':{'drive_get_attempts':0,'oauth_attempts':0,'observed_body_bytes':0}})
    def run_worker(phase,limit,number=None):
        source_check(spec)
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--spec',str(Path(args.spec).resolve()),'--approved-spec-sha256',args.approved_spec_sha256,'--worker',phase]
        if number is not None: command+=['--slice-number',str(number)]
        stamp=f'{phase}-{number or 1}'
        with (root/(stamp+'.log')).open('wb') as log:
            worker_env=dict(os.environ,PARLAYPICKER_OPERATIONS_WORKER_TOKEN=worker_token)
            process=subprocess.Popen(command,stdout=log,stderr=log,env=worker_env)
            deadline=time.monotonic()+limit
            try:
                while process.poll() is None:
                    require(time.monotonic()<=deadline,'STAGE_WALL_LIMIT')
                    require(time.monotonic()-begun<=spec['total_wall_seconds'],'TOTAL_WALL_LIMIT')
                    budget_disk(root,spec)
                    time.sleep(1)
                require(time.monotonic()<=deadline,'STAGE_WALL_LIMIT')
                require(time.monotonic()-begun<=spec['total_wall_seconds'],'TOTAL_WALL_LIMIT')
                budget_disk(root,spec,force=True)
                require(process.returncode==0,'WORKER_BLOCKED_NO_RETRY')
            except BaseException:
                process.kill();process.wait()
                raise
    try:
        full=False
        for number in range(1,spec['max_slices']+1):
            if time.monotonic()-begun>=spec['capture_total_wall_seconds']:
                break
            run_worker('capture',min(spec['slice_wall_seconds'],spec['capture_total_wall_seconds']-(time.monotonic()-begun)),number)
            result=verified_record(root/f'slice-{number:02d}-result.json')
            if result['status']=='RAW_COMPLETE': full=True;break
            require(result['terminal_reason'] in {'PROCESSING_DEADLINE','OBJECT_LIMIT','BYTE_LIMIT','BLOCK_BYTE_LIMIT'},'UNEXPECTED_PARTIAL_STOP')
            if result['terminal_reason']=='BLOCK_BYTE_LIMIT' or result['transport']['observed_body_bytes']>=spec['soft_bytes_total']:
                break
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--check-plan',action='store_true')
    parser.add_argument('--execute-approved-operation',action='store_true')
    parser.add_argument('--approved-spec-sha256')
    parser.add_argument('--worker',choices=('capture','materialize','assess'),help=argparse.SUPPRESS)
    parser.add_argument('--slice-number',type=int)
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
        if args.worker=='capture': capture(spec,args.slice_number)
        elif args.worker=='materialize': materialize(spec)
        else: assess(spec)
    elif args.execute_approved_operation:
        sys.exit(supervise(spec))
    else:
        print(json.dumps({'status':'OFFLINE_PLAN_CHECK_ONLY','spec_sha256':digest(args.spec),'source_revision':spec['source_revision'],'pinned_canonical_objects':len(pinned),'anchor_digest_verified':True,'lane_A_local_qualification':'BLOCKED_INPUT','secure_environment_presence':{k:bool(os.environ.get(k)) for k in ('PARLAYPICKER_DRIVE_FOLDER_ID','PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')},'destination_created':False,'capture_executed':False,'materialization_executed':False,'qualification_executed':False},sort_keys=True,indent=2))
