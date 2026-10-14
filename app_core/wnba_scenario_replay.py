"""WNBA version dispatch in the existing immutable private replay tables."""
from contextlib import closing
import hashlib
from io import BytesIO
import json
import re
import sqlite3
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from app_core import research_replay as store, wnba_scenario_research as scenario

CAPTURE = 'wnba-private-scenario-capture-v1'


def ready(*, path=None):
    try:
        store.scenario_store_ready(path=path)
    except (ValueError, sqlite3.Error):
        raise ValueError('WNBA_SCENARIO_STORE_UNAVAILABLE') from None


def retain(result, *, path=None):
    p = scenario.read_result(result); source = p['input']
    source_raw = scenario.encode(source); raw = scenario.encode(result)
    reference = 'wnba-scenario-input:' + source['sha256']; export_id = 'wnba-scenario-result:' + result['sha256']
    ready(path=path)
    with closing(sqlite3.connect(store._scenario_path(path).as_uri()+'?mode=rw', uri=True)) as db, db:
        store._scenario_schema(db)
        existing = db.execute('SELECT payload,payload_hash FROM research_source_intakes WHERE reference=?', (reference,)).fetchone()
        scenario.require(existing is None or existing == (source_raw, store.digest(source_raw)), 'INTEGRITY')
        db.execute('INSERT OR IGNORE INTO research_source_intakes VALUES (?,?,?)', (reference, source_raw, store.digest(source_raw)))
        existing = db.execute('SELECT package_hash,payload FROM research_replay_exports WHERE export_id=?', (export_id,)).fetchone()
        scenario.require(existing is None or existing == (source['sha256'], raw), 'INTEGRITY')
        db.execute('INSERT OR IGNORE INTO research_replay_exports VALUES (?,?,?)', (export_id, source['sha256'], raw))
    return dict(version=CAPTURE, export_id=export_id, result_sha256=result['sha256'], input_sha256=source['sha256'])


def read(receipt, *, path=None):
    scenario.require(isinstance(receipt, dict) and set(receipt) == {'version', 'export_id', 'result_sha256', 'input_sha256'}
        and receipt['version'] == CAPTURE and all(isinstance(receipt[k], str) and re.fullmatch('[0-9a-f]{64}', receipt[k]) for k in ('result_sha256', 'input_sha256'))
        and receipt['export_id'] == 'wnba-scenario-result:' + receipt['result_sha256'], 'ROUTE_CONFLICT')
    ready(path=path)
    with closing(sqlite3.connect(store._scenario_path(path).as_uri()+'?mode=ro', uri=True)) as db:
        row = db.execute('SELECT package_hash,payload FROM research_replay_exports WHERE export_id=?', (receipt['export_id'],)).fetchone()
        source = db.execute('SELECT payload,payload_hash FROM research_source_intakes WHERE reference=?', ('wnba-scenario-input:' + receipt['input_sha256'],)).fetchone()
    scenario.require(row is not None and source is not None and row[0] == receipt['input_sha256'] and store.digest(source[0]) == source[1], 'INTEGRITY')
    result = json.loads(row[1]); original = scenario.load(source[0].encode())
    scenario.require(result['sha256'] == receipt['result_sha256'] and original['sha256'] == receipt['input_sha256'] and result['payload']['input'] == original, 'INTEGRITY')
    scenario.read_result(result)
    return result


def display(result):
    """Allowlisted private view. No original bodies, offers or public aliases."""
    p = scenario.read_result(result); i = p['input']['payload']
    return dict(labels=p['labels'], evidence_label=i['evidence_label'], event=i['event'], assumption=i['scenario'],
                probability=p['probability'], description='Uncalibrated WNBA model estimate', inference_time=p['inference_time'],
                publisher_availability='UNKNOWN', wager_action='PASS', stake=0)


def download(receipt, *, path=None):
    result = read(receipt, path=path)
    files = {'scenario-result.json': scenario.encode(result).encode(), 'private-receipt.json': scenario.encode(receipt).encode()}
    p = result['payload']['input']['payload']
    for obj in p['scoring']+[p['event_document']]:
        raw, _ = scenario.custody.read(obj)
        files['originals/'+obj['body_sha256']+'.body'] = raw
    manifest = dict(version='wnba-private-scenario-export-v1', purpose='PRIVATE_RESEARCH',
                    files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()})
    files['manifest.json'] = scenario.encode(manifest).encode()
    output = BytesIO()
    with ZipFile(output, 'w') as archive:
        for name, raw in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    return output.getvalue(), manifest
