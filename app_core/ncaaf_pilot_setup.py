"""Explicit isolated prediction-store setup; no provider, Drive or acceptance IO.

Read-only assessment is the default. Authentic copying requires independently
configured, verified private/encrypted local and separate backup destinations.
Existing stores and spent journals are never overwritten by setup or rollback.
"""
from contextlib import closing
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess

VERSION = 'ncaaf-pilot-local-setup-v1'
CORE = ('bundles', 'snapshots', 'snapshot_runtime', 'score_revisions', 'closing_observations', 'validation_plans')
REPLAY = ('research_replay_sources', 'research_replay_exports', 'research_source_intakes')
CORE_COLUMNS = {
    'bundles': ('version', 'frozen_at', 'manifest'),
    'snapshots': ('snapshot_id', 'version', 'generated_at', 'candidates', 'decisions', 'inputs', 'payload_hash'),
    'snapshot_runtime': ('snapshot_id', 'process_instance'),
    'score_revisions': ('snapshot_id', 'evidence_hash', 'recorded_at', 'scores'),
    'closing_observations': ('observation_id', 'snapshot_id', 'candidate_id', 'payload'),
    'validation_plans': ('plan_id', 'sport', 'payload'),
}
# Trusted operator setup configuration, never populated by CLI receipt uploads.
AUTHORIZED_SETUPS = {}


def require(ok, code):
    if not ok:
        raise ValueError(code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_security(path):
    """No write probes. Fail closed on inaccessible ACL/encryption information.

    Current owner, SYSTEM and administrators only; a protected Windows DACL,
    no reparse ancestors, and EFS or a fully protected encrypted volume.
    Non-Windows hosts exercise synthetic setup only, never claim Windows proof.
    """
    p = Path(path)
    result = dict(path=str(p.resolve()), private_acl=False, encrypted=False, no_reparse=False,
                  status='UNKNOWN', evidence='OS_READ_ONLY_INSPECTION')
    if os.name != 'nt' or not p.is_dir():
        return result
    command = r'''
    $ErrorActionPreference = 'Stop'
    $taskPath = $env:NCAAF_SETUP_PATH
    $taskAcl = Get-Acl -LiteralPath $taskPath
    $taskOwner = $taskAcl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value
    $taskCurrent = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $taskAllowed = @($taskCurrent, 'S-1-5-18', 'S-1-5-32-544')
    $taskPrivate = $taskAcl.AreAccessRulesProtected -and ($taskOwner -eq $taskCurrent)
    foreach ($rule in $taskAcl.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier])) {
      if ($rule.AccessControlType -eq 'Allow' -and $rule.FileSystemRights -ne 0 -and $rule.IdentityReference.Value -notin $taskAllowed) { $taskPrivate = $false }
    }
    $taskNoReparse = $true
    $taskNode = Get-Item -LiteralPath $taskPath -Force
    while ($null -ne $taskNode) {
      if ($taskNode.Attributes -band [IO.FileAttributes]::ReparsePoint) { $taskNoReparse = $false }
      $taskNode = $taskNode.Parent
    }
    $taskEncrypted = [bool]((Get-Item -LiteralPath $taskPath -Force).Attributes -band [IO.FileAttributes]::Encrypted)
    $taskEncryptionKnown = $taskEncrypted
    if (-not $taskEncrypted) {
      try {
        $taskVolume = Get-BitLockerVolume -MountPoint ([IO.Path]::GetPathRoot($taskPath)) -ErrorAction Stop
        $taskEncryptionKnown = $true
        $taskEncrypted = ($taskVolume.VolumeStatus -eq 'FullyEncrypted' -and $taskVolume.ProtectionStatus -eq 'On')
      } catch { }
    }
    @{private_acl=[bool]$taskPrivate; encrypted=[bool]$taskEncrypted; no_reparse=[bool]$taskNoReparse; encryption_known=[bool]$taskEncryptionKnown} | ConvertTo-Json -Compress
    '''
    try:
        env = dict(os.environ, NCAAF_SETUP_PATH=str(p.resolve()))
        shell = shutil.which('pwsh') or 'powershell.exe'
        if shell == 'powershell.exe':
            # A caller's PowerShell 7 module path is incompatible with PS5.
            env.pop('PSModulePath', None)
        proc = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-Command', command],
            capture_output=True, text=True, timeout=20, check=False, env=env)
        if proc.returncode == 0:
            found = json.loads(proc.stdout)
            for key in ('private_acl', 'encrypted', 'no_reparse'):
                result[key] = found[key] is True
            result['status'] = 'VERIFIED' if all(result[k] for k in ('private_acl', 'encrypted', 'no_reparse')) else 'BLOCKED' if found['encryption_known'] else 'UNKNOWN'
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        pass
    return result


def _cell(value):
    if isinstance(value, bytes):
        return {'sqlite_blob_b64': base64.b64encode(value).decode()}
    return value


def contents(db):
    """Aggregate content identities, never raw private rows in diagnostics."""
    require(db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'NCAAF_SETUP_INTEGRITY')
    require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'NCAAF_SETUP_FOREIGN_KEYS')
    result = {}
    for table in CORE:
        columns = tuple(row[1] for row in db.execute(f'PRAGMA table_info({table})'))
        require(columns == CORE_COLUMNS[table], 'NCAAF_SETUP_SOURCE_SCHEMA')
        rows = sorted(encode([_cell(v) for v in row]) for row in db.execute(f'SELECT * FROM {table}'))
        h = hashlib.sha256()
        for row in rows:
            h.update(str(len(row)).encode() + b':' + row)
        schema = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
        result[table] = dict(count=len(rows), content_sha256=h.hexdigest(), schema_sha256=hashlib.sha256(schema.encode()).hexdigest())
    return result


def _readonly(path):
    p = Path(path).resolve()
    require(p.is_file(), 'NCAAF_SETUP_SOURCE_MISSING')
    db = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True, timeout=5)
    db.execute('PRAGMA query_only=ON')
    return db


def assess(source, root, backup_root):
    """Read-only. Never connect through prediction_evidence's initializer."""
    source = Path(source).resolve(); root = Path(root).resolve(); backup_root = Path(backup_root).resolve()
    security = {'root': inspect_security(root), 'backup': inspect_security(backup_root)}
    with closing(_readonly(source)) as db:
        core = contents(db)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return dict(version=VERSION, source=str(source), root=str(root), backup_root=str(backup_root),
        source_sha256=file_hash(source), wal_present=Path(str(source) + '-wal').exists(), core=core,
        missing_replay_tables=sorted(set(REPLAY) - tables), security=security,
        status='SECURITY_VERIFIED_AWAITING_EXPLICIT_SETUP' if all(r['status'] == 'VERIFIED' for r in security.values()) else 'BLOCKED',
        write_probes=0, store_mutations=0, provider_requests=0, acceptance=False)


def _exclusive(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def _backup(db, path):
    # Reserve exclusively: setup cannot replace an existing file.
    with Path(path).open('xb'):
        pass
    with closing(sqlite3.connect(path)) as target:
        db.backup(target)
        require(target.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'NCAAF_SETUP_BACKUP_INTEGRITY')


def _replay_schema(db):
    # Compare exact CREATE TABLE/TRIGGER SQL with existing setup's output, not
    # merely names. A same-name altered trigger cannot weaken append-only.
    from app_core import research_replay
    with closing(sqlite3.connect(':memory:')) as expected:
        research_replay.setup(expected)
        schema = expected.execute("SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
    actual = db.execute("SELECT type,name,sql FROM sqlite_master WHERE name IN (" + ','.join('?' for _ in schema) + ") ORDER BY type,name", [s[1] for s in schema]).fetchall()
    require(actual == schema, 'NCAAF_SETUP_REPLAY_SCHEMA')
    return dict(tables=list(REPLAY), immutable_triggers=[s[1] for s in schema if s[0] == 'trigger'])


def setup(source, root, backup_root, *, authorization_ref):
    """New working copy only, explicit configured approval, no rerun overwrite.

    Failure leaves all recovery files and journal in place. No automated rollback
    erases evidence. Before/after backups and separate verified copies survive.
    """
    source = Path(source).resolve(); root = Path(root).resolve(); backup_root = Path(backup_root).resolve()
    require(root != backup_root and root not in backup_root.parents and backup_root not in root.parents
            and root not in source.parents and backup_root not in source.parents, 'NCAAF_SETUP_DESTINATION_CONFLICT')
    permission = dict(version=VERSION, source=str(source), root=str(root), backup_root=str(backup_root))
    require(AUTHORIZED_SETUPS.get(authorization_ref) == hashlib.sha256(encode(permission)).hexdigest(), 'NCAAF_SETUP_AUTHORIZATION_UNTRUSTED')
    report = assess(source, root, backup_root)
    require(report['status'] == 'SECURITY_VERIFIED_AWAITING_EXPLICIT_SETUP', 'NCAAF_SETUP_SECURITY_UNVERIFIED')
    # Destinations must already be secured. Child creation inherits those ACLs
    # and encryption. Setup does not silently change deployment or encryption.
    journal_path = root / 'setup-attempt.jsonl'
    try:
        journal = journal_path.open('xb')
    except FileExistsError:
        raise ValueError('NCAAF_SETUP_ALREADY_ATTEMPTED') from None
    with journal:
        journal.write(encode(dict(version=VERSION, authorization_ref=authorization_ref, source_sha256=report['source_sha256'],
            started_at=datetime.now(timezone.utc).isoformat())) + b'\n')
        journal.flush(); os.fsync(journal.fileno())
        for name in ('prediction', 'backups', 'custody'):
            require(not (root / name).exists(), 'NCAAF_SETUP_EXISTING_DESTINATION')
        prediction = root / 'prediction'; backups = root / 'backups'
        prediction.mkdir(); backups.mkdir(); (root / 'custody').mkdir()
        working = prediction / 'evidence.sqlite3'; pre = backups / 'pre-migration.sqlite3'; post = backups / 'post-migration.sqlite3'
        with closing(_readonly(source)) as db:
            # One coherent backup first; derive recovery copy from that exact
            # snapshot, avoiding two independently sampled source generations.
            _backup(db, pre)
        with closing(_readonly(pre)) as db:
            before = contents(db); _backup(db, working)
        from app_core import research_replay
        with closing(sqlite3.connect(working)) as db:
            db.execute('PRAGMA foreign_keys=ON'); db.execute('BEGIN IMMEDIATE')
            try:
                research_replay.setup(db)
                schema = _replay_schema(db)
                require(contents(db) == before, 'NCAAF_SETUP_CORE_CHANGED')
                db.commit()
            except BaseException:
                db.rollback()
                raise
        # Closed-handle restart read-back: a fresh connection observes commit.
        with closing(_readonly(working)) as db:
            require(contents(db) == before, 'NCAAF_SETUP_RESTART_READBACK')
            require(_replay_schema(db) == schema, 'NCAAF_SETUP_RESTART_READBACK')
            _backup(db, post)
        # Separate backup is exclusive and byte verified, never replacement.
        pre_second = backup_root / 'ncaaf-pre-migration.sqlite3'
        post_second = backup_root / 'ncaaf-post-migration.sqlite3'
        _exclusive(pre_second, pre.read_bytes()); _exclusive(post_second, post.read_bytes())
        require(file_hash(pre) == file_hash(pre_second) and file_hash(post) == file_hash(post_second), 'NCAAF_SETUP_BACKUP_READBACK')
        receipt = dict(version=VERSION, status='WORKING_COPY_READY', source=str(source), source_sha256=report['source_sha256'],
            working_store=str(working), core=before, replay_schema=schema, security=report['security'],
            backups={str(p): file_hash(p) for p in (pre, post, pre_second, post_second)},
            restart_readback=True, accepted=False, inference=False, provider_requests=0,
            rollback='Preserve all files and spent journals. No automatic replacement or post-capture rollback.')
        _exclusive(root / 'setup-receipt.json', encode(receipt))
        require((root / 'setup-receipt.json').read_bytes() == encode(receipt), 'NCAAF_SETUP_PRIVATE_READBACK')
        journal.write(encode(dict(status='COMPLETE', receipt_sha256=hashlib.sha256(encode(receipt)).hexdigest())) + b'\n')
        journal.flush(); os.fsync(journal.fileno())
    return receipt
