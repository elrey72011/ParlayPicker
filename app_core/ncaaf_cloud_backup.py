"""Versioned cloud-backup preparation. No Drive writes, collection or inference.

Local v1 setup is unchanged. This route prepares a working copy but reports
CLOUD_BACKUP_PENDING until authenticated remote read-back AND isolated recovery.
Encryption secrets are supplied in memory, never persisted or uploaded.
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import uuid
import zipfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from app_core import ncaaf_pilot_setup as local

VERSION = 'ncaaf-pilot-cloud-backup-v1'
SETUP_VERSION = 'ncaaf-pilot-cloud-setup-v2'
MAGIC = b'PPNCAAF-CLOUD-1\n'
MAX_BYTES = 64 * 1024 * 1024
MAX_FILES = 512
KDF = dict(name='scrypt', n=32768, r=8, p=1)
AUTHORIZED_SETUPS = {}
# Separate owner backup configuration; never source/model acceptance.
AUTHORIZED_READBACKS = {}
require = local.require
encode = local.encode
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def private(path):
    report = local.inspect_security(path)
    require(report['status'] == 'VERIFIED', 'NCAAF_CLOUD_LOCAL_SECURITY_UNVERIFIED')
    return report


def setup(source, root, *, authorization_ref):
    """Local working copy only. This is NOT completed cloud backup/readiness."""
    source = Path(source).resolve(); root = Path(root).resolve()
    permission = dict(version=SETUP_VERSION, source=str(source), root=str(root))
    require(AUTHORIZED_SETUPS.get(authorization_ref) == sha(encode(permission)),
            'NCAAF_CLOUD_SETUP_AUTHORIZATION_UNTRUSTED')
    require(root not in source.parents, 'NCAAF_CLOUD_DESTINATION_CONFLICT')
    security = private(root)
    before_hash = local.file_hash(source)
    require(not any((root / n).exists() for n in
            ('prediction', 'custody', 'backups', 'setup-attempt.jsonl', 'setup-receipt.json')),
            'NCAAF_CLOUD_SETUP_ALREADY_ATTEMPTED')
    journal = root / 'setup-attempt.jsonl'
    with journal.open('xb') as log:
        log.write(encode(dict(version=SETUP_VERSION, authorization_ref=authorization_ref,
            started_at=datetime.now(timezone.utc).isoformat(), source_sha256=before_hash)) + b'\n')
        log.flush(); os.fsync(log.fileno())
        for name in ('prediction', 'backups', 'custody'):
            (root / name).mkdir()
        pre = root / 'backups/pre-migration.sqlite3'
        working = root / 'prediction/evidence.sqlite3'
        post = root / 'backups/post-migration.sqlite3'
        with closing(local._readonly(source)) as db:
            local.contents(db); local._backup(db, pre)
        with closing(local._readonly(pre)) as db:
            core = local.contents(db); local._backup(db, working)
        from app_core import research_replay
        with closing(sqlite3.connect(working)) as db:
            db.execute('PRAGMA foreign_keys=ON'); db.execute('BEGIN IMMEDIATE')
            try:
                research_replay.setup(db); schema = local._replay_schema(db)
                require(local.contents(db) == core, 'NCAAF_CLOUD_CORE_CHANGED')
                db.commit()
            except BaseException:
                db.rollback()
                raise
        with closing(local._readonly(working)) as db:
            require(local.contents(db) == core and local._replay_schema(db) == schema,
                    'NCAAF_CLOUD_RESTART_READBACK')
            local._backup(db, post)
        receipt = dict(version=SETUP_VERSION, status='LOCAL_PREPARED_CLOUD_BACKUP_PENDING',
            source=str(source), source_sha256=before_hash, root=str(root),
            working_store=str(working), core=core, replay_schema=schema, security=security,
            restart_readback=True, cloud_readback_verified=False, recovery_verified=False,
            collection_ready=False, accepted=False, inference=False, provider_requests=0,
            local_backups={str(p): local.file_hash(p) for p in (pre, post)})
        local._exclusive(root / 'setup-receipt.json', encode(receipt))
        log.write(encode(dict(status=receipt['status'], receipt_sha256=sha(encode(receipt)))) + b'\n')
        log.flush(); os.fsync(log.fileno())
    return receipt


def _path(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'NCAAF_CLOUD_ARCHIVE_PATH')
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(v not in ('', '.', '..') for v in name.split('/'))
            and p.as_posix() == name, 'NCAAF_CLOUD_ARCHIVE_PATH')
    return name


def _snapshot(path, secured_root):
    # Real backup includes committed WAL. Normalize ONLY the new snapshot's
    # journal mode through SQLite, never by rewriting header bytes or source.
    # Memory serialization alone retains WAL header flags and is not portable.
    target = Path(secured_root) / ('.cloud-snapshot-' + uuid.uuid4().hex + '.sqlite3')
    with closing(local._readonly(path)) as source:
        local._backup(source, target)
    with closing(sqlite3.connect(target)) as snapshot:
        require(snapshot.execute('PRAGMA journal_mode=DELETE').fetchone() == ('delete',),
                'NCAAF_CLOUD_SNAPSHOT_JOURNAL_MODE')
        require(snapshot.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
                and not snapshot.execute('PRAGMA foreign_key_check').fetchall(),
                'NCAAF_CLOUD_DATABASE_INTEGRITY')
    raw = _file(target)
    # Only this successfully verified, generated temporary copy is removed.
    # On failure it remains in secured storage for recovery review.
    target.unlink()
    return raw


def _file(path):
    p = Path(path)
    require(not p.is_symlink() and not (p.stat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
            if hasattr(p.stat(), 'st_file_attributes') else False), 'NCAAF_CLOUD_REPARSE')
    require(p.stat().st_size <= MAX_BYTES, 'NCAAF_CLOUD_SIZE_LIMIT')
    raw = p.read_bytes()
    return raw


def _no_credentials(raw):
    from app_core.ncaaf_response_custody import _credentials
    text = raw.decode('utf-8', errors='replace')
    _credentials(text)
    # Preserve incomplete journals byte-for-byte, but reject recognizable
    # credential fields even when their interrupted JSON cannot be parsed.
    require(re.search(r'(?i)[\"\'](?:api[_-]?key|authorization|access_token|refresh_token|client_secret|password|private_key|credentials|cookie|set-cookie)[\"\']\s*:', text) is None,
            'NCAAF_CLOUD_CREDENTIAL_FIELD')
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        _credentials(value)


def inventory(root):
    """All custody files and spent journals; no credentials/config outside root.

    Operators must stop all writers for a whole-bundle snapshot. A second byte
    inventory after SQLite backup/encryption detects observed changes, but cannot
    prove absence of an uncoordinated writer.
    """
    root = Path(root).resolve()
    files = {}
    for name in ('setup-attempt.jsonl', 'setup-receipt.json'):
        require((root / name).is_file(), 'NCAAF_CLOUD_SETUP_EVIDENCE_MISSING')
        files[name] = _file(root / name)
    require((root / 'custody').is_dir(), 'NCAAF_CLOUD_CUSTODY_MISSING')
    for path in sorted((root / 'custody').rglob('*')):
        require(not path.is_symlink() and not (getattr(path.stat(), 'st_file_attributes', 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT), 'NCAAF_CLOUD_REPARSE')
        if path.is_file():
            name = _path(path.relative_to(root).as_posix())
            require(path.name.lower() not in ('.env', 'secrets.toml') and
                path.suffix.lower() not in ('.key', '.pem', '.pfx', '.p12'),
                'NCAAF_CLOUD_CREDENTIAL_FILE')
            files[name] = _file(path)
            _no_credentials(files[name])
    require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_BYTES,
            'NCAAF_CLOUD_SIZE_LIMIT')
    return files


def _key(secret, salt):
    require(isinstance(secret, bytes) and len(secret) >= 20, 'NCAAF_CLOUD_SECRET_REQUIRED')
    return Scrypt(salt=salt, length=32, n=KDF['n'], r=KDF['r'], p=KDF['p']).derive(secret)


def prepare(root, output, *, backup_id, secret, writer_stop_reference, recovery_reference):
    """Create only encrypted outgoing files. No upload; no claimed cloud backup."""
    root = Path(root).resolve(); output = Path(output).resolve()
    private(root); private(output)
    require(re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', backup_id or '') is not None
            and writer_stop_reference and recovery_reference, 'NCAAF_CLOUD_PREPARATION_SCOPE')
    captured = inventory(root)
    receipt = json.loads(captured['setup-receipt.json'])
    require(receipt['version'] == SETUP_VERSION and receipt['root'] == str(root)
            and receipt['status'] == 'LOCAL_PREPARED_CLOUD_BACKUP_PENDING',
            'NCAAF_CLOUD_SETUP_EVIDENCE_MISMATCH')
    for name in ('pre-migration.sqlite3','post-migration.sqlite3'):
        original = root / 'backups' / name
        require(receipt.get('local_backups', {}).get(str(original)) == local.file_hash(original),
                'NCAAF_CLOUD_ORIGINAL_BACKUP_CHANGED')
    files = dict(captured)
    files['database/pre-migration.sqlite3'] = _snapshot(root / 'backups/pre-migration.sqlite3', root)
    files['database/post-migration.sqlite3'] = _snapshot(root / 'backups/post-migration.sqlite3', root)
    files['database/working-snapshot.sqlite3'] = _snapshot(root / 'prediction/evidence.sqlite3', root)
    require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_BYTES,
            'NCAAF_CLOUD_SIZE_LIMIT')
    manifest = dict(version=VERSION, backup_id=backup_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        writer_stop_reference=writer_stop_reference, recovery_reference=recovery_reference,
        files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in sorted(files.items())},
        custody_files=sum(n.startswith('custody/') for n in files),
        journal_files=[n for n in sorted(files) if n.endswith('.jsonl')],
        original_setup_receipt_sha256=sha(captured['setup-receipt.json']))
    archive = BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, raw in sorted(files.items()): z.writestr(name, raw)
        z.writestr('manifest.json', encode(manifest))
    plain = archive.getvalue()
    require(len(plain) <= MAX_BYTES, 'NCAAF_CLOUD_SIZE_LIMIT')
    salt = os.urandom(16); nonce = os.urandom(12)
    header = encode(dict(version=VERSION, cipher='AES-256-GCM', kdf=KDF,
        salt=salt.hex(), nonce=nonce.hex(), backup_id=backup_id))
    prefix = MAGIC + len(header).to_bytes(4, 'big') + header
    encrypted = prefix + AESGCM(_key(secret, salt)).encrypt(nonce, plain, prefix)
    require(inventory(root) == captured, 'NCAAF_CLOUD_FILES_CHANGED')
    first = output / (backup_id + '.ppbackup')
    second = output / (backup_id + '.manifest.json')
    require(not first.exists() and not second.exists(), 'NCAAF_CLOUD_EXISTING_PACKAGE')
    public = dict(version=VERSION, backup_id=backup_id, filename=first.name,
        ciphertext_sha256=sha(encrypted), ciphertext_bytes=len(encrypted),
        status='ENCRYPTED_LOCAL_ONLY_UPLOAD_NOT_AUTHORIZED', cloud_readback_verified=False,
        recovery_verified=False)
    local._exclusive(first, encrypted)
    local._exclusive(second, encode(public))
    return public


def decrypt(raw, secret, expected_sha256):
    require(isinstance(raw, bytes) and len(raw) <= MAX_BYTES + 4096
            and sha(raw) == expected_sha256 and raw.startswith(MAGIC), 'NCAAF_CLOUD_CIPHER_INTEGRITY')
    length = int.from_bytes(raw[len(MAGIC):len(MAGIC)+4], 'big')
    require(0 < length <= 2048, 'NCAAF_CLOUD_HEADER')
    prefix_size = len(MAGIC) + 4 + length
    header = json.loads(raw[len(MAGIC)+4:prefix_size])
    require(set(header) == {'version','cipher','kdf','salt','nonce','backup_id'}
            and header['version'] == VERSION and header['cipher'] == 'AES-256-GCM'
            and header['kdf'] == KDF, 'NCAAF_CLOUD_HEADER')
    salt = bytes.fromhex(header['salt']); nonce = bytes.fromhex(header['nonce'])
    require(len(salt) == 16 and len(nonce) == 12, 'NCAAF_CLOUD_HEADER')
    try:
        plain = AESGCM(_key(secret, salt)).decrypt(nonce, raw[prefix_size:], raw[:prefix_size])
    except Exception:
        raise ValueError('NCAAF_CLOUD_DECRYPTION_FAILED') from None
    with zipfile.ZipFile(BytesIO(plain)) as z:
        names = z.namelist()
        require(len(names) <= MAX_FILES + 1 and len(set(n.casefold() for n in names)) == len(names)
                and 'manifest.json' in names, 'NCAAF_CLOUD_ARCHIVE_FILES')
        require(all(_path(n) == n for n in names), 'NCAAF_CLOUD_ARCHIVE_PATH')
        require(all(i.compress_type == zipfile.ZIP_STORED and i.file_size <= MAX_BYTES
                and not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
                and sum(i.file_size for i in z.infolist()) <= MAX_BYTES, 'NCAAF_CLOUD_SIZE_LIMIT')
        manifest = json.loads(z.read('manifest.json'))
        require(manifest['version'] == VERSION and manifest['backup_id'] == header['backup_id']
                and set(manifest['files']) == set(names) - {'manifest.json'}, 'NCAAF_CLOUD_ARCHIVE_FILES')
        files = {n: z.read(n) for n in manifest['files']}
    require(all(set(v) == {'bytes','sha256'} and len(files[n]) == v['bytes']
                and sha(files[n]) == v['sha256'] for n,v in manifest['files'].items()),
                'NCAAF_CLOUD_ARCHIVE_INTEGRITY')
    allowed = {'setup-attempt.jsonl','setup-receipt.json',
        'database/pre-migration.sqlite3','database/post-migration.sqlite3','database/working-snapshot.sqlite3'}
    require(allowed <= set(files) and all(n in allowed or n.startswith('custody/') for n in files),
            'NCAAF_CLOUD_ARCHIVE_FILES')
    return manifest, files


def recover(raw, secret, expected_sha256, destination):
    """NEW isolated drill only. Never restore a live store/reset spent attempts."""
    target = Path(destination).absolute()
    require(not target.exists(), 'NCAAF_CLOUD_RECOVERY_DESTINATION_EXISTS')
    private(target.parent)
    manifest, files = decrypt(raw, secret, expected_sha256)
    # Verify databases in memory before materializing ANY original bytes.
    for name in (n for n in files if n.startswith('database/')):
        with closing(sqlite3.connect(':memory:')) as db:
            db.deserialize(files[name])
            core = local.contents(db)
            require(db.execute('PRAGMA integrity_check').fetchall() == [('ok',)],
                    'NCAAF_CLOUD_DATABASE_INTEGRITY')
            if name != 'database/pre-migration.sqlite3': local._replay_schema(db)
    target.mkdir()
    for name, content in files.items():
        p = target / name; p.parent.mkdir(parents=True, exist_ok=True)
        local._exclusive(p, content)
    require(all((target/n).read_bytes() == raw for n,raw in files.items()),
            'NCAAF_CLOUD_RECOVERY_READBACK')
    record = dict(version=VERSION, backup_id=manifest['backup_id'], destination=str(target),
        ciphertext_sha256=expected_sha256, recovered_files=len(files), exact_readback=True,
        status='ISOLATED_NON_OPERATIONAL_RECOVERY', spent_journals_preserved=True,
        cloud_origin_verified=False, accepted=False, inference=False, collection_ready=False)
    local._exclusive(target / 'isolated-recovery-receipt.json', encode(record))
    return record


def verify_cloud(reader, *, approval_ref, recovery_parent, secret):
    """Reader performs fresh authenticated metadata/downloads of EXACT approved IDs.

    No upload/list/create/delete API. No local cache qualifies as cloud read-back.
    The controlled operator entry binds full folder ACL evidence and actual file
    IDs after the owner approves the destination and immutable outgoing hashes.
    """
    scope = AUTHORIZED_READBACKS.get(approval_ref)
    require(isinstance(scope, dict) and scope.get('version') == VERSION,
            'NCAAF_CLOUD_READBACK_AUTHORIZATION_UNTRUSTED')
    require(scope.get('independent_key_recovery_reference'), 'NCAAF_CLOUD_KEY_RECOVERY_UNVERIFIED')
    folder = reader.metadata(scope['folder_id'])
    require(folder.get('id') == scope['folder_id'] and folder.get('name') == scope['folder_name']
            and folder.get('mimeType') == 'application/vnd.google-apps.folder'
            and folder.get('trashed') is False and folder.get('permissions_complete') is True,
            'NCAAF_CLOUD_FOLDER_UNVERIFIED')
    permissions = folder.get('permissions')
    require(isinstance(permissions, list) and permissions and all(
        p.get('type') == 'user' and p.get('emailAddress') in scope['allowed_users']
        and p.get('role') in ('owner','writer','reader') for p in permissions)
        and any(p.get('role') == 'owner' and p.get('emailAddress') == scope['owner'] for p in permissions),
        'NCAAF_CLOUD_FOLDER_NOT_PRIVATE')
    def acl(perms):
        require(isinstance(perms, list) and all(p.get('type') == 'user'
            and p.get('emailAddress') in scope['allowed_users'] for p in perms),
            'NCAAF_CLOUD_FOLDER_NOT_PRIVATE')
        return sorted((p.get('type'),p.get('role'),p.get('emailAddress')) for p in perms)
    data = {}
    for kind in ('bundle','manifest'):
        expected = scope[kind]
        meta = reader.metadata(expected['file_id'])
        require(meta.get('id') == expected['file_id'] and meta.get('name') == expected['name']
                and meta.get('parents') == [scope['folder_id']] and meta.get('trashed') is False
                and meta.get('permissions_complete') is True and acl(meta.get('permissions')) == acl(permissions),
                'NCAAF_CLOUD_REMOTE_IDENTITY_CONFLICT')
        # Reader must stream fresh media with this explicit limit and no truncation.
        data[kind] = reader.download(expected['file_id'], max_bytes=MAX_BYTES + 4096)
        require(isinstance(data[kind],bytes) and len(data[kind]) <= MAX_BYTES + 4096
                and sha(data[kind]) == expected['sha256'], 'NCAAF_CLOUD_REMOTE_READBACK_MISMATCH')
    public = json.loads(data['manifest'])
    require(public['filename'] == scope['bundle']['name']
            and public['ciphertext_sha256'] == scope['bundle']['sha256']
            and public['ciphertext_bytes'] == len(data['bundle']), 'NCAAF_CLOUD_REMOTE_READBACK_MISMATCH')
    result = recover(data['bundle'], secret, scope['bundle']['sha256'],
        Path(recovery_parent) / (public['backup_id'] + '-isolated'))
    result.update(cloud_origin_verified=True, cloud_readback_verified=True,
        cloud_folder_id=scope['folder_id'], remote_file_ids=[scope[k]['file_id'] for k in ('bundle','manifest')],
        status='CLOUD_READBACK_AND_ISOLATED_RECOVERY_VERIFIED')
    # Backup proof is not custody/source acceptance or collection authorization.
    local._exclusive(Path(result['destination']) / 'cloud-verification-receipt.json', encode(result))
    return result

