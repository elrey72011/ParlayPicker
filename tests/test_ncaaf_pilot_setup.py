"""SYNTHETIC SQLite stores only; no authentic records or deployment mutation."""
from contextlib import closing
import hashlib
import sqlite3

import pytest
from app_core import ncaaf_pilot_setup as local, prediction_evidence, research_replay


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'SYNTHETIC-original.sqlite3'
    db = prediction_evidence.connect(source)
    # Reproduce the real core-only schema without changing protected setup.
    for table in local.REPLAY:
        db.execute(f'DROP TABLE {table}')
    db.execute("INSERT INTO bundles VALUES ('SYNTHETIC-bundle', '2026-10-09T00:00:00Z', '{}')")
    db.execute("INSERT INTO snapshots VALUES ('SYNTHETIC-snapshot', 'SYNTHETIC-bundle', '2026-10-09T00:00:00Z', '[]', '[]', '{}', 'SYNTHETIC-hash')")
    db.commit(); db.close()
    root = tmp_path / 'SYNTHETIC-private'; root.mkdir()
    backup = tmp_path / 'SYNTHETIC-encrypted-offline-backup'; backup.mkdir()
    monkeypatch.setattr(local, 'inspect_security', lambda p: dict(path=str(p), status='VERIFIED',
        private_acl=True, encrypted=True, no_reparse=True, evidence='SYNTHETIC_OS_PROOF'))
    permission = dict(version=local.VERSION, source=str(source.resolve()), root=str(root.resolve()), backup_root=str(backup.resolve()))
    monkeypatch.setattr(local, 'AUTHORIZED_SETUPS', {'SYNTHETIC-setup': hashlib.sha256(local.encode(permission)).hexdigest()})
    return source, root, backup


def test_default_assessment_is_read_only(fixture):
    source, root, backup = fixture; before = source.read_bytes()
    report = local.assess(source, root, backup)
    assert report['missing_replay_tables'] == sorted(local.REPLAY)
    assert report['store_mutations'] == report['write_probes'] == 0
    assert source.read_bytes() == before and list(root.iterdir()) == list(backup.iterdir()) == []


def test_consistent_copy_migration_restart_backup_and_append_only(fixture):
    source, root, backup = fixture; before = source.read_bytes()
    result = local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert result['status'] == 'WORKING_COPY_READY' and result['restart_readback']
    assert source.read_bytes() == before and len(result['replay_schema']['immutable_triggers']) == 6
    assert not result['accepted'] and not result['inference'] and result['provider_requests'] == 0
    with closing(local._readonly(source)) as db: assert local.contents(db) == result['core']
    with closing(sqlite3.connect(result['working_store'])) as db:
        assert local.contents(db) == result['core']
        for table in local.REPLAY:
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
            # Triggers fire even if no selected rows exist: use source_intakes
            # for a labelled synthetic record to verify mutation prevention.
        db.execute("INSERT INTO research_source_intakes VALUES ('SYNTHETIC-unaccepted', '{}', 'SYNTHETIC-hash')"); db.commit()
        for sql in ("UPDATE research_source_intakes SET payload='bad'", 'DELETE FROM research_source_intakes', "UPDATE bundles SET manifest='bad'"):
            with pytest.raises(sqlite3.IntegrityError, match='append-only'): db.execute(sql)
    retained = (root / 'prediction/evidence.sqlite3').read_bytes()
    with pytest.raises(ValueError, match='ALREADY_ATTEMPTED'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert (root / 'prediction/evidence.sqlite3').read_bytes() == retained and source.read_bytes() == before
    assert (backup / 'ncaaf-pre-migration.sqlite3').read_bytes() == (root / 'backups/pre-migration.sqlite3').read_bytes()


def test_committed_wal_is_included_without_source_checkpoint(fixture):
    source, root, backup = fixture
    with closing(sqlite3.connect(source)) as writer:
        writer.execute('PRAGMA journal_mode=WAL'); writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute("INSERT INTO bundles VALUES ('SYNTHETIC-WAL-only', '2026-10-09T00:00:01Z', '{}')"); writer.commit()
        wal = source.with_name(source.name + '-wal')
        before = source.read_bytes(); wal_before = wal.read_bytes()
        result = local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
        assert source.read_bytes() == before and wal.read_bytes() == wal_before
        with closing(local._readonly(result['working_store'])) as db:
            assert db.execute("SELECT COUNT(*) FROM bundles WHERE version='SYNTHETIC-WAL-only'").fetchone()[0] == 1
        with closing(local._readonly(root / 'backups/pre-migration.sqlite3')) as db:
            assert db.execute('SELECT COUNT(*) FROM bundles').fetchone()[0] == 2


def test_transaction_failure_preserves_pre_migration_and_source(fixture, monkeypatch):
    source, root, backup = fixture; before = source.read_bytes()
    def broken(db):
        db.execute('CREATE TABLE SYNTHETIC_transaction_failure (value TEXT)')
        raise RuntimeError('SYNTHETIC failure')
    monkeypatch.setattr(research_replay, 'setup', broken)
    with pytest.raises(RuntimeError, match='SYNTHETIC failure'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert source.read_bytes() == before and (root / 'backups/pre-migration.sqlite3').exists()
    with closing(local._readonly(root / 'prediction/evidence.sqlite3')) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='SYNTHETIC_transaction_failure'").fetchall()
    assert (root / 'setup-attempt.jsonl').exists() and not (root / 'setup-receipt.json').exists()
    with pytest.raises(ValueError, match='ALREADY_ATTEMPTED'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')


@pytest.mark.parametrize('issue', ['acl', 'encryption', 'reparse', 'unknown'])
def test_security_blocks_before_authentic_copy(fixture, monkeypatch, issue):
    source, root, backup = fixture; before = source.read_bytes()
    def inspect(path):
        return dict(path=str(path), status='UNKNOWN' if issue == 'unknown' else 'BLOCKED', private_acl=issue != 'acl',
            encrypted=issue != 'encryption', no_reparse=issue != 'reparse')
    monkeypatch.setattr(local, 'inspect_security', inspect)
    with pytest.raises(ValueError, match='SECURITY_UNVERIFIED'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert source.read_bytes() == before and list(root.iterdir()) == list(backup.iterdir()) == []


def test_untrusted_setup_does_not_open_source_or_create_directories(fixture, monkeypatch):
    source, root, backup = fixture
    monkeypatch.setattr(local, '_readonly', lambda *a: pytest.fail('Unauthorized source read'))
    with pytest.raises(ValueError, match='AUTHORIZATION_UNTRUSTED'): local.setup(source, root, backup, authorization_ref='unknown')
    assert list(root.iterdir()) == list(backup.iterdir()) == []


def test_existing_store_or_backup_never_overwritten(fixture):
    source, root, backup = fixture
    (root / 'custody').mkdir(); journal = root / 'custody/SYNTHETIC-spent.jsonl'; journal.write_bytes(b'SYNTHETIC spent attempt')
    with pytest.raises(ValueError, match='EXISTING_DESTINATION'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert journal.read_bytes() == b'SYNTHETIC spent attempt'


def test_source_schema_failure_is_read_only(fixture):
    source, root, backup = fixture
    with closing(sqlite3.connect(source)) as db:
        db.execute('DROP TABLE validation_plans'); db.commit()
    before = source.read_bytes()
    with pytest.raises(ValueError, match='SOURCE_SCHEMA'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert source.read_bytes() == before and list(root.iterdir()) == []


def test_same_name_but_weakened_replay_trigger_rejects(fixture):
    source, root, backup = fixture
    with closing(sqlite3.connect(source)) as db:
        research_replay.setup(db)
        db.execute('DROP TRIGGER immutable_research_source_intakes_UPDATE')
        db.execute('CREATE TRIGGER immutable_research_source_intakes_UPDATE BEFORE UPDATE ON research_source_intakes BEGIN SELECT 1; END')
        db.commit()
    before = source.read_bytes()
    with pytest.raises(ValueError, match='REPLAY_SCHEMA'): local.setup(source, root, backup, authorization_ref='SYNTHETIC-setup')
    assert source.read_bytes() == before


def test_production_setup_catalog_empty():
    assert local.AUTHORIZED_SETUPS == {}
