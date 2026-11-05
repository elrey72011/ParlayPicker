"""Synthetic rows in the existing loopback CI PostgreSQL service only (roadmap M1.1)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
import uuid

import psycopg
from psycopg import sql
import pytest

from app_core.pipeline.store import Store, StoreError

KEY = b'SYNTHETIC-pipeline-postgres-test-key-not-production'
OWNER = 'SYNTHETIC owner'


@pytest.fixture
def pipeline_pg(database_url):
    u = urlsplit(database_url)
    assert u.hostname in {'localhost', '127.0.0.1'} and u.path.endswith('_test')
    role = 'pipeline_fixture_' + uuid.uuid4().hex
    password = 'SYNTHETIC-pipeline-store-test-only'
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute('DROP SCHEMA IF EXISTS pipeline_private CASCADE')
        admin.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(sql.Identifier(role), sql.Literal(password)))
        schema = (Path(__file__).resolve().parents[2] / 'docs/operations/pipeline-store-v1.sql').read_text()
        admin.execute(schema.replace('TO pipeline_app', 'TO ' + role))
    info = psycopg.conninfo.conninfo_to_dict(database_url)
    info.update(user=role, password=password)

    def factory(readonly):
        return psycopg.connect(**info, autocommit=True, connect_timeout=5)
    try:
        yield Store(factory, KEY, owner=OWNER), factory, database_url
    finally:
        with psycopg.connect(database_url, autocommit=True) as admin:
            admin.execute('DROP SCHEMA pipeline_private CASCADE')
            admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


def test_pipeline_postgres_claims_once_persists_and_reopens(pipeline_pg):
    store, factory, _ = pipeline_pg
    assert store.verify()['durable_backend'] == 'POSTGRESQL'
    record = store.put('NFL', 'forecast', {'fixture': 'No real results or provider evidence'})

    def click(_):
        try:
            Store(factory, KEY, owner=OWNER).claim('NFL', 'SYNTHETIC-shared-intent', {'calls': 1})
            return True
        except StoreError as exc:
            assert str(exc) == 'PIPELINE_OPERATION_ALREADY_SPENT_NO_RETRY'
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(click, range(12))) == 1
    reopened = Store(factory, KEY, owner=OWNER)
    assert reopened.get('NFL', record['id']) == record and reopened.spent('NFL', 'SYNTHETIC-shared-intent')
    rows = [store.put('NBA', 'forecast', {'n': i}) for i in range(3)]
    pages = [reopened.list('NBA', 'forecast', limit=1, offset=i) for i in range(3)]
    assert {page[0]['id'] for page in pages} == {row['id'] for row in rows}
    with pytest.raises(StoreError, match='PIPELINE_RECORD_MISSING'):
        reopened.get('NBA', record['id'])  # sport is part of the key
    with factory(False) as con:
        for statement in ('DELETE FROM pipeline_private.records', 'UPDATE pipeline_private.records SET kind=kind',
                          'TRUNCATE pipeline_private.claims'):
            with pytest.raises(psycopg.Error):
                con.execute(statement)


def test_pipeline_postgres_owner_signature_and_permission_checks(pipeline_pg):
    store, factory, url = pipeline_pg
    record = store.put('NFL', 'forecast', {'fixture': True})
    with pytest.raises(StoreError, match='MISSING'):
        Store(factory, KEY, owner='SYNTHETIC other owner').get('NFL', record['id'])
    with pytest.raises(StoreError, match='AUTHENTICATION'):
        Store(factory, b'SYNTHETIC-wrong-pipeline-key-more-than32', owner=OWNER).get('NFL', record['id'])
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute('GRANT SELECT ON pipeline_private.records TO PUBLIC')
    with pytest.raises(StoreError, match='ACCESS_TOO_BROAD'):
        store.verify()


def test_pipeline_postgres_missing_protection_and_production_tls_reject(pipeline_pg):
    store, _, url = pipeline_pg
    with pytest.raises(StoreError, match='TLS_REQUIRED'):
        Store.postgres(url, KEY, owner=OWNER).verify()
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute('DROP TRIGGER records_immutable ON pipeline_private.records')
    with pytest.raises(StoreError, match='PROTECTION_MISSING'):
        store.verify()
