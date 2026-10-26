"""Synthetic rows in the existing loopback CI PostgreSQL service only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
import uuid

import psycopg
from psycopg import sql
import pytest

from app_core.nfl_workflow_store import Store, WorkflowError

KEY=b'SYNTHETIC-private-postgres-test-key-not-production'


@pytest.fixture
def private_pg(database_url):
    u=urlsplit(database_url)
    assert u.hostname in {'localhost','127.0.0.1'} and u.path.endswith('_test')
    role='nfl_fixture_'+uuid.uuid4().hex
    password='SYNTHETIC-private-store-test-only'
    with psycopg.connect(database_url,autocommit=True) as admin:
        admin.execute('DROP SCHEMA IF EXISTS nfl_private CASCADE')
        admin.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(sql.Identifier(role),sql.Literal(password)))
        schema=(Path(__file__).resolve().parents[2]/'docs/research/nfl-private-workflow-store-v1.sql').read_text()
        schema=schema.replace('TO nfl_app','TO '+role)
        admin.execute(schema)
    info=psycopg.conninfo.conninfo_to_dict(database_url)
    info.update(user=role,password=password)
    def factory(readonly):
        return psycopg.connect(**info,autocommit=True,connect_timeout=5)
    store=Store(factory,KEY)
    try:
        yield store,factory,database_url
    finally:
        with psycopg.connect(database_url,autocommit=True) as admin:
            admin.execute('DROP SCHEMA nfl_private CASCADE')
            admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


def test_private_nfl_postgres_atomic_claims_persist_and_reopen(private_pg):
    store,factory,_=private_pg
    assert store.verify()['durable_backend']=='POSTGRESQL'
    record=store.put('SYNTHETIC',{'fixture':'No real results or provider evidence'})
    def claim(_):
        try:Store(factory,KEY).claim('SYNTHETIC-shared-intent',{'fixture':True});return True
        except WorkflowError as exc:assert str(exc)=='PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY';return False
    with ThreadPoolExecutor(max_workers=6) as pool:assert sum(pool.map(claim,range(12)))==1
    reopened=Store(factory,KEY)
    assert reopened.get(record['id'])==record
    rows=[store.put('SYNTHETIC-page',{'n':i}) for i in range(3)]
    pages=[reopened.list('SYNTHETIC-page',limit=1,offset=i) for i in range(3)]
    assert {page[0]['id'] for page in pages}=={row['id'] for row in rows}
    assert reopened.spent('SYNTHETIC-shared-intent')
    with factory(False) as con:
        for statement in ('DELETE FROM nfl_private.records','UPDATE nfl_private.records SET kind=kind','TRUNCATE nfl_private.claims'):
            with pytest.raises(psycopg.Error):con.execute(statement)


def test_private_nfl_postgres_owner_signature_and_permission_checks(private_pg):
    store,factory,url=private_pg;record=store.put('SYNTHETIC',{'fixture':True})
    with pytest.raises(ValueError,match='MISSING'):Store(factory,KEY,owner='SYNTHETIC other owner').get(record['id'])
    with pytest.raises(ValueError,match='AUTHENTICATION'):Store(factory,b'SYNTHETIC-wrong-private-key-more-than32').get(record['id'])
    with psycopg.connect(url,autocommit=True) as admin:
        admin.execute('GRANT SELECT ON nfl_private.records TO PUBLIC')
    with pytest.raises(ValueError,match='ACCESS_TOO_BROAD'):store.verify()


def test_private_nfl_postgres_missing_protection_and_production_tls_reject(private_pg):
    store,_,url=private_pg
    with pytest.raises(ValueError,match='TLS_REQUIRED'):Store.postgres(url,KEY).verify()
    with psycopg.connect(url,autocommit=True) as admin:
        admin.execute('DROP TRIGGER records_immutable ON nfl_private.records')
    with pytest.raises(ValueError,match='PROTECTION_MISSING'):store.verify()
