"""SYNTHETIC SQLite behavior of the sport-neutral pipeline store (roadmap M1.1)."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from app_core.pipeline.store import Store, StoreError, SPORTS

KEY = b'SYNTHETIC-pipeline-store-test-key-not-production'
OWNER = 'SYNTHETIC owner'


@pytest.fixture
def path(tmp_path):
    return tmp_path / 'SYNTHETIC-pipeline.sqlite'


@pytest.fixture
def store(path):
    return Store.sqlite(path, KEY, owner=OWNER)


def test_put_get_list_and_reopen_after_restart(store, path):
    assert store.verify() == dict(version='pipeline-store-v1', durable_backend='SQLITE', synthetic=True)
    first = store.put('NFL', 'forecast', {'game': 'SYNTHETIC-1', 'p': 0.53})
    assert first['sport'] == 'NFL' and first['owner'] == OWNER and first['kind'] == 'forecast'
    rows = [store.put('NFL', 'forecast', {'game': f'SYNTHETIC-{i}'}) for i in range(2, 5)]
    reopened = Store.sqlite(path, KEY, owner=OWNER)  # a fresh process/session
    assert reopened.get('NFL', first['id']) == first
    pages = [reopened.list('NFL', 'forecast', limit=1, offset=i) for i in range(4)]
    assert {page[0]['id'] for page in pages} == {first['id'], *(row['id'] for row in rows)}


def test_identical_put_is_idempotent_and_conflicting_put_is_rejected(store):
    one = store.put('NBA', 'forecast', {'p': 0.5}, identity='SYNTHETIC-op')
    assert store.put('NBA', 'forecast', {'p': 0.5}, identity='SYNTHETIC-op') == one
    with pytest.raises(StoreError, match='PIPELINE_RECORD_CONFLICT'):
        store.put('NBA', 'forecast', {'p': 0.6}, identity='SYNTHETIC-op')


def test_sport_is_a_signed_scope(store):
    record = store.put('NHL', 'forecast', {'p': 0.5}, identity='SYNTHETIC-shared-id')
    with pytest.raises(StoreError, match='PIPELINE_RECORD_MISSING'):
        store.get('NBA', record['id'])
    other = store.put('NBA', 'forecast', {'p': 0.7}, identity='SYNTHETIC-shared-id')
    assert store.get('NHL', 'SYNTHETIC-shared-id')['payload'] == {'p': 0.5} and other['payload'] == {'p': 0.7}
    with pytest.raises(StoreError, match='PIPELINE_SPORT_UNSUPPORTED'):
        store.put('XFL', 'forecast', {})
    assert SPORTS == {'NFL', 'NCAAF', 'NBA', 'NHL', 'NCAAB', 'WNBA', 'MLB'}


def test_claim_is_spent_exactly_once_under_concurrent_clicks(path):
    def click(_):
        try:
            Store.sqlite(path, KEY, owner=OWNER).claim('NFL', 'SYNTHETIC-request', {'calls': 1})
            return True
        except StoreError as exc:
            assert str(exc) == 'PIPELINE_OPERATION_ALREADY_SPENT_NO_RETRY'
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(click, range(12))) == 1
    reopened = Store.sqlite(path, KEY, owner=OWNER)
    assert reopened.spent('NFL', 'SYNTHETIC-request') and not reopened.spent('NFL', 'SYNTHETIC-unused')
    assert reopened.claim_record('NFL', 'SYNTHETIC-request')['payload'] == {'calls': 1}
    with pytest.raises(StoreError, match='ALREADY_SPENT'):
        reopened.claim('NFL', 'SYNTHETIC-request', {'calls': 1})


def test_records_and_claims_are_append_only(store, path):
    store.put('NFL', 'forecast', {'p': 0.5})
    store.claim('NFL', 'SYNTHETIC-c', {})
    with sqlite3.connect(path) as con:
        for statement in ('DELETE FROM pipeline_records', 'UPDATE pipeline_records SET kind=kind',
                          'DELETE FROM pipeline_claims'):
            with pytest.raises(sqlite3.DatabaseError, match='append-only'):
                con.execute(statement)


def test_tampering_wrong_key_and_wrong_owner_are_rejected(store, path):
    record = store.put('NFL', 'forecast', {'p': 0.5})
    with pytest.raises(StoreError, match='AUTHENTICATION'):
        Store.sqlite(path, b'SYNTHETIC-wrong-pipeline-key-more-than-32', owner=OWNER).get('NFL', record['id'])
    with pytest.raises(StoreError, match='MISSING'):
        Store.sqlite(path, KEY, owner='SYNTHETIC other owner').get('NFL', record['id'])
    with sqlite3.connect(path) as con:  # bypass the trigger, edit the row, then restore protection
        con.execute('DROP TRIGGER pipeline_records_UPDATE')
        con.execute("UPDATE pipeline_records SET raw=replace(raw,'0.5','0.9')")
        with pytest.raises(StoreError, match='PROTECTION_MISSING'):
            store.get('NFL', record['id'])
        con.execute("CREATE TRIGGER pipeline_records_UPDATE BEFORE UPDATE ON pipeline_records "
                    "BEGIN SELECT RAISE(ABORT,'append-only'); END")
    with pytest.raises(StoreError, match='AUTHENTICATION_FAILED'):
        store.get('NFL', record['id'])


def test_missing_protection_is_detected_before_any_read(store, path):
    with sqlite3.connect(path) as con:
        con.execute('DROP TRIGGER pipeline_claims_DELETE')
    with pytest.raises(StoreError, match='PROTECTION_MISSING'):
        store.verify()


def test_inputs_are_validated(path):
    with pytest.raises(StoreError, match='KEY_REQUIRED'):
        Store.sqlite(path, b'short', owner=OWNER)
    with pytest.raises(StoreError, match='OWNER_REQUIRED'):
        Store.sqlite(path, KEY, owner='')
    store = Store.sqlite(path, KEY, owner=OWNER)
    with pytest.raises(StoreError, match='PIPELINE_RECORD_KIND'):
        store.put('NFL', 'claim', {})
    with pytest.raises(StoreError, match='PIPELINE_RECORD_LIMIT'):
        store.list('NFL', 'forecast', limit=0)
    with pytest.raises(StoreError, match='TLS_REQUIRED'):
        Store.postgres('postgresql://SYNTHETIC@localhost/db', KEY, owner=OWNER).verify()
