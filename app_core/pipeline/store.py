"""Sport-neutral, owner-scoped, append-only signed record store (roadmap M1.1).

Generalizes ``app_core.nfl_workflow_store.Store`` without changing it. Existing
NFL records stay in that store and stay readable through it.

Guarantees, matching the NFL store:
- Every record is HMAC-signed and bound to its owner, sport, identity and kind.
  A read verifies the signature and all four bindings.
- Records and claims are append-only, enforced by database triggers. Writes are
  idempotent only for an identical payload.
- ``claim`` commits atomically before external work. A claim is never leased,
  reset or retried, so a rerender, duplicate click or restart cannot spend twice.
- PostgreSQL is the hosted backend. Its schema is created by an administrator
  from ``docs/operations/pipeline-store-v1.sql``. The application role must not
  own or administer it, and access must not be broader than owner plus app role.
  This code never creates, migrates or alters the schema.
- SQLite is an explicit SYNTHETIC/local backend, chosen only by its constructor.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path
import sqlite3

VERSION = 'pipeline-store-v1'
SPORTS = frozenset({'NFL', 'NCAAF', 'NBA', 'NHL', 'NCAAB', 'WNBA', 'MLB'})
MAX_RECORD = 32 * 1024 * 1024
SCHEMA = 'pipeline_private'
SQLITE_PREFIX = 'pipeline_'
COLUMNS = ['owner_id', 'sport', 'id', 'kind', 'raw', 'signature', 'created_at']
ENVELOPE = frozenset({'version', 'owner', 'sport', 'id', 'kind', 'payload', 'created_at'})
PG_PROTECT = "BEGIN RAISE EXCEPTION 'pipeline records are append-only'; END;"


class StoreError(ValueError):
    """Credential-free, developer-authored error code only."""


def require(ok, code):
    if not ok:
        raise StoreError(code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha256(encode(value).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def _sqlite_trigger(table, action):
    return f"CREATE TRIGGER {table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only'); END"


class Store:
    def __init__(self, factory, key, *, owner, synthetic=False, clock=utc):
        require(isinstance(key, bytes) and len(key) >= 32, 'PIPELINE_STORE_KEY_REQUIRED')
        require(isinstance(owner, str) and 1 <= len(owner) <= 100, 'PIPELINE_STORE_OWNER_REQUIRED')
        self._factory, self._key, self.owner = factory, key, owner
        self.synthetic, self.clock = synthetic, clock
        self.prefix = SQLITE_PREFIX if synthetic else SCHEMA + '.'

    @classmethod
    def postgres(cls, dsn, key, *, owner):
        """Lazy hosted backend: nothing connects until the first operation."""
        def factory(readonly):
            try:
                import psycopg
                from psycopg.conninfo import conninfo_to_dict
                info = conninfo_to_dict(dsn)
                require(info.get('sslmode') == 'verify-full' and info.get('host'), 'PIPELINE_STORE_TLS_REQUIRED')
                return psycopg.connect(dsn, autocommit=True, connect_timeout=5,
                                       options='-c statement_timeout=5000 -c lock_timeout=5000')
            except StoreError:
                raise
            except Exception:
                raise StoreError('PIPELINE_STORE_UNAVAILABLE') from None
        return cls(factory, key, owner=owner)

    @classmethod
    def sqlite(cls, path, key, *, owner, clock=utc):
        """Explicit local/SYNTHETIC backend; never selected by a production setting."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            con.execute(f'CREATE TABLE IF NOT EXISTS {SQLITE_PREFIX}meta (version TEXT PRIMARY KEY)')
            con.execute(f'INSERT OR IGNORE INTO {SQLITE_PREFIX}meta VALUES (?)', (VERSION,))
            for suffix in ('records', 'claims'):
                table = SQLITE_PREFIX + suffix
                con.execute(f'CREATE TABLE IF NOT EXISTS {table} (owner_id TEXT NOT NULL, sport TEXT NOT NULL, '
                            'id TEXT NOT NULL, kind TEXT NOT NULL, raw TEXT NOT NULL, signature TEXT NOT NULL, '
                            'created_at TEXT NOT NULL, PRIMARY KEY(owner_id,sport,id))')
                for action in ('UPDATE', 'DELETE'):
                    con.execute(_sqlite_trigger(table, action).replace('CREATE TRIGGER', 'CREATE TRIGGER IF NOT EXISTS', 1))

        def factory(readonly):
            return sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True, timeout=5)
        return cls(factory, key, owner=owner, synthetic=True, clock=clock)

    # -- transport -------------------------------------------------------
    def _execute(self, con, query, args=()):
        return con.execute(query if self.synthetic else query.replace('?', '%s'), args)

    def _schema(self, con):
        require(self._execute(con, 'SELECT version FROM ' + self.prefix + 'meta').fetchall() == [(VERSION,)],
                'PIPELINE_STORE_SCHEMA_UNSUPPORTED')
        for suffix in ('records', 'claims'):
            table = self.prefix + suffix
            if self.synthetic:
                columns = [row[1] for row in con.execute('PRAGMA table_info(' + table + ')')]
                require(columns == COLUMNS, 'PIPELINE_STORE_SCHEMA_UNSUPPORTED')
                for action in ('UPDATE', 'DELETE'):
                    row = con.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                                      (table + '_' + action,)).fetchone()
                    require(row is not None and row[0] == _sqlite_trigger(table, action), 'PIPELINE_STORE_PROTECTION_MISSING')
                continue
            # The application role must not own or administer its durable store.
            row = con.execute(
                "SELECT r.rolsuper,r.rolcreatedb,r.rolcreaterole,has_schema_privilege(current_user,%s,'CREATE'),"
                "has_table_privilege(current_user,%s,'UPDATE,DELETE,TRUNCATE'),pg_has_role(current_user,c.relowner,'MEMBER') "
                "FROM pg_roles r,pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE r.rolname=current_user AND n.nspname=%s AND c.relname=%s",
                (SCHEMA, table, SCHEMA, suffix)).fetchone()
            require(row is not None and not any(row), 'PIPELINE_STORE_ROLE_NOT_PRIVATE')
            table_other = con.execute(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) a "
                "WHERE n.nspname=%s AND c.relname=%s AND a.grantee NOT IN "
                "(c.relowner,(SELECT oid FROM pg_roles WHERE rolname=current_user))", (SCHEMA, suffix)).fetchone()[0]
            schema_other = con.execute(
                "SELECT count(*) FROM pg_namespace n CROSS JOIN LATERAL aclexplode(COALESCE(n.nspacl,acldefault('n',n.nspowner))) a "
                "WHERE n.nspname=%s AND a.grantee NOT IN (n.nspowner,(SELECT oid FROM pg_roles WHERE rolname=current_user))",
                (SCHEMA,)).fetchone()[0]
            require(table_other == schema_other == 0, 'PIPELINE_STORE_ACCESS_TOO_BROAD')
            triggers = con.execute(
                "SELECT t.tgname,p.prosrc FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_proc p ON p.oid=t.tgfoid "
                "WHERE n.nspname=%s AND c.relname=%s AND NOT t.tgisinternal AND t.tgenabled='O'",
                (SCHEMA, suffix)).fetchall()
            require(set(triggers) == {(suffix + '_immutable', PG_PROTECT)}, 'PIPELINE_STORE_PROTECTION_MISSING')

    @contextmanager
    def _transaction(self, *, readonly):
        con = None
        try:
            con = self._factory(readonly)
            if self.synthetic:
                con.execute('PRAGMA query_only=ON' if readonly else 'BEGIN IMMEDIATE')
            else:
                con.execute('BEGIN READ ONLY' if readonly else 'BEGIN')
            self._schema(con)
            yield con
            con.commit()
        except StoreError:
            if con is not None:
                con.rollback()
            raise
        except Exception:
            if con is not None:
                con.rollback()
            raise StoreError('PIPELINE_STORE_UNAVAILABLE') from None
        finally:
            if con is not None:
                con.close()

    # -- signing ---------------------------------------------------------
    def _signed(self, sport, identity, kind, payload):
        value = dict(version=VERSION, owner=self.owner, sport=sport, id=identity, kind=kind,
                     payload=payload, created_at=self.clock())
        raw = encode(value)
        require(0 < len(raw.encode()) <= MAX_RECORD, 'PIPELINE_RECORD_SIZE')
        return raw, hmac.new(self._key, raw.encode(), 'sha256').hexdigest(), value['created_at']

    def _authenticate(self, row, sport, identity=None, kind=None):
        require(row is not None and isinstance(row[0], str) and len(row[0].encode()) <= MAX_RECORD, 'PIPELINE_RECORD_MISSING')
        raw, signature = row
        expected = hmac.new(self._key, raw.encode(), 'sha256').hexdigest()
        require(isinstance(signature, str) and hmac.compare_digest(signature, expected), 'PIPELINE_RECORD_AUTHENTICATION_FAILED')
        value = json.loads(raw)
        require(set(value) == ENVELOPE and value['version'] == VERSION and value['owner'] == self.owner
                and value['sport'] == sport, 'PIPELINE_RECORD_OWNER_CONFLICT')
        require((identity is None or value['id'] == identity) and (kind is None or value['kind'] == kind),
                'PIPELINE_RECORD_IDENTITY_CONFLICT')
        return value

    @staticmethod
    def _sport(sport):
        require(sport in SPORTS, 'PIPELINE_SPORT_UNSUPPORTED')
        return sport

    # -- API -------------------------------------------------------------
    def verify(self):
        with self._transaction(readonly=True):
            pass
        return dict(version=VERSION, durable_backend='SQLITE' if self.synthetic else 'POSTGRESQL',
                    synthetic=self.synthetic)

    def get(self, sport, identity, *, kind=None):
        sport = self._sport(sport)
        with self._transaction(readonly=True) as con:
            row = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'records WHERE owner_id=? AND sport=? AND id=?',
                                (self.owner, sport, identity)).fetchone()
        return self._authenticate(row, sport, identity, kind)

    def list(self, sport, kind, *, limit=100, offset=0):
        sport = self._sport(sport)
        require(type(limit) is int and 1 <= limit <= 100 and type(offset) is int and 0 <= offset < 100_000_000,
                'PIPELINE_RECORD_LIMIT')
        with self._transaction(readonly=True) as con:
            rows = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'records WHERE owner_id=? AND sport=? AND kind=? '
                                 'ORDER BY created_at DESC,id LIMIT ? OFFSET ?', (self.owner, sport, kind, limit, offset)).fetchall()
        return [self._authenticate(row, sport, kind=kind) for row in rows]

    def put(self, sport, kind, payload, *, identity=None):
        """Append a record. Re-putting the identical payload returns the original."""
        sport = self._sport(sport)
        require(isinstance(kind, str) and 1 <= len(kind) <= 100 and kind != 'claim', 'PIPELINE_RECORD_KIND')
        identity = identity or digest(dict(sport=sport, kind=kind, payload=payload))
        raw, signature, created = self._signed(sport, identity, kind, payload)
        with self._transaction(readonly=False) as con:
            self._execute(con, 'INSERT INTO ' + self.prefix + 'records (owner_id,sport,id,kind,raw,signature,created_at) '
                          'VALUES (?,?,?,?,?,?,?) ON CONFLICT(owner_id,sport,id) DO NOTHING',
                          (self.owner, sport, identity, kind, raw, signature, created))
            row = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'records WHERE owner_id=? AND sport=? AND id=?',
                                (self.owner, sport, identity)).fetchone()
            value = self._authenticate(row, sport, identity, kind)
            require(value['payload'] == payload, 'PIPELINE_RECORD_CONFLICT')
        return value

    def claim(self, sport, identity, payload):
        """Atomically spend ``identity`` once. A second claim raises; nothing resets it."""
        sport = self._sport(sport)
        raw, signature, created = self._signed(sport, identity, 'claim', payload)
        with self._transaction(readonly=False) as con:
            cursor = self._execute(con, 'INSERT INTO ' + self.prefix + 'claims (owner_id,sport,id,kind,raw,signature,created_at) '
                                   'VALUES (?,?,?,?,?,?,?) ON CONFLICT(owner_id,sport,id) DO NOTHING',
                                   (self.owner, sport, identity, 'claim', raw, signature, created))
            inserted = cursor.rowcount == 1
            row = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'claims WHERE owner_id=? AND sport=? AND id=?',
                                (self.owner, sport, identity)).fetchone()
            self._authenticate(row, sport, identity, 'claim')
        require(inserted, 'PIPELINE_OPERATION_ALREADY_SPENT_NO_RETRY')
        return identity

    def claim_record(self, sport, identity):
        sport = self._sport(sport)
        with self._transaction(readonly=True) as con:
            row = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'claims WHERE owner_id=? AND sport=? AND id=?',
                                (self.owner, sport, identity)).fetchone()
        return self._authenticate(row, sport, identity, 'claim')

    def spent(self, sport, identity):
        sport = self._sport(sport)
        with self._transaction(readonly=True) as con:
            row = self._execute(con, 'SELECT raw,signature FROM ' + self.prefix + 'claims WHERE owner_id=? AND sport=? AND id=?',
                                (self.owner, sport, identity)).fetchone()
        if row is not None:
            self._authenticate(row, sport, identity, 'claim')
        return row is not None
