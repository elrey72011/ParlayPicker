"""Owner-scoped durable workflow records; no setup, migration or prediction on read.

PostgreSQL is the hosted backend. SQLite is an explicit SYNTHETIC test backend,
never selected by a production setting. Claims commit before external work and
are never leased, deleted, reset or retried. Native result bodies stay unchanged.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import hmac
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

VERSION = 'nfl-private-workflow-store-v1'
# Aggregate custody includes a prepared original plus two native packets. This
# envelope is separate from, and never raises, the released transport/packet caps.
MAX_RECORD = 32 * 1024 * 1024
PG_PROTECT = "BEGIN RAISE EXCEPTION 'NFL private records are append-only'; END;"


class WorkflowError(ValueError):
    """Credential-free, developer-authored error code only."""


def require(ok, code):
    if not ok:
        raise WorkflowError(code)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha256(encode(value).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, factory, key, *, owner='Robert Velarde', synthetic=False, clock=utc):
        require(isinstance(key, bytes) and len(key) >= 32, 'PRIVATE_STORE_KEY_REQUIRED')
        require(isinstance(owner, str) and 1 <= len(owner) <= 100, 'PRIVATE_STORE_OWNER_REQUIRED')
        self._factory, self._key, self.owner = factory, key, owner
        self.synthetic, self.clock = synthetic, clock
        self.prefix = 'nfl_' if synthetic else 'nfl_private.'

    @classmethod
    def postgres(cls, dsn, key, *, owner='Robert Velarde'):
        # Parsing/connecting is lazy and occurs only after owner activation.
        def factory(readonly):
            try:
                import psycopg
                from psycopg.conninfo import conninfo_to_dict
                info = conninfo_to_dict(dsn)
                require(info.get('sslmode') == 'verify-full' and info.get('host'), 'PRIVATE_STORE_TLS_REQUIRED')
                return psycopg.connect(dsn, autocommit=True, connect_timeout=5,
                    options='-c statement_timeout=5000 -c lock_timeout=5000')
            except WorkflowError:
                raise
            except Exception:
                raise WorkflowError('PRIVATE_STORE_UNAVAILABLE') from None
        return cls(factory, key, owner=owner)

    @classmethod
    def synthetic_fixture(cls, path, key, *, owner='Robert Velarde', clock=utc):
        """Explicit fixture constructor, not an environment/upload/UI option."""
        path = Path(path).resolve()
        require(path.name.startswith('SYNTHETIC-'), 'SYNTHETIC_STORE_PATH_REQUIRED')
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS nfl_meta (version TEXT PRIMARY KEY)')
            con.execute('INSERT OR IGNORE INTO nfl_meta VALUES (?)', (VERSION,))
            for table in ('nfl_records', 'nfl_claims'):
                con.execute(f'CREATE TABLE IF NOT EXISTS {table} (owner_id TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, raw TEXT NOT NULL, signature TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(owner_id,id))')
                for action in ('UPDATE', 'DELETE'):
                    con.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only'); END")
        def factory(readonly):
            return sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True, timeout=5)
        return cls(factory, key, owner=owner, synthetic=True, clock=clock)

    def _execute(self, con, sql, args=()):
        return con.execute(sql if self.synthetic else sql.replace('?', '%s'), args)

    def _schema(self, con):
        require(self._execute(con, 'SELECT version FROM '+self.prefix+'meta').fetchall() == [(VERSION,)], 'PRIVATE_STORE_SCHEMA_UNSUPPORTED')
        for suffix in ('records', 'claims'):
            table = self.prefix + suffix
            if self.synthetic:
                columns = [x[1] for x in con.execute('PRAGMA table_info('+table+')')]
                require(columns == ['owner_id', 'id', 'kind', 'raw', 'signature', 'created_at'], 'PRIVATE_STORE_SCHEMA_UNSUPPORTED')
                for action in ('UPDATE', 'DELETE'):
                    row = con.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (table+'_'+action,)).fetchone()
                    expected = f"CREATE TRIGGER {table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only'); END"
                    require(row is not None and row[0] == expected, 'PRIVATE_STORE_PROTECTION_MISSING')
            else:
                # The application role must not own/administer its durable store.
                row = con.execute("SELECT r.rolsuper,r.rolcreatedb,r.rolcreaterole,has_schema_privilege(current_user,'nfl_private','CREATE'),has_table_privilege(current_user,%s,'UPDATE,DELETE,TRUNCATE'),pg_has_role(current_user,c.relowner,'MEMBER') FROM pg_roles r,pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE r.rolname=current_user AND n.nspname='nfl_private' AND c.relname=%s", (table, suffix)).fetchone()
                require(row is not None and not any(row), 'PRIVATE_STORE_ROLE_NOT_PRIVATE')
                public_or_other = con.execute("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))) a WHERE n.nspname='nfl_private' AND c.relname=%s AND a.grantee NOT IN (c.relowner,(SELECT oid FROM pg_roles WHERE rolname=current_user))", (suffix,)).fetchone()[0]
                schema_other = con.execute("SELECT count(*) FROM pg_namespace n CROSS JOIN LATERAL aclexplode(COALESCE(n.nspacl,acldefault('n',n.nspowner))) a WHERE n.nspname='nfl_private' AND a.grantee NOT IN (n.nspowner,(SELECT oid FROM pg_roles WHERE rolname=current_user))").fetchone()[0]
                require(public_or_other==schema_other==0,'PRIVATE_STORE_ACCESS_TOO_BROAD')
                triggers = con.execute("SELECT t.tgname,p.prosrc FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_proc p ON p.oid=t.tgfoid WHERE n.nspname='nfl_private' AND c.relname=%s AND NOT t.tgisinternal AND t.tgenabled='O'", (suffix,)).fetchall()
                require(set(triggers) == {(suffix+'_immutable', PG_PROTECT)}, 'PRIVATE_STORE_PROTECTION_MISSING')

    @contextmanager
    def _transaction(self, *, readonly):
        con = None
        try:
            con = self._factory(readonly)
            if self.synthetic:
                if readonly:
                    con.execute('PRAGMA query_only=ON')
                else:
                    con.execute('BEGIN IMMEDIATE')
            else:
                con.execute('BEGIN READ ONLY' if readonly else 'BEGIN')
            self._schema(con)
            yield con
            con.commit()
        except WorkflowError:
            if con is not None:
                con.rollback()
            raise
        except Exception:
            if con is not None:
                con.rollback()
            raise WorkflowError('PRIVATE_STORE_UNAVAILABLE') from None
        finally:
            if con is not None:
                con.close()

    def verify(self):
        with self._transaction(readonly=True):
            pass
        return dict(version=VERSION, private=True, durable_backend='SYNTHETIC_SQLITE' if self.synthetic else 'POSTGRESQL', synthetic=self.synthetic)

    def _signed(self, identity, kind, payload):
        value = dict(version=VERSION, owner=self.owner, id=identity, kind=kind,
                     payload=payload, created_at=self.clock())
        raw = encode(value)
        require(0 < len(raw.encode()) <= MAX_RECORD, 'PRIVATE_RECORD_SIZE')
        signature = hmac.new(self._key, raw.encode(), 'sha256').hexdigest()
        return raw, signature, value['created_at']

    def _authenticate(self, row, identity=None, kind=None):
        require(row is not None and isinstance(row[0], str) and len(row[0].encode()) <= MAX_RECORD, 'PRIVATE_RECORD_MISSING')
        raw, signature = row
        require(isinstance(signature, str) and hmac.compare_digest(signature, hmac.new(self._key, raw.encode(), 'sha256').hexdigest()), 'PRIVATE_RECORD_AUTHENTICATION_FAILED')
        value = json.loads(raw)
        require(set(value) == {'version','owner','id','kind','payload','created_at'} and value['version'] == VERSION and value['owner'] == self.owner, 'PRIVATE_RECORD_OWNER_CONFLICT')
        require((identity is None or value['id'] == identity) and (kind is None or value['kind'] == kind), 'PRIVATE_RECORD_IDENTITY_CONFLICT')
        return value

    def get(self, identity, *, kind=None):
        with self._transaction(readonly=True) as con:
            row = self._execute(con, 'SELECT raw,signature FROM '+self.prefix+'records WHERE owner_id=? AND id=?', (self.owner, identity)).fetchone()
        return self._authenticate(row, identity, kind)

    def list(self, kind, *, limit=100, offset=0):
        require(type(limit) is int and 1 <= limit <= 100 and type(offset) is int and 0 <= offset < 100_000_000, 'PRIVATE_RECORD_LIMIT')
        with self._transaction(readonly=True) as con:
            rows = self._execute(con, 'SELECT raw,signature FROM '+self.prefix+'records WHERE owner_id=? AND kind=? ORDER BY created_at DESC,id LIMIT ? OFFSET ?', (self.owner, kind, limit, offset)).fetchall()
        return [self._authenticate(row, kind=kind) for row in rows]

    def put(self, kind, payload, *, identity=None, first=False):
        identity = identity or digest(dict(kind=kind, payload=payload))
        raw, signature, created = self._signed(identity, kind, payload)
        with self._transaction(readonly=False) as con:
            self._execute(con, 'INSERT INTO '+self.prefix+'records (owner_id,id,kind,raw,signature,created_at) VALUES (?,?,?,?,?,?) ON CONFLICT(owner_id,id) DO NOTHING', (self.owner, identity, kind, raw, signature, created))
            row = self._execute(con, 'SELECT raw,signature FROM '+self.prefix+'records WHERE owner_id=? AND id=?', (self.owner, identity)).fetchone()
            value = self._authenticate(row, identity, kind)
            require(first or value['payload'] == payload, 'PRIVATE_RECORD_CONFLICT')
        return value

    def claim(self, identity, payload):
        raw, signature, created = self._signed(identity, 'claim', payload)
        with self._transaction(readonly=False) as con:
            cursor = self._execute(con, 'INSERT INTO '+self.prefix+'claims (owner_id,id,kind,raw,signature,created_at) VALUES (?,?,?,?,?,?) ON CONFLICT(owner_id,id) DO NOTHING', (self.owner, identity, 'claim', raw, signature, created))
            inserted = cursor.rowcount == 1
            row = self._execute(con, 'SELECT raw,signature FROM '+self.prefix+'claims WHERE owner_id=? AND id=?', (self.owner, identity)).fetchone()
            self._authenticate(row, identity, 'claim')
        require(inserted, 'PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY')
        return identity

    def spent(self, identity):
        with self._transaction(readonly=True) as con:
            row = self._execute(con, 'SELECT raw,signature FROM '+self.prefix+'claims WHERE owner_id=? AND id=?', (self.owner, identity)).fetchone()
        if row is not None:
            self._authenticate(row, identity, 'claim')
        return row is not None

    def claim_record(self, identity):
        with self._transaction(readonly=True) as con:
            row=self._execute(con,'SELECT raw,signature FROM '+self.prefix+'claims WHERE owner_id=? AND id=?',(self.owner,identity)).fetchone()
        return self._authenticate(row,identity,'claim')

    def save_native(self, result):
        from app_core import nfl_scenario_research as scenario
        p = scenario.read_result(result)  # Validate the real writer BEFORE signing.
        require(p['input']['payload']['evidence_label'] == ('SYNTHETIC' if self.synthetic else 'RETAINED'), 'PRIVATE_RESULT_LABEL_CONFLICT')
        subject=scenario.subject(p['input']['payload'])
        claim=self.claim_record('native:'+subject)['payload']
        review=self.get(claim['actual_review_id'],kind='inference-review')['payload']
        require(claim['subject_sha256']==subject and claim['maximum_native_calls']==1
            and review['at']==p['input']['payload']['owner_review']['reviewed_at']
            and review['attestation']==p['input']['payload']['owner_review']['attestation'], 'PRIVATE_RESULT_EXECUTION_AUTHORITY_CONFLICT')
        receipt = dict(version='nfl-private-scenario-capture-v1', export_id='nfl-scenario-result:'+result['sha256'], result_sha256=result['sha256'], input_sha256=p['input']['sha256'])
        self.put('native-result', dict(result=result, receipt=receipt, profile=dict(reader=p['reader'], model=p['model'])), identity=receipt['export_id'])
        e, s = p['input']['payload']['event'], p['input']['payload']['scenario']
        self.put('saved-index', dict(receipt=receipt, event=e, scenario=s, inference_time=p['inference_time']), identity='index:'+result['sha256'])
        return receipt

    def read_native(self, receipt):
        from app_core import nfl_scenario_research as scenario, nfl_scenario_zip as replay
        from app_core import nfl_inference_evidence as evidence
        # This anchor comes from the private authenticated native-writer store,
        # NEVER from a user ZIP or a self-hashed upload. No per-game catalog.
        value = self.get(receipt['export_id'], kind='native-result')['payload']
        require(value['receipt'] == receipt, 'PRIVATE_RESULT_RECEIPT_CONFLICT')
        profile = value['profile']
        model=profile['model']
        # Static replay authenticates the recorded producer, not live callable
        # objects (which can be deliberately blocked during read-only viewing).
        # Full model/callable equality is enforced by the native writer above.
        require(profile['reader']['sources'] == scenario.reader_binding()['sources']
            and model['predictor_id']=='score-distribution-v1:nfl'
            and model['target']=='selected_side_full_game_half_point_spread_cover'
            and model['configuration']==evidence.configuration()
            and model['artifacts']==evidence.artifacts()
            and model['feature_order']==list(evidence.FEATURES),'PRIVATE_RESULT_UNSUPPORTED_PRODUCER')
        anchor = dict(input_sha256=receipt['input_sha256'], receipt_sha256=replay.digest(receipt), profile='private-native-writer-v1')
        policy = dict(profiles={'private-native-writer-v1':profile}, records={receipt['result_sha256']:anchor})
        synthetic = replay.SyntheticPolicy(replay.encode(policy['records'])) if self.synthetic else None
        return replay._authenticate(value['result'], receipt, policy, synthetic)

    def download_native(self, receipt):
        self.read_native(receipt)
        result = self.get(receipt['export_id'], kind='native-result')['payload']['result']
        files = {'scenario-result.json':encode(result).encode(), 'private-receipt.json':encode(receipt).encode()}
        manifest = dict(version='nfl-private-scenario-export-v1', purpose='PRIVATE_RESEARCH', files={name:sha256(raw).hexdigest() for name,raw in files.items()})
        files['manifest.json'] = encode(manifest).encode()
        out = BytesIO()
        with ZipFile(out, 'w') as archive:
            for name, raw in sorted(files.items()):
                item = ZipInfo(name, date_time=(1980,1,1,0,0,0)); item.compress_type=ZIP_DEFLATED
                archive.writestr(item, raw)
        return out.getvalue()
