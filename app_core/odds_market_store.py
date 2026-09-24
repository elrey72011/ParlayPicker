"""Sport-isolated, append-only research snapshots for NBA, NCAAB and NHL.

These are observations, not model predictions or production approvals.  Each
sport has its own database and remote prefix so a restore never pools evidence.
"""

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3

from app_core.prospective_sync import sync_records


SPORTS = frozenset({"NBA", "NCAAB", "NHL"})
KINDS = frozenset({"capture", "scores", "pregame_close_candidate"})


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class OddsMarketStore:
    def __init__(self, sport):
        if sport not in SPORTS:
            raise ValueError("unsupported_research_sport")
        self.sport = sport
        self.PREFIX = f"parlaypicker/{sport.lower()}-market-v1/"

    def connect(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS records_{action} BEFORE {action} ON records "
                       "BEGIN SELECT RAISE(ABORT, 'append-only'); END")
        return db

    def insert(self, record, path):
        if (record.get("schema") != 1 or record.get("kind") not in KINDS or
                record.get("data", {}).get("sport") != self.sport):
            raise ValueError("invalid_sport_research_record")
        raw = encode(record)
        key = hashlib.sha256(raw).hexdigest()
        with closing(self.connect(path)) as db, db:
            db.execute("INSERT OR IGNORE INTO records VALUES (?, ?)", (key, raw.decode()))
        return key

    def save(self, kind, data, path):
        if kind not in KINDS or data.get("sport") != self.sport:
            raise ValueError("invalid_sport_research_record")
        if kind in {"capture", "pregame_close_candidate"}:
            from app_core.ncaaf_history import timestamp
            now = datetime.now(timezone.utc)
            if any(timestamp(event.get("start")) is None or
                   timestamp(event.get("response_received_at")) is None or
                   not timestamp(event["response_received_at"]) < timestamp(event["start"]) or
                   not now < timestamp(event["start"])
                   for event in data.get("events", [])):
                raise ValueError("sport_research_capture_started")
        return self.insert({"schema": 1, "kind": kind,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "data": data}, path)

    def records(self, path):
        with closing(self.connect(path)) as db:
            rows = db.execute("SELECT id,payload FROM records ORDER BY rowid").fetchall()
        result = []
        for key, raw in rows:
            if hashlib.sha256(raw.encode()).hexdigest() != key:
                raise ValueError("sport_research_record_integrity")
            record = json.loads(raw)
            if record.get("data", {}).get("sport") != self.sport:
                raise ValueError("sport_research_identity_conflict")
            result.append({"id": key, **record})
        return sorted(result, key=lambda r: (r["created_at"], r["id"]))

    def sync(self, path, *, client, folder, session=None):
        return sync_records(self, path, client, folder, session)

    encode = staticmethod(encode)


STORES = {sport: OddsMarketStore(sport) for sport in sorted(SPORTS)}


def for_sport(sport):
    try:
        return STORES[sport]
    except KeyError:
        raise ValueError("unsupported_research_sport") from None
