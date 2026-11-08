"""A small per-market store for the app: qualified markets without the full evidence store (go-live A).

The app's bridge, qualification and market gate read the canonical prospective store, and
restoring all of it (about 106,000 Drive objects) does not finish inside the app (2026-11-07:
the background restore ran out of memory). Only a qualified market's own records are needed.

**Publish** (decision capture, which holds the full store and has Drive write access): for each
qualified market, ``export`` copies that market's rows (``SLICE``: its events, quotes, closes,
results, predictions, models, calibrations and their result links, plans, artifacts and
deployment reviews) into a fresh store with the same schema, and ``publish`` writes it gzip-
compressed and write-once to Drive under a 15-minute slot key.

**Read** (the app, off unless ``PARLAYPICKER_QUALIFIED_MARKET_FILE=true``): ``refresh`` fetches the
newest slice of each owner-activated market (looking back ``LOOKBACK_SLOTS``), merges them into
one local store (``store_path``), and swaps it in atomically. The live contract and the bridge
then read that store, so the authority chain runs unchanged on it:
``deployment_state`` re-evaluates the plan and checks the artifact hash, and the gate replays the
quote from its retained provider bytes. The slice adds no authority, and a forged or partial
slice fails those same checks.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import gzip
import os
from pathlib import Path
import tempfile

from app_core import prospective_evidence as evidence

PREFIX = "parlaypicker/qualified-market-v1"
MODE_ENV = "PARLAYPICKER_QUALIFIED_MARKET_FILE"
STORE_ENV = "PARLAYPICKER_QUALIFIED_MARKET_STORE"
SLOT_MINUTES = 15
LOOKBACK_SLOTS = 12          # three hours: decision capture runs every 15 minutes through a slate
# (table, filter): rows of one sport and market. Events have no market, so the sport's events.
SLICE = (
    ("prospective_event", "sport=:sport"),
    ("prospective_quote", "sport=:sport AND market_family=:family"),
    ("prospective_close", "sport=:sport AND market_family=:family"),
    ("prospective_result", "sport=:sport AND market_family=:family"),
    ("prospective_model", "sport=:sport AND market_family=:family"),
    ("prospective_model_training_result",
     "model_id IN (SELECT model_id FROM src.prospective_model WHERE sport=:sport AND market_family=:family)"),
    ("prospective_calibration", "sport=:sport AND market_family=:family"),
    ("prospective_calibration_result",
     "calibration_id IN (SELECT calibration_id FROM src.prospective_calibration "
     "WHERE sport=:sport AND market_family=:family)"),
    ("prospective_prediction", "sport=:sport AND market_family=:family"),
    ("prospective_validation_plan", "sport=:sport AND market_family=:family"),
    ("prospective_validation_artifact", "sport=:sport AND market_family=:family"),
    ("prospective_deployment_review", "sport=:sport AND market_family=:family"),
)


def enabled(setting=None) -> bool:
    return str((setting or os.environ.get)(MODE_ENV, "") or "").strip().lower() == "true"


def store_path(setting=None) -> Path:
    return Path((setting or os.environ.get)(STORE_ENV, "") or "data/qualified-market/qualified-markets.sqlite3")


def slot(now: datetime) -> datetime:
    now = now.astimezone(timezone.utc)
    return now.replace(minute=now.minute - now.minute % SLOT_MINUTES, second=0, microsecond=0)


def key(sport: str, family: str, at: datetime) -> str:
    return f"{PREFIX}/{sport}-{family}/{slot(at).strftime('%Y-%m-%dT%H%MZ')}.sqlite.gz"


def _copy(target_db, source: Path, sport: str, family: str) -> dict:
    counts = {}
    target_db.execute("ATTACH DATABASE ? AS src", (str(Path(source).resolve()),))
    try:
        for table, where in SLICE:
            columns = [r[1] for r in target_db.execute(f"PRAGMA main.table_info({table})")]
            names = ",".join(columns)
            cursor = target_db.execute(f"INSERT OR IGNORE INTO main.{table} ({names}) "
                                       f"SELECT {names} FROM src.{table} WHERE {where}",
                                       {"sport": sport, "family": family})
            counts[table] = cursor.rowcount
        target_db.commit()
    finally:
        target_db.execute("DETACH DATABASE src")
    return counts


def export(source, target, markets) -> dict:
    """Copy ``markets`` ({(sport, family)}) from the store at ``source`` into a new store at ``target``."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    counts = {}
    with closing(evidence.connect(target)) as db:
        db.execute("PRAGMA foreign_keys=OFF")  # rows are copied verbatim, parents first or not
        for sport, family in sorted(markets):
            counts[f"{sport}:{family}"] = _copy(db, source, sport, family)
    return counts


def publish(source, client, sport, family, *, now=None) -> dict:
    """Write one market's slice to Drive for this 15-minute slot, once. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        name = key(sport, family, now)
        try:
            client.get_object(Key=name)
            return {"status": "ALREADY_PUBLISHED", "key": name}
        except Exception:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "slice.sqlite3"
            counts = export(source, target, {(sport, family)})
            body = gzip.compress(target.read_bytes())
        try:
            client.put_object(Key=name, Body=body, IfNoneMatch="*")
        except Exception as exc:
            if type(exc).__name__ == "AlreadyExists":
                return {"status": "ALREADY_PUBLISHED", "key": name}
            raise
        return {"status": "PUBLISHED", "key": name, "bytes": len(body), "rows": counts[f"{sport}:{family}"]}
    except Exception as exc:  # decision capture continues without the app file
        return {"status": "ERROR", "error": type(exc).__name__}


def fetch(client, sport, family, *, now=None):
    """(bytes, key) of the newest slice within ``LOOKBACK_SLOTS``, or (None, None)."""
    now = now or datetime.now(timezone.utc)
    for back in range(LOOKBACK_SLOTS):
        name = key(sport, family, now - timedelta(minutes=SLOT_MINUTES * back))
        try:
            return gzip.decompress(client.get_object(Key=name)["Body"].read()), name
        except Exception:
            continue
    return None, None


def refresh(client, markets, *, now=None, target=None) -> dict:
    """Merge the newest slice of each market into one local store and swap it in atomically."""
    now = now or datetime.now(timezone.utc)
    target = Path(target or store_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    found, missing = {}, []
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        merged = Path(tmp) / "merged.sqlite3"
        with closing(evidence.connect(merged)) as db:
            db.execute("PRAGMA foreign_keys=OFF")
            for sport, family in sorted(markets):
                raw, name = fetch(client, sport, family, now=now)
                if raw is None:
                    missing.append(f"{sport}:{family}")
                    continue
                part = Path(tmp) / f"{sport}-{family}.sqlite3"
                part.write_bytes(raw)
                _copy(db, part, sport, family)
                found[f"{sport}:{family}"] = name
        if not found:
            return {"status": "NO_SLICE", "missing": missing}
        os.replace(merged, target)
    return {"status": "REFRESHED", "slices": found, "missing": missing, "store": str(target)}
