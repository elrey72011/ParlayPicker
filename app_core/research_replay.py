"""Private append-only replay of supplied research facts; no remote or authority reader."""
from __future__ import annotations
from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import math
import numpy as np
import pandas as pd
from app_core.research_estimate_trace import SOURCE_FIELDS, EXPORT_FIELDS

REPLAY_COLUMNS = frozenset(SOURCE_FIELDS + EXPORT_FIELDS + """ml_feature_eligible stats_resolution_status football_feature_receipt
provider_quotes
home_team away_team game_date game_time_est Home Away Local Date Commence (Local)
quote_timestamp sportsbook book opposing_odds_source quote_binding_verified
best_available_rank best_available_family_rank best_available_selected
best_available_selection_policy best_available_probability_source
production_win_probability production_edge production_expected_value
production_eligible wager_approved Bettable Trial_Stake Production_Gate_Reason
Status_Reason qualification_reason coverage_reason wager_contract
selection_label edge price_break_even approval_reason reason quote_reason
research_display research_estimate_trace maturity conservative_ev
espn_event_id mlb_game_pk game_number gemini_review_status gemini_reviewed_at
gemini_review_model gemini_review_input_hash gemini_verified_context
gemini_supporting_evidence gemini_missing_information nfl_context_status
feature_home_last_game_summary feature_away_last_game_summary injury_home_summary
injury_away_summary injury_context_source injury_context_status""".split()) | {"Local Date","Commence (Local)"}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


def cell(value, *, contract=False):
    # Lossless scalar states avoid CSV's missing/invalid/bool type coercion.
    if value is None: return {"type":"none"}
    if value is pd.NA: return {"type":"pd.NA"}
    if value is pd.NaT: return {"type":"pd.NaT"}
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,datetime): return {"type":"datetime","value":value.isoformat()}
    if isinstance(value,float) and not math.isfinite(value):
        return {"type":"nonfinite","value":"nan" if math.isnan(value) else "inf" if value>0 else "-inf"}
    if isinstance(value,(str,bool,int,float)): return {"type":"scalar","value":value}
    if contract and isinstance(value,dict):
        from core.live_wager_contract import PUBLIC_FIELDS
        if set(value)==set(PUBLIC_FIELDS):
            return {"type":"contract","value":{k:cell(v) for k,v in value.items()}}
    return {"type":"invalid"}  # Arbitrary objects/configuration never enter retention.


def value(item):
    kind=item["type"]
    if kind=="none": return None
    if kind=="pd.NA": return pd.NA
    if kind=="pd.NaT": return pd.NaT
    if kind=="datetime": return pd.Timestamp(item["value"])
    if kind=="nonfinite": return float(item["value"])
    if kind=="scalar": return item["value"]
    if kind=="contract": return {k:value(v) for k,v in item["value"].items()}
    if kind=="invalid": return float("nan")
    raise ValueError("Unknown retained scalar state")


def frame_payload(frame):
    if frame is None: return None
    if not frame.columns.is_unique: raise ValueError("Replay requires unique frame columns")
    columns=[c for c in frame if c in REPLAY_COLUMNS]
    return {"columns":columns,"rows":[[cell(row[c],contract=c=="wager_contract") for c in columns]
                                     for row in frame.to_dict("records")]}


def frame_from_payload(payload):
    if payload is None: return None
    return pd.DataFrame([[value(c) for c in row] for row in payload["rows"]],
                        columns=payload["columns"],dtype=object)


def setup(db):
    db.execute("CREATE TABLE IF NOT EXISTS research_replay_sources (snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(snapshot_id), export_run_id TEXT NOT NULL, snapshot_payload_hash TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS research_replay_exports (export_id TEXT PRIMARY KEY, package_hash TEXT NOT NULL, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS research_source_intakes (reference TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL)")
    for table in ("research_replay_sources","research_replay_exports","research_source_intakes"):
        for action in ("UPDATE","DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'research replay is append-only'); END")


def original_frames(candidates, card, producer):
    return {"producer":frame_payload(producer),"candidates":frame_payload(candidates),"card":frame_payload(card)}


def retain_source(db, snapshot_id, run_id, snapshot_hash, original, captured, card):
    payload=encode({"version":1,"snapshot_id":snapshot_id,"export_run_id":run_id,
        "snapshot_payload_hash":snapshot_hash,"original":original,
        "captured_candidates":frame_payload(captured),"captured_card":frame_payload(card)})
    db.execute("INSERT INTO research_replay_sources VALUES (?,?,?,?,?)",
               (snapshot_id,run_id,snapshot_hash,payload,digest(payload)))


def _source(db, snapshot_id):
    row=db.execute("SELECT s.payload,s.payload_hash,s.snapshot_payload_hash,p.candidates,p.decisions,p.inputs,p.payload_hash FROM research_replay_sources s JOIN snapshots p USING(snapshot_id) WHERE snapshot_id=?",(snapshot_id,)).fetchone()
    if row is None: return None
    raw,expected,snapshot_hash,*snapshot=row
    if digest(raw)!=expected or snapshot_hash!=snapshot[-1] or digest("\0".join(snapshot[:3]))!=snapshot_hash:
        raise ValueError("Retained research source integrity mismatch")
    return json.loads(raw),expected


def retain_export(boards, package, games, candidates, *, path=None):
    from app_core.prediction_evidence import connect
    from app_core.public_board import validate_package
    validate_package(package)
    identities=set()
    for frame in (games,candidates):
        if isinstance(frame,pd.DataFrame):
            for row in frame.to_dict("records"):
                sid=row.get("snapshot_id");run=row.get("export_run_id")
                if isinstance(sid,str) and sid and isinstance(run,str) and run: identities.add((sid,run))
    with closing(connect(path)) as db,db:
        links=[]
        for sid,run in sorted(identities):
            source=_source(db,sid)
            if source and source[0]["export_run_id"]!=run: raise ValueError("Replay snapshot/run identity mismatch")
            links.append({"snapshot_id":sid,"export_run_id":run,"source_hash":source[1] if source else None,
                          "state":"RETAINED" if source else "UNKNOWN"})
        payload={"version":1,"source_links":links,"source_boundary":"RETAINED" if links and all(l["state"]=="RETAINED" for l in links) else "UNKNOWN",
            "publication_source":{"games":frame_payload(games),"candidates":frame_payload(candidates)},
            "boards":{family:frame_payload(frame) for family,frame in zip(("overall","sides","totals"),boards)},
            "per_game_csv":{family:frame.to_csv(index=False) for family,frame in zip(("overall","sides","totals"),boards)},
            "package_hash":digest(encode(package)),"package":package}
        raw=encode(payload);export_id=digest(raw)
        existing=db.execute("SELECT package_hash,payload FROM research_replay_exports WHERE export_id=?",(export_id,)).fetchone()
        if existing and existing!=(payload["package_hash"],raw): raise ValueError("Replay export conflict")
        db.execute("INSERT OR IGNORE INTO research_replay_exports VALUES (?,?,?)",(export_id,payload["package_hash"],raw))
    return {"export_id":export_id,"package_hash":payload["package_hash"],"source_boundary":payload["source_boundary"]}


def read_export(export_id, *, path=None):
    # Explicit read-only local database; never restore, initialize or acquire.
    from app_core.prediction_evidence import database_path
    target=Path(path or database_path()).resolve()
    with closing(sqlite3.connect(target.as_uri()+"?mode=ro",uri=True)) as db:
        row=db.execute("SELECT package_hash,payload FROM research_replay_exports WHERE export_id=?",(export_id,)).fetchone()
        if row is None: raise ValueError("Retained research export not found")
        package_hash,raw=row
        payload=json.loads(raw)
        if digest(raw)!=export_id or payload["package_hash"]!=package_hash or digest(encode(payload["package"]))!=package_hash:
            raise ValueError("Retained research export integrity mismatch")
        sources={}
        for link in payload["source_links"]:
            if link["state"]=="UNKNOWN": continue
            source=_source(db,link["snapshot_id"])
            if source is None or source[1]!=link["source_hash"] or source[0]["export_run_id"]!=link["export_run_id"]:
                raise ValueError("Retained research source binding mismatch")
            sources[link["snapshot_id"]]=source[0]
        return payload,sources



def download_bundle(receipt, *, expected_package_hash, path=None):
    """Build an owner download only after verifying the immutable local evidence.

    The caller's preview package identity prevents a cached receipt from exporting
    another preview. UNKNOWN links stay explicit; no database repair or fetch.
    """
    from io import BytesIO
    from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
    payload, sources = read_export(receipt["export_id"], path=path)
    expected = {"export_id":receipt["export_id"], "package_hash":payload["package_hash"],
                "source_boundary":payload["source_boundary"]}
    if receipt != expected or expected_package_hash != payload["package_hash"]:
        raise ValueError("Retained research replay receipt/package mismatch")
    boundary = "RETAINED" if payload["source_links"] and all(
        link["state"] == "RETAINED" for link in payload["source_links"]) else "UNKNOWN"
    if boundary != payload["source_boundary"]:
        raise ValueError("Retained research source boundary mismatch")
    files = {"export.json":encode(payload).encode(),
             "package.json":encode(payload["package"]).encode()}
    links = []
    for link in payload["source_links"]:
        state = link["state"]
        if state not in {"RETAINED", "UNKNOWN"} or (state == "UNKNOWN" and link["source_hash"] is not None):
            raise ValueError("Retained research source receipt mismatch")
        source = sources.get(link["snapshot_id"]) if state == "RETAINED" else None
        name = "sources/"+link["source_hash"]+".json" if source is not None else None
        if source is not None:
            if source["snapshot_id"] != link["snapshot_id"]:
                raise ValueError("Retained research snapshot identity mismatch")
            files[name] = encode(source).encode()
            if hashlib.sha256(files[name]).hexdigest() != link["source_hash"]:
                raise ValueError("Retained research source receipt mismatch")
        links.append(dict(link, source_file=name,
                          snapshot_payload_hash=source["snapshot_payload_hash"] if source is not None else None))
    for family in ("overall", "sides", "totals"):
        files["per-game/"+family+".csv"] = payload["per_game_csv"][family].encode()
    if hashlib.sha256(files["export.json"]).hexdigest() != receipt["export_id"]:
        raise ValueError("Retained research export receipt mismatch")
    verified = dict(version=1, **expected, source_links=links,
                    files={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()})
    files["receipt.json"] = encode(verified).encode()
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, raw in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, raw)
    return output.getvalue(), verified
