"""Read-only index of existing private scenario receipts; no inference or setup."""
from contextlib import closing
import re
import sqlite3

from app_core import nfl_scenario_research as scenario, research_replay

PREFIX = 'nfl-scenario-result:'
MAX_LIST = 100
MAX_SELECTED = 4


def store_path():
    return research_replay._scenario_path(None)


def receipt(export_id, input_hash):
    result_hash = export_id.removeprefix(PREFIX) if isinstance(export_id, str) else ''
    scenario.require(export_id == PREFIX + result_hash
                     and re.fullmatch('[0-9a-f]{64}', result_hash) is not None
                     and isinstance(input_hash, str)
                     and re.fullmatch('[0-9a-f]{64}', input_hash) is not None,
                     'NFL_SCENARIO_INTEGRITY')
    return dict(version='nfl-private-scenario-capture-v1', export_id=export_id,
                result_sha256=result_hash, input_sha256=input_hash)


def load(saved_receipt):
    """Authenticate saved facts at their ORIGINAL inference clock, even postgame."""
    result = research_replay.read_scenario(saved_receipt, path=store_path())
    return scenario.read_result(result)


def catalog():
    """Bounded, authenticated descriptors, not stored copies of original bodies."""
    path = store_path()
    research_replay.scenario_store_ready(path=path)
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            rows = db.execute(
                "SELECT export_id,package_hash,length(CAST(payload AS BLOB)) "
                "FROM research_replay_exports WHERE export_id LIKE ? "
                "ORDER BY export_id LIMIT ?", (PREFIX + '%', MAX_LIST + 1)).fetchall()
    except sqlite3.Error:
        raise ValueError('NFL_SCENARIO_STORE_UNAVAILABLE') from None
    records, rejected = [], []
    for export_id, input_hash, size in rows[:MAX_LIST]:
        try:
            scenario.require(0 < size <= scenario.MAX_PACKET, 'NFL_SCENARIO_PACKET_LIMIT')
            saved = receipt(export_id, input_hash)
            p = load(saved)
            e, s = p['input']['payload']['event'], p['input']['payload']['scenario']
            records.append(dict(receipt=saved, selected_team=s['selected_team'],
                signed_handicap=s['signed_handicap'], home=e['home'], away=e['away'],
                inference_time=p['inference_time']))
        except (ValueError, TypeError, KeyError, OSError):
            # A corrupt or incompatible record cannot hide other valid records.
            rejected.append(export_id)
    return dict(store=str(path), records=records, rejected=rejected,
                truncated=len(rows) > MAX_LIST)


def private_download(saved_receipt):
    return research_replay.download_scenario(saved_receipt, path=store_path())
