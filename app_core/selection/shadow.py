"""Shadow run of the EV selector during Refresh (roadmap P1.2).

Comparison only: it never changes analysis, selection, stakes or publication.
It runs when ``PARLAYPICKER_EV_SHADOW=true``. When the pipeline store is
configured (``PARLAYPICKER_PIPELINE_STORE_KEY``; see the orchestrator), every
game's shadow decision is recorded as a signed, append-only
``ev-shadow-decision`` record. Both selectors can then be graded against the
same outcomes before an owner decides on a switch (D7).
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

from app_core.selection import ev_selector
from app_core.pipeline.store import SPORTS, StoreError


def enabled(setting=os.environ.get):
    return str(setting('PARLAYPICKER_EV_SHADOW', 'false')).strip().lower() == 'true'


def _clean(value):
    """JSON-safe scalar: NaN and pandas NA become None."""
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, 'item') else value


def qualified_markets(exports_dir=None, evidence_db=None):
    """Qualified league x market set under owner decision D1, from graded exports and closes.

    Read-only. Any missing or unreadable input yields no qualified markets, never a guess.
    """
    import contextlib
    import io
    from pathlib import Path
    from app_core.evaluation import scoreboard, clv
    from app_core.selection import qualification
    root = Path(__file__).resolve().parents[2]
    exports_dir = Path(exports_dir or root / 'data' / 'backtest_exports')
    if evidence_db is None:
        from app_core.prediction_evidence import database_path
        evidence_db = database_path()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            board = scoreboard.scoreboard(scoreboard.load(exports_dir))
        closes = clv.report(clv.read_observations(evidence_db)) if Path(evidence_db).is_file() else None
        return qualification.qualified_set(qualification.assess(board, closes)), None
    except Exception as exc:  # unreadable evidence qualifies nothing
        return frozenset(), 'QUALIFICATION_UNAVAILABLE:' + type(exc).__name__


def run(analysis_df, current_picks, *, setting=os.environ.get, clock=None, qualified=None):
    """Return (decisions, summary, note). Records decisions when a store is configured."""
    if not enabled(setting):
        return None, dict(games=0), 'EV_SHADOW_DISABLED'
    qualification_note = None
    if qualified is None:
        qualified, qualification_note = qualified_markets()
    decisions = ev_selector.select(analysis_df, qualified=qualified, current=current_picks)
    summary = dict(ev_selector.summary(decisions), qualified_markets=sorted(' '.join(m) for m in qualified),
                   qualification_note=qualification_note)
    from app_core.pipeline.orchestrator import store_from_settings
    store, reason = store_from_settings(setting)
    if store is None or decisions.empty:
        return decisions, summary, reason or 'NO_GAMES'
    at = (clock or (lambda: datetime.now(timezone.utc)))().isoformat()
    recorded = 0
    for record in decisions.to_dict('records'):
        sport = str(record.get('league', '')).upper()
        if sport not in SPORTS:
            continue
        payload = dict({k: _clean(v) for k, v in record.items()}, decided_at=at)
        try:
            store.put(sport, 'ev-shadow-decision', payload)
            recorded += 1
        except StoreError as exc:
            return decisions, dict(summary, recorded=recorded), 'EV_SHADOW_RECORD_FAILED:' + str(exc)
    return decisions, dict(summary, recorded=recorded), None
