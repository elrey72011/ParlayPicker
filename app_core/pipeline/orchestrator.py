"""Batch orchestrator for the shared pipeline (roadmap M1.4).

``run_batch`` processes game/market requests for any registered sport with no
game-specific code. For each request it:
1. checks the sport has an adapter and the game hasn't started;
2. checks quote freshness when a quote time is known (an unknown quote time is
   recorded as UNKNOWN, never invented);
3. creates (or reuses) a calculation operation for the exact subject, so a
   rerender or restart on the same day returns the stored forecast instead of
   recomputing;
4. runs adapter features plus forecast inside that operation.
One request failing never stops the others. Every request ends with a status
and a specific reason.

``shadow`` runs NFL rows from ordinary Refresh through this path for
comparison only. It never changes selection, staking or publication, and it
runs only when ``PARLAYPICKER_PIPELINE_V2`` is ``true``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path

from app_core.pipeline.adapter import AdapterError, adapter_for, instant
from app_core.pipeline.operations import COMPLETED, EXPIRED, FAILED, CANCELLED, OperationError, Operations
from app_core.pipeline.store import Store, StoreError
from app_core.pipeline import nfl as _nfl  # noqa: F401  registers the NFL adapter

MAX_QUOTE_AGE = timedelta(minutes=30)
HISTORY_ATTEMPTS_PER_DAY = 3
HISTORY_WINDOW = timedelta(minutes=10)


@dataclass(frozen=True)
class Request:
    sport: str
    event: dict          # event_id, home, away, kickoff_utc
    market_type: str
    line: float | None
    quote_observed_at: str | None = None
    extra: dict = field(default_factory=dict)  # carried through to the result, e.g. price


def _result(request, status, reason, **values):
    return dict(sport=request.sport, event_id=request.event.get('event_id'), home=request.event.get('home'),
                away=request.event.get('away'), market_type=request.market_type, line=request.line,
                status=status, reason=reason, probability=values.pop('probability', None),
                operation_id=values.pop('operation_id', None), state=values.pop('state', None),
                quote_age_seconds=values.pop('quote_age_seconds', 'UNKNOWN'), **request.extra, **values)


def _code(exc):
    first = exc.args[0] if exc.args else ''
    return first if isinstance(first, str) and first.isupper() and ' ' not in first else type(exc).__name__


def run_batch(requests, *, ops: Operations, history_for, as_of_utc: str, max_quote_age=MAX_QUOTE_AGE, adapters=None):
    """Run every request; return one result row per request, in order.

    ``adapters`` ({sport: adapter}) limits the run to those sports; by default the
    global registry is used.
    """
    now = instant(as_of_utc)
    results = []
    for request in requests:
        try:
            if adapters is None:
                adapter = adapter_for(request.sport)
            elif request.sport in adapters:
                adapter = adapters[request.sport]
            else:
                raise AdapterError('ADAPTER_SPORT_NOT_SUPPORTED')
            kickoff = instant(request.event.get('kickoff_utc'), 'EVENT_KICKOFF_INVALID')
            if now >= kickoff:
                results.append(_result(request, 'UNAVAILABLE', 'GAME_STARTED'))
                continue
            age = 'UNKNOWN'
            if request.quote_observed_at:
                age = (now - instant(request.quote_observed_at, 'QUOTE_TIME_INVALID')).total_seconds()
                if age < 0 or age > max_quote_age.total_seconds():
                    results.append(_result(request, 'UNAVAILABLE', 'QUOTE_STALE', quote_age_seconds=age))
                    continue
            subject = dict(event_id=request.event['event_id'], home=request.event['home'], away=request.event['away'],
                           kickoff_utc=request.event['kickoff_utc'], market_type=request.market_type, line=request.line,
                           quote_observed_at=request.quote_observed_at, features_as_of_date=now.date().isoformat())
            op = ops.create(request.sport, 'calculation', subject, not_after=request.event['kickoff_utc'],
                            review_required=False)
            status = ops.status(request.sport, op['id'])
            if status['state'] == COMPLETED:
                forecast = ops.result(request.sport, op['id'])['output']
            elif status['state'] in {FAILED, EXPIRED, CANCELLED}:
                reason = status['events'][-1].get('reason', status['state'])
                results.append(_result(request, 'UNAVAILABLE', reason, operation_id=op['id'], state=status['state'],
                                       quote_age_seconds=age))
                continue
            else:
                def work(operation, request=request):
                    features = adapter.features(request.event, history_for(request.sport, request), as_of_utc=as_of_utc)
                    return adapter.forecast(features, request.market_type, request.line).record()
                forecast = ops.run(request.sport, op['id'], work)['output']
            results.append(_result(request, forecast['status'], forecast['reason'], probability=forecast['probability'],
                                   operation_id=op['id'], state=COMPLETED, quote_age_seconds=age,
                                   model_id=forecast['model_id'], features_sha256=forecast['features_sha256']))
        except (AdapterError, OperationError, StoreError, ValueError, KeyError, TypeError) as exc:
            results.append(_result(request, 'UNAVAILABLE', _code(exc)))
    return results


# -- NFL history acquisition (one claimed download per as-of date) -----------

def _season(day):
    return day.year if day.month >= 3 else day.year - 1


def fetch_nflverse_schedule(season):
    """Public nflverse schedule, the same source ordinary Refresh already uses."""
    import nfl_data_py as nfl
    from core.nfl_teams import nfl_stats_identity
    rows = []
    for record in nfl.import_schedules([season]).to_dict('records'):
        home, away = record.get('home_score'), record.get('away_score')
        if home is None or away is None or any(isinstance(v, float) and math.isnan(v) for v in (home, away)):
            continue
        rows.append(dict(game_date=str(record.get('gameday'))[:10],
                         home=nfl_stats_identity(record['home_team'], schedule_code=True),
                         away=nfl_stats_identity(record['away_team'], schedule_code=True),
                         home_score=int(home), away_score=int(away)))
    return rows


def nfl_history(ops: Operations, as_of_utc: str, *, fetch=fetch_nflverse_schedule, clock=None):
    """Completed NFL games for the season, downloaded at most once per successful attempt.

    A failed or expired attempt is kept as is. A recorded successor is allowed,
    up to ``HISTORY_ATTEMPTS_PER_DAY`` attempts per as-of date.
    """
    now = instant(as_of_utc)
    season = _season(now.date())
    subject = dict(source='nflverse-schedules', season=season, as_of_date=now.date().isoformat())
    predecessor = None
    for _ in range(HISTORY_ATTEMPTS_PER_DAY):
        op = ops.create('NFL', 'acquisition', subject, not_after=(now + HISTORY_WINDOW).isoformat(),
                        review_required=False, predecessor=predecessor)
        status = ops.reconcile('NFL', op['id'])
        if status['state'] == COMPLETED:
            return ops.result('NFL', op['id'])['output']['games']
        if status['state'] in {FAILED, EXPIRED, CANCELLED}:
            predecessor = op['id']
            continue
        try:
            return ops.run('NFL', op['id'], lambda operation: dict(games=fetch(season)))['output']['games']
        except StoreError:
            raise
        except Exception:
            predecessor = op['id']  # recorded as FAILED by run(); try a recorded successor
    raise AdapterError('NFL_HISTORY_UNAVAILABLE')


# -- Ordinary Refresh shadow ----------------------------------------------------

def enabled(setting=os.environ.get):
    return str(setting('PARLAYPICKER_PIPELINE_V2', 'false')).strip().lower() == 'true'


def store_from_settings(setting=os.environ.get):
    """Configured store, or (None, reason). Never creates a production schema."""
    key = setting('PARLAYPICKER_PIPELINE_STORE_KEY', '') or ''
    if len(key.encode()) < 32:
        return None, 'PIPELINE_STORE_NOT_CONFIGURED'
    owner = setting('PARLAYPICKER_PIPELINE_OWNER', '') or 'owner'
    dsn = setting('PARLAYPICKER_PIPELINE_DATABASE_URL', '') or ''
    if dsn:
        return Store.postgres(dsn, key.encode(), owner=owner), None
    path = setting('PARLAYPICKER_PIPELINE_SQLITE_PATH', '') or 'data/pipeline/pipeline-store.sqlite'
    return Store.sqlite(Path(path), key.encode(), owner=owner), None


def requests_from_analysis(analysis_df, sports=('NFL',)):
    """Spread/total rows for ``sports`` from ordinary Refresh, as pipeline requests."""
    from core.nfl_teams import nfl_stats_identity
    from app_core.pipeline.nba_history import team as nba_team
    identity = {'NFL': nfl_stats_identity, 'NBA': nba_team}
    requests = []
    if analysis_df is None or getattr(analysis_df, 'empty', True):
        return requests
    for row in analysis_df.to_dict('records'):
        sport = str(row.get('league', row.get('League', ''))).upper()
        if sport not in sports:
            continue
        market_type = str(row.get('market_type', '')).lower()
        if market_type not in {'spread_home', 'spread_away', 'total_over', 'total_under'}:
            continue
        line = row.get('total_line') if market_type.startswith('total') else row.get('spread_line')
        try:
            line = float(line)
        except (TypeError, ValueError):
            line = None
        if line is not None and not math.isfinite(line):
            line = None
        event = dict(event_id=str(row.get('provider_event_id') or row.get('matchup_id') or ''),
                     home=identity[sport](row.get('home_team', '')) or str(row.get('home_team', '')),
                     away=identity[sport](row.get('away_team', '')) or str(row.get('away_team', '')),
                     kickoff_utc=_utc_text(row.get('game_start_utc') or row.get('commence_time')))
        observed = row.get('odds_recorded_at') or row.get('quote_time') or None
        requests.append(Request(sport, event, market_type, line, _utc_text(observed) if observed else None,
                                extra=dict(odds_american=row.get('odds_american'))))
    return requests


def _utc_text(value):
    try:
        import pandas as pd
        moment = pd.Timestamp(value)
        moment = moment.tz_localize('UTC') if moment.tzinfo is None else moment.tz_convert('UTC')
        return moment.isoformat()
    except (TypeError, ValueError):
        return ''


def nba_enabled(setting=os.environ.get):
    """NBA research model in the shadow: PARLAYPICKER_PIPELINE_NBA=true and the artifact exists."""
    from app_core.pipeline.nba import ARTIFACT
    return str(setting('PARLAYPICKER_PIPELINE_NBA', 'false')).strip().lower() == 'true' and ARTIFACT.is_file()


def shadow(analysis_df, *, setting=os.environ.get, clock=None, fetch=fetch_nflverse_schedule, fetch_nba_day=None):
    """Shadow results for ordinary Refresh, or ([], reason) when not run."""
    if not enabled(setting):
        return [], 'PIPELINE_V2_DISABLED'
    from app_core.pipeline import nba as nba_adapter, nfl as nfl_adapter
    adapters = {'NFL': nfl_adapter.ADAPTER}
    if nba_enabled(setting):
        adapters['NBA'] = nba_adapter.ADAPTER
    requests = requests_from_analysis(analysis_df, sports=tuple(adapters))
    if not requests:
        return [], 'NO_NFL_SPREAD_OR_TOTAL_ROWS'
    store, reason = store_from_settings(setting)
    if store is None:
        return [], reason
    clock = clock or (lambda: datetime.now(timezone.utc))
    ops = Operations(store, clock=clock)
    as_of = clock().isoformat()
    histories, failures = {}, {}
    for sport in sorted({r.sport for r in requests}):
        try:
            if sport == 'NFL':
                histories[sport] = nfl_history(ops, as_of, fetch=fetch)
            else:
                from app_core.pipeline import nba_history
                histories[sport] = nba_history.history(ops, as_of, **({'fetch': fetch_nba_day} if fetch_nba_day else {}))
        except (AdapterError, OperationError, StoreError) as exc:
            failures[sport] = _code(exc)
    runnable = [r for r in requests if r.sport in histories]
    done = iter(run_batch(runnable, ops=ops, history_for=lambda sport, request: histories[sport],
                          as_of_utc=as_of, adapters=adapters))
    rows = [next(done) if r.sport in histories else _result(r, 'UNAVAILABLE', failures[r.sport]) for r in requests]
    return rows, '; '.join(f'{sport}_HISTORY_UNAVAILABLE' for sport in sorted(failures)) or None
