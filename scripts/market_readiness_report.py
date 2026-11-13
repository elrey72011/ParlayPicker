"""Read-only 12-market readiness report from the Drive evidence (roadmap C0).

Usage (GitHub Actions supplies the credentials):
    python scripts/market_readiness_report.py --directory /tmp/readiness --output readiness.json \
        [--cache-dir /tmp/drive-cache] [--with-sources]

It restores the canonical prospective-evidence store into an empty local
directory and reports
``prospective_evidence.all_market_readiness`` for all 12 sport × market
families, with the next blocker for each. ``--with-sources`` also restores each
sport's research store (descriptive source counts only; much slower).

The canonical store is tens of thousands of small Drive objects. With
``--cache-dir`` they are read through ``DriveStore.read_cached_objects``: bytes
are reused only when their SHA-256 matches Drive's fresh checksum, so a cache
kept between runs (actions/cache) makes later runs fast without trusting it.

It never writes to the evidence. The Drive client it restores through is wrapped so that any
upload raises ``READ_ONLY_VIOLATION`` and fails the run. With ``--publish`` (owner 2026-11-10) the
finished report is uploaded as one object under its own prefix, outside the evidence, for the
app to show (``app_core.readiness_report_remote``); a failed upload never fails the report. Unlike ``load_readiness(authenticate=True)``,
it skips the local reconciliation and the backup that follows it. It makes no
Odds API or CFBD calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ('sport', 'market_family', 'unique_events', 'settled_events', 'verified_entry_prices',
           'verified_closes', 'model_status', 'calibration_status', 'validation_plan_status',
           'validation_count', 'holdout_count', 'close_clv_status', 'deployment_state', 'next_blocker')


from app_core.read_only_drive import ReadOnlyDrive, ReadOnlyViolation  # noqa: E402,F401  (re-exported)


def restore(root: Path, client, folder, *, with_sources=False):
    """Restore the canonical store (and, optionally, every sport's research store). Returns counts."""
    import time
    from app_core.prospective_remote import sync
    started = time.monotonic()
    counts = dict(canonical=sync(root / 'prospective-evidence.sqlite3', client, folder, {}))
    print(f'canonical restore: {time.monotonic() - started:.0f}s', flush=True)
    if not with_sources:
        return counts
    from app_core.mlb_receipt_remote import recover
    from app_core.prospective_readiness_report import SOURCE_FILENAMES
    from app_core.prospective_sport_adapters import ADAPTERS
    for sport, adapter in ADAPTERS.items():
        counts[sport] = adapter.restore(root / adapter.path_name, client, folder, {})
    counts['MLB_RECEIPTS'] = dict(records_restored=recover(client, root / SOURCE_FILENAMES['MLB']))
    return counts


def report(root: Path, *, client=None, folder=None, cache_dir=None, with_sources=False):
    from app_core.prospective_readiness_report import load_readiness
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit('the restore directory must start empty, so no local row can be uploaded')
    if client is None:
        from app_core.evidence_drive import DriveStore
        from app_core.evidence_remote import settings
        folder = folder or settings()[0]
        if not folder or not os.getenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', '').strip():
            raise SystemExit('MISSING_CREDENTIALS: PARLAYPICKER_DRIVE_FOLDER_ID / PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')
        client = DriveStore(folder)
    guarded = ReadOnlyDrive(client, cache_dir)
    counts = restore(root, guarded, folder, with_sources=with_sources)
    import time
    loaded = time.monotonic()
    readiness = load_readiness(root, authenticate=False)  # local read of the restored copy
    print(f'section readiness: {time.monotonic() - loaded:.0f}s', flush=True)
    markets = [dict(row, remote_evidence_status='RESTORED_READ_ONLY', remote_evidence_verified=True)
               for row in readiness['markets']]
    store = root / 'prospective-evidence.sqlite3'

    def timed(name, section, *args, **kwargs):
        """Run one report section and print how long it took: a slow section must be visible in the log."""
        import time
        started = time.monotonic()
        try:
            return section(*args, **kwargs)
        finally:
            print(f'section {name}: {time.monotonic() - started:.0f}s', flush=True)
    return dict(remote=dict(status='RESTORED_READ_ONLY', writes_refused=guarded.writes_refused, counts=counts),
                markets=markets, sources=readiness['sources'],
                tables=timed('tables', table_counts, store),
                plans=timed('plans', plan_evaluations, store),
                research_bindings=timed('research_bindings', research_bindings, store),
                price_edge=timed('price_edge', price_edge, store),
                close_coverage=timed('close_coverage', close_coverage, store),
                plan_progress=timed('plan_progress', _plan_progress, store),
                pinnacle_replay=timed('pinnacle_replay', pinnacle_replay, store, extra=EXTRA_REPLAY_SCOPES),
                pace_comparison=timed('pace_comparison', pace_comparison, store, extra=EXTRA_REPLAY_SCOPES),
                replay_clv=timed('replay_clv', replay_clv, store, extra=EXTRA_REPLAY_SCOPES),
                late_entry_clv=timed('late_entry_clv', late_entry_clv, store, extra=EXTRA_REPLAY_SCOPES),
                replay_results=timed('replay_results', replay_results, store, extra=EXTRA_REPLAY_SCOPES),
                line_movement=timed('line_movement', line_movement, store, extra=EXTRA_REPLAY_SCOPES),
                nightly_record=timed('nightly_record', nightly_record, store, extra=EXTRA_REPLAY_SCOPES),
                pace_recommendation=timed('pace_recommendation', pace_recommendation, store, extra=EXTRA_REPLAY_SCOPES))


REPLAY_WINDOWS = ((1, 'last 24 hours'), (14, 'last 14 days'))


# The lag pool's other sports (owner 2026-11-09), replayed beside the Pinnacle model's own.
EXTRA_REPLAY_SCOPES = {'NHL': ('PUCK_LINE', 'TOTAL')}


def pinnacle_replay(path, now=None, extra=None):
    """Bets a Pinnacle plan would have made on games already started (pinnacle_model.replay_estimate),
    for the stake books, over the last day and the last 14 days. Read-only; results are never read.

    It runs here because this workflow has its own queue. The Bound plan draft prints the same
    estimate for one market, but waits behind research capture for hours on a game night."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            for days, label in REPLAY_WINDOWS:
                try:
                    row = pinnacle_model.replay_estimate(path, sport, family, now=now, lookback_days=days,
                                                         books=pinnacle_model.PLAN_BOOKS)
                except Exception as exc:  # report, never guess
                    row = dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300])
                out.append(dict(row, window=label))
    return out


def pace_comparison(path, now=None, extra=None):
    """Bets found at each capture pace on the last day's started games (pinnacle_model.pace_comparison)."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            try:
                out.append(pinnacle_model.pace_comparison(path, sport, family, now=now, books=pinnacle_model.PLAN_BOOKS))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


CLV_WINDOWS = ((1, 'last 24 hours'), (7, 'last 7 days'))


def replay_clv(path, now=None, extra=None):
    """Closing-line value of the replay's first-edge bets (pinnacle_model.replay_clv): do they beat the close?"""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            for days, label in CLV_WINDOWS:
                try:
                    row = pinnacle_model.replay_clv(path, sport, family, now=now, lookback_days=days,
                                                    books=pinnacle_model.PLAN_BOOKS)
                except Exception as exc:  # report, never guess
                    row = dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300])
                out.append(dict(row, window=label))
    return out


def late_entry_clv(path, now=None, extra=None):
    """The same measurement for a later entry (owner 2026-11-10): only captures within
    ``pinnacle_model.LATE_ENTRY_MINUTES`` of the start can make a bet. A bet taken early often sits at a
    line the book later leaves; this shows what a later entry costs in bets and gains in close coverage."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            for days, label in CLV_WINDOWS:
                try:
                    row = pinnacle_model.replay_clv(path, sport, family, now=now, lookback_days=days,
                                                    books=pinnacle_model.PLAN_BOOKS,
                                                    entry_within_minutes=pinnacle_model.LATE_ENTRY_MINUTES)
                except Exception as exc:  # report, never guess
                    row = dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300])
                out.append(dict(row, window=label))
    return out


RESULT_WINDOWS = ((7, 'last 7 days'), (14, 'last 14 days'))
RESULT_ENTRIES = ((None, 'any time'), ('late', 'last 60 min'))


def replay_results(path, now=None, extra=None):
    """Graded results of the replay's first-edge bets, any-time entry and late entry side by side
    (pinnacle_model.replay_results; owner 2026-11-13). This section READS results; the replay and CLV
    sections above do not. Nothing reads it back."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            for days, label in RESULT_WINDOWS:
                for entry, entry_label in RESULT_ENTRIES:
                    minutes = pinnacle_model.LATE_ENTRY_MINUTES if entry == 'late' else None
                    try:
                        row = pinnacle_model.replay_results(path, sport, family, now=now, lookback_days=days,
                                                            books=pinnacle_model.PLAN_BOOKS,
                                                            entry_within_minutes=minutes)
                    except Exception as exc:  # report, never guess
                        row = dict(sport=sport, market_family=family,
                                   evaluation_error=f'{type(exc).__name__}: {exc}'[:300])
                    out.append(dict(row, window=label, entry=entry_label))
    return out


RESULTS_CAPTION = ('Replay results (owner 2026-11-13): the same first-clearing bets at the stake books, graded against '
                   'the verified final score at the bet\'s line, for any-time entry and for entry only in the last 60 '
                   'minutes. This table READS results, unlike the replay and CLV tables above. A rule chosen from it '
                   'needs a fresh frozen window: these games cannot test it. Small counts; read the SE.')


def _results_markdown(rows):
    rows = [r for r in rows if r.get('evaluation_error') or any(e['bets'] for e in (r.get('by_edge') or {}).values())]
    if not rows:
        return []
    lines = ['', RESULTS_CAPTION, '',
             '| market | window | entry | edge | bets | graded | wins / losses / pushes | units | ROI per bet (SE) |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        name = f"{row['sport']} {row['market_family']} | {row.get('window', '')} | {row.get('entry', '')}"
        if 'evaluation_error' in row:
            lines.append(f"| {name} | | {row['evaluation_error']} | | | | |")
            continue
        for edge, found in row['by_edge'].items():
            if not found['bets']:
                continue
            roi = '' if found['roi_per_bet'] is None else f"{found['roi_per_bet']:+.1%}"
            se = '' if found['roi_se'] is None else f" ({found['roi_se']:.1%})"
            lines.append(f"| {name} | {edge} | {found['bets']} | {found['graded']} | "
                         f"{found['wins']} / {found['losses']} / {found['pushes']} | {found['units']:+.2f} | {roi}{se} |")
    return lines


CLV_CAPTION = ('Replay closing-line value: do the first-clearing bets beat the certified close at the same book, '
               'selection and line? A plan needs a comparable close for 80% of its bets and mean CLV with t of 2 or more. '
               'A bet with no comparable close is left out of the mean, which can flatter it. Results are not read.')
LATE_CLV_CAPTION = ('Later entry: the same measurement when only captures in the last 60 minutes before the start can '
                    'make a bet. Compare bets and comparable closes with the table above: a later entry trades bets '
                    'for close coverage.')


BY_BOOK_WINDOW = 'last 7 days'
BY_BOOK_EDGE = '1%'


def _clv_by_book_markdown(any_time, late):
    """The closing-line table split by the stake book the bet was at (owner 2026-11-11; the stake is at Novig).
    One window and edge, the largest sample: the last 7 days at a 1% edge, for both entry timings."""
    lines = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            if row.get('window') != BY_BOOK_WINDOW or 'evaluation_error' in row:
                continue
            for book, found in ((row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}).get('by_book', {}).items():
                if not found['bets']:
                    continue
                mean = '' if found['mean_price_clv'] is None else f"{found['mean_price_clv']:+.1%}"
                se = '' if found['price_clv_se'] is None else f" ({found['price_clv_se']:.1%})"
                beat = '' if found['beat_close_share'] is None else f"{found['beat_close_share']:.0%}"
                share = '' if found['coverage'] is None else f" ({found['coverage']:.0%})"
                lines.append(f"| {row['sport']} {row['market_family']} | {label} | {book} | {found['bets']} | "
                             f"{found['close_fetched']} | {found['comparable']}{share} | {mean}{se} | {beat} |")
    if not lines:
        return []
    return ['', f'Closing-line value by stake book ({BY_BOOK_WINDOW}, {BY_BOOK_EDGE} edge): the same bets as above, split by '
            'the book the bet was at. Small counts; read the pattern, not the decimals.', '',
            '| market | entry | book | bets | close fetched | comparable (share of bets) | mean CLV (SE) | beat close |',
            '|---|---|---|---|---|---|---|---|'] + lines


def _clv_markdown(rows, caption=CLV_CAPTION):
    rows = [r for r in rows if r.get('evaluation_error') or any(e['bets'] for e in (r.get('by_edge') or {}).values())]
    if not rows:
        return []
    lines = ['', caption, '',
             '| market | window | edge | bets | close fetched | comparable (share of bets) | mean CLV (SE) | t | beat close '
             '| no close: not fetched | book missing | line moved | close before bet |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']

    def pct(value, signed=True):
        return '' if value is None else (f'{value:+.1%}' if signed else f'{value:.0%}')
    for row in rows:
        name = f"{row['sport']} {row['market_family']} | {row.get('window', '')}"
        if 'evaluation_error' in row:
            lines.append(f"| {name} | | {row['evaluation_error']} | | | | | | | | | |")
            continue
        for edge, found in row['by_edge'].items():
            if not found['bets']:
                continue
            se = '' if found['price_clv_se'] is None else f" ({found['price_clv_se']:.1%})"
            t = '' if found['t'] is None else f"{found['t']:.1f}"
            lines.append(f"| {name} | {edge} | {found['bets']} | {found['close_fetched']} | "
                         f"{found['comparable']} ({pct(found['coverage'], signed=False)}) | "
                         f"{pct(found['mean_price_clv'])}{se} | {t} | {pct(found['beat_close_share'], signed=False)} | "
                         + ' | '.join(str((found.get('not_comparable') or {}).get(reason, ''))
                                      for reason in ('CLOSE_NOT_FETCHED', 'BOOK_NOT_IN_CLOSE', 'LINE_MOVED',
                                                     'CLOSE_BEFORE_BET')) + ' |')
    return lines


def _clv_line_adjusted_markdown(any_time, late):
    """Closing-line value with the moved lines counted (owner 2026-11-12): the bets the tables above leave out
    because the book closed at another line, scored with an assumed value per point. Same window and edge as
    the by-book table."""
    lines = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            if row.get('window') != BY_BOOK_WINDOW or 'evaluation_error' in row:
                continue
            found = (row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}
            adjusted = found.get('line_adjusted')
            if not adjusted or not found.get('bets'):
                continue
            points = adjusted['mean_points_in_favour']
            scored = adjusted['scored']
            if adjusted['point_value'] is None:
                result = 'no point value assumed | | | '
            else:
                mean = '' if adjusted['mean_clv'] is None else f"{adjusted['mean_clv']:+.1%}"
                se = '' if adjusted['clv_se'] is None else f" ({adjusted['clv_se']:.1%})"
                t = '' if adjusted['t'] is None else f"{adjusted['t']:.1f}"
                moved = '' if adjusted.get('moved_mean_clv') is None else f"{adjusted['moved_mean_clv']:+.1%}"
                result = f"{mean}{se} | {t} | {moved} | {adjusted['point_value']:.1%}"
            lines.append(f"| {row['sport']} {row['market_family']} | {label} | {found['bets']} | {found['comparable']} | "
                         f"{adjusted['moved_with_close']} | {adjusted['moved_in_favour']} / {adjusted['moved_against']} | "
                         f"{'' if points is None else f'{points:+.2f}'} | {scored} ({scored / found['bets']:.0%}) | "
                         f"{result} |")
    if not lines:
        return []
    return ['', f'Line-adjusted closing-line value ({BY_BOOK_WINDOW}, {BY_BOOK_EDGE} edge): the same bets with the moved lines '
            'counted. A bet whose book closed at another line is scored as its odds times (the close\'s implied '
            'probability plus the points it gained times an ASSUMED value per point) minus 1. "In favour / against" '
            'and the mean points need no assumption: they say whether the bet ended with the better number. A plan '
            'is not held to this measure; it shows whether the edge survives when the moved lines are counted.', '',
            '| market | entry | bets | same line | moved, close found | moved in favour / against | mean points gained '
            '| scored (share of bets) | mean CLV, all scored (SE) | t | mean CLV, moved only | assumed value per point |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|'] + lines


def _clv_large_moves_markdown(any_time, late):
    """The moved bets the line-adjusted table does not score (2026-11-12): a close more than
    ``max_scored_points`` from the bet. Same window and edge as that table."""
    lines = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            if row.get('window') != BY_BOOK_WINDOW or 'evaluation_error' in row:
                continue
            adjusted = ((row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}).get('line_adjusted') or {}
            if 'large_moves' not in adjusted or not adjusted.get('moved_with_close'):
                continue
            median = adjusted.get('median_points_in_favour')
            biggest = adjusted.get('largest_move')
            lines.append(f"| {row['sport']} {row['market_family']} | {label} | {adjusted['moved_with_close']} | "
                         f"{adjusted['moved_with_close'] - adjusted['large_moves']} | "
                         f"{'' if median is None else f'{median:+.2f}'} | {adjusted['large_moves']} | "
                         f"{'' if biggest is None else f'{biggest:+.1f}'} |")
    if not lines:
        return []
    limit = next((((row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}).get('line_adjusted') or {}).get('max_scored_points')
                 for rows in (any_time, late) for row in rows
                 if (((row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}).get('line_adjusted') or {}).get('max_scored_points'))
    traced = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            if row.get('window') != BY_BOOK_WINDOW or 'evaluation_error' in row:
                continue
            adjusted = ((row.get('by_edge') or {}).get(BY_BOOK_EDGE) or {}).get('line_adjusted') or {}
            for bet in adjusted.get('large_move_bets') or []:
                traced.append(f"| {row['sport']} {row['market_family']} | {label} | {str(bet['event_id'])[:12]} | "
                              f"{bet['sportsbook']} | {bet['selection']} | {bet['bet_line']:+g} | {bet['close_line']:+g} | "
                              f"{bet['points']:+.1f} | {'yes' if bet['same_size_opposite_sign'] else 'no'} | "
                              f"{bet['bet_quote_at']} | {bet['close_at']} |")
    if traced:
        lines += ['', 'Each large move (to trace: a real move, another line of the game, or a sign error in the close):', '',
                  '| market | entry | event | book | selection | bet line | close line | points | same size, opposite sign '
                  '| bet quoted at | close at |', '|---|---|---|---|---|---|---|---|---|---|---|'] + traced
    return ['', f'Moved lines, scored and not: a close more than {limit:g} points from the bet is counted in "moved in favour / '
            'against" above but is NOT scored, and is left out of the mean points gained. A value per point holds only near the '
            'line, and a close that far from the bet may be another line of the same game rather than a move of the main one.',
            '', '| market | entry | moved, close found | scored moves | median points gained (scored) | large moves, not scored '
            '| largest |', '|---|---|---|---|---|---|---|'] + lines


def _clv_vs_pinnacle_markdown(any_time, late):
    """The first-edge bets valued at Pinnacle's last no-vig price before the start (owner 2026-11-12)."""
    lines = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            if 'evaluation_error' in row:
                continue
            for edge, found in (row.get('by_edge') or {}).items():
                against = found.get('vs_pinnacle')
                if not against or not against.get('bets'):
                    continue
                mean = '' if against['mean_closing_ev'] is None else f"{against['mean_closing_ev']:+.1%}"
                se = '' if against['se'] is None else f" ({against['se']:.1%})"
                t = '' if against['t'] is None else f"{against['t']:.1f}"
                beat = '' if against['beat_close_share'] is None else f"{against['beat_close_share']:.0%}"
                lines.append(f"| {row['sport']} {row['market_family']} | {row.get('window', '')} | {label} | {edge} | "
                             f"{against['bets']} | {against['priced']} ({against['coverage']:.0%}) | {mean}{se} | {t} | "
                             f"{beat} | {against['pinnacle_moved_in_favour']} / {against['pinnacle_moved_against']} | "
                             f"{against['no_late_pinnacle_capture']} |")
    if not lines:
        return []
    waits = []
    for label, rows in (('any time', any_time), ('last 60 min', late)):
        for row in rows:
            for edge, found in ((row.get('by_edge') or {}) if 'evaluation_error' not in row else {}).items():
                against = found.get('vs_pinnacle') or {}
                if against.get('bets') and 'entered_at_last_capture' in against:
                    wait = against.get('median_minutes_entry_to_close')
                    waits.append(f"| {row['sport']} {row['market_family']} | {row.get('window', '')} | {label} | {edge} | "
                                 f"{against['entered_at_last_capture']} | {'' if wait is None else f'{wait:.0f}'} |")
    if waits:
        lines += ['', 'Not priced above: a bet entered at Pinnacle\'s last capture was chosen because it beat Pinnacle in that '
                  'same capture, so pricing it there would only repeat the entry edge. The median minutes show how long after '
                  'entry the priced bets were valued.', '',
                  '| market | window | entry | edge | entered at the last capture (not priced) | median minutes from entry to close |',
                  '|---|---|---|---|---|---|'] + waits
    return ['', "Closing value against Pinnacle: the same bets valued at Pinnacle's last no-vig price before the start "
            '(its last capture, within 30 minutes of the start, at the bet\'s line): bet odds times that probability, minus '
            '1. Pinnacle\'s close is the usual benchmark, and it does not depend on the stake book\'s closing snapshot, which '
            'for an exchange can sit far from the market minutes before tip-off. When Pinnacle closed at another line the '
            'bet is counted by direction, not priced. These captures are not certified closes and a plan is not held to '
            'this measure.', '',
            '| market | window | entry | edge | bets | priced at Pinnacle\'s close (share) | mean closing EV (SE) | t '
            '| positive | Pinnacle moved in favour / against | no late Pinnacle capture |',
            '|---|---|---|---|---|---|---|---|---|---|---|'] + lines


def line_movement(path, now=None, extra=None):
    """Pinnacle's line movement before each started game and whether the stake books followed
    (pinnacle_model.line_movement), over the last 24 hours. Read-only."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            try:
                out.append(pinnacle_model.line_movement(path, sport, family, now=now))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def _movement_markdown(rows):
    rows = [r for r in rows if r.get('evaluation_error') or r.get('games_tracked')]
    if not rows:
        return []
    lines = ['', "Line movement (last 24 hours): how Pinnacle's main line moved between its first and last capture "
             'before each game, and whether the stake books (Novig, DraftKings, FanDuel, BetMGM) followed. The nearest '
             'thing to market sentiment in the stored prices; no betting splits are captured. Captures are hours apart '
             "until a game's last three hours, so an early move shows only as a difference between snapshots.", '',
             '| market | games tracked / started | line moved | mean move (points) | price shift when the line held | '
             "stake books at Pinnacle's final line | median minutes to follow | captures per game |",
             '|---|---|---|---|---|---|---|---|']

    def pct(value, digits=0):
        return '' if value is None else f'{value:.{digits}%}'
    for row in rows:
        name = f"{row['sport']} {row['market_family']}"
        if 'evaluation_error' in row:
            lines.append(f"| {name} | {row['evaluation_error']} | | | | | | |")
            continue
        move = '' if row['mean_abs_move'] is None else f"{row['mean_abs_move']:.2f}"
        follow = '' if row['median_follow_minutes'] is None else f"{row['median_follow_minutes']:g}"
        lines.append(f"| {name} | {row['games_tracked']} / {row['games_started']} | "
                     f"{row['games_moved']} ({pct(row['share_moved'])}) | {move} | "
                     f"{pct(row['mean_prob_shift_when_held'], 1)} | "
                     f"{row['stake_books_at_final_line']} of {row['stake_book_games']} ({pct(row['share_followed'])}) | "
                     f"{follow} | {row['captures_per_game']:g} |")
    return lines


GO_BETS_PER_NIGHT = 2        # the go/no-go line for a plan (roadmap M2)
RECORD_NIGHTS = 7


def nightly_record(path, now=None, extra=None):
    """Per ET slate night, games started and first-edge bets per market (pinnacle_model.nightly_record)."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport, families in {**pinnacle_model.FAMILIES, **(extra or {})}.items():
        for family in families:
            try:
                out.append(pinnacle_model.nightly_record(path, sport, family, now=now, nights=RECORD_NIGHTS,
                                                         books=pinnacle_model.PLAN_BOOKS))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=family, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def pace_recommendation(path, now=None, extra=None):
    """Per sport, the slowest capture pace that kept 85% of the bets over the last 7 days
    (price_watch.recommended): what the watch's automatic pace would use. Read-only."""
    from datetime import datetime, timezone
    from app_core import pinnacle_model, price_watch
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    out = []
    for sport in {**pinnacle_model.FAMILIES, **(extra or {})}:
        try:
            out.append(dict(price_watch.recommended(path, sport, now), sport=sport,
                            current_minutes=price_watch.interval_minutes(sport, {}.get)))
        except Exception as exc:  # report, never guess
            out.append(dict(sport=sport, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


HEALTH_MIN_GAMES = 3
HEALTH_MIN_PINNACLE_SHARE = 0.5


def capture_health(replay, paces):
    """Warnings from the last 24 hours of the replay and pace rows: no new store read.

    * ``WATCH_NOT_CAPTURING``: games started but each has only its one research capture, so
      the price watch did not run for that sport (a scheduled start that never fired looks like this);
    * ``PINNACLE_LINE_LOW``: under half the started games had Pinnacle at a stake book's line.
    """
    flags = []
    for row in paces:
        captured = next((p for p in row.get('paces') or [] if p['pace'] == 'as captured'), None)
        if captured and row['games_started'] >= HEALTH_MIN_GAMES and captured['captures_per_game'] <= 1:
            flags.append(dict(sport=row['sport'], market_family=row['market_family'], flag='WATCH_NOT_CAPTURING',
                              detail=f"{row['games_started']} games started with one capture each"))
    for row in replay:
        if row.get('window') != REPLAY_WINDOWS[0][1] or row.get('status') != 'ESTIMATED':
            continue
        if row['games_started'] >= HEALTH_MIN_GAMES and row['plan_games'] / row['games_started'] < HEALTH_MIN_PINNACLE_SHARE:
            flags.append(dict(sport=row['sport'], market_family=row['market_family'], flag='PINNACLE_LINE_LOW',
                              detail=f"{row['plan_games']} of {row['games_started']} games at Pinnacle's line"))
    return flags


def _pipeline_markdown(result):
    """Nightly record, pace recommendation and capture warnings (2026-11-10). Research only."""
    lines = []
    record = [r for r in result.get('nightly_record') or [] if r.get('nights')]
    if record:
        nights = sorted({n['night'] for r in record for n in r['nights']})
        lines += ['', f'Nightly record: first-clearing bets at 1% / 2% per ET slate night (games started in brackets). '
                  f'The go line for a plan is {GO_BETS_PER_NIGHT} bets a night. Volume only: results are not read.', '',
                  '| market | ' + ' | '.join(nights) + f' | nights at {GO_BETS_PER_NIGHT}+ (1%) |',
                  '|---|' + '---|' * (len(nights) + 1)]
        for row in record:
            by_night = {n['night']: n for n in row['nights']}
            cells = [f"{by_night[n]['bets'].get('1%', '')} / {by_night[n]['bets'].get('2%', '')} [{by_night[n]['games_started']}]"
                     if n in by_night else '' for n in nights]
            reached = sum(1 for n in row['nights'] if n['bets'].get('1%', 0) >= GO_BETS_PER_NIGHT)
            lines.append(f"| {row['sport']} {row['market_family']} | " + ' | '.join(cells) + f" | {reached} of {len(row['nights'])} |")
    recommended = result.get('pace_recommendation') or []
    if recommended:
        lines += ['', 'Capture pace recommendation (last 7 days): the slowest poll that kept 85% of the bets found as '
                  'captured. The watch uses it only when automatic pace is switched on, and only to slow a sport down.', '',
                  '| sport | normal pace | recommended | status | bets as captured (1%) | captures per game | share kept at 15 / 30 / 60 min |',
                  '|---|---|---|---|---|---|---|']
        for row in recommended:
            if 'evaluation_error' in row:
                lines.append(f"| {row['sport']} | | | {row['evaluation_error']} | | | |")
                continue
            kept = row.get('kept') or {}
            shares = ' / '.join(f"{kept[k]:.0%}" if k in kept else '' for k in ('15 min', '30 min', '60 min'))
            lines.append(f"| {row['sport']} | {row['current_minutes']} min | "
                         f"{str(row['minutes']) + ' min' if row.get('minutes') else ''} | {row['status']} | "
                         f"{row['bets_as_captured']:g} | {row.get('captures_per_game', '')} | {shares} |")
    flags = capture_health(result.get('pinnacle_replay') or [], result.get('pace_comparison') or [])
    if flags:
        lines += ['', '| capture warning (last 24 hours) | market | detail |', '|---|---|---|']
        lines += [f"| {f['flag']} | {f['sport']} {f['market_family']} | {f['detail']} |" for f in flags]
    elif result.get('pace_comparison'):
        lines += ['', 'Capture warnings (last 24 hours): none.']
    return lines


def _pace_markdown(rows):
    rows = [r for r in rows if r.get('paces') or r.get('evaluation_error')]
    if not rows:
        return []
    lines = ['', 'Capture pace: bets the first-clearing rule finds on the last 24 hours of started games when the stored '
             'captures are thinned to a slower poll. Each cell is bets at 1% / bets at 2%, with captures per game in '
             'brackets; slower paces are the mean of three alignments (range). An estimate: a slower poll would have '
             'landed at times that were not captured.', '',
             '| market | games started | as captured | every 15 min | every 30 min | every 60 min |', '|---|---|---|---|---|---|']

    def cell(pace):
        def one(found):
            return str(found['mean']) if found['low'] == found['high'] else f"{found['mean']} ({found['low']}-{found['high']})"
        bets = pace['bets']
        return f"{one(bets.get('1%', {'mean': '', 'low': 0, 'high': 0}))} / {one(bets.get('2%', {'mean': '', 'low': 0, 'high': 0}))} [{pace['captures_per_game']}]"
    for row in rows:
        name = f"{row['sport']} {row['market_family']}"
        if 'evaluation_error' in row:
            lines.append(f"| {name} | {row['evaluation_error']} | | | | |")
            continue
        by_pace = {p['pace']: p for p in row['paces']}
        lines.append(f"| {name} | {row['games_started']} | " + ' | '.join(
            cell(by_pace[label]) if label in by_pace else '' for label in ('as captured', '15 min', '30 min', '60 min')) + ' |')
    return lines


def _replay_markdown(rows):
    if not rows:
        return []
    lines = ['', "Pinnacle replay: bets a plan would have made on games already started, at the stake books "
             "(Novig, DraftKings, FanDuel, BetMGM). A game not yet started is not counted. Results are not read.", '',
             '| market | window | games started | at Pinnacle\'s line | junk quotes dropped | earliest capture: bets at 1% | at 2% '
             '| first capture that clears: bets at 1% | at 2% |', '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        name = f"{row['sport']} {row['market_family']} | {row.get('window', '')}"
        if row.get('status') != 'ESTIMATED':
            lines.append(f"| {name} | {row.get('games_started', row.get('evaluation_error', ''))} | | | | | | |")
            continue
        cells = {}
        for variant in row['variants']:
            if variant['probability'] == 'PINNACLE_NO_VIG':
                for edge, found in variant['by_edge'].items():
                    cells[(variant['entry_rule'], edge)] = f"{found['bets']} ({found['bets_per_day']}/day)"
        lines.append(f"| {name} | {row['games_started']} | {row['plan_games']} | {row.get('junk_quotes_dropped', '')} | "
                     f"{cells.get(('EARLIEST_SHARED_LINE', '1%'), '')} | {cells.get(('EARLIEST_SHARED_LINE', '2%'), '')} | "
                     f"{cells.get(('FIRST_EDGE_CAPTURE', '1%'), '')} | {cells.get(('FIRST_EDGE_CAPTURE', '2%'), '')} |")
    return lines


def _plan_progress(path):
    """Frozen bound plans' pace against their windows (app_core.plan_progress); read-only."""
    from app_core import plan_progress
    try:
        return plan_progress.from_store(path)
    except Exception as exc:  # a progress view must never break the readiness report
        return [{'plan': 'ALL', 'sport': '', 'market_family': '', 'error': type(exc).__name__}]


def price_edge(path):
    """Cross-book price screen per research market (app_core.price_edge_screen), read-only."""
    from app_core import price_edge_screen
    from app_core.prospective_research_models import SCOPES
    if not Path(path).is_file():
        return []
    out = []
    for sport, markets in SCOPES.items():
        for market in markets:
            try:
                out.append(price_edge_screen.from_store(path, sport, market))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=market, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def close_coverage(path):
    """Comparable certified-close coverage by entry timing (app_core.close_coverage), read-only."""
    from app_core import close_coverage as coverage
    from app_core.prospective_research_models import SCOPES
    if not Path(path).is_file():
        return []
    out = []
    for sport, markets in SCOPES.items():
        for market in markets:
            try:
                out.append(coverage.from_store(path, sport, market))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=market, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def research_bindings(path, now=None):
    """Each research market's place on the C2 chain (prospective_calibration.binding_status), read-only."""
    from app_core import prospective_calibration as calibration
    from app_core.prospective_research_models import SCOPES
    if not Path(path).is_file():
        return []
    out = []
    for sport, markets in SCOPES.items():
        for market in markets:
            try:
                out.append(calibration.binding_status(path, sport, market, now=now))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=market, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def _scalars(value):
    """The scalar fields of a nested report section (counts, metrics, statuses)."""
    plain = (str, int, float, bool, type(None))
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, plain)] if all(isinstance(v, plain) for v in value) else len(value)
    if not isinstance(value, dict):
        return value if isinstance(value, plain) else type(value).__name__
    return {k: v for k, v in value.items() if isinstance(v, plain)}


def plan_evaluations(path):
    """Each frozen validation plan's own evaluation, read-only (prospective_evidence.evaluate_validation_plan)."""
    from app_core import prospective_evidence as evidence
    if not Path(path).is_file():
        return []
    out = []
    for plan in evidence.read_records(path, 'prospective_validation_plan'):
        row = dict(validation_plan_id=plan['validation_plan_id'], sport=plan['sport'],
                   market_family=plan['market_family'], version=plan.get('version'),
                   model_id=plan.get('model_id'), calibration_id=plan.get('calibration_id'))
        try:
            result = evidence.evaluate_validation_plan(path, plan['validation_plan_id'])
            row.update({k: _scalars(v) for k, v in result.items()
                        if k not in row and k not in ('plan_artifact_hash',)})
        except Exception as exc:  # report, never guess
            row['evaluation_error'] = f'{type(exc).__name__}: {exc}'[:300]
        out.append(row)
    return out


def table_counts(path):
    """Row counts per canonical table, split by sport where the table has one (read-only)."""
    import sqlite3
    if not Path(path).is_file():
        return {}
    out = {}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for name in names:
            columns = {r[1] for r in db.execute(f'PRAGMA table_info("{name}")')}
            if 'sport' in columns:
                out[name] = {str(sport): n for sport, n in db.execute(
                    f'SELECT sport, COUNT(*) FROM "{name}" GROUP BY sport ORDER BY sport')}
            else:
                out[name] = db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    return out


def markdown(result):
    lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
    for row in result['markets']:
        lines.append('| ' + ' | '.join('' if row.get(c) is None else str(row.get(c)) for c in COLUMNS) + ' |')
    tables = result.get('tables') or {}
    if tables:
        lines += ['', '| canonical table | rows |', '|---|---|']
        for name, count in tables.items():
            shown = ', '.join(f'{k}: {v}' for k, v in count.items()) if isinstance(count, dict) else str(count)
            lines.append(f'| {name} | {shown or 0} |')
    plans = result.get('plans') or []
    if plans:
        lines += ['', '| validation plan | version | model | status | blockers |', '|---|---|---|---|---|']
        for plan in plans:
            blockers = plan.get('blockers') or plan.get('reasons') or plan.get('evaluation_error') or ''
            lines.append(f"| {plan['validation_plan_id']} | {plan.get('version')} | {plan.get('model_id') or ''} | "
                         f"{plan.get('status', '')} | {blockers} |")
    bindings = result.get('research_bindings') or []
    if bindings:
        lines += ['', '| research market | training events | candidate decided | calibration | games/day | '
                  'window days | bound plan | next blocker |', '|---|---|---|---|---|---|---|---|']
        for row in bindings:
            lines.append(f"| {row['sport']} {row['market_family']} | {row.get('most_training_events', '')} | "
                         f"{row.get('candidate_decided', '')} | {row.get('calibration_id') or ''} | "
                         f"{row.get('games_per_day_14d', '')} | {row.get('proposed_window_days') or ''} | "
                         f"{row.get('bound_plan_id') or ''} | "
                         f"{row.get('next_blocker') or row.get('evaluation_error', '')} |")
    edges = result.get('price_edge') or []
    if edges:
        lines += ['', '| price screen (best price, all candidates) | candidates | EV >= 2% picks | mean EV | closes | mean CLV | beat close | settled | ROI |',
                  '|---|---|---|---|---|---|---|---|---|']
        def pct(v):
            return '' if v is None else f'{v:+.1%}'
        for row in edges:
            t = (row.get('by_threshold') or {}).get('all') or {}
            flagged = ((row.get('by_threshold') or {}).get('0.02') or {}).get('picks', '')
            lines.append(f"| {row['sport']} {row['market_family']} | {row.get('candidates', row.get('evaluation_error', ''))} | "
                         f"{flagged} | {pct(t.get('mean_ev'))} | {t.get('with_comparable_close', '')} | "
                         f"{pct(t.get('mean_price_clv'))} | {pct(t.get('beat_close_share'))} | {t.get('settled', '')} | "
                         f"{pct(t.get('roi_per_pick'))} |")
        lines += _price_breakdown(edges, pct)
    covered = [row for row in result.get('close_coverage') or [] if row.get('by_minutes_before_start')]
    if covered:
        from app_core.close_coverage import BUCKETS
        labels = [label for _, _, label in BUCKETS]
        lines += ['', 'Comparable certified close (same book, selection and line) for entry quotes on started games, '
                  'by how long before the start the quote was taken. A plan needs 80% for its bets.', '',
                  '| market | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
        for row in covered:
            cells = []
            for label in labels:
                b = row['by_minutes_before_start'][label]
                cells.append('' if b['share'] is None else f"{b['share']:.0%} ({b['comparable']}/{b['with_close_fetched']})")
            lines.append(f"| {row['sport']} {row['market_family']} | " + ' | '.join(cells) + ' |')
    lines += _replay_markdown(result.get('pinnacle_replay') or [])
    lines += _pace_markdown(result.get('pace_comparison') or [])
    lines += _clv_markdown(result.get('replay_clv') or [])
    lines += _clv_markdown(result.get('late_entry_clv') or [], LATE_CLV_CAPTION)
    lines += _clv_by_book_markdown(result.get('replay_clv') or [], result.get('late_entry_clv') or [])
    lines += _clv_line_adjusted_markdown(result.get('replay_clv') or [], result.get('late_entry_clv') or [])
    lines += _clv_large_moves_markdown(result.get('replay_clv') or [], result.get('late_entry_clv') or [])
    lines += _clv_vs_pinnacle_markdown(result.get('replay_clv') or [], result.get('late_entry_clv') or [])
    lines += _results_markdown(result.get('replay_results') or [])
    lines += _movement_markdown(result.get('line_movement') or [])
    lines += _pipeline_markdown(result)
    if result.get('plan_progress'):  # only once a frozen bound plan has a window running
        from app_core import plan_progress
        lines += plan_progress.markdown(result['plan_progress'])
    return '\n'.join(lines) + '\n'


def _price_breakdown(edges, pct):
    """Price screen by the book with the best price, and by quote timing (roadmap M5). Research only."""
    lines = []
    for key, title in (('by_book', 'book with the best price'), ('by_timing', 'best price quoted before the start')):
        rows = [(row, name, g) for row in edges for name, g in (row.get(key) or {}).items()]
        if not rows:
            continue
        lines += ['', f'| price screen by {title} | picks | mean EV | closes | mean CLV (SE) | beat close | settled | '
                  'wins | ROI (SE) |', '|---|---|---|---|---|---|---|---|---|']

        def with_se(value, se):
            return '' if value is None else pct(value) + ('' if se is None else f' ({se:.1%})')
        for row, name, g in rows:
            lines.append(f"| {row['sport']} {row['market_family']}: {name} | {g['picks']} | {pct(g['mean_ev'])} | "
                         f"{g['with_comparable_close']} | {with_se(g['mean_price_clv'], g['price_clv_se'])} | "
                         f"{pct(g['beat_close_share'])} | {g['settled']} | {g['wins']} | "
                         f"{with_se(g['roi_per_pick'], g['roi_se'])} |")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--directory', type=Path, required=True, help='empty directory to restore into')
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--cache-dir', type=Path, help='checksum-verified Drive object cache kept between runs')
    parser.add_argument('--with-sources', action='store_true', help="also restore each sport's research store")
    parser.add_argument('--publish', action='store_true',
                        help='also upload the finished report to Drive for the app (one object, outside the evidence)')
    args = parser.parse_args(argv)
    result = report(args.directory, cache_dir=args.cache_dir, with_sources=args.with_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    table = markdown(result)
    args.output.with_suffix('.md').write_text(table, encoding='utf-8')
    summary = os.getenv('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write('## Prospective market readiness (read-only restore)\n\n' + table)
    print(table)
    if args.publish:
        print('publish: ' + json.dumps(publish(result, table)), flush=True)
    return 0


def publish(result, table, client=None, folder=None):
    """Upload the finished report for the app. The report is already written: an error here is reported, not raised."""
    from app_core import readiness_report_remote
    try:
        if client is None:
            from app_core.evidence_drive import DriveStore
            from app_core.evidence_remote import settings
            folder = settings()[0]
            client = DriveStore(folder)
        return readiness_report_remote.publish(client, folder, result, table)
    except Exception as exc:
        return dict(status='PUBLISH_FAILED', error=type(exc).__name__)


if __name__ == '__main__':
    raise SystemExit(main())
